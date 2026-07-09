"""
ZO-LDSD optimizers adapted for OLMo training pipeline.

Ports ZO_MUON, ZO_SignSGD, ZO_RL, and ZO_RL_AdaMM from zo_ldsd/ to OLMo's
ZeroOrderOptimizer interface.
Key adaptations vs the original zo_ldsd implementations:
  - step(closure, z_seed) accepts an external seed so distributed ranks stay in sync
  - get_post_step_metrics() exposes training diagnostics for W&B / console logging
  - per-device generators (consistent with existing MeZO / ZoAdam convention)
  - weight_decay applied as decoupled (multiplicative) decay on all parameters
  - k candidate seeds are derived deterministically from z_seed (LDSDRl / LDSDRlAdaMM)
    so all DDP ranks agree on the candidate set and the optimal seed selection
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch

from .zo_optim import VectorSampler, ZeroOrderOptimizer


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalisation (from ZO-LDSD / Muon)
# ---------------------------------------------------------------------------

def _zeropower_via_newtonschulz5(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Newton-Schulz iteration to orthogonalise G.

    Returns something like U S' V^T where S' has singular values roughly in
    [0.5, 1.5].  This approximation is sufficient for update normalisation
    and avoids a full SVD.  Operates in bfloat16 for speed, returns in the
    original dtype.

    Reference: ZO-LDSD repo, optimizers/opt_utils/newton_schulz.py.
    """
    assert G.ndim == 2, f"Expected 2-D tensor, got shape {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# LDSDMuon
# ---------------------------------------------------------------------------

class LDSDMuon(ZeroOrderOptimizer):
    """ZO-MUON from ZO-LDSD, adapted to OLMo's interface.

    Zero-order optimizer that applies Newton-Schulz orthogonalisation to the
    gradient estimate for 2-D parameters and sign-compression for 1-D
    parameters.

    Update rule per parameter p
    ---------------------------
    1. seed = z_seed
    2. Sample z_i sequentially (perturb phase):
         z_i ~ N(0, I)  (generator advances through all params)
    3. Compute projected scalar:
         g = (f(θ + ε z) − f(θ − ε z)) / 2   [two_side]
    4. For the update, re-seed before each param (faithful to ZO_MUON design,
       meaning all params use the same z sampled fresh from z_seed):
         z_fresh ~ N(0, I)   (generator re-seeded to z_seed before each param)
         grad_update = g * z_fresh / ε
    5. Apply normalised update:
         for 2-D: p ← p − lr * NS(grad_update)
         for 1-D: p ← p − lr * sign(grad_update)
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        vector_sampling_type: str = "standard_normal",
        newtonschulz_steps: int = 5,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.vector_sampler = VectorSampler(vector_sampling_type)
        self.newtonschulz_steps = newtonschulz_steps
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def _seed_all_devices(self, z_seed: int) -> None:
        """Seed each unique device's generator once (for sequential perturb)."""
        seen: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.device not in seen:
                    self._get_generator(p.device).manual_seed(z_seed)
                    seen.add(p.device)

    # ------------------------------------------------------------------
    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics from the last step.

        projected_grad_abs
            |g| where g = (f+ − f−)/2ε (or (f+ − f−)/ε for one_side).
        grad_est_norm
            L2 norm of ``g * z_fresh / ε`` concatenated across all params
            (raw estimate before Newton-Schulz/sign, before lr scaling).
        grad_est_norm_per_z_rms
            ``grad_est_norm`` over the RMS norm of raw ``z``; equals ``|g|/ε``
            with ``g = (f⁺−f⁻)/2`` from the step (same scaling as MeZO).
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor],
        z_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        # --- forward perturbation (sequential z per param) ---
        self._perturb_sequential(z_seed, scaling_factor=+1.0)
        loss_plus = closure()

        if self.defaults["perturbation_mode"] == "two_side":
            self._perturb_sequential(z_seed, scaling_factor=-2.0)
            loss_minus = closure()
            projected_grad = (loss_plus - loss_minus).item() / 2.0
            self._perturb_sequential(z_seed, scaling_factor=+1.0)  # restore
        else:
            self._perturb_sequential(z_seed, scaling_factor=-1.0)
            loss_minus = closure()
            projected_grad = (loss_plus - loss_minus).item()

        self._apply_muon_update(z_seed, projected_grad)
        self._last_metrics["projected_grad_abs"] = abs(projected_grad)
        return loss_plus

    # ------------------------------------------------------------------
    def _perturb_sequential(self, z_seed: int, scaling_factor: float) -> None:
        """Add ε * scaling_factor * z_i to each param, z_i sampled sequentially."""
        self._seed_all_devices(z_seed)
        for group in self.param_groups:
            eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                p.data.add_(z, alpha=scaling_factor * eps)

    def _apply_muon_update(self, z_seed: int, projected_grad: float) -> None:
        """Apply the MUON parameter update.

        Faithful to ZO_MUON: generator is re-seeded to z_seed *before* each
        parameter's sample, so every parameter gets the same fresh z sampled
        from z_seed (re-shaped to its own dimensions).
        """
        grad_sum_sq = 0.0
        z_sum_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["zo_eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                # Re-seed before each param — all params get z(seed, shape_i).
                gen = self._get_generator(p.device)
                gen.manual_seed(z_seed)
                z = self.vector_sampler.sample(p.shape, p.device, gen)

                z_sum_sq += z.to(torch.float32).norm().item() ** 2
                grad_update = z.mul(projected_grad / eps)
                grad_sum_sq += grad_update.to(torch.float32).norm().item() ** 2

                if p.ndim >= 2:
                    grad_final = _zeropower_via_newtonschulz5(
                        grad_update, steps=self.newtonschulz_steps
                    )
                else:
                    grad_final = torch.sign(grad_update)

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
                p.data.add_(grad_final, alpha=-lr)

        ge = math.sqrt(grad_sum_sq)
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["grad_est_norm"] = ge
        self._last_metrics["grad_est_norm_per_z_rms"] = (ge / z_rms) if z_rms > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# LDSDSignSgd
# ---------------------------------------------------------------------------


class LDSDSignSgd(ZeroOrderOptimizer):
    """ZO-SignSGD from ZO-LDSD, adapted to OLMo's interface.

    Same finite-difference probing as MeZO, but the projected scalar gradient
    g = (f+ − f−)/2ε is replaced by its sign before scaling the update:

        update_i = z_i * sign(g) / ε

    This makes the effective step magnitude insensitive to the absolute size of
    the loss difference, acting like a sign-based first-order method applied to
    the zero-order estimate.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        vector_sampling_type: str = "standard_normal",
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        super().__init__(params, defaults)
        self.vector_sampler = VectorSampler(vector_sampling_type)
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def _reset_generators(self, z_seed: int) -> None:
        seen: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.device not in seen:
                    self._get_generator(p.device).manual_seed(z_seed)
                    seen.add(p.device)

    # ------------------------------------------------------------------
    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics from the last step.

        projected_grad_abs
            |g| where g = (f+ − f−)/2  (raw, before sign compression).
        grad_est_norm
            L2 norm of ``sign(g) * z / ε`` across all params.
        grad_est_norm_per_z_rms
            Ratio to RMS ``||z||₂``; with ``|sign(g)|=1`` equals ``1/ε``; scales
            with ``|g|`` if you compare to raw ``g`` via ``projected_grad_abs/ε``
            only when the sign is non-degenerate.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor],
        z_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        self._perturb(z_seed, scaling_factor=+1.0)
        loss_plus = closure()

        if self.defaults["perturbation_mode"] == "two_side":
            self._perturb(z_seed, scaling_factor=-2.0)
            loss_minus = closure()
            raw_grad = (loss_plus - loss_minus).item() / 2.0
            self._perturb(z_seed, scaling_factor=+1.0)  # restore
        else:
            self._perturb(z_seed, scaling_factor=-1.0)
            loss_minus = closure()
            raw_grad = (loss_plus - loss_minus).item()

        # sign compression of the scalar gradient
        signed_grad = math.copysign(1.0, raw_grad) if raw_grad != 0.0 else 0.0

        self._apply_update(z_seed, signed_grad)
        self._last_metrics["projected_grad_abs"] = abs(raw_grad)
        return loss_plus

    # ------------------------------------------------------------------
    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        self._reset_generators(z_seed)
        for group in self.param_groups:
            eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                p.data.add_(z, alpha=scaling_factor * eps)

    def _apply_update(self, z_seed: int, signed_grad: float) -> None:
        self._reset_generators(z_seed)
        grad_sum_sq = 0.0
        z_sum_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["zo_eps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                z_sum_sq += z.to(torch.float32).norm().item() ** 2
                grad_est = z.mul(signed_grad / eps)
                grad_sum_sq += grad_est.to(torch.float32).norm().item() ** 2
                if weight_decay != 0.0:
                    grad_est.add_(p.data, alpha=weight_decay)
                if momentum != 0.0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = grad_est.clone()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad_est)
                        grad_est = buf
                p.data.add_(grad_est, alpha=-lr)
        ge = math.sqrt(grad_sum_sq)
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["grad_est_norm"] = ge
        self._last_metrics["grad_est_norm_per_z_rms"] = (ge / z_rms) if z_rms > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# Shared μ_0-from-FO-gradient init (LDSDRl / LDSDRlAdaMM / LDSDRlSgd)
# ---------------------------------------------------------------------------

class _MuFromFoGradMixin:
    """Seed the per-parameter trust direction μ from a true first-order gradient.

    Requires a prior FO backward pass so every trainable parameter's ``.grad``
    is populated (e.g. via ``Trainer.train_batch``). Overwrites ``state["mu"]``
    (and ``state["mu_old"]`` / ``state["mu_old_norm_sq"]``) in place, so it must
    be called before the first ``step()``.
    """

    def init_mu_from_fo_grad(self, normalize: bool = False) -> dict[str, float]:
        grads: dict[int, torch.Tensor] = {}
        for group in self.param_groups:  # type: ignore[attr-defined]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    raise RuntimeError(
                        "init_mu_from_fo_grad requires a populated .grad on every trainable "
                        "parameter; run a first-order backward pass first."
                    )
                grads[id(p)] = p.grad.detach().float()

        fo_norm = math.sqrt(sum(g.norm().item() ** 2 for g in grads.values()))
        scale = 1.0 / fo_norm if (normalize and fo_norm > 0) else 1.0

        for group in self.param_groups:  # type: ignore[attr-defined]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]  # type: ignore[attr-defined]
                mu = state["mu"]
                mu.copy_((grads[id(p)] * scale).to(mu.dtype))
                state["mu_old"].copy_(mu)
                state["mu_old_norm_sq"] = mu.float().norm().item() ** 2

        return {"fo_grad_norm": fo_norm, "normalized": float(normalize)}


# ---------------------------------------------------------------------------
# LDSDRl
# ---------------------------------------------------------------------------

class LDSDRl(_MuFromFoGradMixin, ZeroOrderOptimizer):
    """ZO_RL from ZO-LDSD, adapted to OLMo's interface.

    Explores k random perturbation directions per step, selects the seed that
    minimises the loss, then refines with a two-sided finite difference along
    that direction. Maintains a per-parameter "trust direction" μ updated via
    an evolution-strategies natural-gradient step.

    Parameter update: sign(grad_accum) — sign-SGD style.
    Only a ``params_ratio`` fraction of parameters is perturbed per step.

    k-seed synchronisation
    ----------------------
    All k candidate seeds are derived deterministically from z_seed via
    ``np.random.RandomState(z_seed)``. All DDP ranks receive the same z_seed
    (synced by the trainer), so they agree on candidates and the optimal seed.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        beta: float = 0.9,
        k: int = 10,
        variance: float = 1e-3,
        lr_mu: Optional[float] = None,
        params_ratio: float = 0.1,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"Invalid beta: {beta}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not (0.0 < params_ratio <= 1.0):
            raise ValueError(f"params_ratio must be in (0, 1], got {params_ratio}")

        defaults = dict(lr=lr, zo_eps=zo_eps, weight_decay=weight_decay, beta=beta)
        super().__init__(params, defaults)
        self.k = k
        self.variance = variance
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self.params_ratio = params_ratio
        self._perturbation_mode = perturbation_mode
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

        # Flat list of trainable parameters for sparse subset selection.
        self._all_trainable = [
            p for g in self.param_groups for p in g["params"] if p.requires_grad
        ]

        # Initialise per-parameter state.
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state["step"] = 0
                state["grad_accum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # μ: initial random unit vector.
                mu = torch.randn_like(p, memory_format=torch.preserve_format)
                norm = torch.linalg.norm(mu)
                state["mu"] = mu.div_(norm) if norm > 0 else mu
                state["mu_old"] = state["mu"].detach().clone()
                state["mu_old_norm_sq"] = state["mu_old"].norm().item() ** 2

    # ------------------------------------------------------------------
    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics from the last step.

        projected_grad_abs
            |(f+ − f−) / 2| along the optimal direction.
        avg_mu_norm
            Mean L2 norm of μ across all trainable parameters.
        mu_alignment
            Cosine similarity between μ before and after the μ update
            (pooled across selected parameters).
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor],
        z_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        # Derive k candidate seeds deterministically — all DDP ranks agree.
        seed_rng = np.random.RandomState(z_seed)
        candidate_seeds = [int(seed_rng.randint(0, 1_000_000_000)) for _ in range(self.k)]

        # --- Phase 1: evaluate k candidates, pick the one with lowest loss ---
        loss_per_seed: dict[int, float] = {}
        for seed in candidate_seeds:
            selected_ids = self._sparse_perturb(seed, +1.0)
            loss_per_seed[seed] = closure().item()
            self._sparse_restore(seed, selected_ids, -1.0)

        optimal_seed = min(loss_per_seed, key=loss_per_seed.__getitem__)

        # --- Phase 2: two-sided finite difference along optimal direction ---
        # Reuse loss_plus from the candidate evaluation (same seed, same θ, same z).
        # Re-run the sparse perturb to recover selected_ids for the optimal seed.
        selected_ids = self._sparse_perturb(optimal_seed, +1.0)
        loss_plus = torch.tensor(loss_per_seed[optimal_seed])
        # Params are now at θ + ε·z; go directly to θ - ε·z.
        self._sparse_restore(optimal_seed, selected_ids, -2.0)
        loss_minus = closure()
        projected_grad = (loss_plus - loss_minus).item() / 2.0
        self._sparse_restore(optimal_seed, selected_ids, +1.0)  # restore to θ

        # --- Phase 3: parameter and μ update ---
        f_vals = torch.tensor([loss_per_seed[s] for s in candidate_seeds])
        f_sum = f_vals.sum()
        coeff = (f_vals * self.k - f_sum) / max(self.k - 1, 1)

        dot_sum = 0.0
        new_norm_sq = 0.0
        old_norm_sq = 0.0

        for group in self.param_groups:
            lr = group["lr"]
            zo_eps = group["zo_eps"]
            weight_decay = group["weight_decay"]
            beta = group["beta"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state["step"] += 1
                mu = state["mu"]

                # Gradient accumulation (selected params only).
                if id(p) in selected_ids:
                    gen = self._get_generator(p.device)
                    gen.manual_seed(optimal_seed)
                    z = torch.normal(mean=mu, std=self.variance, generator=gen)
                    grad_est = z.mul(projected_grad / zo_eps)
                    state["grad_accum"].mul_(beta).add_(grad_est, alpha=1.0 - beta)

                # Weight decay (all params).
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # SignSGD update (all params).
                p.data.add_(torch.sign(state["grad_accum"]), alpha=-lr)

                # μ update (selected params only).
                if id(p) in selected_ids:
                    mu_diff = torch.zeros_like(mu)
                    for i, seed in enumerate(candidate_seeds):
                        gen = self._get_generator(p.device)
                        gen.manual_seed(seed)
                        z_i = torch.normal(mean=mu, std=self.variance, generator=gen)
                        mu_diff.add_(mu - z_i, alpha=coeff[i].item())

                    g_mu = mu_diff.neg_().div_(self.k * self.variance ** 2)
                    state["mu"].add_(g_mu, alpha=-self.lr_mu)

                    dot_sum += torch.dot(state["mu_old"].view(-1), state["mu"].view(-1)).item()
                    new_norm_sq += state["mu"].norm().item() ** 2
                    old_norm_sq += state["mu_old_norm_sq"]

                    state["mu_old"].copy_(state["mu"])
                    state["mu_old_norm_sq"] = state["mu"].norm().item() ** 2

        # Metrics.
        n = max(len(self._all_trainable), 1)
        self._last_metrics["projected_grad_abs"] = abs(projected_grad)
        self._last_metrics["avg_mu_norm"] = (
            sum(self.state[p]["mu"].norm().item() for p in self._all_trainable) / n
        )
        if new_norm_sq > 0 and old_norm_sq > 0:
            self._last_metrics["mu_alignment"] = dot_sum / (
                math.sqrt(new_norm_sq) * math.sqrt(old_norm_sq)
            )
        return loss_plus

    # ------------------------------------------------------------------
    def _sparse_perturb(self, seed: int, scaling_factor: float) -> set[int]:
        """Perturb a sparse subset of params; return their Python ids."""
        ref_device = self._all_trainable[0].device
        gen = self._get_generator(ref_device)
        gen.manual_seed(seed)
        n = max(1, int(len(self._all_trainable) * self.params_ratio))
        perm = torch.randperm(len(self._all_trainable), device=ref_device, generator=gen)[:n]
        selected_ids = {id(self._all_trainable[int(i)]) for i in perm}

        for group in self.param_groups:
            zo_eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad or id(p) not in selected_ids:
                    continue
                mu = self.state[p]["mu"]
                gen = self._get_generator(p.device)
                gen.manual_seed(seed)  # reseed per param (faithful to ZO_RL)
                z = torch.normal(mean=mu, std=self.variance, generator=gen)
                p.data.add_(z, alpha=scaling_factor * zo_eps)

        return selected_ids

    def _sparse_restore(
        self, seed: int, selected_ids: set[int], scaling_factor: float
    ) -> None:
        """Re-apply the same z's with a new scaling_factor to restore/offset params."""
        for group in self.param_groups:
            zo_eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad or id(p) not in selected_ids:
                    continue
                mu = self.state[p]["mu"]
                gen = self._get_generator(p.device)
                gen.manual_seed(seed)
                z = torch.normal(mean=mu, std=self.variance, generator=gen)
                p.data.add_(z, alpha=scaling_factor * zo_eps)


# ---------------------------------------------------------------------------
# LDSDRlAdaMM
# ---------------------------------------------------------------------------

class LDSDRlAdaMM(_MuFromFoGradMixin, ZeroOrderOptimizer):
    """ZO_RL_AdaMM from ZO-LDSD, adapted to OLMo's interface.

    Combines RL-based direction learning (same k-candidate exploration as
    LDSDRl) with AMSGrad-style second-moment tracking for the parameter update.
    All parameters are perturbed every step (no sparse selection).

    The generator is seeded once per perturbation call and advances
    sequentially through all parameters (no per-param reseeding).
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        k: int = 10,
        variance: float = 1e-3,
        lr_mu: Optional[float] = None,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 < b2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        defaults = dict(lr=lr, zo_eps=zo_eps, weight_decay=weight_decay, betas=betas)
        super().__init__(params, defaults)
        self.k = k
        self.variance = variance
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self._perturbation_mode = perturbation_mode
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

        self._all_trainable = [
            p for g in self.param_groups for p in g["params"] if p.requires_grad
        ]

        # Initialise moments and μ (μ starts at zero, matches ZO_RL_AdaMM default).
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state["step"] = 0
                # Store moments and μ in bf16 to halve optimizer state memory.
                # With amp_bf16, params are fp32 master weights; 5 fp32 buffers
                # would cost ~30 GB for 1.5B params.  bf16 costs ~7.5 GB.
                _bf16 = torch.bfloat16
                state["exp_avg"] = torch.zeros_like(p, dtype=_bf16, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=_bf16, memory_format=torch.preserve_format)
                state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=_bf16, memory_format=torch.preserve_format)
                state["mu"] = torch.zeros_like(p, dtype=_bf16, memory_format=torch.preserve_format)
                state["mu_old"] = state["mu"].detach().clone()
                state["mu_old_norm_sq"] = 0.0

    # ------------------------------------------------------------------
    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics from the last step.

        projected_grad_abs
            |(f+ − f−) / 2| along the optimal direction.
        avg_mu_norm
            Mean L2 norm of μ across all trainable parameters.
        avg_mu_norm_diff
            Mean per-parameter change in ‖μ‖ this step.
        avg_mu_grad_norm
            Mean per-parameter L2 norm of the μ gradient.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor],
        z_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        seed_rng = np.random.RandomState(z_seed)
        candidate_seeds = [int(seed_rng.randint(0, 1_000_000_000)) for _ in range(self.k)]

        # --- Phase 1: evaluate k candidates ---
        loss_per_seed: dict[int, float] = {}
        for seed in candidate_seeds:
            self._perturb_full(seed, +1.0)
            loss_per_seed[seed] = closure().item()
            self._perturb_full(seed, -1.0)  # restore

        optimal_seed = min(loss_per_seed, key=loss_per_seed.__getitem__)

        # --- Phase 2: two-sided FD along optimal direction ---
        # Perturb to θ - ε·z directly (no need for +1 first).
        self._perturb_full(optimal_seed, -1.0)
        loss_minus = closure()
        # Recreate f+ as a tensor on the same device/dtype as closure() output.
        loss_plus = loss_minus.detach().new_tensor(loss_per_seed[optimal_seed])
        projected_grad = (loss_plus - loss_minus).item() / 2.0
        self._perturb_full(optimal_seed, +1.0)  # restore to θ

        # --- Phase 3: AdaMM parameter update ---
        f_vals = torch.tensor([loss_per_seed[s] for s in candidate_seeds])
        f_sum = f_vals.sum()
        coeff = (f_vals * self.k - f_sum) / max(self.k - 1, 1)

        # Seed once; generator advances sequentially through all params.
        ref_device = self._all_trainable[0].device
        ref_gen = self._get_generator(ref_device)
        ref_gen.manual_seed(optimal_seed)

        for group in self.param_groups:
            lr = group["lr"]
            zo_eps = group["zo_eps"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state["step"] += 1
                mu = state["mu"]
                pgen = self._get_generator(p.device)
                if pgen is not ref_gen:
                    pgen.manual_seed(optimal_seed)
                z = torch.normal(mean=mu, std=self.variance, generator=pgen)
                # In-place scale: avoids allocating a separate grad tensor.
                z.mul_(projected_grad / zo_eps)
                grad = z

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # out= avoids allocating a new tensor on every step.
                torch.maximum(state["max_exp_avg_sq"], state["exp_avg_sq"], out=state["max_exp_avg_sq"])
                # AMSGrad update: cast bf16 states to param dtype (fp32 in amp_bf16).
                p_dtype = p.data.dtype
                p.data.addcdiv_(
                    state["exp_avg"].to(p_dtype),
                    (state["max_exp_avg_sq"].sqrt() + 1e-10).to(p_dtype),
                    value=-lr,
                )

        # --- Phase 4: μ update ---
        mu_norm_diff_sq = 0.0
        mu_grad_norm_sq = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                mu = state["mu"]
                mu_diff = torch.zeros_like(mu)

                for i, seed in enumerate(candidate_seeds):
                    gen = self._get_generator(p.device)
                    gen.manual_seed(seed)
                    z_i = torch.normal(mean=mu, std=self.variance, generator=gen)
                    # Compute (mu - z_i) in-place to avoid an extra allocation.
                    z_i.neg_().add_(mu)
                    mu_diff.add_(z_i, alpha=coeff[i].item())

                g_mu = mu_diff.neg_().div_(self.k * self.variance ** 2)
                state["mu"].add_(g_mu, alpha=-self.lr_mu)

                mu_norm_diff_sq += (state["mu"] - state["mu_old"]).norm().item() ** 2
                mu_grad_norm_sq += g_mu.norm().item() ** 2

                state["mu_old"].copy_(state["mu"])
                state["mu_old_norm_sq"] = state["mu"].norm().item() ** 2

        # Metrics.
        n = max(len(self._all_trainable), 1)
        self._last_metrics["projected_grad_abs"] = abs(projected_grad)
        self._last_metrics["avg_mu_norm"] = (
            sum(self.state[p]["mu"].norm().item() for p in self._all_trainable) / n
        )
        self._last_metrics["avg_mu_norm_diff"] = math.sqrt(max(mu_norm_diff_sq, 0.0)) / n
        self._last_metrics["avg_mu_grad_norm"] = math.sqrt(max(mu_grad_norm_sq, 0.0)) / n
        return loss_plus

    # ------------------------------------------------------------------
    def _perturb_full(self, seed: int, scaling_factor: float) -> None:
        """Perturb ALL params; generator seeded once and advances sequentially."""
        ref_device = self._all_trainable[0].device
        ref_gen = self._get_generator(ref_device)
        ref_gen.manual_seed(seed)
        for group in self.param_groups:
            zo_eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                mu = self.state[p]["mu"]
                pgen = self._get_generator(p.device)
                if pgen is not ref_gen:
                    pgen.manual_seed(seed)
                z = torch.normal(mean=mu, std=self.variance, generator=pgen)
                # mu is bf16; cast z to param dtype (fp32 in amp_bf16) for in-place add.
                p.data.add_(z.to(p.data.dtype), alpha=scaling_factor * zo_eps)


# ---------------------------------------------------------------------------
# LDSDRlSgd
# ---------------------------------------------------------------------------

class LDSDRlSgd(_MuFromFoGradMixin, ZeroOrderOptimizer):
    """ZO_RL_SGD from ZO-LDSD adapted to OLMo's interface.

    RL-based direction learning (k-candidate exploration, same as LDSDRlAdaMM)
    with vanilla SGD (optional momentum) for the parameter update.
    All parameters are perturbed every step (no sparse selection).

    Algorithm:
      Phase 1 — evaluate k candidate perturbations, select the one with lowest loss.
      Phase 2 — two-sided FD along the optimal direction (reuses loss_plus from phase 1).
      Phase 3 — SGD (+ optional momentum) parameter update using the projected gradient.
      Phase 4 — μ direction update via ES natural gradient.

    Total forward passes per step: k + 1 (k exploration + 1 for loss_minus in FD).
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        momentum: float = 0.0,
        k: int = 10,
        variance: float = 1e-3,
        lr_mu: Optional[float] = None,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        defaults = dict(lr=lr, zo_eps=zo_eps, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.k = k
        self.variance = variance
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self._perturbation_mode = perturbation_mode
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

        self._all_trainable = [
            p for g in self.param_groups for p in g["params"] if p.requires_grad
        ]

        # μ in bf16 to halve optimizer state memory (same rationale as LDSDRlAdaMM).
        # momentum_buffer is created lazily on the first step that uses it.
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state["step"] = 0
                _bf16 = torch.bfloat16
                state["mu"] = torch.zeros_like(p, dtype=_bf16, memory_format=torch.preserve_format)
                state["mu_old"] = state["mu"].detach().clone()

    # ------------------------------------------------------------------
    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics from the last step.

        projected_grad_abs
            |(f+ − f−) / 2| along the optimal direction.
        avg_mu_norm
            Mean L2 norm of μ across all trainable parameters.
        avg_mu_norm_diff
            Mean per-parameter change in ‖μ‖ this step.
        avg_mu_grad_norm
            Mean per-parameter L2 norm of the μ gradient.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor],
        z_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        seed_rng = np.random.RandomState(z_seed)
        candidate_seeds = [int(seed_rng.randint(0, 1_000_000_000)) for _ in range(self.k)]

        # --- Phase 1: evaluate k candidates ---
        loss_per_seed: dict[int, float] = {}
        for seed in candidate_seeds:
            self._perturb_full(seed, +1.0)
            loss_per_seed[seed] = closure().item()
            self._perturb_full(seed, -1.0)  # restore

        optimal_seed = min(loss_per_seed, key=loss_per_seed.__getitem__)

        # --- Phase 2: two-sided FD along optimal direction ---
        # Reuse loss_plus already computed in phase 1 (k+1 passes total, not k+2).
        self._perturb_full(optimal_seed, -1.0)
        loss_minus = closure()
        # Recreate f+ as a tensor on the same device/dtype as closure() output.
        loss_plus = loss_minus.detach().new_tensor(loss_per_seed[optimal_seed])
        projected_grad = (loss_plus - loss_minus).item() / 2.0
        self._perturb_full(optimal_seed, +1.0)  # restore to θ

        # --- Phase 3: SGD (+ optional momentum) parameter update ---
        f_vals = torch.tensor([loss_per_seed[s] for s in candidate_seeds])
        f_sum = f_vals.sum()
        coeff = (f_vals * self.k - f_sum) / max(self.k - 1, 1)

        ref_device = self._all_trainable[0].device
        ref_gen = self._get_generator(ref_device)
        ref_gen.manual_seed(optimal_seed)

        for group in self.param_groups:
            lr = group["lr"]
            zo_eps = group["zo_eps"]
            weight_decay = group["weight_decay"]
            mom = group["momentum"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state["step"] += 1
                mu = state["mu"]
                pgen = self._get_generator(p.device)
                if pgen is not ref_gen:
                    pgen.manual_seed(optimal_seed)
                z = torch.normal(mean=mu, std=self.variance, generator=pgen)
                # In-place scale to grad; cast to param dtype (fp32 in amp_bf16).
                z.mul_(projected_grad / zo_eps)
                grad = z.to(p.data.dtype)

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                if mom != 0.0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(grad)
                    p.data.add_(buf.to(p.data.dtype), alpha=-lr)
                else:
                    p.data.add_(grad, alpha=-lr)

        # --- Phase 4: μ update ---
        mu_norm_diff_sq = 0.0
        mu_grad_norm_sq = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                mu = state["mu"]
                mu_diff = torch.zeros_like(mu)

                for i, seed in enumerate(candidate_seeds):
                    gen = self._get_generator(p.device)
                    gen.manual_seed(seed)
                    z_i = torch.normal(mean=mu, std=self.variance, generator=gen)
                    z_i.neg_().add_(mu)
                    mu_diff.add_(z_i, alpha=coeff[i].item())

                g_mu = mu_diff.neg_().div_(self.k * self.variance ** 2)
                state["mu"].add_(g_mu, alpha=-self.lr_mu)

                mu_norm_diff_sq += (state["mu"] - state["mu_old"]).norm().item() ** 2
                mu_grad_norm_sq += g_mu.norm().item() ** 2

                state["mu_old"].copy_(state["mu"])

        # Metrics.
        n = max(len(self._all_trainable), 1)
        self._last_metrics["projected_grad_abs"] = abs(projected_grad)
        self._last_metrics["avg_mu_norm"] = (
            sum(self.state[p]["mu"].norm().item() for p in self._all_trainable) / n
        )
        self._last_metrics["avg_mu_norm_diff"] = math.sqrt(max(mu_norm_diff_sq, 0.0)) / n
        self._last_metrics["avg_mu_grad_norm"] = math.sqrt(max(mu_grad_norm_sq, 0.0)) / n
        return loss_plus

    # ------------------------------------------------------------------
    def _perturb_full(self, seed: int, scaling_factor: float) -> None:
        """Perturb ALL params; generator seeded once and advances sequentially."""
        ref_device = self._all_trainable[0].device
        ref_gen = self._get_generator(ref_device)
        ref_gen.manual_seed(seed)
        for group in self.param_groups:
            zo_eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                mu = self.state[p]["mu"]
                pgen = self._get_generator(p.device)
                if pgen is not ref_gen:
                    pgen.manual_seed(seed)
                z = torch.normal(mean=mu, std=self.variance, generator=pgen)
                # mu is bf16; cast z to param dtype (fp32 in amp_bf16) for in-place add.
                p.data.add_(z.to(p.data.dtype), alpha=scaling_factor * zo_eps)


# ---------------------------------------------------------------------------
# LDSDRlKron
# ---------------------------------------------------------------------------

class LDSDRlKron(_MuFromFoGradMixin, ZeroOrderOptimizer):
    """ZO-RL's learned N(μ, σ²) policy wrapped in KronZO's directional selection.

    This combines two optimizers:

      * **Perturbation sampler (ZO-RL).** Directions are drawn ``z ~ N(μ, σ²)``
        where μ is a per-parameter "trust direction" learned online. This is
        KronZO Algorithm 2's Line 9 (``Z_i = A_i ⊗ B``) replaced by the ZO-RL
        sampler.
      * **Direction selection + acceptance (KronZO, Lines 8-19).** Each step
        samples ``q`` probes; for each it forms the SPSA scalar
        ``c_i = (L(θ+εz_i) − L(θ−εz_i)) / 2ε`` and *evaluates the candidate step*
        ``L(θ − α c_i z_i)``. Only the single best-decreasing direction is kept,
        and it is applied only if its loss beats the worst of the last ``h``
        accepted losses (sliding window); otherwise the step is skipped. Cost:
        ``1 + 3q`` forward passes (two_side) or ``1 + 2q`` (one_side).

    The μ direction is still learned with ZO-RL's ES / score-function update
    (Phase 4), **unchanged** — it is fed by the ``q`` probe returns ``f_i``
    (perturbed loss ``L(θ+εz_i)`` by default, or the candidate loss).

    Configurable via ``apply`` (how θ consumes the chosen direction: ``kron`` /
    ``sign_sgd`` / ``sgd`` / ``adamm``), ``mu_return`` (``perturbed`` / ``candidate``)
    and ``mu_init`` (``zero`` / ``random``; FO-gradient init is orthogonal, via the
    trainer's ``ldsd_rl_mu_init_from_fo`` path — this class subclasses the same
    ``_MuFromFoGradMixin``).

    A single active subset of parameters (``params_ratio``) is chosen per step
    from ``z_seed`` and shared by all ``q`` probes, the θ update and the μ update,
    so the ES gradient stays well-defined. ``params_ratio=1.0`` gives the dense,
    KronZO-like regime.

    DDP: all ranks receive the same ``z_seed`` (synced by the trainer), so they
    agree on the subset, the ``q`` candidate seeds, every loss (the closure
    all-reduces), the accept/skip decision and the update, with no gradient
    communication. FSDP is not supported (blocked in ``build_optimizer``).
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        query_budget: int = 10,
        history_length: int = 10,
        variance: float = 1e-3,
        beta: float = 0.9,
        betas: tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.0,
        lr_mu: Optional[float] = None,
        params_ratio: float = 1.0,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        apply: str = "kron",
        mu_return: str = "perturbed",
        mu_init: str = "zero",
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if query_budget < 1:
            raise ValueError(f"query_budget must be >= 1, got {query_budget}")
        if history_length < 0:
            raise ValueError(f"history_length must be >= 0, got {history_length}")
        if not (0.0 < params_ratio <= 1.0):
            raise ValueError(f"params_ratio must be in (0, 1], got {params_ratio}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        if apply not in ("kron", "sign_sgd", "sgd", "adamm"):
            raise ValueError(f"apply must be kron|sign_sgd|sgd|adamm, got {apply!r}")
        if mu_return not in ("perturbed", "candidate"):
            raise ValueError(f"mu_return must be 'perturbed' or 'candidate', got {mu_return!r}")
        if mu_init not in ("zero", "random"):
            raise ValueError(f"mu_init must be 'zero' or 'random', got {mu_init!r}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            weight_decay=weight_decay,
            beta=beta,
            betas=betas,
            momentum=momentum,
            # Exposed so Trainer._zo_forward_passes_per_step accounts for 1 + (3|2)q.
            query_budget=query_budget,
            perturbation_mode=perturbation_mode,
        )
        super().__init__(params, defaults)
        self.q = query_budget
        self.variance = variance
        self.lr_mu = lr_mu if lr_mu is not None else lr
        self.params_ratio = params_ratio
        self._perturbation_mode = perturbation_mode
        self._apply = apply
        self._mu_return = mu_return
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

        self._window: list[float] = [float("inf")] * history_length
        self._step_count = 0
        self._n_steps = 0
        self._n_accepted = 0

        self._all_trainable = [
            p for g in self.param_groups for p in g["params"] if p.requires_grad
        ]

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state["step"] = 0
                if mu_init == "random":
                    mu = torch.randn_like(p, memory_format=torch.preserve_format)
                    norm = torch.linalg.norm(mu)
                    mu = mu.div_(norm) if norm > 0 else mu
                else:  # "zero"
                    mu = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["mu"] = mu
                state["mu_old"] = mu.detach().clone()
                state["mu_old_norm_sq"] = mu.float().norm().item() ** 2
                # apply-mode-specific buffers (allocated only when needed).
                if apply == "sign_sgd":
                    state["grad_accum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                elif apply == "adamm":
                    _bf16 = torch.bfloat16
                    state["exp_avg"] = torch.zeros_like(p, dtype=_bf16, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=_bf16, memory_format=torch.preserve_format)
                    state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=_bf16, memory_format=torch.preserve_format)
                # "sgd" momentum_buffer is created lazily on first use; "kron" needs no buffer.

    # ------------------------------------------------------------------
    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """KronZO/ZO-RL diagnostics from the last step.

        projected_grad_abs   mean |c_i| over the q probes (SPSA signal strength).
        grad_est_norm        ‖applied update direction‖ (0.0 when the step was skipped).
        update_accepted      1.0 if the directional step was applied this step, else 0.0.
        accept_rate          running fraction of accepted steps.
        avg_mu_norm          mean ‖μ‖ over trainable params.
        avg_mu_grad_norm     mean per-param ‖g_μ‖ this step.
        avg_mu_norm_diff     mean per-param ‖μ − μ_old‖ this step.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    # Subset selection and perturbation helpers
    # ------------------------------------------------------------------
    def _select_ids(self, seed: int) -> set[int]:
        """Deterministically pick the active parameter subset for this step."""
        if self.params_ratio >= 1.0:
            return {id(p) for p in self._all_trainable}
        ref_device = self._all_trainable[0].device
        gen = self._get_generator(ref_device)
        gen.manual_seed(seed)
        n = max(1, int(len(self._all_trainable) * self.params_ratio))
        perm = torch.randperm(len(self._all_trainable), device=ref_device, generator=gen)[:n]
        return {id(self._all_trainable[int(i)]) for i in perm}

    def _z(self, p: torch.Tensor, seed: int) -> torch.Tensor:
        """Regenerate the ZO-RL probe direction z ~ N(μ, σ²) for one param from seed.

        Reseeds per parameter (faithful to ZO_RL); μ is frozen within the step so
        every regeneration point (probe, candidate, apply, μ-update) agrees.
        """
        mu = self.state[p]["mu"]
        gen = self._get_generator(p.device)
        gen.manual_seed(seed)
        return torch.normal(mean=mu, std=self.variance, generator=gen)

    def _apply_eps(self, seed: int, ids: set[int], sign: float) -> None:
        """θ += sign · ε · z   (finite-difference probe; single global ε like KronZO)."""
        eps = self.defaults["zo_eps"]
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad or id(p) not in ids:
                    continue
                z = self._z(p, seed)
                p.data.add_(z.to(p.data.dtype), alpha=sign * eps)

    def _apply_step(self, seed: int, ids: set[int], c: float, sign: float) -> None:
        """θ += sign · lr · c · z   (candidate / trial step, per-group lr)."""
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if not p.requires_grad or id(p) not in ids:
                    continue
                z = self._z(p, seed)
                p.data.add_(z.to(p.data.dtype), alpha=sign * lr * c)

    def _push_window(self, loss: float) -> None:
        """Replace the worst (max) loss in the acceptance window with ``loss``."""
        if not self._window:
            return
        j = max(range(len(self._window)), key=lambda k: self._window[k])
        self._window[j] = loss

    # ------------------------------------------------------------------
    # Parameter update (once the best direction is chosen)
    # ------------------------------------------------------------------
    def _apply_update(self, seed: int, c: float, ids: set[int]) -> float:
        """Consume the chosen direction ``grad = c · z`` per the ``apply`` mode.

        Returns the L2 norm of the raw ZO update direction ``c · z`` (before the
        apply-mode transform), summed over the active subset.
        """
        grad_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if not p.requires_grad or id(p) not in ids:
                    continue
                state = self.state[p]
                state["step"] += 1
                z = self._z(p, seed)
                grad = z.mul(c)  # ZO-RL gradient estimate: z · (projected_grad/ε) = z · c
                grad_sq += grad.float().norm().item() ** 2
                p_dtype = p.data.dtype

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                if self._apply == "kron":
                    p.data.add_(grad.to(p_dtype), alpha=-lr)
                elif self._apply == "sign_sgd":
                    beta = group["beta"]
                    buf = state["grad_accum"]
                    buf.mul_(beta).add_(grad, alpha=1.0 - beta)
                    p.data.add_(torch.sign(buf).to(p_dtype), alpha=-lr)
                elif self._apply == "sgd":
                    mom = group["momentum"]
                    if mom != 0.0:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )
                        buf = state["momentum_buffer"]
                        buf.mul_(mom).add_(grad)
                        p.data.add_(buf.to(p_dtype), alpha=-lr)
                    else:
                        p.data.add_(grad.to(p_dtype), alpha=-lr)
                elif self._apply == "adamm":
                    beta1, beta2 = group["betas"]
                    state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    torch.maximum(state["max_exp_avg_sq"], state["exp_avg_sq"], out=state["max_exp_avg_sq"])
                    p.data.addcdiv_(
                        state["exp_avg"].to(p_dtype),
                        (state["max_exp_avg_sq"].sqrt() + 1e-10).to(p_dtype),
                        value=-lr,
                    )
        return math.sqrt(max(grad_sq, 0.0))

    # ------------------------------------------------------------------
    # μ update (ES / score-function natural gradient) — unchanged from ZO-RL
    # ------------------------------------------------------------------
    def _mu_update(self, candidate_seeds: list[int], returns: list[float], ids: set[int]) -> None:
        q = len(candidate_seeds)
        f_vals = torch.tensor(returns)
        coeff = (f_vals * q - f_vals.sum()) / max(q - 1, 1)

        dot_sum = 0.0
        new_norm_sq = 0.0
        old_norm_sq = 0.0
        mu_norm_diff_sq = 0.0
        mu_grad_norm_sq = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad or id(p) not in ids:
                    continue
                state = self.state[p]
                mu = state["mu"]
                mu_diff = torch.zeros_like(mu)
                for i, seed in enumerate(candidate_seeds):
                    gen = self._get_generator(p.device)
                    gen.manual_seed(seed)
                    z_i = torch.normal(mean=mu, std=self.variance, generator=gen)
                    mu_diff.add_(mu - z_i, alpha=coeff[i].item())

                g_mu = mu_diff.neg_().div_(q * self.variance ** 2)
                state["mu"].add_(g_mu, alpha=-self.lr_mu)

                dot_sum += torch.dot(state["mu_old"].view(-1).float(), state["mu"].view(-1).float()).item()
                new_norm_sq += state["mu"].float().norm().item() ** 2
                old_norm_sq += state["mu_old_norm_sq"]
                mu_norm_diff_sq += (state["mu"] - state["mu_old"]).float().norm().item() ** 2
                mu_grad_norm_sq += g_mu.float().norm().item() ** 2

                state["mu_old"].copy_(state["mu"])
                state["mu_old_norm_sq"] = state["mu"].float().norm().item() ** 2

        n = max(len(self._all_trainable), 1)
        self._last_metrics["avg_mu_norm"] = (
            sum(self.state[p]["mu"].float().norm().item() for p in self._all_trainable) / n
        )
        self._last_metrics["avg_mu_norm_diff"] = math.sqrt(max(mu_norm_diff_sq, 0.0)) / n
        self._last_metrics["avg_mu_grad_norm"] = math.sqrt(max(mu_grad_norm_sq, 0.0)) / n
        if new_norm_sq > 0 and old_norm_sq > 0:
            self._last_metrics["mu_alignment"] = dot_sum / (
                math.sqrt(new_norm_sq) * math.sqrt(old_norm_sq)
            )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor],
        z_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        q = self.q
        eps = self.defaults["zo_eps"]
        two_side = self._perturbation_mode == "two_side"

        # One active subset per step, shared by all probes / updates.
        ids = self._select_ids(z_seed)
        seed_rng = np.random.RandomState(z_seed)
        candidate_seeds = [int(seed_rng.randint(0, 1_000_000_000)) for _ in range(q)]

        # KronZO Line 5: centre loss at θ_k (also reported to the trainer).
        center_loss = closure()
        center = center_loss.item()
        l_best = center
        best_seed: Optional[int] = None
        best_c: Optional[float] = None
        abs_c_sum = 0.0
        returns: list[float] = []

        for seed_i in candidate_seeds:
            # SPSA scalar c_i along z_i.
            self._apply_eps(seed_i, ids, +1.0)
            loss_plus = closure().item()
            if two_side:
                self._apply_eps(seed_i, ids, -2.0)
                loss_minus = closure().item()
                self._apply_eps(seed_i, ids, +1.0)  # restore θ
                projected_grad = (loss_plus - loss_minus) / 2.0
            else:
                self._apply_eps(seed_i, ids, -1.0)  # restore θ
                projected_grad = loss_plus - center
            c_i = projected_grad / eps
            abs_c_sum += abs(c_i)

            # Evaluate the candidate step L(θ − lr·c_i·z_i), then restore.
            self._apply_step(seed_i, ids, c_i, -1.0)
            loss_cand = closure().item()
            self._apply_step(seed_i, ids, c_i, +1.0)

            returns.append(loss_cand if self._mu_return == "candidate" else loss_plus)

            if loss_cand < l_best:
                l_best = loss_cand
                best_seed = seed_i
                best_c = c_i

        # Directional update with sliding-window acceptance (KronZO Lines 16-19).
        window_max = max(self._window) if self._window else float("inf")
        accepted = (best_seed is not None) and (l_best <= window_max)
        grad_norm = 0.0
        if accepted:
            assert best_seed is not None and best_c is not None
            grad_norm = self._apply_update(best_seed, best_c, ids)
        self._push_window(l_best)

        # μ update (ES / score-function) — unchanged; fed by the q probe returns.
        self._mu_update(candidate_seeds, returns, ids)

        self._step_count += 1
        self._n_steps += 1
        if accepted:
            self._n_accepted += 1

        self._last_metrics["projected_grad_abs"] = abs_c_sum / q
        self._last_metrics["grad_est_norm"] = grad_norm
        self._last_metrics["update_accepted"] = 1.0 if accepted else 0.0
        self._last_metrics["accept_rate"] = self._n_accepted / self._n_steps
        return center_loss
