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

        self._last_metrics["grad_est_norm"] = math.sqrt(grad_sum_sq)


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
        self._last_metrics["grad_est_norm"] = math.sqrt(grad_sum_sq)


# ---------------------------------------------------------------------------
# LDSDRl
# ---------------------------------------------------------------------------

class LDSDRl(ZeroOrderOptimizer):
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

class LDSDRlAdaMM(ZeroOrderOptimizer):
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
        # Reuse loss_plus from the candidate evaluation (same seed, same θ, same z).
        loss_plus = torch.tensor(loss_per_seed[optimal_seed])
        # Perturb to θ - ε·z directly (no need for +1 first).
        self._perturb_full(optimal_seed, -1.0)
        loss_minus = closure()
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
