"""
MeZO, LOZO, ZoAdam, and ZOMuon zero-order optimizers for OLMo training.

Adapted from ``llm-base-zero-order`` (see also LOZO paper / optsuite).
ZoAdam combines the MeZO / SPSA gradient estimate with AdamW-style moment updates.
ZOMuon implements "Powering Up Zeroth-Order Training via Subspace Gradient
Orthogonalization" (ZO-Muon): projects perturbations through a Haar-distributed
orthogonal matrix P, accumulates multiple gradient estimates in the low-rank
subspace, then orthogonalizes the result with Newton-Schulz before updating.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import Optimizer


class ZeroOrderOptimizer(Optimizer):
    """Marker base class; the trainer uses this to skip backward and gradient clipping."""

    pass


class VectorSampler:
    def __init__(self, sampler_type: str = "standard_normal"):
        if sampler_type not in ("standard_normal", "lp_sphere"):
            raise ValueError(
                f"Unknown sampler_type: {sampler_type!r}. Choose 'standard_normal' or 'lp_sphere'."
            )
        self.sampler_type = sampler_type

    def sample(self, shape: torch.Size, device: torch.device, generator: torch.Generator) -> torch.Tensor:
        if self.sampler_type == "standard_normal":
            return torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
        x = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
        norm = x.norm()
        return x / norm if norm > 0 else x


class MeZO(ZeroOrderOptimizer):
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

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def _reset_generators(self, z_seed: int) -> None:
        """Seed each device's generator once before iterating parameters.

        Calling manual_seed inside the parameter loop would reset the RNG state
        for every parameter, causing all parameters of the same shape to receive
        the identical perturbation vector.  Seeding once here lets the generator
        advance sequentially through all parameters so each gets an independent z.
        Both _perturb and _apply_update call this before their loops, so they
        reproduce the same per-parameter z from the same seed.
        """
        seen: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.device not in seen:
                    self._get_generator(p.device).manual_seed(z_seed)
                    seen.add(p.device)

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics collected during the last step.

        Metrics
        -------
        projected_grad_abs
            Absolute value of the SPSA scalar estimate ``(f⁺ − f⁻) / (2ε)``
            (or ``(f⁺ − f⁻) / ε`` for one_side mode).  Indicates the
            magnitude of the loss difference along the perturbation direction.
        grad_est_norm
            Global L2 norm of the gradient estimate vector
            ``z * projected_grad / ε`` concatenated across all parameters
            (before lr scaling and weight decay).  Comparable to a gradient
            norm in first-order training.
        grad_est_norm_per_z_rms
            ``grad_est_norm`` divided by the RMS norm of the raw probe
            directions ``z`` (same samples as in the estimate).  For MeZO this
            equals ``|projected_grad| / ε`` and removes the ``√(num_params)``
            scale from ``||z||₂``.  Not the norm of a direction-normalized
            ``ĝ`` (that would be trivially 1).
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        self._perturb(z_seed, scaling_factor=+1.0)
        loss_plus = closure()

        if self.defaults["perturbation_mode"] == "two_side":
            self._perturb(z_seed, scaling_factor=-2.0)
            loss_minus = closure()
            projected_grad = (loss_plus - loss_minus).item() / 2.0
            self._perturb(z_seed, scaling_factor=+1.0)
        else:
            self._perturb(z_seed, scaling_factor=-1.0)
            loss_minus = closure()
            projected_grad = (loss_plus - loss_minus).item()

        self._apply_update(z_seed, projected_grad)
        self._last_metrics["projected_grad_abs"] = abs(projected_grad)
        return loss_plus

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

    def _apply_update(self, z_seed: int, projected_grad: float) -> None:
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
                grad_est = z.mul_(projected_grad / eps)
                # Accumulate before weight decay to track pure SPSA estimate norm.
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


class ZoAdam(ZeroOrderOptimizer):
    """
    Zeroth-order Adam (AdamW-style): same random-probe SPSA estimate as MeZO, then
    first/second moment tracking and bias-corrected update like AdamW (decoupled weight decay).
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        vector_sampling_type: str = "standard_normal",
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        b1, b2 = betas
        if not (0.0 <= b1 <= 1.0 and 0.0 <= b2 <= 1.0):
            raise ValueError(f"betas must be in [0, 1], got {betas}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            betas=betas,
            eps=eps,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.vector_sampler = VectorSampler(vector_sampling_type)
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def _reset_generators(self, z_seed: int) -> None:
        """Seed each device's generator once before iterating parameters.

        See MeZO._reset_generators for rationale.
        """
        seen: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.device not in seen:
                    self._get_generator(p.device).manual_seed(z_seed)
                    seen.add(p.device)

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics collected during the last step.

        Metrics
        -------
        projected_grad_abs
            Absolute value of the SPSA scalar ``(f⁺ − f⁻) / (2ε)``.
        grad_est_norm
            Global L2 norm of ``z * projected_grad / ε`` across all parameters
            (raw SPSA estimate, before Adam preconditioning and lr scaling).
        update_norm
            Global L2 norm of the actual Adam-preconditioned update applied
            to the parameters.  Reflects the effect of moment scaling.
        grad_est_norm_per_z_rms
            ``grad_est_norm`` divided by the RMS norm of the raw ``z`` probes
            (same as in MeZO).  Equals ``|projected_grad| / ε`` for the SPSA
            block before Adam.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        self._perturb(z_seed, scaling_factor=+1.0)
        loss_plus = closure()

        if self.defaults["perturbation_mode"] == "two_side":
            self._perturb(z_seed, scaling_factor=-2.0)
            loss_minus = closure()
            projected_grad = (loss_plus - loss_minus).item() / 2.0
            self._perturb(z_seed, scaling_factor=+1.0)
        else:
            self._perturb(z_seed, scaling_factor=-1.0)
            loss_minus = closure()
            projected_grad = (loss_plus - loss_minus).item()

        self._apply_update(z_seed, projected_grad)
        self._last_metrics["projected_grad_abs"] = abs(projected_grad)
        return loss_plus

    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        self._reset_generators(z_seed)
        for group in self.param_groups:
            zo_eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                p.data.add_(z, alpha=scaling_factor * zo_eps)

    def _apply_update(self, z_seed: int, projected_grad: float) -> None:
        self._reset_generators(z_seed)
        grad_sum_sq = 0.0
        z_sum_sq = 0.0
        update_sum_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            zo_eps = group["zo_eps"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            adam_eps = group["eps"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                gen = self._get_generator(p.device)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                z_sum_sq += z.to(dtype=torch.float32).norm().item() ** 2
                g = z.mul(projected_grad / zo_eps)

                g32 = g.to(dtype=torch.float32)
                # Track raw SPSA estimate norm before moment updates.
                grad_sum_sq += g32.norm().item() ** 2

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=torch.device("cpu"))
                    state["exp_avg"] = torch.zeros_like(g32)
                    state["exp_avg_sq"] = torch.zeros_like(g32)

                state["step"] += 1
                step = int(state["step"].item())
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)

                bias_c1 = 1.0 - beta1**step
                bias_c2 = 1.0 - beta2**step
                step_size_neg = -lr / bias_c1
                denom = exp_avg_sq.sqrt().div(math.sqrt(bias_c2)).add(adam_eps)
                update = (exp_avg * step_size_neg / denom).to(dtype=p.dtype)
                update_sum_sq += update.to(torch.float32).norm().item() ** 2
                p.data.add_(update)

        ge = math.sqrt(grad_sum_sq)
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["grad_est_norm"] = ge
        self._last_metrics["grad_est_norm_per_z_rms"] = (ge / z_rms) if z_rms > 0.0 else float("nan")
        self._last_metrics["update_norm"] = math.sqrt(update_sum_sq)


class LOZO(ZeroOrderOptimizer):
    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        rank: int = 4,
        step_interval: int = 1,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            step_interval=step_interval,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._last_metrics: dict[str, float] = {}

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics collected during the last step.

        Metrics
        -------
        projected_grad_abs
            Absolute value of the SPSA scalar ``(f⁺ − f⁻) / (2ε)``.
        grad_est_norm
            Global L2 norm of the low-rank gradient estimate ``projected_grad * Z``
            across all parameters (before lr scaling).  For 2D params the
            Frobenius norm of ``U @ V.T`` is computed efficiently via the
            ``rank × rank`` Gram matrices; for 1D params the standard L2 norm.
        grad_est_norm_per_z_rms
            ``grad_est_norm`` divided by the RMS norm of the probe directions
            (``U @ V.T`` Frobenius contribution per 2-D param, ``z`` L2 per 1-D
            param).  For this optimizer equals ``|projected_grad|`` from
            ``step()`` (the scalar already includes the ``1/ε`` factor).
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))
        refresh_v = self._step_count % self.defaults["step_interval"] == 0

        self._perturb(z_seed, scaling_factor=+1.0, refresh_v=refresh_v)
        loss_plus = closure()

        if self.defaults["perturbation_mode"] == "two_side":
            self._perturb(z_seed, scaling_factor=-2.0, refresh_v=False)
            loss_minus = closure()
            projected_grad = (loss_plus - loss_minus).item() / (2.0 * self.defaults["zo_eps"])
            self._perturb(z_seed, scaling_factor=+1.0, refresh_v=False)
        else:
            self._perturb(z_seed, scaling_factor=-1.0, refresh_v=False)
            loss_minus = closure()
            projected_grad = (loss_plus - loss_minus).item() / self.defaults["zo_eps"]

        self._apply_update(z_seed, projected_grad)
        self._last_metrics["projected_grad_abs"] = abs(projected_grad)
        self._step_count += 1
        return loss_plus

    def _flat_params(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p, idx
                idx += 1

    def _get_u(self, p: torch.Tensor, z_seed: int, param_idx: int) -> torch.Tensor:
        rank = self.defaults["rank"]
        gen = torch.Generator(device=p.device)
        gen.manual_seed((z_seed + param_idx * 1_000_003) % (2**31))
        return torch.randn(p.shape[0], rank, device=p.device, dtype=p.dtype, generator=gen)

    def _get_z1d(self, p: torch.Tensor, z_seed: int, param_idx: int) -> torch.Tensor:
        gen = torch.Generator(device=p.device)
        gen.manual_seed((z_seed + param_idx * 999_983) % (2**31))
        return torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen)

    def _perturb(self, z_seed: int, scaling_factor: float, refresh_v: bool) -> None:
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            eps = group["zo_eps"]
            if p.dim() == 2:
                rank = group["rank"]
                state = self.state[p]
                if refresh_v or "v" not in state:
                    gen_v = torch.Generator(device=p.device)
                    gen_v.manual_seed((z_seed * 1_000_003 + idx * 999_983) % (2**31))
                    v = torch.randn(p.shape[1], rank, device=p.device, dtype=p.dtype, generator=gen_v)
                    state["v"] = v
                v = state["v"]
                u = self._get_u(p, z_seed, idx)
                p.data.addmm_(u, v.t(), alpha=scaling_factor * eps)
            else:
                z = self._get_z1d(p, z_seed, idx)
                p.data.add_(z, alpha=scaling_factor * eps)

    def _apply_update(self, z_seed: int, projected_grad: float) -> None:
        grad_sum_sq = 0.0
        z_sum_sq = 0.0
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            if p.dim() == 2:
                v = self.state[p]["v"]
                u = self._get_u(p, z_seed, idx)
                # ||projected_grad * U @ V.T||_F^2 = projected_grad^2 * tr(U.T@U @ V.T@V)
                # Computed via two (rank x rank) Gram matrices — no full m×n materialisation.
                ugu = u.to(torch.float32).T @ u.to(torch.float32)  # (rank, rank)
                vgv = v.to(torch.float32).T @ v.to(torch.float32)  # (rank, rank)
                z_sum_sq += (ugu @ vgv).trace().item()
                grad_sum_sq += projected_grad ** 2 * (ugu @ vgv).trace().item()
                p.data.addmm_(u, v.t(), alpha=-lr * projected_grad)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
            else:
                z = self._get_z1d(p, z_seed, idx)
                z_sum_sq += z.to(torch.float32).norm().item() ** 2
                grad_sum_sq += (projected_grad * z.to(torch.float32)).norm().item() ** 2
                p.data.add_(z, alpha=-lr * projected_grad)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
        ge = math.sqrt(max(grad_sum_sq, 0.0))
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["grad_est_norm"] = ge
        self._last_metrics["grad_est_norm_per_z_rms"] = (ge / z_rms) if z_rms > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization  (Muon / ZO-Muon building block)
# ---------------------------------------------------------------------------

def _newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate matrix sign via Newton-Schulz iteration.

    Maps G → U Vᵀ where G = U Σ Vᵀ, i.e. normalizes all singular values to ~1.
    This is the "Muon" update rule from Kosson et al. and Vyas et al.
    Always operates in float32 and returns float32.

    Convergence requires spectral norm < sqrt(3); the initial normalisation
    by ``X.norm()`` ensures this.
    """
    assert G.ndim == 2, "G must be a 2-D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


# ---------------------------------------------------------------------------
# ZOMuon
# ---------------------------------------------------------------------------

class ZOMuon(ZeroOrderOptimizer):
    """Zeroth-Order optimizer with subspace gradient orthogonalization (ZO-Muon).

    For 2-D weight matrices the algorithm:
      1. Samples a Haar-distributed orthogonal projection ``P`` of shape
         ``(m, rank)`` via QR; cached and refreshed every ``step_interval`` steps.
      2. Accumulates ``num_samples`` gradient scalars in the low-rank subspace:
           ``lowdim_rge = (1/N) Σᵢ scalar_i · uᵢ``   shape ``(rank, n)``
         where each ``uᵢ ~ N(0,I)`` is the per-sample in-subspace direction.
         With ``num_samples=1`` the antithetic two-sided estimator is used
         (2 forward passes); with ``num_samples>1`` a one-sided centre-based
         estimator is used (1+N forward passes, shared ``P``).
      3. Applies Newton-Schulz orthogonalization:
           ``M_sign = NS(lowdim_rge)``   (≈ U Vᵀ, singular values → 1)
      4. Updates: ``W -= lr · P @ M_sign``

    For 1-D parameters (biases, norms) a standard averaged MeZO-style update
    is used without Newton-Schulz.

    DDP: ``P`` is seeded from ``_step_count`` (globally synchronized across
    ranks); ``uᵢ`` from the externally broadcast ``z_seed``.  The loss is
    all-reduced inside the closure so all ranks compute identical scalars and
    apply the same update without gradient communication.

    FSDP is not supported (blocked in ``build_optimizer``).

    Reference: "Powering Up Zeroth-Order Training via Subspace Gradient
    Orthogonalization", 2024.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        rank: int = 4,
        step_interval: int = 1,
        num_samples: int = 1,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if num_samples < 1:
            raise ValueError(f"Invalid num_samples: {num_samples}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            step_interval=step_interval,
            num_samples=num_samples,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._last_metrics: dict[str, float] = {}

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZOMuon diagnostics from the last step.

        Metrics
        -------
        projected_grad_abs_mean
            Mean absolute SPSA scalar across all ``num_samples`` samples.
            Measures the average signal strength in the probed subspace.
        grad_est_norm
            Global L2 / Frobenius norm of the update direction
            ``P @ M_sign`` for 2-D params (plus standard norm for 1-D),
            before ``lr`` scaling.  Because Newton-Schulz normalizes singular
            values to ~1, this is roughly ``sqrt(rank · n_2d_params)``.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat_params(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p, idx
                idx += 1

    def _get_or_refresh_p(self, p: torch.Tensor, param_idx: int) -> torch.Tensor:
        """Return the cached Haar-orthogonal P, refreshing if the interval has elapsed."""
        effective_rank = min(self.defaults["rank"], p.shape[0])
        state = self.state[p]
        last_refresh = state.get("p_refresh_step", -1)
        should_refresh = (last_refresh < 0) or (
            self._step_count - last_refresh >= self.defaults["step_interval"]
        )
        if should_refresh:
            gen = torch.Generator(device=p.device)
            gen.manual_seed(
                (self._step_count * 1_000_003 + param_idx * 999_983) % (2**31)
            )
            Q = torch.randn(
                p.shape[0], effective_rank, device=p.device, dtype=torch.float32, generator=gen
            )
            P, R = torch.linalg.qr(Q, mode="reduced")
            # Haar sign correction: ensure uniform distribution over O(m, rank).
            signs = R.diagonal().sign()
            signs[signs == 0] = 1.0
            P = P * signs  # float32, shape (m, effective_rank)
            state["p_mat"] = P
            state["p_refresh_step"] = self._step_count
        return state["p_mat"]

    def _get_u(
        self, p: torch.Tensor, z_seed: int, param_idx: int, sample_idx: int
    ) -> torch.Tensor:
        """Sample uᵢ ~ N(0, I) of shape (rank, n_cols) for sample i. Always float32."""
        effective_rank = min(self.defaults["rank"], p.shape[0])
        gen = torch.Generator(device=p.device)
        gen.manual_seed(
            (z_seed + param_idx * 1_000_003 + sample_idx * 999_983) % (2**31)
        )
        return torch.randn(
            effective_rank, p.shape[1], device=p.device, dtype=torch.float32, generator=gen
        )

    def _get_z1d(
        self, p: torch.Tensor, z_seed: int, param_idx: int, sample_idx: int
    ) -> torch.Tensor:
        """Sample z ~ N(0, I) for 1-D params. Always float32."""
        gen = torch.Generator(device=p.device)
        gen.manual_seed(
            (z_seed + param_idx * 999_983 + sample_idx * 111_317) % (2**31)
        )
        return torch.randn(p.shape, device=p.device, dtype=torch.float32, generator=gen)

    def _perturb_sample(self, z_seed: int, sample_idx: int, scaling_factor: float) -> None:
        """Perturb all parameters for one sample: θ += scaling * eps * P @ u (2D) or z (1D)."""
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            eps = group["zo_eps"]
            if p.dim() == 2:
                P = self._get_or_refresh_p(p, idx)
                u = self._get_u(p, z_seed, idx, sample_idx)
                p.data.addmm_(P.to(p.dtype), u.to(p.dtype), alpha=scaling_factor * eps)
            else:
                z = self._get_z1d(p, z_seed, idx, sample_idx)
                p.data.add_(z.to(p.dtype), alpha=scaling_factor * eps)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        zo_eps = self.defaults["zo_eps"]
        num_samples = self.defaults["num_samples"]
        scalars: dict[int, float] = {}

        if num_samples == 1:
            # Two-sided antithetic: 2 forward passes, lower-variance estimator.
            self._perturb_sample(z_seed, 0, +1.0)
            loss_plus = closure()
            self._perturb_sample(z_seed, 0, -2.0)
            loss_minus = closure()
            scalars[0] = (loss_plus - loss_minus).item() / (2.0 * zo_eps)
            self._perturb_sample(z_seed, 0, +1.0)  # restore θ
        else:
            # One-sided centre-based: 1 + num_samples forward passes, shared P.
            loss_center = closure()
            loss_plus = loss_center  # fallback; overwritten below
            for i in range(num_samples):
                self._perturb_sample(z_seed, i, +1.0)
                loss_plus_i = closure()
                if i == 0:
                    loss_plus = loss_plus_i  # return first sample's loss for logging
                self._perturb_sample(z_seed, i, -1.0)  # restore θ
                scalars[i] = (loss_plus_i - loss_center).item() / zo_eps

        self._apply_update(scalars, z_seed)
        self._last_metrics["projected_grad_abs_mean"] = (
            sum(abs(s) for s in scalars.values()) / len(scalars)
        )
        self._step_count += 1
        return loss_plus

    def _apply_update(self, scalars: dict[int, float], z_seed: int) -> None:
        ns_steps = self.defaults["ns_steps"]
        n_samples = len(scalars)
        grad_sum_sq = 0.0

        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if p.dim() == 2:
                P = self.state[p]["p_mat"]  # (m, effective_rank), float32
                effective_rank = P.shape[1]

                # Accumulate low-rank gradient estimate: (1/N) Σᵢ scalarᵢ · uᵢ
                lowdim_rge = torch.zeros(
                    effective_rank, p.shape[1], device=p.device, dtype=torch.float32
                )
                for sample_idx, scalar in scalars.items():
                    u = self._get_u(p, z_seed, idx, sample_idx)
                    lowdim_rge.add_(u, alpha=scalar)
                lowdim_rge.div_(n_samples)

                # Newton-Schulz: lowdim_rge → M_sign ≈ U Vᵀ  (singular values → 1)
                M_sign = _newtonschulz5(lowdim_rge, steps=ns_steps)

                # Full update direction G = P @ M_sign, shape (m, n)
                G = P @ M_sign
                grad_sum_sq += G.norm().item() ** 2

                p.data.add_(G.to(p.dtype), alpha=-lr)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

            else:
                # 1-D: average the scalar gradient estimates, no orthogonalization.
                grad_1d = torch.zeros(p.shape, device=p.device, dtype=torch.float32)
                for sample_idx, scalar in scalars.items():
                    z = self._get_z1d(p, z_seed, idx, sample_idx)
                    grad_1d.add_(z, alpha=scalar)
                grad_1d.div_(n_samples)

                grad_sum_sq += grad_1d.norm().item() ** 2
                p.data.add_(grad_1d.to(p.dtype), alpha=-lr)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

        self._last_metrics["grad_est_norm"] = math.sqrt(grad_sum_sq)
