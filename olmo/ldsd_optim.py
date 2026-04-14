"""
ZO-LDSD optimizers adapted for OLMo training pipeline.

Ports ZO_MUON and ZO_SignSGD from zo_ldsd/ to OLMo's ZeroOrderOptimizer interface.
Key adaptations vs the original zo_ldsd implementations:
  - step(closure, z_seed) accepts an external seed so distributed ranks stay in sync
  - get_post_step_metrics() exposes training diagnostics for W&B / console logging
  - per-device generators (consistent with existing MeZO / ZoAdam convention)
  - weight_decay applied as decoupled (multiplicative) decay on all parameters
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
