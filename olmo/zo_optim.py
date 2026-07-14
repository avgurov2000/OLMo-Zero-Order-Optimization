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

from .zo_fo_direction_strategy import normalize_fo_direction


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
        num_pert_samples: int = 1,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        if num_pert_samples < 1:
            raise ValueError(f"Invalid num_pert_samples: {num_pert_samples}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            momentum=momentum,
            num_pert_samples=num_pert_samples,
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

        num_pert = self.defaults["num_pert_samples"]
        samples: list[tuple[int, float]] = []
        first_loss: Optional[torch.Tensor] = None

        for i in range(num_pert):
            seed_i = (z_seed + i * 987_654_321) % (2**31)
            self._perturb(seed_i, scaling_factor=+1.0)
            loss_plus = closure()
            if first_loss is None:
                first_loss = loss_plus

            if self.defaults["perturbation_mode"] == "two_side":
                self._perturb(seed_i, scaling_factor=-2.0)
                loss_minus = closure()
                scalar = (loss_plus - loss_minus).item() / 2.0
                self._perturb(seed_i, scaling_factor=+1.0)
            else:
                self._perturb(seed_i, scaling_factor=-1.0)
                loss_minus = closure()
                scalar = (loss_plus - loss_minus).item()

            samples.append((seed_i, scalar))

        self._apply_update(samples)
        self._last_metrics["projected_grad_abs"] = sum(abs(s) for _, s in samples) / num_pert
        return first_loss  # type: ignore[return-value]

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

    def _apply_update(self, samples: list[tuple[int, float]]) -> None:
        n = len(samples)

        # Float32 accumulators: sum_i scalar_i * z_i / eps, averaged over n.
        accumulators: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    accumulators[id(p)] = torch.zeros_like(p, dtype=torch.float32)

        z_sum_sq = 0.0
        for seed_i, scalar_i in samples:
            self._reset_generators(seed_i)
            for group in self.param_groups:
                eps = group["zo_eps"]
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    gen = self._get_generator(p.device)
                    z = self.vector_sampler.sample(p.shape, p.device, gen)
                    z_sum_sq += z.float().norm().item() ** 2
                    accumulators[id(p)].add_(z.float(), alpha=scalar_i / eps)

        grad_sum_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                grad_est = accumulators[id(p)].div_(n)
                grad_sum_sq += grad_est.norm().item() ** 2
                if weight_decay != 0.0:
                    grad_est.add_(p.data.float(), alpha=weight_decay)
                if momentum != 0.0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = grad_est.clone()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad_est)
                        grad_est = buf
                p.data.add_(grad_est.to(p.dtype), alpha=-lr)
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
        num_pert_samples: int = 1,
        normalize_direction: bool = False,
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
        if num_pert_samples < 1:
            raise ValueError(f"Invalid num_pert_samples: {num_pert_samples}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            betas=betas,
            eps=eps,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            num_pert_samples=num_pert_samples,
        )
        super().__init__(params, defaults)
        self.vector_sampler = VectorSampler(vector_sampling_type)
        self.normalize_direction = normalize_direction
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}
        self._last_pert_samples: list[tuple[int, float]] = []

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

        num_pert = self.defaults["num_pert_samples"]
        samples: list[tuple[int, float]] = []
        first_loss: Optional[torch.Tensor] = None

        for i in range(num_pert):
            seed_i = (z_seed + i * 987_654_321) % (2**31)
            self._perturb(seed_i, scaling_factor=+1.0)
            loss_plus = closure()
            if first_loss is None:
                first_loss = loss_plus

            if self.defaults["perturbation_mode"] == "two_side":
                self._perturb(seed_i, scaling_factor=-2.0)
                loss_minus = closure()
                scalar = (loss_plus - loss_minus).item() / 2.0
                self._perturb(seed_i, scaling_factor=+1.0)
            else:
                self._perturb(seed_i, scaling_factor=-1.0)
                loss_minus = closure()
                scalar = (loss_plus - loss_minus).item()

            samples.append((seed_i, scalar))

        self._last_pert_samples = list(samples)
        self._apply_update(samples)
        self._last_metrics["projected_grad_abs"] = sum(abs(s) for _, s in samples) / num_pert
        return first_loss  # type: ignore[return-value]

    def _sample_probe_direction(self, z_seed: int) -> dict[int, torch.Tensor]:
        """Sample per-parameter ``z`` for ``z_seed``; optionally normalize globally."""
        self._reset_generators(z_seed)
        direction: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                direction[id(p)] = self.vector_sampler.sample(p.shape, p.device, gen)
        if self.normalize_direction:
            direction, _ = normalize_fo_direction(direction)
        return direction

    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        direction = self._sample_probe_direction(z_seed)
        for group in self.param_groups:
            zo_eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                p.data.add_(direction[id(p)], alpha=scaling_factor * zo_eps)

    def _apply_update(self, samples: list[tuple[int, float]]) -> None:
        n = len(samples)

        # Float32 accumulators: (1/n) * sum_i scalar_i * z_i / eps
        accumulators: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    accumulators[id(p)] = torch.zeros_like(p, dtype=torch.float32)

        z_sum_sq = 0.0
        for seed_i, scalar_i in samples:
            direction = self._sample_probe_direction(seed_i)
            for group in self.param_groups:
                zo_eps = group["zo_eps"]
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    z = direction[id(p)].float()
                    z_sum_sq += z.norm().item() ** 2
                    accumulators[id(p)].add_(z, alpha=scalar_i / zo_eps)

        grad_sum_sq = 0.0
        update_sum_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            adam_eps = group["eps"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                g32 = accumulators[id(p)].div_(n)
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

    def _direction_for(self, p: torch.Tensor, direction: dict[int, torch.Tensor]) -> torch.Tensor:
        z = direction.get(id(p))
        if z is None:
            raise KeyError(f"Missing FO direction for parameter id={id(p)}")
        return z

    @torch.no_grad()
    def _perturb_direction(self, direction: dict[int, torch.Tensor], scaling_factor: float) -> None:
        for group in self.param_groups:
            zo_eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                z = self._direction_for(p, direction)
                p.data.add_(z.to(dtype=p.dtype), alpha=scaling_factor * zo_eps)

    @torch.no_grad()
    def estimate_with_direction(
        self,
        closure: Callable[[], torch.Tensor],
        direction: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, float]:
        """Two-sided (or one-sided) SPSA scalar along fixed ``direction``; restores θ."""
        self._perturb_direction(direction, scaling_factor=+1.0)
        loss_plus = closure()

        if self.defaults["perturbation_mode"] == "two_side":
            self._perturb_direction(direction, scaling_factor=-2.0)
            loss_minus = closure()
            scalar = (loss_plus - loss_minus).item() / 2.0
            self._perturb_direction(direction, scaling_factor=+1.0)
        else:
            self._perturb_direction(direction, scaling_factor=-1.0)
            loss_minus = closure()
            scalar = (loss_plus - loss_minus).item()
            self._perturb_direction(direction, scaling_factor=+1.0)

        return loss_plus, scalar

    @torch.no_grad()
    def apply_direction_update(self, scalar: float, direction: dict[int, torch.Tensor]) -> None:
        """Apply ZoAdam update using ``g_est = (scalar/ε) · z`` from ``direction``."""
        zo_eps = self.defaults["zo_eps"]
        self._last_pert_samples = [(0, scalar)]

        accumulators: dict[int, torch.Tensor] = {}
        z_sum_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                z = self._direction_for(p, direction).float()
                z_sum_sq += z.norm().item() ** 2
                accumulators[id(p)] = z.mul(scalar / zo_eps)

        grad_sum_sq = 0.0
        update_sum_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            adam_eps = group["eps"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                g32 = accumulators[id(p)]
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

        scalar_s_abs = abs(scalar / zo_eps)
        ge = math.sqrt(grad_sum_sq)
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["projected_grad_abs"] = abs(scalar)
        self._last_metrics["scalar_S_abs"] = scalar_s_abs
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
        num_pert_samples: int = 1,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        if num_pert_samples < 1:
            raise ValueError(f"Invalid num_pert_samples: {num_pert_samples}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            step_interval=step_interval,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            num_pert_samples=num_pert_samples,
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

        num_pert = self.defaults["num_pert_samples"]
        samples: list[tuple[int, float]] = []
        first_loss: Optional[torch.Tensor] = None

        for i in range(num_pert):
            seed_i = (z_seed + i * 987_654_321) % (2**31)
            # V is refreshed only on the first sample of a refresh step.
            refresh_v = (i == 0) and (self._step_count % self.defaults["step_interval"] == 0)

            self._perturb(seed_i, scaling_factor=+1.0, refresh_v=refresh_v)
            loss_plus = closure()
            if first_loss is None:
                first_loss = loss_plus

            if self.defaults["perturbation_mode"] == "two_side":
                self._perturb(seed_i, scaling_factor=-2.0, refresh_v=False)
                loss_minus = closure()
                projected_grad = (loss_plus - loss_minus).item() / (2.0 * self.defaults["zo_eps"])
                self._perturb(seed_i, scaling_factor=+1.0, refresh_v=False)
            else:
                self._perturb(seed_i, scaling_factor=-1.0, refresh_v=False)
                loss_minus = closure()
                projected_grad = (loss_plus - loss_minus).item() / self.defaults["zo_eps"]

            samples.append((seed_i, projected_grad))

        self._apply_update(samples)
        self._last_metrics["projected_grad_abs"] = sum(abs(s) for _, s in samples) / num_pert
        self._step_count += 1
        return first_loss  # type: ignore[return-value]

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

    def _apply_update(self, samples: list[tuple[int, float]]) -> None:
        n = len(samples)
        grad_sum_sq = 0.0
        z_sum_sq = 0.0
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if p.dim() == 2:
                v = self.state[p]["v"]  # (n_cols, rank), p.dtype
                effective_rank = v.shape[1]
                vgv = v.float().T @ v.float()

                u_acc = torch.zeros(p.shape[0], effective_rank, device=p.device, dtype=torch.float32)
                for seed_i, pg in samples:
                    u_i = self._get_u(p, seed_i, idx).float()
                    u_acc.add_(u_i, alpha=pg)
                    ugu = u_i.T @ u_i
                    z_sum_sq += (ugu @ vgv).trace().item()
                u_acc.div_(n)

                ugu = u_acc.T @ u_acc
                grad_sum_sq += (ugu @ vgv).trace().item()

                p.data.addmm_(u_acc.to(p.dtype), v.t(), alpha=-lr)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
            else:
                z_acc = torch.zeros(p.shape, device=p.device, dtype=torch.float32)
                for seed_i, pg in samples:
                    z_i = self._get_z1d(p, seed_i, idx).float()
                    z_acc.add_(z_i, alpha=pg)
                    z_sum_sq += z_i.norm().item() ** 2
                z_acc.div_(n)

                grad_sum_sq += z_acc.norm().item() ** 2
                p.data.add_(z_acc.to(p.dtype), alpha=-lr)
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


# ---------------------------------------------------------------------------
# KronZO
# ---------------------------------------------------------------------------

def _best_pair(n: int) -> tuple[int, int]:
    """Two divisors of ``n`` closest to ``sqrt(n)`` (Algorithm 1 ``BestPair`` in KronZO).

    Returns ``(1, n)`` for primes (the paper notes this is rare for LM layer widths).
    """
    if n <= 1:
        return (1, n)
    root = int(math.isqrt(n))
    for offset in range(root + 1):
        for cand in (root - offset, root + offset):
            if cand >= 1 and n % cand == 0:
                return (cand, n // cand)
    return (1, n)  # fallback for primes


class KronZO(ZeroOrderOptimizer):
    """Zeroth-order Kronecker optimization for LM pretraining (KronZO).

    Implements "Zeroth-order Kronecker optimization for pretraining language models"
    (Allaire, Le Digabel, Orban, Partovi Nia, GERAD G-2025-44).

    Two ideas vs. MeZO / LOZO:

    1. **Kronecker-factored perturbations** (Algorithm 1).  For a 2-D weight
       ``W ∈ R^{m×n}`` the probe direction is ``Z = A ⊗ B`` where
       ``A ∈ R^{m1×n1}``, ``B ∈ R^{m2×n2}`` with ``m1·m2 = m`` and ``n1·n2 = n``,
       each factor chosen as close to square as possible.  Because
       ``rank(A⊗B) = rank(A)·rank(B)``, ``Z`` is almost surely **full rank** (so it
       explores the whole subspace, unlike low-rank LOZO) while only the tiny
       factors are stored.  ``B`` is frozen for ``step_interval`` steps (like ``V``
       in LOZO); ``A`` is resampled for every probe.  1-D params use a plain
       Gaussian probe.

    2. **Criterion-driven directional update** (Algorithm 2, "lucky RGE").  Each
       step samples ``query_budget`` (``q``) probes; for each it forms the SPSA
       scalar ``c_i = (L(θ+εZ_i) − L(θ−εZ_i)) / 2ε`` and *evaluates the candidate
       step* ``L(θ − α c_i Z_i)``.  Only the single best-decreasing direction is
       kept (not the average).  The step is then accepted only if that best loss
       beats the worst of the last ``history_length`` (``h``) accepted losses;
       otherwise the step is skipped.  Cost per step: ``1 + 3q`` forward passes
       (two-sided) or ``1 + 2q`` (one-sided).

    Probes are regenerated from seeds rather than stored, keeping the live memory
    footprint at one parameter-sized temporary.  DDP: ``A`` is seeded from the
    externally broadcast ``z_seed`` and ``B`` from the globally-synchronized
    ``_step_count``, while the closure all-reduces the loss, so every rank derives
    identical scalars, accept/reject decisions and updates without gradient
    communication.  FSDP is not supported (blocked in ``build_optimizer``).
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        step_interval: int = 50,
        query_budget: int = 10,
        history_length: int = 10,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        orthogonal_probes: bool = False,
        momentum: float = 0.0,
        line_search_steps: int = 0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if step_interval < 1:
            raise ValueError(f"Invalid step_interval: {step_interval}")
        if query_budget < 1:
            raise ValueError(f"Invalid query_budget: {query_budget}")
        if history_length < 0:
            raise ValueError(f"Invalid history_length: {history_length}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1): {momentum}")
        if line_search_steps < 0:
            raise ValueError(f"line_search_steps must be >= 0: {line_search_steps}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            step_interval=step_interval,
            query_budget=query_budget,
            history_length=history_length,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            orthogonal_probes=orthogonal_probes,
            momentum=momentum,
            line_search_steps=line_search_steps,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        # Sliding window of the last `history_length` accepted losses; prefilled with
        # +inf so the first `h` steps always pass the acceptance test.
        self._window: list[float] = [float("inf")] * history_length
        self._n_steps = 0
        self._n_accepted = 0
        self._last_metrics: dict[str, float] = {}
        # Per-step probe bookkeeping for the optional ZO-vs-FO cosine diagnostic:
        # the (seed, scalar) of every probe and the index of the lowest-loss one.
        self._probe_samples: list[tuple[int, float]] = []
        self._best_probe_idx: int = -1
        # Orthogonal-probe support: seeds of the current step's q probes and a per-step,
        # per-parameter cache of the QR-orthogonalized A factors (rebuilt each step).
        self._probe_seeds: list[int] = []
        self._seed_pos: dict[int, int] = {}
        self._ortho_cache: dict[int, torch.Tensor] = {}

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return KronZO diagnostics from the last step.

        Metrics
        -------
        projected_grad_abs
            Mean ``|c_i|`` over the ``q`` probes (SPSA signal strength).
        grad_est_norm
            Frobenius norm of the applied update direction ``c_best · Z_best``
            (0.0 when the step was skipped).  Computed cheaply via the Kronecker
            identity ``||A⊗B||_F = ||A||_F · ||B||_F`` without materializing ``Z``.
        update_accepted
            1.0 if the directional update was applied this step, else 0.0
            (a step is skipped when no probe improved on the center loss, or the
            best loss did not beat the acceptance window).
        accept_rate
            Running fraction of accepted steps (the paper observes ~90%).
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def fo_cosine_metrics(self) -> dict[str, float]:
        """Cosine similarity between KronZO's ZO gradient estimate and the true FO gradient.

        Must be called right after ``step()`` and after a real backward has populated
        ``p.grad`` for these parameters at the *same* weights θ_k (the trainer arranges
        this on its probe interval).  Reconstructs the probe directions from the seeds
        stored during the last step and reports two alignments:

        ``cosine_selected``
            cos(∠) between the *lucky* probe KronZO actually uses (lowest candidate
            loss, ``c_best·Z_best``) and ∇L.  How good the chosen step direction is.
        ``cosine_qrge``
            cos(∠) between the unbiased q-RGE average ``(1/q)·Σ c_i Z_i`` and ∇L.
            How good the raw averaged estimator is (improves with q, degrades with
            dimension — the "dilution" the directional update is designed to beat).

        Returns ``{}`` if no probe data or no gradients are available.
        """
        samples = self._probe_samples
        if not samples or self._best_probe_idx < 0:
            return {}
        q = len(samples)
        best_seed, best_c = samples[self._best_probe_idx]

        dot_sel = norm_sel = 0.0
        dot_avg = norm_avg = 0.0
        norm_g = 0.0
        for _group, p, idx in self._flat_params():
            if not p.requires_grad or p.grad is None:
                continue
            g = p.grad.float()
            norm_g += g.pow(2).sum().item()

            # Selected ("lucky") direction c_best · Z_best.
            sel = self._build_z(p, idx, best_seed).float().mul_(best_c)
            dot_sel += (g * sel).sum().item()
            norm_sel += sel.pow(2).sum().item()

            # Unbiased q-RGE average (1/q) Σ c_i Z_i.
            avg = torch.zeros_like(g)
            for seed_i, c_i in samples:
                avg.add_(self._build_z(p, idx, seed_i).float(), alpha=c_i)
            avg.div_(q)
            dot_avg += (g * avg).sum().item()
            norm_avg += avg.pow(2).sum().item()

        if norm_g == 0.0:
            return {}
        eps = 1e-12
        return {
            "cosine_selected": dot_sel / (math.sqrt(norm_g * norm_sel) + eps),
            "cosine_qrge": dot_avg / (math.sqrt(norm_g * norm_avg) + eps),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat_params(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p, idx
                idx += 1

    def _get_a(self, p: torch.Tensor, dims: tuple, z_seed: int, idx: int) -> torch.Tensor:
        m1, n1, _m2, _n2 = dims
        gen = torch.Generator(device=p.device)
        gen.manual_seed((z_seed + idx * 1_000_003) % (2**31))
        return torch.randn(m1, n1, device=p.device, dtype=p.dtype, generator=gen)

    def _a_for(self, p: torch.Tensor, dims: tuple, seed: int, idx: int) -> torch.Tensor:
        """Return the A factor for one probe — raw Gaussian, or (if enabled) the
        QR-orthogonalized variant so the step's q probes are mutually orthogonal.

        Since ``<A_i⊗B, A_j⊗B> = <A_i,A_j>·||B||²`` (B is frozen within the step),
        orthogonalizing the small A factors makes the full perturbations Z_i orthogonal,
        maximizing coverage of the B-induced subspace per step. The orthonormal columns
        are rescaled to the Gaussian norm √(m1·n1) so ε / lr keep their scale. The whole
        set is reproducible from the step's seeds, so every regeneration point agrees.
        """
        if not self.defaults.get("orthogonal_probes", False):
            return self._get_a(p, dims, seed, idx)
        m1, n1, _m2, _n2 = dims
        q = len(self._probe_seeds)
        # Can't orthogonalize more vectors than the subspace dimension — fall back.
        if q < 2 or m1 * n1 < q:
            return self._get_a(p, dims, seed, idx)
        cache = self._ortho_cache.get(idx)
        if cache is None:
            raws = torch.stack(
                [self._get_a(p, dims, s, idx).float().reshape(-1) for s in self._probe_seeds],
                dim=1,
            )  # (m1*n1, q)
            qmat, _ = torch.linalg.qr(raws, mode="reduced")  # orthonormal columns (m1*n1, q)
            cache = (qmat * math.sqrt(m1 * n1)).t().reshape(q, m1, n1).to(p.dtype)  # (q, m1, n1)
            self._ortho_cache[idx] = cache
        return cache[self._seed_pos[seed]]

    def _refresh_b(self, p: torch.Tensor, dims: tuple, idx: int) -> torch.Tensor:
        _m1, _n1, m2, n2 = dims
        gen = torch.Generator(device=p.device)
        gen.manual_seed((self._step_count * 1_000_003 + idx * 999_983) % (2**31))
        return torch.randn(m2, n2, device=p.device, dtype=p.dtype, generator=gen)

    def _get_z1d(self, p: torch.Tensor, z_seed: int, idx: int) -> torch.Tensor:
        gen = torch.Generator(device=p.device)
        gen.manual_seed((z_seed + idx * 999_983) % (2**31))
        return torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen)

    def _ensure_state(self) -> None:
        """Cache per-param Kronecker dims and (re)sample the frozen factor ``B``."""
        refresh = self._step_count % self.defaults["step_interval"] == 0
        for _group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            if p.dim() != 2:
                continue
            state = self.state[p]
            if "kron_dims" not in state:
                m, n = p.shape
                m1, m2 = _best_pair(int(m))
                n1, n2 = _best_pair(int(n))
                state["kron_dims"] = (m1, n1, m2, n2)
            if refresh or "B" not in state:
                state["B"] = self._refresh_b(p, state["kron_dims"], idx)

    def _build_z(self, p: torch.Tensor, idx: int, seed: int) -> torch.Tensor:
        """Regenerate the full perturbation ``Z`` for one parameter from ``seed``."""
        if p.dim() == 2:
            dims = self.state[p]["kron_dims"]
            a = self._a_for(p, dims, seed, idx)
            b = self.state[p]["B"]
            return torch.kron(a, b)
        return self._get_z1d(p, seed, idx)

    def _perturb_eps(self, seed: int, sign: float) -> None:
        """θ += sign · ε · Z   (finite-difference probe, single global ε)."""
        eps = self.defaults["zo_eps"]
        for _group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            p.data.add_(self._build_z(p, idx, seed), alpha=sign * eps)

    def _perturb_step(self, seed: int, c: float, sign: float) -> None:
        """θ += sign · lr · c · Z   (candidate / trial step, per-group lr)."""
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            p.data.add_(self._build_z(p, idx, seed), alpha=sign * group["lr"] * c)

    def _apply_best(self, seed: int, c: float) -> float:
        """Apply the accepted update ``θ -= lr · c · Z`` (+ decoupled weight decay).

        Returns the Frobenius norm of ``c · Z`` summed over params (the update
        direction magnitude), computed via ``||A⊗B||_F = ||A||_F · ||B||_F``.
        """
        grad_sq = 0.0
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            if p.dim() == 2:
                dims = self.state[p]["kron_dims"]
                a = self._a_for(p, dims, seed, idx)
                b = self.state[p]["B"]
                grad_sq += (c * a.float().norm().item() * b.float().norm().item()) ** 2
                z = torch.kron(a, b)
            else:
                z = self._get_z1d(p, seed, idx)
                grad_sq += (c * z.float().norm().item()) ** 2
            p.data.add_(z, alpha=-lr * c)
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)
        return math.sqrt(max(grad_sq, 0.0))

    def _momentum_step(self, seed: Optional[int], c: Optional[float], accepted: bool) -> float:
        """Heavy-ball momentum update.

        Keeps a per-parameter buffer ``m`` and integrates the step's chosen ZO direction
        into it: ``m ← β·m + c_best·Z_best`` when the step is accepted (otherwise ``m`` just
        decays, ``m ← β·m``). The parameters always move along the accumulated direction,
        ``θ ← θ − lr·m`` (+ decoupled weight decay).

        Because the per-step estimates are (near-)unbiased, this averages many probes over
        time, so the effective alignment with the true gradient grows roughly like
        ``√(1/(1−β) / d)`` instead of ``1/√d`` for a single step — the cheapest way to
        "guess" the gradient better without any backward pass. Costs one extra full-size
        buffer per parameter (still far below first-order, which also stores activations).
        """
        beta = self.defaults["momentum"]
        upd_sq = 0.0
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            state = self.state[p]
            mom = state.get("mom")
            if mom is None:
                mom = torch.zeros_like(p, dtype=torch.float32)
                state["mom"] = mom
            if accepted:
                assert seed is not None and c is not None
                # EMA so ||m|| stays on the scale of a single estimate -> lr keeps its meaning
                # (heavy-ball would amplify the step by ~1/(1-β) and needs a smaller lr).
                mom.mul_(beta).add_(self._build_z(p, idx, seed).float(), alpha=c * (1.0 - beta))
            else:
                mom.mul_(beta)
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            p.data.add_(mom.to(p.dtype), alpha=-lr)
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)
            upd_sq += mom.norm().item() ** 2
        return math.sqrt(max(upd_sq, 0.0))

    def _step_along_momentum(self, sign: float) -> None:
        """θ += sign · lr · m   (move one more step along the momentum buffer)."""
        for group, p, _idx in self._flat_params():
            if not p.requires_grad:
                continue
            mom = self.state[p].get("mom")
            if mom is None:
                continue
            p.data.add_(mom.to(p.dtype), alpha=sign * group["lr"])

    def _line_search(self, extend: Callable[[], None], undo: Callable[[], None],
                     closure: Callable[[], torch.Tensor], k_max: int) -> int:
        """Greedily keep stepping along the just-applied direction while the loss drops.

        Once a direction has been accepted, this rides it: re-apply the same update and
        evaluate the (same-batch) loss; keep it while it strictly improves, otherwise undo
        the last extension and stop. Amortizes the expensive 1+3q probe sampling over
        several real moves and acts as an adaptive step length. ``extend``/``undo`` move
        one step forward/back along the direction. Returns how many extra steps were taken.
        """
        cur = closure().item()
        taken = 0
        for _ in range(k_max):
            extend()
            nxt = closure().item()
            if nxt < cur:
                cur = nxt
                taken += 1
            else:
                undo()  # revert the non-improving extension
                break
        return taken

    def _push_window(self, loss: float) -> None:
        """Replace the worst (max) loss in the acceptance window with ``loss``."""
        if not self._window:
            return
        j = max(range(len(self._window)), key=lambda k: self._window[k])
        self._window[j] = loss

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        self._ensure_state()
        eps = self.defaults["zo_eps"]
        q = self.defaults["query_budget"]
        two_side = self.defaults["perturbation_mode"] == "two_side"

        # Fix this step's q probe seeds up front so every regeneration point agrees,
        # and reset the per-step orthogonalization cache (depends on these seeds).
        self._probe_seeds = [(z_seed + i * 987_654_321) % (2**31) for i in range(q)]
        self._seed_pos = {s: i for i, s in enumerate(self._probe_seeds)}
        self._ortho_cache = {}

        # Line 5: center loss at θ_k. Also the value reported to the trainer.
        center_loss = closure()
        l_best = center_loss.item()
        best_seed: Optional[int] = None
        best_c: Optional[float] = None
        abs_c_sum = 0.0
        samples: list[tuple[int, float]] = []
        best_probe_idx = -1
        best_probe_loss = float("inf")

        for i in range(q):
            seed_i = self._probe_seeds[i]

            # SPSA scalar c_i along Z_i.
            self._perturb_eps(seed_i, +1.0)
            loss_plus = closure().item()
            if two_side:
                self._perturb_eps(seed_i, -2.0)
                loss_minus = closure().item()
                self._perturb_eps(seed_i, +1.0)  # restore θ
                c_i = (loss_plus - loss_minus) / (2.0 * eps)
            else:
                self._perturb_eps(seed_i, -1.0)  # restore θ
                c_i = (loss_plus - l_best) / eps
            abs_c_sum += abs(c_i)
            samples.append((seed_i, c_i))

            # Evaluate the candidate step L(θ − lr·c_i·Z_i), then restore.
            self._perturb_step(seed_i, c_i, -1.0)
            loss_cand = closure().item()
            self._perturb_step(seed_i, c_i, +1.0)

            if loss_cand < best_probe_loss:
                best_probe_loss = loss_cand
                best_probe_idx = i
            if loss_cand < l_best:
                l_best = loss_cand
                best_seed = seed_i
                best_c = c_i

        self._probe_samples = samples
        self._best_probe_idx = best_probe_idx

        # Directional update with sliding-window acceptance (Lines 16-19).
        window_max = max(self._window) if self._window else float("inf")
        accepted = (best_c is not None) and (l_best <= window_max)
        grad_norm = 0.0
        use_momentum = self.defaults["momentum"] > 0.0
        if use_momentum:
            # Momentum mode: the gate decides what enters the buffer; we always step along it.
            grad_norm = self._momentum_step(best_seed, best_c, accepted)
        elif accepted:
            assert best_seed is not None and best_c is not None
            grad_norm = self._apply_best(best_seed, best_c)

        # Line search: once a step is applied, keep riding the same direction while the
        # (same-batch) loss keeps dropping. Extends along the momentum buffer in momentum
        # mode, otherwise along c_best·Z_best.
        ls_max = self.defaults["line_search_steps"]
        n_extends = 0
        if ls_max > 0 and accepted:
            if use_momentum:
                extend = lambda: self._step_along_momentum(-1.0)
                undo = lambda: self._step_along_momentum(+1.0)
            else:
                extend = lambda: self._perturb_step(best_seed, best_c, -1.0)
                undo = lambda: self._perturb_step(best_seed, best_c, +1.0)
            n_extends = self._line_search(extend, undo, closure, ls_max)

        self._push_window(l_best)

        self._step_count += 1
        self._n_steps += 1
        if accepted:
            self._n_accepted += 1

        self._last_metrics["projected_grad_abs"] = abs_c_sum / q
        self._last_metrics["grad_est_norm"] = grad_norm
        self._last_metrics["update_accepted"] = 1.0 if accepted else 0.0
        self._last_metrics["accept_rate"] = self._n_accepted / self._n_steps
        self._last_metrics["line_search_extends"] = float(n_extends)

        return center_loss


# ---------------------------------------------------------------------------
# HiZOO
# ---------------------------------------------------------------------------

class HiZOO(ZeroOrderOptimizer):
    """Hessian-Informed Zeroth-Order Optimizer (HiZOO / HiZOO).

    Implements "Second-Order Fine-Tuning without Pain for LLMs: A Hessian Informed
    Zeroth-Order Optimizer" (Zhao, Dang, Ye, Dai, Qian, Tsang, ICLR 2025),
    Algorithm 1 (see ``papers/HiZO.pdf``).

    Idea
    ----
    MeZO uses an isotropic Gaussian probe ``u`` and takes a step of fixed scale
    regardless of the local curvature.  Heterogeneous curvature across LLM
    parameters makes this converge slowly / towards saddle points.  HiZOO
    additionally estimates a **diagonal Hessian** with one extra forward pass and
    uses it as a pre-conditioner, so each coordinate is scaled by its curvature —
    a diagonal-Newton step obtained from forward passes only.

    Per parameter we store the diagonal ``Σ⁻¹`` (an estimate of the diagonal
    Hessian, initialised to ``1``), EMA-updated each step.  ``Σ^{1/2} = 1/√Σ⁻¹``
    pre-conditions both the perturbation and the descent direction.

    Each step (Algorithm 1) costs **3 forward passes** and, per parameter,
    ``u ~ N(0, I)`` reproduced from the shared seed:

      1. ``ℓ   = L(θ)``                                   (center)
      2. ``ℓ₊  = L(θ + μ Σ^{1/2} u)``
      3. ``ℓ₋  = L(θ − μ Σ^{1/2} u)``
      4. scalar curvature ``S = (ℓ₊ + ℓ₋ − 2ℓ) / (2μ²)`` and
         projected gradient ``g = (ℓ₊ − ℓ₋) / (2μ)``.
      5. diagonal Hessian estimate ``diag(Σ'ₜ) = S · Σ⁻¹ₜ₋₁ · u²`` and EMA update
         ``Σ⁻¹ₜ = (1−α) Σ⁻¹ₜ₋₁ + α |diag(Σ'ₜ)|`` (``α`` = smooth scale, Eq. 5).
      6. descent (Eq. 1/2), using the **updated** ``Σₜ``:
         ``θᵢ ← θᵢ − ηₜ · g · Σ^{1/2}ₜ,ᵢ · uᵢ``.

    Memory: one extra ``O(d)`` float32 buffer (the diagonal Hessian).  The probe
    ``u`` is regenerated from the seed rather than stored, matching MeZO's
    memory-frugal pattern.

    DDP: ``u`` is seeded from the externally broadcast ``z_seed`` and the loss is
    all-reduced inside the closure, so every rank derives identical scalars, an
    identical Hessian state and an identical update without gradient
    communication.  FSDP is not supported (blocked in ``build_optimizer``).

    Notes
    -----
    * The paper's Algorithm 1 is inherently three-pass (the Hessian needs the
      center loss ``ℓ``), so ``perturbation_mode`` is ignored here.
    * ``num_pert_samples = 1`` (``n = 1``) is the paper's default; larger ``n``
      averages several probes per step (HiZOO-multi) for a lower-variance Hessian
      estimate at the cost of ``1 + 2n`` forward passes.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        smooth_scale: float = 1e-6,
        hessian_min: float = 1e-8,
        weight_decay: float = 0.0,
        vector_sampling_type: str = "standard_normal",
        num_pert_samples: int = 1,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if not (0.0 <= smooth_scale <= 1.0):
            raise ValueError(f"smooth_scale (EMA α) must be in [0, 1], got {smooth_scale}")
        if hessian_min <= 0:
            raise ValueError(f"hessian_min must be > 0, got {hessian_min}")
        if num_pert_samples < 1:
            raise ValueError(f"Invalid num_pert_samples: {num_pert_samples}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            smooth_scale=smooth_scale,
            hessian_min=hessian_min,
            weight_decay=weight_decay,
            num_pert_samples=num_pert_samples,
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

        See MeZO._reset_generators for the rationale (seeding once lets the
        generator advance sequentially so each parameter gets an independent
        probe, while the same seed reproduces the same probes later).
        """
        seen: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.device not in seen:
                    self._get_generator(p.device).manual_seed(z_seed)
                    seen.add(p.device)

    def _hess(self, p: torch.Tensor) -> torch.Tensor:
        """Return the per-parameter diagonal Hessian estimate ``Σ⁻¹`` (float32).

        Lazily initialised to ones (``Σ₀ = I``) on first access.
        """
        state = self.state[p]
        if "hess" not in state:
            state["hess"] = torch.ones_like(p, dtype=torch.float32)
        return state["hess"]

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return HiZOO diagnostics collected during the last step.

        Metrics
        -------
        projected_grad_abs
            Absolute SPSA scalar ``|(ℓ₊ − ℓ₋) / (2μ)|`` (mean over ``n`` probes).
        curvature_abs
            Absolute finite-difference curvature ``|(ℓ₊ + ℓ₋ − 2ℓ) / (2μ²)|``
            (mean over ``n`` probes) — the raw signal driving the Hessian EMA.
        grad_est_norm
            Global L2 norm of the applied pre-conditioned update direction
            ``g · Σ^{1/2} · u`` across all parameters (before ``lr`` scaling).
        hessian_mean
            Mean of the diagonal Hessian estimate ``Σ⁻¹`` across all parameters
            (1.0 at init; drifts towards the local curvature scale).
        precond_sqrt_mean
            Mean pre-conditioner ``Σ^{1/2} = 1/√Σ⁻¹`` — the effective per-coordinate
            step-scale multiplier relative to MeZO.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        mu = self.defaults["zo_eps"]
        num_pert = self.defaults["num_pert_samples"]

        # Center loss ℓ = L(θ). One evaluation shared by all n probes.
        center_loss = closure()
        center = center_loss.item()

        pg_abs_sum = 0.0
        curv_abs_sum = 0.0
        for i in range(num_pert):
            seed_i = (z_seed + i * 987_654_321) % (2**31)

            self._perturb(seed_i, scaling_factor=+1.0)   # θ + μ Σ^{1/2} u
            loss_plus = closure().item()
            self._perturb(seed_i, scaling_factor=-2.0)   # θ − μ Σ^{1/2} u
            loss_minus = closure().item()
            self._perturb(seed_i, scaling_factor=+1.0)   # restore θ

            projected_grad = (loss_plus - loss_minus) / (2.0 * mu)
            curvature = (loss_plus + loss_minus - 2.0 * center) / (2.0 * mu * mu)
            pg_abs_sum += abs(projected_grad)
            curv_abs_sum += abs(curvature)

            # Update the diagonal Hessian (Σ⁻¹) and apply the pre-conditioned
            # descent step for this probe, both reproduced from ``seed_i``.
            self._hessian_and_descent(seed_i, projected_grad, curvature)

        self._last_metrics["projected_grad_abs"] = pg_abs_sum / num_pert
        self._last_metrics["curvature_abs"] = curv_abs_sum / num_pert
        return center_loss

    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        """θ ← θ + scaling_factor · μ · Σ^{1/2} · u, with ``Σ^{1/2} = 1/√Σ⁻¹``.

        Uses the *current* (pre-update) Hessian, so the ±μ / −2μ sequence cancels
        exactly and restores θ.
        """
        self._reset_generators(z_seed)
        mu = self.defaults["zo_eps"]
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                u = self.vector_sampler.sample(p.shape, p.device, gen).float()
                precond = torch.rsqrt(self._hess(p))  # Σ^{1/2} = 1/√Σ⁻¹
                delta = u.mul_(scaling_factor * mu).mul_(precond)
                p.data.add_(delta.to(p.dtype))

    def _hessian_and_descent(self, z_seed: int, projected_grad: float, curvature: float) -> None:
        """EMA-update Σ⁻¹ then take the pre-conditioned descent step for one probe.

        Σ'ₜ diagonal:  ``S · Σ⁻¹ₜ₋₁ · u²``  (S = curvature, Eq. 4).
        EMA (Eq. 5):   ``Σ⁻¹ₜ = (1−α) Σ⁻¹ₜ₋₁ + α |diag(Σ'ₜ)|``.
        Descent (Eq. 1/2), with the **updated** Σₜ:
                       ``θᵢ ← θᵢ − lr · g · Σ^{1/2}ₜ,ᵢ · uᵢ``.
        """
        self._reset_generators(z_seed)
        num_pert = self.defaults["num_pert_samples"]
        alpha = self.defaults["smooth_scale"]
        hess_min = self.defaults["hessian_min"]

        grad_sum_sq = 0.0
        hess_sum = 0.0
        precond_sum = 0.0
        n_elems = 0
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                u = self.vector_sampler.sample(p.shape, p.device, gen).float()
                hess = self._hess(p)  # Σ⁻¹ₜ₋₁ (float32, in-place buffer)

                # Diagonal Hessian estimate and EMA (use old Σ⁻¹, then update).
                sigma_prime = (curvature * hess).mul_(u.mul(u))  # S · Σ⁻¹ · u²
                hess.mul_(1.0 - alpha).add_(sigma_prime.abs_(), alpha=alpha)
                hess.clamp_(min=hess_min)

                # Pre-conditioned descent with the updated Σₜ, averaged over n probes.
                precond = torch.rsqrt(hess)  # Σ^{1/2}ₜ = 1/√Σ⁻¹ₜ
                grad_est = u.mul_(projected_grad / num_pert).mul_(precond)
                grad_sum_sq += grad_est.pow(2).sum().item()
                if weight_decay != 0.0:
                    grad_est.add_(p.data.float(), alpha=weight_decay)
                p.data.add_(grad_est.to(p.dtype), alpha=-lr)

                hess_sum += hess.sum().item()
                precond_sum += precond.sum().item()
                n_elems += hess.numel()

        self._last_metrics["grad_est_norm"] = math.sqrt(max(grad_sum_sq, 0.0))
        if n_elems > 0:
            self._last_metrics["hessian_mean"] = hess_sum / n_elems
            self._last_metrics["precond_sqrt_mean"] = precond_sum / n_elems
