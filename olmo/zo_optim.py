"""
MeZO, LOZO, and ZoAdam zero-order optimizers for OLMo training.

Adapted from ``llm-base-zero-order`` (see also LOZO paper / optsuite).
ZoAdam combines the MeZO / SPSA gradient estimate with AdamW-style moment updates.
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

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

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
        return loss_plus

    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        for group in self.param_groups:
            eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                gen.manual_seed(z_seed)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                p.data.add_(z, alpha=scaling_factor * eps)

    def _apply_update(self, z_seed: int, projected_grad: float) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["zo_eps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                gen.manual_seed(z_seed)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                grad_est = z.mul_(projected_grad / eps)
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

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

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
        return loss_plus

    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        for group in self.param_groups:
            zo_eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                gen.manual_seed(z_seed)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                p.data.add_(z, alpha=scaling_factor * zo_eps)

    def _apply_update(self, z_seed: int, projected_grad: float) -> None:
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
                gen.manual_seed(z_seed)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                g = z.mul(projected_grad / zo_eps)

                g32 = g.to(dtype=torch.float32)
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
                p.data.add_(update)


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
            if p.dim() >= 2:
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
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            if p.dim() >= 2:
                v = self.state[p]["v"]
                u = self._get_u(p, z_seed, idx)
                p.data.addmm_(u, v.t(), alpha=-lr * projected_grad)
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
            else:
                z = self._get_z1d(p, z_seed, idx)
                p.data.add_(z, alpha=-lr * projected_grad)
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
