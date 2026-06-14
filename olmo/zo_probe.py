"""ZO probes: divergence (AdamW vs MeZO/ZOMuon) and FO vs ZoAdam gradient norms.

ZO Divergence Probe: measures alignment between AdamW and ZO update directions.

On each probe step, computes a MeZO and/or ZOMuon update direction (forward-only,
no backward) on the same batch that AdamW just processed, then logs cosine similarity
between p.grad (true gradient) and each ZO update direction.

Usage: call ``maybe_compute(step, seed, loss_fn)`` from ``Trainer.train_step`` after
backward() but before optimizer.step(). The ``loss_fn`` closure must run inference-mode
forward passes on the same batch and return a scalar loss.

DDP: pass a ``seed`` that has been broadcast from rank 0 (so all ranks use the same
perturbation directions). The ``loss_fn`` closure must already all_reduce its loss
(like the ZO training closure does). Since ``p.grad`` is all_reduced by DDP backward
and the ZO direction is identical on all ranks, the cosine sim is the same everywhere.
"""
from __future__ import annotations

import math
from typing import Callable, TYPE_CHECKING

import torch

from .zo_optim import _newtonschulz5

if TYPE_CHECKING:
    from .zo_optim import ZoAdam


class ZODivergenceProbe:
    def __init__(
        self,
        model: torch.nn.Module,
        zo_eps: float = 1e-3,
        mezo_enabled: bool = True,
        zomuon_enabled: bool = True,
        zomuon_rank: int = 4,
        zomuon_ns_steps: int = 5,
        probe_interval: int = 1,
    ):
        self.model = model
        self.zo_eps = zo_eps
        self.mezo_enabled = mezo_enabled
        self.zomuon_enabled = zomuon_enabled
        self.zomuon_rank = zomuon_rank
        self.zomuon_ns_steps = zomuon_ns_steps
        self.probe_interval = probe_interval
        self._probe_step = 0  # incremented each time compute() actually runs

    # ------------------------------------------------------------------
    # Seeded vector generators (independent from the training optimizers)
    # ------------------------------------------------------------------

    def _mezo_z(self, p: torch.Tensor, seed: int, idx: int) -> torch.Tensor:
        gen = torch.Generator(device=p.device)
        gen.manual_seed((seed + idx * 1_000_003) % (2**31))
        return torch.randn(p.shape, device=p.device, dtype=torch.float32, generator=gen)

    def _zomuon_p(self, p: torch.Tensor, idx: int) -> torch.Tensor:
        rank = min(self.zomuon_rank, p.shape[0])
        gen = torch.Generator(device=p.device)
        gen.manual_seed((self._probe_step * 1_000_003 + idx * 999_983) % (2**31))
        Q = torch.randn(p.shape[0], rank, device=p.device, dtype=torch.float32, generator=gen)
        P, R = torch.linalg.qr(Q, mode="reduced")
        signs = R.diagonal().sign()
        signs[signs == 0] = 1.0
        return P * signs  # Haar-correct orthogonal matrix, float32

    def _zomuon_u(self, p: torch.Tensor, seed: int, idx: int) -> torch.Tensor:
        rank = min(self.zomuon_rank, p.shape[0])
        gen = torch.Generator(device=p.device)
        # Salt 987_654_321 prevents seed collision with _mezo_z at idx=0
        # (both would otherwise evaluate to `seed` when idx=0).
        gen.manual_seed((seed + idx * 999_983 + 987_654_321) % (2**31))
        return torch.randn(rank, p.shape[1], device=p.device, dtype=torch.float32, generator=gen)

    def _zomuon_z1d(self, p: torch.Tensor, seed: int, idx: int) -> torch.Tensor:
        gen = torch.Generator(device=p.device)
        gen.manual_seed((seed + idx * 111_317 + 987_654_321) % (2**31))
        return torch.randn(p.shape, device=p.device, dtype=torch.float32, generator=gen)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_compute(
        self,
        step: int,
        seed: int,
        loss_fn: Callable[[], torch.Tensor],
    ) -> dict[str, float]:
        """Run probe and return cosine-similarity metrics; returns {} on off-steps.

        Parameters
        ----------
        step:
            Current global training step (used to check probe_interval).
        seed:
            RNG seed broadcast from rank-0 (ensures all DDP ranks sample identical
            perturbation vectors).
        loss_fn:
            Closure that runs a forward-only pass on the current batch and returns
            the global scalar loss (all_reduced across ranks).
        """
        if step % self.probe_interval != 0:
            return {}

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params or not any(p.grad is not None for p in params):
            return {}

        metrics: dict[str, float] = {}

        if self.mezo_enabled:
            metrics.update(self._run_mezo(params, seed, loss_fn))

        if self.zomuon_enabled:
            metrics.update(self._run_zomuon(params, seed, loss_fn))

        self._probe_step += 1
        return metrics

    # ------------------------------------------------------------------
    # MeZO probe: θ ± ε·z, antithetic estimator, 2 forward passes
    # ------------------------------------------------------------------

    def _run_mezo(
        self,
        params: list[torch.Tensor],
        seed: int,
        loss_fn: Callable[[], torch.Tensor],
    ) -> dict[str, float]:
        with torch.no_grad():
            for idx, p in enumerate(params):
                z = self._mezo_z(p, seed, idx)
                p.data.add_(z.to(p.dtype), alpha=self.zo_eps)

        # _current_scale tracks params position relative to θ:
        # +1.0 → params at θ + ε·z  (restore with -ε·z)
        # -1.0 → params at θ - ε·z  (restore with +ε·z)
        _current_scale = 1.0
        try:
            loss_plus = loss_fn()

            with torch.no_grad():
                for idx, p in enumerate(params):
                    z = self._mezo_z(p, seed, idx)
                    p.data.add_(z.to(p.dtype), alpha=-2.0 * self.zo_eps)
            _current_scale = -1.0

            loss_minus = loss_fn()
        finally:
            with torch.no_grad():
                for idx, p in enumerate(params):
                    z = self._mezo_z(p, seed, idx)
                    p.data.add_(z.to(p.dtype), alpha=-_current_scale * self.zo_eps)

        scalar = (loss_plus - loss_minus).item() / (2.0 * self.zo_eps)

        # Cosine between p.grad and the gradient estimate scalar·z
        dot = 0.0
        sq_g = 0.0
        sq_zo = 0.0
        with torch.no_grad():
            for idx, p in enumerate(params):
                if p.grad is None:
                    continue
                z = self._mezo_z(p, seed, idx)
                zo_dir = z * scalar  # gradient estimate direction
                g = p.grad.float()
                dot += (g * zo_dir).sum().item()
                sq_g += g.pow(2).sum().item()
                sq_zo += zo_dir.pow(2).sum().item()

        cos = dot / (math.sqrt(sq_g * sq_zo) + 1e-8)
        return {"mezo_cosine": cos, "mezo_scalar_abs": abs(scalar)}

    # ------------------------------------------------------------------
    # ZOMuon probe: θ ± ε·P@u, antithetic estimator, 2 forward passes
    # ------------------------------------------------------------------

    def _run_zomuon(
        self,
        params: list[torch.Tensor],
        seed: int,
        loss_fn: Callable[[], torch.Tensor],
    ) -> dict[str, float]:
        # Pre-compute per-param perturbation directions (stored to avoid recomputing P@u).
        pert_dirs: list[torch.Tensor] = []  # delta in float32
        ps_list: list[torch.Tensor | None] = []  # P matrix for 2D params
        us_list: list[torch.Tensor] = []  # u (2D) or z (1D)

        with torch.no_grad():
            for idx, p in enumerate(params):
                if p.dim() == 2:
                    P = self._zomuon_p(p, idx)
                    u = self._zomuon_u(p, seed, idx)
                    delta = P @ u  # (m, n) float32
                    ps_list.append(P)
                    us_list.append(u)
                else:
                    z = self._zomuon_z1d(p, seed, idx)
                    ps_list.append(None)
                    us_list.append(z)
                    delta = z
                pert_dirs.append(delta)
                p.data.add_(delta.to(p.dtype), alpha=self.zo_eps)

        _current_scale = 1.0
        try:
            loss_plus = loss_fn()

            with torch.no_grad():
                for p, delta in zip(params, pert_dirs):
                    p.data.add_(delta.to(p.dtype), alpha=-2.0 * self.zo_eps)
            _current_scale = -1.0

            loss_minus = loss_fn()
        finally:
            with torch.no_grad():
                for p, delta in zip(params, pert_dirs):
                    p.data.add_(delta.to(p.dtype), alpha=-_current_scale * self.zo_eps)

        scalar = (loss_plus - loss_minus).item() / (2.0 * self.zo_eps)

        # Cosine between p.grad and the ZOMuon update direction P @ NS(scalar·u)
        dot = 0.0
        sq_g = 0.0
        sq_zo = 0.0
        with torch.no_grad():
            for p, P_or_none, u_or_z in zip(params, ps_list, us_list):
                if p.grad is None:
                    continue
                g = p.grad.float()
                if P_or_none is not None:
                    lowdim = u_or_z * scalar  # (rank, n), scale u by scalar
                    M_sign = _newtonschulz5(lowdim, steps=self.zomuon_ns_steps)
                    zo_dir = P_or_none @ M_sign  # (m, n)
                else:
                    zo_dir = u_or_z * scalar
                dot += (g * zo_dir).sum().item()
                sq_g += g.pow(2).sum().item()
                sq_zo += zo_dir.pow(2).sum().item()

        cos = dot / (math.sqrt(sq_g * sq_zo) + 1e-8)
        return {"zomuon_cosine": cos, "zomuon_scalar_abs": abs(scalar)}


def register_zo_fo_compare_wandb_metrics() -> None:
    """Register custom x-axes for scalar-vs-alignment W&B charts."""
    import wandb

    wandb.define_metric("zo_fo_compare/scalar_sim", step_metric="zo_fo_compare/sim_fo_z")
    wandb.define_metric(
        "zo_fo_compare/scalar_grad_aligment_fo_z",
        step_metric="zo_fo_compare/grad_aligment_fo_z",
    )


class ZOAdamFOGradCompare:
    """FO vs ZoAdam gradient diagnostics for W&B (``zo_fo_compare/*`` metrics).

    Call ``compute_metrics`` after ``ZoAdam.step`` while ``p.grad`` still holds the FO
    backward from the same batch.  Uses the same probe vectors and scalars as ZoAdam.
    """

    def __init__(self, probe_interval: int = 1):
        self.probe_interval = probe_interval

    def should_run(self, step: int) -> bool:
        return step % self.probe_interval == 0

    @staticmethod
    def _trainable_params(optim: "ZoAdam") -> list[torch.Tensor]:
        params: list[torch.Tensor] = []
        for group in optim.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
        return params

    @staticmethod
    def _fo_grad_norm(params: list[torch.Tensor]) -> float:
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += p.grad.detach().float().norm().item() ** 2
        return math.sqrt(sq)

    def compute_metrics(self, optim: "ZoAdam") -> dict[str, float]:
        samples = optim._last_pert_samples
        if not samples:
            return {}

        params = self._trainable_params(optim)
        if not params or not any(p.grad is not None for p in params):
            return {}

        zo_eps = optim.defaults["zo_eps"]
        n = len(samples)
        num_params = sum(p.numel() for p in params)
        sqrt_d = math.sqrt(num_params)

        fo_grad_norm = self._fo_grad_norm(params)

        zo_acc: dict[int, torch.Tensor] = {id(p): torch.zeros_like(p, dtype=torch.float32) for p in params}
        zo_scalar_sum = 0.0
        norm_z_sum = 0.0
        norm_proj_fo_sum = 0.0
        sim_sum = 0.0
        grad_aligment_sum = 0.0

        for seed_i, scalar_i in samples:
            # S = (L(θ+μz) − L(θ−μz)) / (2μ); ZoAdam stores (L+−L−)/2 in scalar_i.
            s = scalar_i / zo_eps
            zo_scalar_sum += s

            optim._reset_generators(seed_i)
            z_sq = 0.0
            dot = 0.0
            for p in params:
                gen = optim._get_generator(p.device)
                z = optim.vector_sampler.sample(p.shape, p.device, gen)
                z_f = z.float()
                z_sq += z_f.norm().item() ** 2
                if p.grad is not None:
                    dot += (p.grad.detach().float() * z_f).sum().item()
                zo_acc[id(p)].add_(z_f, alpha=s)

            norm_z_i = math.sqrt(z_sq)
            norm_z_sum += norm_z_i
            norm_proj_fo_sum += abs(dot) / norm_z_i if norm_z_i > 0.0 else 0.0
            sim_sum += dot / (fo_grad_norm * norm_z_i) if fo_grad_norm > 0.0 and norm_z_i > 0.0 else 0.0
            grad_aligment_sum += dot * dot

        zo_scalar = zo_scalar_sum / n
        norm_z = norm_z_sum / n
        norm_proj_fo = norm_proj_fo_sum / n
        sim_fo_z = sim_sum / n
        grad_aligment_fo_z = grad_aligment_sum / n

        zo_grad_sq = 0.0
        for p in params:
            zo_vec = zo_acc[id(p)].div_(n)
            zo_grad_sq += zo_vec.norm().item() ** 2
        norm_zo = math.sqrt(zo_grad_sq)

        diff_norm_zo_fo = abs(norm_zo - fo_grad_norm)
        diff_norm_zo_fo_scaled = abs(norm_zo / sqrt_d - fo_grad_norm) if sqrt_d > 0.0 else float("nan")
        diff_norm_zo_proj_fo = abs(norm_zo - norm_proj_fo)
        diff_norm_zo_proj_fo_scaled = (
            abs(norm_zo / (norm_z * norm_z) - norm_proj_fo) if norm_z > 0.0 else float("nan")
        )

        return {
            "zo_scalar": zo_scalar,
            "norm_z": norm_z,
            "norm_zo": norm_zo,
            "norm_fo": fo_grad_norm,
            "norm_proj_fo": norm_proj_fo,
            "sim_fo_z": sim_fo_z,
            "grad_aligment_fo_z": grad_aligment_fo_z,
            "diff_norm_zo_fo": diff_norm_zo_fo,
            "diff_norm_zo_fo_scaled": diff_norm_zo_fo_scaled,
            "diff_norm_zo_proj_fo": diff_norm_zo_proj_fo,
            "diff_norm_zo_proj_fo_scaled": diff_norm_zo_proj_fo_scaled,
            # Same S on Y; custom W&B x-axis via define_metric (see register_zo_fo_compare_wandb_metrics).
            "scalar_sim": zo_scalar,
            "scalar_grad_aligment_fo_z": zo_scalar,
        }
