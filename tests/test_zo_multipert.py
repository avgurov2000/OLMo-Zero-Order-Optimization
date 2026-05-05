"""Tests for multi-sample SPSA (num_pert_samples > 1) in MeZO, ZoAdam, and LOZO.

Three invariants:
  1. Exactly 2*n forward passes per step.
  2. Parameters are restored to θ between consecutive samples (no accumulation).
  3. The gradient estimate equals (1/n) * Σᵢ scalar_i * zᵢ (averaged over n pairs).
"""
import torch
import pytest

from olmo.zo_optim import MeZO, ZoAdam, LOZO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param(vals):
    return torch.nn.Parameter(torch.tensor(vals, dtype=torch.float32))


def _mezo_z(seed: int, shape: torch.Size) -> torch.Tensor:
    """Replicate the z vector MeZO draws for a single CPU param with generator seeded to `seed`."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.normal(0.0, 1.0, size=shape, generator=gen)


def _lozo_z1d(seed: int, shape: torch.Size, param_idx: int = 0) -> torch.Tensor:
    """Replicate LOZO's _get_z1d for a 1-D CPU param."""
    gen = torch.Generator()
    gen.manual_seed((seed + param_idx * 999_983) % (2**31))
    return torch.randn(shape, generator=gen)


def _sample_seeds(z_seed: int, n: int):
    return [(z_seed + i * 987_654_321) % (2**31) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Forward pass count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("opt_cls,extra", [
    (MeZO,   {}),
    (ZoAdam, {"betas": (0.9, 0.95), "eps": 1e-8}),
    (LOZO,   {"rank": 2}),
])
def test_forward_count(opt_cls, extra, n):
    p = _param([1.0, 2.0, 3.0])
    calls = [0]

    def closure():
        calls[0] += 1
        return (p * p).sum()

    opt = opt_cls([p], lr=1e-3, zo_eps=1e-3, num_pert_samples=n, **extra)
    opt.step(closure, z_seed=7)
    assert calls[0] == 2 * n, f"{opt_cls.__name__} n={n}: expected {2*n} calls, got {calls[0]}"


# ---------------------------------------------------------------------------
# 2. Weights restored between samples (no perturbation accumulation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 3])
def test_mezo_weights_restored(n):
    """Between samples i and i+1 params must return to θ, not accumulate ε*z offsets."""
    zo_eps = 1e-2
    z_seed = 42
    p = _param([1.0, -2.0, 3.0])
    theta = p.data.clone()
    snapshots = []

    def closure():
        snapshots.append(p.data.clone())
        return p.sum()

    # lr=0 so the final update doesn't move θ; we only care about in-step positions.
    opt = MeZO([p], lr=0.0, zo_eps=zo_eps, num_pert_samples=n, weight_decay=0.0, momentum=0.0)
    opt.step(closure, z_seed=z_seed)

    assert len(snapshots) == 2 * n

    seeds = _sample_seeds(z_seed, n)
    for i, seed_i in enumerate(seeds):
        z_i = _mezo_z(seed_i, p.shape)
        # f+_i: θ + ε*z_i (independent of previous samples)
        torch.testing.assert_close(
            snapshots[2 * i], theta + zo_eps * z_i, atol=1e-6, rtol=0,
            msg=f"sample {i} f+: params should be θ+ε·z_{i}, not accumulated"
        )
        # f-_i: θ - ε*z_i
        torch.testing.assert_close(
            snapshots[2 * i + 1], theta - zo_eps * z_i, atol=1e-6, rtol=0,
            msg=f"sample {i} f-: params should be θ−ε·z_{i}"
        )


@pytest.mark.parametrize("n", [1, 3])
def test_zo_adam_weights_restored(n):
    """ZoAdam uses the same _perturb as MeZO — same positions expected."""
    zo_eps = 1e-2
    z_seed = 17
    p = _param([0.5, -1.0, 2.0])
    theta = p.data.clone()
    snapshots = []

    def closure():
        snapshots.append(p.data.clone())
        return p.sum()

    opt = ZoAdam([p], lr=0.0, zo_eps=zo_eps, num_pert_samples=n,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    opt.step(closure, z_seed=z_seed)

    assert len(snapshots) == 2 * n

    seeds = _sample_seeds(z_seed, n)
    for i, seed_i in enumerate(seeds):
        z_i = _mezo_z(seed_i, p.shape)
        torch.testing.assert_close(
            snapshots[2 * i], theta + zo_eps * z_i, atol=1e-6, rtol=0,
            msg=f"ZoAdam sample {i} f+: accumulated perturbation detected"
        )
        torch.testing.assert_close(
            snapshots[2 * i + 1], theta - zo_eps * z_i, atol=1e-6, rtol=0,
            msg=f"ZoAdam sample {i} f-: wrong restore position"
        )


@pytest.mark.parametrize("n", [1, 3])
def test_lozo_weights_restored_1d(n):
    """LOZO 1-D param: restored to θ before each new sample."""
    zo_eps = 1e-2
    z_seed = 99
    p = _param([1.0, -1.0])
    theta = p.data.clone()
    snapshots = []

    def closure():
        snapshots.append(p.data.clone())
        return p.sum()

    opt = LOZO([p], lr=0.0, zo_eps=zo_eps, rank=1, num_pert_samples=n, weight_decay=0.0)
    opt.step(closure, z_seed=z_seed)

    assert len(snapshots) == 2 * n

    seeds = _sample_seeds(z_seed, n)
    for i, seed_i in enumerate(seeds):
        z_i = _lozo_z1d(seed_i, p.shape, param_idx=0).to(p.dtype)
        torch.testing.assert_close(
            snapshots[2 * i], theta + zo_eps * z_i, atol=1e-5, rtol=0,
            msg=f"LOZO 1D sample {i} f+: accumulated perturbation"
        )
        torch.testing.assert_close(
            snapshots[2 * i + 1], theta - zo_eps * z_i, atol=1e-5, rtol=0,
            msg=f"LOZO 1D sample {i} f-: wrong restore"
        )


# ---------------------------------------------------------------------------
# 3. Gradient estimate = (1/n) * Σᵢ scalar_i * zᵢ
#
# For a linear loss f(θ) = <a, θ>:
#   scalar_i = (f(θ+εzᵢ) − f(θ−εzᵢ)) / 2 = ε·<a, zᵢ>
#   g_est_i  = scalar_i / ε · zᵢ = <a, zᵢ>·zᵢ
#   g_avg    = (1/n) · Σᵢ <a, zᵢ>·zᵢ
#   θ_new    = θ − lr · g_avg
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 4])
def test_mezo_gradient_average(n):
    zo_eps = 1e-3
    lr = 1.0
    z_seed = 123
    a = torch.tensor([0.5, -0.3, 0.8, 1.2])
    p = _param([1.0, 2.0, -1.0, 0.5])
    theta = p.data.clone()

    opt = MeZO([p], lr=lr, zo_eps=zo_eps, num_pert_samples=n, weight_decay=0.0, momentum=0.0)
    opt.step(lambda: (a * p).sum(), z_seed=z_seed)

    expected_g = torch.zeros_like(theta)
    for seed_i in _sample_seeds(z_seed, n):
        z_i = _mezo_z(seed_i, p.shape)
        scalar_i = zo_eps * (a * z_i).sum().item()   # ε·<a, zᵢ>
        expected_g += z_i * (scalar_i / zo_eps)       # <a, zᵢ>·zᵢ
    expected_g /= n

    torch.testing.assert_close(p.data, theta - lr * expected_g, atol=1e-5, rtol=0)


@pytest.mark.parametrize("n", [1, 4])
def test_lozo_gradient_average_1d(n):
    """1-D LOZO params: gradient estimate matches manual computation."""
    zo_eps = 1e-3
    lr = 1.0
    z_seed = 55
    a = torch.tensor([1.0, -2.0])
    p = _param([0.5, 1.5])
    theta = p.data.clone()

    opt = LOZO([p], lr=lr, zo_eps=zo_eps, rank=1, num_pert_samples=n, weight_decay=0.0)
    opt.step(lambda: (a * p).sum(), z_seed=z_seed)

    expected_z_acc = torch.zeros_like(theta)
    for seed_i in _sample_seeds(z_seed, n):
        z_i = _lozo_z1d(seed_i, p.shape, param_idx=0).to(theta.dtype)
        # projected_grad_i = (f+ - f-) / (2*eps) = <a, z_i>
        proj_i = (a * z_i).sum().item()
        expected_z_acc += z_i * proj_i
    expected_z_acc /= n

    torch.testing.assert_close(p.data, theta - lr * expected_z_acc, atol=1e-5, rtol=0)
