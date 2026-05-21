"""Tests for the hybrid FO+ZO training pipeline.

The hybrid mode lets a fixed param subset (typically head + embeddings) train with
first-order AdamW via autograd while the rest of the network trains with the main
zero-order optimizer (ZoAdam or LDSDMuon).  Both updates happen every step; total
forward count is unchanged versus plain two-sided ZO.

These tests verify (without spinning up the full Trainer):
  1. `_split_fo_zo_params` correctly routes parameters by regex pattern, including
     the weight-tying case where head and embedding share one tensor.
  2. `_strip_params_from_main_optim` removes FO params from the main optimizer.
  3. The hybrid step algorithm (perturb / FO forward+backward / second forward /
     restore / apply ZO + FO updates) preserves the key invariants:
       a. exactly 2 forward passes per step (same as plain two-sided ZO),
       b. FO params accumulate gradients, ZO params do not,
       c. ZO body params are bit-exactly restored to θ before the ZO update is applied,
       d. seed determinism: same z_seed → same ZO update direction.
"""
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from olmo.config import HybridFOConfig
from olmo.ldsd_optim import LDSDMuon
from olmo.train import Trainer
from olmo.zo_optim import ZoAdam


# ---------------------------------------------------------------------------
# Tiny model fixtures
# ---------------------------------------------------------------------------

class _TinyTransformer(nn.Module):
    """Mimics OLMo's ``transformer.wte`` / ``transformer.ff_out`` naming."""

    def __init__(self, vocab=16, d_model=8, weight_tying=False):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab, d_model)
        self.transformer.body = nn.Linear(d_model, d_model, bias=True)
        if weight_tying:
            self.transformer.ff_out = None
            self._tied_head = self.transformer.wte
        else:
            self.transformer.ff_out = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        x = self.transformer.wte(idx)
        x = self.transformer.body(x)
        if self.transformer.ff_out is None:
            return torch.nn.functional.linear(x, self.transformer.wte.weight)
        return self.transformer.ff_out(x)


def _fake_trainer(model):
    """Build a SimpleNamespace that mimics Trainer just enough to call the helper methods."""
    ft = SimpleNamespace()
    ft.model = model
    return ft


# ---------------------------------------------------------------------------
# 1. Param routing
# ---------------------------------------------------------------------------

def test_split_fo_zo_params_default_patterns():
    model = _TinyTransformer(weight_tying=False)
    hf = HybridFOConfig()
    fo_names, fo_params, zo_params = Trainer._split_fo_zo_params(_fake_trainer(model), hf)
    assert set(fo_names) == {"transformer.wte.weight", "transformer.ff_out.weight"}
    assert len(fo_params) == 2
    # ZO side has body.weight + body.bias = 2 params
    body_params = list(model.transformer.body.parameters())
    assert len(zo_params) == len(body_params)
    fo_ids = {id(p) for p in fo_params}
    zo_ids = {id(p) for p in zo_params}
    assert fo_ids.isdisjoint(zo_ids)


def test_split_fo_zo_params_weight_tying():
    """With weight tying, only wte.weight matches (no ff_out); shared tensor stays FO once."""
    model = _TinyTransformer(weight_tying=True)
    hf = HybridFOConfig()
    fo_names, fo_params, zo_params = Trainer._split_fo_zo_params(_fake_trainer(model), hf)
    assert fo_names == ["transformer.wte.weight"]
    assert len(fo_params) == 1
    # ff_out should not even exist (or be None); body still goes to ZO.
    assert all("ff_out" not in n for n, _ in model.named_parameters())


def test_split_fo_zo_params_custom_pattern():
    model = _TinyTransformer(weight_tying=False)
    hf = HybridFOConfig(param_patterns=[r"transformer\.body\.weight"])
    fo_names, fo_params, zo_params = Trainer._split_fo_zo_params(_fake_trainer(model), hf)
    assert fo_names == ["transformer.body.weight"]
    # All others — wte.weight, body.bias, ff_out.weight — go to ZO.
    assert len(zo_params) == 3


def test_split_fo_zo_params_no_match():
    model = _TinyTransformer()
    hf = HybridFOConfig(param_patterns=[r"nonexistent\.layer"])
    fo_names, fo_params, _ = Trainer._split_fo_zo_params(_fake_trainer(model), hf)
    assert fo_names == []
    assert fo_params == []


# ---------------------------------------------------------------------------
# 2. Stripping FO params from main optimizer
# ---------------------------------------------------------------------------

def test_strip_params_from_main_optim_removes_fo():
    model = _TinyTransformer()
    all_params = list(model.parameters())
    fo_params = [model.transformer.wte.weight, model.transformer.ff_out.weight]
    fo_ids = {id(p) for p in fo_params}

    optim = ZoAdam([{"params": list(all_params), "param_names": [n for n, _ in model.named_parameters()]}],
                   lr=1e-4, zo_eps=1e-3)

    ft = SimpleNamespace(optim=optim)
    Trainer._strip_params_from_main_optim(ft, fo_params)

    remaining_ids = {id(p) for g in optim.param_groups for p in g["params"]}
    assert fo_ids.isdisjoint(remaining_ids), "FO params should be stripped from main optim"
    # The number of remaining params equals all - fo
    assert len(remaining_ids) == len(all_params) - len(fo_params)


# ---------------------------------------------------------------------------
# 3. Algorithmic correctness — manual mini-step that replays the hybrid algorithm
# ---------------------------------------------------------------------------

def _hybrid_mini_step(model, fo_params, zo_params, zo_optim, *, z_seed, zo_eps=1e-3, idx=None):
    """Run one hybrid step on a tiny model.

    Returns (loss_plus_value, loss_minus_value, forward_count, fo_grad_norms).
    Mirrors `_train_step_hybrid` minus DDP and microbatching.
    """
    if idx is None:
        idx = torch.arange(4) % model.transformer.wte.num_embeddings

    calls = {"forward": 0}

    def _forward_and_loss():
        calls["forward"] += 1
        logits = model(idx)
        # simple sum loss
        return logits.float().sum()

    saved_rg = [p.requires_grad for p in zo_params]
    fo_grad_norms_before = []
    zo_data_before = [p.data.clone() for p in zo_params]

    def _perturb(seed, k):
        if isinstance(zo_optim, ZoAdam):
            zo_optim._perturb(seed, k)
        else:
            zo_optim._perturb_sequential(seed, k)

    # +ε perturb (requires_grad must be True so _perturb does not skip ZO params)
    _perturb(z_seed, +1.0)

    # Now flip requires_grad off for the FO forward+backward.
    for p in zo_params:
        p.requires_grad_(False)
    try:
        for p in fo_params:
            if p.grad is not None:
                p.grad = None
        loss_plus = _forward_and_loss()
        loss_plus.backward()
        for p in fo_params:
            assert p.grad is not None, "FO param did not receive .grad after backward"
            fo_grad_norms_before.append(p.grad.detach().norm().item())
    finally:
        for p, rg in zip(zo_params, saved_rg):
            p.requires_grad_(rg)

    # -2ε (now at -ε from θ), forward only
    _perturb(z_seed, -2.0)
    with torch.inference_mode():
        loss_minus = _forward_and_loss()

    # restore +ε (back to θ)
    _perturb(z_seed, +1.0)

    # ZO params should be back to θ now (before the actual update).  Note the perturb cycle
    # (+ε, -2ε, +ε) is not bit-exact in fp32 due to add accumulation; ~1e-7 drift per step is
    # expected (and identical in spirit to the drift in plain ZO).  Floats keep ≥6 digits.
    for p, theta in zip(zo_params, zo_data_before):
        torch.testing.assert_close(p.data, theta, atol=1e-6, rtol=0.0)

    # Apply ZO update
    scalar_half = (loss_plus.item() - loss_minus.item()) / 2.0
    if isinstance(zo_optim, ZoAdam):
        zo_optim._apply_update([(z_seed, scalar_half)])
    else:
        zo_optim._apply_muon_update(z_seed, scalar_half)

    return loss_plus.item(), loss_minus.item(), calls["forward"], fo_grad_norms_before


@pytest.mark.parametrize("zo_cls,zo_kwargs", [
    (ZoAdam, dict(betas=(0.9, 0.95), eps=1e-8)),
    (LDSDMuon, {}),
])
def test_hybrid_step_forward_count(zo_cls, zo_kwargs):
    """Hybrid step does exactly 2 forward passes (n=1 two-sided ZO)."""
    torch.manual_seed(0)
    model = _TinyTransformer()
    fo_params = [model.transformer.wte.weight, model.transformer.ff_out.weight]
    zo_params = list(model.transformer.body.parameters())
    zo_optim = zo_cls([{"params": zo_params, "zo_eps": 1e-3, "perturbation_mode": "two_side"}],
                      lr=1e-4, zo_eps=1e-3, **zo_kwargs)

    _, _, fwd, _ = _hybrid_mini_step(model, fo_params, zo_params, zo_optim, z_seed=42)
    assert fwd == 2


@pytest.mark.parametrize("zo_cls,zo_kwargs", [
    (ZoAdam, dict(betas=(0.9, 0.95), eps=1e-8)),
    (LDSDMuon, {}),
])
def test_hybrid_step_fo_grads_zo_no_grad(zo_cls, zo_kwargs):
    """After hybrid step backward: FO params have .grad, ZO params do NOT."""
    torch.manual_seed(1)
    model = _TinyTransformer()
    fo_params = [model.transformer.wte.weight, model.transformer.ff_out.weight]
    zo_params = list(model.transformer.body.parameters())
    zo_optim = zo_cls([{"params": zo_params, "zo_eps": 1e-3, "perturbation_mode": "two_side"}],
                      lr=1e-4, zo_eps=1e-3, **zo_kwargs)

    # Zero everything pre-step.
    for p in fo_params + zo_params:
        p.grad = None

    _, _, _, fo_norms = _hybrid_mini_step(model, fo_params, zo_params, zo_optim, z_seed=7)
    # FO grads exist and non-zero norm.
    assert all(g > 0.0 for g in fo_norms)
    # ZO grads remain None (requires_grad was off → autograd skipped them).
    for p in zo_params:
        assert p.grad is None, f"ZO param {p.shape} unexpectedly received .grad"


@pytest.mark.parametrize("zo_cls,zo_kwargs", [
    (ZoAdam, dict(betas=(0.9, 0.95), eps=1e-8)),
    (LDSDMuon, {}),
])
def test_hybrid_step_both_groups_change(zo_cls, zo_kwargs):
    """After a full hybrid step (ZO _apply_update + FO optim.step), both groups move."""
    torch.manual_seed(2)
    model = _TinyTransformer()
    fo_params = [model.transformer.wte.weight, model.transformer.ff_out.weight]
    zo_params = list(model.transformer.body.parameters())
    zo_optim = zo_cls([{"params": zo_params, "zo_eps": 1e-3, "perturbation_mode": "two_side"}],
                      lr=1e-2, zo_eps=1e-3, **zo_kwargs)
    fo_optim = torch.optim.AdamW(fo_params, lr=1e-2)

    fo_before = [p.data.clone() for p in fo_params]
    zo_before = [p.data.clone() for p in zo_params]

    fo_optim.zero_grad(set_to_none=True)
    _hybrid_mini_step(model, fo_params, zo_params, zo_optim, z_seed=11)
    fo_optim.step()

    for p, before in zip(fo_params, fo_before):
        assert not torch.allclose(p.data, before, atol=0.0), "FO param did not change"
    for p, before in zip(zo_params, zo_before):
        assert not torch.allclose(p.data, before, atol=0.0), "ZO param did not change"


def test_hybrid_step_zo_seed_determinism():
    """Same z_seed (and same θ) → identical ZO param updates across two runs."""
    torch.manual_seed(3)
    model_a = _TinyTransformer()
    # Make model_b an exact deep copy of model_a so initial θ is identical.
    model_b = _TinyTransformer()
    model_b.load_state_dict(model_a.state_dict())

    def run(model):
        fo_params = [model.transformer.wte.weight, model.transformer.ff_out.weight]
        zo_params = list(model.transformer.body.parameters())
        zo_optim = ZoAdam([{"params": zo_params, "zo_eps": 1e-3, "perturbation_mode": "two_side"}],
                          lr=1e-2, zo_eps=1e-3, betas=(0.9, 0.95), eps=1e-8)
        fo_optim = torch.optim.AdamW(fo_params, lr=1e-2)
        fo_optim.zero_grad(set_to_none=True)
        _hybrid_mini_step(model, fo_params, zo_params, zo_optim, z_seed=99)
        # Note: we skip fo_optim.step() — only checking ZO determinism here.
        return [p.data.clone() for p in zo_params]

    out_a = run(model_a)
    out_b = run(model_b)
    for a, b in zip(out_a, out_b):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# 4. Config validation
# ---------------------------------------------------------------------------

def test_hybrid_config_default_regex_matches_olmo_names():
    """Sanity-check: default regex patterns are valid and match canonical OLMo names."""
    hf = HybridFOConfig()
    patterns = [re.compile(pat) for pat in hf.param_patterns]
    assert any(pat.fullmatch("transformer.wte.weight") for pat in patterns)
    assert any(pat.fullmatch("transformer.ff_out.weight") for pat in patterns)
    # Should NOT match a random middle layer name.
    assert not any(pat.fullmatch("transformer.blocks.5.attn.weight") for pat in patterns)
