"""Tests for ZoAdam FO-cached direction training."""

import torch

from olmo.config import DistributedStrategy, OptimizerType, TrainConfig, ZoFODirectionConfig
from olmo.model import OLMo
from olmo.optim import build_optimizer
from olmo.zo_optim import ZoAdam


def _make_closure(model, batch):
    def closure():
        out = model(input_ids=batch["input_ids"])
        return out.logits.sum()
    return closure


def test_zo_adam_estimate_and_apply_direction_cpu():
    cfg = TrainConfig()
    cfg.distributed_strategy = DistributedStrategy.single
    cfg.optimizer.name = OptimizerType.zo_adam
    cfg.optimizer.zo_eps = 1e-2

    model = OLMo(cfg.model)
    model.train()
    opt = build_optimizer(cfg, model)
    assert isinstance(opt, ZoAdam)
    batch = {"input_ids": torch.randint(0, model.config.vocab_size, (1, 16))}

    model.zero_grad(set_to_none=True)
    loss = model(input_ids=batch["input_ids"]).logits.sum()
    loss.backward()

    direction = {
        id(p): p.grad.detach().float().clone()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    }
    params_before = [p.data.clone() for p in model.parameters() if p.requires_grad]

    _, scalar = opt.estimate_with_direction(_make_closure(model, batch), direction)
    params_after_probe = [p.data.clone() for p in model.parameters() if p.requires_grad]
    max_probe_drift = max(
        (before - after).abs().max().item() for before, after in zip(params_before, params_after_probe)
    )
    assert max_probe_drift < 1e-2

    opt.apply_direction_update(scalar, direction)
    params_after_update = [p.data for p in model.parameters() if p.requires_grad]
    assert any(not torch.allclose(b, a) for b, a in zip(params_before, params_after_update))
    assert opt._last_metrics["scalar_S_abs"] == abs(scalar / opt.defaults["zo_eps"])


def test_zo_fo_direction_config_loads():
    cfg = TrainConfig()
    cfg.zo_fo_direction = ZoFODirectionConfig(enabled=True, scalar_abs_threshold=1e-4)
    assert cfg.zo_fo_direction.scalar_abs_threshold == 1e-4
