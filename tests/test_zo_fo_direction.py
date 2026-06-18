"""Tests for ZOAdamFOGradCompare (FO vs ZoAdam gradient norm metrics)."""

import math

import torch

from olmo.config import DistributedStrategy, OptimizerType, TrainConfig, ZoFODirectionConfig, ZoFOSamplingStrategyConfig
from olmo.model import OLMo
from olmo.optim import build_optimizer
from olmo.zo_probe import ZOAdamFOGradCompare


def _make_closure(model, batch):
    def closure():
        out = model(input_ids=batch["input_ids"])
        return out.logits.sum()
    return closure


def test_zo_adam_fo_compare_metrics_cpu():
    cfg = TrainConfig()
    cfg.distributed_strategy = DistributedStrategy.single
    cfg.optimizer.name = OptimizerType.zo_adam
    cfg.optimizer.learning_rate = 1e-3
    cfg.optimizer.zo_eps = 1e-2

    model = OLMo(cfg.model)
    model.train()
    opt = build_optimizer(cfg, model)
    batch = {"input_ids": torch.randint(0, model.config.vocab_size, (1, 16))}

    model.zero_grad(set_to_none=True)
    loss = model(input_ids=batch["input_ids"]).logits.sum()
    loss.backward()

    compare = ZOAdamFOGradCompare()
    opt.step(_make_closure(model, batch), z_seed=11)

    metrics = compare.compute_metrics(opt)
    expected_keys = {
        "zo_scalar",
        "norm_z",
        "norm_zo",
        "norm_fo",
        "norm_proj_fo",
        "sim_fo_z",
        "grad_aligment_fo_z",
        "diff_norm_zo_fo",
        "diff_norm_zo_fo_scaled",
        "diff_norm_zo_proj_fo",
        "diff_norm_zo_proj_fo_scaled",
        "scalar_sim",
        "scalar_grad_aligment_fo_z",
    }
    assert set(metrics) == expected_keys

    post = opt.get_post_step_metrics()
    assert abs(metrics["norm_zo"] - post["grad_est_norm"].item()) <= max(1.0, 1e-6 * post["grad_est_norm"].item())


def test_zo_fo_direction_config_loads():
    cfg = TrainConfig()
    cfg.zo_fo_direction = ZoFODirectionConfig(
        enabled=True,
        normalize_direction=True,
        sampling_strategy=ZoFOSamplingStrategyConfig(
            strategy_type="fo_direction_norm_thrs",
            scalar_abs_fo_norm_ratio=0.1,
        ),
    )
    assert cfg.zo_fo_direction.normalize_direction is True
    assert cfg.zo_fo_direction.sampling_strategy.scalar_abs_fo_norm_ratio == 0.1


def test_zo_fo_direction_yaml_loads_sampling_strategy():
    cfg = TrainConfig.load(
        "configs/wiki/OLMo2-360M-wiki-ddp-zo-adam-fo-direction-scalar-thrs.yaml",
        validate_paths=False,
    )
    assert cfg.zo_fo_direction is not None
    assert cfg.zo_fo_direction.sampling_strategy.strategy_type == "fo_direction_scalar_thrs"
    assert cfg.zo_fo_direction.sampling_strategy.scalar_abs_threshold == 1e-4
