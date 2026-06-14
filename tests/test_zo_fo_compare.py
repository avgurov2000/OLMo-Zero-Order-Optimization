"""Tests for ZOAdamFOGradCompare (FO vs ZoAdam gradient norm metrics)."""

import math

import torch

from olmo.config import DistributedStrategy, OptimizerType, TrainConfig
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
    for key in expected_keys:
        assert metrics[key] == metrics[key]  # not NaN
    assert metrics["scalar_sim"] == metrics["zo_scalar"]
    assert metrics["scalar_grad_aligment_fo_z"] == metrics["zo_scalar"]
    assert metrics["norm_zo"] >= 0.0
    assert metrics["norm_fo"] >= 0.0
    assert metrics["diff_norm_zo_fo"] == abs(metrics["norm_zo"] - metrics["norm_fo"])
    assert metrics["diff_norm_zo_proj_fo"] == abs(metrics["norm_zo"] - metrics["norm_proj_fo"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert metrics["diff_norm_zo_fo_scaled"] == abs(
        metrics["norm_zo"] / math.sqrt(num_params) - metrics["norm_fo"]
    )
    assert metrics["diff_norm_zo_proj_fo_scaled"] == abs(
        metrics["norm_zo"] / (metrics["norm_z"] ** 2) - metrics["norm_proj_fo"]
    )

    post = opt.get_post_step_metrics()
    assert abs(metrics["norm_zo"] - post["grad_est_norm"].item()) <= max(1.0, 1e-6 * post["grad_est_norm"].item())
