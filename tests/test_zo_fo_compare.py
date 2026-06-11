"""Tests for ZOAdamFOGradCompare (FO vs ZoAdam gradient norm metrics)."""

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
        "fo_grad_norm",
        "zo_grad_est_norm",
        "grad_norm_diff_abs",
        "grad_est_norm_ratio",
        "grad_est_norm_per_fo_z_norm",
        "fo_zo_recon_norm",
        "recon_norm_diff_abs",
    }
    assert set(metrics) == expected_keys
    for key in expected_keys:
        assert metrics[key] >= 0.0
        assert metrics[key] == metrics[key]  # not NaN
    assert abs(metrics["grad_est_norm_ratio"] - metrics["zo_grad_est_norm"] / metrics["fo_grad_norm"]) < 1e-6

    post = opt.get_post_step_metrics()
    zo_ref = post["grad_est_norm"].item()
    assert abs(metrics["zo_grad_est_norm"] - zo_ref) <= max(1.0, 1e-6 * zo_ref)
