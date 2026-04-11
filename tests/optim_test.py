import pytest
import torch

from olmo.config import DistributedStrategy, ModelConfig, OptimizerType, TrainConfig
from olmo.exceptions import OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import BoltOnWarmupScheduler, LinearWithWarmup, build_optimizer
from olmo.train import cross_entropy_loss


def test_linear_with_warmup_scheduler():
    initial_lr = 1.0
    max_steps = 10_000
    scheduler = LinearWithWarmup(
        grad_clip_warmup_steps=None, grad_clip_warmup_factor=None, warmup_steps=2000, warmup_min_lr=None
    )
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 0.1
    assert scheduler.get_lr(initial_lr, 2000, max_steps) == 1.0
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 0.1
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) > scheduler.get_lr(initial_lr, 5_000, max_steps)


def test_bolt_on_warmup_scheduler():
    initial_lr = 1.0
    max_steps = 11_000
    alpha_f = 0.1
    scheduler = LinearWithWarmup(
        grad_clip_warmup_steps=None,
        grad_clip_warmup_factor=None,
        warmup_steps=1000,
        alpha_f=alpha_f,
        warmup_min_lr=None,
    )
    scheduler2 = BoltOnWarmupScheduler.wrap(scheduler, 5000, 6000)
    assert scheduler.get_lr(initial_lr, 100, max_steps) > 0.0
    assert scheduler2.get_lr(initial_lr, 100, max_steps) == 0.0
    assert scheduler2.get_lr(initial_lr, 5000, max_steps) == 0.0
    assert scheduler2.get_lr(initial_lr, 5500, max_steps) == pytest.approx(0.25 * (1 + alpha_f))
    assert scheduler2.get_lr(initial_lr, 6000, max_steps) == pytest.approx(0.5 * (1 + alpha_f))
    assert scheduler2.get_lr(initial_lr, 7000, max_steps) == scheduler.get_lr(initial_lr, 7000, max_steps)


def test_lozo_not_supported_with_fsdp():
    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.lozo
    cfg.distributed_strategy = DistributedStrategy.fsdp
    model = OLMo(ModelConfig())
    with pytest.raises(OLMoConfigurationError, match="FSDP"):
        build_optimizer(cfg, model)


def test_zo_adam_not_supported_with_fsdp():
    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.zo_adam
    cfg.distributed_strategy = DistributedStrategy.fsdp
    model = OLMo(ModelConfig())
    with pytest.raises(OLMoConfigurationError, match="FSDP"):
        build_optimizer(cfg, model)


def test_lozo_one_step_cpu():
    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.lozo
    cfg.distributed_strategy = DistributedStrategy.single
    cfg.optimizer.learning_rate = 1e-3
    cfg.optimizer.zo_eps = 1e-2
    model = OLMo(ModelConfig())
    model.train()
    opt = build_optimizer(cfg, model)
    batch = {"input_ids": torch.randint(0, model.config.vocab_size, (1, 16))}

    def closure():
        logits = model(input_ids=batch["input_ids"]).logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["input_ids"][..., 1:].contiguous().view(-1)
        loss, _ = cross_entropy_loss(logits, labels, ignore_index=-100, compute_z_loss=False)
        return loss

    opt.step(closure, z_seed=42)
    assert isinstance(closure().item(), float)


def test_zo_adam_one_step_cpu():
    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.zo_adam
    cfg.distributed_strategy = DistributedStrategy.single
    cfg.optimizer.learning_rate = 1e-3
    cfg.optimizer.zo_eps = 1e-2
    model = OLMo(ModelConfig())
    model.train()
    opt = build_optimizer(cfg, model)
    batch = {"input_ids": torch.randint(0, model.config.vocab_size, (1, 16))}

    def closure():
        logits = model(input_ids=batch["input_ids"]).logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["input_ids"][..., 1:].contiguous().view(-1)
        loss, _ = cross_entropy_loss(logits, labels, ignore_index=-100, compute_z_loss=False)
        return loss

    opt.step(closure, z_seed=7)
    assert isinstance(closure().item(), float)
