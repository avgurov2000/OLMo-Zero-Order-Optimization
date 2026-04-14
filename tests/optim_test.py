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


def test_ldsd_muon_not_supported_with_fsdp():
    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.ldsd_muon
    cfg.distributed_strategy = DistributedStrategy.fsdp
    model = OLMo(ModelConfig())
    with pytest.raises(OLMoConfigurationError, match="FSDP"):
        build_optimizer(cfg, model)


def test_ldsd_sign_sgd_not_supported_with_fsdp():
    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.ldsd_sign_sgd
    cfg.distributed_strategy = DistributedStrategy.fsdp
    model = OLMo(ModelConfig())
    with pytest.raises(OLMoConfigurationError, match="FSDP"):
        build_optimizer(cfg, model)


def _make_closure(model, batch):
    def closure():
        logits = model(input_ids=batch["input_ids"]).logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["input_ids"][..., 1:].contiguous().view(-1)
        loss, _ = cross_entropy_loss(logits, labels, ignore_index=-100, compute_z_loss=False)
        return loss
    return closure


def test_ldsd_muon_one_step_cpu():
    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.ldsd_muon
    cfg.distributed_strategy = DistributedStrategy.single
    cfg.optimizer.learning_rate = 1e-3
    cfg.optimizer.zo_eps = 1e-2
    model = OLMo(ModelConfig())
    model.train()
    opt = build_optimizer(cfg, model)
    batch = {"input_ids": torch.randint(0, model.config.vocab_size, (1, 16))}
    closure = _make_closure(model, batch)

    loss = opt.step(closure, z_seed=42)
    assert isinstance(loss.item(), float)

    metrics = opt.get_post_step_metrics()
    assert "projected_grad_abs" in metrics
    assert "grad_est_norm" in metrics
    assert metrics["projected_grad_abs"].item() >= 0.0


def test_ldsd_muon_deterministic_with_seed():
    """Same z_seed must produce identical updates across two model copies.

    Newton-Schulz uses bfloat16 internally; allow a small absolute tolerance
    (~1e-5) to accommodate rounding in multi-threaded BLAS on CPU.
    """
    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.ldsd_muon
    cfg.distributed_strategy = DistributedStrategy.single
    cfg.optimizer.learning_rate = 1e-3
    cfg.optimizer.zo_eps = 1e-2

    # Single-threaded matmul → deterministic Newton-Schulz across both runs.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        model_a = OLMo(ModelConfig())
        model_a.train()

        torch.manual_seed(0)
        model_b = OLMo(ModelConfig())
        model_b.train()

        opt_a = build_optimizer(cfg, model_a)
        opt_b = build_optimizer(cfg, model_b)
        batch = {"input_ids": torch.randint(0, model_a.config.vocab_size, (1, 16))}

        opt_a.step(_make_closure(model_a, batch), z_seed=99)
        opt_b.step(_make_closure(model_b, batch), z_seed=99)

        for (name, pa), (_, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
            assert torch.allclose(pa, pb, atol=1e-5), (
                f"Parameter '{name}' diverged despite identical z_seed; "
                f"max diff = {(pa - pb).abs().max().item():.2e}"
            )
    finally:
        torch.set_num_threads(prev_threads)


def test_ldsd_sign_sgd_one_step_cpu():
    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.ldsd_sign_sgd
    cfg.distributed_strategy = DistributedStrategy.single
    cfg.optimizer.learning_rate = 1e-3
    cfg.optimizer.zo_eps = 1e-2
    model = OLMo(ModelConfig())
    model.train()
    opt = build_optimizer(cfg, model)
    batch = {"input_ids": torch.randint(0, model.config.vocab_size, (1, 16))}
    closure = _make_closure(model, batch)

    loss = opt.step(closure, z_seed=7)
    assert isinstance(loss.item(), float)

    metrics = opt.get_post_step_metrics()
    assert "projected_grad_abs" in metrics
    assert "grad_est_norm" in metrics


def test_ldsd_sign_sgd_step_size_insensitive_to_loss_scale():
    """SignSGD: doubling the loss should not change the update direction or magnitude."""
    import copy

    cfg = TrainConfig()
    cfg.optimizer.name = OptimizerType.ldsd_sign_sgd
    cfg.distributed_strategy = DistributedStrategy.single
    cfg.optimizer.learning_rate = 1e-3
    cfg.optimizer.zo_eps = 1e-2

    torch.manual_seed(0)
    model_a = OLMo(ModelConfig())
    model_a.train()
    params_before = {n: p.clone() for n, p in model_a.named_parameters()}

    torch.manual_seed(0)
    model_b = copy.deepcopy(model_a)
    model_b.train()

    batch = {"input_ids": torch.randint(0, model_a.config.vocab_size, (1, 16))}

    def closure_scaled():
        logits = model_b(input_ids=batch["input_ids"]).logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["input_ids"][..., 1:].contiguous().view(-1)
        loss, _ = cross_entropy_loss(logits, labels, ignore_index=-100, compute_z_loss=False)
        return loss * 2.0  # scaled loss

    opt_a = build_optimizer(cfg, model_a)
    cfg2 = TrainConfig()
    cfg2.optimizer.name = OptimizerType.ldsd_sign_sgd
    cfg2.distributed_strategy = DistributedStrategy.single
    cfg2.optimizer.learning_rate = 1e-3
    cfg2.optimizer.zo_eps = 1e-2
    opt_b = build_optimizer(cfg2, model_b)

    opt_a.step(_make_closure(model_a, batch), z_seed=55)
    opt_b.step(closure_scaled, z_seed=55)

    for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.allclose(pa, pb), f"Param {na} differs despite sign compression"
