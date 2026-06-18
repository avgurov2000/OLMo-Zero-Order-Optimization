"""Tests for ZoAdam FO-direction sampling strategies."""

import pytest

from olmo.config import ZoFOSamplingStrategyConfig
from olmo.exceptions import OLMoConfigurationError
from olmo.zo_fo_direction_strategy import (
    FoDirectionIntervalStrategy,
    FoDirectionNormThrsStrategy,
    FoDirectionScalarThrsStrategy,
    ZoFODirectionRuntime,
    build_zo_fo_direction_strategy,
    fo_direction_global_norm,
    normalize_fo_direction,
)


def _runtime(cache=None, last_refresh_step=-1):
    return ZoFODirectionRuntime(cache=cache, last_refresh_step=last_refresh_step)


def test_build_scalar_thrs_strategy():
    cfg = ZoFOSamplingStrategyConfig(
        strategy_type="fo_direction_scalar_thrs",
        scalar_abs_threshold=1e-4,
    )
    strategy = build_zo_fo_direction_strategy(cfg)
    assert isinstance(strategy, FoDirectionScalarThrsStrategy)


def test_build_unknown_strategy_raises():
    cfg = ZoFOSamplingStrategyConfig(strategy_type="does_not_exist")
    with pytest.raises(OLMoConfigurationError):
        build_zo_fo_direction_strategy(cfg)


def test_scalar_thrs_rejects_low_abs_S():
    cfg = ZoFOSamplingStrategyConfig(
        strategy_type="fo_direction_scalar_thrs",
        scalar_abs_threshold=1.0,
    )
    strategy = build_zo_fo_direction_strategy(cfg)
    runtime = _runtime(cache={1: None})  # type: ignore[arg-type]
    strategy.on_direction_refreshed(0, runtime, norm_fo=5.0)

    apply, forced = strategy.should_apply_update(0.5, runtime, refresh_attempts=1, max_refresh_retries=10)
    assert not apply and not forced
    assert strategy.should_refresh_before_probe(0, runtime)

    apply, forced = strategy.should_apply_update(1.5, runtime, refresh_attempts=1, max_refresh_retries=10)
    assert apply and not forced


def test_norm_thrs_sets_dynamic_threshold():
    cfg = ZoFOSamplingStrategyConfig(
        strategy_type="fo_direction_norm_thrs",
        scalar_abs_fo_norm_ratio=0.1,
    )
    strategy = build_zo_fo_direction_strategy(cfg)
    runtime = _runtime(cache={1: None})  # type: ignore[arg-type]
    strategy.on_direction_refreshed(3, runtime, norm_fo=10.0)
    assert runtime.scalar_threshold == pytest.approx(1.0)


def test_interval_refreshes_every_n_steps():
    cfg = ZoFOSamplingStrategyConfig(
        strategy_type="fo_direction_interval",
        refresh_interval=5,
    )
    strategy = build_zo_fo_direction_strategy(cfg)
    runtime = _runtime()

    assert strategy.should_refresh_before_probe(0, runtime)
    strategy.on_direction_refreshed(0, runtime, norm_fo=2.0)
    runtime.cache = {1: None}  # type: ignore[assignment]

    assert not strategy.should_refresh_before_probe(4, runtime)
    assert strategy.should_refresh_before_probe(5, runtime)

    apply, forced = strategy.should_apply_update(0.0, runtime, 1, 10)
    assert apply and not forced


def test_normalize_fo_direction_unit_norm():
    import torch

    direction = {0: torch.tensor([3.0, 4.0]), 1: torch.tensor([0.0, 0.0])}
    assert fo_direction_global_norm(direction) == pytest.approx(5.0)

    normalized, norm = normalize_fo_direction(direction)
    assert norm == pytest.approx(5.0)
    assert fo_direction_global_norm(normalized) == pytest.approx(1.0)
    assert normalized[0] == pytest.approx(torch.tensor([0.6, 0.8]))
