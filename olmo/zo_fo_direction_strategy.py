"""Pluggable refresh policies for ZoAdam training with ``z = g_fo``."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .config import ZoFOSamplingStrategyConfig
from .exceptions import OLMoConfigurationError

if TYPE_CHECKING:
    import torch


@dataclass
class ZoFODirectionRuntime:
    """Mutable per-run state for cached FO probe directions."""

    cache: Optional[dict[int, torch.Tensor]] = None
    norm_fo: Optional[float] = None
    scalar_threshold: Optional[float] = None
    last_refresh_step: int = -1
    _probe_rejected: bool = field(default=False, repr=False)

    def clear_probe_rejection(self) -> None:
        self._probe_rejected = False

    def mark_probe_rejected(self) -> None:
        self._probe_rejected = True


class ZoFODirectionSamplingStrategy(ABC):
    """Decides when to refresh ``z = g_fo`` and when to accept a ZO probe."""

    strategy_type: str

    def __init__(self, cfg: ZoFOSamplingStrategyConfig):
        self.cfg = cfg

    @abstractmethod
    def on_direction_refreshed(self, global_step: int, runtime: ZoFODirectionRuntime, norm_fo: float) -> None:
        """Update runtime after a new ``g_fo`` snapshot was taken."""

    @abstractmethod
    def should_refresh_before_probe(self, global_step: int, runtime: ZoFODirectionRuntime) -> bool:
        """Whether to run FO backward before the next ZO probe on this training step."""

    @abstractmethod
    def should_apply_update(
        self,
        abs_S: float,
        runtime: ZoFODirectionRuntime,
        refresh_attempts: int,
        max_refresh_retries: int,
    ) -> tuple[bool, bool]:
        """Return ``(apply_update, forced_update)`` after a ZO probe."""

    def extra_metrics(self, runtime: ZoFODirectionRuntime) -> dict[str, float]:
        return {}


class FoDirectionScalarThrsStrategy(ZoFODirectionSamplingStrategy):
    """Refresh when ``|S|`` falls below a fixed constant threshold."""

    strategy_type = "fo_direction_scalar_thrs"

    def __init__(self, cfg: ZoFOSamplingStrategyConfig):
        super().__init__(cfg)
        if cfg.scalar_abs_threshold is None or cfg.scalar_abs_threshold <= 0:
            raise OLMoConfigurationError(
                "fo_direction_scalar_thrs requires sampling_strategy.scalar_abs_threshold > 0"
            )

    def on_direction_refreshed(self, global_step: int, runtime: ZoFODirectionRuntime, norm_fo: float) -> None:
        runtime.norm_fo = norm_fo
        runtime.scalar_threshold = self.cfg.scalar_abs_threshold
        runtime.last_refresh_step = global_step
        runtime.clear_probe_rejection()

    def should_refresh_before_probe(self, global_step: int, runtime: ZoFODirectionRuntime) -> bool:
        del global_step
        return runtime.cache is None or runtime._probe_rejected

    def should_apply_update(
        self,
        abs_S: float,
        runtime: ZoFODirectionRuntime,
        refresh_attempts: int,
        max_refresh_retries: int,
    ) -> tuple[bool, bool]:
        assert runtime.scalar_threshold is not None
        if abs_S >= runtime.scalar_threshold:
            runtime.clear_probe_rejection()
            return True, False
        if refresh_attempts >= max_refresh_retries:
            runtime.clear_probe_rejection()
            return True, True
        runtime.mark_probe_rejected()
        return False, False


class FoDirectionNormThrsStrategy(ZoFODirectionSamplingStrategy):
    """Refresh when ``|S|`` falls below ``‖g_fo‖ · scalar_abs_fo_norm_ratio``."""

    strategy_type = "fo_direction_norm_thrs"

    def __init__(self, cfg: ZoFOSamplingStrategyConfig):
        super().__init__(cfg)
        if cfg.scalar_abs_fo_norm_ratio is None or cfg.scalar_abs_fo_norm_ratio <= 0:
            raise OLMoConfigurationError(
                "fo_direction_norm_thrs requires sampling_strategy.scalar_abs_fo_norm_ratio > 0"
            )

    def on_direction_refreshed(self, global_step: int, runtime: ZoFODirectionRuntime, norm_fo: float) -> None:
        runtime.norm_fo = norm_fo
        runtime.scalar_threshold = norm_fo * self.cfg.scalar_abs_fo_norm_ratio  # type: ignore[operator]
        runtime.last_refresh_step = global_step
        runtime.clear_probe_rejection()

    def should_refresh_before_probe(self, global_step: int, runtime: ZoFODirectionRuntime) -> bool:
        del global_step
        return runtime.cache is None or runtime._probe_rejected

    def should_apply_update(
        self,
        abs_S: float,
        runtime: ZoFODirectionRuntime,
        refresh_attempts: int,
        max_refresh_retries: int,
    ) -> tuple[bool, bool]:
        assert runtime.scalar_threshold is not None
        if abs_S >= runtime.scalar_threshold:
            runtime.clear_probe_rejection()
            return True, False
        if refresh_attempts >= max_refresh_retries:
            runtime.clear_probe_rejection()
            return True, True
        runtime.mark_probe_rejected()
        return False, False


class FoDirectionIntervalStrategy(ZoFODirectionSamplingStrategy):
    """Refresh ``z = g_fo`` every ``refresh_interval`` global training steps."""

    strategy_type = "fo_direction_interval"

    def __init__(self, cfg: ZoFOSamplingStrategyConfig):
        super().__init__(cfg)
        if cfg.refresh_interval is None or cfg.refresh_interval < 1:
            raise OLMoConfigurationError(
                "fo_direction_interval requires sampling_strategy.refresh_interval >= 1"
            )

    def on_direction_refreshed(self, global_step: int, runtime: ZoFODirectionRuntime, norm_fo: float) -> None:
        runtime.norm_fo = norm_fo
        runtime.scalar_threshold = None
        runtime.last_refresh_step = global_step
        runtime.clear_probe_rejection()

    def should_refresh_before_probe(self, global_step: int, runtime: ZoFODirectionRuntime) -> bool:
        if runtime.cache is None:
            return True
        if runtime.last_refresh_step < 0:
            return True
        return (global_step - runtime.last_refresh_step) >= self.cfg.refresh_interval  # type: ignore[operator]

    def should_apply_update(
        self,
        abs_S: float,
        runtime: ZoFODirectionRuntime,
        refresh_attempts: int,
        max_refresh_retries: int,
    ) -> tuple[bool, bool]:
        del abs_S, runtime, refresh_attempts, max_refresh_retries
        return True, False

    def extra_metrics(self, runtime: ZoFODirectionRuntime) -> dict[str, float]:
        return {
            "refresh_interval": float(self.cfg.refresh_interval),  # type: ignore[arg-type]
            "last_refresh_step": float(runtime.last_refresh_step),
        }


_STRATEGY_TYPES: dict[str, type[ZoFODirectionSamplingStrategy]] = {
    FoDirectionScalarThrsStrategy.strategy_type: FoDirectionScalarThrsStrategy,
    FoDirectionNormThrsStrategy.strategy_type: FoDirectionNormThrsStrategy,
    FoDirectionIntervalStrategy.strategy_type: FoDirectionIntervalStrategy,
}


def build_zo_fo_direction_strategy(cfg: ZoFOSamplingStrategyConfig) -> ZoFODirectionSamplingStrategy:
    cls = _STRATEGY_TYPES.get(cfg.strategy_type)
    if cls is None:
        known = ", ".join(sorted(_STRATEGY_TYPES))
        raise OLMoConfigurationError(
            f"Unknown zo_fo_direction sampling_strategy.strategy_type={cfg.strategy_type!r}; "
            f"known types: {known}"
        )
    return cls(cfg)
