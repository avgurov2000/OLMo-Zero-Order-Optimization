from __future__ import annotations

import cProfile
import functools
import gc
import logging
import math
import os
import random
import shutil
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils
import torch.utils.hooks
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .aliases import PathOrStr
from .checkpoint import Checkpointer, FullCheckpointer, build_sharded_checkpointer
from .config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    HybridFOConfig,
    PhaseSwitchConfig,
    SchedulerUnits,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
    TrainConfig,
    ZOAdamFOGradCompareConfig,
    ZoFODirectionConfig,
    ZOProbeConfig,
)
from .data import IterableDataset
from .eval import Evaluator
from .exceptions import OLMoConfigurationError
from .model import OLMo
from torch.optim import Optimizer as TorchOptimizer

from .optim import CosWithWarmup, Scheduler
from .torch_util import (
    SingleAccelerator,
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    is_distributed,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value,
)
from .ldsd_optim import LDSDMuon, LDSDRl, LDSDRlAdaMM, LDSDRlSgd
from .zo_optim import ZeroOrderOptimizer, ZoAdam
from .zo_fo_direction_strategy import (
    ZoFODirectionRuntime,
    build_zo_fo_direction_strategy,
    fo_direction_global_norm,
    normalize_fo_direction,
)
from .zo_probe import ZOAdamFOGradCompare, ZODivergenceProbe, register_zo_fo_compare_wandb_metrics
from .util import upload

__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]

log = logging.getLogger(__name__)


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_total_tokens: int = 0
    total_training_Gflops: float = 0
    device_interval_tokens: Deque[int] = field(default_factory=lambda: deque([]))

    def batch_start(
        self,
        global_total_tokens: int,
        device_batch_num_tokens: int,
        num_fwd_flops: int,
        num_bck_flops: int,
        record: bool = True,
    ) -> None:
        self.global_total_tokens = global_total_tokens
        # num_fwd_flops and num_bck_flops from the OLMo model computes flops per token
        # converting to GFLOPs here prevents numerical issues while logging
        self.total_training_Gflops = (num_fwd_flops + num_bck_flops) * global_total_tokens / 1e9

        if record:
            if len(self.start_times) >= self.cfg.window_size:
                self.start_times.popleft()
                self.device_interval_tokens.popleft()
            self.start_times.append(time.monotonic())
            self.device_interval_tokens.append(device_batch_num_tokens)

    def reset(self) -> None:
        self.start_times.clear()
        self.device_interval_tokens.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}

        # plot flops related metrics
        metrics["throughput/total_training_Gflops"] = self.total_training_Gflops
        metrics["throughput/total_training_log_Gflops"] = math.log(self.total_training_Gflops)

        if self.start_times:
            interval_seconds = time.monotonic() - self.start_times[0]
            interval_batches = len(self.start_times)
            interval_tokens = sum(self.device_interval_tokens)
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


def cross_entropy_loss(
    logits,
    labels,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss


fused_loss_fn: Optional[Callable]

try:
    import flash_attn
    from flash_attn.ops.triton.cross_entropy import (
        cross_entropy_loss as flash_cross_entropy_loss,  # type: ignore
    )

    def fused_loss_fn(
        logits,
        labels,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
    ):
        # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
        ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

        if ce_loss_use_ignore_index_param:
            ignore_index_kwarg = {"ignore_index": ignore_index}
        else:
            ignore_index_kwarg = {"ignored_index": ignore_index}

        loss, z_loss = flash_cross_entropy_loss(
            logits,
            labels,
            label_smoothing=0.0,
            logit_scale=1.0,
            lse_square_scale=z_loss_multiplier,
            inplace_backward=False,
            process_group=None,
            **ignore_index_kwarg,
        )

        mask = labels != ignore_index

        if reduction == "mean":
            loss = loss.sum() / mask.sum()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss

        if not compute_z_loss:
            return loss, None

        if reduction == "mean":
            z_loss = z_loss.sum() / mask.sum()
        elif reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss

        return loss, z_loss

except ImportError:
    fused_loss_fn = None


@dataclass
class Trainer:
    cfg: TrainConfig
    model: OLMo
    dist_model: Union[DDP, FSDP, SingleAccelerator]
    optim: TorchOptimizer
    scheduler: Scheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[Evaluator]
    epoch: Optional[int] = None
    global_step: int = 0
    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""
    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""
    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    indices_file: Optional[TextIO] = None
    _start_time: float = 0.0
    _gc_init_state: bool = True
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None

    def __post_init__(self):
        if self.cfg.fused_loss:
            if fused_loss_fn is not None:
                self.loss_fn = fused_loss_fn
            else:
                raise NameError("`fused_loss_fn` is not defined. Please ensure that `flash_attn` is installed.")

        probe_cfg: Optional[ZOProbeConfig] = self.cfg.zo_probe
        if probe_cfg is not None and probe_cfg.enabled and not isinstance(self.optim, ZeroOrderOptimizer):
            self._zo_probe: Optional[ZODivergenceProbe] = ZODivergenceProbe(
                model=self.model,
                zo_eps=probe_cfg.zo_eps,
                mezo_enabled=probe_cfg.mezo_enabled,
                zomuon_enabled=probe_cfg.zomuon_enabled,
                zomuon_rank=probe_cfg.zomuon_rank,
                zomuon_ns_steps=probe_cfg.zomuon_ns_steps,
                probe_interval=probe_cfg.probe_interval,
            )
        else:
            self._zo_probe = None

        compare_cfg: Optional[ZOAdamFOGradCompareConfig] = self.cfg.zo_adam_fo_compare
        fo_dir_cfg: Optional[ZoFODirectionConfig] = self.cfg.zo_fo_direction
        if (
            compare_cfg is not None
            and compare_cfg.enabled
            and fo_dir_cfg is not None
            and fo_dir_cfg.enabled
        ):
            raise OLMoConfigurationError(
                "zo_adam_fo_compare and zo_fo_direction cannot both be enabled; "
                "use zo_fo_direction for training with cached g_fo probes."
            )
        if compare_cfg is not None and compare_cfg.enabled and isinstance(self.optim, ZoAdam):
            self._zo_fo_compare: Optional[ZOAdamFOGradCompare] = ZOAdamFOGradCompare(
                probe_interval=compare_cfg.probe_interval,
            )
        else:
            self._zo_fo_compare = None

        # LDSDRl / LDSDRlAdaMM / LDSDRlSgd: μ_0-from-FO-gradient init runs once, on the
        # first zero-order training step (see `_train_step_zero_order`).
        self._ldsd_rl_mu_fo_initialized = False

        if fo_dir_cfg is not None and fo_dir_cfg.enabled:
            if not isinstance(self.optim, ZoAdam):
                raise OLMoConfigurationError("zo_fo_direction requires optimizer.name=zo_adam")
            self._zo_fo_direction_runtime = ZoFODirectionRuntime()
            self._zo_fo_direction_strategy = build_zo_fo_direction_strategy(fo_dir_cfg.sampling_strategy)
        else:
            self._zo_fo_direction_runtime = ZoFODirectionRuntime()
            self._zo_fo_direction_strategy = None

        ps: Optional[PhaseSwitchConfig] = self.cfg.phase_switch
        if ps is not None:
            all_params = [p for group in self.optim.param_groups for p in group["params"]]
            self._phase1_optim: Optional[torch.optim.AdamW] = torch.optim.AdamW(
                all_params,
                lr=ps.learning_rate,
                betas=tuple(ps.betas),  # type: ignore[arg-type]
                eps=ps.eps,
                weight_decay=ps.weight_decay,
            )
            self._phase1_scheduler: Optional[CosWithWarmup] = CosWithWarmup(
                grad_clip_warmup_steps=None,
                grad_clip_warmup_factor=None,
                warmup_steps=ps.t_warmup,
                alpha_f=ps.alpha_f,
                warmup_min_lr=ps.warmup_min_lr,
            )
        else:
            self._phase1_optim = None
            self._phase1_scheduler = None

        # Hybrid FO+ZO setup: split params into FO subset (head+embed) and ZO rest.
        hf: Optional[HybridFOConfig] = self.cfg.hybrid_fo
        if hf is not None:
            if self.cfg.phase_switch is not None:
                raise OLMoConfigurationError(
                    "hybrid_fo and phase_switch are mutually exclusive — pick one."
                )
            if not isinstance(self.optim, (ZoAdam, LDSDMuon)):
                raise OLMoConfigurationError(
                    "hybrid_fo currently supports main optimizer of type ZoAdam or LDSDMuon, "
                    f"got {type(self.optim).__name__}.  Other ZO variants (LOZO, ZOMuon, LDSDRl, "
                    "etc.) have different perturb/apply contracts and are not wired into the "
                    "hybrid step yet."
                )
            self._fo_param_names, self._fo_params, self._zo_params = self._split_fo_zo_params(hf)
            if len(self._fo_params) == 0:
                raise OLMoConfigurationError(
                    f"hybrid_fo.param_patterns matched zero parameters; patterns: {hf.param_patterns}"
                )
            self._strip_params_from_main_optim(self._fo_params)
            self._fo_optim: Optional[torch.optim.AdamW] = torch.optim.AdamW(
                self._fo_params,
                lr=hf.learning_rate,
                betas=tuple(hf.betas),  # type: ignore[arg-type]
                eps=hf.eps,
                weight_decay=hf.weight_decay,
            )
            self._fo_scheduler: Optional[CosWithWarmup] = CosWithWarmup(
                grad_clip_warmup_steps=None,
                grad_clip_warmup_factor=None,
                warmup_steps=hf.t_warmup,
                alpha_f=hf.alpha_f,
                warmup_min_lr=hf.warmup_min_lr,
            )
            log.info(
                "Hybrid FO+ZO mode: %d FO params (%s), %d ZO params remain in main optimizer",
                len(self._fo_params),
                ", ".join(self._fo_param_names),
                len(self._zo_params),
            )
        else:
            self._fo_param_names: List[str] = []
            self._fo_params: List[torch.nn.Parameter] = []
            self._zo_params: List[torch.nn.Parameter] = []
            self._fo_optim = None
            self._fo_scheduler = None

    def _split_fo_zo_params(
        self, hf: HybridFOConfig
    ) -> Tuple[List[str], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        import re

        patterns = [re.compile(pat) for pat in hf.param_patterns]
        fo_names: List[str] = []
        fo_params: List[torch.nn.Parameter] = []
        seen_fo_ids: set = set()
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(pat.fullmatch(name) for pat in patterns):
                if id(p) in seen_fo_ids:
                    continue  # weight-tying: same tensor matched twice
                fo_names.append(name)
                fo_params.append(p)
                seen_fo_ids.add(id(p))

        zo_params: List[torch.nn.Parameter] = []
        seen_zo_ids: set = set()
        for _, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen_fo_ids or id(p) in seen_zo_ids:
                continue
            zo_params.append(p)
            seen_zo_ids.add(id(p))
        return fo_names, fo_params, zo_params

    def _strip_params_from_main_optim(self, fo_params: List[torch.nn.Parameter]) -> None:
        """Remove fo_params from self.optim.param_groups so ZO _perturb/_apply_update ignores them."""
        fo_ids = {id(p) for p in fo_params}
        for group in self.optim.param_groups:
            kept_indices = [i for i, p in enumerate(group["params"]) if id(p) not in fo_ids]
            group["params"] = [group["params"][i] for i in kept_indices]
            if "param_names" in group:
                group["param_names"] = [group["param_names"][i] for i in kept_indices]

    @property
    def dataset(self) -> IterableDataset:
        assert isinstance(self.train_loader.dataset, IterableDataset)
        return self.train_loader.dataset

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        return math.ceil(self.max_steps / self.batches_per_epoch)

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = math.ceil(tokens_remaining / self.tokens_per_batch)
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(int(float(self.cfg.max_duration)) - self.global_step, 0) * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch or 0,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        self.checkpoints = [
            path
            for path in state_dict["checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.unsharded_checkpoints = [
            path
            for path in state_dict["unsharded_checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.ephemeral_checkpoints = [
            path
            for path in state_dict.get("ephemeral_checkpoints", [])
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch") or 0
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict.get(
            "global_train_examples_seen_this_epoch",
            state_dict.get(  # for backwards compatibility
                "global_train_examples_seen",
                state_dict.get("global_data_step", self.global_step) * self.cfg.global_train_batch_size,
            ),
        )
        self.global_train_tokens_seen = state_dict.get(
            "global_train_tokens_seen",
            state_dict.get("global_data_step", self.global_step)  # for backwards compatibility
            * self.cfg.global_train_batch_size
            * self.cfg.model.max_sequence_length,
        )

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_step = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0

        assert self.epoch is not None
        # Reshuffle dataset if needed.
        if self.dataset.epoch != self.epoch:
            log.info(f"Reshuffling data loader for epoch {self.epoch}...")
            self.dataset.reshuffle(self.epoch)

        if self.cfg.fast_forward_batches:
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        if self.global_train_examples_seen_this_epoch > 0:
            assert isinstance(self.dataset, IterableDataset)
            log.info(f"Data loader will start at instance index {self.global_train_examples_seen_this_epoch:,d}")
            self.dataset.start_index = self.global_train_examples_seen_this_epoch

        # Reset learning rate and weight decay to the values from the config, not the checkpoint.
        log.info("Resetting learning rate...")
        new_learning_rate = self.scheduler.get_lr(
            self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
        )
        for group in self.optim.param_groups:
            group["lr"] = new_learning_rate
            group["initial_lr"] = self.cfg.optimizer.learning_rate
            if "weight_decay" in group and group["weight_decay"] > 0.0:
                group["weight_decay"] = self.cfg.optimizer.weight_decay

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available():
            if rng_state["cuda"] is not None:
                torch.cuda.set_rng_state(rng_state["cuda"])
            else:
                log.warning("CUDA is available, but no RNG state was provided.")
        if torch.backends.mps.is_available():
            if rng_state["mps"] is not None:
                torch.mps.set_rng_state(rng_state["mps"])
            else:
                log.warning("MPS is available, but no RNG state was provided.")

    def _save_checkpoint(
        self, checkpointer: Checkpointer, checkpoint_type: CheckpointType
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.unsharded:
            suffix = "-unsharded"
            current_checkpoints = self.unsharded_checkpoints
            link_latest = get_global_rank() == 0
            num_checkpoints_to_keep = self.cfg.save_num_unsharded_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            link_latest = get_fs_local_rank() == 0
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        # Flush data indices file.
        # TODO: upload the indices files?
        if self.indices_file is not None:
            self.indices_file.flush()

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}{suffix}"
        remote_checkpoint_dir: Optional[str] = None
        if self.cfg.remote_save_folder is not None:
            remote_checkpoint_dir = f"{self.cfg.remote_save_folder.rstrip('/')}/{checkpoint_dir.name}"
        current_checkpoints.append(checkpoint_dir)

        # Save the checkpoint.
        try:
            checkpointer.save_checkpoint(
                checkpoint_dir,
                self.dist_model,
                self.optim,
                self.trainer_state_dict(),
                upload_to=remote_checkpoint_dir,
            )
        except FileExistsError:
            raise OLMoConfigurationError(
                f"Checkpoint for step {self.global_step} already exists, use --save_overwrite to overwrite it"
            )

        if link_latest:
            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / f"latest{suffix}"
            latest_path.unlink(missing_ok=True)
            try:
                latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)
            except FileExistsError:
                # Same as above, caught when another (file-system) local rank 0 has already made the 'latest' symlink.
                # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                # file-systems.
                if latest_path.resolve().name != checkpoint_dir.name:
                    raise

        # Remove old checkpoints.
        # For DDP, checkpoint_type being passed to remove_checkpoint is always `unsharded`.
        if num_checkpoints_to_keep > 0:
            while len(current_checkpoints) > num_checkpoints_to_keep:
                self.remove_checkpoint(0, checkpoint_type)

        barrier()

        if remote_checkpoint_dir is not None:
            return remote_checkpoint_dir, checkpoint_dir
        else:
            return checkpoint_dir, None

    def save_sharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def save_ephemeral_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = build_sharded_checkpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.sharded_ephemeral)
        self.last_sharded_checkpoint_step = self.global_step
        return result

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def remove_sharded_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.checkpoints)

    def remove_ephemeral_checkpoint(self, idx: int = 0):
        self._remove_sharded_checkpoint(idx, self.ephemeral_checkpoints)

    def restore_sharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = build_sharded_checkpointer(self.cfg, name=sharded_checkpointer)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_unsharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        checkpointer = FullCheckpointer(self.cfg)
        result = self._save_checkpoint(checkpointer, CheckpointType.unsharded)
        self.last_unsharded_checkpoint_step = self.global_step
        return result

    def remove_unsharded_checkpoint(self, idx: int = 0):
        barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_unsharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
    ):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)
        checkpointer = FullCheckpointer(self.cfg)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            self.optim,
            local_cache=local_cache,
            load_optimizer_state=load_optimizer_state,
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
        barrier()

    def save_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.sharded
    ) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        result: Tuple[PathOrStr, Optional[PathOrStr]]
        if checkpoint_type == CheckpointType.sharded:
            result = self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            result = self.save_unsharded_checkpoint()
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            result = self.save_ephemeral_checkpoint()
        else:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()
        return result

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        checkpoint_type: Optional[CheckpointType] = None,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        sharded_checkpointer: Optional[ShardedCheckpointerType] = None,
    ):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).rstrip("/").endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
            )
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(
                load_path,
                local_cache=local_cache,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
                sharded_checkpointer=sharded_checkpointer,
            )
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()

    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            self.remove_ephemeral_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def _setup_module_output_save_hooks(self, micro_batch_idx: int) -> List[torch.utils.hooks.RemovableHandle]:
        if (
            self.cfg.module_outputs_save_steps is None
            or self.global_step not in self.cfg.module_outputs_save_steps
        ):
            return []

        if micro_batch_idx != 0 or get_global_rank() != 0:
            # Hook is currently only used on the first microbatch of rank 0
            return []

        trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
        if trace_save_folder.exists():
            if self.cfg.save_overwrite:
                shutil.rmtree(trace_save_folder)
            else:
                raise OLMoConfigurationError(
                    f"Attempting to overwrite traces at step {self.global_step} without --save_overwrite"
                )
        trace_save_folder.mkdir(parents=True)

        def trace_outputs_hook(
            module_name: str, _: torch.nn.Module, args: Tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            if len(args) == 0:
                log.info("No input args for module %s, output %s", module_name, output)

            module_input = args[0] if len(args) > 0 else torch.tensor(())
            trace_save_folder = Path(self.cfg.save_folder) / f"traces/step{self.global_step}"
            trace_save_folder.mkdir(parents=True, exist_ok=True)

            module_occurence_num = 0
            while (
                module_input_filepath := trace_save_folder / f"{module_name}_{module_occurence_num}_input.pt"
            ).exists():
                module_occurence_num += 1
            torch.save(module_input, module_input_filepath)

            module_output_filepath = trace_save_folder / f"{module_name}_{module_occurence_num}_output.pt"
            torch.save(output, module_output_filepath)

        output_hooks = []
        for module_name, module in self.model.named_modules(prefix="model"):
            output_hooks.append(module.register_forward_hook(functools.partial(trace_outputs_hook, module_name)))

        return output_hooks

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.dist_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            max_doc_lens=batch.get("max_doc_lens"),
        ).logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = self.get_labels(batch)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, z_loss, logits

    def train_micro_batch(
        self, micro_batch: Dict[str, Any], batch_size_in_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ce_loss, z_loss, logits = self.model_forward(
            micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss, loss_reduction="sum"
        )
        ce_loss = ce_loss / batch_size_in_tokens

        # In case this helps with memory utilization.
        del micro_batch

        # Get loss to optimize for.
        if self.cfg.softmax_auxiliary_loss:
            assert z_loss is not None
            z_loss = z_loss / batch_size_in_tokens
            loss = ce_loss + z_loss
        else:
            loss = ce_loss

        del logits

        return loss, ce_loss, z_loss

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        batch_size_in_tokens = batch["input_ids"].numel()

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            # setup sync context for DDP for all micro-batches except the last
            grad_sync_context = nullcontext
            if (
                self.cfg.distributed_strategy == DistributedStrategy.ddp
                and self.cfg.ddp is not None
                and self.cfg.ddp.grad_sync_mode == DDPGradSyncMode.batch
            ):
                if micro_batch_idx != num_micro_batches - 1:
                    grad_sync_context = self.dist_model.no_sync

            # Register output hooks
            output_hooks: List[torch.utils.hooks.RemovableHandle] = []
            output_hooks += self._setup_module_output_save_hooks(micro_batch_idx)

            with grad_sync_context():
                autocast_device = "mps" if self.device.type == "mps" else "cuda"
                with torch.autocast(autocast_device, enabled=True, dtype=self.cfg.autocast_precision):
                    # Run forward pass.
                    loss, ce_loss, z_loss = self.train_micro_batch(micro_batch, batch_size_in_tokens)

                    # Update overall CE batch loss.
                    ce_batch_loss += ce_loss.detach()

                    # Update overall Z batch loss.
                    if z_loss is not None:
                        assert z_batch_loss is not None
                        z_batch_loss += z_loss.detach()

                # Run backward pass.
                loss.backward()

            # Remove output hooks
            for hook in output_hooks:
                hook.remove()

        return ce_batch_loss, z_batch_loss

    def _snapshot_fo_direction(self) -> dict[int, torch.Tensor]:
        direction: dict[int, torch.Tensor] = {}
        for group in self.optim.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    if p.grad is None:
                        raise RuntimeError("FO backward did not populate gradients for zo_fo_direction")
                    direction[id(p)] = p.grad.detach().float().clone()
        return direction

    @staticmethod
    def _fo_direction_norm(direction: dict[int, torch.Tensor]) -> float:
        return fo_direction_global_norm(direction)

    def _refresh_zo_fo_direction_cache(self, batch: Dict[str, Any]) -> float:
        self.optim.zero_grad(set_to_none=True)
        self.train_batch(batch)
        raw = self._snapshot_fo_direction()
        norm_fo = self._fo_direction_norm(raw)
        fo_dir = self.cfg.zo_fo_direction
        assert fo_dir is not None
        if fo_dir.normalize_direction:
            self._zo_fo_direction_runtime.cache, _ = normalize_fo_direction(raw)
        else:
            self._zo_fo_direction_runtime.cache = raw
        return norm_fo

    def _train_step_zo_fo_direction(
        self, batch: Dict[str, Any], reduce_global_loss: bool = True
    ) -> Dict[str, float]:
        """ZoAdam with cached ``z = g_fo``; refresh policy from ``sampling_strategy``."""
        assert isinstance(self.optim, ZoAdam)
        fo_dir = self.cfg.zo_fo_direction
        assert fo_dir is not None and fo_dir.enabled
        strategy = self._zo_fo_direction_strategy
        assert strategy is not None
        runtime = self._zo_fo_direction_runtime

        metrics: Dict[str, float] = {}

        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        batch = move_to_device(batch, self.device)
        batch_size_in_tokens = batch["input_ids"].numel()
        micro_batches = self.split_batch(batch)

        def closure() -> torch.Tensor:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            autocast_device = "mps" if self.device.type == "mps" else "cuda"
            with torch.inference_mode():
                with torch.autocast(autocast_device, enabled=True, dtype=self.cfg.autocast_precision):
                    for micro_batch in micro_batches:
                        loss, _, _ = self.train_micro_batch(micro_batch, batch_size_in_tokens)
                        total_loss = total_loss + loss.float()
            if is_distributed():
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                total_loss = total_loss / get_world_size()
            return total_loss

        for group in self.optim.param_groups:
            group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
            )

        zo_eps = self.optim.defaults["zo_eps"]
        refresh_attempts = 0
        backward_passes = 0
        forced_update = False

        loss: Optional[torch.Tensor] = None
        while True:
            if strategy.should_refresh_before_probe(self.global_step, runtime):
                norm_fo = self._refresh_zo_fo_direction_cache(batch)
                strategy.on_direction_refreshed(self.global_step, runtime, norm_fo)
                backward_passes += 1
                refresh_attempts += 1

            assert runtime.cache is not None
            loss, raw_scalar = self.optim.estimate_with_direction(closure, runtime.cache)
            abs_S = abs(raw_scalar / zo_eps)

            apply_update, forced = strategy.should_apply_update(
                abs_S, runtime, refresh_attempts, fo_dir.max_refresh_retries
            )
            if apply_update:
                if forced:
                    forced_update = True
                    log.warning(
                        "zo_fo_direction[%s]: |S|=%.3e below threshold %.3e after %d refresh attempts; "
                        "applying update anyway.",
                        strategy.strategy_type,
                        abs_S,
                        runtime.scalar_threshold if runtime.scalar_threshold is not None else float("nan"),
                        refresh_attempts,
                    )
                self.optim.apply_direction_update(raw_scalar, runtime.cache)
                break

        assert loss is not None
        ce_batch_loss = loss.detach()
        if ce_batch_loss.device != self.device:
            ce_batch_loss = ce_batch_loss.to(self.device)
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())

        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")

        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        if should_log_optim_metrics_this_step and hasattr(self.optim, "get_post_step_metrics"):
            optim_metrics = self.optim.get_post_step_metrics(
                self.dist_model, process_group=self.dist_model.process_group
            )
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        metrics["zo_fo_direction/abs_S"] = abs_S
        metrics["zo_fo_direction/norm_fo"] = runtime.norm_fo or 0.0
        metrics["zo_fo_direction/norm_z"] = (
            self._fo_direction_norm(runtime.cache) if runtime.cache is not None else 0.0
        )
        metrics["zo_fo_direction/normalize_direction"] = float(fo_dir.normalize_direction)
        metrics["zo_fo_direction/scalar_threshold"] = (
            runtime.scalar_threshold if runtime.scalar_threshold is not None else 0.0
        )
        metrics["zo_fo_direction/steps_since_refresh"] = (
            float(self.global_step - runtime.last_refresh_step)
            if runtime.last_refresh_step >= 0
            else 0.0
        )
        for key, value in strategy.extra_metrics(runtime).items():
            metrics[f"zo_fo_direction/{key}"] = value
        metrics["zo_fo_direction/refresh_attempts"] = float(refresh_attempts)
        metrics["zo_fo_direction/backward_passes"] = float(backward_passes)
        metrics["zo_fo_direction/direction_refreshed"] = float(backward_passes > 0)
        metrics["zo_fo_direction/forced_update"] = float(forced_update)

        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)

        return metrics

    def _train_step_zero_order(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        """
        MeZO / LOZO: no backward; optimizer step runs 2–3 forward passes via ``closure``.
        Loss inside ``closure`` is averaged over distributed ranks so all replicas apply the same update.
        """
        metrics: Dict[str, float] = {}

        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        self.optim.zero_grad(set_to_none=True)
        batch = move_to_device(batch, self.device)
        batch_size_in_tokens = batch["input_ids"].numel()
        micro_batches = self.split_batch(batch)

        run_fo_compare = (
            self._zo_fo_compare is not None and self._zo_fo_compare.should_run(self.global_step)
        )
        should_init_mu_from_fo = (
            self.cfg.optimizer.ldsd_rl_mu_init_from_fo
            and isinstance(self.optim, (LDSDRl, LDSDRlAdaMM, LDSDRlSgd))
            and not self._ldsd_rl_mu_fo_initialized
            # global_step is incremented before train_step() is called, so `1` is the very
            # first batch of a fresh run; guards against clobbering μ on checkpoint resume.
            and self.global_step == 1
        )
        if run_fo_compare or should_init_mu_from_fo:
            self.train_batch(batch)
        if should_init_mu_from_fo:
            fo_init_metrics = self.optim.init_mu_from_fo_grad(
                normalize=self.cfg.optimizer.ldsd_rl_mu_init_normalize
            )
            metrics["train/ldsd_rl_mu_fo_init_norm"] = fo_init_metrics["fo_grad_norm"]
            self._ldsd_rl_mu_fo_initialized = True
            if not run_fo_compare:
                self.optim.zero_grad(set_to_none=True)

        z_seed: Optional[int] = None
        if is_distributed():
            z_seed_local = int(np.random.randint(0, 1_000_000_000)) if get_global_rank() == 0 else 0
            z_seed = int(synchronize_value(z_seed_local, self.device))

        def closure() -> torch.Tensor:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            autocast_device = "mps" if self.device.type == "mps" else "cuda"
            with torch.inference_mode():
                with torch.autocast(autocast_device, enabled=True, dtype=self.cfg.autocast_precision):
                    for micro_batch in micro_batches:
                        loss, _, _ = self.train_micro_batch(micro_batch, batch_size_in_tokens)
                        total_loss = total_loss + loss.float()
            if is_distributed():
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                total_loss = total_loss / get_world_size()
            return total_loss

        for group in self.optim.param_groups:
            group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
            )
            if "max_grad_norm" in group:
                group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
                )
            if "max_grad_norm_ratio" in group:
                group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                    self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
                )

        loss = self.optim.step(closure, z_seed=z_seed)  # type: ignore[call-arg]

        ce_batch_loss = loss.detach()
        if ce_batch_loss.device != self.device:
            ce_batch_loss = ce_batch_loss.to(self.device)
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())

        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")

        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        if should_log_optim_metrics_this_step and hasattr(self.optim, "get_post_step_metrics"):
            optim_metrics = self.optim.get_post_step_metrics(
                self.dist_model, process_group=self.dist_model.process_group
            )
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        if run_fo_compare and isinstance(self.optim, ZoAdam):
            for _k, _v in self._zo_fo_compare.compute_metrics(self.optim).items():
                metrics[f"zo_fo_compare/{_k}"] = _v

        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)

        return metrics

    @property
    def _phase1_sched_current(self) -> int:
        ps = self.cfg.phase_switch
        assert ps is not None
        return self.global_step if ps.switch_units == "steps" else self.global_train_tokens_seen

    @property
    def _in_phase1(self) -> bool:
        if self._phase1_optim is None:
            return False
        return self._phase1_sched_current < self.cfg.phase_switch.switch_after  # type: ignore[union-attr]

    def _train_step_phase1(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        """FO AdamW warm-up step (phase 1).  Runs forward+backward; no ZO closures."""
        metrics: Dict[str, float] = {}
        ps = self.cfg.phase_switch
        assert ps is not None and self._phase1_optim is not None and self._phase1_scheduler is not None

        self._phase1_optim.zero_grad(set_to_none=True)
        batch = move_to_device(batch, self.device)
        ce_batch_loss, z_batch_loss = self.train_batch(batch)

        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())

        # Gradient clipping (mirrors main FO path; no fancy metric collection).
        if self.cfg.max_grad_norm is not None and self.cfg.max_grad_norm > 0:
            all_params = [p for g in self._phase1_optim.param_groups for p in g["params"]]
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.cfg.max_grad_norm)
            metrics["train/phase1_grad_norm"] = grad_norm.item()

        # LR from phase1 scheduler.
        lr = self._phase1_scheduler.get_lr(ps.learning_rate, self._phase1_sched_current, ps.switch_after)
        for group in self._phase1_optim.param_groups:
            group["lr"] = lr

        self._phase1_optim.step()

        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")

        metrics["train/ce_loss"] = ce_batch_loss.item()
        if z_batch_loss is not None:
            metrics["train/z_loss"] = z_batch_loss.item()
        metrics["train/phase1_lr"] = lr
        metrics["train/phase1_active"] = 1.0
        return metrics

    @property
    def _fo_sched_current(self) -> int:
        return self.scheduler_current

    @property
    def _fo_sched_max(self) -> int:
        hf = self.cfg.hybrid_fo
        if hf is not None and hf.t_max is not None:
            return hf.t_max
        return self.scheduler_max

    def _train_step_hybrid(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        """Hybrid FO+ZO step: FO AdamW on head+embed via autograd, ZO body via SPSA.

        Total forward passes per step: 2 (same as plain two-sided ZO).  The FO
        backward piggy-backs on the +ε perturbed forward, so no extra forward
        is needed.  ZO body params are temporarily set to ``requires_grad=False``
        around the FO forward so autograd does not trace them (saves memory and
        avoids spurious ``.grad`` writes).
        """
        assert self.cfg.hybrid_fo is not None
        assert self._fo_optim is not None and self._fo_scheduler is not None
        hf = self.cfg.hybrid_fo
        metrics: Dict[str, float] = {}

        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        # Sync z_seed across DDP ranks so all ranks perturb identically.
        z_seed: int
        if is_distributed():
            z_seed_local = int(np.random.randint(0, 1_000_000_000)) if get_global_rank() == 0 else 0
            z_seed = int(synchronize_value(z_seed_local, self.device))
        else:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        batch = move_to_device(batch, self.device)
        batch_size_in_tokens = batch["input_ids"].numel()
        micro_batches = self.split_batch(batch)
        num_micro_batches = len(micro_batches)

        self._fo_optim.zero_grad(set_to_none=True)

        # NOTE on ordering: ZoAdam._perturb / LDSDMuon._perturb_sequential skip params with
        # requires_grad=False.  We therefore flip RG on ZO params *between* perturb calls,
        # not around them, so each perturb sees requires_grad=True on the params it should touch.

        zo_rg_saved = [p.requires_grad for p in self._zo_params]

        # ---- Phase A: +ε perturb on ZO body (RG still True) ----
        self._zo_perturb(z_seed, +1.0)

        # Disable autograd on ZO params for the FO forward+backward only.
        for p in self._zo_params:
            p.requires_grad_(False)

        try:
            ce_batch_loss = torch.zeros((), device=self.device)
            z_batch_loss: Optional[torch.Tensor] = (
                None if not self.cfg.softmax_auxiliary_loss else torch.zeros((), device=self.device)
            )
            loss_plus_total = torch.zeros((), device=self.device, dtype=torch.float32)

            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                grad_sync_context = nullcontext
                if (
                    self.cfg.distributed_strategy == DistributedStrategy.ddp
                    and self.cfg.ddp is not None
                    and self.cfg.ddp.grad_sync_mode == DDPGradSyncMode.batch
                    and micro_batch_idx != num_micro_batches - 1
                ):
                    grad_sync_context = self.dist_model.no_sync

                with grad_sync_context():
                    autocast_device = "mps" if self.device.type == "mps" else "cuda"
                    with torch.autocast(autocast_device, enabled=True, dtype=self.cfg.autocast_precision):
                        loss, ce_loss, z_loss = self.train_micro_batch(micro_batch, batch_size_in_tokens)
                        ce_batch_loss = ce_batch_loss + ce_loss.detach()
                        loss_plus_total = loss_plus_total + loss.detach().float()
                        if z_loss is not None:
                            assert z_batch_loss is not None
                            z_batch_loss = z_batch_loss + z_loss.detach()
                    loss.backward()

            if is_distributed():
                dist.all_reduce(loss_plus_total, op=dist.ReduceOp.SUM)
                loss_plus_total = loss_plus_total / get_world_size()
        finally:
            # Restore requires_grad before next perturb so _perturb sees ZO params again.
            for p, rg in zip(self._zo_params, zo_rg_saved):
                p.requires_grad_(rg)

        # ---- Phase B: -2ε perturb (now at -ε from θ), forward-only ----
        self._zo_perturb(z_seed, -2.0)

        loss_minus_total = torch.zeros((), device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            autocast_device = "mps" if self.device.type == "mps" else "cuda"
            with torch.autocast(autocast_device, enabled=True, dtype=self.cfg.autocast_precision):
                for micro_batch in micro_batches:
                    loss, _, _ = self.train_micro_batch(micro_batch, batch_size_in_tokens)
                    loss_minus_total = loss_minus_total + loss.float()

        if is_distributed():
            dist.all_reduce(loss_minus_total, op=dist.ReduceOp.SUM)
            loss_minus_total = loss_minus_total / get_world_size()

        # ---- Phase C: restore +ε on ZO body (back to θ) ----
        self._zo_perturb(z_seed, +1.0)

        # ---- Apply ZO update via low-level optimizer helpers ----
        lp = loss_plus_total.item()
        lm = loss_minus_total.item()
        scalar_half = (lp - lm) / 2.0  # (f⁺ − f⁻)/2; both ZoAdam and LDSDMuon expect this scaling.

        if isinstance(self.optim, ZoAdam):
            self.optim._apply_update([(z_seed, scalar_half)])
            self.optim._last_metrics["projected_grad_abs"] = abs(scalar_half)
        elif isinstance(self.optim, LDSDMuon):
            self.optim._apply_muon_update(z_seed, scalar_half)
            self.optim._last_metrics["projected_grad_abs"] = abs(scalar_half)
        else:  # pragma: no cover — guarded in __post_init__
            raise RuntimeError(f"Unsupported hybrid ZO optimizer: {type(self.optim).__name__}")

        # ---- FO update: clip + LR + step ----
        if self.cfg.max_grad_norm is not None and self.cfg.max_grad_norm > 0:
            fo_grad_norm = torch.nn.utils.clip_grad_norm_(self._fo_params, self.cfg.max_grad_norm)
            metrics["optim/fo/grad_norm"] = fo_grad_norm.item()

        fo_lr = self._fo_scheduler.get_lr(hf.learning_rate, self._fo_sched_current, self._fo_sched_max)
        for group in self._fo_optim.param_groups:
            group["lr"] = fo_lr
        self._fo_optim.step()

        # ---- Set LR for the main ZO optimizer (no .step needed — _apply_update already applied) ----
        zo_lr = self.scheduler.get_lr(
            self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
        )
        for group in self.optim.param_groups:
            group["lr"] = zo_lr

        # ---- Loss reduction for logging ----
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())

        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")

        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = z_batch_loss.item()

        metrics["optim/fo/lr"] = fo_lr
        metrics["optim/zo/lr"] = zo_lr
        metrics["optim/zo/loss_plus"] = lp
        metrics["optim/zo/loss_minus"] = lm

        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        if should_log_optim_metrics_this_step and hasattr(self.optim, "get_post_step_metrics"):
            optim_metrics = self.optim.get_post_step_metrics(
                self.dist_model, process_group=self.dist_model.process_group
            )
            for k, v in optim_metrics.items():
                metrics[f"optim/zo/{k}"] = v.item()

        if torch.cuda.is_available() and self.global_step % max(1, self.cfg.console_log_interval) == 0:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            metrics["train/peak_mem_mb"] = peak_mb
            torch.cuda.reset_peak_memory_stats()

        return metrics

    def _zo_perturb(self, z_seed: int, scaling_factor: float) -> None:
        """Dispatch to the main optimizer's _perturb helper (signature matches both)."""
        if isinstance(self.optim, ZoAdam):
            self.optim._perturb(z_seed, scaling_factor)
        elif isinstance(self.optim, LDSDMuon):
            self.optim._perturb_sequential(z_seed, scaling_factor)
        else:  # pragma: no cover — guarded in __post_init__
            raise RuntimeError(f"Unsupported hybrid ZO optimizer: {type(self.optim).__name__}")

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        if self._in_phase1:
            return self._train_step_phase1(batch, reduce_global_loss=reduce_global_loss)
        if self._fo_optim is not None:
            return self._train_step_hybrid(batch, reduce_global_loss=reduce_global_loss)
        if (
            isinstance(self.optim, ZoAdam)
            and self.cfg.zo_fo_direction is not None
            and self.cfg.zo_fo_direction.enabled
        ):
            return self._train_step_zo_fo_direction(batch, reduce_global_loss=reduce_global_loss)
        if isinstance(self.optim, ZeroOrderOptimizer):
            return self._train_step_zero_order(batch, reduce_global_loss=reduce_global_loss)

        metrics: Dict[str, float] = {}

        # Write data-indices to file.
        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss = self.train_batch(batch)

        # Collect loss, potentially reducing over all ranks.
        if reduce_global_loss:
            dist.reduce(ce_batch_loss, 0)
            ce_batch_loss.div_(get_world_size())
            if z_batch_loss is not None:
                dist.reduce(z_batch_loss, 0)
                z_batch_loss.div_(get_world_size())

        # Clip gradient norms and collect param/gradient/optim metrics.
        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        optim_metrics = self.optim.clip_grads_and_collect_metrics(
            self.global_step,
            collect_param_metrics=should_log_optim_metrics_this_step,
            # passing this process group here ensures metrics are reduced correctly when we're using
            # HYBRID sharding.
            process_group=self.dist_model.process_group,
        )

        # ZO divergence probe: measure cosine similarity between p.grad and ZO update directions.
        # Runs after backward (p.grad available) and before optimizer.step (params still at θ_t).
        if self._zo_probe is not None:
            if is_distributed():
                _probe_seed_local = int(np.random.randint(0, 1_000_000_000)) if get_global_rank() == 0 else 0
                _probe_seed = int(synchronize_value(_probe_seed_local, self.device))
            else:
                _probe_seed = int(np.random.randint(0, 1_000_000_000))

            _probe_micro_batches = self.split_batch(batch)
            _probe_batch_tokens = batch["input_ids"].numel()

            def _probe_loss_fn() -> torch.Tensor:
                total = torch.zeros((), device=self.device, dtype=torch.float32)
                _ac_dev = "mps" if self.device.type == "mps" else "cuda"
                with torch.inference_mode():
                    with torch.autocast(_ac_dev, enabled=True, dtype=self.cfg.autocast_precision):
                        for _mb in _probe_micro_batches:
                            _loss, _, _ = self.train_micro_batch(_mb, _probe_batch_tokens)
                            total += _loss.float()
                if is_distributed():
                    dist.all_reduce(total, op=dist.ReduceOp.SUM)
                    total /= get_world_size()
                return total

            _probe_metrics = self._zo_probe.maybe_compute(self.global_step, _probe_seed, _probe_loss_fn)
            for _k, _v in _probe_metrics.items():
                metrics[f"zo_probe/{_k}"] = _v

        # Adjust the learning rate.
        for group in self.optim.param_groups:
            # TODO (epwalsh): if we want to enable different LRs or gradient clipping settings per group
            # we should pass `group["initial_lr"]` or `group["initial_max_grad_norm"]` here instead of
            # the corresponding values from `self.cfg`.
            group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
            )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        if torch.isnan(ce_batch_loss):
            raise ValueError("nan loss encountered")
        if z_batch_loss is not None and torch.isnan(z_batch_loss):
            raise ValueError("nan loss encountered")
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = ce_batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/CrossEntropyLoss"] = self.cur_train_loss
        metrics["train/Perplexity"] = math.exp(self.cur_train_loss)
        if z_batch_loss is not None:
            metrics["train/ZLoss"] = z_batch_loss.item()

        # Maybe collect post-step optimizer-specific metrics.
        if should_log_optim_metrics_this_step:
            optim_metrics = self.optim.get_post_step_metrics(
                self.dist_model, process_group=self.dist_model.process_group
            )
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        return metrics

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            ce_loss, _, logits = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, ce_loss, logits
        )  # batch includes all keys that the downstream evaluation needs

        barrier()

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if name == "optim/total_grad_norm"
                    or not name.startswith("optim/")  # there's too many optimizer metrics
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.wandb is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = self.cfg.wandb.log_interval
        else:
            optim_log_interval = max(optim_log_interval, self.cfg.wandb.log_interval)
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        else:
            return False

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.dist_model.eval()

        eval_metrics = {}
        for evaluator in self.evaluators:
            log.info(f"Running evaluation for '{evaluator.label}'...")

            # Reset metrics.
            evaluator.reset_metrics()

            # Initialize data loader iterator.
            eval_batches = iter(evaluator.eval_loader)

            # Adjust how many batches to evaluate on.
            num_eval_batches = (
                evaluator.subset_num_batches
                if evaluator.subset_num_batches is not None
                else self.cfg.eval_subset_num_batches
            )
            if num_eval_batches > 0:
                num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
                eval_batches = islice(eval_batches, num_eval_batches)

            # Run model over batches.
            for eval_step, eval_batch in enumerate(eval_batches):
                self.eval_step(eval_batch, evaluator)

                # Log to console.
                if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.cfg.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

            # Get final metrics.
            metrics = evaluator.compute_metrics()
            eval_metrics.update(metrics)
            self.log_metrics_to_console(f"{evaluator.label}", metrics)

            del eval_batches

        # Eval compiles a bunch more versions, and the result is terrible. This way we get back to zero.
        if self.cfg.compile is not None:
            torch.compiler.reset()

        return eval_metrics

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = False
        cancel_reason: Optional[str] = None
        extra_steps = 0
        if get_global_rank() == 0:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"
                extra_steps = self.cfg.extra_steps_after_cancel
            elif (
                self.cfg.early_stopping_factor is not None
                and self.global_step > self.cfg.scheduler.t_warmup
                and self.cur_train_loss > self.cfg.early_stopping_factor * self.min_train_loss
            ):
                # Next check if early stopping loss criteria is met.
                should_cancel = True
                cancel_reason = "early stopping from loss increase"
            elif wandb.run is not None and (api_key := os.environ.get("WANDB_API_KEY")) is not None:
                # Finally, check if someone canceled the run from W&B by adding the 'cancel' / 'canceled' tag..
                # We won't see it in the run object. So we have to use the import/export API to check.
                from requests.exceptions import RequestException
                from wandb.errors import CommError

                try:
                    api = wandb.Api(api_key=api_key)
                    run = api.run(wandb.run.path)
                    for tag in run.tags or []:
                        if tag.lower() in {"cancel", "canceled", "cancelled"}:
                            should_cancel = True
                            cancel_reason = "Weights & Biases tag"
                            extra_steps = self.cfg.extra_steps_after_cancel
                            break
                except (RequestException, CommError):
                    log.info("Failed to check if W&B run is cancelled, continuing run.")

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            extra_steps = synchronize_value(extra_steps, self.device)
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")

        return run_canceled, extra_steps

    def fit(self):
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)
        if self.cfg.stop_at is None:
            self.cfg.stop_at = self.max_steps + 10

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

        # Set model to 'train' mode.
        self.dist_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)

        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(
                    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
                )
                if self.cfg.remote_save_folder is not None:
                    upload_folder = f"{self.cfg.remote_save_folder.rstrip('/')}/profiler"
                    log.info(f"Tracing complete, uploading results to '{upload_folder}'...")
                    upload(trace_path, f"{upload_folder}/{trace_path.name}")

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: int = self.cfg.stop_at
        save_checkpoints: bool = True

        with torch_profiler as p:
            for epoch in range(self.epoch or 0, self.max_epochs):
                for batch in self.train_loader:
                    # Bookkeeping.
                    # NOTE: To track the global batch size / number of tokens per batch we make the assumption that all
                    # batches see the same number of tokens, which should be the case for language model pre-training
                    # (at least when drop_last=True).
                    # Alternatively we'd have to use a distributed all reduce over seq_len here, but I don't want that
                    # overhead. So for now I'm putting these assertions here so if the assumption is violated it will
                    # fail loudly.
                    batch_size, seq_len = batch["input_ids"].shape
                    assert seq_len == self.cfg.model.max_sequence_length
                    assert batch_size == self.cfg.device_train_batch_size
                    global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
                    self.global_step += 1
                    self.global_train_examples_seen_this_epoch += global_batch_size
                    self.global_train_tokens_seen += global_batch_size * seq_len
                    speed_monitor.batch_start(
                        global_total_tokens=self.global_train_tokens_seen,
                        device_batch_num_tokens=batch_size * seq_len,  # num tokens in batch for this device
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        num_fwd_flops=self.model.num_fwd_flops,  # this is per token
                        num_bck_flops=self.model.num_bck_flops,  # this is per token
                        record=not first_batch,
                    )

                    should_log_this_step = self.should_log_this_step()

                    # Run train step on batch.
                    metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)

                    # Maybe collect other metrics.
                    if should_log_this_step:
                        # Speed metrics.
                        metrics.update(speed_monitor.check())
                        # System metrics.
                        metrics.update(self.system_metrics())
                        # Learning rate metrics.
                        metrics.update(lr_monitor.check())

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        if get_global_rank() == 0:
                            self.log_metrics_to_console(
                                f"[step={self.global_step}/{self.max_steps},epoch={epoch}]",
                                metrics,
                            )
                        else:
                            log.info(f"[step={self.global_step}/{self.max_steps},epoch={epoch}]")

                    # Log metrics to W&B.
                    if (
                        wandb.run is not None
                        and self.cfg.wandb is not None
                        and self.global_step % self.cfg.wandb.log_interval == 0
                    ):
                        wandb.log(metrics, step=self.global_step)

                    # Check if/when run should be canceled.
                    if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                        cancel_initiated, extra_steps = self.check_if_cancelled()
                        if cancel_initiated:
                            stop_at = min(stop_at, self.global_step + extra_steps)

                    # Maybe save sharded checkpoint.
                    if self.cfg.distributed_strategy == DistributedStrategy.fsdp:
                        if save_checkpoints and (
                            cancel_initiated
                            or (
                                self.cfg.save_interval is not None
                                and self.global_step % self.cfg.save_interval == 0
                                and self.cfg.save_num_checkpoints_to_keep != 0
                            )
                        ):
                            log.info("Saving checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Remove any ephemeral checkpoints.
                            while self.ephemeral_checkpoints:
                                self.remove_ephemeral_checkpoint()

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                            # If the run was just canceled this will be the final checkpoint.
                            if cancel_initiated:
                                save_checkpoints = False
                        elif (
                            self.cfg.save_interval_ephemeral is not None
                            and self.global_step % self.cfg.save_interval_ephemeral == 0
                        ):
                            log.info("Saving ephemeral checkpoint...")
                            checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded_ephemeral)
                            log.info(f"Checkpoint saved to {checkpoint_path}")

                            # Reset speed monitor so that we don't count the time taken to save checkpoints.
                            speed_monitor.reset()

                    # Maybe save unsharded checkpoint.
                    # This code snippet should always execute when running DDP.
                    if (
                        save_checkpoints
                        and self.cfg.save_interval_unsharded is not None
                        and self.global_step % self.cfg.save_interval_unsharded == 0
                        and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
                    ):
                        log.info("Saving unsharded checkpoint...")
                        checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe run evaluations.
                    if not cancel_initiated and (
                        self.global_step % self.cfg.eval_interval == 0 or self.global_step >= stop_at
                    ):
                        eval_metrics = self.eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.dist_model.train()

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if self.global_step >= stop_at:
                        break

                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    self.dataset.start_index = 0
                    if self.epoch < self.max_epochs:
                        log.info(f"Reshuffling data loader for epoch {self.epoch}...")
                        self.dataset.reshuffle(self.epoch)
                    continue

                break

        # Save final checkpoint.
        if save_checkpoints:
            if (
                self.cfg.save_interval_unsharded is not None
                and self.last_unsharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final unsharded model checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.unsharded)
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")
            elif (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
                and self.cfg.distributed_strategy == DistributedStrategy.fsdp
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path, _ = self.save_checkpoint(CheckpointType.sharded)
                log.info(f"Checkpoint saved to {checkpoint_path}")

    def close(self, exit_code: int = 0) -> None:
        gc_cuda()

        if self.indices_file is not None:
            self.indices_file.flush()
            self.indices_file.close()
        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code, quiet=True)

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)
