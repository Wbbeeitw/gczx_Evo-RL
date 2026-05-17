#!/usr/bin/env python

"""
Standalone tactile-prompt training entrypoint.

This file does not modify the original LeRobot/Evo-RL training code. It creates a
separate training command that approximates the thesis training style under a
simplified assumption:

1. The dataset already contains four tactile image streams.
2. We do not add new model-side tactile alignment logic.
3. We only enforce a fixed seven-image order and inject tactile semantics into
   the task prompt so the policy sees an instruction-following style prompt.

Assumed dataset image keys:
    - observation.images.left_top
    - observation.images.left_wrist
    - observation.images.right_wrist
    - observation.images.tactile_left_outer
    - observation.images.tactile_left_inner
    - observation.images.tactile_right_outer
    - observation.images.tactile_right_inner
"""

import dataclasses
import logging
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from accelerate import Accelerator
import torch
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.acp_dataset_stats import compute_acp_indicator_stats
from lerobot.rl.acp_hook import build_acp_raw_batch_hook
from lerobot.rl.wandb_utils import make_logger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging

DEFAULT_ORDERED_IMAGE_KEYS = [
    "observation.images.left_top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.tactile_left_outer",
    "observation.images.tactile_left_inner",
    "observation.images.tactile_right_outer",
    "observation.images.tactile_right_inner",
]

DEFAULT_SENSOR_TEXT = {
    "observation.images.left_top": "Global RGB camera: observe the whole scene and object layout.",
    "observation.images.left_wrist": "Left wrist RGB camera: observe the local left gripper view.",
    "observation.images.right_wrist": "Right wrist RGB camera: observe the local right gripper view.",
    "observation.images.tactile_left_outer": (
        "Left gripper outer tactile heatmap: describe contact on the outer surface of the left gripper."
    ),
    "observation.images.tactile_left_inner": (
        "Left gripper inner tactile heatmap: describe contact on the inner grasping surface of the left gripper."
    ),
    "observation.images.tactile_right_outer": (
        "Right gripper outer tactile heatmap: describe contact on the outer surface of the right gripper."
    ),
    "observation.images.tactile_right_inner": (
        "Right gripper inner tactile heatmap: describe contact on the inner grasping surface of the right gripper."
    ),
}


@dataclass
class TactilePromptConfig:
    enable: bool = True
    task_field: str = "task"
    strict_image_keys: bool = True
    ordered_image_keys: str = ",".join(DEFAULT_ORDERED_IMAGE_KEYS)
    prompt_prefix: str = (
        "You control a bimanual robot from multi-view RGB observations, tactile heatmaps, and robot state."
    )
    prompt_suffix: str = "Generate the next action chunk that follows the instruction."

    def validate(self) -> None:
        if not self.task_field:
            raise ValueError("'tactile_prompt.task_field' must be non-empty.")
        if len(self.image_keys()) == 0:
            raise ValueError("'tactile_prompt.ordered_image_keys' must contain at least one key.")

    def image_keys(self) -> list[str]:
        return [key.strip() for key in self.ordered_image_keys.split(",") if key.strip()]


@dataclass
class ThesisPresetConfig:
    enable: bool = True
    batch_size: int = 1
    steps: int = 10_000
    log_freq: int = 100
    save_freq: int = 1_000
    chunk_size: int = 30
    n_action_steps: int = 30
    max_state_dim: int = 14
    max_action_dim: int = 14
    optimizer_lr: float = 5e-5
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 6_000
    scheduler_decay_lr: float = 5e-6
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_visual_projector: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("'thesis_preset.batch_size' must be > 0.")
        if self.steps <= 0:
            raise ValueError("'thesis_preset.steps' must be > 0.")
        if self.chunk_size <= 0 or self.n_action_steps <= 0:
            raise ValueError("'thesis_preset.chunk_size' and 'thesis_preset.n_action_steps' must be > 0.")
        if self.max_state_dim <= 0 or self.max_action_dim <= 0:
            raise ValueError("'thesis_preset.max_state_dim' and 'thesis_preset.max_action_dim' must be > 0.")
        if self.optimizer_lr <= 0:
            raise ValueError("'thesis_preset.optimizer_lr' must be > 0.")

    def apply(self, cfg: "TactileTrainPipelineConfig") -> None:
        if not self.enable:
            return

        cfg.batch_size = self.batch_size
        cfg.steps = self.steps
        cfg.log_freq = self.log_freq
        cfg.save_freq = self.save_freq

        if cfg.policy is None:
            raise ValueError("Policy must be configured before applying thesis presets.")
        if cfg.policy.type != "pi05":
            raise ValueError("This standalone tactile training entrypoint currently targets --policy.type=pi05.")

        policy_attrs: dict[str, Any] = {
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "optimizer_lr": self.optimizer_lr,
            "scheduler_warmup_steps": self.scheduler_warmup_steps,
            "scheduler_decay_steps": self.scheduler_decay_steps,
            "scheduler_decay_lr": self.scheduler_decay_lr,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "train_expert_only": self.train_expert_only,
        }
        for attr_name, attr_value in policy_attrs.items():
            if hasattr(cfg.policy, attr_name):
                setattr(cfg.policy, attr_name, attr_value)


@dataclass
class TactileTrainPipelineConfig(TrainPipelineConfig):
    tactile_prompt: TactilePromptConfig = field(default_factory=TactilePromptConfig)
    thesis_preset: ThesisPresetConfig = field(default_factory=ThesisPresetConfig)

    def validate(self) -> None:
        super().validate()
        self.tactile_prompt.validate()
        self.thesis_preset.validate()
        self.thesis_preset.apply(self)

        if self.use_policy_training_preset and not self.resume and self.policy is not None:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()


def _label_for_image_key(image_key: str) -> str:
    if image_key in DEFAULT_SENSOR_TEXT:
        return DEFAULT_SENSOR_TEXT[image_key]

    suffix = image_key.split(".")[-1].replace("_", " ")
    return f"{suffix.title()}: auxiliary observation stream available for action prediction."


def build_tactile_prompt(task: str, image_keys: list[str], cfg: TactilePromptConfig) -> str:
    parts: list[str] = []
    if cfg.prompt_prefix.strip():
        parts.append(cfg.prompt_prefix.strip())

    for image_key in image_keys:
        parts.append(_label_for_image_key(image_key))

    parts.append(f"Instruction: {task.strip()}")

    if cfg.prompt_suffix.strip():
        parts.append(cfg.prompt_suffix.strip())

    return " ".join(part for part in parts if part)


class TactilePromptHook:
    def __init__(self, cfg: TactilePromptConfig):
        self.cfg = cfg
        self.image_keys = cfg.image_keys()

    def __call__(self, batch: Any, _: int) -> Any:
        if not isinstance(batch, dict):
            raise TypeError(f"Tactile prompt batch must be dict, got {type(batch).__name__}.")

        task_field = self.cfg.task_field
        if task_field not in batch:
            raise KeyError(f"Tactile prompt requires '{task_field}' in batch.")

        tasks = batch[task_field]
        if not isinstance(tasks, list) or any(not isinstance(task, str) for task in tasks):
            raise TypeError(f"Tactile prompt batch['{task_field}'] must be list[str].")

        present_image_keys = [image_key for image_key in self.image_keys if image_key in batch]
        missing_image_keys = [image_key for image_key in self.image_keys if image_key not in batch]

        if self.cfg.strict_image_keys and missing_image_keys:
            raise KeyError(
                "Missing expected tactile/image keys for tactile prompt hook: "
                f"{missing_image_keys}. Present keys: {list(batch.keys())}"
            )

        batch[task_field] = [
            build_tactile_prompt(task=task, image_keys=present_image_keys, cfg=self.cfg) for task in tasks
        ]
        return batch


def build_tactile_prompt_hook(cfg: TactilePromptConfig) -> TactilePromptHook | None:
    if not cfg.enable:
        return None
    return TactilePromptHook(cfg)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()

    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    with accelerator.autocast():
        if rabc_batch_weights is not None:
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if hasattr(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def configure_policy_features_for_tactile_dataset(
    cfg: TactileTrainPipelineConfig,
    dataset,
) -> None:
    if cfg.policy is None:
        raise ValueError("Policy must be configured before feature ordering.")

    all_features = dataset_to_policy_features(dataset.meta.features)
    ordered_image_keys = cfg.tactile_prompt.image_keys()

    output_features = {key: ft for key, ft in all_features.items() if ft.type is FeatureType.ACTION}
    ordered_input_features: dict[str, Any] = {}

    missing_requested_image_keys = []
    for image_key in ordered_image_keys:
        feature = all_features.get(image_key)
        if feature is None:
            missing_requested_image_keys.append(image_key)
            continue
        if feature.type is not FeatureType.VISUAL:
            raise ValueError(f"Expected a visual feature for '{image_key}', got type={feature.type}.")
        ordered_input_features[image_key] = feature

    if cfg.tactile_prompt.strict_image_keys and missing_requested_image_keys:
        raise ValueError(
            "Dataset is missing required tactile/image keys: "
            f"{missing_requested_image_keys}. Available image keys: {list(dataset.meta.camera_keys)}"
        )

    for key, feature in all_features.items():
        if key in output_features or key in ordered_input_features:
            continue
        ordered_input_features[key] = feature

    cfg.policy.input_features = ordered_input_features
    cfg.policy.output_features = output_features


def _resolve_pi05_visual_projector(policy: PreTrainedPolicy):
    pi05_model = getattr(policy, "model", None)
    if pi05_model is None:
        return None

    paligemma_with_expert = getattr(pi05_model, "paligemma_with_expert", None)
    if paligemma_with_expert is None:
        return None

    paligemma = getattr(paligemma_with_expert, "paligemma", None)
    if paligemma is None:
        return None

    paligemma_model = getattr(paligemma, "model", None)
    if paligemma_model is not None and hasattr(paligemma_model, "multi_modal_projector"):
        return paligemma_model.multi_modal_projector

    if hasattr(paligemma, "multi_modal_projector"):
        return paligemma.multi_modal_projector

    return None


def _resolve_pi05_action_expert(policy: PreTrainedPolicy):
    pi05_model = getattr(policy, "model", None)
    if pi05_model is None:
        return None

    paligemma_with_expert = getattr(pi05_model, "paligemma_with_expert", None)
    if paligemma_with_expert is None:
        return None

    return getattr(paligemma_with_expert, "gemma_expert", None)


def configure_trainable_parameters_for_tactile_finetune(
    cfg: TactileTrainPipelineConfig,
    policy: PreTrainedPolicy,
) -> None:
    if cfg.policy is None or cfg.policy.type != "pi05":
        return

    if not cfg.thesis_preset.enable:
        return

    if not cfg.thesis_preset.train_expert_only:
        return

    if not cfg.thesis_preset.train_visual_projector:
        return

    visual_projector = _resolve_pi05_visual_projector(policy)
    if visual_projector is None:
        raise ValueError(
            "Unable to locate the PI05 visual projector. Expected PaliGemma multi_modal_projector to exist."
        )
    action_expert = _resolve_pi05_action_expert(policy)
    if action_expert is None:
        raise ValueError(
            "Unable to locate the PI05 action expert. Expected PaliGemma-with-expert gemma_expert to exist."
        )

    for param in policy.parameters():
        param.requires_grad = False

    for param in action_expert.parameters():
        param.requires_grad = True

    for param in visual_projector.parameters():
        param.requires_grad = True


def validate_tactile_finetune_parameter_subset(
    cfg: TactileTrainPipelineConfig,
    policy: PreTrainedPolicy,
) -> None:
    if cfg.policy is None or cfg.policy.type != "pi05":
        return

    if not cfg.thesis_preset.enable or not cfg.thesis_preset.train_expert_only:
        return

    allowed_substrings: list[str] = [
        ".paligemma_with_expert.gemma_expert.",
    ]
    if cfg.thesis_preset.train_visual_projector:
        allowed_substrings.extend(
            [
                ".paligemma_with_expert.paligemma.model.multi_modal_projector.",
                ".paligemma_with_expert.paligemma.multi_modal_projector.",
            ]
        )

    unexpected_trainable = []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if not any(token in name for token in allowed_substrings):
            unexpected_trainable.append(name)

    if unexpected_trainable:
        preview = unexpected_trainable[:20]
        raise ValueError(
            "Unexpected trainable parameters remain after tactile finetune filtering: "
            f"{preview}{' ...' if len(unexpected_trainable) > len(preview) else ''}"
        )


def log_tactile_finetune_parameter_subset(policy: PreTrainedPolicy) -> None:
    bucket_rules = {
        "gemma_expert": ".paligemma_with_expert.gemma_expert.",
        "visual_projector": ".paligemma_with_expert.paligemma.model.multi_modal_projector.",
        "visual_projector_alt": ".paligemma_with_expert.paligemma.multi_modal_projector.",
    }

    summary: dict[str, int] = {}
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue

        bucket = "other"
        for bucket_name, token in bucket_rules.items():
            if token in name:
                bucket = "visual_projector" if bucket_name == "visual_projector_alt" else bucket_name
                break

        summary[bucket] = summary.get(bucket, 0) + param.numel()

    logging.info("Tactile finetune trainable parameter buckets: %s", summary)


@parser.wrap()
def train(
    cfg: TactileTrainPipelineConfig,
    accelerator: "Accelerator | None" = None,
):
    cfg.validate()
    acp_raw_batch_hook = build_acp_raw_batch_hook(cfg.acp, cfg.seed)
    tactile_prompt_hook = build_tactile_prompt_hook(cfg.tactile_prompt)

    if accelerator is None:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    wandb_logger = make_logger(cfg) if is_main_process else None
    if wandb_logger is None and is_main_process:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        configure_policy_features_for_tactile_dataset(cfg, dataset)

        if cfg.acp.enable:
            indicator_stats = compute_acp_indicator_stats(dataset, cfg.acp.indicator_field)
            if indicator_stats is None:
                logging.warning(
                    "ACP is enabled but indicator statistics are unavailable for field '%s'.",
                    cfg.acp.indicator_field,
                )
            else:
                if indicator_stats.total_count >= 0:
                    logging.info(
                        "ACP indicator stats (%s): field='%s' ratio=%.6f positive=%d total=%d",
                        indicator_stats.source,
                        indicator_stats.indicator_field,
                        indicator_stats.positive_ratio,
                        indicator_stats.positive_count,
                        indicator_stats.total_count,
                    )
                else:
                    logging.info(
                        "ACP indicator stats (%s): field='%s' ratio=%.6f",
                        indicator_stats.source,
                        indicator_stats.indicator_field,
                        indicator_stats.positive_ratio,
                    )
                if indicator_stats.invalid_count > 0:
                    logging.warning(
                        "ACP indicator field '%s' contains %d non-binary values (expected only 0/1).",
                        cfg.acp.indicator_field,
                        indicator_stats.invalid_count,
                    )

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset(cfg)
        configure_policy_features_for_tactile_dataset(cfg, dataset)

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Policy image order: %s", list(cfg.policy.image_features.keys()))
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    configure_trainable_parameters_for_tactile_finetune(cfg, policy)
    validate_tactile_finetune_parameter_subset(cfg, policy)

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    accelerator.wait_for_everyone()

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        log_tactile_finetune_parameter_subset(policy)
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start offline tactile-prompt training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    logged_first_prompt = False
    prompt_keys: list[str] = []
    policy_task_field = getattr(cfg.policy, "task_field", None)
    if isinstance(policy_task_field, str) and policy_task_field:
        prompt_keys.append(policy_task_field)
    for key in ("task", "subtask"):
        if key not in prompt_keys:
            prompt_keys.append(key)

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        if acp_raw_batch_hook is not None:
            batch = acp_raw_batch_hook(batch, step)
        if tactile_prompt_hook is not None:
            batch = tactile_prompt_hook(batch, step)
        batch = preprocessor(batch)

        if is_main_process and not logged_first_prompt:
            for key in prompt_keys:
                if key not in batch:
                    continue
                prompt_batch = batch[key]
                first_prompt = None
                if isinstance(prompt_batch, str):
                    first_prompt = prompt_batch
                elif isinstance(prompt_batch, (list, tuple)) and len(prompt_batch) > 0:
                    first_item = prompt_batch[0]
                    if isinstance(first_item, str):
                        first_prompt = first_item
                if first_prompt is not None:
                    logging.info("First policy prompt (%s):\n%s", key, first_prompt)
                    logged_first_prompt = True
                    break
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                aggregated = eval_info["overall"]

                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("avg_rwd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main() -> None:
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
