#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


INFO_PATH = Path("meta/info.json")


EPISODE_SUCCESS = "success"
EPISODE_FAILURE = "failure"


@dataclass
class EpisodeInfo:
    episode_index: int
    task_name: str
    length: int
    success: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute advantage and ACP indicator from an existing frame-level value field."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the LeRobot dataset root.")
    parser.add_argument(
        "--value-field",
        type=str,
        default="complementary_info.value",
        help="Existing value field to read.",
    )
    parser.add_argument(
        "--advantage-field",
        type=str,
        default="complementary_info.advantage",
        help="Advantage field to overwrite/write.",
    )
    parser.add_argument(
        "--indicator-field",
        type=str,
        default="complementary_info.acp_indicator",
        help="Indicator field to overwrite/write.",
    )
    parser.add_argument(
        "--intervention-field",
        type=str,
        default="complementary_info.is_intervention",
        help="Human intervention field. Frames with values > 0.5 can be forced positive.",
    )
    parser.add_argument("--n-step", type=int, default=50, help="N-step horizon for advantage computation.")
    parser.add_argument(
        "--positive-ratio",
        type=float,
        default=0.3,
        help="Per-task positive ratio used to binarize advantages into indicators.",
    )
    parser.add_argument(
        "--c-fail-coef",
        type=float,
        default=1.0,
        help="Failure penalty coefficient used in normalized value targets.",
    )
    parser.add_argument(
        "--success-field",
        type=str,
        default="episode_success",
        help="Episode-level success label field in meta/episodes parquet.",
    )
    parser.add_argument(
        "--default-success",
        type=str,
        default="failure",
        help="Fallback success label when the episode metadata lacks an explicit label.",
    )
    parser.add_argument(
        "--force-intervention-positive",
        action="store_true",
        default=True,
        help="Force intervention-marked frames to indicator=1. Enabled by default.",
    )
    parser.add_argument(
        "--no-force-intervention-positive",
        dest="force_intervention_positive",
        action="store_false",
        help="Do not force intervention-marked frames to indicator=1.",
    )
    return parser.parse_args()


def normalize_episode_success_label(label: str | None) -> str | None:
    if label is None:
        return None
    normalized = str(label).strip().lower()
    if normalized not in {EPISODE_SUCCESS, EPISODE_FAILURE}:
        raise ValueError(
            f"`episode_success` must be one of {[EPISODE_FAILURE, EPISODE_SUCCESS]}, got '{label}'."
        )
    return normalized


def resolve_episode_success_label(
    explicit_label: str | None,
    default_label: str | None = None,
    require_label: bool = False,
) -> str | None:
    explicit = normalize_episode_success_label(explicit_label)
    if explicit is not None:
        return explicit

    default = normalize_episode_success_label(default_label)
    if default is not None:
        return default

    if require_label:
        raise ValueError("Missing episode_success label and no valid default was provided.")
    return None


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_episode_metadata(dataset_root: Path) -> pd.DataFrame:
    episodes_dir = dataset_root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {episodes_dir}")
    df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    if "episode_index" not in df.columns:
        raise KeyError("Episode metadata is missing 'episode_index'.")
    return df.sort_values("episode_index").reset_index(drop=True)


def load_frame_core(dataset_root: Path, value_field: str, intervention_field: str) -> tuple[pd.DataFrame, list[Path]]:
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No frame parquet files found under {data_dir}")

    frames = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        required = {"index", "episode_index", "frame_index", value_field}
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required frame columns in {path}: {missing}")
        keep_cols = ["index", "episode_index", "frame_index", value_field]
        if intervention_field in df.columns:
            keep_cols.append(intervention_field)
        frames.append(df[keep_cols].copy())

    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values("index").reset_index(drop=True), parquet_files


def build_episode_info(episodes: pd.DataFrame, success_field: str, default_success: str) -> tuple[dict[int, EpisodeInfo], dict[str, int]]:
    has_success = success_field in episodes.columns
    episode_info: dict[int, EpisodeInfo] = {}
    task_max_length: dict[str, int] = {}

    for _, row in episodes.iterrows():
        ep_idx = int(row["episode_index"])
        ep_length = int(row["length"])
        tasks = row["tasks"]
        task_name = tasks[0] if isinstance(tasks, list) else str(tasks)
        explicit_success = row[success_field] if has_success else None
        resolved_success = resolve_episode_success_label(
            explicit_success,
            default_label=default_success,
            require_label=True,
        )
        success = resolved_success == EPISODE_SUCCESS

        episode_info[ep_idx] = EpisodeInfo(
            episode_index=ep_idx,
            task_name=task_name,
            length=ep_length,
            success=success,
        )
        task_max_length[task_name] = max(task_max_length.get(task_name, 0), ep_length)

    return episode_info, task_max_length


def compute_normalized_value_targets(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_info: dict[int, EpisodeInfo],
    task_max_lengths: dict[str, int],
    c_fail_coef: float,
    *,
    clip_min: float = -1.0,
    clip_max: float = 0.0,
) -> np.ndarray:
    if episode_indices.shape != frame_indices.shape:
        raise ValueError("episode_indices and frame_indices must have the same shape.")
    if c_fail_coef < 0:
        raise ValueError("'c_fail_coef' must be non-negative.")

    targets = np.zeros(episode_indices.shape[0], dtype=np.float32)
    for i in range(episode_indices.shape[0]):
        ep_idx = int(episode_indices[i])
        if ep_idx not in episode_info:
            raise KeyError(f"Missing episode metadata for episode_index={ep_idx}.")
        ep = episode_info[ep_idx]
        task_max = task_max_lengths.get(ep.task_name)
        if task_max is None:
            raise KeyError(f"Missing task max length for task '{ep.task_name}'.")
        if task_max <= 0:
            raise ValueError(f"Invalid task max length {task_max} for task '{ep.task_name}'.")

        remaining_steps = ep.length - int(frame_indices[i]) - 1
        c_fail = float(task_max) * c_fail_coef
        g = -float(remaining_steps)
        if not ep.success:
            g -= c_fail

        denom = float(task_max) + c_fail
        g_norm = g / denom
        targets[i] = np.clip(g_norm, clip_min, clip_max)

    return targets


def compute_dense_rewards_from_targets(
    targets: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    rewards = np.zeros_like(targets, dtype=np.float32)
    n = targets.shape[0]

    for i in range(n):
        is_next_in_episode = (
            i + 1 < n
            and episode_indices[i + 1] == episode_indices[i]
            and frame_indices[i + 1] == frame_indices[i] + 1
        )
        if is_next_in_episode:
            rewards[i] = float(targets[i] - targets[i + 1])
        else:
            rewards[i] = float(targets[i])

    return rewards


def compute_n_step_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> np.ndarray:
    if n_step <= 0:
        raise ValueError("'n_step' must be > 0.")

    n = rewards.shape[0]
    advantages = np.zeros(n, dtype=np.float32)

    for i in range(n):
        ep_i = episode_indices[i]
        fi = frame_indices[i]

        discounted_sum = 0.0
        j = i
        steps = 0
        while steps < n_step and j < n:
            same_episode = episode_indices[j] == ep_i
            contiguous = frame_indices[j] == fi + steps
            if not same_episode or not contiguous:
                break

            discounted_sum += float(rewards[j])
            steps += 1
            j += 1

        if steps == n_step and j < n and episode_indices[j] == ep_i and frame_indices[j] == fi + n_step:
            bootstrap = float(values[j])
        else:
            bootstrap = 0.0

        advantages[i] = float(discounted_sum + bootstrap - values[i])

    return advantages


def compute_task_thresholds(task_ids: np.ndarray, advantages: np.ndarray, positive_ratio: float) -> dict[int, float]:
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("'positive_ratio' must be within [0, 1].")

    thresholds: dict[int, float] = {}
    quantile = 1.0 - positive_ratio
    for task_id in np.unique(task_ids):
        task_adv = advantages[task_ids == task_id]
        if task_adv.size == 0:
            thresholds[int(task_id)] = float("inf")
        else:
            thresholds[int(task_id)] = float(np.quantile(task_adv, quantile))
    return thresholds


def binarize_advantages(
    task_ids: np.ndarray,
    advantages: np.ndarray,
    thresholds: dict[int, float],
    interventions: np.ndarray,
    force_intervention_positive: bool,
) -> np.ndarray:
    indicators = np.zeros_like(advantages, dtype=np.int64)
    for i in range(advantages.shape[0]):
        task_id = int(task_ids[i])
        indicators[i] = 1 if float(advantages[i]) >= thresholds[task_id] else 0

    if force_intervention_positive:
        intervention_mask = interventions.astype(np.float32) > 0.5
        indicators[intervention_mask] = 1

    return indicators


def update_feature_metadata(dataset_root: Path, feature_infos: dict[str, dict[str, Any]]) -> None:
    info_path = dataset_root / INFO_PATH
    info = load_json(info_path)
    features = info.setdefault("features", {})
    for feature_name, feature_info in feature_infos.items():
        features[feature_name] = {
            "dtype": feature_info["dtype"],
            "shape": list(feature_info["shape"]),
            "names": feature_info.get("names"),
        }
    write_json(info, info_path)


def write_columns_in_place(
    dataset_root: Path,
    absolute_indices: np.ndarray,
    columns: dict[str, np.ndarray],
    feature_infos: dict[str, dict[str, Any]],
) -> None:
    if absolute_indices.ndim != 1:
        raise ValueError("'absolute_indices' must be rank-1.")

    max_index = int(np.max(absolute_indices))
    selected = np.zeros(max_index + 1, dtype=np.bool_)
    selected[absolute_indices] = True

    lookups: dict[str, np.ndarray] = {}
    for field, values in columns.items():
        field_dtype = feature_infos[field]["dtype"]
        lookup_dtype = np.float32 if field_dtype == "float32" else np.int64
        lookup = np.zeros(max_index + 1, dtype=lookup_dtype)
        if values.shape[0] != absolute_indices.shape[0]:
            raise ValueError(
                f"Column '{field}' length mismatch: expected {absolute_indices.shape[0]}, got {values.shape[0]}."
            )
        lookup[absolute_indices] = values.astype(lookup_dtype, copy=False)
        lookups[field] = lookup

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_root / 'data'}")

    for parquet_path in data_files:
        table = pq.read_table(parquet_path)
        idx_np = table["index"].to_numpy().astype(np.int64, copy=False)

        in_range = (idx_np >= 0) & (idx_np <= max_index)
        in_subset = np.zeros_like(in_range)
        in_subset[in_range] = selected[idx_np[in_range]]

        new_table = table
        for field, lookup in lookups.items():
            field_dtype = feature_infos[field]["dtype"]
            if field_dtype == "float32":
                default_value = np.nan
                target_dtype = np.float32
                pa_type = pa.float32()
            elif field_dtype == "int64":
                default_value = 0
                target_dtype = np.int64
                pa_type = pa.int64()
            else:
                raise ValueError(f"Unsupported annotation dtype '{field_dtype}' for field '{field}'.")

            if field in new_table.schema.names:
                current = new_table[field].to_numpy().astype(target_dtype, copy=True)
            else:
                current = np.full(idx_np.shape[0], default_value, dtype=target_dtype)

            if np.any(in_subset):
                current[in_subset] = lookup[idx_np[in_subset]]

            array = pa.array(current, type=pa_type)
            if field in new_table.schema.names:
                col_idx = new_table.schema.names.index(field)
                new_table = new_table.set_column(col_idx, field, array)
            else:
                new_table = new_table.append_column(field, array)

        pq.write_table(new_table, parquet_path, compression="snappy")

    update_feature_metadata(dataset_root=dataset_root, feature_infos=feature_infos)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    frames, _ = load_frame_core(dataset_root, args.value_field, args.intervention_field)
    episodes = load_episode_metadata(dataset_root)
    episode_info, task_max_lengths = build_episode_info(
        episodes=episodes,
        success_field=args.success_field,
        default_success=args.default_success,
    )

    values = frames[args.value_field].to_numpy(dtype=np.float32, copy=False)
    absolute_indices = frames["index"].to_numpy(dtype=np.int64, copy=False)
    episode_indices = frames["episode_index"].to_numpy(dtype=np.int64, copy=False)
    frame_indices = frames["frame_index"].to_numpy(dtype=np.int64, copy=False)

    if args.intervention_field in frames.columns:
        interventions = frames[args.intervention_field].to_numpy(dtype=np.float32, copy=False)
    else:
        interventions = np.zeros(len(frames), dtype=np.float32)

    task_name_to_id: dict[str, int] = {}
    task_ids = np.zeros(len(frames), dtype=np.int64)
    for i, ep_idx in enumerate(episode_indices):
        task_name = episode_info[int(ep_idx)].task_name
        if task_name not in task_name_to_id:
            task_name_to_id[task_name] = len(task_name_to_id)
        task_ids[i] = task_name_to_id[task_name]

    value_targets = compute_normalized_value_targets(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_lengths,
        c_fail_coef=float(args.c_fail_coef),
    )
    rewards = compute_dense_rewards_from_targets(
        targets=value_targets,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
    )
    advantages = compute_n_step_advantages(
        rewards=rewards,
        values=values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        n_step=int(args.n_step),
    )
    thresholds = compute_task_thresholds(
        task_ids=task_ids,
        advantages=advantages,
        positive_ratio=float(args.positive_ratio),
    )
    indicators = binarize_advantages(
        task_ids=task_ids,
        advantages=advantages,
        thresholds=thresholds,
        interventions=interventions,
        force_intervention_positive=bool(args.force_intervention_positive),
    )

    columns = {
        args.advantage_field: advantages.astype(np.float32),
        args.indicator_field: indicators.astype(np.int64),
    }
    feature_infos = {
        args.advantage_field: {"dtype": "float32", "shape": (1,), "names": None},
        args.indicator_field: {"dtype": "int64", "shape": (1,), "names": None},
    }
    write_columns_in_place(
        dataset_root=dataset_root,
        absolute_indices=absolute_indices,
        columns=columns,
        feature_infos=feature_infos,
    )

    payload = {
        "dataset_root": str(dataset_root),
        "value_field": args.value_field,
        "advantage_field": args.advantage_field,
        "indicator_field": args.indicator_field,
        "intervention_field": args.intervention_field,
        "force_intervention_positive": bool(args.force_intervention_positive),
        "n_step": int(args.n_step),
        "positive_ratio_target": float(args.positive_ratio),
        "positive_ratio_observed": float(np.mean(indicators.astype(np.float32))),
        "num_rows": int(len(frames)),
        "num_intervention_rows": int(np.sum(interventions > 0.5)),
        "value_min": float(np.min(values)),
        "value_max": float(np.max(values)),
        "value_mean": float(np.mean(values)),
        "advantage_min": float(np.min(advantages)),
        "advantage_max": float(np.max(advantages)),
        "advantage_mean": float(np.mean(advantages)),
        "thresholds_by_task_id": {str(k): float(v) for k, v in thresholds.items()},
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
