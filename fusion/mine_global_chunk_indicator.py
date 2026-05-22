#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


INFO_PATH = Path("meta/info.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute chunk-level GAE-like advantages from a value field, select globally top chunks by "
            "coverage ratio, and write back a frame-level indicator."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the LeRobot dataset root.")
    parser.add_argument(
        "--value-field",
        type=str,
        default="complementary_info.value",
        help="Frame-level value field used to score chunks.",
    )
    parser.add_argument(
        "--intervention-field",
        type=str,
        default="complementary_info.is_intervention",
        help="Human intervention field. Frames > 0.5 are always forced to indicator=1.",
    )
    parser.add_argument(
        "--chunk-advantage-field",
        type=str,
        default="complementary_info.global_chunk_advantage",
        help="Field used to store chunk advantage at valid chunk-start frames. Non-start frames are NaN.",
    )
    parser.add_argument(
        "--chunk-start-indicator-field",
        type=str,
        default="complementary_info.global_chunk_start_indicator",
        help="Field used to mark selected chunk starts with 1.",
    )
    parser.add_argument(
        "--indicator-field",
        type=str,
        default="complementary_info.acp_indicator",
        help="Frame-level indicator field to overwrite/write.",
    )
    parser.add_argument("--chunk-size", type=int, default=50, help="Chunk size K.")
    parser.add_argument("--lam", type=float, default=0.95, help="Lambda used in chunk GAE-like accumulation.")
    parser.add_argument(
        "--l-max-percentile",
        type=float,
        default=95.0,
        help="Percentile of episode lengths used to compute L_max.",
    )
    parser.add_argument(
        "--value-normalization",
        type=str,
        choices=["none", "clip", "episode_minmax"],
        default="none",
        help="Optional per-episode value normalization mode before chunk-advantage computation.",
    )
    parser.add_argument(
        "--value-smoothing-window",
        type=int,
        default=1,
        help="Optional moving-average smoothing window on the value sequence before chunk-advantage computation.",
    )
    parser.add_argument(
        "--global-top-ratio",
        type=float,
        default=0.3,
        help="Target non-intervention frame coverage ratio for selected chunks.",
    )
    parser.add_argument(
        "--global-top-k",
        type=int,
        default=0,
        help="If > 0, select a fixed number of globally top chunks after NMS instead of ratio-based coverage.",
    )
    parser.add_argument(
        "--global-min-candidates",
        type=int,
        default=1,
        help="Minimum number of chunks to keep when using coverage-ratio selection.",
    )
    parser.add_argument(
        "--global-nms-overlap-ratio",
        type=float,
        default=0.5,
        help="Temporal NMS overlap ratio threshold used before global selection.",
    )
    parser.add_argument(
        "--exclude-intervention-chunks",
        action="store_true",
        default=True,
        help="Exclude chunks that touch any intervention-marked frame from the global ranking pool.",
    )
    parser.add_argument(
        "--no-exclude-intervention-chunks",
        dest="exclude_intervention_chunks",
        action="store_false",
        help="Allow chunks overlapping intervention-marked frames to participate in global ranking.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def normalize_values(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("'values' must contain only finite numbers.")

    if mode == "none":
        return values.astype(np.float32, copy=True)
    if mode == "clip":
        return np.clip(values, -1.0, 0.0).astype(np.float32)
    if mode == "episode_minmax":
        if values.size == 0:
            return values.astype(np.float32, copy=True)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        span = max_value - min_value
        if span <= 1e-8:
            return np.clip(values, -1.0, 0.0).astype(np.float32)
        scaled = (values - min_value) / span
        return np.clip(scaled - 1.0, -1.0, 0.0).astype(np.float32)
    raise ValueError("'mode' must be one of {'none', 'clip', 'episode_minmax'}.")


def smooth_values(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if window <= 1 or values.size == 0:
        return values.astype(np.float32, copy=True)

    radius = window // 2
    smoothed = np.empty_like(values, dtype=np.float32)
    for idx in range(values.size):
        left = max(0, idx - radius)
        right = min(values.size, idx + radius + 1)
        smoothed[idx] = float(np.mean(values[left:right]))
    return smoothed


def compute_chunk_advantages_batch(
    values: np.ndarray,
    chunk_size: int,
    l_max: float,
    lam: float = 0.95,
) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("'chunk_size' must be > 0.")
    if l_max <= 0:
        raise ValueError("'l_max' must be > 0.")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("'lam' must be within [0, 1].")

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("'values' must contain only finite numbers.")
    if values.shape[0] < int(chunk_size) + 1:
        return np.empty((0,), dtype=np.float64)

    rho = 1.0 / float(l_max)
    deltas = -rho + values[1:] - values[:-1]
    powers = float(lam) ** np.arange(int(chunk_size), dtype=np.float64)
    return np.convolve(deltas, powers[::-1], mode="valid").astype(np.float64, copy=False)


def percentile_or_max(lengths: list[int], percentile: float) -> float:
    if not lengths:
        raise ValueError("Cannot compute L_max from an empty length list.")
    arr = np.asarray(lengths, dtype=np.float32)
    if percentile >= 100.0:
        return float(np.max(arr))
    return float(np.percentile(arr, percentile))


def safe_l_max(value: float, chunk_size: int) -> float:
    return float(max(float(value), float(chunk_size + 1)))


def valid_chunk_start_count(episode_length: int, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError("'chunk_size' must be > 0.")
    return max(episode_length - chunk_size, 0)


def iter_episode_slices(episode_indices: np.ndarray):
    episode_indices = np.asarray(episode_indices)
    if episode_indices.ndim != 1:
        raise ValueError(f"'episode_indices' must be rank-1, got shape={tuple(episode_indices.shape)}.")
    if episode_indices.size == 0:
        return

    start = 0
    current = episode_indices[0]
    for idx in range(1, episode_indices.size):
        if episode_indices[idx] != current:
            yield int(current), slice(start, idx)
            start = idx
            current = episode_indices[idx]
    yield int(current), slice(start, episode_indices.size)


def temporal_overlap_ratio(start_a: int, start_b: int, length: int) -> float:
    if length <= 0:
        raise ValueError("'length' must be > 0.")
    end_a = start_a + length
    end_b = start_b + length
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    return float(inter) / float(length)


def temporal_nms(
    starts: np.ndarray,
    scores: np.ndarray,
    length: int,
    threshold: float,
    top_k: int = 0,
) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if starts.ndim != 1 or scores.ndim != 1:
        raise ValueError("'starts' and 'scores' must be rank-1 arrays.")
    if starts.shape[0] != scores.shape[0]:
        raise ValueError("'starts' and 'scores' must have the same length.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("'threshold' must be within [0, 1].")
    if starts.size == 0:
        return np.empty((0,), dtype=np.int64)

    order = np.argsort(-scores, kind="mergesort")
    kept: list[int] = []
    for candidate_idx in order:
        candidate_start = int(starts[candidate_idx])
        if all(temporal_overlap_ratio(candidate_start, int(starts[kept_idx]), length) <= threshold for kept_idx in kept):
            kept.append(int(candidate_idx))
            if top_k > 0 and len(kept) >= top_k:
                break
    return np.asarray(kept, dtype=np.int64)


def feature_infos_for_columns(columns: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    infos: dict[str, dict[str, Any]] = {}
    for field, values in columns.items():
        if np.issubdtype(values.dtype, np.integer):
            infos[field] = {"dtype": "int64", "shape": (1,), "names": None}
        elif np.issubdtype(values.dtype, np.floating):
            infos[field] = {"dtype": "float32", "shape": (1,), "names": None}
        else:
            raise ValueError(f"Unsupported annotation dtype for field '{field}': {values.dtype}")
    return infos


def update_feature_metadata(dataset_root: Path, feature_infos: dict[str, dict[str, Any]]) -> None:
    info = load_json(dataset_root / INFO_PATH)
    for feature_name, feature_info in feature_infos.items():
        info["features"][feature_name] = {
            "dtype": feature_info["dtype"],
            "shape": list(feature_info["shape"]),
            "names": feature_info.get("names"),
        }
    write_json(info, dataset_root / INFO_PATH)


def write_columns_in_place(
    dataset_root: Path,
    absolute_indices: np.ndarray,
    columns: dict[str, np.ndarray],
    feature_infos: dict[str, dict[str, Any]] | None = None,
) -> None:
    absolute_indices = np.asarray(absolute_indices, dtype=np.int64)
    if absolute_indices.ndim != 1:
        raise ValueError("'absolute_indices' must be rank-1.")
    if absolute_indices.size == 0:
        raise ValueError("Cannot write annotations for an empty dataset.")

    if feature_infos is None:
        feature_infos = feature_infos_for_columns(columns)

    max_index = int(np.max(absolute_indices))
    selected = np.zeros(max_index + 1, dtype=np.bool_)
    selected[absolute_indices] = True

    lookups: dict[str, np.ndarray] = {}
    for field, values in columns.items():
        values = np.asarray(values)
        if values.shape[0] != absolute_indices.shape[0]:
            raise ValueError(
                f"Column '{field}' length mismatch: expected {absolute_indices.shape[0]}, got {values.shape[0]}."
            )
        lookup_dtype = np.float32 if feature_infos[field]["dtype"] == "float32" else np.int64
        lookup = np.zeros(max_index + 1, dtype=lookup_dtype)
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
            ftype = feature_infos[field]["dtype"]
            if ftype == "float32":
                default_value = np.nan
                target_dtype = np.float32
                pa_type = pa.float32()
            elif ftype == "int64":
                default_value = 0
                target_dtype = np.int64
                pa_type = pa.int64()
            else:
                raise ValueError(f"Unsupported annotation dtype '{ftype}' for field '{field}'.")

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


def load_frames(dataset_root: Path, value_field: str, intervention_field: str) -> pd.DataFrame:
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No frame parquet files found under {data_dir}")

    frames = []
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        required = {"index", "episode_index", "frame_index", value_field}
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in {parquet_path}: {missing}")
        keep_cols = ["index", "episode_index", "frame_index", value_field]
        if intervention_field in df.columns:
            keep_cols.append(intervention_field)
        frames.append(df[keep_cols].copy())

    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values("index").reset_index(drop=True)


def choose_prefix_by_coverage(
    ordered_starts: list[int],
    total_rows: int,
    chunk_size: int,
    target_coverage_ratio: float,
    global_top_k: int,
    min_candidates: int,
    eligible_mask: np.ndarray,
) -> tuple[int, int]:
    num_candidates = len(ordered_starts)
    if num_candidates <= 0:
        return 0, 0
    if global_top_k > 0:
        keep = min(global_top_k, num_candidates)
        coverage = np.zeros(total_rows, dtype=np.bool_)
        for start in ordered_starts[:keep]:
            end = min(int(start) + int(chunk_size), int(total_rows))
            coverage[int(start):end] |= eligible_mask[int(start):end]
        return keep, int(np.sum(coverage))
    if target_coverage_ratio <= 0.0:
        return 0, 0

    eligible_total = int(np.sum(eligible_mask))
    target_frames = float(target_coverage_ratio) * float(eligible_total)
    coverage = np.zeros(total_rows, dtype=np.bool_)
    covered_frames = 0
    best_count = 0
    best_diff = float("inf")

    for rank, start in enumerate(ordered_starts, start=1):
        end = min(int(start) + int(chunk_size), int(total_rows))
        newly = int(np.sum((~coverage[int(start):end]) & eligible_mask[int(start):end]))
        if newly > 0:
            coverage[int(start):end] |= eligible_mask[int(start):end]
            covered_frames += newly

        if rank < min_candidates:
            continue

        diff = abs(float(covered_frames) - target_frames)
        if diff < best_diff:
            best_diff = diff
            best_count = rank

        if covered_frames >= target_frames:
            break

    if best_count <= 0 and min_candidates > 0:
        best_count = min(num_candidates, min_candidates)

    coverage = np.zeros(total_rows, dtype=np.bool_)
    for start in ordered_starts[:best_count]:
        end = min(int(start) + int(chunk_size), int(total_rows))
        coverage[int(start):end] |= eligible_mask[int(start):end]
    return best_count, int(np.sum(coverage))


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if args.chunk_size <= 0:
        raise ValueError("'chunk_size' must be > 0.")
    if args.value_smoothing_window <= 0:
        raise ValueError("'value_smoothing_window' must be > 0.")
    if not 0.0 <= args.lam <= 1.0:
        raise ValueError("'lam' must be within [0, 1].")
    if not 0.0 <= args.global_top_ratio <= 1.0:
        raise ValueError("'global_top_ratio' must be within [0, 1].")
    if args.global_top_k < 0:
        raise ValueError("'global_top_k' must be >= 0.")
    if args.global_min_candidates < 0:
        raise ValueError("'global_min_candidates' must be >= 0.")
    if not 0.0 <= args.global_nms_overlap_ratio <= 1.0:
        raise ValueError("'global_nms_overlap_ratio' must be within [0, 1].")
    if not 0.0 <= args.l_max_percentile:
        raise ValueError("'l_max_percentile' must be >= 0.")

    frames = load_frames(dataset_root, args.value_field, args.intervention_field)
    num_rows = len(frames)
    if num_rows == 0:
        raise ValueError("Dataset has no frames.")

    absolute_indices = frames["index"].to_numpy(dtype=np.int64, copy=False)
    episode_indices = frames["episode_index"].to_numpy(dtype=np.int64, copy=False)
    values = frames[args.value_field].to_numpy(dtype=np.float32, copy=False)

    if args.intervention_field in frames.columns:
        intervention_mask = frames[args.intervention_field].to_numpy(dtype=np.float32, copy=False) > 0.5
    else:
        intervention_mask = np.zeros(num_rows, dtype=np.bool_)

    eligible_mask = ~intervention_mask

    episode_lengths = [int(ep_slice.stop - ep_slice.start) for _, ep_slice in iter_episode_slices(episode_indices)]
    l_max = safe_l_max(percentile_or_max(episode_lengths, float(args.l_max_percentile)), int(args.chunk_size))

    chunk_advantage = np.full(num_rows, np.nan, dtype=np.float32)
    chunk_start_indicator = np.zeros(num_rows, dtype=np.int64)
    indicator = np.zeros(num_rows, dtype=np.int64)

    candidate_starts: list[int] = []
    candidate_scores: list[float] = []
    valid_start_count_total = 0
    excluded_intervention_candidate_count = 0

    all_positions = np.arange(num_rows, dtype=np.int64)
    for _, ep_slice in iter_episode_slices(episode_indices):
        positions = all_positions[ep_slice]
        episode_length = int(positions.size)
        num_starts = valid_chunk_start_count(episode_length, int(args.chunk_size))
        if num_starts == 0:
            continue

        valid_start_count_total += num_starts

        ep_values = smooth_values(values[positions], int(args.value_smoothing_window))
        ep_values = normalize_values(ep_values, args.value_normalization)
        ep_adv = compute_chunk_advantages_batch(ep_values, int(args.chunk_size), float(l_max), float(args.lam))
        ep_intervention = intervention_mask[positions]

        for local_start in range(num_starts):
            global_start = int(positions[local_start])
            score = float(ep_adv[local_start])
            chunk_advantage[global_start] = np.float32(score)

            local_end = int(local_start) + int(args.chunk_size)
            if args.exclude_intervention_chunks and bool(np.any(ep_intervention[int(local_start):local_end])):
                excluded_intervention_candidate_count += 1
                continue

            candidate_starts.append(global_start)
            candidate_scores.append(score)

    candidate_starts_np = np.asarray(candidate_starts, dtype=np.int64)
    candidate_scores_np = np.asarray(candidate_scores, dtype=np.float32)

    kept_relative_idx = temporal_nms(
        candidate_starts_np,
        candidate_scores_np,
        length=int(args.chunk_size),
        threshold=float(args.global_nms_overlap_ratio),
        top_k=0,
    )

    nms_starts = candidate_starts_np[kept_relative_idx].tolist()
    nms_scores = candidate_scores_np[kept_relative_idx]
    if len(nms_starts) > 1:
        order = np.argsort(-nms_scores, kind="mergesort")
        ordered_starts = [int(nms_starts[i]) for i in order]
    else:
        ordered_starts = [int(s) for s in nms_starts]

    keep_count, selected_eligible_frames = choose_prefix_by_coverage(
        ordered_starts=ordered_starts,
        total_rows=num_rows,
        chunk_size=int(args.chunk_size),
        target_coverage_ratio=float(args.global_top_ratio),
        global_top_k=int(args.global_top_k),
        min_candidates=int(args.global_min_candidates),
        eligible_mask=eligible_mask,
    )

    for start in ordered_starts[:keep_count]:
        chunk_start_indicator[int(start)] = 1
        end = min(int(start) + int(args.chunk_size), int(num_rows))
        indicator[int(start):end] = 1

    indicator[intervention_mask] = 1

    columns = {
        args.chunk_advantage_field: chunk_advantage.astype(np.float32, copy=False),
        args.chunk_start_indicator_field: chunk_start_indicator.astype(np.int64, copy=False),
        args.indicator_field: indicator.astype(np.int64, copy=False),
    }
    feature_infos = feature_infos_for_columns(columns)
    write_columns_in_place(
        dataset_root=dataset_root,
        absolute_indices=absolute_indices,
        columns=columns,
        feature_infos=feature_infos,
    )

    eligible_total = int(np.sum(eligible_mask))
    intervention_total = int(np.sum(intervention_mask))
    final_positive_total = int(np.sum(indicator))
    payload = {
        "dataset_root": str(dataset_root),
        "value_field": args.value_field,
        "chunk_advantage_field": args.chunk_advantage_field,
        "chunk_start_indicator_field": args.chunk_start_indicator_field,
        "indicator_field": args.indicator_field,
        "intervention_field": args.intervention_field,
        "num_rows": num_rows,
        "chunk_size": int(args.chunk_size),
        "lam": float(args.lam),
        "l_max": float(l_max),
        "l_max_percentile": float(args.l_max_percentile),
        "value_normalization": args.value_normalization,
        "value_smoothing_window": int(args.value_smoothing_window),
        "global_top_ratio_target": float(args.global_top_ratio),
        "global_top_k": int(args.global_top_k),
        "global_min_candidates": int(args.global_min_candidates),
        "global_nms_overlap_ratio": float(args.global_nms_overlap_ratio),
        "exclude_intervention_chunks": bool(args.exclude_intervention_chunks),
        "valid_chunk_start_count": int(valid_start_count_total),
        "candidate_chunk_count": int(candidate_starts_np.shape[0]),
        "excluded_intervention_candidate_count": int(excluded_intervention_candidate_count),
        "nms_chunk_count": int(len(ordered_starts)),
        "selected_chunk_count": int(keep_count),
        "eligible_frame_count": int(eligible_total),
        "selected_eligible_frame_count": int(selected_eligible_frames),
        "selected_eligible_coverage_ratio": float(selected_eligible_frames / eligible_total) if eligible_total > 0 else 0.0,
        "intervention_frame_count": int(intervention_total),
        "final_positive_frame_count": int(final_positive_total),
        "final_positive_ratio": float(final_positive_total / num_rows) if num_rows > 0 else 0.0,
        "chunk_advantage_min": float(np.nanmin(chunk_advantage)) if np.any(~np.isnan(chunk_advantage)) else None,
        "chunk_advantage_max": float(np.nanmax(chunk_advantage)) if np.any(~np.isnan(chunk_advantage)) else None,
        "chunk_advantage_mean": float(np.nanmean(chunk_advantage)) if np.any(~np.isnan(chunk_advantage)) else None,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
