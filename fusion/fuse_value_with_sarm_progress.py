#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse dataset value field with shifted SARM progress and write the fused value back into the dataset."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the LeRobot dataset root.")
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help="Optional explicit path to sarm_progress.parquet. Defaults to <dataset_root>/sarm_progress.parquet.",
    )
    parser.add_argument(
        "--value-field",
        type=str,
        default="complementary_info.value",
        help="Frame-level value field to overwrite.",
    )
    parser.add_argument(
        "--progress-field",
        type=str,
        default="progress_dense",
        help="Progress field to read from sarm_progress.parquet.",
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=0.5,
        help="Weight for the original value field.",
    )
    parser.add_argument(
        "--progress-weight",
        type=float,
        default=0.5,
        help="Weight for the shifted progress term.",
    )
    parser.add_argument(
        "--progress-shift",
        type=float,
        default=-1.0,
        help="Shift added to the progress field before fusion. Default -1.0 implements progress - 1.",
    )
    return parser.parse_args()


def resolve_progress_path(dataset_root: Path, progress_path: Path | None) -> Path:
    if progress_path is not None:
        return progress_path.resolve()
    return (dataset_root / "sarm_progress.parquet").resolve()


def load_progress_lookup(progress_path: Path, progress_field: str, progress_shift: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(progress_path, columns=["index", progress_field])
    if "index" not in df.columns:
        raise KeyError(f"'index' column not found in {progress_path}")
    if progress_field not in df.columns:
        raise KeyError(f"'{progress_field}' column not found in {progress_path}")

    index_np = df["index"].to_numpy(dtype=np.int64, copy=False)
    progress_np = df[progress_field].to_numpy(dtype=np.float32, copy=False)

    if index_np.size == 0:
        raise ValueError(f"{progress_path} is empty.")
    if np.any(index_np < 0):
        raise ValueError("Progress parquet contains negative frame indexes.")
    if pd.Index(index_np).duplicated().any():
        raise ValueError(f"{progress_path} contains duplicated 'index' values.")
    if np.isnan(progress_np).any():
        nan_count = int(np.isnan(progress_np).sum())
        raise ValueError(f"{progress_path} contains {nan_count} NaN values in '{progress_field}'.")

    max_index = int(np.max(index_np))
    shifted_lookup = np.full(max_index + 1, np.nan, dtype=np.float32)
    seen = np.zeros(max_index + 1, dtype=np.bool_)
    shifted_lookup[index_np] = progress_np + np.float32(progress_shift)
    seen[index_np] = True
    return shifted_lookup, seen


def fuse_dataset_values(
    dataset_root: Path,
    value_field: str,
    shifted_progress_lookup: np.ndarray,
    progress_seen: np.ndarray,
    value_weight: float,
    progress_weight: float,
) -> dict[str, float | int | str]:
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_root / 'data'}")

    max_index = shifted_progress_lookup.shape[0] - 1
    total_rows = 0
    orig_values_all: list[np.ndarray] = []
    shifted_progress_all: list[np.ndarray] = []
    fused_values_all: list[np.ndarray] = []

    for parquet_path in data_files:
        table = pq.read_table(parquet_path)
        if value_field not in table.schema.names:
            raise KeyError(f"Field '{value_field}' not found in {parquet_path}")
        if "index" not in table.schema.names:
            raise KeyError(f"'index' field not found in {parquet_path}")

        idx_np = table["index"].to_numpy().astype(np.int64, copy=False)
        if np.any(idx_np < 0) or np.any(idx_np > max_index):
            bad_min = int(np.min(idx_np))
            bad_max = int(np.max(idx_np))
            raise ValueError(
                f"Dataset frame index out of progress range in {parquet_path}: min={bad_min}, max={bad_max}, "
                f"progress_max_index={max_index}"
            )

        if not np.all(progress_seen[idx_np]):
            missing_count = int((~progress_seen[idx_np]).sum())
            raise ValueError(f"{missing_count} rows in {parquet_path} are missing progress values.")

        original_values = table[value_field].to_numpy().astype(np.float32, copy=False)
        shifted_progress = shifted_progress_lookup[idx_np]
        fused_values = value_weight * original_values + progress_weight * shifted_progress
        fused_values = fused_values.astype(np.float32, copy=False)

        col_idx = table.schema.names.index(value_field)
        new_table = table.set_column(col_idx, value_field, pa.array(fused_values, type=pa.float32()))
        pq.write_table(new_table, parquet_path, compression="snappy")

        total_rows += int(len(idx_np))
        orig_values_all.append(original_values.copy())
        shifted_progress_all.append(shifted_progress.copy())
        fused_values_all.append(fused_values.copy())

    orig_values_np = np.concatenate(orig_values_all)
    shifted_progress_np = np.concatenate(shifted_progress_all)
    fused_values_np = np.concatenate(fused_values_all)

    return {
        "num_data_files": len(data_files),
        "num_rows": total_rows,
        "original_value_min": float(np.min(orig_values_np)),
        "original_value_max": float(np.max(orig_values_np)),
        "original_value_mean": float(np.mean(orig_values_np)),
        "shifted_progress_min": float(np.min(shifted_progress_np)),
        "shifted_progress_max": float(np.max(shifted_progress_np)),
        "shifted_progress_mean": float(np.mean(shifted_progress_np)),
        "fused_value_min": float(np.min(fused_values_np)),
        "fused_value_max": float(np.max(fused_values_np)),
        "fused_value_mean": float(np.mean(fused_values_np)),
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    progress_path = resolve_progress_path(dataset_root, args.progress_path)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not progress_path.exists():
        raise FileNotFoundError(f"Progress parquet not found: {progress_path}")

    shifted_progress_lookup, progress_seen = load_progress_lookup(
        progress_path=progress_path,
        progress_field=args.progress_field,
        progress_shift=args.progress_shift,
    )
    summary = fuse_dataset_values(
        dataset_root=dataset_root,
        value_field=args.value_field,
        shifted_progress_lookup=shifted_progress_lookup,
        progress_seen=progress_seen,
        value_weight=float(args.value_weight),
        progress_weight=float(args.progress_weight),
    )

    payload = {
        "dataset_root": str(dataset_root),
        "progress_path": str(progress_path),
        "value_field": args.value_field,
        "progress_field": args.progress_field,
        "value_weight": float(args.value_weight),
        "progress_weight": float(args.progress_weight),
        "progress_shift": float(args.progress_shift),
        **summary,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
