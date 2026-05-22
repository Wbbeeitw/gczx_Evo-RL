#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Back up one frame-level column from a LeRobot dataset into a standalone parquet."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the dataset root.")
    parser.add_argument("column_name", type=str, help="Column name to back up.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Optional backup parquet path. "
            "If omitted, writes to <dataset_parent>/<dataset_name>_<column_name>_backup.parquet."
        ),
    )
    return parser.parse_args()


def sanitize_column_name(column_name: str) -> str:
    return column_name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")


def resolve_output_path(dataset_root: Path, column_name: str, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path

    dataset_root = dataset_root.resolve()
    backup_name = f"{dataset_root.name}_{sanitize_column_name(column_name)}_backup.parquet"
    return dataset_root.parent / backup_name


def build_backup(dataset_root: Path, column_name: str) -> tuple[pd.DataFrame, int]:
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    frames: list[pd.DataFrame] = []
    absolute_index = 0

    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in {parquet_path}")

        keep_cols = [col for col in ("episode_index", "frame_index") if col in df.columns]
        chunk = df[keep_cols + [column_name]].copy()
        chunk.insert(0, "absolute_index", range(absolute_index, absolute_index + len(chunk)))
        chunk.insert(1, "row_in_file", range(len(chunk)))
        chunk.insert(2, "source_file", str(parquet_path.relative_to(dataset_root)))
        absolute_index += len(chunk)
        frames.append(chunk)

    return pd.concat(frames, ignore_index=True), len(parquet_files)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_path = resolve_output_path(dataset_root, args.column_name, args.output_path)

    backup_df, source_parquet_files = build_backup(dataset_root, args.column_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_df.to_parquet(output_path, index=False)

    payload = {
        "dataset_root": str(dataset_root),
        "column_name": args.column_name,
        "output_path": str(output_path),
        "num_rows": int(len(backup_df)),
        "num_non_null": int(backup_df[args.column_name].notna().sum()),
        "num_null": int(backup_df[args.column_name].isna().sum()),
        "source_parquet_files": source_parquet_files,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
