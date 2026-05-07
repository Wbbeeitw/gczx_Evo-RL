#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.utils import load_info, write_info


def column_to_1d_array(values: Any, dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[-1] == 1:
        return arr[:, 0]
    return arr.reshape(arr.shape[0], -1)[:, 0].astype(dtype, copy=False)


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
    info = load_info(dataset_root)
    for feature_name, feature_info in feature_infos.items():
        info["features"][feature_name] = {
            "dtype": feature_info["dtype"],
            "shape": tuple(feature_info["shape"]),
            "names": feature_info.get("names"),
        }
    write_info(info, dataset_root)


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
                subset_indices = idx_np[in_subset]
                current[in_subset] = lookup[subset_indices]

            array = pa.array(current, type=pa_type)
            if field in new_table.schema.names:
                col_idx = new_table.schema.names.index(field)
                new_table = new_table.set_column(col_idx, field, array)
            else:
                new_table = new_table.append_column(field, array)

        pq.write_table(new_table, parquet_path, compression="snappy")

    update_feature_metadata(dataset_root=dataset_root, feature_infos=feature_infos)

