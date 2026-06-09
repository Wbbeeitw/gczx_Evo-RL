#!/usr/bin/env python

"""Compute native XR0 32D action statistics from a LeRobot dataset.

The output is meant to be passed to ``XR0Config.xr0_stats_path``. For ALOHA
datasets this script packs 14D actions into Xiaomi's native 32D layout and,
by default, converts absolute future joint targets into XR0-style deltas.
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Iterable

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.xr0.processor_xr0 import (
    XR0_DIM,
    _aloha14_to_xr032,
    _ensure_batched_action,
    _ensure_batched_state,
    _fit_horizon,
    _fit_horizon_mask,
    _xr0_action_mask,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _as_int(value) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _parse_episodes(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    episodes = []
    for value in values:
        episodes.extend(int(part) for part in value.split(",") if part.strip())
    return episodes or None


def _iter_indices(num_frames: int, stride: int, max_samples: int | None) -> Iterable[int]:
    indices = range(0, num_frames, stride)
    if max_samples is not None:
        return islice(indices, max_samples)
    return indices


def compute_xr0_stats(
    *,
    repo_id: str,
    root: Path | None,
    output: Path,
    horizon: int,
    action_layout: str,
    actions_are_delta: bool,
    episodes: list[int] | None,
    revision: str | None,
    tolerance_s: float,
    stride: int,
    max_samples: int | None,
    min_std: float,
) -> dict:
    metadata = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    delta_timestamps = {ACTION: [i / metadata.fps for i in range(horizon)]}
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        revision=revision,
        tolerance_s=tolerance_s,
        download_videos=False,
    )

    action_sum = torch.zeros(horizon, XR0_DIM, dtype=torch.float64)
    action_sumsq = torch.zeros(horizon, XR0_DIM, dtype=torch.float64)
    action_count = torch.zeros(horizon, XR0_DIM, dtype=torch.float64)
    layout_mask = _xr0_action_mask(
        1,
        horizon,
        action_layout=action_layout,
        device=torch.device("cpu"),
        dtype=torch.bool,
    )[0]

    used_samples = 0
    for idx in _iter_indices(len(dataset), stride, max_samples):
        item = dataset.hf_dataset[idx]
        ep_idx = _as_int(item["episode_index"])
        abs_idx = _as_int(item["index"])
        query_indices, padding = dataset._get_query_indices(abs_idx, ep_idx)
        query_result = dataset._query_hf_dataset({ACTION: query_indices[ACTION]})

        state32 = _aloha14_to_xr032(_ensure_batched_state(item[OBS_STATE]), is_action=False)[0]
        action32 = _fit_horizon(_ensure_batched_action(query_result[ACTION]), horizon)[0]
        action32 = _aloha14_to_xr032(action32, is_action=True)
        if not actions_are_delta:
            action32 = action32 - state32.unsqueeze(0)

        valid_steps = ~_fit_horizon_mask(padding[f"{ACTION}_is_pad"], horizon)[0]
        valid_mask = layout_mask & valid_steps.unsqueeze(-1)
        if not torch.any(valid_mask):
            continue

        values = action32.double()
        mask = valid_mask.double()
        action_sum += values * mask
        action_sumsq += values.square() * mask
        action_count += mask
        used_samples += 1

    has_count = action_count > 0
    mean = torch.zeros_like(action_sum)
    mean[has_count] = action_sum[has_count] / action_count[has_count]

    var = torch.zeros_like(action_sum)
    var[has_count] = action_sumsq[has_count] / action_count[has_count] - mean[has_count].square()
    std = torch.ones_like(action_sum)
    std[has_count] = var[has_count].clamp_min(min_std * min_std).sqrt()

    payload = {
        ACTION: {
            "mean": mean.float(),
            "std": std.float(),
            "count": action_count.float(),
        },
        "meta": {
            "repo_id": repo_id,
            "root": str(root) if root is not None else None,
            "horizon": horizon,
            "action_layout": action_layout,
            "actions_are_delta": actions_are_delta,
            "episodes": episodes,
            "stride": stride,
            "used_samples": used_samples,
            "fps": metadata.fps,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id.")
    parser.add_argument("--root", type=Path, default=None, help="Local LeRobot dataset root.")
    parser.add_argument("--output", required=True, type=Path, help="Where to write xr0_stats.pt.")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--action-layout", choices=["aloha14", "xr0_32"], default="aloha14")
    parser.add_argument("--actions-are-delta", action="store_true")
    parser.add_argument("--episodes", nargs="*", default=None, help="Optional episode ids, space or comma separated.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame when computing stats.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-std", type=float, default=1e-6)
    args = parser.parse_args()

    payload = compute_xr0_stats(
        repo_id=args.repo_id,
        root=args.root,
        output=args.output,
        horizon=args.horizon,
        action_layout=args.action_layout,
        actions_are_delta=args.actions_are_delta,
        episodes=_parse_episodes(args.episodes),
        revision=args.revision,
        tolerance_s=args.tolerance_s,
        stride=args.stride,
        max_samples=args.max_samples,
        min_std=args.min_std,
    )
    stats = payload[ACTION]
    print(f"wrote: {args.output}")
    print(f"mean shape: {tuple(stats['mean'].shape)}")
    print(f"std shape: {tuple(stats['std'].shape)}")
    print(f"used samples: {payload['meta']['used_samples']}")


if __name__ == "__main__":
    main()
