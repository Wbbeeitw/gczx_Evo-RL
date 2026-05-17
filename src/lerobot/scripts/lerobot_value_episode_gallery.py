#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_VALUE_FIELD = "complementary_info.value"
DEFAULT_ADVANTAGE_FIELD = "complementary_info.value_infer_advantage"
DEFAULT_INDICATOR_FIELD = "complementary_info.value_infer_indicator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one value/advantage/indicator figure per episode from a LeRobot dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save episode figures.")
    parser.add_argument(
        "--value-field",
        type=str,
        default=DEFAULT_VALUE_FIELD,
        help="Frame-level value field to plot.",
    )
    parser.add_argument(
        "--advantage-field",
        type=str,
        default=DEFAULT_ADVANTAGE_FIELD,
        help="Frame-level advantage field to plot.",
    )
    parser.add_argument(
        "--indicator-field",
        type=str,
        default=DEFAULT_INDICATOR_FIELD,
        help="Frame-level binary indicator field to plot.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Comma-separated episode indices to render, or 'all'.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI.")
    return parser.parse_args()


def parse_episode_selection(raw: str) -> list[int] | None:
    if raw.strip().lower() == "all":
        return None
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def load_episode_metadata(dataset_root: Path) -> pd.DataFrame:
    episodes_dir = dataset_root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {episodes_dir}")
    df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    if "episode_index" not in df.columns:
        raise KeyError("Episode metadata is missing 'episode_index'.")
    return df.sort_values("episode_index").reset_index(drop=True)


def load_frame_annotations(
    dataset_root: Path,
    value_field: str,
    advantage_field: str,
    indicator_field: str,
) -> pd.DataFrame:
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No frame parquet files found under {data_dir}")

    required_cols = ["episode_index", "frame_index", value_field, advantage_field, indicator_field]
    frames = []
    for path in parquet_files:
        df = pd.read_parquet(path, columns=required_cols)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    missing = [col for col in required_cols if col not in merged.columns]
    if missing:
        raise KeyError(f"Missing required frame columns: {missing}")
    return merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def find_positive_spans(indicator: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(indicator.astype(bool).tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(indicator) - 1))
    return spans


def add_indicator_spans(axes: list[plt.Axes], spans: list[tuple[int, int]]) -> None:
    for axis in axes:
        for start, end in spans:
            axis.axvspan(start, end, color="#f15b5b", alpha=0.12, linewidth=0)


def render_episode_figure(
    episode_index: int,
    episode_meta: pd.Series,
    episode_frames: pd.DataFrame,
    output_path: Path,
    *,
    value_field: str,
    advantage_field: str,
    indicator_field: str,
    dpi: int,
) -> dict[str, object]:
    frame_ids = episode_frames["frame_index"].to_numpy(dtype=np.int64)
    values = episode_frames[value_field].to_numpy(dtype=np.float32)
    advantages = episode_frames[advantage_field].to_numpy(dtype=np.float32)
    indicators = episode_frames[indicator_field].fillna(0).to_numpy(dtype=np.int64)
    spans = find_positive_spans(indicators)

    label = str(episode_meta.get("episode_success", "unknown"))
    pos_count = int(indicators.sum())
    length = int(episode_meta.get("length", len(episode_frames)))
    pos_ratio = pos_count / max(len(indicators), 1)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 2.2, 1.0]},
    )
    fig.patch.set_facecolor("#f5f1e8")

    add_indicator_spans([axes[0], axes[1]], spans)

    axes[0].plot(frame_ids, values, color="#1f4e79", linewidth=1.8)
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.25)
    axes[0].set_title(
        f"Episode {episode_index} | label={label} | length={length} | positive_frames={pos_count} ({pos_ratio:.1%})",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )

    axes[1].plot(frame_ids, advantages, color="#0b8f87", linewidth=1.4)
    axes[1].axhline(0.0, color="#7f7f7f", linewidth=1.0, linestyle="--", alpha=0.8)
    axes[1].set_ylabel("Advantage")
    axes[1].grid(alpha=0.25)

    axes[2].step(frame_ids, indicators, where="mid", color="#b42318", linewidth=1.5)
    axes[2].fill_between(frame_ids, 0, indicators, step="mid", color="#f15b5b", alpha=0.35)
    axes[2].set_ylabel("Mask")
    axes[2].set_xlabel("Frame Index")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks([0, 1])
    axes[2].grid(alpha=0.2)

    summary_text = (
        f"value[min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}]\n"
        f"adv[min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}]"
    )
    axes[0].text(
        0.995,
        0.02,
        summary_text,
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d5c7a3", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "episode_index": episode_index,
        "episode_success": label,
        "length": length,
        "positive_frames": pos_count,
        "positive_ratio": pos_ratio,
        "figure_path": str(output_path),
    }


def main() -> None:
    args = parse_args()
    requested_episodes = parse_episode_selection(args.episodes)

    episodes = load_episode_metadata(args.dataset_root)
    frames = load_frame_annotations(
        args.dataset_root,
        args.value_field,
        args.advantage_field,
        args.indicator_field,
    )

    available_episodes = episodes["episode_index"].astype(int).tolist()
    target_episodes = available_episodes if requested_episodes is None else requested_episodes
    missing_requested = [ep for ep in target_episodes if ep not in available_episodes]
    if missing_requested:
        raise KeyError(f"Requested episodes are not present in dataset: {missing_requested}")

    report: list[dict[str, object]] = []
    for episode_index in target_episodes:
        episode_meta = episodes.loc[episodes["episode_index"] == episode_index].iloc[0]
        episode_frames = frames.loc[frames["episode_index"] == episode_index].copy()
        if episode_frames.empty:
            raise ValueError(f"No frame rows found for episode {episode_index}.")

        output_path = args.output_dir / f"episode_{episode_index:04d}_value_gallery.png"
        item = render_episode_figure(
            episode_index,
            episode_meta,
            episode_frames,
            output_path,
            value_field=args.value_field,
            advantage_field=args.advantage_field,
            indicator_field=args.indicator_field,
            dpi=args.dpi,
        )
        report.append(item)
        print(f"saved {output_path}")

    report_path = args.output_dir / "value_episode_gallery_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
