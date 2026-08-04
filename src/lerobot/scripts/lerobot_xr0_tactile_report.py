#!/usr/bin/env python

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_summary(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No diagnostic records found in {summary_path}")
    return rows


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row.get(name) or 0.0) for row in rows], dtype=np.float64)


def _load_action_record(
    diagnostics_dir: Path, row: dict[str, str]
) -> dict[str, np.ndarray]:
    action_path = diagnostics_dir / row["action_npz"]
    if not action_path.is_file():
        raise FileNotFoundError(f"Missing action record: {action_path}")
    with np.load(action_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _right_action_dimensions(action_dim: int) -> tuple[list[int], int]:
    if action_dim >= 27:
        return list(range(21, 27)), 20
    if action_dim >= 14:
        return list(range(7, 13)), 13
    raise ValueError(f"Unsupported action dimension for XR0 report: {action_dim}")


def generate_report(diagnostics_dir: Path, output_dir: Path | None = None) -> Path:
    diagnostics_dir = diagnostics_dir.expanduser().resolve()
    summary_path = diagnostics_dir / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing diagnostic summary: {summary_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_summary(summary_path)
    output_dir = (output_dir or diagnostics_dir / "reports").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    action_time = _float_column(rows, "action_time_s")
    suffix_delta = _float_column(rows, "suffix_mean_abs_delta")
    next_delta = _float_column(rows, "next_generated_action_delta")
    gripper_delta = _float_column(rows, "right_gripper_suffix_delta")
    tactile_delta = _float_column(rows, "tactile_pixel_mean_delta")

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(
        action_time,
        suffix_delta,
        label="Generated suffix mean action delta",
        linewidth=2,
    )
    axis.plot(action_time, next_delta, label="First new action delta", linewidth=1.5)
    axis.plot(
        action_time, gripper_delta, label="Right gripper suffix delta", linewidth=1.5
    )
    axis.set_xlabel("Nominal action time (s)")
    axis.set_ylabel("Absolute action delta")
    axis.grid(alpha=0.3)
    tactile_axis = axis.twinx()
    tactile_axis.plot(
        action_time,
        tactile_delta,
        color="black",
        linestyle="--",
        alpha=0.65,
        label="Tactile pixel mean delta",
    )
    tactile_axis.set_ylabel("Normalized tactile pixel delta")
    handles, labels = axis.get_legend_handles_labels()
    tactile_handles, tactile_labels = tactile_axis.get_legend_handles_labels()
    axis.legend(handles + tactile_handles, labels + tactile_labels, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_dir / "action_delta_timeline.png", dpi=180)
    plt.close(figure)

    gripper_record_index = int(np.argmax(gripper_delta))
    gripper_record = _load_action_record(diagnostics_dir, rows[gripper_record_index])
    primary = gripper_record["primary_actions"]
    masked = gripper_record["masked_actions"]
    prefix_length = int(gripper_record["rtc_prefix_length"])
    joint_dimensions, gripper_dimension = _right_action_dimensions(primary.shape[-1])
    horizon = np.arange(primary.shape[-2])

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        horizon, primary[:, gripper_dimension], label="Current tactile", linewidth=2
    )
    axis.plot(
        horizon,
        masked[:, gripper_dimension],
        label="Unloaded tactile baseline",
        linestyle="--",
        linewidth=2,
    )
    axis.axvspan(
        0, max(0, prefix_length - 1), color="gray", alpha=0.2, label="RTC fixed prefix"
    )
    axis.set_xlabel("Action horizon step")
    axis.set_ylabel("Right gripper position")
    axis.set_title(
        f"Largest right-gripper tactile effect at timestep {rows[gripper_record_index]['timestep']}"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "right_gripper_comparison.png", dpi=180)
    plt.close(figure)

    strongest_record_index = int(np.argmax(suffix_delta))
    strongest_record = _load_action_record(
        diagnostics_dir, rows[strongest_record_index]
    )
    strongest_primary = strongest_record["primary_actions"]
    strongest_masked = strongest_record["masked_actions"]
    strongest_prefix = int(strongest_record["rtc_prefix_length"])
    per_joint_delta = np.abs(
        strongest_primary[strongest_prefix:, joint_dimensions]
        - strongest_masked[strongest_prefix:, joint_dimensions]
    ).mean(axis=0)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar([f"joint_{index}" for index in range(1, 7)], per_joint_delta)
    axis.set_ylabel("Mean absolute suffix delta")
    axis.set_title(
        f"Right-joint tactile effect at timestep {rows[strongest_record_index]['timestep']}"
    )
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "right_joint_deltas.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(
        tactile_delta,
        suffix_delta,
        c=action_time,
        cmap="viridis",
        s=42,
        alpha=0.85,
    )
    axis.set_xlabel("Normalized tactile pixel mean delta")
    axis.set_ylabel("Generated suffix mean action delta")
    axis.grid(alpha=0.3)
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Nominal action time (s)")
    figure.tight_layout()
    figure.savefig(output_dir / "tactile_vs_action_delta.png", dpi=180)
    plt.close(figure)

    report_text = "\n".join(
        [
            "=== XR0 Tactile Counterfactual Report ===",
            f"diagnostics_dir: {diagnostics_dir}",
            f"records: {len(rows)}",
            f"max_suffix_mean_abs_delta: {suffix_delta.max():.6f}",
            f"max_next_generated_action_delta: {next_delta.max():.6f}",
            f"max_right_gripper_suffix_delta: {gripper_delta.max():.6f}",
            f"max_tactile_pixel_mean_delta: {tactile_delta.max():.6f}",
        ]
    )
    (output_dir / "report.txt").write_text(report_text + "\n", encoding="utf-8")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate plots from an XR0 tactile counterfactual diagnostic session."
    )
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = generate_report(args.diagnostics_dir, args.output_dir)
    print(f"XR0_TACTILE_REPORT_PASS: {output_dir}")


if __name__ == "__main__":
    main()
