#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms.functional import to_pil_image

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

_RESAMPLING = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _short_camera_label(camera_key: str) -> str:
    return camera_key.split(".")[-1]


def output_tag_from_prefix(output_prefix: str) -> str:
    tag = output_prefix
    if "vgsacm_" in tag:
        tag = tag.split("vgsacm_", maxsplit=1)[1]
    if tag.endswith("_chunkmask"):
        tag = tag[: -len("_chunkmask")]
    return tag.replace(".", "_")


def _parse_requested_camera_keys(requested_camera_keys: str | None) -> list[str]:
    if requested_camera_keys is None or not requested_camera_keys.strip():
        return []

    camera_keys: list[str] = []
    for raw_key in requested_camera_keys.split(","):
        key = raw_key.strip()
        if key and key not in camera_keys:
            camera_keys.append(key)
    return camera_keys


def select_camera_keys(
    available_camera_keys: list[str],
    requested_camera_keys: str | None,
    max_cameras: int,
) -> list[str]:
    if len(available_camera_keys) == 0:
        raise ValueError("Dataset does not contain any camera keys for visualization.")

    requested = _parse_requested_camera_keys(requested_camera_keys)
    if requested:
        missing = [key for key in requested if key not in available_camera_keys]
        if missing:
            raise ValueError(
                f"Unknown visualization camera keys: {missing}. Available camera keys: {available_camera_keys}"
            )
        return requested

    if max_cameras <= 0:
        return list(available_camera_keys)
    return list(available_camera_keys[:max_cameras])


def verify_episode_chunk_mask(
    *,
    ep_chunk_start_indicator: np.ndarray,
    ep_indicator: np.ndarray,
    chunk_size: int,
) -> dict[str, Any]:
    ep_chunk_start_indicator = np.asarray(ep_chunk_start_indicator, dtype=np.int64).reshape(-1)
    ep_indicator = np.asarray(ep_indicator, dtype=np.int64).reshape(-1)

    starts = np.flatnonzero(ep_chunk_start_indicator > 0)
    expected_indicator = np.zeros_like(ep_indicator, dtype=np.int64)
    valid_start_limit = max(0, ep_indicator.shape[0] - int(chunk_size) + 1)
    invalid_start_positions = [int(start) for start in starts if int(start) >= valid_start_limit]

    for start in starts:
        start = int(start)
        if start >= valid_start_limit:
            continue
        end = min(start + int(chunk_size), ep_indicator.shape[0])
        expected_indicator[start:end] = 1

    mismatch_positions = np.flatnonzero(expected_indicator != ep_indicator)
    invalid_indicator_values = [int(v) for v in np.unique(ep_indicator) if int(v) not in {0, 1}]
    invalid_start_values = [int(v) for v in np.unique(ep_chunk_start_indicator) if int(v) not in {0, 1}]

    passed = (
        len(invalid_indicator_values) == 0
        and len(invalid_start_values) == 0
        and len(invalid_start_positions) == 0
        and mismatch_positions.size == 0
    )
    return {
        "pass": bool(passed),
        "selected_chunk_count": int(starts.size),
        "expected_positive_frames": int(np.sum(expected_indicator)),
        "actual_positive_frames": int(np.sum(ep_indicator)),
        "indicator_mismatch_count": int(mismatch_positions.size),
        "indicator_mismatch_positions": [int(pos) for pos in mismatch_positions[:64]],
        "invalid_indicator_values": invalid_indicator_values,
        "invalid_chunk_start_values": invalid_start_values,
        "invalid_start_positions": invalid_start_positions,
    }


def verify_chunk_mask_consistency(
    *,
    episode_indices: np.ndarray,
    chunk_start_indicator: np.ndarray,
    indicator: np.ndarray,
    chunk_size: int,
) -> dict[str, Any]:
    episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    chunk_start_indicator = np.asarray(chunk_start_indicator, dtype=np.int64).reshape(-1)
    indicator = np.asarray(indicator, dtype=np.int64).reshape(-1)

    per_episode: dict[str, Any] = {}
    failed_episodes: list[int] = []

    for episode_index in sorted(int(v) for v in np.unique(episode_indices)):
        positions = np.flatnonzero(episode_indices == episode_index)
        episode_report = verify_episode_chunk_mask(
            ep_chunk_start_indicator=chunk_start_indicator[positions],
            ep_indicator=indicator[positions],
            chunk_size=chunk_size,
        )
        per_episode[str(episode_index)] = episode_report
        if not bool(episode_report["pass"]):
            failed_episodes.append(int(episode_index))

    return {
        "pass": len(failed_episodes) == 0,
        "episodes_checked": len(per_episode),
        "failed_episodes": failed_episodes,
        "per_episode": per_episode,
    }


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / max(image.width, 1), height / max(image.height, 1))
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        resample=_RESAMPLING,
    )
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    offset_x = (width - resized.width) // 2
    offset_y = (height - resized.height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _build_labeled_thumbnail(
    image: Image.Image,
    *,
    width: int,
    height: int,
    label: str,
    accent: tuple[int, int, int],
) -> Image.Image:
    label_h = 28
    frame = _fit_image(image, width, height)
    thumb = Image.new("RGB", (width, height + label_h), (252, 252, 252))
    thumb.paste(frame, (0, 0))

    draw = ImageDraw.Draw(thumb)
    draw.rectangle((0, 0, width - 1, height - 1), outline=accent, width=4)
    draw.rectangle((0, height, width - 1, height + label_h - 1), fill=(245, 245, 245))
    font = _load_font(16)
    draw.text((10, height + 5), label, fill=(40, 40, 40), font=font)
    return thumb


def _frame_to_pil_image(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))

    if array.dtype != np.uint8:
        array = array.astype(np.float32, copy=False)
        max_value = float(np.max(array)) if array.size > 0 else 1.0
        if max_value <= 1.0 + 1e-6:
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
        array = array.astype(np.uint8)

    if array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        return Image.fromarray(array).convert("RGB")
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    return to_pil_image(array).convert("RGB")


def _curve_points(
    values: np.ndarray,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return []

    x_denom = max(1, values.size - 1)
    y_denom = max(1e-6, y_max - y_min)
    points: list[tuple[int, int]] = []
    for idx, value in enumerate(values):
        x = int(round(x0 + width * (idx / x_denom)))
        y_norm = np.clip((float(value) - y_min) / y_denom, 0.0, 1.0)
        y = int(round(y0 + (1.0 - y_norm) * height))
        points.append((x, y))
    return points


def _draw_timeline_panel(
    *,
    ep_values: np.ndarray,
    ep_frame_indices: np.ndarray,
    ep_chunk_advantage: np.ndarray,
    ep_chunk_start_indicator: np.ndarray,
    ep_indicator: np.ndarray,
    chunk_size: int,
    width: int,
) -> Image.Image:
    panel_h = 360
    panel = Image.new("RGBA", (width, panel_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(panel)

    margin_left = 70
    margin_right = 40
    margin_top = 35
    plot_h = 220
    indicator_h = 28
    gap = 22
    plot_w = width - margin_left - margin_right
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_y1 = plot_y0 + plot_h
    indicator_y0 = plot_y1 + gap
    indicator_y1 = indicator_y0 + indicator_h

    values = np.asarray(ep_values, dtype=np.float32).reshape(-1)
    frame_indices = np.asarray(ep_frame_indices, dtype=np.int64).reshape(-1)
    chunk_advantage = np.asarray(ep_chunk_advantage, dtype=np.float32).reshape(-1)
    indicator = np.asarray(ep_indicator, dtype=np.int64).reshape(-1)
    selected_starts = [int(v) for v in np.flatnonzero(np.asarray(ep_chunk_start_indicator, dtype=np.int64) > 0)]

    y_min = float(np.min(values)) if values.size > 0 else 0.0
    y_max = float(np.max(values)) if values.size > 0 else 1.0
    if abs(y_max - y_min) < 1e-6:
        y_min -= 0.5
        y_max += 0.5
    padding = max(1e-3, (y_max - y_min) * 0.08)
    y_min -= padding
    y_max += padding

    draw.rounded_rectangle((plot_x0, plot_y0, plot_x0 + plot_w, plot_y1), radius=10, fill=(250, 250, 252))
    draw.rounded_rectangle(
        (plot_x0, indicator_y0, plot_x0 + plot_w, indicator_y1),
        radius=8,
        fill=(246, 246, 246),
    )

    grid_color = (220, 224, 230, 255)
    for frac in (0.25, 0.5, 0.75):
        gy = int(round(plot_y0 + frac * plot_h))
        draw.line((plot_x0, gy, plot_x0 + plot_w, gy), fill=grid_color, width=1)

    if values.size > 1:
        x_denom = max(1, values.size - 1)
        for chunk_id, start in enumerate(selected_starts, start=1):
            end = min(start + int(chunk_size), values.size)
            x0 = int(round(plot_x0 + plot_w * (start / x_denom)))
            x1 = int(round(plot_x0 + plot_w * ((max(end - 1, start)) / x_denom)))
            draw.rectangle((x0, plot_y0, x1, plot_y1), fill=(255, 110, 110, 42))
            draw.rectangle((x0, indicator_y0, x1, indicator_y1), fill=(255, 110, 110, 72))
            label_x = min(max(plot_x0 + 6, x0 + 6), plot_x0 + plot_w - 36)
            draw.text((label_x, plot_y0 + 6), f"C{chunk_id}", fill=(160, 20, 20), font=_load_font(14))

    points = _curve_points(
        values,
        x0=plot_x0,
        y0=plot_y0,
        width=plot_w,
        height=plot_h,
        y_min=y_min,
        y_max=y_max,
    )
    if len(points) >= 2:
        draw.line(points, fill=(70, 120, 220, 255), width=3)

    indicator_segments: list[tuple[int, int]] = []
    start_idx: int | None = None
    for idx, flag in enumerate(indicator):
        if int(flag) > 0 and start_idx is None:
            start_idx = idx
        elif int(flag) <= 0 and start_idx is not None:
            indicator_segments.append((start_idx, idx))
            start_idx = None
    if start_idx is not None:
        indicator_segments.append((start_idx, indicator.size))

    if values.size > 1:
        x_denom = max(1, values.size - 1)
        for start, end in indicator_segments:
            x0 = int(round(plot_x0 + plot_w * (start / x_denom)))
            x1 = int(round(plot_x0 + plot_w * ((max(end - 1, start)) / x_denom)))
            draw.rectangle((x0, indicator_y0, x1, indicator_y1), fill=(225, 60, 60, 255))

    font = _load_font(16)
    small_font = _load_font(14)
    draw.text((plot_x0, 8), "Value Timeline", fill=(35, 35, 35), font=_load_font(20))
    draw.text((plot_x0, indicator_y0 - 22), "Indicator", fill=(55, 55, 55), font=font)

    if values.size > 0:
        start_frame = int(frame_indices[0])
        end_frame = int(frame_indices[-1])
        summary = (
            f"frames {start_frame}-{end_frame} | selected chunks {len(selected_starts)} | "
            f"positive frames {int(np.sum(indicator))}"
        )
        draw.text((plot_x0 + 150, indicator_y0 - 22), summary, fill=(85, 85, 85), font=small_font)

    draw.text((12, plot_y0 - 6), f"{y_max:.3f}", fill=(80, 80, 80), font=small_font)
    draw.text((12, plot_y1 - 10), f"{y_min:.3f}", fill=(80, 80, 80), font=small_font)

    for start in selected_starts:
        label = f"{float(chunk_advantage[start]):.3f}" if np.isfinite(chunk_advantage[start]) else "nan"
        if values.size > 1:
            x = int(round(plot_x0 + plot_w * (start / max(1, values.size - 1))))
            draw.line((x, plot_y0, x, plot_y1), fill=(190, 25, 25, 180), width=2)
            draw.text((min(x + 6, plot_x0 + plot_w - 52), plot_y1 - 22), label, fill=(140, 20, 20), font=small_font)

    return panel.convert("RGB")


def _compose_episode_figure(
    *,
    dataset: LeRobotDataset,
    episode_index: int,
    output_prefix: str,
    episode_positions: np.ndarray,
    ep_frame_indices: np.ndarray,
    ep_values: np.ndarray,
    ep_chunk_advantage: np.ndarray,
    ep_chunk_start_indicator: np.ndarray,
    ep_indicator: np.ndarray,
    chunk_size: int,
    camera_keys: list[str],
    output_path: Path,
    episode_verification: dict[str, Any],
) -> None:
    figure_w = 1600
    margin_x = 36
    header_h = 72
    timeline_panel = _draw_timeline_panel(
        ep_values=ep_values,
        ep_frame_indices=ep_frame_indices,
        ep_chunk_advantage=ep_chunk_advantage,
        ep_chunk_start_indicator=ep_chunk_start_indicator,
        ep_indicator=ep_indicator,
        chunk_size=chunk_size,
        width=figure_w - 2 * margin_x,
    )

    selected_starts = [int(v) for v in np.flatnonzero(np.asarray(ep_chunk_start_indicator, dtype=np.int64) > 0)]

    thumb_w = 250
    thumb_h = 180
    chunk_title_h = 28
    row_gap = 16
    chunk_gap = 24
    block_h = chunk_title_h + 2 * (thumb_h + 28) + row_gap + 14
    content_h = 40 if len(selected_starts) == 0 else len(selected_starts) * block_h + (len(selected_starts) - 1) * chunk_gap
    figure_h = header_h + timeline_panel.height + 28 + content_h + 36

    canvas = Image.new("RGB", (figure_w, figure_h), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(26)
    body_font = _load_font(16)
    small_font = _load_font(14)

    tag = output_tag_from_prefix(output_prefix)
    status = "PASS" if bool(episode_verification["pass"]) else "FAIL"
    title = f"Episode {episode_index:03d} | {tag}"
    summary = (
        f"selected chunks={len(selected_starts)} | positive frames={int(np.sum(ep_indicator))} | "
        f"mask check={status}"
    )
    draw.text((margin_x, 18), title, fill=(30, 30, 30), font=title_font)
    draw.text((margin_x, 46), summary, fill=(80, 80, 80), font=body_font)

    timeline_y = header_h
    canvas.paste(timeline_panel, (margin_x, timeline_y))

    item_cache: dict[int, dict[str, Any]] = {}
    content_y = timeline_y + timeline_panel.height + 28

    if len(selected_starts) == 0:
        draw.text((margin_x, content_y), "No selected chunks in this episode.", fill=(90, 90, 90), font=body_font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return

    for chunk_idx, local_start in enumerate(selected_starts, start=1):
        local_end = min(local_start + int(chunk_size) - 1, episode_positions.shape[0] - 1)
        start_row = int(episode_positions[local_start])
        end_row = int(episode_positions[local_end])
        if start_row not in item_cache:
            item_cache[start_row] = dataset[start_row]
        if end_row not in item_cache:
            item_cache[end_row] = dataset[end_row]

        start_frame_idx = int(ep_frame_indices[local_start])
        end_frame_idx = int(ep_frame_indices[local_end])
        advantage = float(ep_chunk_advantage[local_start]) if np.isfinite(ep_chunk_advantage[local_start]) else float("nan")
        chunk_title = (
            f"Chunk {chunk_idx}: frames {start_frame_idx}-{end_frame_idx} | "
            f"rows {local_start}-{local_end} | advantage={advantage:.4f}"
        )
        draw.text((margin_x, content_y), chunk_title, fill=(35, 35, 35), font=body_font)

        start_y = content_y + chunk_title_h
        end_y = start_y + thumb_h + 28 + row_gap
        for cam_idx, camera_key in enumerate(camera_keys):
            x = margin_x + cam_idx * (thumb_w + 22)
            camera_label = _short_camera_label(camera_key)
            start_thumb = _build_labeled_thumbnail(
                _frame_to_pil_image(item_cache[start_row][camera_key]),
                width=thumb_w,
                height=thumb_h,
                label=f"start | {camera_label} | f={start_frame_idx}",
                accent=(215, 70, 70),
            )
            end_thumb = _build_labeled_thumbnail(
                _frame_to_pil_image(item_cache[end_row][camera_key]),
                width=thumb_w,
                height=thumb_h,
                label=f"end   | {camera_label} | f={end_frame_idx}",
                accent=(70, 105, 215),
            )
            canvas.paste(start_thumb, (x, start_y))
            canvas.paste(end_thumb, (x, end_y))

        note_x = margin_x + len(camera_keys) * (thumb_w + 22)
        note_y = start_y + 4
        draw.text((note_x, note_y), f"C{chunk_idx}", fill=(170, 20, 20), font=_load_font(22))
        draw.text((note_x, note_y + 34), "Top row: chunk start", fill=(90, 90, 90), font=small_font)
        draw.text((note_x, note_y + 56), "Bottom row: chunk end", fill=(90, 90, 90), font=small_font)
        draw.text(
            (note_x, note_y + 96),
            f"start mask={int(ep_chunk_start_indicator[local_start])}",
            fill=(90, 90, 90),
            font=small_font,
        )

        content_y = end_y + thumb_h + 28 + chunk_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def generate_stage_chunk_episode_figures(
    *,
    dataset: LeRobotDataset,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    chunk_advantage: np.ndarray,
    chunk_start_indicator: np.ndarray,
    indicator: np.ndarray,
    chunk_size: int,
    output_dir: Path,
    output_prefix: str,
    requested_camera_keys: str | None = None,
    max_cameras: int = 3,
    overwrite: bool = True,
) -> dict[str, Any]:
    episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    chunk_advantage = np.asarray(chunk_advantage, dtype=np.float32).reshape(-1)
    chunk_start_indicator = np.asarray(chunk_start_indicator, dtype=np.int64).reshape(-1)
    indicator = np.asarray(indicator, dtype=np.int64).reshape(-1)

    camera_keys = select_camera_keys(
        available_camera_keys=list(dataset.meta.camera_keys),
        requested_camera_keys=requested_camera_keys,
        max_cameras=max_cameras,
    )
    verification = verify_chunk_mask_consistency(
        episode_indices=episode_indices,
        chunk_start_indicator=chunk_start_indicator,
        indicator=indicator,
        chunk_size=chunk_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_episodes = sorted(int(v) for v in np.unique(episode_indices))
    pad_width = max(3, len(str(max(unique_episodes) if unique_episodes else 0)))
    figure_tag = output_tag_from_prefix(output_prefix)
    figure_paths: list[str] = []

    for episode_index in unique_episodes:
        positions = np.flatnonzero(episode_indices == episode_index)
        if positions.size == 0:
            continue
        sort_order = np.argsort(frame_indices[positions], kind="stable")
        positions = positions[sort_order]

        output_path = output_dir / f"episode_{episode_index:0{pad_width}d}_{figure_tag}_chunk_gallery.png"
        if not output_path.exists() or overwrite:
            _compose_episode_figure(
                dataset=dataset,
                episode_index=episode_index,
                output_prefix=output_prefix,
                episode_positions=positions,
                ep_frame_indices=frame_indices[positions],
                ep_values=values[positions],
                ep_chunk_advantage=chunk_advantage[positions],
                ep_chunk_start_indicator=chunk_start_indicator[positions],
                ep_indicator=indicator[positions],
                chunk_size=chunk_size,
                camera_keys=camera_keys,
                output_path=output_path,
                episode_verification=verification["per_episode"][str(episode_index)],
            )
        figure_paths.append(str(output_path))

    return {
        "figure_dir": str(output_dir),
        "figure_count": len(figure_paths),
        "figure_paths": figure_paths,
        "camera_keys": camera_keys,
        "verification": verification,
    }
