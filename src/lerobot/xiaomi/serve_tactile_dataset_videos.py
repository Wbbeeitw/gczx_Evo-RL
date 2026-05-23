#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _ensure_repo_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


_ensure_repo_src_on_path()

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


@dataclass
class EpisodeRecord:
    episode_index: int
    left_video_relpath: str
    right_video_relpath: str
    left_start_s: float
    left_end_s: float
    right_start_s: float
    right_end_s: float
    duration_s: float
    episode_success: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a browser page to inspect recorded left/right tactile MP4 videos from a LeRobot dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the recorded LeRobot dataset root.")
    parser.add_argument(
        "--left-key",
        default="observation.images.left_tactile",
        help="Dataset video key for the left tactile stream.",
    )
    parser.add_argument(
        "--right-key",
        default="observation.images.right_tactile",
        help="Dataset video key for the right tactile stream.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address.")
    parser.add_argument("--http-port", type=int, default=8767, help="HTTP port for the viewer page.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


class DatasetVideoIndex:
    def __init__(self, dataset_root: Path, *, left_key: str, right_key: str) -> None:
        self.dataset_root = dataset_root.resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        self.meta = LeRobotDatasetMetadata(repo_id=self.dataset_root.name, root=self.dataset_root)
        self.left_key = left_key
        self.right_key = right_key

        missing = [key for key in (left_key, right_key) if key not in self.meta.video_keys]
        if missing:
            raise KeyError(
                "Missing tactile video key(s) in dataset: "
                f"{missing}. Available video keys: {sorted(self.meta.video_keys)}"
            )

        self.episodes = self._build_episode_records()

    def _build_episode_records(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        total_episodes = len(self.meta.episodes)
        for episode_index in range(total_episodes):
            episode = self.meta.episodes[episode_index]
            left_relpath = str(self.meta.get_video_file_path(episode_index, self.left_key))
            right_relpath = str(self.meta.get_video_file_path(episode_index, self.right_key))

            left_start_s = float(_scalar(episode[f"videos/{self.left_key}/from_timestamp"]))
            left_end_s = float(_scalar(episode[f"videos/{self.left_key}/to_timestamp"]))
            right_start_s = float(_scalar(episode[f"videos/{self.right_key}/from_timestamp"]))
            right_end_s = float(_scalar(episode[f"videos/{self.right_key}/to_timestamp"]))
            duration_s = max(left_end_s - left_start_s, right_end_s - right_start_s, 0.0)

            episode_success = None
            if "episode_success" in episode:
                raw_label = _scalar(episode["episode_success"])
                episode_success = None if raw_label is None else str(raw_label)

            records.append(
                EpisodeRecord(
                    episode_index=episode_index,
                    left_video_relpath=left_relpath,
                    right_video_relpath=right_relpath,
                    left_start_s=left_start_s,
                    left_end_s=left_end_s,
                    right_start_s=right_start_s,
                    right_end_s=right_end_s,
                    duration_s=duration_s,
                    episode_success=episode_success,
                )
            )

        return records

    def get_episode(self, episode_index: int) -> EpisodeRecord:
        if episode_index < 0 or episode_index >= len(self.episodes):
            raise IndexError(f"Episode index {episode_index} out of range: 0 <= idx < {len(self.episodes)}")
        return self.episodes[episode_index]

    def resolve_video_path(self, episode_index: int, side: str) -> Path:
        episode = self.get_episode(episode_index)
        relpath = episode.left_video_relpath if side == "left" else episode.right_video_relpath
        path = (self.dataset_root / relpath).resolve()
        if self.dataset_root not in path.parents and path != self.dataset_root:
            raise ValueError(f"Resolved video path escapes dataset root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        return path

    def to_payload(self) -> dict:
        return {
            "dataset_root": str(self.dataset_root),
            "left_key": self.left_key,
            "right_key": self.right_key,
            "episodes": [asdict(record) for record in self.episodes],
        }


def _read_http_range(range_header: str | None, file_size: int) -> tuple[int, int] | None:
    if not range_header or not range_header.startswith("bytes="):
        return None

    range_spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
    if "-" not in range_spec:
        return None

    start_text, end_text = range_spec.split("-", 1)
    if start_text == "":
        suffix_length = int(end_text)
        if suffix_length <= 0:
            return None
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
        return start, end

    start = int(start_text)
    end = file_size - 1 if end_text == "" else int(end_text)
    if start < 0 or end < start:
        return None
    return start, min(end, file_size - 1)


def make_handler(index: DatasetVideoIndex, logger: logging.Logger) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            logger.info("viewer %s", fmt % args)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._serve_index()
                return
            if parsed.path == "/episodes.json":
                self._serve_episodes()
                return
            if parsed.path == "/video":
                self._serve_video(parsed.query)
                return
            if parsed.path == "/healthz":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"ok")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def _serve_index(self) -> None:
            html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tactile Dataset Viewer</title>
  <style>
    body {
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #0f1419;
      color: #e6edf3;
      min-height: 100vh;
    }
    .wrap {
      width: min(1200px, 96vw);
      margin: 24px auto 40px;
    }
    .topbar {
      display: grid;
      gap: 12px;
      margin-bottom: 18px;
    }
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    select, button {
      background: #161b22;
      color: #e6edf3;
      border: 1px solid #30363d;
      padding: 8px 10px;
      border-radius: 8px;
      font: inherit;
    }
    .meta {
      opacity: 0.85;
      line-height: 1.7;
      white-space: pre-wrap;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 16px;
    }
    .panel {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 14px;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 16px;
    }
    video {
      width: 100%;
      background: #000;
      border-radius: 8px;
    }
    .hint {
      margin-top: 10px;
      opacity: 0.75;
      line-height: 1.6;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="controls">
        <label for="episodeSelect">Episode</label>
        <select id="episodeSelect"></select>
        <button id="restartBtn" type="button">Restart</button>
        <button id="playBtn" type="button">Play Both</button>
        <button id="pauseBtn" type="button">Pause Both</button>
      </div>
      <div id="meta" class="meta">Loading episode list...</div>
    </div>

    <div class="grid">
      <div class="panel">
        <h2>Left tactile</h2>
        <video id="leftVideo" controls preload="metadata" playsinline></video>
      </div>
      <div class="panel">
        <h2>Right tactile</h2>
        <video id="rightVideo" controls preload="metadata" playsinline></video>
      </div>
    </div>

    <div class="hint">
      If multiple episodes share the same MP4 file, the page will auto-seek to the selected episode start timestamp and stop at the selected end timestamp.
    </div>
  </div>

  <script>
    const state = {
      payload: null,
      currentEpisode: null,
    };

    const selectEl = document.getElementById("episodeSelect");
    const metaEl = document.getElementById("meta");
    const leftVideo = document.getElementById("leftVideo");
    const rightVideo = document.getElementById("rightVideo");
    const restartBtn = document.getElementById("restartBtn");
    const playBtn = document.getElementById("playBtn");
    const pauseBtn = document.getElementById("pauseBtn");

    function episodeLabel(ep) {
      const status = ep.episode_success ? ` | ${ep.episode_success}` : "";
      return `episode ${ep.episode_index}${status} | ${ep.duration_s.toFixed(2)}s`;
    }

    function installBounds(videoEl, startS, endS) {
      const applyBounds = () => {
        const clampedStart = Math.max(0, startS);
        if (!Number.isFinite(videoEl.duration) || clampedStart > videoEl.duration) {
          return;
        }
        if (Math.abs(videoEl.currentTime - clampedStart) > 0.35) {
          videoEl.currentTime = clampedStart;
        }
      };

      videoEl.onloadedmetadata = applyBounds;
      videoEl.ontimeupdate = () => {
        if (videoEl.currentTime >= endS) {
          videoEl.pause();
        }
      };
    }

    function loadEpisode(ep) {
      state.currentEpisode = ep;
      metaEl.textContent =
        `dataset: ${state.payload.dataset_root}\\n` +
        `episode: ${ep.episode_index}\\n` +
        `duration: ${ep.duration_s.toFixed(2)}s\\n` +
        `label: ${ep.episode_success ?? "unlabeled"}\\n` +
        `left: ${ep.left_video_relpath} [${ep.left_start_s.toFixed(2)}, ${ep.left_end_s.toFixed(2)}]\\n` +
        `right: ${ep.right_video_relpath} [${ep.right_start_s.toFixed(2)}, ${ep.right_end_s.toFixed(2)}]`;

      leftVideo.src = `/video?episode=${ep.episode_index}&side=left`;
      rightVideo.src = `/video?episode=${ep.episode_index}&side=right`;
      installBounds(leftVideo, ep.left_start_s, ep.left_end_s);
      installBounds(rightVideo, ep.right_start_s, ep.right_end_s);
      leftVideo.load();
      rightVideo.load();
    }

    async function init() {
      const response = await fetch("/episodes.json", { cache: "no-store" });
      const payload = await response.json();
      state.payload = payload;

      selectEl.innerHTML = "";
      payload.episodes.forEach((ep) => {
        const option = document.createElement("option");
        option.value = String(ep.episode_index);
        option.textContent = episodeLabel(ep);
        selectEl.appendChild(option);
      });

      if (payload.episodes.length === 0) {
        metaEl.textContent = "No episodes found in dataset.";
        return;
      }

      selectEl.addEventListener("change", () => {
        const ep = payload.episodes.find((item) => item.episode_index === Number(selectEl.value));
        if (ep) {
          loadEpisode(ep);
        }
      });

      restartBtn.addEventListener("click", () => {
        if (!state.currentEpisode) {
          return;
        }
        loadEpisode(state.currentEpisode);
      });

      playBtn.addEventListener("click", async () => {
        if (!state.currentEpisode) {
          return;
        }
        leftVideo.currentTime = state.currentEpisode.left_start_s;
        rightVideo.currentTime = state.currentEpisode.right_start_s;
        await Promise.allSettled([leftVideo.play(), rightVideo.play()]);
      });

      pauseBtn.addEventListener("click", () => {
        leftVideo.pause();
        rightVideo.pause();
      });

      selectEl.value = String(payload.episodes[0].episode_index);
      loadEpisode(payload.episodes[0]);
    }

    init().catch((error) => {
      metaEl.textContent = `Failed to load dataset viewer: ${error}`;
      console.error(error);
    });
  </script>
</body>
</html>
"""
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_episodes(self) -> None:
            body = json.dumps(index.to_payload(), ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_video(self, query: str) -> None:
            params = parse_qs(query)
            try:
                episode_index = int(params["episode"][0])
                side = params["side"][0].strip().lower()
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Invalid query parameters: {exc}")
                return

            if side not in {"left", "right"}:
                self.send_error(HTTPStatus.BAD_REQUEST, "side must be 'left' or 'right'")
                return

            try:
                path = index.resolve_video_path(episode_index, side)
            except Exception as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                return

            try:
                self._send_file(path)
            except Exception as exc:
                logger.exception("Failed to serve %s", path)
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def _send_file(self, path: Path) -> None:
            file_size = path.stat().st_size
            range_bounds = _read_http_range(self.headers.get("Range"), file_size)
            start = 0
            end = file_size - 1
            status = HTTPStatus.OK
            if range_bounds is not None:
                start, end = range_bounds
                status = HTTPStatus.PARTIAL_CONTENT

            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("tactile-dataset-viewer")

    index = DatasetVideoIndex(
        args.dataset_root,
        left_key=args.left_key,
        right_key=args.right_key,
    )
    logger.info("Loaded %d episode(s) from %s", len(index.episodes), index.dataset_root)
    if index.episodes:
        logger.info("First available episode: %d", index.episodes[0].episode_index)

    server = ThreadingHTTPServer((args.host, args.http_port), make_handler(index, logger))
    logger.info("Tactile dataset viewer available at http://%s:%d", args.host, args.http_port)
    logger.info("If running on a remote server, use SSH port forwarding before opening the page in your browser.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping tactile dataset viewer")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
