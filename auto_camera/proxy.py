"""Low-resolution proxy copies of the source videos.

A proxy keeps every source frame, one for one, at 480x270. Motion analysis,
labelling and model training read proxies instead of the 3072x1728 originals,
which turns minutes of decoding into seconds. Frame ``n`` of a proxy is frame
``n`` of its source.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import CACHE_ROOT

PROXY_WIDTH = 480
PROXY_HEIGHT = 270
VIDEO_SUFFIXES = {".mov", ".m4v", ".mp4"}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = Path.home() / "Videos" / "auto-camera-data"
DEFAULT_PROXY_DIR = CACHE_ROOT / "proxies"


def slugify(name: str) -> str:
    """Stable id for a video, from its file name without the extension."""
    return re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")


def list_videos(video_dir: Path = DEFAULT_VIDEO_DIR) -> list[Path]:
    return sorted(
        (p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda p: p.name.lower(),
    )


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=int(num) / int(den),
        frames=int(stream["nb_frames"]),
        duration=float(data["format"]["duration"]),
    )


def proxy_path(video: Path, proxy_dir: Path = DEFAULT_PROXY_DIR) -> Path:
    return proxy_dir / f"{slugify(video.name)}.mp4"


def _taps(source: int, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each target sample along one axis, the two nearest source samples and the weight of the second.

    This is how FFmpeg's ``scale_cuda`` shrinks with its default interpolation
    (bicubic with A = 0): the two source samples around the target's centre,
    weighted ``1 - s`` and ``s`` with ``s = 3f^2 - 2f^3``, and no filtering of
    what lies between them (measured against ``scale_cuda`` output while
    porting the pipeline to the browser, ``web/src/video/sampler.ts``).
    """
    x = (np.arange(target) + 0.5) * source / target - 0.5
    x0 = np.floor(x)
    f = x - x0
    s = 3 * f * f - 2 * f * f * f
    x0 = x0.astype(np.int64)
    return np.clip(x0, 0, source - 1), np.clip(x0 + 1, 0, source - 1), s


def shrink_plane(plane: np.ndarray, width: int, height: int) -> np.ndarray:
    """One 8-bit image plane shrunk to ``width`` x ``height`` as ``scale_cuda`` does, the result truncated."""
    y0, y1, sy = _taps(plane.shape[0], height)
    x0, x1, sx = _taps(plane.shape[1], width)
    rows = plane[y0].astype(np.float32) * (1 - sy)[:, None].astype(np.float32) + plane[y1].astype(np.float32) * sy[:, None].astype(np.float32)
    out = rows[:, x0] * (1 - sx).astype(np.float32) + rows[:, x1] * sx.astype(np.float32)
    return np.floor(np.clip(out, 0, 255) + 1e-4).astype(np.uint8)


def _build_proxy_cpu(video: Path, encode: list[str], threads: int) -> None:
    """The proxy without a GPU, sampled the way ``scale_cuda`` samples.

    FFmpeg's own scalers average over the source pixels a target pixel covers;
    ``scale_cuda`` takes the two nearest per axis. Print stays sharper that
    way, and everything here was measured on such proxies: an area-averaged
    proxy gave *Einar* 119 page landings instead of 125. So the frames are
    decoded at full size and shrunk here, plane by plane, and handed to the
    same encoder.
    """
    info = probe(video)
    width, height = info.width, info.height
    num, den = (subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                                "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout.strip().split("/"))
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", str(threads), "-i", str(video), "-an",
                                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{PROXY_WIDTH}x{PROXY_HEIGHT}",
                                "-r", f"{num}/{den}", "-i", "-", *encode], stdin=subprocess.PIPE)
    assert decoder.stdout is not None and encoder.stdin is not None
    luma = width * height
    chroma = ((width + 1) // 2) * ((height + 1) // 2)
    try:
        while True:
            raw = decoder.stdout.read(luma + 2 * chroma)
            if len(raw) < luma + 2 * chroma:
                break
            frame = np.frombuffer(raw, dtype=np.uint8)
            y = frame[:luma].reshape(height, width)
            u = frame[luma:luma + chroma].reshape((height + 1) // 2, (width + 1) // 2)
            v = frame[luma + chroma:].reshape((height + 1) // 2, (width + 1) // 2)
            encoder.stdin.write(shrink_plane(y, PROXY_WIDTH, PROXY_HEIGHT).tobytes())
            encoder.stdin.write(shrink_plane(u, PROXY_WIDTH // 2, PROXY_HEIGHT // 2).tobytes())
            encoder.stdin.write(shrink_plane(v, PROXY_WIDTH // 2, PROXY_HEIGHT // 2).tobytes())
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()
    if decoder.returncode or encoder.returncode:
        raise RuntimeError(f"proxy of {video.name}: ffmpeg failed (decoder {decoder.returncode}, encoder {encoder.returncode})")


def build_proxy(video: Path, out: Path, *, use_gpu: bool = True, threads: int = 2) -> None:
    """Write a 480x270 proxy with the same number of frames as ``video``.

    With ``use_gpu`` FFmpeg decodes and shrinks on an NVIDIA card
    (``scale_cuda``); without, the same sampling is done on the CPU
    (``_build_proxy_cpu``), so both give the same proxy to within a level.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.mp4")
    encode = [
        "-an", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-g", "11", "-bf", "0", "-threads", str(threads), str(tmp),
    ]
    if use_gpu:
        subprocess.run([
            "ffmpeg", "-v", "error", "-y",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(video),
            "-vf", f"scale_cuda={PROXY_WIDTH}:{PROXY_HEIGHT},hwdownload,format=nv12",
            *encode,
        ], check=True)
    else:
        _build_proxy_cpu(video, encode, threads)
    tmp.replace(out)


def read_proxy(path: Path, *, gray: bool = False) -> np.ndarray:
    """Decode a whole proxy into memory.

    Returns ``(frames, 270, 480)`` uint8 when ``gray`` else
    ``(frames, 270, 480, 3)`` uint8 in BGR order. A 28-minute video is about
    7 GB in colour and 2.4 GB in gray, so prefer ``gray`` or ``iter_proxy`` for
    long videos.
    """
    pix_fmt = "gray" if gray else "bgr24"
    channels = 1 if gray else 3
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", "2", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-threads", "2", "-"],
        capture_output=True, check=True,
    ).stdout
    frame_bytes = PROXY_WIDTH * PROXY_HEIGHT * channels
    count = len(raw) // frame_bytes
    shape = (count, PROXY_HEIGHT, PROXY_WIDTH) if gray else (count, PROXY_HEIGHT, PROXY_WIDTH, 3)
    return np.frombuffer(raw, dtype=np.uint8, count=count * frame_bytes).reshape(shape)


def iter_proxy(path: Path, *, gray: bool = False, chunk_frames: int = 256):
    """Yield ``(first_index, frames)`` chunks of a proxy without holding it all."""
    pix_fmt = "gray" if gray else "bgr24"
    channels = 1 if gray else 3
    frame_bytes = PROXY_WIDTH * PROXY_HEIGHT * channels
    # stderr is dropped: a caller that stops reading early makes ffmpeg report a
    # broken pipe, which is expected and only noise.
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-threads", "2", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-threads", "2", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    index = 0
    try:
        while True:
            raw = proc.stdout.read(frame_bytes * chunk_frames)
            count = len(raw) // frame_bytes
            if count == 0:
                break
            shape = (count, PROXY_HEIGHT, PROXY_WIDTH) if gray else (count, PROXY_HEIGHT, PROXY_WIDTH, 3)
            yield index, np.frombuffer(raw, dtype=np.uint8, count=count * frame_bytes).reshape(shape)
            index += count
    finally:
        proc.stdout.close()
        proc.wait()


def find_video(slug: str, video_dir: Path = DEFAULT_VIDEO_DIR) -> Path:
    """The source video whose slug is ``slug``."""
    for video in list_videos(video_dir):
        if slugify(video.name) == slug:
            return video
    raise FileNotFoundError(f"no source video with slug {slug!r} in {video_dir}")


def extract_source_frame(video: Path, frame: int, fps: float, *, width: int | None = None, threads: int = 2) -> np.ndarray:
    """Decode one frame of a source video, by frame index, as BGR.

    Seeks to just before the frame's timestamp so the first decoded frame at or
    after it is the one asked for. ``width`` downsizes, keeping aspect ratio.
    """
    info = probe(video)
    out_w = width or info.width
    out_h = int(round(info.height * out_w / info.width / 2)) * 2
    seek = max(0.0, (frame - 0.3) / fps)
    cmd = [
        "ffmpeg", "-v", "error", "-threads", str(threads), "-ss", f"{seek:.6f}", "-i", str(video),
        "-frames:v", "1", "-vf", f"scale={out_w}:{out_h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3).copy()
