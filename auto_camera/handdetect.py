"""Find hands in every frame of a video's proxy with two trained models from Google's MediaPipe.

- the hand landmarker: 21 points per hand, when it can see the palm;
- the multiclass segmenter (selfie_multiclass_256x256): per pixel, the
  chance that it is skin. It needs no palm, so it also marks fingers that
  come in from the edge of the picture and the thumb that holds the book.

For each video ``detect`` writes ``handnet/SLUG.npz`` under the cache root
with, per frame, up to ``MAX_HANDS`` hands: 21 landmarks each (x, y as shares
of the frame's width and height, NaN where there is no hand) and the model's
handedness score; and the skin mask (body or face skin more likely than not),
packed to bits, at the proxy's size. Every frame is looked at on its own
(image mode): a hand found in one frame is never carried into the next, so a
hand that has left is not reported. ``auto_camera/handnet.py`` reads what it
writes.

``auto_camera.pipeline.find_hands`` runs this in worker processes of its
own, and it can be run in a separate environment that holds MediaPipe
alone. So nothing of the rest of the package
is imported here (``auto_camera.paths`` is standard library only), NumPy is
the one import at load time, and OpenCV and MediaPipe are imported when the
models are run.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import types
from concurrent.futures import Executor
from pathlib import Path

import numpy as np

from .paths import CACHE_ROOT, MODELS_DIR

PROXY_DIR = CACHE_ROOT / "proxies"
OUT_DIR = CACHE_ROOT / "handnet"
MODEL = MODELS_DIR / "hand_landmarker.task"
SEGMENTER = MODELS_DIR / "selfie_multiclass_256x256.tflite"
SKIN_LEVEL = 0.5         # body skin and face skin together more likely than this
MAX_HANDS = 4
MIN_CONFIDENCE = 0.3     # detection and presence; low, so that a hand half out of the picture is still found
VERSION = 2          # 2: skin masks


def _stub_matplotlib() -> None:
    """Let ``import mediapipe`` work where Matplotlib is absent (the Windows app leaves it out).

    MediaPipe's drawing helpers import ``matplotlib.pyplot`` when the package
    is imported, and nothing used here draws. Where Matplotlib is installed it
    is left alone.
    """
    if "matplotlib" in sys.modules:
        return
    try:
        if importlib.util.find_spec("matplotlib") is not None:
            return
    except (ImportError, ValueError):
        pass
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot  # type: ignore[attr-defined]
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot


def import_mediapipe():
    """``mediapipe``, imported; raises ImportError when it is not installed."""
    _stub_matplotlib()
    import mediapipe

    return mediapipe


def mediapipe_available() -> bool:
    try:
        import_mediapipe()
    except Exception:      # a broken install raises more than ImportError
        return False
    return True


def quiet_worker() -> None:
    """For a worker process: send the models' own log lines (TensorFlow Lite, glog) to nowhere.

    When this ran as a separate program its output was captured and dropped;
    run in worker processes, the same lines would otherwise reach whoever
    reads the pipeline's standard error. Errors still come back as exceptions.
    """
    os.environ.setdefault("GLOG_minloglevel", "2")
    try:
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
        os.close(null)
    except OSError:
        pass


def probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip().split(",")
    num, den = out[2].split("/")
    return int(out[0]), int(out[1]), float(num) / float(den)


def detect_range(slug: str, start: int, stop: int, proxy_dir: Path = PROXY_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """The models on frames ``start`` to ``stop - 1`` of a proxy (``stop`` 0: to the end)."""
    import cv2

    mp = import_mediapipe()
    from mediapipe.tasks.python import BaseOptions, vision

    proxy = proxy_dir / f"{slug}.mp4"
    width, height, fps = probe(proxy)
    options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.IMAGE,
                                           num_hands=MAX_HANDS, min_hand_detection_confidence=MIN_CONFIDENCE,
                                           min_hand_presence_confidence=MIN_CONFIDENCE, min_tracking_confidence=MIN_CONFIDENCE)
    detector = vision.HandLandmarker.create_from_options(options)
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(SEGMENTER)), running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True, output_category_mask=False))
    select = [] if start == 0 and stop == 0 else ["-vf", f"select=between(n\\,{start}\\,{(stop or 10**9) - 1})", "-fps_mode", "passthrough"]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(proxy), *select, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height * 3
    landmarks: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    skins: list[np.ndarray] = []
    try:
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = detector.detect(image)
            masks = segmenter.segment(image).confidence_masks
            skin = np.maximum(np.squeeze(masks[2].numpy_view()), np.squeeze(masks[3].numpy_view())).astype(np.float32)
            if skin.shape != (height, width):
                skin = cv2.resize(skin, (width, height), interpolation=cv2.INTER_LINEAR)
            skins.append(np.packbits(skin > SKIN_LEVEL))
            points = np.full((MAX_HANDS, 21, 2), np.nan, dtype=np.float32)
            score = np.zeros(MAX_HANDS, dtype=np.float32)
            for index, (hand, handed) in enumerate(zip(result.hand_landmarks[:MAX_HANDS], result.handedness[:MAX_HANDS])):
                points[index] = [[p.x, p.y] for p in hand]
                score[index] = handed[0].score
            landmarks.append(points)
            scores.append(score)
    finally:
        proc.stdout.close()
        proc.wait()
        detector.close()
        segmenter.close()
    return np.array(landmarks, dtype=np.float32), np.array(scores, dtype=np.float32), np.array(skins, dtype=np.uint8), width, height, fps


def _range_job(args: tuple[str, int, int, Path]):
    return detect_range(*args)


def frame_count(slug: str, proxy_dir: Path = PROXY_DIR) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                          "-of", "csv=p=0", str(proxy_dir / f"{slug}.mp4")], capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def save(path: Path, landmarks: np.ndarray, scores: np.ndarray, skins: np.ndarray, width: int, height: int, fps: float) -> None:
    """Write one video's detections as ``handnet.load`` reads them (written beside, then moved into place)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial.npz")
    np.savez_compressed(tmp, landmarks=landmarks.astype(np.float16), scores=scores.astype(np.float16),
                        skin=skins, width=width, height=height, fps=fps, version=VERSION)
    tmp.replace(path)


def detect(slug: str, proxy_dir: Path = PROXY_DIR, out_dir: Path = OUT_DIR, parts: int = 1, pool: Executor | None = None) -> str:
    """The models on every frame of one proxy, written to ``out_dir/SLUG.npz``.

    With a ``pool`` the work is done there, split into ``parts`` stretches run
    side by side; without, here in this process.
    """
    started = time.time()
    if pool is not None and parts > 1:
        total = frame_count(slug, proxy_dir)
        edges = np.linspace(0, total, parts + 1).round().astype(int)
        jobs = [(slug, int(a), int(b), proxy_dir) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        pieces = list(pool.map(_range_job, jobs))
    elif pool is not None:
        pieces = [pool.submit(_range_job, (slug, 0, 0, proxy_dir)).result()]
    else:
        pieces = [detect_range(slug, 0, 0, proxy_dir)]
    width, height, fps = pieces[0][3], pieces[0][4], pieces[0][5]
    landmarks = np.concatenate([p[0] for p in pieces if len(p[0])])
    scores = np.concatenate([p[1] for p in pieces if len(p[1])])
    skins = np.concatenate([p[2] for p in pieces if len(p[2])])
    save(out_dir / f"{slug}.npz", landmarks, scores, skins, width, height, fps)
    seen = int((scores > 0).any(axis=1).sum())
    return f"{slug[:50]:50s} {len(scores):6d} frames, hands in {seen:6d}, {time.time() - started:6.1f}s"
