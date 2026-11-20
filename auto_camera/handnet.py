"""Hands found by trained models (``handdetect.py``), as masks and as cover of the print.

Two models from Google's MediaPipe are run on every frame of the proxy:

- a skin segmenter, which marks per pixel whatever is skin. It needs no
  palm, so it also marks fingers that come in from the edge of the picture and
  the thumb that holds the book. This is what is measured.
- a hand landmarker, which gives 21 points per hand when it sees the palm.
  It decides which patches of skin are hands (see ``Detections.mask``); its
  points are not drawn, because it also places fingers it cannot see.

The colour model in ``hands.py`` sees a hand only where its colour over the
page matches its colour over the background, and that fails under the light
of a lamp or from the side. These models know what a hand looks like.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .paths import CACHE_ROOT

DEFAULT_HANDNET_DIR = CACHE_ROOT / "handnet"
FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
PALM = (0, 1, 5, 9, 13, 17)
FINGER_WIDTH = 0.24      # a finger is this wide, as a share of the length of the palm (wrist to the middle finger's knuckle): about 2 cm to 8.5 cm
THUMB_WIDTH = 0.28       # the thumb a little wider
PALM_MARGIN = 0.10       # and the palm reaches this much beyond the polygon of its joints
MIN_SCORE = 0.5          # handedness score under which a landmark detection is not trusted
MIN_SKIN = 12            # proxy pixels; smaller patches of skin are ignored
SKIN_OPEN = 5            # proxy pixels: skin thinner than this is not a finger (the segmenter bleeds onto paper between lines of print)


def silhouette(hands: np.ndarray, scores: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bool image of what the landmark hands of one frame cover. ``hands`` is (H, 21, 2) as shares of width and height, NaN where no hand."""
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    for points, score in zip(hands, scores):
        if score < MIN_SCORE or not np.isfinite(points).all():
            continue
        xy = points * np.array([width, height], dtype=np.float32)
        palm_length = float(np.linalg.norm(xy[9] - xy[0]))
        if palm_length < 2:
            continue
        palm = cv2.convexHull(xy[list(PALM)].round().astype(np.int32))
        cv2.fillConvexPoly(mask, palm, 1)
        grow = max(1, int(round(PALM_MARGIN * palm_length)))
        cv2.polylines(mask, [palm], True, 1, thickness=2 * grow)
        for finger in FINGERS:
            band = (THUMB_WIDTH if finger[0] == 1 else FINGER_WIDTH) * palm_length
            joints = xy[list(finger)]
            # The last landmark is the fingertip itself: the band is drawn as
            # flat-ended segments, so it does not run on past the tip.
            for a, b in zip(joints[:-1], joints[1:]):
                direction = b - a
                length = float(np.linalg.norm(direction))
                if length < 1e-3:
                    continue
                side = np.array([-direction[1], direction[0]]) / length * band / 2
                quad = np.array([a + side, b + side, b - side, a - side]).round().astype(np.int32)
                cv2.fillConvexPoly(mask, quad, 1)
                cv2.circle(mask, tuple(int(v) for v in a.round()), max(1, int(round(band / 2))), 1, -1)
    return mask.astype(bool)


hand_mask = silhouette      # the name used before the skin segmenter was added


def fine_detail(gray: np.ndarray) -> np.ndarray:
    """Fine-detail energy around each pixel that runs both ways, across and down: high on print and pictures.

    Letters have strokes in every direction. The stack of page edges beside
    the print is all vertical lines, the edge of the book block above and
    below it all horizontal ones; each is as fine as print one way and flat
    the other, so the weaker of the two directions leaves them out. Skin is
    smooth both ways.
    """
    wide = gray.astype(np.int16)
    across = np.zeros(gray.shape, dtype=np.float32)
    down = np.zeros(gray.shape, dtype=np.float32)
    across[:, 1:] = np.abs(wide[:, 1:] - wide[:, :-1])
    down[1:, :] = np.abs(wide[1:, :] - wide[:-1, :])
    return np.minimum(cv2.blur(across, (5, 5)), cv2.blur(down, (5, 5))).round().clip(0, 255).astype(np.uint8)


class Detections:
    """What the models found in one video, frame by frame."""

    def __init__(self, landmarks: np.ndarray, scores: np.ndarray, skin: np.ndarray | None, shape: tuple[int, int]):
        self.landmarks = landmarks
        self.scores = scores
        self.skin_bits = skin
        self.shape = shape

    def __len__(self) -> int:
        return len(self.scores)

    def skin(self, index: int) -> np.ndarray:
        if self.skin_bits is None:
            return np.zeros(self.shape, dtype=bool)
        height, width = self.shape
        return np.unpackbits(self.skin_bits[index])[:height * width].reshape(height, width).astype(bool)

    def mask(self, index: int, region: np.ndarray | None = None) -> np.ndarray:
        """Bool image of what hands cover in frame ``index``.

        Skin, in patches that reach the edge of the picture (a hand is on an
        arm, and the arm goes out of the picture) or that hold the palm of a
        hand the landmarker found. A patch of skin colour wholly inside the
        picture with no palm in it is printed or lit that way: a sepia
        portrait, warm paper under a lamp.

        The landmarks are not drawn themselves: the model places the fingers
        of the hand that holds the book even where they are behind the page,
        and drawn they lie across the print. ``region`` is not used; it is
        kept so that callers need not change.
        """
        if self.skin_bits is None:
            return silhouette(self.landmarks[index], self.scores[index], self.shape)
        skin = self.skin(index).astype(np.uint8)
        if not skin.any():
            return skin.astype(bool)
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SKIN_OPEN, SKIN_OPEN)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
        keep = np.zeros(count, dtype=bool)
        edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        keep[np.unique(edge)] = True
        palms = self.palms(index)
        if palms.any():
            keep[np.unique(labels[palms])] = True
        keep[0] = False
        keep[1:] &= stats[1:, cv2.CC_STAT_AREA] >= MIN_SKIN
        return keep[labels]

    def palms(self, index: int) -> np.ndarray:
        """Bool image of the palms of the hands the landmarker found in frame ``index``."""
        height, width = self.shape
        out = np.zeros(self.shape, dtype=np.uint8)
        for points, score in zip(self.landmarks[index], self.scores[index]):
            if score < MIN_SCORE or not np.isfinite(points).all():
                continue
            xy = (points[list(PALM)] * np.array([width, height], dtype=np.float32)).round().astype(np.int32)
            cv2.fillConvexPoly(out, cv2.convexHull(xy), 1)
        return out.astype(bool)


def load(slug: str, cache_dir: Path = DEFAULT_HANDNET_DIR) -> Detections | None:
    """What ``handdetect.detect`` found in this video; None if it was never run."""
    path = cache_dir / f"{slug}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        skin = data["skin"] if "skin" in data else None
        return Detections(data["landmarks"].astype(np.float32), data["scores"].astype(np.float32), skin,
                          (int(data["height"]), int(data["width"])))


def cover(detections: Detections | None, where: np.ndarray, region: np.ndarray, frames: int) -> np.ndarray | None:
    """Per frame, the share of ``where`` (the print) that hands cover, without looking at the frames; None without detections.

    ``hands.measure`` does the same with each frame's print detail as well.
    """
    if detections is None or not where.any():
        return None
    area = float(where.sum())
    out = np.zeros(frames, dtype=np.float32)
    for index in range(min(frames, len(detections))):
        out[index] = float((detections.mask(index, region) & where).sum()) / area
    return out


def key(slug: str, cache_dir: Path = DEFAULT_HANDNET_DIR) -> str:
    """Changes when the detections do, for caches built from them."""
    path = cache_dir / f"{slug}.npz"
    return f"-net{int(path.stat().st_mtime)}" if path.exists() else ""
