"""Read the digits printed in the header and footer bands of a still.

Used only by the page-number audit. The OCR engine (rapidocr-onnxruntime) lives
in its own environment, ``.venv-ocr``, and is imported on first use.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

NUMBER = re.compile(r"\d{1,4}")
MIN_SCORE = 0.6
DET_MAX_SIDE = 1600       # the text detector sees each band at most this wide
MAX_ASPECT = 4.5          # a box wider than this many times its height is a line of text, not a page number
END_ASPECT = 4.0          # how much of each end of a head or foot line is read, in line heights
MIN_HEIGHT = 8            # pixels
BAND_OUT = 0.12           # how far a band reaches beyond the print into the margin, as a share of the page height
BAND_IN = 0.07            # and how far into the print
LEADING = re.compile(r"^(\d{1,4})(?=\D)")     # digits that open a line of text
TRAILING = re.compile(r"(?<=\D)(\d{1,4})$")   # digits that close one
_detector = None
_reader = None


def init_engine(threads: int = 1) -> None:
    """Start the detector and the recogniser, each limited to ``threads``.

    Detection runs on a downsized band and only short boxes are recognised.
    Recognising every line in the band took six times as long for the same
    digits.
    """
    global _detector, _reader
    from rapidocr_onnxruntime import RapidOCR
    _detector = RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1, det_limit_side_len=DET_MAX_SIDE, det_limit_type="max")
    _reader = RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1)


def bands(image: np.ndarray, box: list[float], which: str = "both", rows: list[float] | None = None) -> dict[str, np.ndarray]:
    """Crops of the strips where page numbers sit: around the first and the last line of print.

    ``rows`` is where the head and the foot of the print may lie in this still
    (fractions of the frame height: ``[top_from, top_to, bottom_from,
    bottom_to]`` from ``export.band_rows``, or just ``[top, bottom]``). A page number is printed in
    the head or foot line or just outside it, so each strip reaches a little
    into the text and further out into the margin. Without ``rows`` the strips
    are taken at the top and bottom of the page box.
    """
    x0, x1, y0, y1 = box
    h, w = image.shape[:2]
    xa, xb = int(max(0.0, x0 - 0.02) * w), int(min(1.0, x1 + 0.02) * w)
    height = y1 - y0
    if rows is None:
        top = (y0 - 0.03, y0 + 0.17 * height)
        bottom = (y1 - 0.14 * height, y1 + 0.03)
    else:
        top_from, top_to, bottom_from, bottom_to = rows if len(rows) == 4 else (rows[0], rows[0], rows[1], rows[1])
        top = (top_from - BAND_OUT * height, top_to + BAND_IN * height)
        bottom = (bottom_from - BAND_IN * height, bottom_to + BAND_OUT * height)
    out = {}
    if which in ("top", "both"):
        out["top"] = image[int(max(0.0, top[0]) * h):int(min(1.0, top[1]) * h), xa:xb]
    if which in ("bottom", "both"):
        out["bottom"] = image[int(max(0.0, bottom[0]) * h):int(min(1.0, bottom[1]) * h), xa:xb]
    return out


def _recognise(crop: np.ndarray) -> tuple[str, float]:
    if crop.size == 0:
        return "", 0.0
    read = _reader(crop, use_det=False, use_cls=False, use_rec=True)
    if not read or not read[0]:
        return "", 0.0
    return str(read[0][0][0]).strip().replace(" ", ""), float(read[0][0][1])


def read_still(path: str | Path, box: list[float], which: str = "both", rows: list[float] | None = None) -> list[dict]:
    """Every run of one to four digits found in the bands of one still.

    A short box is read whole and must be all digits. A long box is a line of
    text; only the head line and the foot line of each page are looked at, and
    only their two ends, where a number printed beside the running head sits
    ("46  REFORMATION THOUGHT", "THE NEW PHILOSOPHY  47").
    """
    if _detector is None:
        init_engine()
    image = cv2.imread(str(path))
    tokens: list[dict] = []
    if image is None:
        return tokens
    for name, band in bands(image, box, which, rows).items():
        if band.size == 0:
            continue
        found = _detector(band, use_det=True, use_cls=False, use_rec=False)
        boxes = found[0] if found and found[0] is not None else []
        lines: list[tuple[float, float, float, float]] = []
        for points in boxes:
            points = np.asarray(points, dtype=np.float32)
            x0, y0, x1, y1 = points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()
            width, height = x1 - x0, y1 - y0
            if height < MIN_HEIGHT:
                continue
            if width > MAX_ASPECT * height:
                lines.append((float(x0), float(y0), float(x1), float(y1)))
                continue
            pad = int(0.15 * height) + 2
            text, score = _recognise(band[max(0, int(y0) - pad):int(y1) + pad, max(0, int(x0) - pad):int(x1) + pad])
            if NUMBER.fullmatch(text) and score >= MIN_SCORE:
                tokens.append({"band": name, "x": round(float((x0 + x1) / 2 / band.shape[1]), 3), "text": text, "score": round(score, 2)})
        # The outermost line of each page: the head line in the top band, the foot line in the bottom one.
        middle = band.shape[1] / 2
        for side in (lambda line: (line[0] + line[2]) / 2 < middle, lambda line: (line[0] + line[2]) / 2 >= middle):
            page_lines = [line for line in lines if side(line)]
            if not page_lines:
                continue
            x0, y0, x1, y1 = min(page_lines, key=lambda line: line[1]) if name == "top" else max(page_lines, key=lambda line: line[3])
            height = y1 - y0
            pad = int(0.15 * height) + 2
            reach = END_ASPECT * height
            for end, (xa, xb), pattern in (("left", (x0, min(x1, x0 + reach)), LEADING), ("right", (max(x0, x1 - reach), x1), TRAILING)):
                text, score = _recognise(band[max(0, int(y0) - pad):int(y1) + pad, max(0, int(xa) - pad):int(xb) + pad])
                match = pattern.search(text)
                if match and score >= MIN_SCORE:
                    centre = (xa + 0.5 * height) if end == "left" else (xb - 0.5 * height)
                    tokens.append({"band": name, "x": round(float(centre / band.shape[1]), 3), "text": match.group(1), "score": round(score, 2), "line_end": end})
    return tokens
