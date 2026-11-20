"""One clean, flat image per page from a still of a two-page spread.

A still shows the open book from a camera above and in front of it: each page
is a sheet that curves down into the fold and is seen at an angle. For OCR, for
reading and for a PDF each page should come out flat, upright and cropped to
the page. This module does that from the lines of text on the page, which
should be straight, level and parallel on the flat page:

1. The spread is split at the fold (``find_fold``), the gap between the two
   blocks of print.
2. On each half, lines of text are found (``text_spans``): ink, joined along
   the line and kept where it forms long thin runs; each run is sampled at
   regular steps along its middle.
3. A model of the page is fitted to those samples (``fit_page``): a sheet
   that is straight from top to bottom and curves as a cubic from left to
   right, seen through a pinhole camera from some position. Each run of text
   is a line of constant height on the sheet. This is the model of Matt
   Zucker's *page_dewarp* (2016); the fit here is a sparse least-squares
   solve rather than a general minimiser, which takes a fraction of a second.
4. The sheet is unrolled onto a flat page and the frame resampled
   (``render``), cropped to the print with a margin.

When too few lines are found (a plate, a chapter opening with a few lines)
or the fit does not explain them, the page is only cut out and straightened
with the outline of its print (``flat_crop``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

WORK_LINE = 30.0         # pixels from one line of text to the next at the working scale
FOCAL = 1.4              # camera focal length, as a share of half the frame's width (about a 70-degree view)
STEP = 0.8               # samples along a line of text, this many line pitches apart
MIN_SPANS = 8            # fewer lines than this and a page is only straightened, not unrolled
MARGIN = 0.05            # white kept around the print, as a share of the print's height
ONE_SIDED = 0.12         # under this share of the print on one side of the fold, that page is taken to be blank
WIDE_GAP = 0.02          # the fold's gap is looked for this wide first, as a share of the frame's width
NARROW_GAP = 0.006       # then this narrow (a tightly bound book: the print runs almost into the fold)
NARROW_BALANCE = 0.25    # a narrow gap is the fold only with at least this share of the print on either side
NARROW_NEAR = 0.15       # and only this near the middle of the book, as a share of the book's width
BLANK_NEAR = 0.25        # with one page blank, the fold is looked for this near the middle of the book
MAX_RESIDUAL = 0.004     # median distance of a sample from its fitted line, as a share of the frame's width, beyond which the fit is not trusted


@dataclass
class Spans:
    """Lines of text on one page, in frame pixels: per line, the sampled points along its middle."""

    points: list[np.ndarray] = field(default_factory=list)   # each (n, 2) float32, x and y
    pitch: float = 0.0                                        # line pitch in frame pixels

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class PageFit:
    """A fitted page: camera pose, curvature, and where the print lies on the flat sheet."""

    rvec: np.ndarray
    tvec: np.ndarray
    curve: np.ndarray          # alpha, beta: slopes of the sheet at its left and right edge
    width: float               # extent of the sheet in the model (normalised frame units)
    height: float
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 of the print on the sheet
    residual: float            # median distance of a sample from its line, as a share of frame width
    unrolled: bool             # False: a plane fitted to the outline only


@dataclass
class Page:
    image: np.ndarray          # BGR
    side: str                  # "left" or "right"
    lines: int                 # lines of text found
    unrolled: bool             # curvature removed (else only cut out and straightened)
    residual: float


# --- the fold -----------------------------------------------------------------------------------

def ink_mask(gray: np.ndarray, pitch: float) -> np.ndarray:
    """Dark marks on paper: print, as a bool image."""
    block = int(pitch * 2) | 1
    return cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 18).astype(bool)


def find_fold(gray: np.ndarray, region: np.ndarray | None = None) -> int:
    """Column of the fold between the two pages.

    The fold is a wide gap in the print with about as much print on one
    side as on the other: of the columns with little ink around them, the
    one that divides the print most evenly. (The emptiest column alone can be
    the outer margin of a page, between its print and the stack of leaves.)
    In a tightly bound book the print runs almost into the fold and the gap
    is narrow; a narrow gap is taken only near the middle of the book and
    with print on both sides (a two-column page has one too). When one page
    is blank, it is the empty column nearest the middle of the book, in the
    middle half of the book (an empty column at the book's edge, beside a
    title page of a few lines, is not the fold).
    """
    small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4), interpolation=cv2.INTER_AREA)
    ink = ink_mask(small, 8).astype(np.uint8)
    # The edges of the stacked leaves beside a page are long upright lines,
    # as dense as print in a column; letters are short. Long upright runs
    # are taken out.
    upright = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))
    ink = (ink & ~cv2.dilate(upright, np.ones((3, 3), np.uint8))).astype(np.float32)
    if region is not None:
        ink *= cv2.resize(region.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    profile = cv2.blur(ink.sum(axis=0, keepdims=True), (5, 1))[0]
    width = small.shape[1]
    if profile.sum() <= 0:
        return gray.shape[1] // 2
    busy = np.quantile(profile[profile > 0], 0.75)
    # Only columns dense enough to be print are weighed: blank paper shows
    # faint marks too (its grain, print through the leaf), and a blank page
    # would otherwise weigh as much as a page of text in a narrow column.
    text = np.where(profile > 0.3 * busy, profile, 0.0)
    if text.sum() <= 0:
        return gray.shape[1] // 2
    total = np.cumsum(text)
    balance = np.minimum(total, total[-1] - total) / total[-1]
    # The book: where print usually is, else the whole frame.
    if region is not None and region.any():
        columns = np.flatnonzero(cv2.resize(region.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).any(axis=0))
        left, right = int(columns.min()), int(columns.max())
    else:
        left, right = 0, width
    middle = (left + right) / 2
    distance = np.abs(np.arange(width) - middle)

    def gap(share: float) -> np.ndarray:
        """How empty of print each column is, over a band this share of the width."""
        wide = cv2.blur(profile[None, :], (max(3, int(width * share)) | 1, 1))[0]
        return np.clip(1.0 - wide / busy, 0.0, 1.0)

    # The fold is a wide gap (the inner margins of two pages); a column
    # between two words inside a block of print is narrow.
    emptiness = gap(WIDE_GAP)
    score = balance * emptiness
    score[: int(width * 0.2)] = -1.0
    score[int(width * 0.8):] = -1.0
    fold = int(np.argmax(score))
    if balance[fold] >= ONE_SIDED and emptiness[fold] >= 0.6:
        return fold * 4
    narrow = gap(NARROW_GAP)
    score = balance * narrow
    score[: int(width * 0.2)] = -1.0
    score[int(width * 0.8):] = -1.0
    score[distance > NARROW_NEAR * (right - left)] = -1.0
    fold = int(np.argmax(score))
    if balance[fold] >= NARROW_BALANCE and narrow[fold] >= 0.6:
        return fold * 4
    # Nearly all the print on one side, or no gap with print on both sides:
    # one page is blank (a verso, the end of a chapter). Then the fold is
    # the empty column nearest the middle of the book.
    empty = np.flatnonzero(emptiness > 0.8)
    empty = empty[(empty > width * 0.2) & (empty < width * 0.8) & (distance[empty] <= BLANK_NEAR * (right - left))]
    if not len(empty):
        return int(middle) * 4
    return int(empty[np.argmin(np.abs(empty - middle))]) * 4


# --- lines of text ------------------------------------------------------------------------------

def text_spans(image: np.ndarray, x0: int, x1: int, region: np.ndarray | None = None) -> Spans:
    """Lines of text between columns ``x0`` and ``x1`` of a frame, sampled along their middles.

    Ink is averaged along the row over about a line pitch: in the band of
    the small letters of a line that average is high, in the ascenders and
    descenders that touch the next line it is low, so cutting it at a level
    leaves one band per line even where the lines are close. Bands that are
    long, thin and near level are the lines.
    """
    gray = cv2.cvtColor(image[:, x0:x1], cv2.COLOR_BGR2GRAY)
    pitch = estimate_pitch(gray, None if region is None else region[:, x0:x1])
    scale = WORK_LINE / pitch
    work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = ink_mask(work, WORK_LINE).astype(np.float32)
    if region is not None:
        part = region[:, x0:x1].astype(np.uint8)
        ink *= cv2.resize(part, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST)
    density = cv2.blur(ink, (int(WORK_LINE), 1))
    level = float(np.quantile(density[density > 0], 0.9)) if (density > 0).any() else 1.0
    core = (density > 0.4 * level).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core)
    spans = Spans(pitch=pitch)
    step = max(4, int(WORK_LINE * STEP))
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if w < 3 * WORK_LINE:
            continue
        mask = labels[y:y + h, x:x + w] == index
        heights = mask.sum(axis=0)
        thickness = float(np.median(heights[heights > 0]))
        if thickness > 0.75 * WORK_LINE or thickness < 0.12 * WORK_LINE:
            continue
        points = []
        for start in range(0, w - step // 2, step):
            column = mask[:, start:start + step]
            rows = np.nonzero(column)[0]
            if len(rows) < 3 or np.ptp(rows) > 1.2 * WORK_LINE:
                continue
            points.append((x + start + min(step, w - start) / 2, y + rows.mean()))
        if len(points) < 3:
            continue
        points = np.array(points, dtype=np.float32)
        slope = np.polyfit(points[:, 0], points[:, 1], 1)[0]
        if abs(slope) > 0.35:
            continue
        spans.points.append(points / scale + np.array([x0, 0], dtype=np.float32))
    return _main_block(spans)


def _main_block(spans: Spans) -> Spans:
    """Only lines that run like most of the others and have another line a pitch or two above or below them.

    That drops what is long and thin but not print: the edge of the desk, a
    cable, the stand, a crease of the hand.
    """
    if len(spans) < 4:
        return spans
    angles = np.array([np.arctan(np.polyfit(p[:, 0], p[:, 1], 1)[0]) for p in spans.points])
    usual = float(np.median(angles))
    level = [p for p, a in zip(spans.points, angles) if abs(a - usual) < np.radians(6)]
    keep = []
    for index, points in enumerate(level):
        lo, hi = points[:, 0].min(), points[:, 0].max()
        y = float(np.median(points[:, 1]))
        for other_index, other in enumerate(level):
            if other_index == index:
                continue
            olo, ohi = other[:, 0].min(), other[:, 0].max()
            if min(hi, ohi) - max(lo, olo) <= 0:
                continue
            gap = abs(float(np.median(other[:, 1])) - y)
            if 0.5 * spans.pitch < gap < 2.5 * spans.pitch:
                keep.append(points)
                break
    return Spans(points=keep, pitch=spans.pitch)


def estimate_pitch(gray: np.ndarray, where: np.ndarray | None = None) -> float:
    """Distance between lines of text in pixels: the strongest period of the ink's row profile, the page turned to its best angle.

    ``where`` (bool, the size of ``gray``) limits it to the print: a spiral
    binding or the fingers of a hand are periodic too.
    """
    small = cv2.resize(gray, (max(64, gray.shape[1] // 2), max(64, gray.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    ink = ink_mask(small, 30).astype(np.float32)
    if where is not None:
        ink *= cv2.resize(where.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = ink.shape
    best_power, best_period = 0.0, 20.0
    for angle in (-14, -10, -6, -3, 0, 3, 6, 10, 14):
        turned = cv2.warpAffine(ink, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), (w, h))
        profile = turned[:, w // 4: 3 * w // 4].sum(axis=1)
        profile = profile - profile.mean()
        spectrum = np.abs(np.fft.rfft(profile * np.hanning(len(profile))))
        periods = len(profile) / np.maximum(np.arange(len(spectrum)), 1)
        usable = (periods > 6) & (periods < len(profile) / 8)
        if not usable.any():
            continue
        index = int(np.argmax(np.where(usable, spectrum, 0)))
        if spectrum[index] > best_power:
            best_power, best_period = float(spectrum[index]), float(periods[index])
    return best_period * 2


# --- the model ----------------------------------------------------------------------------------

def _camera(shape: tuple[int, int]) -> tuple[np.ndarray, float, np.ndarray]:
    height, width = shape
    scale = 2.0 / width
    centre = np.array([width / 2, height / 2], dtype=np.float64)
    k = np.array([[FOCAL, 0, 0], [0, FOCAL, 0], [0, 0, 1]], dtype=np.float64)
    return k, scale, centre


def _sheet_z(x: np.ndarray, width: float, curve: np.ndarray) -> np.ndarray:
    """Height of the sheet above its plane at ``x`` (0 to ``width``): a cubic, flat at neither end, zero at both."""
    alpha, beta = curve
    u = x / width
    return width * (((alpha + beta) * u - (2 * alpha + beta)) * u + alpha) * u


def _project(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(points.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, k, None)
    return projected.reshape(-1, 2)


def fit_page(spans: Spans, outline: np.ndarray, shape: tuple[int, int]) -> PageFit:
    """Fit the curved-sheet model to the lines of one page.

    ``outline`` is the page's print as four corners in frame pixels (top
    left, top right, bottom right, bottom left); ``shape`` the frame's
    height and width.
    """
    k, scale, centre = _camera(shape)
    corners = (outline.astype(np.float64) - centre) * scale
    width = float(np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])) / 2
    height = float(np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])) / 2
    plane = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(plane, corners, k, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(plane, corners, k, None)
    rvec, tvec = rvec.reshape(3), tvec.reshape(3)
    if len(spans) < MIN_SPANS:
        return PageFit(rvec, tvec, np.zeros(2), width, height, (0.0, 0.0, width, height), 0.0, False)

    # Where each sample lies on the plane of the outline, to start from.
    to_plane = cv2.getPerspectiveTransform(corners.astype(np.float32), plane[:, :2].astype(np.float32))
    observed, span_of, start_x, start_y = [], [], [], []
    for index, points in enumerate(spans.points):
        normal = (points.astype(np.float64) - centre) * scale
        on_plane = cv2.perspectiveTransform(normal.reshape(-1, 1, 2).astype(np.float32), to_plane).reshape(-1, 2)
        observed.append(normal)
        span_of.extend([index] * len(points))
        start_x.extend(on_plane[:, 0])
        start_y.append(float(on_plane[:, 1].mean()))
    observed = np.concatenate(observed)
    span_of = np.array(span_of)
    spans_count, points_count = len(spans), len(observed)
    params = np.concatenate([rvec, tvec, [0.0, 0.0], start_y, start_x])

    def residuals(p: np.ndarray) -> np.ndarray:
        r, t, curve = p[:3], p[3:6], p[6:8]
        ys = p[8:8 + spans_count][span_of]
        xs = p[8 + spans_count:]
        sheet = np.stack([xs, ys, _sheet_z(xs, width, curve)], axis=1)
        return (_project(sheet, r, t, k) - observed).ravel()

    sparsity = lil_matrix((2 * points_count, len(params)), dtype=np.uint8)
    rows = np.arange(points_count)
    for d in range(2):
        sparsity[2 * rows + d, :8] = 1
        sparsity[2 * rows + d, 8 + span_of] = 1
        sparsity[2 * rows + d, 8 + spans_count + rows] = 1
    pitch = spans.pitch * scale
    result = least_squares(residuals, params, jac_sparsity=sparsity, method="trf", loss="soft_l1", f_scale=pitch * 0.15,
                           x_scale="jac", max_nfev=200)
    p = result.x
    xs, ys = p[8 + spans_count:], p[8:8 + spans_count]
    residual = float(np.median(np.abs(result.fun))) / 2      # frame width is 2 units
    box = (float(np.quantile(xs, 0.01)), float(ys.min()), float(np.quantile(xs, 0.99)), float(ys.max()))
    # The first and last line of print: their middles are half a pitch inside the print.
    box = (box[0], box[1] - pitch * 0.5, box[2], box[3] + pitch * 0.5)
    return PageFit(p[:3], p[3:6], p[6:8], width, height, box, residual, True)


def render(image: np.ndarray, fit: PageFit, *, pad: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """The page unrolled flat: the print and ``pad`` (across, down; sheet units) around it, at about the resolution it has in the frame."""
    k, scale, centre = _camera(image.shape[:2])
    x0, y0, x1, y1 = fit.box
    x0, y0, x1, y1 = x0 - pad[0], y0 - pad[1], x1 + pad[0], y1 + pad[1]
    # Output size: as many pixels per sheet unit as the frame shows in the middle of the print.
    middle = np.array([[(x0 + x1) / 2, (y0 + y1) / 2, 0.0], [(x0 + x1) / 2, (y0 + y1) / 2 + 0.01, 0.0]])
    middle[:, 2] = _sheet_z(middle[:, 0], fit.width, fit.curve)
    pts = _project(middle, fit.rvec, fit.tvec, k)
    per_unit = float(np.linalg.norm(pts[1] - pts[0])) / 0.01 / scale
    out_w, out_h = int(round((x1 - x0) * per_unit)), int(round((y1 - y0) * per_unit))
    out_w, out_h = max(16, min(out_w, 6000)), max(16, min(out_h, 8000))
    # The map on a coarse grid, then smoothed up to full size.
    gw, gh = max(2, out_w // 16), max(2, out_h // 16)
    gx, gy = np.meshgrid(np.linspace(x0, x1, gw), np.linspace(y0, y1, gh))
    sheet = np.stack([gx.ravel(), gy.ravel(), _sheet_z(gx.ravel(), fit.width, fit.curve)], axis=1)
    pixels = _project(sheet, fit.rvec, fit.tvec, k) / scale + centre
    map_x = cv2.resize(pixels[:, 0].reshape(gh, gw).astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(pixels[:, 1].reshape(gh, gw).astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# --- the page outline -----------------------------------------------------------------------------

def print_outline(spans: Spans) -> np.ndarray | None:
    """Four corners around the lines of text (top left, top right, bottom right, bottom left), or None.

    The rectangle is aligned with the lines (their median direction), and
    its sides cut off the outermost hundredth of the samples, so that a stray
    run does not tilt or stretch it.
    """
    if len(spans) < 3:
        return None
    angles = [np.arctan(np.polyfit(p[:, 0], p[:, 1], 1)[0]) for p in spans.points]
    angle = float(np.median(angles))
    along = np.array([np.cos(angle), np.sin(angle)])
    across = np.array([-np.sin(angle), np.cos(angle)])
    points = np.concatenate(spans.points).astype(np.float64)
    u, v = points @ along, points @ across
    u0, u1 = np.quantile(u, [0.005, 0.995])
    v0, v1 = v.min() - spans.pitch * 0.5, v.max() + spans.pitch * 0.5
    return np.array([u0 * along + v0 * across, u1 * along + v0 * across, u1 * along + v1 * across, u0 * along + v1 * across])


def crop_to_print(page: np.ndarray, pitch: float, core: tuple[int, int, int, int], *, margin: float = MARGIN) -> np.ndarray:
    """Cut a flattened page down to its print and a margin.

    ``core`` (x0, y0, x1, y1, pixels) is where the fitted lines lie. Print
    near it that the lines did not include is added: the running head, the
    page number, notes in the margin. Marks that reach the edge of the
    rendered area are the edge of the paper, the stand or the hand, and are
    left out, as is anything further than three lines from the rest.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(gray, pitch).astype(np.uint8)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(pitch * 0.6)), max(3, int(pitch * 0.3)))))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    x0, y0, x1, y1 = core
    reach = 3 * pitch
    height, width = gray.shape
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue
        if h > 2 * pitch:
            continue        # taller than two lines and outside the print: the stack of leaves, the stand, a hand
        if x + w < x0 - reach or x > x1 + reach or y + h < y0 - reach or y > y1 + reach:
            continue
        x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x + w), max(y1, y + h)
    pad = int(round(margin * (core[3] - core[1])))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
    return page[y0:y1, x0:x1]


def even_paper(page: np.ndarray, pitch: float) -> np.ndarray:
    """The page with its paper made evenly light: the shading of the curve and of the lamp divided out.

    The paper's brightness is estimated by closing over the print (a
    neighbourhood larger than a letter takes the lightest value around it)
    and smoothing; each pixel's lightness is divided by it. Colour is kept,
    so plates and red print stay as they are.
    """
    lab = cv2.cvtColor(page, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)
    size = max(7, int(pitch * 1.5)) | 1
    paper = cv2.morphologyEx(light, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    paper = cv2.GaussianBlur(paper, (0, 0), size)
    lab[:, :, 0] = np.clip(light / np.maximum(paper, 1.0) * 245.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def page_images(image: np.ndarray, region: np.ndarray | None = None, *, even: bool = True) -> list[Page]:
    """The two pages of a spread, each flat and cropped; fewer if a half holds no print.

    ``region``, if given, is a bool image (any size) of where the pages are in
    the frame; print outside it (a calendar on the wall) is ignored.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    full_region = None
    if region is not None:
        full_region = cv2.resize(region.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    fold = find_fold(gray, full_region)
    pages = []
    for side, (x0, x1) in (("left", (0, fold)), ("right", (fold, image.shape[1]))):
        spans = text_spans(image, x0, x1, full_region)
        outline = print_outline(spans)
        if outline is None:
            # No lines of text: a plate, a blank page, a title page. It is
            # kept, cut out by where print usually is on this side.
            plain = _plain_crop(image, x0, x1, full_region)
            if plain is not None:
                pages.append(Page(even_paper(plain, 30.0) if even else plain, side, 0, False, 0.0))
            continue
        fit = fit_page(spans, outline, image.shape[:2])
        if fit.unrolled and fit.residual > MAX_RESIDUAL:
            fit = fit_page(Spans(pitch=spans.pitch), outline, image.shape[:2])
        flat, core, pitch = _render_with_room(image, fit, spans.pitch)
        page = crop_to_print(flat, pitch, core)
        if even:
            page = even_paper(page, pitch)
        pages.append(Page(page, side, len(spans), fit.unrolled, fit.residual))
    return pages


def _plain_crop(image: np.ndarray, x0: int, x1: int, region: np.ndarray | None) -> np.ndarray | None:
    """The part of ``region`` between columns ``x0`` and ``x1``, cut out along its tilted outline; None without a region there."""
    if region is None:
        return None
    part = np.zeros(region.shape, dtype=np.uint8)
    part[:, x0:x1] = region[:, x0:x1]
    if part.sum() < 0.02 * part.size:
        return None
    ys, xs = np.nonzero(part)
    (cx, cy), (w, h), angle = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
    if w < h * 0.2 or h < w * 0.2:
        return None
    if angle > 45:
        angle -= 90
        w, h = h, w
    turn = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    turn[:, 2] += np.array([w / 2 - cx, h / 2 - cy])
    return cv2.warpAffine(image, turn, (int(w), int(h)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _render_with_room(image: np.ndarray, fit: PageFit, pitch_px: float) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """The page flattened with four lines' room around the fitted print; also where the print lies in it, and the line pitch there."""
    _, scale, _ = _camera(image.shape[:2])
    x0, y0, x1, y1 = fit.box
    room = 4 * pitch_px * scale
    flat = render(image, fit, pad=(room, room))
    per_unit_x = flat.shape[1] / (x1 - x0 + 2 * room)
    per_unit_y = flat.shape[0] / (y1 - y0 + 2 * room)
    core = (int(room * per_unit_x), int(room * per_unit_y), int(flat.shape[1] - room * per_unit_x), int(flat.shape[0] - room * per_unit_y))
    return flat, core, pitch_px * scale * per_unit_y
