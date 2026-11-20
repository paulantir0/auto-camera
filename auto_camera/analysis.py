"""One decode pass per video, then everything else from tiny thumbnails.

``analyse`` reads a proxy once and stores a 96x54 gray thumbnail of every frame
(5 KB per frame) plus a median colour frame. The page mask, motion, turn
detection and frame choice are all computed from those afterwards, so changing
any of them does not need the video again. Nothing here holds a whole video in
memory.

Motion is measured on 10x10-pixel block averages, not on pixels. On a page of
dense print, individual pixels flicker by more than ten gray levels between
frames from compression noise and sub-pixel wobble even when the book is
still, which makes a per-pixel measure read 30% "motion" on a resting page. A
block average does not move unless something really passes over the block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .paths import CACHE_ROOT
from .proxy import DEFAULT_PROXY_DIR, PROXY_HEIGHT, PROXY_WIDTH, REPO_ROOT, iter_proxy, probe

DEFAULT_ANALYSIS_DIR = CACHE_ROOT / "analysis"

THUMB_H, THUMB_W = 54, 96   # stored thumbnail: 5x5-pixel averages of the proxy
CELL = 10                   # pixels per motion cell side
GRID_H, GRID_W = PROXY_HEIGHT // CELL, PROXY_WIDTH // CELL   # 27 x 48
CELL_THRESHOLD = 6          # gray levels a block average must change by to count as moved
STRONG_THRESHOLD = 25       # gray levels, per pixel: "this pixel shows different content"
DETAIL_SCALE = 1.0          # stored detail = mean |Laplacian| of the cell, rounded
MEDIAN_SAMPLE_FRAMES = 240
PAPER_CHROMA_TOLERANCE = 9  # Lab a/b distance from the paper colour that still counts as paper
PAPER_VALUE_SHARE = 0.7     # and this share of the paper's brightness
TEXT_LEVEL = 0.5            # a print cell's usual detail, as a share of the strongest print
DIP_LEVEL = 0.5             # a cell's detail has collapsed when it is under this share of its usual value
DIP_SHARE = 0.3             # a print cell collapses at least this often, relative to the cells that do so most
MIN_DIP_RATE = 0.02         # and in at least this share of frames
MIN_TEXT_FILL = 0.4          # print cells must fill this share of their own outline to be trusted as the pages (real books: 0.58 and up; scattered background: 0.22)
ACTIVITY_SHARE = 0.35       # fallback mask only: a page cell is swept at least this often, relative to the busiest bright cells
MARGIN_X, MARGIN_Y = 0.09, 0.13   # how far paper reaches beyond the print, as shares of the print's width and height
ROW_SLACK = 0.1             # a frame's first line may sit this far below the usual one (share of the print's height) and still be the first line


def grab_frames(proxy: Path, indices, *, gray: bool = False) -> dict[int, np.ndarray]:
    """Return the requested frames, by index, in one streaming pass."""
    wanted = sorted({int(i) for i in indices})
    if not wanted:
        return {}
    out: dict[int, np.ndarray] = {}
    position = 0
    last = wanted[-1]
    for first, chunk in iter_proxy(proxy, gray=gray, chunk_frames=128):
        stop = first + len(chunk)
        while position < len(wanted) and wanted[position] < stop:
            out[wanted[position]] = chunk[wanted[position] - first].copy()
            position += 1
        if stop > last:
            break
    return out


def median_frame(proxy: Path, frames: int) -> np.ndarray:
    """Per-pixel median colour over evenly spaced frames: the typical scene."""
    step = max(1, frames // MEDIAN_SAMPLE_FRAMES)
    sample = grab_frames(proxy, range(0, frames, step))
    stack = np.stack([sample[i] for i in sorted(sample)])
    return np.median(stack, axis=0).astype(np.uint8)


def text_cells(detail: np.ndarray) -> np.ndarray:
    """(27, 48) bool: cells that hold print on a page that gets turned.

    Two things mark such a cell. Its fine detail is usually strong, as print
    is. And that detail regularly collapses for a few frames, because a leaf
    passing over the cell is blurred; a textured wall, a pen cup or a shelf of
    book spines is just as detailed but never does this. Both are measured
    from the video, with no idea of colour or of where the book sits.
    """
    if len(detail) == 0:
        return np.zeros((GRID_H, GRID_W), dtype=bool)
    usual = np.median(detail[:: max(1, len(detail) // 600)], axis=0).astype(np.float32)
    dips = (detail < DIP_LEVEL * usual[None]).mean(axis=0)
    strong = usual >= 0.6 * TEXT_LEVEL * np.percentile(usual, 97)
    if not strong.any():
        return strong
    turned = dips >= max(MIN_DIP_RATE, DIP_SHARE * float(np.percentile(dips[strong], 90)))
    if not (strong & turned).any():
        return strong & turned
    cells = (usual >= TEXT_LEVEL * float(np.percentile(usual[strong & turned], 90))) & turned
    # A block of print is at least three cells each way; a strip thinner than that is a page edge or a wire.
    square = np.ones((3, 3), np.uint8)
    solid = cv2.morphologyEx(cells.astype(np.uint8), cv2.MORPH_OPEN, square)
    cells &= cv2.dilate(solid, square) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cells.astype(np.uint8), connectivity=4)
    if count > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        cells = np.isin(labels, [i + 1 for i, area in enumerate(areas) if area >= 0.15 * areas.max()])
    return cells


def text_fill(text: np.ndarray) -> float:
    """How much of their own outline the print cells fill: near 1 for two blocks of print, low for scattered clutter."""
    ys, xs = np.where(text)
    if len(xs) < 3:
        return 0.0
    hull = np.zeros(text.shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32)), 1)
    return float(text.sum() / max(1, int(hull.sum())))


def page_mask_from_brightness(median_bgr: np.ndarray, activity: np.ndarray) -> np.ndarray:
    """The paper where there is too little print to find it by: bright, paper-coloured and swept by turning pages.

    This was the only method before the page was grown from its print. It lets
    a white wall in and cuts shaded pages (see ``page_mask``), so it is used
    only for a recording with hardly any print in it, such as the blank and
    title leaves at the front of a book, where the print cells found are
    scattered bits of background. ``activity`` is the per-cell share of frames
    in which the cell moved.
    """
    hsv = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    threshold, _ = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = value > threshold
    if not bright.any():
        return bright
    lab = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    activity_px = cv2.resize(activity.astype(np.float32), (PROXY_WIDTH, PROXY_HEIGHT), interpolation=cv2.INTER_NEAREST)
    busy_level = float(np.percentile(activity_px[bright], 75))
    swept = activity_px >= ACTIVITY_SHARE * busy_level
    core = bright & swept & (value >= np.median(value[bright & swept])) if (bright & swept).any() else bright
    paper_a, paper_b = np.median(lab[..., 1][core]), np.median(lab[..., 2][core])
    chroma = np.hypot(lab[..., 1] - paper_a, lab[..., 2] - paper_b)
    mask = (bright & swept & (chroma < PAPER_CHROMA_TOLERANCE)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i, area in enumerate(areas) if area >= 0.15 * areas.max()]
        mask = np.isin(labels, keep).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    return mask.astype(bool)


def paper_colour(median_bgr: np.ndarray, where: np.ndarray) -> tuple[float, float, float]:
    """Brightness and Lab a, b of the paper: the median over ``where`` (print and the paper between its lines)."""
    value = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    lab = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2LAB)
    return float(np.median(value[where])), float(np.median(lab[..., 1][where])), float(np.median(lab[..., 2][where]))


def paper_like(frame_bgr: np.ndarray, paper: tuple[float, float, float]) -> np.ndarray:
    """Bool image: pixels about as bright and as neutral as the paper."""
    value = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - paper[1], lab[..., 2] - paper[2])
    return (value >= PAPER_VALUE_SHARE * paper[0]) & (chroma < PAPER_CHROMA_TOLERANCE)


def page_mask(median_bgr: np.ndarray, text: np.ndarray) -> np.ndarray:
    """The paper, as a bool image, grown out from the blocks of print.

    ``text`` is ``text_cells``. The page is the print plus its margins: pixels
    as bright and as neutral as the paper between the lines, no further from
    the print than a margin can be. A white wall or desk beyond that distance
    is left out however much it looks like paper, and the pale hand holding
    the book is left out by its colour.
    """
    text_px = cv2.resize(text.astype(np.uint8), (PROXY_WIDTH, PROXY_HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
    if not text_px.any():
        return text_px
    paper = paper_like(median_bgr, paper_colour(median_bgr, text_px))
    ys, xs = np.where(text_px)
    hull = np.zeros(text_px.shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32)), 1)
    reach_x = max(CELL, int(MARGIN_X * (xs.max() - xs.min() + 1)))
    reach_y = max(CELL, int(MARGIN_Y * (ys.max() - ys.min() + 1)))
    limit = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach_x + 1, 2 * reach_y + 1))) > 0
    mask = ((paper & limit) | text_px).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) | text_px.astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(mask)
    if count > 2:
        mask = np.isin(labels, np.unique(labels[text_px])).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    return mask.astype(bool)


def gutter_column(text: np.ndarray) -> float | None:
    """Proxy x of the gutter: the emptiest columns between the two blocks of print."""
    columns = np.where(text.any(axis=0))[0]
    if len(columns) < 6:
        return None
    x0, x1 = int(columns.min()), int(columns.max())
    profile = text.sum(axis=0).astype(np.float32)
    lo, hi = x0 + int(0.3 * (x1 - x0)), x1 - int(0.3 * (x1 - x0))
    if hi <= lo:
        return None
    lowest = np.where(profile[lo:hi + 1] == profile[lo:hi + 1].min())[0] + lo
    return float((lowest.min() + lowest.max() + 1) / 2 * CELL)


def page_halves(mask: np.ndarray, text: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Left-page and right-page regions of the mask, leaving out the gutter."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    columns = np.where(inner.any(axis=0))[0]
    if len(columns) == 0:
        return inner, inner
    x0, x1 = int(columns.min()), int(columns.max())
    centre = gutter_column(text) if text is not None else None
    if centre is None:
        centre = (x0 + x1) / 2
    gap = 0.04 * (x1 - x0)
    xs = np.arange(mask.shape[1])[None, :]
    return inner & (xs < centre - gap), inner & (xs > centre + gap)


def to_grid(mask: np.ndarray) -> np.ndarray:
    """Pixel mask to cell mask: a cell is in when most of its pixels are."""
    trimmed = mask[: GRID_H * CELL, : GRID_W * CELL].reshape(GRID_H, CELL, GRID_W, CELL)
    return trimmed.mean(axis=(1, 3)) > 0.5


@dataclass
class Analysis:
    slug: str
    fps: float
    thumbs: np.ndarray          # (N, 54, 96) uint8 gray thumbnails
    detail: np.ndarray          # (N, 27, 48) uint8, fine detail (print) energy per cell, see DETAIL_SCALE
    moved: np.ndarray           # (N, 27, 48) bool, cell changed since the previous frame
    median: np.ndarray          # (270, 480, 3) uint8
    mask: np.ndarray            # (270, 480) bool, the page region
    grid_page: np.ndarray       # (27, 48) bool cell masks
    grid_left: np.ndarray
    grid_right: np.ndarray
    page_motion: np.ndarray     # (N,) share of the page that moved since the previous frame
    left_motion: np.ndarray     # (N,) the same over the left page
    right_motion: np.ndarray    # (N,) the same over the right page
    page_detail: np.ndarray     # (N,) mean fine-detail energy over the page: sharp, uncovered print scores high

    @property
    def frames(self) -> int:
        return len(self.page_motion)

    @property
    def inner(self) -> np.ndarray:
        """The page region pulled in from its edge, for comparing frames."""
        return cv2.erode(self.mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)


def _decode(proxy: Path) -> tuple[np.ndarray, np.ndarray, float]:
    info = probe(proxy)
    thumbs = np.zeros((info.frames, THUMB_H, THUMB_W), dtype=np.uint8)
    detail = np.zeros((info.frames, GRID_H, GRID_W), dtype=np.uint8)
    index = 0
    for _, chunk in iter_proxy(proxy, gray=True, chunk_frames=128):
        for frame in chunk:
            thumbs[index] = cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
            edges = cv2.convertScaleAbs(cv2.Laplacian(frame, cv2.CV_16S, ksize=1))
            detail[index] = cv2.resize(edges, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
            index += 1
            if index >= info.frames:
                break
    return thumbs[:index], detail[:index], info.fps


def cell_motion(thumbs: np.ndarray, threshold: float = CELL_THRESHOLD) -> np.ndarray:
    """(N, 27, 48) bool: which 10x10-pixel blocks changed since the previous frame."""
    count = len(thumbs)
    moved = np.zeros((count, GRID_H, GRID_W), dtype=bool)
    step = 2048
    previous: np.ndarray | None = None
    for start in range(0, count, step):
        part = thumbs[start:start + step].astype(np.float32)
        blocks = part.reshape(len(part), GRID_H, 2, GRID_W, 2).mean(axis=(2, 4))
        if previous is not None:
            moved[start] = np.abs(blocks[0] - previous) > threshold
        moved[start + 1:start + len(part)] = np.abs(np.diff(blocks, axis=0)) > threshold
        previous = blocks[-1]
    return moved


def _region_motion(moved: np.ndarray, region: np.ndarray) -> np.ndarray:
    if not region.any():
        return np.zeros(len(moved), dtype=np.float32)
    return moved[:, region].mean(axis=1).astype(np.float32)


def derive(slug: str, fps: float, thumbs: np.ndarray, detail: np.ndarray, median: np.ndarray) -> Analysis:
    """Mask and motion from the stored thumbnails. Cheap; no video needed."""
    moved = cell_motion(thumbs)
    text = text_cells(detail)
    if text_fill(text) >= MIN_TEXT_FILL:
        mask = page_mask(median, text)
        left, right = page_halves(mask, text)
    else:
        mask = page_mask_from_brightness(median, moved.mean(axis=0))
        left, right = page_halves(mask)
    grid_page, grid_left, grid_right = to_grid(mask), to_grid(left), to_grid(right)
    return Analysis(
        slug=slug, fps=fps, thumbs=thumbs, detail=detail, moved=moved, median=median, mask=mask,
        grid_page=grid_page, grid_left=grid_left, grid_right=grid_right,
        page_motion=_region_motion(moved, grid_page),
        left_motion=_region_motion(moved, grid_left),
        right_motion=_region_motion(moved, grid_right),
        page_detail=(detail[:, grid_page].mean(axis=1) / DETAIL_SCALE).astype(np.float32) if grid_page.any() else np.zeros(len(detail), np.float32),
    )


def analyse(slug: str, *, proxy_dir: Path = DEFAULT_PROXY_DIR, cache_dir: Path = DEFAULT_ANALYSIS_DIR, refresh: bool = False) -> Analysis:
    """Thumbnails and median for one video, cached; mask and motion derived on load."""
    proxy = proxy_dir / f"{slug}.mp4"
    cache = cache_dir / f"{slug}.npz"
    if cache.exists() and not refresh and cache.stat().st_mtime >= proxy.stat().st_mtime:
        with np.load(cache) as data:
            return derive(slug, float(data["fps"]), data["thumbs"], data["detail"], data["median"])
    thumbs, detail, fps = _decode(proxy)
    median = median_frame(proxy, len(thumbs))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, fps=fps, thumbs=thumbs, detail=detail, median=median)
    return derive(slug, fps, thumbs, detail, median)


PRINT_LEVEL = 0.5           # a cell is print area when its usual detail is this share of the page's strongest
MIN_PRINT_CELLS = 40        # fewer than this on a page and the whole page is used instead


def print_cells(analysis: Analysis) -> np.ndarray:
    """(27, 48) bool: cells where print usually is on this book's pages.

    A cell belongs when its median fine-detail energy over the video is at
    least half that of the page's busiest cells. Margins, the gutter and the
    spot where the holding hand rests fall outside it.
    """
    page = analysis.grid_page
    if not page.any():
        return np.zeros(page.shape, dtype=bool)
    step = max(1, analysis.frames // 600)
    usual = np.median(analysis.detail[::step], axis=0)
    strong = float(np.percentile(usual[page], 90))
    return (usual >= PRINT_LEVEL * strong) & page


def _printed_rows(cells: np.ndarray, page: np.ndarray) -> tuple[int, int] | None:
    """First and one-past-last grid row in which a fair share of the page's cells hold print."""
    width = int(page.sum(axis=1).max()) if page.any() else 0
    if width == 0:
        return None
    # Measured against the full width of the page: the paper's own edge is as
    # sharp as print, and a row that only clips a corner of the page has few
    # cells, most of them edge.
    printed = list((cells & page).sum(axis=1) >= 0.25 * width) + [False]
    runs, start = [], None
    for row, on in enumerate(printed):
        if on and start is None:
            start = row
        elif not on and start is not None:
            runs.append((start, row))
            start = None
    # Lines of print fill several rows on end. A single row by itself is the
    # top or bottom edge of the book against whatever is behind it.
    blocks = [run for run in runs if run[1] - run[0] >= 3] or runs
    return (blocks[0][0], blocks[-1][1]) if blocks else None


def usual_text_rows(analysis: Analysis) -> tuple[float, float] | None:
    """Where the print usually starts and ends, as fractions of the frame height."""
    rows = _printed_rows(print_cells(analysis), analysis.grid_page)
    return (rows[0] * CELL / PROXY_HEIGHT, rows[1] * CELL / PROXY_HEIGHT) if rows else None


def text_rows(analysis: Analysis, frame: int, usual: tuple[float, float] | None = None) -> tuple[float, float] | None:
    """Top and bottom of the print in one frame, as fractions of the frame height.

    The book creeps up and down the stand during a recording, so the first and
    last lines are looked for in the frame itself, inside the page region.
    ``usual`` (``usual_text_rows``) guards against a page that is short of
    print: where a chapter ends early or starts low, the first or last line of
    this frame lies well inside the usual block, and the usual position is
    used for that end. ``None`` when neither is known.
    """
    page = analysis.grid_page
    detail = analysis.detail[frame].astype(np.float32)
    on_page = detail[page]
    level = TEXT_LEVEL * float(np.percentile(on_page, 90)) if on_page.size else 0.0
    rows = _printed_rows(detail >= level, page) if level > 0 else None
    here = (rows[0] * CELL / PROXY_HEIGHT, rows[1] * CELL / PROXY_HEIGHT) if rows else None
    if usual is None or here is None:
        return here or usual
    slack = ROW_SLACK * (usual[1] - usual[0])
    return (here[0] if here[0] <= usual[0] + slack else usual[0], here[1] if here[1] >= usual[1] - slack else usual[1])


def find_valleys(motion: np.ndarray, *, window: int = 3, ceiling: float = 0.15,
                 dip_ceiling: float = 0.45, min_prominence: float = 0.20, reach: int = 22) -> list[int]:
    """Frames where a motion signal is a local minimum: candidate moments of rest.

    A local minimum (lowest within ``window`` frames either side) qualifies in
    one of two ways. Either its motion is below ``ceiling``, the ordinary case
    of a page at rest. Or it is a clear dip between two bursts: below
    ``dip_ceiling`` and at least ``min_prominence`` lower than the highest
    motion within ``reach`` frames on each side. The second rule is what keeps
    spreads that were flipped past without ever coming to rest.
    """
    count = len(motion)
    valleys: list[int] = []
    for i in range(1, count):
        value = float(motion[i])
        if value >= dip_ceiling:
            continue
        if value > motion[max(0, i - window): i + window + 1].min():
            continue
        if value >= ceiling:
            before = float(motion[max(0, i - reach): i].max()) if i > 0 else 0.0
            after = float(motion[i + 1: i + 1 + reach].max()) if i + 1 < count else 0.0
            if min(before, after) - value < min_prominence:
                continue
        if valleys and i - valleys[-1] <= window:
            if value < motion[valleys[-1]]:
                valleys[-1] = i
        else:
            valleys.append(i)
    return valleys


def page_changed(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    """Share of ``region`` whose brightness differs strongly between two gray frames."""
    area = max(1, int(region.sum()))
    return float(((cv2.absdiff(a, b) > STRONG_THRESHOLD) & region).sum() / area)
