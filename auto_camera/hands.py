"""Find hands (and anything else that reaches in) over the page.

No fixed idea of skin colour is used: warm paper, page edges and plates defeat
that. The colour of whatever reaches into the picture is learned from each
video. Next to the book there is a strip of fixed background, and the only
thing that ever crosses it is the reader's hand and arm. Pixels that differ
from the median frame there are therefore hand, and their chroma becomes this
video's "intruder" colours. The page's own chroma (paper and ink, both nearly
neutral) is learned from inside the page region. A pixel on the page counts as
covered when its chroma is far more typical of the intruder than of the page.

Dark, neutral things such as a black sleeve look like ink and are not found.

Reported per frame: how much of the page is covered, and how much of the
print. The hand that holds the book open sits on the outer margin in almost
every frame; that is expected and only cover of the print counts against a
frame.

"The print" is taken two ways (see ``measure``). The cells that usually hold
print leave out the corner on which a hand rests in most frames: by what is
usual there is no print there, and a hand lying on it covers nothing. So where
the print is, is also learned per spot from the frames in which the book lies
still and nothing hand-coloured is over the spot (``print_maps``): fine detail
that changes from spread to spread.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import handnet
from .analysis import Analysis, grab_frames, paper_colour, paper_like, print_cells
from .paths import CACHE_ROOT
from .proxy import DEFAULT_PROXY_DIR, REPO_ROOT, iter_proxy

DEFAULT_HANDS_DIR = CACHE_ROOT / "hands"
SAMPLE_FRAMES = 300
BINS = 64                 # chroma histogram is BINS x BINS over a*, b* in [64, 192)
LOW, SPAN = 64, 128
FOREGROUND = 35           # a pixel this far from the median frame is not background
MIN_LIGHT = 70            # darker than this is ink or shadow, never counted as a hand
LIKELIHOOD_RATIO = 3.0    # how much more typical of the intruder than of the page
MIN_BLOB = 60             # proxy pixels; smaller specks are ignored
MIN_SAMPLES = 4000        # intruder pixels needed to trust the model
CHROMA_CHANGE = 10        # Lab a/b distance from the background's own colour at that spot; a shadow darkens a wall without changing its colour
PRINT_PARTS = 1           # stretches the video is cut into, each with its own print map. One: with eight of 60 frames the maps came out ragged
PRINT_FRAMES = 300        # frames with the book at rest from which each stretch's map is learned
MIN_PART_FRAMES = 30      # with fewer frames at rest than this per stretch, fewer stretches are made
PRINT_CALM = 0.03         # page motion at or under this, before and after a frame: the book lies still
PRINT_LEVEL = 0.4         # fine detail this strong, relative to the video's usual print, is print
PRINT_SHARE = 0.35        # print is there in at least this share of the frames in which nothing covers the spot
PRINT_CHANGE = 0.4        # and the spot's brightness varies from spread to spread this much, relative to print: the edge of the book does not
PRINT_REACH = 15          # proxy pixels beyond the page region in which print is still looked for
MIN_SEEN = 10             # frames in which a spot must have been uncovered to say anything about it
PRINT_INSIDE = 4          # proxy pixels in from the edge of the print map before a hand counts as on the print
PRINT_JOIN = 11           # proxy pixels: a line this close above or below the body of the page belongs to it
PRINT_NEAR = 10           # proxy pixels (2% of the frame's width): a hand this close to the print counts, for choosing a frame, as on it
PRINT_SHARED = 0.2        # a patch of the map is kept when this share of it lies on cells that usually hold print
CACHE_VERSION = "-v23"    # v2: only patches that reach past the page edge count; paper share added. v13: a second share, over the print map


@dataclass
class HandModel:
    lut: np.ndarray        # (BINS, BINS) bool: this chroma means "covered"
    samples: int           # intruder pixels the model was built from
    usable: bool


def _bins(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.clip((lab[..., 1].astype(np.int32) - LOW) * BINS // SPAN, 0, BINS - 1)
    b = np.clip((lab[..., 2].astype(np.int32) - LOW) * BINS // SPAN, 0, BINS - 1)
    return a, b


def build_model(analysis: Analysis, *, proxy_dir: Path = DEFAULT_PROXY_DIR, page: np.ndarray | None = None, print_level: float = 0.0) -> HandModel:
    """Learn this video's intruder colours and page colours.

    Called plainly, anything outside the page that differs from the median
    frame is an intruder, and the page's colours are all that is ever seen
    inside the page region. That is the model the covered share of the print
    cells (``Hands.on_print``) has always been measured with.

    With ``page`` and ``print_level`` (from ``print_maps``) a second, stricter
    model is made for the print map, which reaches further than the page
    region and has to hold up where a hand lies in most frames:

    - an intruder also has to change the colour of the background, not only
      its brightness: the arm throws a shadow on the wall beside the book,
      and a shadowed wall is not a hand;
    - the page's own colours are taken over ``page``, everywhere except where
      the colour is not the median frame's (a hand that lies on the page in
      half the frames would otherwise make its colour the page's), and
      wherever a frame shows print, whatever its colour (the book creeps, and
      paper in the light of a lamp ends up where the median has shadow).
    """
    refined = page is not None
    proxy = proxy_dir / f"{analysis.slug}.mp4"
    step = max(1, analysis.frames // SAMPLE_FRAMES)
    frames = grab_frames(proxy, range(0, analysis.frames, step))
    median = analysis.median.astype(np.int16)
    median_lab = cv2.cvtColor(analysis.median, cv2.COLOR_BGR2LAB).astype(np.int16)
    outside = cv2.dilate(analysis.mask.astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
    inside = page if refined else analysis.inner
    intruder = np.zeros((BINS, BINS), dtype=np.float64)
    any_change = np.zeros((BINS, BINS), dtype=np.float64)
    own_colours = np.zeros((BINS, BINS), dtype=np.float64)
    for frame in frames.values():
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        a, b = _bins(lab)
        moved = np.abs(frame.astype(np.int16) - median).max(axis=2) > FOREGROUND
        light = lab[..., 0] > MIN_LIGHT
        chosen = moved & outside & light
        np.add.at(any_change, (a[chosen], b[chosen]), 1)
        own = inside
        if refined:
            coloured = np.hypot(lab[..., 1] - median_lab[..., 1], lab[..., 2] - median_lab[..., 2]) > CHROMA_CHANGE
            np.add.at(intruder, (a[chosen & coloured], b[chosen & coloured]), 1)
            own = inside & ~(moved & coloured)
            if print_level > 0:
                own |= inside & (_fine_detail(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) >= print_level)
        np.add.at(own_colours, (a[own], b[own]), 1)
    if not refined or intruder.sum() < MIN_SAMPLES:
        intruder = any_change       # the plain model; or a hand of the background's own colour, where all that changed has to do
    samples = int(intruder.sum())
    if samples < MIN_SAMPLES or own_colours.sum() == 0:
        return HandModel(np.zeros((BINS, BINS), dtype=bool), samples, False)
    intruder = cv2.GaussianBlur(intruder / intruder.sum(), (0, 0), 1.0)
    own_colours = cv2.GaussianBlur(own_colours / own_colours.sum(), (0, 0), 1.0)
    lut = (intruder > LIKELIHOOD_RATIO * own_colours) & (intruder > 1e-5)
    return HandModel(lut, samples, True)


def covered(frame_bgr: np.ndarray, model: HandModel, region: np.ndarray) -> np.ndarray:
    """Bool image of the page pixels of one proxy frame that something covers.

    A hand is attached to an arm, so whatever covers the page reaches it from
    beyond its edge. A patch of the same colours that lies wholly inside the
    page is printed there: a colour plate, a rubricated title, a sepia
    engraving. Only patches that extend past the page region are counted.
    """
    if not model.usable:
        return np.zeros(region.shape, dtype=bool)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    a, b = _bins(lab)
    hit = (model.lut[a, b] & (lab[..., 0] > MIN_LIGHT)).astype(np.uint8)
    hit = cv2.morphologyEx(hit, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hit)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_BLOB
    reaches_out = np.zeros(count, dtype=bool)
    reaches_out[np.unique(labels[~region])] = True
    return (keep & reaches_out)[labels] & region


def print_area(analysis: Analysis) -> np.ndarray:
    """Bool image of the cells in which print usually is on this book's pages."""
    cells = print_cells(analysis)
    full = cv2.resize(cells.astype(np.uint8), (analysis.mask.shape[1], analysis.mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (full > 0) & analysis.inner


_fine_detail = handnet.fine_detail


def _hand_coloured(frame_bgr: np.ndarray, model: HandModel) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    a, b = _bins(lab)
    return model.lut[a, b] & (lab[..., 0] > MIN_LIGHT)


def calm_frames(analysis: Analysis, count: int) -> list[int]:
    """Up to ``count`` frames, spread over the video, in which the book lies still (or, in a video that never rests, its calmest)."""
    motion = analysis.page_motion
    moving = np.maximum(motion, np.concatenate([motion[1:], motion[-1:]]))
    calm = np.flatnonzero(moving <= PRINT_CALM)
    if len(calm) < count // 3:
        calm = np.sort(np.argsort(moving, kind="stable")[:count])
    picks = calm[np.linspace(0, len(calm) - 1, min(count, len(calm))).astype(int)] if len(calm) else calm
    return sorted({int(i) for i in picks})


def _print_map(analysis: Analysis, cells: np.ndarray, detail: np.ndarray, visible: np.ndarray, gray: np.ndarray, usual_detail: float) -> np.ndarray:
    """The print map of one stretch of the video, from its frames at rest (see ``print_maps``)."""
    seen = visible.sum(axis=0).astype(np.float32)
    count = np.maximum(seen, 1)
    printed = ((detail >= PRINT_LEVEL * usual_detail) & visible).sum(axis=0) / count
    light = gray.astype(np.float32)
    mean = (light * visible).sum(axis=0) / count
    change = np.sqrt(np.maximum((light * light * visible).sum(axis=0) / count - mean * mean, 0))
    usual_change = float(np.median(change[cells]))
    found = (printed >= PRINT_SHARE) & (seen >= min(MIN_SEEN, len(detail) // 3)) & (change >= PRINT_CHANGE * usual_change)
    reach = 2 * PRINT_REACH + 1
    found &= cv2.dilate(analysis.mask.astype(np.uint8), np.ones((reach, reach), np.uint8)) > 0
    found = cv2.morphologyEx(found.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # A running head or a folio stands a line or two clear of the body of the
    # page; joining upwards and downwards makes them one patch with it. The
    # stack of page edges beside the print is a patch of its own across a
    # margin, and is dropped for sharing nothing with the cells that usually
    # hold print.
    found = cv2.morphologyEx(found, cv2.MORPH_CLOSE, np.ones((PRINT_JOIN, 3), np.uint8))
    found = cv2.morphologyEx(found, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(found)
    if labels_count > 1:
        shared = np.bincount(labels[cells], minlength=labels_count)
        keep = shared >= PRINT_SHARED * stats[:, cv2.CC_STAT_AREA]
        keep[0] = False
        found = keep[labels].astype(np.uint8)
    found = found > 0
    # Too little found to be the pages (few frames at rest, or a book of plates): what usually holds print has to do.
    if found.sum() < 0.5 * cells.sum():
        return cells
    # A page of print is a rectangle, seen a little askew. Where a hand rests
    # in nearly every frame, no frame shows the print under it and the map
    # has a hole there: most often the top outer corner of the right page,
    # with the running head and the page number. Filled out page by page to
    # the smallest rectangle around what was found, a hand lying there counts
    # as lying on the print, which it is. (A convex hull is not enough: with
    # the corner missing it is cut off diagonally, IMG-0044, Balthasar.)
    # The two pages are told apart at the fold, as a vertical line between the
    # page cells of the left page and those of the right; not by those cells
    # themselves, nor by the page region: both were grown from where print is
    # usually seen, and stop short exactly where the hand rests (the outer
    # top corner of the right page, IMG-0056, Les paraphrases, Balthasar).
    fold = fold_column(analysis, found.shape[1])
    filled = found.copy()
    for lo, hi in ((0, fold), (fold, found.shape[1])):
        part = np.zeros_like(found)
        part[:, lo:hi] = found[:, lo:hi]
        if part.sum() < 50:
            continue
        ys, xs = np.nonzero(part)
        box = cv2.boxPoints(cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32)))
        block = np.zeros(found.shape, dtype=np.uint8)
        cv2.fillConvexPoly(block, box.round().astype(np.int32), 1)
        block[:, :lo] = 0
        block[:, hi:] = 0
        filled |= block > 0
    return filled


def fold_column(analysis: Analysis, width: int) -> int:
    """Column of the fold in an image ``width`` wide: midway between the page cells of the left page and those of the right."""
    left_columns = np.flatnonzero(analysis.grid_left.any(axis=0))
    right_columns = np.flatnonzero(analysis.grid_right.any(axis=0))
    if len(left_columns) and len(right_columns):
        return int(round((left_columns.max() + 1 + right_columns.min()) / 2 * width / analysis.grid_left.shape[1]))
    return width // 2


def print_maps(analysis: Analysis, model: HandModel, *, proxy_dir: Path = DEFAULT_PROXY_DIR,
               detections: handnet.Detections | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Where print is on this book's pages when nothing covers it, for each stretch of the video.

    Returns the maps (K, 270, 480) bool, the frame from which each applies
    (K,), and the ``_fine_detail`` level from which a spot shows print.

    A spot holds print when, in the frames in which the book lies still and
    nothing hand-coloured is over the spot, it mostly shows fine detail, and
    its brightness changes from spread to spread as print does. The second
    condition leaves out what is as detailed but fixed: the clips of the
    stand, a textured wall.

    This is asked per spot of the frames in which the spot is uncovered, not
    of all frames. Where a hand rests most of the time (the outer top corner
    of the right page, with some readers) the usual picture is the hand, and
    a map made from what is usual has a hole exactly where it is needed.

    The book does not stay put: as the leaves move from the right side to
    the left, the pages of a thick book shift by more than the width of a
    margin, and one map for the whole video puts print on the margin of its
    first pages. A map per stretch of the video (``PRINT_PARTS``) follows
    that, but from the few frames at rest a stretch has it comes out ragged,
    so one map is made; the second look in ``measure`` (print one can see is
    not covered) is what keeps a lit margin from counting.
    """
    cells = print_area(analysis)
    one = (cells[None], np.zeros(1, dtype=np.int64), 0.0)
    if not cells.any():
        return one
    frames = grab_frames(proxy_dir / f"{analysis.slug}.mp4", calm_frames(analysis, PRINT_FRAMES * PRINT_PARTS))
    if not frames:
        return one
    order = np.array(sorted(frames))
    shape = analysis.mask.shape
    detail = np.zeros((len(order), *shape), dtype=np.uint8)
    gray = np.zeros((len(order), *shape), dtype=np.uint8)
    visible = np.ones((len(order), *shape), dtype=bool)
    grow = np.ones((5, 5), np.uint8)
    for position, index in enumerate(order):
        frame = frames[int(index)]
        gray[position] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detail[position] = _fine_detail(gray[position])
        if model.usable:
            visible[position] = cv2.dilate(_hand_coloured(frame, model).astype(np.uint8), grow) == 0
    usual_detail = float(np.median([np.median(d[cells]) for d in detail]))
    if detections is not None:
        # The trained models see hands the colour model misses; where they
        # see one, the frame says nothing about whether there is print.
        for position, index in enumerate(order):
            if int(index) < len(detections):
                hand = detections.mask(int(index))
                visible[position] &= cv2.dilate(hand.astype(np.uint8), grow) == 0
    parts = max(1, min(PRINT_PARTS, len(order) // MIN_PART_FRAMES))
    groups = np.array_split(np.arange(len(order)), parts)
    maps = np.array([_print_map(analysis, cells, detail[g], visible[g], gray[g], usual_detail) for g in groups])
    starts = np.array([0] + [(int(order[a[-1]]) + int(order[b[0]])) // 2 + 1 for a, b in zip(groups, groups[1:])], dtype=np.int64)
    return maps, starts, PRINT_LEVEL * usual_detail


@dataclass
class Hands:
    usable: bool
    samples: int
    share: np.ndarray        # (N,) share of the page covered in each frame
    on_print: np.ndarray     # (N,) share of the cells that usually hold print that is covered in each frame
    paper: np.ndarray        # (N,) share of the page region that looks like paper: near 1 for an open book, near 0 for a cover or an empty stand
    on_map: np.ndarray | None = None   # (N,) share of the print map covered in each frame: sees a hand where one often lies (see ``measure``)
    where: np.ndarray | None = None    # (270, 480) bool: the print map
    net: np.ndarray | None = None      # (N,) share of the print covered by hands the trained model found (``handnet``); None if it was not run
    net_page: np.ndarray | None = None # (N,) share of the page region the same hands cover, print or margin
    net_near: np.ndarray | None = None # (N,) the same hands on the print or within PRINT_NEAR of it, as a share of the print's area
    net_left: np.ndarray | None = None # (N,) as ``net_near``, for the left page alone (left of the fold)
    net_right: np.ndarray | None = None  # (N,) and for the right page alone
    net_inside: np.ndarray | None = None # (N,) as ``net``, of the print PRINT_INSIDE in from its edge: what is flagged


def _mask_key(analysis: Analysis) -> str:
    """Changes whenever the page region or the trained models' detections do, so stale measurements are not reused."""
    return hashlib.md5(np.packbits(analysis.mask).tobytes()).hexdigest() + CACHE_VERSION + handnet.key(analysis.slug)


def measure(analysis: Analysis, *, proxy_dir: Path = DEFAULT_PROXY_DIR, cache_dir: Path = DEFAULT_HANDS_DIR, refresh: bool = False) -> Hands:
    """What covers the page in every frame; cached on disk.

    Two shares of covered print are measured, and they are kept apart because
    they fail differently.

    ``on_print`` is over the cells that usually hold print, with the plain
    colour model. It is blind where a hand lies most of the time (those cells
    do not "usually hold print"), but what it does report has been checked
    against printed page numbers over the whole corpus, and it can be compared
    between any two frames of a video.

    ``on_map`` is over the print map, from a little inside its edge, with the
    stricter colour model, and only where the frame shows no print. It sees
    the hand on the corner the first one misses. It also has fixed errors of
    its own, the same in every frame of a spread: the map is made for the
    whole video and the book creeps, so on some spreads a lit margin counts.
    It therefore only ever says that one frame of a spread has more over the
    print than another frame of the same spread.
    """
    cache = cache_dir / f"{analysis.slug}.npz"
    proxy = proxy_dir / f"{analysis.slug}.mp4"
    key = _mask_key(analysis)
    if cache.exists() and not refresh and cache.stat().st_mtime >= proxy.stat().st_mtime:
        with np.load(cache) as data:
            if "key" in data and str(data["key"]) == key:
                where = np.unpackbits(data["where"])[:analysis.mask.size].reshape(analysis.mask.shape).astype(bool)
                return Hands(bool(data["usable"]), int(data["samples"]), data["share"], data["on_print"], data["paper"], data["on_map"], where,
                             data["net"] if "net" in data and data["net"].size else None,
                             data["net_page"] if "net_page" in data and data["net_page"].size else None,
                             data["net_near"] if "net_near" in data and data["net_near"].size else None,
                             data["net_left"] if "net_left" in data and data["net_left"].size else None,
                             data["net_right"] if "net_right" in data and data["net_right"].size else None,
                             data["net_inside"] if "net_inside" in data and data["net_inside"].size else None)
    model = build_model(analysis, proxy_dir=proxy_dir)
    inner = analysis.inner
    cells = print_area(analysis)
    area, cells_area = max(1, int(inner.sum())), max(1, int(cells.sum()))
    detections = handnet.load(analysis.slug)
    maps, starts, level = print_maps(analysis, model, proxy_dir=proxy_dir, detections=detections)
    everywhere = maps.any(axis=0)
    strict = build_model(analysis, proxy_dir=proxy_dir, page=inner | everywhere, print_level=level)
    # The hand that holds the book lies along the edge of the print in almost
    # every frame, and whether it touches the first letters of a line cannot
    # be told at this size. What counts is a hand further in than that.
    edge = 2 * PRINT_INSIDE + 1
    stretches = [(inner | text,       # the map reaches into corners the page region leaves out
                  cv2.erode(text.astype(np.uint8), np.ones((edge, edge), np.uint8)) > 0, max(1, int(text.sum()))) for text in maps]
    part_of = np.searchsorted(starts, np.arange(analysis.frames), side="right") - 1
    share = np.zeros(analysis.frames, dtype=np.float32)
    on_print = np.zeros(analysis.frames, dtype=np.float32)
    on_map = np.zeros(analysis.frames, dtype=np.float32)
    paper = np.ones(analysis.frames, dtype=np.float32)
    colour = paper_colour(analysis.median, cells) if cells.any() else None
    text_all = everywhere | cells
    text_all_area = max(1, int(text_all.sum()))
    # What is flagged is a hand well inside the print (``net_inside``): the
    # map is filled out to a rectangle per page, and the thumb that holds the
    # book lies on its edge in nearly every frame (Einar: 68 of 116 captures
    # flagged that the checkers all call clean). Choosing a frame within a
    # spread uses the near zone (``net_near``), and choosing between
    # candidates of a spread the whole map (``net``).
    inside = cv2.erode(text_all.astype(np.uint8), np.ones((edge, edge), np.uint8)) > 0
    inside_area = max(1, int(inside.sum()))
    net = np.zeros(analysis.frames, dtype=np.float32) if detections is not None and text_all.any() else None
    net_page = np.zeros(analysis.frames, dtype=np.float32) if net is not None else None
    net_near = np.zeros(analysis.frames, dtype=np.float32) if net is not None else None
    net_inside = np.zeros(analysis.frames, dtype=np.float32) if net is not None else None
    # The hand that turns the leaves takes hold of the corner of the right
    # page above and beside its print, and lifts it; its fingers lie just
    # outside the print map. Counted only on the print, it costs a frame
    # less than the thumb that holds the book on the bottom line of the left
    # page, and the frame with the leaf lifted wins (Balthasar, IMG-0053).
    near = cv2.dilate(text_all.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PRINT_NEAR + 1,) * 2)) > 0
    # And per page, so that each page can be taken from the frame in which it is clearest (``spreads.group_spreads``).
    fold = fold_column(analysis, text_all.shape[1])
    near_left, near_right = near.copy(), near.copy()
    near_left[:, fold:] = False
    near_right[:, :fold] = False
    left_area = max(1, int(text_all[:, :fold].sum()))
    right_area = max(1, int(text_all[:, fold:].sum()))
    net_left = np.zeros(analysis.frames, dtype=np.float32) if net is not None else None
    net_right = np.zeros(analysis.frames, dtype=np.float32) if net is not None else None
    page_area = max(1, int(analysis.mask.sum()))
    index = 0
    for _, chunk in iter_proxy(proxy, chunk_frames=64):
        for frame in chunk:
            if index < analysis.frames:
                detail = None
                if net is not None and index < len(detections):
                    detail = _fine_detail(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    hand = detections.mask(index)
                    net[index] = (hand & text_all).sum() / text_all_area
                    net_inside[index] = (hand & inside).sum() / inside_area
                    net_page[index] = (hand & analysis.mask).sum() / page_area
                    net_near[index] = (hand & near).sum() / text_all_area
                    net_left[index] = (hand & near_left).sum() / left_area
                    net_right[index] = (hand & near_right).sum() / right_area
                if model.usable:
                    hit = covered(frame, model, inner)
                    share[index] = hit.sum() / area
                    on_print[index] = (hit & cells).sum() / cells_area
                if strict.usable:
                    region, well_inside, text_area = stretches[part_of[index]]
                    hit = covered(frame, strict, region) & well_inside
                    if hit.any():
                        # Paper in the light of a lamp is the colour of skin,
                        # and so is a sepia page. Skin is smooth, though:
                        # where the frame shows print, nothing lies over it.
                        hit &= (detail if detail is not None else _fine_detail(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) < level
                    on_map[index] = hit.sum() / text_area
                if colour is not None:
                    paper[index] = (paper_like(frame, colour) & inner).sum() / area
            index += 1
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, usable=model.usable, samples=model.samples, share=share, on_print=on_print, paper=paper, on_map=on_map, key=key,
                        where=np.packbits(everywhere), net=net if net is not None else np.zeros(0, dtype=np.float32),
                        net_page=net_page if net_page is not None else np.zeros(0, dtype=np.float32),
                        net_near=net_near if net_near is not None else np.zeros(0, dtype=np.float32),
                        net_left=net_left if net_left is not None else np.zeros(0, dtype=np.float32),
                        net_right=net_right if net_right is not None else np.zeros(0, dtype=np.float32),
                        net_inside=net_inside if net_inside is not None else np.zeros(0, dtype=np.float32))
    return Hands(model.usable, model.samples, share, on_print, paper, on_map, everywhere, net, net_page, net_near, net_left, net_right, net_inside)
