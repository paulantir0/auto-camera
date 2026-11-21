"""Split a video into spreads by finding page turns.

The detector rests on one physical fact. Turning a leaf lifts it off the right
page and lands it on the left page, so every turn ends with the whole left page
being swept and then coming to rest. A hand that reaches in, presses a page
flat or re-grips the book moves only part of a page. Counting full sweeps of
the left page therefore counts turns, with no model of what a hand looks like
and no dependence on what is printed.

Between two landings the spread is on show. The leaf for the next turn lifts
off the right page before it lands, so the usable part of that interval ends
when the right page has been fully swept again.

Everything here works on the motion cells from ``analysis``; the video is only
touched to fetch the few frames compared in ``merge_repeats``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .analysis import MIN_PRINT_CELLS, Analysis, find_valleys, grab_frames, page_changed, print_cells
from .proxy import DEFAULT_PROXY_DIR

LAND_COVERAGE = 0.90         # share of the left page that must have moved for a landing
LIFT_COVERAGE = 0.90         # once this much of the right page has moved, the next leaf is up
SHARP_ENOUGH = 0.60          # a frame this sharp, relative to the spread's best, may be chosen for being hand-free
HAND_STEP = 0.0025           # covered print is compared in steps of a quarter of a percent
SAME_SPREAD_CHANGED = 0.03   # two frames of one spread differ in less than this share of the page
SAME_SPREAD_SHARE = 0.3      # ... and in less than this share of what two different spreads of the video differ in
FAR_APART = 8                # spreads this many apart are certainly different
CALM_MOTION = 0.15           # page motion at or under this is "at rest"
CALM_SLACK = 0.10            # when a spread is never at rest, frames this close to its calmest moment may be chosen
WEAK_LAND_COVERAGE = 0.75    # a sweep this complete counts as a landing inside an interval that is far too long
LONG_INTERVAL = 1.8          # ... that is, one at least this many times the video's usual interval
SAME_FALL = 3                # frames: a left page swept again this soon after a "rest" that was only a slow-down is the same leaf, still coming down
HIDDEN_LEVEL = 0.4           # a cell holds print in a spread when its calm frames show this share of the video's strong print detail there
HIDDEN_DROP = 0.5            # and a frame hides that print when it shows under this share of it
HIDDEN_GROUP = 4             # in at least this many cells side by side
HIDDEN_PAPER = 0.9           # ... that are darker than this share of the frame's bare paper: a blank stretch of page hides nothing
CLEAR_MARGIN = 0.005         # the two newer measures count in whole steps of this size: under half a percent they are noise
HAND_NEAR = 2                # frames either side: in a frame that moves, a hand the models found this close counts too (a moving hand is a blur they miss)
NEAR_MOTION = 0.10           # ... page motion above which a frame counts as moving for that
PAGE_HAND_STEP = 0.01        # hands anywhere on the page are compared in steps of a percent of it
UNSETTLED_STEPS = 8          # between captures of one spread, never coming to rest counts as much as this many HAND_STEPs over the print (2%)
UNSETTLED = 0.15             # page motion around the chosen frame above which a capture never came to rest
TWO_FRAMES = False           # put a spread together from two frames, one per page; tried and off (v14, 2026-11-13): the spreads it joined were no cleaner
TWO_FRAMES_STEPS = 2         # ... when that has this many HAND_STEPs less hand than the best single frame


@dataclass
class Spread:
    index: int
    start: int                 # frame at which the leaf had covered the left page
    end: int                   # last frame before the next leaf covers the left page
    best: int = -1             # chosen frame
    best_motion: float = -1.0  # page motion around the chosen frame; high means it never came to rest
    detail: float = 0.0        # fine-detail energy of the chosen frame
    hand: float = 0.0          # share of the page covered by a hand in the chosen frame
    hand_on_print: float = 0.0 # share of the print covered in the chosen frame, by the largest of the three measures in ``group_spreads``
    hand_inside: float = 0.0   # the same with the trained models' hand counted only well inside the print: what review lists use
    print_hidden: float = 0.0  # share of the spread's printed cells the chosen frame does not show though another of its calm frames does
    hand_on_page: float = 0.0  # share of the page region the trained models' hands cover in the chosen frame, print or margin
    hand_free_exists: bool = True  # some frame of this spread had no hand on the print
    before_lift: bool = True   # the chosen frame precedes the next full sweep of the right page
    landing_cover: float = 0.0 # how much of the left page the landing swept
    paper: float = 1.0         # share of the page region that looks like paper in the chosen frame
    weak_landing: bool = False # found on the second look, in an interval that was far too long
    left_frame: int = -1       # frame the left page is taken from, when not ``best`` (see TWO_FRAMES_STEPS); -1: ``best``
    right_frame: int = -1      # and the right page
    fold: float = 0.5          # where the two halves meet, as a share of the frame's width

    def as_dict(self, fps: float) -> dict:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "best": self.best,
            "best_seconds": round(self.best / fps, 3) if self.best >= 0 else None,
            "best_motion": round(self.best_motion, 4),
            "detail": round(self.detail, 2),
            "hand": round(self.hand, 4),
            "hand_on_print": round(self.hand_on_print, 4),
            "hand_inside": round(self.hand_inside, 4),
            "print_hidden": round(self.print_hidden, 4),
            "hand_on_page": round(self.hand_on_page, 4),
            "hand_free_exists": self.hand_free_exists,
            "before_lift": self.before_lift,
            "landing_cover": round(self.landing_cover, 3),
            "paper": round(self.paper, 3),
            "weak_landing": self.weak_landing,
            "left_frame": self.left_frame if self.left_frame >= 0 else self.best,
            "right_frame": self.right_frame if self.right_frame >= 0 else self.best,
            "fold": round(self.fold, 4),
        }


@dataclass
class Turns:
    landings: list[int]         # frames at which a leaf had covered the left page
    confirmed: list[int]        # frames at which the left page next came to rest
    landing_cover: list[float]  # left-page coverage when each landing was confirmed
    weak: list[bool]            # the landing was found on the second look (see ``rescue_weak_landings``)
    cover_left: np.ndarray      # (N,) share of the left page moved since the last confirmed landing
    cover_right: np.ndarray     # (N,) share of the right page moved since the last landing
    settled: list[int] | None = None  # per landing, the last full sweep of the same fall (``one_landing_per_fall``); None: the landings themselves


def detect_turns(analysis: Analysis, *, land_coverage: float = LAND_COVERAGE) -> Turns:
    """Find page turns as full sweeps of the left page.

    Coverage of the left page accumulates frame by frame. When it passes
    ``land_coverage`` a leaf has come down, and that frame starts the new
    spread. The accumulator is only cleared once the left page is next at rest,
    so the tail of a landing and the leaf settling are not counted again.
    """
    count = analysis.frames
    cover_left = np.zeros(count, dtype=np.float32)
    cover_right = np.zeros(count, dtype=np.float32)
    rest_points = set(find_valleys(analysis.left_motion))
    left, right = analysis.grid_left, analysis.grid_right
    # Coverage is counted over cells that normally hold print. A blank margin
    # does not change when a leaf lands on it, so counting it lets a real
    # landing fall short of the threshold.
    printed = print_cells(analysis)
    if int((left & printed).sum()) >= MIN_PRINT_CELLS:
        left = left & printed
    if int((right & printed).sum()) >= MIN_PRINT_CELLS:
        right = right & printed
    left_area, right_area = max(1, int(left.sum())), max(1, int(right.sum()))
    swept_left = np.zeros(left.shape, dtype=bool)
    swept_right = np.zeros(left.shape, dtype=bool)
    landings: list[int] = []
    confirmed: list[int] = []
    landing_cover: list[float] = []
    crossed = -1
    for index in range(1, count):
        moved = analysis.moved[index]
        swept_left |= moved
        swept_right |= moved
        cover_left[index] = swept_left[left].sum() / left_area
        if crossed < 0 and cover_left[index] >= land_coverage:
            crossed = index
            swept_right[:] = False
        cover_right[index] = swept_right[right].sum() / right_area
        if crossed >= 0 and index in rest_points:
            landings.append(crossed)
            confirmed.append(index)
            landing_cover.append(float(cover_left[index]))
            swept_left[:] = False
            crossed = -1
    turns = rescue_weak_landings(Turns(landings, confirmed, landing_cover, [False] * len(landings), cover_left, cover_right), rest_points)
    return one_landing_per_fall(turns, analysis.left_motion)


def one_landing_per_fall(turns: Turns, left_motion: np.ndarray) -> Turns:
    """Count a leaf that slows down in the air once, not twice.

    A landing is confirmed at the left page's next moment of rest, and
    ``find_valleys`` also takes a clear dip between two bursts of motion for
    one, so that a spread flipped past is not lost. A falling leaf dips like
    that too: it slows for a frame halfway down. The landing was then
    confirmed and the sweep cleared with the leaf still in the air, the rest
    of the fall swept the left page a second time, and the few frames between
    the two stood as a spread of their own in which nothing is still (Roberts,
    frames 3155 to 3159: left-page motion 0.75, 0.42, 0.81, two hands and the
    leaf in every frame). A second leaf cannot be lifted off the right page
    and brought down on the left within ``SAME_FALL`` frames of the first, so
    a full sweep that follows that soon on a "rest" that was only such a dip
    belongs to the same fall, and the two are one landing.

    The spread starts at the first sweep and ``settled`` records the last,
    from where the right page is watched for the next leaf lifting. The
    frames between the two stay with the spread they lead into: they are
    chosen from like any others, which is to say only when the spread never
    gets calmer than they are. Handing them to the spread before instead
    (tried first) threw away the only clear view of a spread whose landing
    was followed by a hand (IMG-0032, 2500 for 2508) and dropped frames that
    had tied a mid-turn view to its spread (IMG-0037, 8849). A landing
    confirmed with the left page at rest is left alone.
    """
    count = len(turns.landings)
    same_fall = [k + 1 < count and left_motion[turns.confirmed[k]] >= CALM_MOTION and turns.landings[k + 1] - turns.confirmed[k] <= SAME_FALL
                 for k in range(count)]
    if not any(same_fall):
        return turns
    landings, confirmed, cover, weak, settled = [], [], [], [], []
    first = 0
    while first < count:
        last = first
        while same_fall[last]:
            last += 1
        landings.append(turns.landings[first])
        settled.append(turns.landings[last])
        confirmed.append(turns.confirmed[last])
        cover.append(turns.landing_cover[last])
        weak.append(any(turns.weak[first:last + 1]))
        first = last + 1
    return Turns(landings, confirmed, cover, weak, turns.cover_left, turns.cover_right, settled)


def rescue_weak_landings(turns: Turns, rest_points: set[int]) -> Turns:
    """Look again inside intervals that are far too long for one spread.

    A leaf that lands askew, or on a page whose print cells the mask only
    partly covers, can sweep a little under ``LAND_COVERAGE`` of the left page.
    The turn is then missed and two spreads share one interval, which shows as
    an interval about twice the usual length. Only in such intervals is a
    weaker sweep (``WEAK_LAND_COVERAGE``) followed by the page coming to rest
    taken as a landing. A hand pressing the page can sweep that much too, so
    this is not done everywhere; and where it splits one spread in two, both
    halves show the same pages and are merged again when captures are chosen.
    """
    if len(turns.landings) < 6:
        return turns
    lengths = np.diff(turns.landings)
    usual = float(np.median(lengths))
    added: list[tuple[int, int, float]] = []
    for k in range(len(turns.landings) - 1):
        start, stop = turns.confirmed[k], turns.landings[k + 1]
        if stop - turns.landings[k] < LONG_INTERVAL * usual:
            continue
        crossing = next((i for i in range(start + 1, stop) if turns.cover_left[i] >= WEAK_LAND_COVERAGE), None)
        if crossing is None:
            continue
        rest = next((i for i in range(crossing, stop) if i in rest_points), None)
        # Each half must be long enough to be a spread of its own.
        if rest is None or crossing - turns.landings[k] < 0.5 * usual or stop - rest < 0.5 * usual:
            continue
        added.append((crossing, rest, float(turns.cover_left[rest])))
    if not added:
        return turns
    merged = sorted([(l, c, v, False) for l, c, v in zip(turns.landings, turns.confirmed, turns.landing_cover)] + [(l, c, v, True) for l, c, v in added])
    return Turns([m[0] for m in merged], [m[1] for m in merged], [m[2] for m in merged], [m[3] for m in merged], turns.cover_left, turns.cover_right)


def frame_quality(analysis: Analysis) -> np.ndarray:
    """Per-frame score for "this is a good still of the spread".

    Print that is sharp and not covered has high fine-detail energy; a hand over
    the text, a leaf in the air and motion blur all lower it. That is multiplied
    by how still the page was on either side of the frame. Nothing here knows
    what a hand looks like.
    """
    motion = analysis.page_motion
    after = np.concatenate([motion[1:], motion[-1:]])
    return analysis.page_detail * (1.0 - 0.5 * (motion + after))


def cell_light(thumbs: np.ndarray) -> np.ndarray:
    """Mean brightness per motion cell, (F, rows, columns), from the stored thumbnails."""
    frames, height, width = thumbs.shape
    return thumbs.reshape(frames, height // 2, 2, width // 2, 2).mean(axis=(2, 4))


def hidden_print(detail: np.ndarray, reference: np.ndarray, cells: np.ndarray, level: float, light: np.ndarray | None = None) -> np.ndarray:
    """Per frame of one spread, the share of the spread's printed cells whose print the frame does not show.

    ``detail`` is (F, rows, columns) for the spread's frames and ``reference``
    (F,) marks those that show it at rest. What the spread looks like is taken
    from them: per cell, the fine detail the middle one of them shows. Where a
    frame shows under ``HIDDEN_DROP`` of that, something lies over the print:
    a hand, a sleeve, a leaf standing up. This needs no idea of what a hand
    looks like, so it also sees one the colour model does not.

    The middle one and not the most: one frame at rest with a hand in it, or
    a leaf held up, has fine detail of its own where the page is blank, and
    measured against the most that any frame shows, every clean frame "hides"
    it (Dale, pp. 190-191: 4 to 8% on each clean frame, none on the one with
    the hand). The price is that a hand lying there in more than half of the
    frames at rest becomes what the spread looks like, and is not seen.

    A held book is never in quite the same place in two frames, and a cell on
    the edge of the print has print in one and margin in the next. So only
    cells whose neighbours all hold print are judged, and only
    ``HIDDEN_GROUP`` or more such cells side by side count: a hand is a
    patch, the noise is single cells. A fingertip on the first letters of a
    line is too small for this; that is left to the colour model.

    With ``light`` (cell brightness, same shape as ``detail``) a cell counts
    only if it is darker than bare paper in that frame: a blank stretch of
    page is as bright as paper gets, and hides nothing whatever the other
    frames show there.

    A frame that is a little softer all over is not penalised: its typical
    ratio is divided out. And a hand that lies still for the whole spread is
    not seen: no frame shows what is under it.
    """
    frames = len(detail)
    hidden = np.zeros(frames, dtype=np.float32)
    if not reference.any():
        return hidden
    shown = np.median(detail[reference], axis=0).astype(np.float32)
    solid = cv2.erode(shown, np.ones((3, 3), np.uint8))
    printed = cells & (solid >= max(level, 1.0))
    count = int(printed.sum())
    if count < MIN_PRINT_CELLS:
        return hidden
    for index in range(frames):
        overall = float(np.median(detail[index][printed] / shown[printed]))
        low = printed & (detail[index] < HIDDEN_DROP * overall * shown)
        if light is not None and low.any():
            low &= light[index] < HIDDEN_PAPER * float(np.percentile(light[index][cells], 95))
        if int(low.sum()) < HIDDEN_GROUP:
            continue
        groups, labels, stats, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), connectivity=8)
        patches = stats[1:, cv2.CC_STAT_AREA]
        hidden[index] = float(patches[patches >= HIDDEN_GROUP].sum()) / count
    return hidden


def strong_print(analysis: Analysis) -> float:
    """Fine-detail energy of this video's densest print cells (the scale ``print_cells`` measures against)."""
    page = analysis.grid_page
    if not page.any():
        return 0.0
    usual = np.median(analysis.detail[:: max(1, analysis.frames // 600)], axis=0)
    return float(np.percentile(usual[page], 90))


def group_spreads(turns: Turns, analysis: Analysis, hands=None, *, lift_coverage: float = LIFT_COVERAGE) -> list[Spread]:
    """One spread per interval between landings, with its best frame chosen.

    Among the calm frames at least ``SHARP_ENOUGH`` as good as the best, the
    one that hides the least print is taken, and among those the best: a
    slightly softer frame with the text clear beats a sharper one with a hand
    on it. How much print a frame hides is the largest of three measures: the
    covered share of the print cells (``hands.on_print``), and, beyond what
    the spread's cleanest calm frame has, the covered share of the print map
    (``hands.on_map``) and the print that other frames of the spread show and
    this one does not (``hidden_print``).
    """
    motion = analysis.page_motion
    count = len(motion)
    quality = frame_quality(analysis)
    use_hands = hands is not None and hands.usable
    level = HIDDEN_LEVEL * strong_print(analysis)
    cells = cv2.dilate(analysis.grid_page.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    light = cell_light(analysis.thumbs)
    from .hands import fold_column
    fold_share = fold_column(analysis, analysis.grid_left.shape[1] * 100) / (analysis.grid_left.shape[1] * 100)
    starts = [0] + turns.landings
    settles = [0] + (turns.settled if turns.settled is not None else turns.landings)
    covers = [0.0] + turns.landing_cover
    weak = [False] + turns.weak
    spreads: list[Spread] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] - 1 if position + 1 < len(starts) else count - 1
        if end < start:
            continue
        window = quality[start:end + 1]
        # Only the calm part of the interval is eligible: frames at rest, or,
        # when the spread never comes to rest, those close to its calmest
        # moment. A frame taken while a leaf is in the air must not win for
        # being sharp or for having less hand in it.
        moving = np.maximum(motion[start:end + 1], np.concatenate([motion[start + 1:end + 1], motion[end:end + 1]]))
        calm = moving <= max(CALM_MOTION, float(moving.min()) + CALM_SLACK)
        window = np.where(calm, window, -np.inf)
        # The right page is watched from where this spread's own leaf came down: when it slowed in the air and swept the
        # left page twice (``one_landing_per_fall``), that is the second sweep, where the right page's count was last cleared.
        lift = settles[position]
        for frame in range(lift + 1, end + 1):
            if turns.cover_right[frame] >= lift_coverage:
                break
            lift = frame
        # What the spread looks like is taken from its calm frames before the
        # next leaf is up, when there are any: after that the right page shows
        # the spread to come.
        reference = calm.copy()
        if calm[:lift - start + 1].any():
            reference[lift - start + 1:] = False
        hidden = hidden_print(analysis.detail[start:end + 1], reference, cells, level, light[start:end + 1])
        # What is over the print, measured four ways. The colour model's
        # cover of the print cells counts as it stands (``hands.on_print``).
        # The others only say that a frame has clearly more over the print
        # than the calm frame of this spread that has least: each carries
        # errors that are the same in every frame of a spread (a lit margin
        # in the print map, the thumb that holds the book shifting by a few
        # pixels against the edge of the print) and wavers by a few tenths of
        # a percent between frames that look alike. So they count beyond the
        # spread's least, and in whole steps of CLEAR_MARGIN.
        net = getattr(hands, "net", None) if hands is not None else None
        net_page = getattr(hands, "net_page", None) if hands is not None else None
        beyond = hidden - float(hidden[calm].min())
        if use_hands and getattr(hands, "on_map", None) is not None:
            beyond = np.maximum(beyond, hands.on_map[start:end + 1] - float(hands.on_map[start:end + 1][calm].min()))
        moving_now = moving > NEAR_MOTION
        primary = np.floor(np.maximum(beyond, 0.0) / CLEAR_MARGIN)
        if use_hands:
            primary = np.maximum(primary, np.round(hands.on_print[start:end + 1] / HAND_STEP))
        if net is not None:
            # A hand on the move is a blur the models miss: in a frame that
            # moves, with a hand found on the print within HAND_NEAR frames
            # before it and within HAND_NEAR after, the hand counts as there.
            # (One side only is a hand arriving or leaving, IMG-0052.) The
            # trained models are exact enough to count in HAND_STEPs, beyond
            # the spread's calm frame with least: the thumb that holds the
            # book is in all of them and shifts by a pixel or two.
            # A hand close beside the print counts as on it here
            # (``hands.net_near``): the hand that lifts the corner of the
            # leaf lies just outside the print.
            ranked = getattr(hands, "net_near", None)
            ranked = net if ranked is None else ranked
            own = ranked[start:end + 1]
            before = np.array([ranked[max(0, i - HAND_NEAR):i].max() if i > 0 else 0.0 for i in range(start, end + 1)], dtype=np.float32)
            after = np.array([ranked[i + 1:i + HAND_NEAR + 1].max() if i + 1 < len(ranked) else 0.0 for i in range(start, end + 1)], dtype=np.float32)
            seen = np.where(moving_now, np.maximum(own, np.minimum(before, after)), own)
            primary = np.maximum(primary, np.round(np.maximum(seen - float(own[calm].min()), 0.0) / HAND_STEP))
        # Next, a hand anywhere on the page: the hand that turns the leaves
        # rests on its corner before it turns, and lifts it.
        if net_page is not None:
            page_own = net_page[start:end + 1]
            secondary = np.round(np.maximum(page_own - float(page_own[calm].min()), 0.0) / PAGE_HAND_STEP)
        else:
            secondary = np.zeros(end - start + 1)
        allowed = calm & (window >= SHARP_ENOUGH * float(window.max()))
        rank = np.where(allowed, primary * 1000 + secondary, np.inf)
        least = rank.min()
        best = start + int(np.argmax(np.where(rank == least, window, -np.inf)))
        # Recorded, and flagged on, is what the chosen frame itself has over
        # the print: the larger of the colour model's and the trained
        # models' share, and the newer measures beyond the spread's least.
        covered = np.floor(np.maximum(hidden - float(hidden[calm].min()), 0.0) / CLEAR_MARGIN) * CLEAR_MARGIN
        if use_hands:
            covered = np.maximum(covered, hands.on_print[start:end + 1])
        if net is not None:
            covered = np.maximum(covered, net[start:end + 1])
        hand_free = bool(np.where(allowed, np.round(covered / HAND_STEP), np.inf).min() == 0)
        # What a person is told to look at counts the trained models' hand
        # only well inside the print (``hands.net_inside``): the holding thumb
        # on the edge of the print map is in nearly every frame. The selection
        # rules after this (repeats, glimpses, the number check) keep using
        # ``covered``.
        net_inside = getattr(hands, "net_inside", None) if hands is not None else None
        inside = np.floor(np.maximum(hidden - float(hidden[calm].min()), 0.0) / CLEAR_MARGIN) * CLEAR_MARGIN
        if use_hands:
            inside = np.maximum(inside, hands.on_print[start:end + 1])
        if net_inside is not None:
            inside = np.maximum(inside, net_inside[start:end + 1])
        elif net is not None:
            inside = np.maximum(inside, net[start:end + 1])
        # Each page from the frame in which it is clearest, when no one frame
        # has both clear: the hand that turns the leaves lies on the right
        # page early and the hand that holds the book on the left page late,
        # or the other way round. The right page is taken only from before
        # the next leaf rises over it.
        left_frame = right_frame = best
        net_left = getattr(hands, "net_left", None) if hands is not None else None
        net_right = getattr(hands, "net_right", None) if hands is not None else None
        if TWO_FRAMES and net_left is not None and net_right is not None:
            offsets = np.arange(end - start + 1)

            def page_steps(per_frame: np.ndarray) -> np.ndarray:
                own = per_frame[start:end + 1]
                before = np.array([per_frame[max(0, i - HAND_NEAR):i].max() if i > 0 else 0.0 for i in range(start, end + 1)], dtype=np.float32)
                after = np.array([per_frame[i + 1:i + HAND_NEAR + 1].max() if i + 1 < len(per_frame) else 0.0 for i in range(start, end + 1)], dtype=np.float32)
                seen = np.where(moving_now, np.maximum(own, np.minimum(before, after)), own)
                return np.round(np.maximum(seen - float(own[calm].min()), 0.0) / HAND_STEP)

            def clearest(steps: np.ndarray, usable: np.ndarray) -> int:
                rank = np.where(usable, steps, np.inf)
                if not np.isfinite(rank).any():
                    return best
                return start + int(np.argmax(np.where(rank == rank.min(), window, -np.inf)))

            left_steps, right_steps = page_steps(net_left), page_steps(net_right)
            left_pick = clearest(left_steps, allowed)
            right_pick = clearest(right_steps, allowed & (offsets <= lift - start))
            single = left_steps[best - start] + right_steps[best - start]
            paired = left_steps[left_pick - start] + right_steps[right_pick - start]
            if (left_pick, right_pick) != (best, best) and paired <= single - TWO_FRAMES_STEPS:
                left_frame, right_frame = left_pick, right_pick
                # What the joined image has over its print: each page's own
                # share, which is the larger of the two in proportion.
                covered = covered.copy()
                covered[best - start] = min(float(covered[best - start]), max(float(net_left[left_pick]), float(net_right[right_pick])))
        after = motion[best + 1] if best + 1 < count else motion[best]
        spreads.append(Spread(index=len(spreads) + 1, start=start, end=end, best=best,
                              best_motion=max(float(motion[best]), float(after)),
                              detail=float(analysis.page_detail[best]),
                              hand=float(hands.share[best]) if use_hands else 0.0,
                              hand_on_print=float(covered[best - start]),
                              hand_inside=float(inside[best - start]),
                              print_hidden=float(max(0.0, hidden[best - start] - hidden[calm].min())),
                              hand_on_page=float(net_page[best]) if net_page is not None else 0.0,
                              hand_free_exists=hand_free,
                              left_frame=left_frame, right_frame=right_frame,
                              fold=fold_share,
                              before_lift=best <= lift, landing_cover=covers[position],
                              paper=float(hands.paper[best]) if hands is not None else 1.0, weak_landing=weak[position]))
    return spreads


def capture_rank(capture: dict) -> tuple[float, float]:
    """Higher is better, between captures of one spread (a candidate's dict, or a capture's in ``spreads.json``).

    First what lies over the print, in HAND_STEPs, with UNSETTLED_STEPS more
    for a capture that never came to rest: a frame taken while a leaf is in
    the air must not win over a still one for having a little less thumb on
    the print (IMG-0055, IMG-0034: a mid-turn frame chosen over a still one
    with the holding thumb on two letters). Then sharp print on a page at
    rest. A hand anywhere on the page is not compared here: between two
    stretches of a video the hand that holds the book moves, and that, not
    the turning hand, is what such a share mostly measures (IMG-0035); it is
    used within a stretch (``group_spreads``).
    """
    steps = round(capture.get("hand_on_print", 0.0) / HAND_STEP)
    if capture["best_motion"] > UNSETTLED:
        steps += UNSETTLED_STEPS
    return (-steps, capture["detail"] * (1.0 - min(1.0, capture["best_motion"])))


def merge_repeats(spreads: list[Spread], analysis: Analysis, *, proxy_dir: Path = DEFAULT_PROXY_DIR,
                  same_changed: float = SAME_SPREAD_CHANGED) -> tuple[list[Spread], list[tuple[int, int, float]]]:
    """Fold a spread into its predecessor when both show the same pages.

    A full sweep of the left page with no change of content is a leaf that was
    lifted and laid back, or a hand passing over the page. Returns the merged
    list and the ``(kept_index, dropped_index, changed)`` pairs.
    """
    if len(spreads) < 2:
        return spreads, []
    inner = analysis.inner
    frames = grab_frames(proxy_dir / f"{analysis.slug}.mp4", [s.best for s in spreads], gray=True)
    blurred = {i: cv2.blur(f, (3, 3)) for i, f in frames.items()}
    # How much two certainly different spreads of this video differ, as this
    # measure sees it. That depends on the print and on how the proxy was
    # scaled (a smoother scaler leaves fewer strongly changed pixels), so "the
    # same" is set relative to it and not only as a fixed share of the page.
    apart = [page_changed(blurred[a.best], blurred[b.best], inner) for a, b in zip(spreads[::2], spreads[FAR_APART::2])]
    if len(apart) >= 5:
        same_changed = min(same_changed, SAME_SPREAD_SHARE * float(np.percentile(apart, 10)))
    merged: list[Spread] = [spreads[0]]
    dropped: list[tuple[int, int, float]] = []
    for spread in spreads[1:]:
        keep = merged[-1]
        changed = page_changed(blurred[keep.best], blurred[spread.best], inner)
        if changed < same_changed:
            dropped.append((keep.index, spread.index, changed))
            keep.end = spread.end
            if capture_rank(spread.as_dict(1.0)) > capture_rank(keep.as_dict(1.0)):
                keep.best, keep.best_motion, keep.detail, keep.before_lift = spread.best, spread.best_motion, spread.detail, spread.before_lift
                keep.hand, keep.hand_on_print, keep.print_hidden, keep.paper = spread.hand, spread.hand_on_print, spread.print_hidden, spread.paper
                keep.hand_on_page = spread.hand_on_page
            keep.hand_free_exists = keep.hand_free_exists or spread.hand_free_exists
        else:
            merged.append(spread)
    for number, spread in enumerate(merged, 1):
        spread.index = number
    return merged, dropped


def segment(analysis: Analysis, hands=None, *, proxy_dir: Path = DEFAULT_PROXY_DIR) -> list[Spread]:
    turns = detect_turns(analysis)
    spreads = group_spreads(turns, analysis, hands)
    spreads, _ = merge_repeats(spreads, analysis, proxy_dir=proxy_dir)
    return spreads
