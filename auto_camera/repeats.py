"""Merge captures that show the same pages.

Some books do not lie flat: a turned leaf springs back up and is pressed down
again, sometimes several times. Each of those is a full sweep of the left page,
so turn detection reports several spreads where there is one, and the frames
differ too much (curled leaf, hand in the way) for pixel comparison to tell.

What does survive is the print itself. Keypoints on the two captures are
matched and fitted to one homography; captures of the same pages agree on
dozens to hundreds of points, captures of different pages on a handful
(measured against pairs labelled by their printed page numbers).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .spreads import capture_rank

MATCH_WIDTH = 1536        # captures are compared at this width
LOOK_BACK = 3             # a capture is compared with this many members of the current group
STATIC_EDGE = 60          # gradient of the median frame above which a pixel is fixed structure
ELSEWHERE_INLIERS = 18    # keypoints one page of a flagged capture must share with the same page of an unflagged one close by
ELSEWHERE_WINDOW = 8      # how many captures either way "close by" reaches
LEGIBLE_SHARE = 0.25      # a half holding this share of a capture's keypoints is legible; less is a blurred leaf or a hand
LEGIBLE_USUAL = 0.5       # ... and so is one with this share of the keypoints that side usually has in the video's unflagged captures
PAGE_PAPER = 0.3          # share of the page region that must look like paper for a frame to show an open book
MAX_SCALE = 1.2           # a page does not grow or shrink by more than this between two captures
MAX_ROTATION = 12.0       # degrees
PAGE_INLIERS = 12         # agreeing keypoints each page needs for two captures to be the same spread
SPREAD_INLIERS = 60       # and the two pages together need this many
MIN_PAGE_KEYPOINTS = 60   # a page with fewer keypoints than this is treated as blank
HALF_PAGE_INLIERS = 40    # agreeing keypoints on one page needed to call a mid-turn frame part of a neighbour
UNSETTLED_MOTION = 0.15   # a candidate whose chosen frame moved this much never came to rest
FAR_APART = 8             # candidates this many positions apart show different spreads
BASELINE_PAIRS = 120      # far-apart pairs sampled to measure what different spreads agree on

_orb = None
_matcher = None


def _tools():
    global _orb, _matcher
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
        _matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    return _orb, _matcher


def print_region(mask: np.ndarray, median_bgr: np.ndarray) -> np.ndarray:
    """The page region with fixed structure removed.

    Stand clips, the gutter, page edges and the holding hand sit in the same
    place in every frame, so they match between any two captures and make two
    different, nearly blank spreads look alike. In the median frame print
    averages to a smooth gray while those fixed things stay sharp, so the
    sharp edges of the median frame mark what to leave out.
    """
    gray = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    fixed = (cv2.magnitude(gx, gy) > STATIC_EDGE).astype(np.uint8)
    fixed = cv2.dilate(fixed, np.ones((9, 9), np.uint8))
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
    return (inner > 0) & (fixed == 0)


def page_features(image_path: Path, region: np.ndarray):
    """Keypoints and descriptors inside ``region`` (a proxy-sized bool mask) of a capture."""
    orb, _ = _tools()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return [], None
    height = int(round(image.shape[0] * MATCH_WIDTH / image.shape[1]))
    image = cv2.resize(image, (MATCH_WIDTH, height), interpolation=cv2.INTER_AREA)
    where = cv2.resize(region.astype(np.uint8) * 255, (MATCH_WIDTH, height), interpolation=cv2.INTER_NEAREST)
    return orb.detectAndCompute(image, where)


def agreeing_points(a, b) -> int:
    """Matched keypoints of two captures that fit one rigid placement of the page.

    Each keypoint may be matched once, and the fit is a similarity (shift,
    rotation, uniform scale) close to "nothing moved". A full homography is not
    used: it can collapse a line of print onto a single point and report dozens
    of agreeing points between two different pages.
    """
    _, matcher = _tools()
    (keys_a, desc_a), (keys_b, desc_b) = a, b
    if desc_a is None or desc_b is None or len(keys_a) < 20 or len(keys_b) < 20:
        return 0
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    best: dict[int, cv2.DMatch] = {}
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            match = pair[0]
            if match.trainIdx not in best or match.distance < best[match.trainIdx].distance:
                best[match.trainIdx] = match
    good = list(best.values())
    if len(good) < 12:
        return 0
    src = np.float32([keys_a[m.queryIdx].pt for m in good])
    dst = np.float32([keys_b[m.trainIdx].pt for m in good])
    fit, keep = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
    if fit is None or keep is None:
        return 0
    scale = float(np.hypot(fit[0, 0], fit[1, 0]))
    turn = abs(float(np.degrees(np.arctan2(fit[1, 0], fit[0, 0]))))
    if not (1 / MAX_SCALE <= scale <= MAX_SCALE) or turn > MAX_ROTATION:
        return 0
    return int(keep.sum())


def fixed_agreement(pages: list) -> tuple[int, int]:
    """How many keypoints two *different* spreads of this video agree on, per page.

    Stand clips, page edges, a ruled border or a repeated running head match
    between any two spreads. Their level differs from video to video, so it is
    measured: candidates far apart in the sequence are certainly different
    spreads, and the 90th percentile of their agreement is taken as the level
    that means nothing.
    """
    count = len(pages)
    if count <= FAR_APART + 1:
        return 0, 0
    step = max(1, (count - FAR_APART) // BASELINE_PAIRS)
    left, right = [], []
    for i in range(0, count - FAR_APART, step):
        left.append(agreeing_points(pages[i][0], pages[i + FAR_APART][0]))
        right.append(agreeing_points(pages[i][1], pages[i + FAR_APART][1]))
    return int(np.percentile(left, 90)), int(np.percentile(right, 90))


def same_spread(a_pages, b_pages, *, threshold: int = PAGE_INLIERS, fixed: tuple[int, int] = (0, 0)) -> bool:
    """Do two captures show the same two pages?

    ``a_pages`` and ``b_pages`` are ``(left, right)`` features. Both pages have
    to match. Matching on the whole spread is not enough: a frame taken in the
    middle of a turn shares its left page with the spread before and its right
    page with the spread after, and would chain the two together. A page that
    is blank in either capture cannot confirm anything, so the other page then
    has to match twice as strongly; two blank pages are never called the same.

    Each page must clear a minimum and the two together a higher total. With
    only per-page minimums, two different spreads of IMG-0037 that agreed on 21
    and 12 keypoints were merged; frames of one spread agree on 76 and up.
    """
    (a_left, a_right), (b_left, b_right) = a_pages, b_pages
    left_ok = min(len(a_left[0]), len(b_left[0])) >= MIN_PAGE_KEYPOINTS
    right_ok = min(len(a_right[0]), len(b_right[0])) >= MIN_PAGE_KEYPOINTS
    need_left, need_right = max(threshold, 2 * fixed[0] + 2), max(threshold, 2 * fixed[1] + 2)
    need_both = max(SPREAD_INLIERS, 3 * (fixed[0] + fixed[1]) + 20)
    if left_ok and right_ok:
        left, right = agreeing_points(a_left, b_left), agreeing_points(a_right, b_right)
        return left >= need_left and right >= need_right and left + right >= need_both
    if left_ok:
        return agreeing_points(a_left, b_left) >= max(2 * need_left, need_both)
    if right_ok:
        return agreeing_points(a_right, b_right) >= max(2 * need_right, need_both)
    return False


def group_same_pages(pages: list, *, threshold: int = PAGE_INLIERS, fixed: tuple[int, int] = (0, 0)) -> list[list[int]]:
    """Consecutive captures that show the same spread, as lists of positions.

    ``pages`` holds ``(left, right)`` features per capture.
    """
    groups: list[list[int]] = []
    for position, current in enumerate(pages):
        if groups and any(same_spread(pages[other], current, threshold=threshold, fixed=fixed) for other in groups[-1][-LOOK_BACK:]):
            groups[-1].append(position)
            continue
        groups.append([position])
    return groups


def one_page(features, centre_x: float, side: str):
    """The keypoints and descriptors of the left or the right page only."""
    keys, desc = features
    if desc is None or len(keys) == 0:
        return [], None
    xs = np.array([k.pt[0] for k in keys])
    pick = np.where(xs < centre_x)[0] if side == "left" else np.where(xs >= centre_x)[0]
    return [keys[i] for i in pick], (desc[pick] if len(pick) else None)


def usual_keypoints(best: list[int], candidates: list[dict], pages: list) -> list[float]:
    """How many keypoints each side usually has in this video when a spread is shown properly (its unflagged captures)."""
    return [float(np.median([len(pages[position][side][0]) for position in best if not candidates[position].get("review")] or [0.0])) for side in (0, 1)]


def drop_transitional(groups: list[list[int]], candidates: list[dict], pages: list,
                      *, threshold: int = HALF_PAGE_INLIERS, fixed: tuple[int, int] = (0, 0)) -> tuple[list[list[int]], list[tuple[int, str]]]:
    """Remove groups that are a disturbed view of a neighbouring spread, not a spread of their own.

    Halfway through a turn the left page is still the previous spread's and
    the right page is already the next spread's; a hand pressing one page of a
    settled spread leaves the other page matching. A real spread shares
    neither page with its neighbours. Only groups whose best frame never came
    to rest are considered, so a spread that was properly shown is never
    dropped here. Returns the remaining groups and ``(candidate position,
    reason)`` for each dropped one.

    Where the view before is one that never came to rest either, the two are
    views of one page and the first is not the one to keep for being first: a
    spread that is never at rest is often seen first with the leaf still
    coming down or a hand across it, and then more calmly (Dale 16471, a hand
    over the right page, was kept and 16485, at rest and clear, dropped as a
    view of it). The later view takes the earlier one's place when it ranks
    higher and shows all the earlier one does: see ``fuller_view``.
    """
    best = [max(group, key=lambda position: quality(candidates[position])) for group in groups]
    keep = [True] * len(groups)
    dropped: list[tuple[int, str]] = []
    need = [max(threshold, 3 * fixed[index] + 6) for index in (0, 1)]
    usual = usual_keypoints(best, candidates, pages)
    sides = ("left", "right")
    seen: dict[tuple[int, int, int], int] = {}

    def agreeing(a: int, b: int, index: int) -> int:
        if (a, b, index) not in seen:
            seen[a, b, index] = agreeing_points(pages[a][index], pages[b][index])
        return seen[a, b, index]

    def shared(neighbour: int | None, position: int) -> int | None:
        """The first side on which the two show the same page."""
        if neighbour is None:
            return None
        return next((index for index in (0, 1) if agreeing(neighbour, position, index) >= need[index]), None)

    def unsettled(position: int) -> bool:
        return candidates[position]["best_motion"] > UNSETTLED_MOTION

    def fuller_view(position: int, earlier: int, side: int) -> bool:
        """Whether a later view that never came to rest should stand in place of an earlier one that shares its ``side`` page.

        It has to rank higher, and the earlier view must have nothing it
        lacks: the earlier view's other page is hidden (it holds under
        ``LEGIBLE_USUAL`` of the keypoints that side usually has: a leaf in
        the air, a hand), or the two agree on that page as well. Where the
        earlier view shows another page in full that this one does not, this
        one is a frame from the next turn and the earlier view is the record
        of that page.
        """
        other = 1 - side
        if quality(candidates[position]) <= quality(candidates[earlier]):
            return False
        hidden = len(pages[earlier][other][0]) < LEGIBLE_USUAL * usual[other]
        return hidden or agreeing(earlier, position, other) >= max(PAGE_INLIERS, 3 * fixed[other] + 6)

    for number, position in enumerate(best):
        if not unsettled(position):
            continue
        kept_before = [k for k in range(number - 1, -1, -1) if keep[k]]
        before = best[kept_before[0]] if kept_before else None
        # Only a neighbour that was properly shown can vouch that this frame is
        # a disturbed view of it. Judging against the next candidate whatever
        # its state dropped every view of a spread that was never at rest: each
        # one "matched the next", which was dropped in turn.
        after = next((best[k] for k in range(number + 1, min(len(groups), number + 4)) if not unsettled(best[k])), None)
        reason = None
        side = shared(before, position)
        if side is not None:
            reason = f"{sides[side]} page matches the previous spread"
            if unsettled(before) and shared(after, position) is None and fuller_view(position, before, side):
                # It must be a view in its own right once the earlier one is gone: not a disturbed view of the capture kept before that.
                further = best[kept_before[1]] if len(kept_before) > 1 else None
                if shared(further, position) is None:
                    keep[kept_before[0]] = False
                    dropped.extend((p, f"a better view of its {sides[side]} page follows") for p in groups[kept_before[0]])
                    continue
        else:
            side = shared(after, position)
            if side is not None:
                reason = f"{sides[side]} page matches the next spread"
        if reason:
            keep[number] = False
            dropped.extend((p, reason) for p in groups[number])
    return [g for g, k in zip(groups, keep) if k], dropped


def drop_shown_elsewhere(groups: list[list[int]], candidates: list[dict], pages: list,
                         *, fixed: tuple[int, int] = (0, 0)) -> tuple[list[list[int]], list[tuple[int, str]]]:
    """Remove captures that never came to rest and in which everything legible is shown properly close by.

    A reader who grabs two leaves sees a later spread for a moment, with a
    leaf in the air or a hand across it, and goes back; a leaf that springs up
    is pressed down again. Such a capture never came to rest or has a hand on
    the print, and what can be read in it is a page that an unflagged capture
    within ``ELSEWHERE_WINDOW`` places also shows. It is dropped only if that
    holds for every legible half: a half with its share of the keypoints (of
    the capture's own, or of what that side usually has in this video) that
    matches nothing nearby may be the only record of a page, and then the
    capture stays. Unflagged captures are never dropped here and flagged ones
    never vouch, so the rule cannot chain.

    Measured on the development videos: of 60 captures
    the printed numbers show to be one too many it removes 29, and none of the
    47 they show to be a spread of their own. Matching against the two
    neighbours only, at 40 keypoints, removed 5.
    """
    best = [max(group, key=lambda position: quality(candidates[position])) for group in groups]
    flagged = [bool(candidates[position].get("review")) for position in best]
    keep = [True] * len(groups)
    dropped: list[tuple[int, str]] = []
    # How many keypoints each side usually has when a spread is shown
    # properly. A page of dense print takes most of a capture's keypoints,
    # and the plain page facing it, in full view, is left with under a
    # quarter: judged by its share alone it passes for hidden, and the
    # capture for a glimpse of the dense page only. In Kloppenborg that
    # dropped the only capture of pp. 50-51 (twice, with different frame
    # choices). Kept by this, a few more second views stay (IMG-0041 and
    # IMG-0042, about ten); a missing page is the worse fault.
    usual = usual_keypoints(best, candidates, pages)
    for number, position in enumerate(best):
        # Only a capture that never came to rest. One that did, with a hand at
        # its edge, is a proper view of its spread: in Olson such a capture of
        # a blank page facing "Scripture Index" was dropped because that page
        # agreed on 30 keypoints with "Author Index" on the next spread.
        if candidates[position]["best_motion"] <= UNSETTLED_MOTION:
            continue
        keys = [len(pages[position][0][0]), len(pages[position][1][0])]
        total = max(1, sum(keys))
        found: list[str] = []
        alone = False
        for index, side in ((0, "left"), (1, "right")):
            need = max(ELSEWHERE_INLIERS, 3 * fixed[index] + 6)
            where = None
            for distance in range(1, ELSEWHERE_WINDOW + 1):
                for other in (number + distance, number - distance):
                    if 0 <= other < len(groups) and not flagged[other] and where is None:
                        if agreeing_points(pages[best[other]][index], pages[position][index]) >= need:
                            where = other - number
                if where is not None:
                    break
            if where is not None:
                found.append(f"{side} page is shown {abs(where)} capture(s) {'later' if where > 0 else 'earlier'}")
            elif keys[index] >= LEGIBLE_SHARE * total or (usual[index] > 0 and keys[index] >= LEGIBLE_USUAL * usual[index]):
                alone = True
        if found and not alone:
            keep[number] = False
            dropped.extend((p, "; ".join(found)) for p in groups[number])
    return [g for g, k in zip(groups, keep) if k], dropped


def trim_not_pages(groups: list[list[int]], candidates: list[dict]) -> tuple[list[list[int]], list[tuple[int, str]]]:
    """Remove what comes before the book is open and after it is closed.

    A recording starts and ends with something that is not a spread: the
    cover, the empty stand, the next book being put up. In those frames the
    page region shows no paper. Only an unbroken run of such captures at the
    very start or the very end is removed; in the middle of a book a spread of
    two dark plates looks the same, and is kept.
    """
    best = [max(group, key=lambda position: quality(candidates[position])) for group in groups]
    paper = [candidates[position].get("paper", 1.0) for position in best]
    first = 0
    while first < len(groups) and paper[first] < PAGE_PAPER:
        first += 1
    last = len(groups)
    while last > first and paper[last - 1] < PAGE_PAPER:
        last -= 1
    dropped = [(p, "no paper in the page region: not an open book") for number in list(range(first)) + list(range(last, len(groups))) for p in groups[number]]
    return groups[first:last], dropped


def quality(capture: dict) -> tuple[float, float, float]:
    """Higher is better: see ``spreads.capture_rank``."""
    return capture_rank(capture)


def select_captures(run_dir: Path, region: np.ndarray, *, threshold: int = PAGE_INLIERS) -> dict:
    """Choose the final captures from the candidates of one video.

    Runs of candidates that show the same pages are reduced to their best
    member. ``captures/`` is rebuilt as hard links to the cached stills, so
    this can be repeated with different settings without decoding again, and
    ``spreads.json`` gains a ``captures`` list.
    """
    summary_path = run_dir / "spreads.json"
    summary = json.loads(summary_path.read_text())
    candidates = summary["candidates"]
    source = Path(summary["stills_dir"])
    features = [page_features(source / c["still"], region) for c in candidates]
    columns = np.where(region.any(axis=0))[0]
    centre_x = (columns.min() + columns.max()) / 2 * MATCH_WIDTH / region.shape[1] if len(columns) else MATCH_WIDTH / 2
    pages = [(one_page(f, centre_x, "left"), one_page(f, centre_x, "right")) for f in features]
    fixed = fixed_agreement(pages)
    groups = group_same_pages(pages, threshold=threshold, fixed=fixed)
    groups, transitional = drop_transitional(groups, candidates, pages, fixed=fixed)
    groups, elsewhere = drop_shown_elsewhere(groups, candidates, pages, fixed=fixed)
    groups, not_pages = trim_not_pages(groups, candidates)
    target = run_dir / "captures"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    captures: list[dict] = []
    for number, group in enumerate(groups, 1):
        best = max(group, key=lambda position: quality(candidates[position]))
        chosen = dict(candidates[best])
        chosen["candidate_index"] = chosen["index"]
        chosen["repeats"] = [candidates[p]["best"] for p in group if p != best]
        chosen["index"] = number
        chosen["file"] = f"sequence_{number:04d}_t{chosen['best'] / summary['fps']:07.2f}s.jpg"
        link = target / chosen["file"]
        try:
            link.hardlink_to(source / chosen["still"])
        except OSError:
            shutil.copy2(source / chosen["still"], link)
        captures.append(chosen)
    summary["captures"] = captures
    summary.pop("captures_before_number_check", None)
    summary.pop("removed_by_number", None)
    summary["spreads"] = len(captures)
    summary["fixed_agreement"] = list(fixed)
    summary["transitional"] = [{"frame": candidates[p]["best"], "reason": reason} for p, reason in transitional]
    summary["shown_elsewhere"] = [{"frame": candidates[p]["best"], "reason": reason} for p, reason in elsewhere]
    summary["not_pages"] = [{"frame": candidates[p]["best"], "reason": reason} for p, reason in not_pages]
    summary["repeats_merged"] = len(candidates) - len(captures) - len(transitional) - len(elsewhere) - len(not_pages)
    summary["flagged"] = sum(1 for c in captures if c.get("review"))
    summary_path.write_text(json.dumps(summary, indent=1))
    return summary
