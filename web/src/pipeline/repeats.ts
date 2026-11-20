// Port of auto_camera/repeats.py and the capture list of export.py: from the
// candidates (one per spread) to the final captures, by keypoint matching of
// the full-size frames.

import {
  ELSEWHERE_INLIERS, ELSEWHERE_WINDOW, HALF_PAGE_INLIERS, LEGIBLE_SHARE, LEGIBLE_USUAL, LOOK_BACK, MATCH_WIDTH, MIN_PAGE_KEYPOINTS,
  PAGE_INLIERS, PAGE_PAPER, PROXY_HEIGHT, PROXY_WIDTH, REPEATS_FAR_APART, BASELINE_PAIRS, SPREAD_INLIERS, STATIC_EDGE,
  UNSETTLED_MOTION, EXPORT_UNSETTLED_MOTION, HAND_ON_PRINT,
} from './constants.ts';
import { bgrToGray } from './colour.ts';
import { dilate, erode, median, percentile, rectKernel, roundHalfEven, sobelMagnitude } from './imgproc.ts';
import { agreeingPoints as defaultAgree, emptyPage, type PageFeatures } from './matching.ts';
import { pyFixed, pyRound } from './pyround.ts';
import { captureRank, type Spread } from './spreads.ts';

/** print_region: the page region with fixed structure (clips, edges, the holding hand) removed. */
export function printRegion(mask: Uint8Array, medianBgr: Uint8Array): Uint8Array {
  const W = PROXY_WIDTH, H = PROXY_HEIGHT;
  const gray = bgrToGray(medianBgr);
  const mag = sobelMagnitude(gray, W, H);
  const fixed0 = new Uint8Array(W * H);
  // The gradients are whole numbers, so only a magnitude of exactly
  // STATIC_EDGE is in doubt; OpenCV's vectorised square root returns one unit
  // in the last place above it (60.000004 for 60), so it counts as fixed.
  for (let i = 0; i < fixed0.length; i++) fixed0[i] = mag[i] >= STATIC_EDGE ? 1 : 0;
  const fixed = dilate(fixed0, W, H, rectKernel(9, 9));
  const inner = erode(mask, W, H, rectKernel(9, 9));
  const out = new Uint8Array(W * H);
  for (let i = 0; i < out.length; i++) out[i] = inner[i] && !fixed[i] ? 1 : 0;
  return out;
}

/** The match image size for a frame: MATCH_WIDTH wide, height as int(round(h * MATCH_WIDTH / w)). */
export function matchSize(width: number, height: number): [number, number] {
  return [MATCH_WIDTH, roundHalfEven(height * MATCH_WIDTH / width)];
}

/** centre_x of select_captures: middle of the region's columns, in match-image pixels. */
export function regionCentreX(region: Uint8Array): number {
  let x0 = Infinity, x1 = -Infinity;
  for (let i = 0; i < region.length; i++) if (region[i]) { const x = i % PROXY_WIDTH; x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
  return x0 === Infinity ? MATCH_WIDTH / 2 : (x0 + x1) / 2 * MATCH_WIDTH / PROXY_WIDTH;
}

/** one_page for both sides: keypoints left of centre_x, and the rest. */
export function splitPages(all: PageFeatures, centreX: number): [PageFeatures, PageFeatures] {
  const pick = (left: boolean): PageFeatures => {
    const idx: number[] = [];
    for (let i = 0; i < all.xs.length; i++) if ((all.xs[i] < centreX) === left) idx.push(i);
    if (!idx.length) return emptyPage();
    const xs = new Float32Array(idx.length), ys = new Float32Array(idx.length), desc = new Uint8Array(idx.length * 32);
    idx.forEach((i, k) => { xs[k] = all.xs[i]; ys[k] = all.ys[i]; desc.set(all.desc.subarray(i * 32, i * 32 + 32), k * 32); });
    return { xs, ys, desc };
  };
  return [pick(true), pick(false)];
}

/** A candidate as spreads.json holds it (Spread.as_dict plus the review flags). */
export interface Candidate {
  index: number;
  start: number;
  end: number;
  best: number;
  best_seconds: number | null;
  best_motion: number;
  detail: number;
  hand: number;
  hand_on_print: number;
  hand_inside: number;
  print_hidden: number;
  hand_on_page: number;
  hand_free_exists: boolean;
  before_lift: boolean;
  landing_cover: number;
  paper: number;
  weak_landing: boolean;
  left_frame: number;
  right_frame: number;
  fold: number;
  review: string[];
}

/** review_flags of export.py. */
export function reviewFlags(s: Spread): string[] {
  const reasons: string[] = [];
  if (s.handOnPrint > HAND_ON_PRINT) reasons.push('hand_on_print');
  if (s.bestMotion > EXPORT_UNSETTLED_MOTION) reasons.push('never_at_rest');
  return reasons;
}

export function toCandidate(s: Spread, fps: number): Candidate {
  return {
    index: s.index, start: s.start, end: s.end, best: s.best,
    best_seconds: s.best >= 0 ? pyRound(s.best / fps, 3) : null,
    best_motion: pyRound(s.bestMotion, 4), detail: pyRound(s.detail, 2), hand: pyRound(s.hand, 4),
    hand_on_print: pyRound(s.handOnPrint, 4), hand_inside: pyRound(s.handInside, 4), print_hidden: pyRound(s.printHidden, 4),
    hand_on_page: pyRound(s.handOnPage, 4), hand_free_exists: s.handFreeExists, before_lift: s.beforeLift,
    landing_cover: pyRound(s.landingCover, 3), paper: pyRound(s.paper, 3), weak_landing: s.weakLanding,
    left_frame: s.best, right_frame: s.best, fold: pyRound(s.fold, 4),
    review: reviewFlags(s),
  };
}

/** audit.shown_reasons: the review reasons a person is told about; a hand over the print only when it lies well inside it. */
export function shownReasons(c: Pick<Candidate, 'review' | 'hand_inside' | 'hand_on_print'>): string[] {
  const reasons = [...c.review];
  const at = reasons.indexOf('hand_on_print');
  if (at >= 0 && (c.hand_inside ?? c.hand_on_print ?? 0) <= HAND_ON_PRINT) reasons.splice(at, 1);
  return reasons;
}

type Pages = [PageFeatures, PageFeatures];
type Agree = (a: PageFeatures, b: PageFeatures) => number;

/** quality(): spreads.capture_rank. */
export function quality(c: Candidate): [number, number] {
  return captureRank(c);
}

const better = (x: [number, number], y: [number, number]) => x[0] > y[0] || (x[0] === y[0] && x[1] > y[1]);

/** max(group, key=quality): the first of the best. */
export function bestOf(group: number[], candidates: Candidate[]): number {
  let best = group[0], q = quality(candidates[group[0]]);
  for (const p of group.slice(1)) {
    const qp = quality(candidates[p]);
    if (better(qp, q)) { best = p; q = qp; }
  }
  return best;
}

/** fixed_agreement: what two certainly different spreads of this video agree on, per page. */
export function fixedAgreement(pages: Pages[], agree: Agree): [number, number] {
  const count = pages.length;
  if (count <= REPEATS_FAR_APART + 1) return [0, 0];
  const step = Math.max(1, Math.floor((count - REPEATS_FAR_APART) / BASELINE_PAIRS));
  const left: number[] = [], right: number[] = [];
  for (let i = 0; i < count - REPEATS_FAR_APART; i += step) {
    left.push(agree(pages[i][0], pages[i + REPEATS_FAR_APART][0]));
    right.push(agree(pages[i][1], pages[i + REPEATS_FAR_APART][1]));
  }
  return [Math.trunc(percentile(left, 90)), Math.trunc(percentile(right, 90))];
}

/** same_spread: do two captures show the same two pages? */
export function sameSpread(a: Pages, b: Pages, agree: Agree, threshold = PAGE_INLIERS, fixed: [number, number] = [0, 0]): boolean {
  const leftOk = Math.min(a[0].xs.length, b[0].xs.length) >= MIN_PAGE_KEYPOINTS;
  const rightOk = Math.min(a[1].xs.length, b[1].xs.length) >= MIN_PAGE_KEYPOINTS;
  const needLeft = Math.max(threshold, 2 * fixed[0] + 2), needRight = Math.max(threshold, 2 * fixed[1] + 2);
  const needBoth = Math.max(SPREAD_INLIERS, 3 * (fixed[0] + fixed[1]) + 20);
  if (leftOk && rightOk) {
    const left = agree(a[0], b[0]), right = agree(a[1], b[1]);
    return left >= needLeft && right >= needRight && left + right >= needBoth;
  }
  if (leftOk) return agree(a[0], b[0]) >= Math.max(2 * needLeft, needBoth);
  if (rightOk) return agree(a[1], b[1]) >= Math.max(2 * needRight, needBoth);
  return false;
}

/** group_same_pages: consecutive captures that show the same spread, as lists of positions. */
export function groupSamePages(pages: Pages[], agree: Agree, threshold = PAGE_INLIERS, fixed: [number, number] = [0, 0]): number[][] {
  const groups: number[][] = [];
  pages.forEach((current, position) => {
    if (groups.length) {
      const last = groups[groups.length - 1];
      if (last.slice(-LOOK_BACK).some(other => sameSpread(pages[other], current, agree, threshold, fixed))) {
        last.push(position);
        return;
      }
    }
    groups.push([position]);
  });
  return groups;
}

/** drop_transitional: remove groups that are a disturbed view of a neighbouring spread. */
export function dropTransitional(groups: number[][], candidates: Candidate[], pages: Pages[], agree: Agree,
  threshold = HALF_PAGE_INLIERS, fixed: [number, number] = [0, 0]): [number[][], [number, string][]] {
  const best = groups.map(g => bestOf(g, candidates));
  const keep = groups.map(() => true);
  const dropped: [number, string][] = [];
  best.forEach((position, number) => {
    if (candidates[position].best_motion <= UNSETTLED_MOTION) return;
    let before: number | null = null;
    for (let k = number - 1; k >= 0; k--) if (keep[k]) { before = best[k]; break; }
    let after: number | null = null;
    for (let k = number + 1; k < Math.min(groups.length, number + 4); k++) {
      if (candidates[best[k]].best_motion <= UNSETTLED_MOTION) { after = best[k]; break; }
    }
    let reason: string | null = null;
    for (const [neighbour, name] of [[before, 'previous'], [after, 'next']] as [number | null, string][]) {
      if (neighbour === null || reason) continue;
      for (const [side, index] of [['left', 0], ['right', 1]] as [string, 0 | 1][]) {
        if (agree(pages[neighbour][index], pages[position][index]) >= Math.max(threshold, 3 * fixed[index] + 6)) {
          reason = `${side} page matches the ${name} spread`;
          break;
        }
      }
    }
    if (reason) {
      keep[number] = false;
      for (const p of groups[number]) dropped.push([p, reason]);
    }
  });
  return [groups.filter((_, k) => keep[k]), dropped];
}

/** drop_shown_elsewhere: remove captures that never came to rest and in which everything legible is shown properly close by. */
export function dropShownElsewhere(groups: number[][], candidates: Candidate[], pages: Pages[], agree: Agree,
  fixed: [number, number] = [0, 0]): [number[][], [number, string][]] {
  const best = groups.map(g => bestOf(g, candidates));
  const flagged = best.map(p => candidates[p].review.length > 0);
  const keep = groups.map(() => true);
  const dropped: [number, string][] = [];
  // How many keypoints each side usually has when a spread is shown properly: a plain page beside a dense one is legible though it holds few.
  const usual = ([0, 1] as const).map(side => {
    const counts = best.filter((_, k) => !flagged[k]).map(p => pages[p][side].xs.length);
    return counts.length ? median(counts) : 0;
  });
  best.forEach((position, number) => {
    // Only a capture that never came to rest; one that did, with a hand at its edge, is a proper view of its spread.
    if (candidates[position].best_motion <= UNSETTLED_MOTION) return;
    const keys = [pages[position][0].xs.length, pages[position][1].xs.length];
    const total = Math.max(1, keys[0] + keys[1]);
    const found: string[] = [];
    let alone = false;
    for (const [index, side] of [[0, 'left'], [1, 'right']] as [0 | 1, string][]) {
      const need = Math.max(ELSEWHERE_INLIERS, 3 * fixed[index] + 6);
      let where: number | null = null;
      for (let distance = 1; distance <= ELSEWHERE_WINDOW; distance++) {
        for (const other of [number + distance, number - distance]) {
          if (other >= 0 && other < groups.length && !flagged[other] && where === null) {
            if (agree(pages[best[other]][index], pages[position][index]) >= need) where = other - number;
          }
        }
        if (where !== null) break;
      }
      if (where !== null) found.push(`${side} page is shown ${Math.abs(where)} capture(s) ${where > 0 ? 'later' : 'earlier'}`);
      else if (keys[index] >= LEGIBLE_SHARE * total || (usual[index] > 0 && keys[index] >= LEGIBLE_USUAL * usual[index])) alone = true;
    }
    if (found.length && !alone) {
      keep[number] = false;
      for (const p of groups[number]) dropped.push([p, found.join('; ')]);
    }
  });
  return [groups.filter((_, k) => keep[k]), dropped];
}

/** trim_not_pages: remove what comes before the book is open and after it is closed. */
export function trimNotPages(groups: number[][], candidates: Candidate[]): [number[][], [number, string][]] {
  const best = groups.map(g => bestOf(g, candidates));
  const paper = best.map(p => candidates[p].paper ?? 1.0);
  let first = 0;
  while (first < groups.length && paper[first] < PAGE_PAPER) first++;
  let last = groups.length;
  while (last > first && paper[last - 1] < PAGE_PAPER) last--;
  const dropped: [number, string][] = [];
  for (const number of [...Array(first).keys(), ...Array.from({ length: groups.length - last }, (_, k) => last + k)]) {
    for (const p of groups[number]) dropped.push([p, 'no paper in the page region: not an open book']);
  }
  return [groups.slice(first, last), dropped];
}

export interface Capture extends Candidate {
  candidate_index: number;
  repeats: number[];
  file: string;
}

export interface Selection {
  captures: Capture[];
  fixed: [number, number];
  transitional: { frame: number; reason: string }[];
  shownElsewhere: { frame: number; reason: string }[];
  notPages: { frame: number; reason: string }[];
  repeatsMerged: number;
  flagged: number;
}

/** The file name select_captures gives a capture. */
export const captureName = (number: number, frame: number, fps: number) =>
  `sequence_${String(number).padStart(4, '0')}_t${pyFixed(frame / fps, 7, 2)}s.jpg`;

/** select_captures, given each candidate's (left, right) page features. */
export function selectCaptures(candidates: Candidate[], pages: Pages[], fps: number, agree: Agree = defaultAgree,
  threshold = PAGE_INLIERS): Selection {
  const fixed = fixedAgreement(pages, agree);
  let groups = groupSamePages(pages, agree, threshold, fixed);
  let transitional: [number, string][], elsewhere: [number, string][], notPages: [number, string][];
  [groups, transitional] = dropTransitional(groups, candidates, pages, agree, HALF_PAGE_INLIERS, fixed);
  [groups, elsewhere] = dropShownElsewhere(groups, candidates, pages, agree, fixed);
  [groups, notPages] = trimNotPages(groups, candidates);
  const captures: Capture[] = groups.map((group, k) => {
    const number = k + 1;
    const best = bestOf(group, candidates);
    const chosen = candidates[best];
    return {
      ...chosen, candidate_index: chosen.index, repeats: group.filter(p => p !== best).map(p => candidates[p].best),
      index: number, file: captureName(number, chosen.best, fps),
    };
  });
  const asList = (xs: [number, string][]) => xs.map(([p, reason]) => ({ frame: candidates[p].best, reason }));
  return {
    captures, fixed, transitional: asList(transitional), shownElsewhere: asList(elsewhere), notPages: asList(notPages),
    repeatsMerged: candidates.length - captures.length - transitional.length - elsewhere.length - notPages.length,
    flagged: captures.filter(c => c.review.length > 0).length,
  };
}

/** Memoise agreeing_points by page identity; the same pairs are asked for by several steps. */
export function memoAgree(agree: Agree = defaultAgree): Agree {
  const ids = new WeakMap<PageFeatures, number>();
  let next = 0;
  const id = (p: PageFeatures) => { let v = ids.get(p); if (v === undefined) { v = next++; ids.set(p, v); } return v; };
  const cache = new Map<string, number>();
  return (a, b) => {
    const key = `${id(a)}:${id(b)}`;
    let v = cache.get(key);
    if (v === undefined) { v = agree(a, b); cache.set(key, v); }
    return v;
  };
}
