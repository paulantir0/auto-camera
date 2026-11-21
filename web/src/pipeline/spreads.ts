// Port of auto_camera/spreads.py: page turns from full sweeps of the left
// page, one candidate frame per interval, and the early same-spread merge.

import {
  CALM_MOTION, CALM_SLACK, CLEAR_MARGIN, FAR_APART, GRID_H, GRID_W, HAND_NEAR, HAND_STEP, HIDDEN_DROP, HIDDEN_GROUP, HIDDEN_LEVEL,
  HIDDEN_PAPER, LAND_COVERAGE, LIFT_COVERAGE, LONG_INTERVAL, MIN_PRINT_CELLS, NEAR_MOTION, PAGE_HAND_STEP, SAME_FALL, SAME_SPREAD_CHANGED,
  SAME_SPREAD_SHARE, SHARP_ENOUGH, UNSETTLED, UNSETTLED_STEPS, WEAK_LAND_COVERAGE,
} from './constants.ts';
import { CELLS, cellMedian, findValleys, pageChanged, printCells, type Analysis } from './analysis.ts';
import { boxBlur3, connectedComponents, dilate, percentile, rectKernel, roundHalfEven } from './imgproc.ts';
import { PROXY_HEIGHT, PROXY_WIDTH } from './constants.ts';
import { foldColumn } from './printmap.ts';
import { pyRound } from './pyround.ts';

const f32 = Math.fround;

export interface Spread {
  index: number;
  start: number;
  end: number;
  best: number;
  bestMotion: number;
  detail: number;
  hand: number;
  handOnPrint: number;   // share of the print covered in the chosen frame, by the largest of the measures in groupSpreads
  handInside: number;    // the same with the trained models' hand counted only well inside the print: what the review list uses
  printHidden: number;   // share of the spread's printed cells the chosen frame does not show though another of its calm frames does
  handOnPage: number;    // share of the page region the trained models' hands cover in the chosen frame
  handFreeExists: boolean;
  beforeLift: boolean;
  landingCover: number;
  paper: number;
  weakLanding: boolean;
  fold: number;          // where the two halves meet, as a share of the frame's width
}

export interface Turns {
  landings: number[];
  confirmed: number[];
  landingCover: number[];
  weak: boolean[];
  coverLeft: Float32Array;
  coverRight: Float32Array;
  /** per landing, the last full sweep of the same fall (one_landing_per_fall); absent: the landings themselves */
  settled?: number[];
}

export interface Hands {
  usable: boolean;
  samples: number;
  share: Float32Array;
  onPrint: Float32Array;
  paper: Float32Array;
  onMap?: Float32Array | null;     // the stricter colour model's cover of the print map
  net?: Float32Array | null;       // the trained models' hands over the print; null when they did not run
  netPage?: Float32Array | null;   // ... over the page region, print or margin
  netNear?: Float32Array | null;   // ... over the print or within PRINT_NEAR of it
  netInside?: Float32Array | null; // ... PRINT_INSIDE in from the edge of the print
}

/** cell_light: mean brightness per motion cell, from the 2x2 sums of 5x5 thumbnail pixels kept per frame. */
export function cellLight(lightSums: Uint16Array, frame: number): Float64Array {
  const out = new Float64Array(CELLS);
  const o = frame * CELLS;
  for (let c = 0; c < CELLS; c++) out[c] = lightSums[o + c] / 4;
  return out;
}

/** strong_print: fine-detail energy of this video's densest print cells. */
export function strongPrint(a: Analysis): number {
  const page: number[] = [];
  let any = false;
  for (let c = 0; c < CELLS; c++) if (a.gridPage[c]) any = true;
  if (!any) return 0;
  const usual = cellMedian(a.detail, a.frames, Math.max(1, Math.floor(a.frames / 600)));
  for (let c = 0; c < CELLS; c++) if (a.gridPage[c]) page.push(usual[c]);
  return percentile(page, 90);
}

/**
 * hidden_print: per frame of one spread, the share of the spread's printed
 * cells whose print the frame does not show though the spread's frames at
 * rest (`reference`) do. Frames start..end of the analysis.
 */
export function hiddenPrint(a: Analysis, start: number, end: number, reference: Uint8Array, cells: Uint8Array, level: number,
  lightSums: Uint16Array | null): Float32Array {
  const frames = end - start + 1;
  const hidden = new Float32Array(frames);
  const refs: number[] = [];
  for (let k = 0; k < frames; k++) if (reference[k]) refs.push(start + k);
  if (!refs.length) return hidden;
  // per cell, the median over the reference frames (uint8 -> float64 -> float32)
  const shown = new Float32Array(CELLS);
  const counts = new Uint32Array(256);
  const n = refs.length, k1 = Math.floor((n - 1) / 2), k2 = Math.floor(n / 2);
  for (let c = 0; c < CELLS; c++) {
    counts.fill(0);
    for (const f of refs) counts[a.detail[f * CELLS + c]]++;
    let seen = 0, lo = -1, hi = -1;
    for (let v = 0; v < 256; v++) {
      seen += counts[v];
      if (lo < 0 && seen > k1) lo = v;
      if (seen > k2) { hi = v; break; }
    }
    shown[c] = (lo + hi) / 2;
  }
  // cv2.erode with a 3x3 square on float32: the least of each cell's neighbours (the border never erodes)
  const solid = new Float32Array(CELLS);
  for (let y = 0; y < GRID_H; y++) {
    for (let x = 0; x < GRID_W; x++) {
      let m = Infinity;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < GRID_H && xx >= 0 && xx < GRID_W) m = Math.min(m, shown[yy * GRID_W + xx]);
      }
      solid[y * GRID_W + x] = m;
    }
  }
  const floor = f32(Math.max(level, 1.0));
  const printed = new Uint8Array(CELLS);
  const at: number[] = [];
  for (let c = 0; c < CELLS; c++) if (cells[c] && solid[c] >= floor) { printed[c] = 1; at.push(c); }
  const count = at.length;
  if (count < MIN_PRINT_CELLS) return hidden;
  const low = new Uint8Array(CELLS);
  const ratios = new Float32Array(count);
  for (let k = 0; k < frames; k++) {
    const o = (start + k) * CELLS;
    // overall: the median of detail / shown over the printed cells, in float32 (a frame a little softer all over is not penalised)
    for (let j = 0; j < count; j++) ratios[j] = f32(a.detail[o + at[j]] / shown[at[j]]);
    const sorted = Float32Array.from(ratios).sort();
    const overall = count % 2 ? sorted[(count - 1) / 2] : f32(f32(sorted[count / 2 - 1] + sorted[count / 2]) / 2);
    const scale = f32(HIDDEN_DROP * overall);
    low.fill(0);
    let lows = 0;
    for (const c of at) if (a.detail[o + c] < f32(scale * shown[c])) { low[c] = 1; lows++; }
    if (lightSums && lows) {
      const light = cellLight(lightSums, start + k);
      const onCells: number[] = [];
      for (let c = 0; c < CELLS; c++) if (cells[c]) onCells.push(light[c]);
      const paper = HIDDEN_PAPER * percentile(onCells, 95);
      lows = 0;
      for (const c of at) { if (low[c] && !(light[c] < paper)) low[c] = 0; lows += low[c]; }
    }
    if (lows < HIDDEN_GROUP) continue;
    const cc = connectedComponents(low, GRID_W, GRID_H, 8);
    let patches = 0;
    for (let l = 1; l < cc.count; l++) if (cc.areas[l] >= HIDDEN_GROUP) patches += cc.areas[l];
    hidden[k] = patches / count;
  }
  return hidden;
}

/** detect_turns: page turns as full sweeps of the left page. */
export function detectTurns(a: Analysis, landCoverage = LAND_COVERAGE): Turns {
  const count = a.frames;
  const coverLeft = new Float32Array(count), coverRight = new Float32Array(count);
  const rest = new Set(findValleys(a.leftMotion));
  let left = a.gridLeft, right = a.gridRight;
  const printed = printCells(a);
  const and = (x: Uint8Array, y: Uint8Array) => x.map((v, i) => v & y[i]);
  const sum = (x: Uint8Array) => x.reduce((s, v) => s + v, 0);
  if (sum(and(left, printed)) >= MIN_PRINT_CELLS) left = and(left, printed);
  if (sum(and(right, printed)) >= MIN_PRINT_CELLS) right = and(right, printed);
  const leftCells: number[] = [], rightCells: number[] = [];
  for (let c = 0; c < CELLS; c++) { if (left[c]) leftCells.push(c); if (right[c]) rightCells.push(c); }
  const leftArea = Math.max(1, leftCells.length), rightArea = Math.max(1, rightCells.length);
  const sweptLeft = new Uint8Array(CELLS), sweptRight = new Uint8Array(CELLS);
  const landings: number[] = [], confirmed: number[] = [], landingCover: number[] = [];
  let crossed = -1;
  for (let index = 1; index < count; index++) {
    const o = index * CELLS;
    for (let c = 0; c < CELLS; c++) if (a.moved[o + c]) { sweptLeft[c] = 1; sweptRight[c] = 1; }
    let s = 0;
    for (const c of leftCells) s += sweptLeft[c];
    coverLeft[index] = s / leftArea;
    if (crossed < 0 && coverLeft[index] >= landCoverage) {
      crossed = index;
      sweptRight.fill(0);
    }
    s = 0;
    for (const c of rightCells) s += sweptRight[c];
    coverRight[index] = s / rightArea;
    if (crossed >= 0 && rest.has(index)) {
      landings.push(crossed);
      confirmed.push(index);
      landingCover.push(coverLeft[index]);
      sweptLeft.fill(0);
      crossed = -1;
    }
  }
  const turns = rescueWeakLandings({ landings, confirmed, landingCover, weak: landings.map(() => false), coverLeft, coverRight }, rest);
  return oneLandingPerFall(turns, a.leftMotion);
}

/**
 * one_landing_per_fall: a leaf that slows down in the air is counted once. A full sweep that follows within
 * SAME_FALL frames on a "rest" that was only a dip (the left page still moving) belongs to the same fall: the
 * spread starts at the first sweep, `settled` records the last, and the frames between stay with that spread.
 */
export function oneLandingPerFall(t: Turns, leftMotion: ArrayLike<number>): Turns {
  const count = t.landings.length;
  const sameFall = t.landings.map((_, k) => k + 1 < count && leftMotion[t.confirmed[k]] >= CALM_MOTION && t.landings[k + 1] - t.confirmed[k] <= SAME_FALL);
  if (!sameFall.some(Boolean)) return t;
  const landings: number[] = [], confirmed: number[] = [], landingCover: number[] = [], weak: boolean[] = [], settled: number[] = [];
  for (let first = 0; first < count;) {
    let last = first;
    while (sameFall[last]) last++;
    landings.push(t.landings[first]);
    settled.push(t.landings[last]);
    confirmed.push(t.confirmed[last]);
    landingCover.push(t.landingCover[last]);
    weak.push(t.weak.slice(first, last + 1).some(Boolean));
    first = last + 1;
  }
  return { landings, confirmed, landingCover, weak, coverLeft: t.coverLeft, coverRight: t.coverRight, settled };
}

/** rescue_weak_landings: a second look inside intervals far too long for one spread. */
export function rescueWeakLandings(t: Turns, rest: Set<number>): Turns {
  if (t.landings.length < 6) return t;
  const lengths = t.landings.slice(1).map((v, i) => v - t.landings[i]);
  const sorted = [...lengths].sort((x, y) => x - y);
  const n = sorted.length;
  const usual = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const added: [number, number, number][] = [];
  for (let k = 0; k < t.landings.length - 1; k++) {
    const start = t.confirmed[k], stop = t.landings[k + 1];
    if (stop - t.landings[k] < LONG_INTERVAL * usual) continue;
    let crossing: number | null = null;
    for (let i = start + 1; i < stop; i++) if (t.coverLeft[i] >= WEAK_LAND_COVERAGE) { crossing = i; break; }
    if (crossing === null) continue;
    let restAt: number | null = null;
    for (let i = crossing; i < stop; i++) if (rest.has(i)) { restAt = i; break; }
    if (restAt === null || crossing - t.landings[k] < 0.5 * usual || stop - restAt < 0.5 * usual) continue;
    added.push([crossing, restAt, t.coverLeft[restAt]]);
  }
  if (!added.length) return t;
  const merged: [number, number, number, boolean][] = t.landings.map((l, i) => [l, t.confirmed[i], t.landingCover[i], false] as [number, number, number, boolean])
    .concat(added.map(([l, c, v]) => [l, c, v, true] as [number, number, number, boolean]));
  // Python sorts the tuples: by landing, then confirmed, then cover, then the flag
  merged.sort((x, y) => x[0] - y[0] || x[1] - y[1] || x[2] - y[2] || Number(x[3]) - Number(y[3]));
  return {
    landings: merged.map(m => m[0]), confirmed: merged.map(m => m[1]), landingCover: merged.map(m => m[2]),
    weak: merged.map(m => m[3]), coverLeft: t.coverLeft, coverRight: t.coverRight,
  };
}

/** frame_quality: sharp, uncovered print on a page that is still. float32 like NumPy. */
export function frameQuality(a: Analysis): Float32Array {
  const m = a.pageMotion, n = m.length;
  const q = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const after = i + 1 < n ? m[i + 1] : m[n - 1];
    q[i] = f32(a.pageDetail[i] * f32(1.0 - f32(0.5 * f32(m[i] + after))));
  }
  return q;
}

/**
 * group_spreads: one spread per interval between landings, with its best frame chosen.
 *
 * Among the calm frames at least SHARP_ENOUGH as good as the best, the one
 * with least over the print is taken, then least hand on the page, then the
 * best. What is over the print is the largest of: the colour model's cover
 * of the print cells, and, beyond what the spread's cleanest calm frame has,
 * the stricter colour model's cover of the print map, the print other calm
 * frames show and this one does not, and the trained models' hands on or
 * beside the print. `lightSums` are the per-cell brightness sums kept by the
 * first pass (cellLight).
 */
export function groupSpreads(t: Turns, a: Analysis, hands: Hands | null, lightSums: Uint16Array | null = null, liftCoverage = LIFT_COVERAGE): Spread[] {
  const motion = a.pageMotion;
  const count = motion.length;
  const quality = frameQuality(a);
  const useHands = hands !== null && hands.usable;
  const level = HIDDEN_LEVEL * strongPrint(a);
  const cells = dilate(a.gridPage, GRID_W, GRID_H, rectKernel(3, 3));
  const foldShare = foldColumn(a, GRID_W * 100) / (GRID_W * 100);
  const net = hands?.net ?? null, netPage = hands?.netPage ?? null, netInside = hands?.netInside ?? null;
  const ranked = net ? (hands?.netNear ?? net) : null;
  const onMap = useHands ? hands!.onMap ?? null : null;
  const starts = [0, ...t.landings];
  const settles = [0, ...(t.settled ?? t.landings)];
  const covers = [0.0, ...t.landingCover];
  const weak = [false, ...t.weak];
  const spreads: Spread[] = [];
  const step32 = f32(HAND_STEP), margin32 = f32(CLEAR_MARGIN), pageStep32 = f32(PAGE_HAND_STEP), near32 = f32(NEAR_MOTION);
  starts.forEach((start, position) => {
    const end = position + 1 < starts.length ? starts[position + 1] - 1 : count - 1;
    if (end < start) return;
    const len = end - start + 1;
    const moving = new Float32Array(len);
    let lowest = Infinity;
    for (let k = 0; k < len; k++) {
      const i = start + k;
      moving[k] = Math.max(motion[i], motion[i < end ? i + 1 : end]);
      lowest = Math.min(lowest, moving[k]);
    }
    const calmLimit = f32(Math.max(CALM_MOTION, lowest + CALM_SLACK));
    const calm = new Uint8Array(len);
    const window = new Float32Array(len);
    let windowMax = -Infinity;
    for (let k = 0; k < len; k++) {
      calm[k] = moving[k] <= calmLimit ? 1 : 0;
      window[k] = calm[k] ? quality[start + k] : -Infinity;
      windowMax = Math.max(windowMax, window[k]);
    }
    // the right page is watched from where this spread's own leaf came down (the last sweep of its fall)
    let lift = settles[position];
    for (let frame = lift + 1; frame <= end; frame++) {
      if (t.coverRight[frame] >= liftCoverage) break;
      lift = frame;
    }
    // what the spread looks like is taken from its calm frames before the next leaf is up, when there are any
    const reference = calm.slice();
    let before = false;
    for (let k = 0; k <= lift - start; k++) if (calm[k]) { before = true; break; }
    if (before) for (let k = lift - start + 1; k < len; k++) reference[k] = 0;
    const hidden = hiddenPrint(a, start, end, reference, cells, level, lightSums);
    const calmMin = (x: ArrayLike<number>, offset = 0) => {
      let m = Infinity;
      for (let k = 0; k < len; k++) if (calm[k]) m = Math.min(m, x[offset + k]);
      return m;
    };
    const hiddenMin = calmMin(hidden);
    const beyond = new Float32Array(len);
    for (let k = 0; k < len; k++) beyond[k] = f32(hidden[k] - hiddenMin);
    if (onMap) {
      const m = calmMin(onMap, start);
      for (let k = 0; k < len; k++) beyond[k] = Math.max(beyond[k], f32(onMap[start + k] - m));
    }
    const primary = new Float32Array(len);
    for (let k = 0; k < len; k++) primary[k] = Math.floor(f32(Math.max(beyond[k], 0) / margin32));
    if (useHands) for (let k = 0; k < len; k++) primary[k] = Math.max(primary[k], roundHalfEven(f32(hands!.onPrint[start + k] / step32)));
    if (ranked) {
      // in a frame that moves, a hand found on the print within HAND_NEAR frames before it and after it counts as there
      const ownMin = calmMin(ranked, start);
      for (let k = 0; k < len; k++) {
        const i = start + k;
        let seen = ranked[i];
        if (moving[k] > near32) {
          let b = 0, af = 0;
          if (i > 0) { b = -Infinity; for (let j = Math.max(0, i - HAND_NEAR); j < i; j++) b = Math.max(b, ranked[j]); }
          if (i + 1 < ranked.length) { af = -Infinity; for (let j = i + 1; j < Math.min(ranked.length, i + HAND_NEAR + 1); j++) af = Math.max(af, ranked[j]); }
          seen = Math.max(seen, Math.min(b, af));
        }
        primary[k] = Math.max(primary[k], roundHalfEven(f32(Math.max(f32(seen - ownMin), 0) / step32)));
      }
    }
    // next, a hand anywhere on the page: the hand that turns the leaves rests on its corner before it lifts it
    const secondary = new Float32Array(len);
    if (netPage) {
      const m = calmMin(netPage, start);
      for (let k = 0; k < len; k++) secondary[k] = roundHalfEven(f32(Math.max(f32(netPage[start + k] - m), 0) / pageStep32));
    }
    const sharp = f32(SHARP_ENOUGH * windowMax);
    const allowed = new Uint8Array(len);
    let least = Infinity;
    const rank = new Float64Array(len);
    for (let k = 0; k < len; k++) {
      allowed[k] = calm[k] && window[k] >= sharp ? 1 : 0;
      rank[k] = allowed[k] ? primary[k] * 1000 + secondary[k] : Infinity;
      least = Math.min(least, rank[k]);
    }
    let bestK = 0, bestV = -Infinity;
    for (let k = 0; k < len; k++) {
      const v = rank[k] === least ? window[k] : -Infinity;
      if (v > bestV) { bestV = v; bestK = k; }
    }
    const best = start + bestK;
    // recorded, and flagged on: what the chosen frame has over the print
    const covered = new Float32Array(len), inside = new Float32Array(len);
    for (let k = 0; k < len; k++) {
      const steps = f32(Math.floor(f32(Math.max(f32(hidden[k] - hiddenMin), 0) / margin32)) * margin32);
      covered[k] = steps; inside[k] = steps;
      if (useHands) { covered[k] = Math.max(covered[k], hands!.onPrint[start + k]); inside[k] = Math.max(inside[k], hands!.onPrint[start + k]); }
      if (net) covered[k] = Math.max(covered[k], net[start + k]);
      if (netInside) inside[k] = Math.max(inside[k], netInside[start + k]);
      else if (net) inside[k] = Math.max(inside[k], net[start + k]);
    }
    let freeLeast = Infinity;
    for (let k = 0; k < len; k++) if (allowed[k]) freeLeast = Math.min(freeLeast, roundHalfEven(f32(covered[k] / step32)));
    const after = best + 1 < count ? motion[best + 1] : motion[best];
    spreads.push({
      index: spreads.length + 1, start, end, best,
      bestMotion: Math.max(motion[best], after),
      detail: a.pageDetail[best],
      hand: useHands ? hands!.share[best] : 0.0,
      handOnPrint: covered[bestK],
      handInside: inside[bestK],
      printHidden: Math.max(0, f32(hidden[bestK] - hiddenMin)),
      handOnPage: netPage ? netPage[best] : 0.0,
      handFreeExists: freeLeast === 0,
      beforeLift: best <= lift,
      landingCover: covers[position],
      paper: hands !== null ? hands.paper[best] : 1.0,
      weakLanding: weak[position],
      fold: foldShare,
    });
  });
  return spreads;
}

/**
 * The frames whose hands groupSpreads looks at: the calm frames of every
 * interval, and the HAND_NEAR frames either side of a calm frame that moves
 * (its hand is judged by its neighbours'). The trained models need to run on
 * these only; on any other frame their result is never read. Python runs
 * them on every frame, with the same result for these.
 */
export function framesForHands(t: Turns, a: Analysis): Uint8Array {
  const motion = a.pageMotion, count = motion.length;
  const needed = new Uint8Array(count);
  const starts = [0, ...t.landings];
  starts.forEach((start, position) => {
    const end = position + 1 < starts.length ? starts[position + 1] - 1 : count - 1;
    if (end < start) return;
    const len = end - start + 1;
    const moving = new Float32Array(len);
    let lowest = Infinity;
    for (let k = 0; k < len; k++) {
      const i = start + k;
      moving[k] = Math.max(motion[i], motion[i < end ? i + 1 : end]);
      lowest = Math.min(lowest, moving[k]);
    }
    const calmLimit = f32(Math.max(CALM_MOTION, lowest + CALM_SLACK)), near32 = f32(NEAR_MOTION);
    for (let k = 0; k < len; k++) {
      if (moving[k] > calmLimit) continue;
      const i = start + k;
      needed[i] = 1;
      if (moving[k] > near32) for (let j = Math.max(0, i - HAND_NEAR); j <= Math.min(count - 1, i + HAND_NEAR); j++) needed[j] = 1;
    }
  });
  return needed;
}

/**
 * capture_rank: higher is better, between captures of one spread (rounded as
 * spreads.json holds them): what lies over the print in HAND_STEPs, with
 * UNSETTLED_STEPS more for one that never came to rest; then sharp print on
 * a page at rest.
 */
export function captureRank(c: { hand_on_print: number; best_motion: number; detail: number }): [number, number] {
  let steps = roundHalfEven(c.hand_on_print / HAND_STEP);
  if (c.best_motion > UNSETTLED) steps += UNSETTLED_STEPS;
  return [-steps, c.detail * (1.0 - Math.min(1.0, c.best_motion))];
}

/** A spread's measures as Spread.as_dict rounds them, for captureRank. */
export const rounded = (s: Spread) => ({ hand_on_print: pyRound(s.handOnPrint, 4), best_motion: pyRound(s.bestMotion, 4), detail: pyRound(s.detail, 2) });

/**
 * merge_repeats: fold a spread into its predecessor when both show the same
 * pages. `gray(best)` gives the proxy-sized gray frame chosen for a spread.
 */
export function mergeRepeats(spreads: Spread[], a: Analysis, gray: (frame: number) => Uint8Array,
  sameChanged = SAME_SPREAD_CHANGED): { merged: Spread[]; dropped: [number, number, number][] } {
  if (spreads.length < 2) return { merged: spreads, dropped: [] };
  const inner = a.inner;
  const blurred = new Map<number, Uint8Array>();
  for (const s of spreads) if (!blurred.has(s.best)) blurred.set(s.best, boxBlur3(gray(s.best), PROXY_WIDTH, PROXY_HEIGHT));
  const apart: number[] = [];
  for (let i = 0, j = FAR_APART; j < spreads.length; i += 2, j += 2) {
    apart.push(pageChanged(blurred.get(spreads[i].best)!, blurred.get(spreads[j].best)!, inner));
  }
  if (apart.length >= 5) sameChanged = Math.min(sameChanged, SAME_SPREAD_SHARE * percentile(apart, 10));
  const merged: Spread[] = [{ ...spreads[0] }];
  const dropped: [number, number, number][] = [];
  for (const spread of spreads.slice(1)) {
    const keep = merged[merged.length - 1];
    const changed = pageChanged(blurred.get(keep.best)!, blurred.get(spread.best)!, inner);
    if (changed < sameChanged) {
      dropped.push([keep.index, spread.index, changed]);
      keep.end = spread.end;
      const [ks, kq] = captureRank(rounded(keep)), [ss, sq] = captureRank(rounded(spread));
      if (ss > ks || (ss === ks && sq > kq)) {
        keep.best = spread.best; keep.bestMotion = spread.bestMotion; keep.detail = spread.detail; keep.beforeLift = spread.beforeLift;
        keep.hand = spread.hand; keep.handOnPrint = spread.handOnPrint; keep.printHidden = spread.printHidden; keep.paper = spread.paper;
        keep.handOnPage = spread.handOnPage;
        // Python's merge_repeats leaves hand_inside (and left_frame, right_frame) at the kept spread's own frame, so its
        // review flag and its picture are those of the frame that lost; here they follow the frame that won.
        keep.handInside = spread.handInside;
      }
      keep.handFreeExists = keep.handFreeExists || spread.handFreeExists;
    } else merged.push({ ...spread });
  }
  merged.forEach((s, i) => { s.index = i + 1; });
  return { merged, dropped };
}

/** The frames whose proxy gray merge_repeats will look at. */
export function framesForMerge(spreads: Spread[]): number[] {
  return [...new Set(spreads.map(s => s.best))].sort((x, y) => x - y);
}
