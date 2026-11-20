// Port of auto_camera/analysis.py: page region, motion and print sharpness
// from the per-frame cell measurements. Grid arrays are GRID_H x GRID_W
// (27 x 48) row-major; pixel arrays are PROXY_HEIGHT x PROXY_WIDTH.

import {
  ACTIVITY_SHARE, MIN_TEXT_FILL, CELL, DIP_LEVEL, DIP_SHARE, GRID_H, GRID_W, MARGIN_X, MARGIN_Y, MIN_DIP_RATE, PAPER_CHROMA_TOLERANCE,
  PAPER_VALUE_SHARE, PRINT_LEVEL, PROXY_HEIGHT, PROXY_WIDTH, ROW_SLACK, STRONG_THRESHOLD, TEXT_LEVEL,
  VALLEY_CEILING, VALLEY_DIP_CEILING, VALLEY_MIN_PROMINENCE, VALLEY_REACH, VALLEY_WINDOW,
} from './constants.ts';
import { bgrImageToLab, hsvValue } from './colour.ts';
import {
  close, connectedComponents, convexHull, dilate, ellipseKernel, erode, fillConvexPoly, median, open, percentile,
  rectKernel, resizeNearest,
} from './imgproc.ts';

export const CELLS = GRID_H * GRID_W;
export const PIXELS = PROXY_WIDTH * PROXY_HEIGHT;
const f32 = Math.fround;

/** Per-cell median over every step-th frame of a (frames x CELLS) uint8 array. */
export function cellMedian(detail: Uint8Array, frames: number, step: number): Float64Array {
  const usual = new Float64Array(CELLS);
  const picked: number[] = [];
  for (let f = 0; f < frames; f += step) picked.push(f);
  const counts = new Uint32Array(256);
  const n = picked.length;
  const k1 = Math.floor((n - 1) / 2), k2 = Math.floor(n / 2);
  for (let c = 0; c < CELLS; c++) {
    counts.fill(0);
    for (const f of picked) counts[detail[f * CELLS + c]]++;
    let seen = 0, a = -1, b = -1;
    for (let v = 0; v < 256; v++) {
      seen += counts[v];
      if (a < 0 && seen > k1) a = v;
      if (seen > k2) { b = v; break; }
    }
    usual[c] = (a + b) / 2;
  }
  return usual;
}

/** text_cells: (27, 48) cells that hold print on a page that gets turned. */
export function textCells(detail: Uint8Array, frames: number): Uint8Array {
  const cells = new Uint8Array(CELLS);
  if (frames === 0) return cells;
  const usual = cellMedian(detail, frames, Math.max(1, Math.floor(frames / 600)));
  const usual32 = Float32Array.from(usual);
  const dips = new Float64Array(CELLS);
  for (let c = 0; c < CELLS; c++) {
    const limit = f32(DIP_LEVEL * usual32[c]);
    let count = 0;
    for (let f = 0; f < frames; f++) if (detail[f * CELLS + c] < limit) count++;
    dips[c] = count / frames;
  }
  const level = 0.6 * TEXT_LEVEL * percentile(usual32, 97);
  const strong = new Uint8Array(CELLS);
  let anyStrong = false;
  const level32 = f32(level);
  for (let c = 0; c < CELLS; c++) if (usual32[c] >= level32) { strong[c] = 1; anyStrong = true; }
  if (!anyStrong) return strong;
  const strongDips: number[] = [];
  for (let c = 0; c < CELLS; c++) if (strong[c]) strongDips.push(dips[c]);
  const dipLevel = Math.max(MIN_DIP_RATE, DIP_SHARE * percentile(strongDips, 90));
  const turned = new Uint8Array(CELLS);
  const both: number[] = [];
  for (let c = 0; c < CELLS; c++) {
    if (dips[c] >= dipLevel) turned[c] = 1;
    if (strong[c] && turned[c]) both.push(usual32[c]);
  }
  if (both.length === 0) return cells; // strong & turned is empty
  const printLevel = f32(TEXT_LEVEL * percentile(both, 90));
  for (let c = 0; c < CELLS; c++) cells[c] = usual32[c] >= printLevel && turned[c] ? 1 : 0;
  // A block of print is at least three cells each way; a strip thinner than that is a page edge or a wire.
  const square = rectKernel(3, 3);
  const solid = open(cells, GRID_W, GRID_H, square);
  const grown = dilate(solid, GRID_W, GRID_H, square);
  for (let c = 0; c < CELLS; c++) cells[c] = cells[c] & grown[c];
  const cc = connectedComponents(cells, GRID_W, GRID_H, 4);
  if (cc.count > 2) {
    let largest = 0;
    for (let l = 1; l < cc.count; l++) largest = Math.max(largest, cc.areas[l]);
    for (let c = 0; c < CELLS; c++) {
      const l = cc.labels[c];
      cells[c] = l > 0 && cc.areas[l] >= 0.15 * largest ? 1 : 0;
    }
  }
  return cells;
}

/** text_fill: how much of their own outline the print cells fill. */
export function textFill(text: Uint8Array): number {
  const xs: number[] = [], ys: number[] = [];
  for (let c = 0; c < CELLS; c++) if (text[c]) { xs.push(c % GRID_W); ys.push(Math.floor(c / GRID_W)); }
  if (xs.length < 3) return 0;
  const hull = new Uint8Array(CELLS);
  fillConvexPoly(hull, GRID_W, GRID_H, convexHull(xs, ys));
  let area = 0;
  for (const v of hull) area += v;
  return xs.length / Math.max(1, area);
}

/** cv2.threshold(..., THRESH_OTSU) on an 8-bit image: the level that best splits its histogram. */
export function otsuThreshold(values: Uint8Array): number {
  const h = new Float64Array(256);
  for (const v of values) h[v]++;
  const scale = 1 / values.length;
  let mu = 0;
  for (let i = 0; i < 256; i++) mu += i * h[i];
  mu *= scale;
  let mu1 = 0, q1 = 0, maxSigma = 0, maxVal = 0;
  const eps = 1.1920928955078125e-7;
  for (let i = 0; i < 256; i++) {
    const p = h[i] * scale;
    mu1 *= q1;
    q1 += p;
    const q2 = 1 - q1;
    if (Math.min(q1, q2) < eps || Math.max(q1, q2) > 1 - eps) continue;
    mu1 = (mu1 + i * p) / q1;
    const mu2 = (mu - q1 * mu1) / q2;
    const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > maxSigma) { maxSigma = sigma; maxVal = i; }
  }
  return maxVal;
}

/** page_mask_from_brightness: the fallback page region for a recording with hardly any print. */
export function pageMaskFromBrightness(medianBgr: Uint8Array, activity: Float64Array): Uint8Array {
  const W = PROXY_WIDTH, H = PROXY_HEIGHT;
  const value = hsvValue(medianBgr);
  const threshold = otsuThreshold(value);
  const bright = new Uint8Array(PIXELS);
  let anyBright = false;
  for (let i = 0; i < PIXELS; i++) if (value[i] > threshold) { bright[i] = 1; anyBright = true; }
  if (!anyBright) return bright;
  const lab = bgrImageToLab(medianBgr);
  const activityPx = new Float32Array(PIXELS);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) activityPx[y * W + x] = activity[Math.floor(y / CELL) * GRID_W + Math.floor(x / CELL)];
  const brightActivity: number[] = [];
  for (let i = 0; i < PIXELS; i++) if (bright[i]) brightActivity.push(activityPx[i]);
  const busy = percentile(Float32Array.from(brightActivity), 75);
  const sweptLevel = f32(ACTIVITY_SHARE * busy);
  const swept = new Uint8Array(PIXELS);
  const bsValues: number[] = [];
  for (let i = 0; i < PIXELS; i++) {
    if (activityPx[i] >= sweptLevel) swept[i] = 1;
    if (bright[i] && swept[i]) bsValues.push(value[i]);
  }
  const core = new Uint8Array(PIXELS);
  if (bsValues.length) {
    const level = median(bsValues);
    for (let i = 0; i < PIXELS; i++) core[i] = bright[i] && swept[i] && value[i] >= level ? 1 : 0;
  } else core.set(bright);
  const as: number[] = [], bs: number[] = [];
  for (let i = 0; i < PIXELS; i++) if (core[i]) { as.push(lab[3 * i + 1]); bs.push(lab[3 * i + 2]); }
  const pa = f32(median(as)), pb = f32(median(bs));
  let mask: Uint8Array = new Uint8Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) {
    if (!bright[i] || !swept[i]) continue;
    const chroma = f32(Math.hypot(f32(lab[3 * i + 1] - pa), f32(lab[3 * i + 2] - pb)));
    if (chroma < PAPER_CHROMA_TOLERANCE) mask[i] = 1;
  }
  mask = open(mask, W, H, rectKernel(9, 9));
  const cc = connectedComponents(mask, W, H, 8);
  if (cc.count > 2) {
    let largest = 0;
    for (let l = 1; l < cc.count; l++) largest = Math.max(largest, cc.areas[l]);
    for (let i = 0; i < PIXELS; i++) { const l = cc.labels[i]; mask[i] = l > 0 && cc.areas[l] >= 0.15 * largest ? 1 : 0; }
  }
  return close(mask, W, H, rectKernel(21, 21));
}

export type PaperColour = [number, number, number];

/** paper_colour: brightness and Lab a, b of the paper (medians over `where`). */
export function paperColour(medianBgr: Uint8Array, where: Uint8Array): PaperColour {
  const value = hsvValue(medianBgr);
  const lab = bgrImageToLab(medianBgr);
  const v: number[] = [], a: number[] = [], b: number[] = [];
  for (let i = 0; i < where.length; i++) {
    if (!where[i]) continue;
    v.push(value[i]); a.push(lab[3 * i + 1]); b.push(lab[3 * i + 2]);
  }
  return [median(v), median(a), median(b)];
}

/** paper_like: pixels about as bright and as neutral as the paper. Takes the frame's Lab and V if already known. */
export function paperLike(frameBgr: Uint8Array, paper: PaperColour, lab?: Uint8Array, value?: Uint8Array): Uint8Array {
  const v = value ?? hsvValue(frameBgr);
  const l = lab ?? bgrImageToLab(frameBgr);
  const n = v.length;
  const out = new Uint8Array(n);
  const vMin = PAPER_VALUE_SHARE * paper[0];
  const pa = f32(paper[1]), pb = f32(paper[2]);
  for (let i = 0; i < n; i++) {
    if (v[i] < vMin) continue;
    const da = f32(l[3 * i + 1] - pa), db = f32(l[3 * i + 2] - pb);
    if (f32(Math.hypot(da, db)) < PAPER_CHROMA_TOLERANCE) out[i] = 1;
  }
  return out;
}

/** page_mask: the paper, grown out from the blocks of print. */
export function pageMask(medianBgr: Uint8Array, text: Uint8Array): Uint8Array {
  const W = PROXY_WIDTH, H = PROXY_HEIGHT;
  const textPx = resizeNearest(text, GRID_W, GRID_H, W, H);
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < PIXELS; i++) if (textPx[i]) { xs.push(i % W); ys.push(Math.floor(i / W)); }
  if (xs.length === 0) return textPx;
  const paper = paperLike(medianBgr, paperColour(medianBgr, textPx));
  const hull = new Uint8Array(PIXELS);
  fillConvexPoly(hull, W, H, convexHull(xs, ys));
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let k = 0; k < xs.length; k++) { x0 = Math.min(x0, xs[k]); x1 = Math.max(x1, xs[k]); y0 = Math.min(y0, ys[k]); y1 = Math.max(y1, ys[k]); }
  const reachX = Math.max(CELL, Math.trunc(MARGIN_X * (x1 - x0 + 1)));
  const reachY = Math.max(CELL, Math.trunc(MARGIN_Y * (y1 - y0 + 1)));
  const limit = dilate(hull, W, H, ellipseKernel(2 * reachX + 1, 2 * reachY + 1));
  let mask: Uint8Array = new Uint8Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) mask[i] = (paper[i] & limit[i]) | textPx[i];
  mask = open(mask, W, H, rectKernel(5, 5));
  for (let i = 0; i < PIXELS; i++) mask[i] |= textPx[i];
  const cc = connectedComponents(mask, W, H, 8);
  if (cc.count > 2) {
    const keep = new Uint8Array(cc.count);
    for (let i = 0; i < PIXELS; i++) if (textPx[i]) keep[cc.labels[i]] = 1;
    for (let i = 0; i < PIXELS; i++) mask[i] = keep[cc.labels[i]] && cc.labels[i] > 0 ? 1 : 0;
  }
  return close(mask, W, H, rectKernel(21, 21));
}

/** gutter_column: proxy x of the gutter, or null. */
export function gutterColumn(text: Uint8Array): number | null {
  const profile = new Float64Array(GRID_W);
  const columns: number[] = [];
  for (let x = 0; x < GRID_W; x++) {
    let s = 0;
    for (let y = 0; y < GRID_H; y++) s += text[y * GRID_W + x];
    profile[x] = s;
    if (s > 0) columns.push(x);
  }
  if (columns.length < 6) return null;
  const x0 = columns[0], x1 = columns[columns.length - 1];
  const lo = x0 + Math.trunc(0.3 * (x1 - x0)), hi = x1 - Math.trunc(0.3 * (x1 - x0));
  if (hi <= lo) return null;
  let least = Infinity;
  for (let x = lo; x <= hi; x++) least = Math.min(least, profile[x]);
  let first = -1, last = -1;
  for (let x = lo; x <= hi; x++) if (profile[x] === least) { if (first < 0) first = x; last = x; }
  return (first + last + 1) / 2 * CELL;
}

export function innerMask(mask: Uint8Array): Uint8Array {
  return erode(mask, PROXY_WIDTH, PROXY_HEIGHT, rectKernel(7, 7));
}

/** page_halves: left-page and right-page regions of the mask, leaving out the gutter. */
export function pageHalves(mask: Uint8Array, text: Uint8Array | null): [Uint8Array, Uint8Array] {
  const W = PROXY_WIDTH;
  const inner = innerMask(mask);
  let x0 = Infinity, x1 = -Infinity;
  for (let i = 0; i < inner.length; i++) if (inner[i]) { const x = i % W; x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
  if (x0 === Infinity) return [inner, inner];
  let centre = text ? gutterColumn(text) : null;
  if (centre === null) centre = (x0 + x1) / 2;
  const gap = 0.04 * (x1 - x0);
  const left = new Uint8Array(inner.length), right = new Uint8Array(inner.length);
  for (let i = 0; i < inner.length; i++) {
    if (!inner[i]) continue;
    const x = i % W;
    if (x < centre - gap) left[i] = 1;
    if (x > centre + gap) right[i] = 1;
  }
  return [left, right];
}

/** to_grid: a cell is in when most of its pixels are. */
export function toGrid(mask: Uint8Array): Uint8Array {
  const grid = new Uint8Array(CELLS);
  for (let gy = 0; gy < GRID_H; gy++) {
    for (let gx = 0; gx < GRID_W; gx++) {
      let s = 0;
      for (let y = 0; y < CELL; y++) for (let x = 0; x < CELL; x++) s += mask[(gy * CELL + y) * PROXY_WIDTH + gx * CELL + x] ? 1 : 0;
      grid[gy * GRID_W + gx] = s / (CELL * CELL) > 0.5 ? 1 : 0;
    }
  }
  return grid;
}

/** _region_motion: per frame, the share of the region's cells that moved. */
export function regionMotion(moved: Uint8Array, frames: number, region: Uint8Array): Float32Array {
  const out = new Float32Array(frames);
  const cells: number[] = [];
  for (let c = 0; c < CELLS; c++) if (region[c]) cells.push(c);
  if (cells.length === 0) return out;
  for (let f = 0; f < frames; f++) {
    let s = 0;
    const o = f * CELLS;
    for (const c of cells) s += moved[o + c];
    out[f] = s / cells.length;
  }
  return out;
}

export interface Analysis {
  fps: number;
  frames: number;
  detail: Uint8Array; // frames x CELLS
  moved: Uint8Array; // frames x CELLS, 0/1
  median: Uint8Array; // PIXELS x 3, BGR
  text: Uint8Array; // CELLS
  mask: Uint8Array; // PIXELS
  inner: Uint8Array; // PIXELS
  gridPage: Uint8Array;
  gridLeft: Uint8Array;
  gridRight: Uint8Array;
  pageMotion: Float32Array;
  leftMotion: Float32Array;
  rightMotion: Float32Array;
  pageDetail: Float32Array;
}

/** derive: mask and motion from the stored per-frame measurements. */
export function derive(fps: number, frames: number, detail: Uint8Array, moved: Uint8Array, medianBgr: Uint8Array): Analysis {
  const text = textCells(detail, frames);
  let mask: Uint8Array, left: Uint8Array, right: Uint8Array;
  if (textFill(text) >= MIN_TEXT_FILL) {
    mask = pageMask(medianBgr, text);
    [left, right] = pageHalves(mask, text);
  } else {
    const activity = new Float64Array(CELLS);
    for (let f = 0; f < frames; f++) for (let c = 0; c < CELLS; c++) activity[c] += moved[f * CELLS + c];
    for (let c = 0; c < CELLS; c++) activity[c] /= Math.max(1, frames);
    mask = pageMaskFromBrightness(medianBgr, activity);
    [left, right] = pageHalves(mask, null);
  }
  const gridPage = toGrid(mask), gridLeft = toGrid(left), gridRight = toGrid(right);
  const pageDetail = new Float32Array(frames);
  const pageCells: number[] = [];
  for (let c = 0; c < CELLS; c++) if (gridPage[c]) pageCells.push(c);
  if (pageCells.length) {
    for (let f = 0; f < frames; f++) {
      let s = 0;
      const o = f * CELLS;
      for (const c of pageCells) s += detail[o + c];
      pageDetail[f] = s / pageCells.length;
    }
  }
  return {
    fps, frames, detail, moved, median: medianBgr, text, mask, inner: innerMask(mask), gridPage, gridLeft, gridRight,
    pageMotion: regionMotion(moved, frames, gridPage), leftMotion: regionMotion(moved, frames, gridLeft),
    rightMotion: regionMotion(moved, frames, gridRight), pageDetail,
  };
}

/** print_cells: cells where print usually is on this book's pages. */
export function printCells(a: Pick<Analysis, 'gridPage' | 'detail' | 'frames'>): Uint8Array {
  const out = new Uint8Array(CELLS);
  let any = false;
  for (let c = 0; c < CELLS; c++) if (a.gridPage[c]) any = true;
  if (!any) return out;
  const usual = cellMedian(a.detail, a.frames, Math.max(1, Math.floor(a.frames / 600)));
  const onPage: number[] = [];
  for (let c = 0; c < CELLS; c++) if (a.gridPage[c]) onPage.push(usual[c]);
  const strong = percentile(onPage, 90);
  for (let c = 0; c < CELLS; c++) out[c] = usual[c] >= PRINT_LEVEL * strong && a.gridPage[c] ? 1 : 0;
  return out;
}

/** _printed_rows: first and one-past-last grid row in which a fair share of the page's cells hold print. */
export function printedRows(cells: Uint8Array, page: Uint8Array): [number, number] | null {
  let width = 0;
  for (let y = 0; y < GRID_H; y++) {
    let s = 0;
    for (let x = 0; x < GRID_W; x++) s += page[y * GRID_W + x];
    width = Math.max(width, s);
  }
  if (width === 0) return null;
  const printed: boolean[] = [];
  for (let y = 0; y < GRID_H; y++) {
    let s = 0;
    for (let x = 0; x < GRID_W; x++) s += cells[y * GRID_W + x] & page[y * GRID_W + x];
    printed.push(s >= 0.25 * width);
  }
  printed.push(false);
  const runs: [number, number][] = [];
  let start: number | null = null;
  printed.forEach((on, row) => {
    if (on && start === null) start = row;
    else if (!on && start !== null) { runs.push([start, row]); start = null; }
  });
  const long = runs.filter(r => r[1] - r[0] >= 3);
  const blocks = long.length ? long : runs;
  return blocks.length ? [blocks[0][0], blocks[blocks.length - 1][1]] : null;
}

export function usualTextRows(a: Analysis): [number, number] | null {
  const rows = printedRows(printCells(a), a.gridPage);
  return rows ? [rows[0] * CELL / PROXY_HEIGHT, rows[1] * CELL / PROXY_HEIGHT] : null;
}

/** text_rows: top and bottom of the print in one frame, as fractions of the frame height. */
export function textRows(a: Analysis, frame: number, usual: [number, number] | null): [number, number] | null {
  const onPage: number[] = [];
  const o = frame * CELLS;
  for (let c = 0; c < CELLS; c++) if (a.gridPage[c]) onPage.push(a.detail[o + c]);
  const level = onPage.length ? TEXT_LEVEL * percentile(onPage, 90) : 0;
  let here: [number, number] | null = null;
  if (level > 0) {
    const cells = new Uint8Array(CELLS);
    for (let c = 0; c < CELLS; c++) cells[c] = a.detail[o + c] >= level ? 1 : 0;
    const rows = printedRows(cells, a.gridPage);
    if (rows) here = [rows[0] * CELL / PROXY_HEIGHT, rows[1] * CELL / PROXY_HEIGHT];
  }
  if (usual === null || here === null) return here ?? usual;
  const slack = ROW_SLACK * (usual[1] - usual[0]);
  return [here[0] <= usual[0] + slack ? here[0] : usual[0], here[1] >= usual[1] - slack ? here[1] : usual[1]];
}

/** find_valleys: frames where a motion signal is a local minimum (candidate moments of rest). */
export function findValleys(motion: Float32Array, window = VALLEY_WINDOW, ceiling = VALLEY_CEILING,
  dipCeiling = VALLEY_DIP_CEILING, minProminence = VALLEY_MIN_PROMINENCE, reach = VALLEY_REACH): number[] {
  const count = motion.length;
  const valleys: number[] = [];
  for (let i = 1; i < count; i++) {
    const value = motion[i];
    if (value >= dipCeiling) continue;
    let lowest = Infinity;
    for (let k = Math.max(0, i - window); k < Math.min(count, i + window + 1); k++) lowest = Math.min(lowest, motion[k]);
    if (value > lowest) continue;
    if (value >= ceiling) {
      let before = 0, after = 0;
      if (i > 0) { before = -Infinity; for (let k = Math.max(0, i - reach); k < i; k++) before = Math.max(before, motion[k]); }
      if (i + 1 < count) { after = -Infinity; for (let k = i + 1; k < Math.min(count, i + 1 + reach); k++) after = Math.max(after, motion[k]); }
      if (Math.min(before, after) - value < minProminence) continue;
    }
    if (valleys.length && i - valleys[valleys.length - 1] <= window) {
      if (value < motion[valleys[valleys.length - 1]]) valleys[valleys.length - 1] = i;
    } else valleys.push(i);
  }
  return valleys;
}

/** page_changed: share of `region` whose brightness differs strongly between two gray frames. */
export function pageChanged(a: Uint8Array, b: Uint8Array, region: Uint8Array): number {
  let area = 0, changed = 0;
  for (let i = 0; i < region.length; i++) {
    if (!region[i]) continue;
    area++;
    if (Math.abs(a[i] - b[i]) > STRONG_THRESHOLD) changed++;
  }
  return changed / Math.max(1, area);
}
