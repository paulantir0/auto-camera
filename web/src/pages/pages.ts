// Port of auto_camera/pages.py: one clean, flat image per page from a picture
// of a two-page spread, with OpenCV.js for the image operations Python does
// with OpenCV, and src/pages/sheet.ts for the fit SciPy does there.
//
// 1. The spread is split at the fold (findFold): the wide gap in the print
//    with about as much print on either side.
// 2. On each half the lines of text are found (textSpans) and sampled along
//    their middles.
// 3. The curved-sheet model is fitted to them (fitPage), the sheet unrolled
//    (render), cut to the print and a margin (cropToPrint), and the paper
//    made evenly light (evenPaper).
// A half without lines of text is only cut out straight (plainCrop).

import { percentile } from '../pipeline/imgproc.ts';
import { FOCAL, fitSheet, project, sheetZ } from './sheet.ts';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type CV = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Mat = any;

export const WORK_LINE = 30.0;   // pages.WORK_LINE: pixels from one line of text to the next at the working scale
export const STEP = 0.8;         // pages.STEP: samples along a line of text, this many line pitches apart
export const MIN_SPANS = 8;      // pages.MIN_SPANS: fewer lines and a page is only straightened, not unrolled
export const MARGIN = 0.05;      // pages.MARGIN: white kept around the print, as a share of the print's height
export const ONE_SIDED = 0.12;   // pages.ONE_SIDED: under this share of the print on one side of the fold, that page is blank
export const WIDE_GAP = 0.02;        // the fold's gap is looked for this wide first, as a share of the frame's width
export const NARROW_GAP = 0.006;     // then this narrow (a tightly bound book: the print runs almost into the fold)
export const NARROW_BALANCE = 0.25;  // a narrow gap is the fold only with at least this share of the print on either side
export const NARROW_NEAR = 0.15;     // and only this near the middle of the book, as a share of the book's width
export const BLANK_NEAR = 0.25;      // with one page blank, the fold is looked for this near the middle of the book
export const MAX_RESIDUAL = 0.004; // pages.MAX_RESIDUAL: median distance of a sample from its fitted line, as a share of the width

export interface Spans { points: Float64Array[]; pitch: number }   // per line, x0 y0 x1 y1 ... in frame pixels

export interface PageFit {
  rvec: number[]; tvec: number[]; curve: [number, number];
  width: number; height: number;
  box: [number, number, number, number];
  residual: number; unrolled: boolean;
}

export interface Page { image: Mat; side: 'left' | 'right'; lines: number; unrolled: boolean; residual: number }

/** Run `f` and delete the Mats it lists afterwards. */
function using<T>(mats: Mat[], f: () => T): T {
  try { return f(); } finally { for (const m of mats) if (m && !m.isDeleted()) m.delete(); }
}

/** ink_mask: dark marks on paper, as a CV_8U image of 0 and 1. */
export function inkMask(cv: CV, gray: Mat, pitch: number): Mat {
  const out = new cv.Mat();
  const block = Math.trunc(pitch * 2) | 1;
  cv.adaptiveThreshold(gray, out, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, block, 18);
  return out;
}

/** cv2.resize of a 0/1 region (Uint8Array, w x h) to a Mat of the given size, nearest neighbour. */
function regionMat(cv: CV, region: Uint8Array, w: number, h: number, outW: number, outH: number): Mat {
  const src = cv.matFromArray(h, w, cv.CV_8UC1, region);
  const out = new cv.Mat();
  cv.resize(src, out, new cv.Size(outW, outH), 0, 0, cv.INTER_NEAREST);
  src.delete();
  return out;
}

/** cv2.blur of one row with a box `k` wide (BORDER_REFLECT_101). */
function blurRow(v: Float64Array, k: number): Float64Array {
  const n = v.length, half = Math.floor(k / 2), out = new Float64Array(n);
  const at = (i: number) => (n === 1 ? v[0] : i < 0 ? v[-i] : i >= n ? v[2 * n - 2 - i] : v[i]);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = i - half; j < i - half + k; j++) s += at(j);
    out[i] = s / k;
  }
  return out;
}

/** find_fold: column of the fold between the two pages. `region` is full size (or null). */
export function findFold(cv: CV, gray: Mat, region: Mat | null): number {
  const W = gray.cols, H = gray.rows;
  const sw = Math.floor(W / 4), sh = Math.floor(H / 4);
  const small = new cv.Mat(), upright = new cv.Mat(), grown = new cv.Mat();
  const smallRegion = region ? new cv.Mat() : null;
  return using([small, upright, grown, smallRegion], () => {
    cv.resize(gray, small, new cv.Size(sw, sh), 0, 0, cv.INTER_AREA);
    const ink = inkMask(cv, small, 8);
    const rect = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 15));
    const ones = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.morphologyEx(ink, upright, cv.MORPH_OPEN, rect);
    cv.dilate(upright, grown, ones);
    rect.delete(); ones.delete();
    if (smallRegion) cv.resize(region, smallRegion, new cv.Size(sw, sh), 0, 0, cv.INTER_NEAREST);
    const profileRaw = new Float64Array(sw);
    const inkD = ink.data, gD = grown.data, rD = smallRegion?.data;
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
      const i = y * sw + x;
      if (inkD[i] && !gD[i] && (!rD || rD[i])) profileRaw[x] += 1;
    }
    ink.delete();
    const profile = blurRow(profileRaw, 5);
    let sum = 0;
    for (const v of profile) sum += v;
    if (sum <= 0) return Math.floor(W / 2);
    const positive = Array.from(profile).filter(v => v > 0);
    const busy = percentile(positive, 75);
    const gap = (share: number) => blurRow(profile, Math.max(3, Math.trunc(sw * share)) | 1).map(v => Math.min(1, Math.max(0, 1 - v / busy)));
    const text = profile.map(v => (v > 0.3 * busy ? v : 0));
    let textSum = 0;
    for (const v of text) textSum += v;
    if (textSum <= 0) return Math.floor(W / 2);
    const total = new Float64Array(sw);
    let acc = 0;
    for (let x = 0; x < sw; x++) { acc += text[x]; total[x] = acc; }
    const last = total[sw - 1];
    const balance = (x: number) => Math.min(total[x], last - total[x]) / last;
    // the book: where print usually is, else the whole frame
    let left = 0, right = sw;
    if (rD) {
      let c0 = Infinity, c1 = -Infinity;
      for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) if (rD[y * sw + x]) { c0 = Math.min(c0, x); c1 = Math.max(c1, x); }
      if (c0 !== Infinity) { left = c0; right = c1; }
    }
    const middle = (left + right) / 2;
    const lo = Math.trunc(sw * 0.2), hi = Math.trunc(sw * 0.8);
    const best = (emptiness: Float64Array | number[], near: number) => {
      let fold = 0, top = -Infinity;
      for (let x = 0; x < sw; x++) {
        const score = x < lo || x >= hi || Math.abs(x - middle) > near ? -1 : balance(x) * emptiness[x];
        if (score > top) { top = score; fold = x; }
      }
      return fold;
    };
    // a wide gap (the inner margins of two pages); a column between two words inside a block of print is narrow
    const emptiness = gap(WIDE_GAP);
    let fold = best(emptiness, Infinity);
    if (balance(fold) >= ONE_SIDED && emptiness[fold] >= 0.6) return fold * 4;
    // a tightly bound book: a narrow gap, near the middle of the book, with print on both sides
    const narrow = gap(NARROW_GAP);
    fold = best(narrow, NARROW_NEAR * (right - left));
    if (balance(fold) >= NARROW_BALANCE && narrow[fold] >= 0.6) return fold * 4;
    // one page blank: the empty column nearest the middle, in the middle half of the book
    let pick = -1, distance = Infinity;
    for (let x = 0; x < sw; x++) {
      if (!(emptiness[x] > 0.8) || !(x > sw * 0.2 && x < sw * 0.8) || Math.abs(x - middle) > BLANK_NEAR * (right - left)) continue;
      if (Math.abs(x - middle) < distance) { distance = Math.abs(x - middle); pick = x; }
    }
    return pick < 0 ? Math.trunc(middle) * 4 : pick * 4;
  });
}

/** np.hanning */
const hanning = (m: number) => Float64Array.from({ length: m }, (_, n) => (m === 1 ? 1 : 0.5 - 0.5 * Math.cos(2 * Math.PI * n / (m - 1))));

/** estimate_pitch: distance between lines of text in pixels, the page turned to its best angle. `where` is a Mat the size of `gray`, or null. */
export function estimatePitch(cv: CV, gray: Mat, where: Mat | null): number {
  const w0 = Math.max(64, Math.floor(gray.cols / 2)), h0 = Math.max(64, Math.floor(gray.rows / 2));
  const small = new cv.Mat(), inkF = new cv.Mat(), smallWhere = where ? new cv.Mat() : null;
  return using([small, inkF, smallWhere], () => {
    cv.resize(gray, small, new cv.Size(w0, h0), 0, 0, cv.INTER_AREA);
    const ink = inkMask(cv, small, 30);
    if (smallWhere) {
      cv.resize(where, smallWhere, new cv.Size(w0, h0), 0, 0, cv.INTER_NEAREST);
      const d = ink.data, r = smallWhere.data;
      for (let i = 0; i < d.length; i++) if (!r[i]) d[i] = 0;
    }
    ink.convertTo(inkF, cv.CV_32F);
    ink.delete();
    const h = h0, w = w0;
    let bestPower = 0, bestPeriod = 20;
    const turned = new cv.Mat();
    const window = hanning(h);
    for (const angle of [-14, -10, -6, -3, 0, 3, 6, 10, 14]) {
      const m = cv.getRotationMatrix2D(new cv.Point(w / 2, h / 2), angle, 1.0);
      cv.warpAffine(inkF, turned, m, new cv.Size(w, h));
      m.delete();
      const t = turned.data32F as Float32Array;
      const profile = new Float64Array(h);
      const a = Math.floor(w / 4), b = Math.floor(3 * w / 4);
      for (let y = 0; y < h; y++) { let s = 0; for (let x = a; x < b; x++) s += t[y * w + x]; profile[y] = s; }
      let mean = 0;
      for (const v of profile) mean += v;
      mean /= h;
      for (let y = 0; y < h; y++) profile[y] = (profile[y] - mean) * window[y];
      // |rfft| at the frequencies whose period is usable: 6 < h / k < h / 8
      let index = -1, power = 0;
      for (let k = 1; k <= Math.floor(h / 2); k++) {
        const period = h / k;
        if (!(period > 6 && period < h / 8)) continue;
        let re = 0, im = 0;
        for (let y = 0; y < h; y++) { const ph = -2 * Math.PI * k * y / h; re += profile[y] * Math.cos(ph); im += profile[y] * Math.sin(ph); }
        const p = Math.hypot(re, im);
        if (p > power) { power = p; index = k; }
      }
      if (index < 0) continue;
      if (power > bestPower) { bestPower = power; bestPeriod = h / index; }
    }
    turned.delete();
    return bestPeriod * 2;
  });
}

const slopeOf = (p: Float64Array) => {
  const n = p.length / 2;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += p[2 * i]; my += p[2 * i + 1]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { const dx = p[2 * i] - mx; sxy += dx * (p[2 * i + 1] - my); sxx += dx * dx; }
  return sxx > 0 ? sxy / sxx : 0;
};
const medianOf = (v: number[]) => { const s = [...v].sort((a, b) => a - b), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

/** text_spans: lines of text between columns x0 and x1 of a frame, sampled along their middles. */
export function textSpans(cv: CV, gray: Mat, x0: number, x1: number, region: Mat | null): Spans {
  const rect = new cv.Rect(x0, 0, x1 - x0, gray.rows);
  const part = gray.roi(rect);
  const partRegion = region ? region.roi(rect) : null;
  const work = new cv.Mat(), workRegion = region ? new cv.Mat() : null, inkF = new cv.Mat(), density = new cv.Mat(), core = new cv.Mat();
  const labels = new cv.Mat(), stats = new cv.Mat(), centroids = new cv.Mat();
  return using([part, partRegion, work, workRegion, inkF, density, core, labels, stats, centroids], () => {
    const pitch = estimatePitch(cv, part, partRegion);
    const scale = WORK_LINE / pitch;
    cv.resize(part, work, new cv.Size(0, 0), scale, scale, cv.INTER_AREA);
    const ink = inkMask(cv, work, WORK_LINE);
    if (workRegion) {
      cv.resize(partRegion, workRegion, new cv.Size(work.cols, work.rows), 0, 0, cv.INTER_NEAREST);
      const d = ink.data, r = workRegion.data;
      for (let i = 0; i < d.length; i++) if (!r[i]) d[i] = 0;
    }
    ink.convertTo(inkF, cv.CV_32F);
    ink.delete();
    cv.blur(inkF, density, new cv.Size(Math.trunc(WORK_LINE), 1));
    const dens = density.data32F as Float32Array;
    const positive: number[] = [];
    for (let i = 0; i < dens.length; i++) if (dens[i] > 0) positive.push(dens[i]);
    const level = positive.length ? percentile(positive, 90) : 1.0;
    const raw = new cv.Mat(work.rows, work.cols, cv.CV_8UC1);
    const rd = raw.data;
    for (let i = 0; i < dens.length; i++) rd[i] = dens[i] > 0.4 * level ? 1 : 0;
    const k3 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    cv.morphologyEx(raw, core, cv.MORPH_OPEN, k3);
    raw.delete(); k3.delete();
    const count = cv.connectedComponentsWithStats(core, labels, stats, centroids);
    const L = labels.data32S as Int32Array, st = stats.data32S as Int32Array, ww = work.cols;
    const step = Math.max(4, Math.trunc(WORK_LINE * STEP));
    const spans: Spans = { points: [], pitch };
    for (let index = 1; index < count; index++) {
      const x = st[index * 5], y = st[index * 5 + 1], w = st[index * 5 + 2], h = st[index * 5 + 3];
      if (w < 3 * WORK_LINE) continue;
      const heights: number[] = [];
      for (let cx = 0; cx < w; cx++) {
        let n = 0;
        for (let cy = 0; cy < h; cy++) if (L[(y + cy) * ww + x + cx] === index) n++;
        if (n > 0) heights.push(n);
      }
      const thickness = medianOf(heights);
      if (thickness > 0.75 * WORK_LINE || thickness < 0.12 * WORK_LINE) continue;
      const pts: number[] = [];
      for (let start = 0; start < w - Math.floor(step / 2); start += step) {
        let n = 0, sum = 0, lo = Infinity, hi = -Infinity;
        for (let cy = 0; cy < h; cy++) {
          for (let cx = start; cx < Math.min(w, start + step); cx++) {
            if (L[(y + cy) * ww + x + cx] === index) { n++; sum += cy; lo = Math.min(lo, cy); hi = Math.max(hi, cy); }
          }
        }
        if (n < 3 || hi - lo > 1.2 * WORK_LINE) continue;
        pts.push(x + start + Math.min(step, w - start) / 2, y + sum / n);
      }
      if (pts.length < 6) continue;
      const points = Float64Array.from(pts);
      if (Math.abs(slopeOf(points)) > 0.35) continue;
      for (let i = 0; i < points.length; i += 2) { points[i] = points[i] / scale + x0; points[i + 1] /= scale; }
      spans.points.push(points);
    }
    return mainBlock(spans);
  });
}

/** _main_block: only lines that run like most of the others and have another line a pitch or two above or below. */
export function mainBlock(spans: Spans): Spans {
  if (spans.points.length < 4) return spans;
  const angles = spans.points.map(p => Math.atan(slopeOf(p)));
  const usual = medianOf(angles);
  const level = spans.points.filter((_, k) => Math.abs(angles[k] - usual) < 6 * Math.PI / 180);
  const keep: Float64Array[] = [];
  const range = (p: Float64Array) => { let lo = Infinity, hi = -Infinity; for (let i = 0; i < p.length; i += 2) { lo = Math.min(lo, p[i]); hi = Math.max(hi, p[i]); } return [lo, hi]; };
  const middleY = (p: Float64Array) => medianOf(Array.from({ length: p.length / 2 }, (_, i) => p[2 * i + 1]));
  level.forEach((points, index) => {
    const [lo, hi] = range(points), y = middleY(points);
    for (let other = 0; other < level.length; other++) {
      if (other === index) continue;
      const [olo, ohi] = range(level[other]);
      if (Math.min(hi, ohi) - Math.max(lo, olo) <= 0) continue;
      const gap = Math.abs(middleY(level[other]) - y);
      if (0.5 * spans.pitch < gap && gap < 2.5 * spans.pitch) { keep.push(points); break; }
    }
  });
  return { points: keep, pitch: spans.pitch };
}

/** print_outline: four corners around the lines of text (top left, top right, bottom right, bottom left), or null. */
export function printOutline(spans: Spans): number[][] | null {
  if (spans.points.length < 3) return null;
  const angle = medianOf(spans.points.map(p => Math.atan(slopeOf(p))));
  const along = [Math.cos(angle), Math.sin(angle)], across = [-Math.sin(angle), Math.cos(angle)];
  const u: number[] = [], v: number[] = [];
  for (const p of spans.points) for (let i = 0; i < p.length; i += 2) { u.push(p[i] * along[0] + p[i + 1] * along[1]); v.push(p[i] * across[0] + p[i + 1] * across[1]); }
  const u0 = percentile(u, 0.5), u1 = percentile(u, 99.5);
  const v0 = Math.min(...v) - spans.pitch * 0.5, v1 = Math.max(...v) + spans.pitch * 0.5;
  const at = (a: number, b: number) => [a * along[0] + b * across[0], a * along[1] + b * across[1]];
  return [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)];
}

/** fit_page: the curved-sheet model fitted to the lines of one page. */
export function fitPage(cv: CV, spans: Spans, outline: number[][], width: number, height: number): PageFit {
  const scale = 2.0 / width, cx = width / 2, cy = height / 2;
  const corners = outline.map(([x, y]) => [(x - cx) * scale, (y - cy) * scale]);
  const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const sheetW = (dist(corners[1], corners[0]) + dist(corners[2], corners[3])) / 2;
  const sheetH = (dist(corners[3], corners[0]) + dist(corners[2], corners[1])) / 2;
  const plane = [[0, 0, 0], [sheetW, 0, 0], [sheetW, sheetH, 0], [0, sheetH, 0]];
  const obj = cv.matFromArray(4, 3, cv.CV_64F, plane.flat());
  const img = cv.matFromArray(4, 2, cv.CV_64F, corners.flat());
  const K = cv.matFromArray(3, 3, cv.CV_64F, [FOCAL, 0, 0, 0, FOCAL, 0, 0, 0, 1]);
  const dist0 = new cv.Mat(), rv = new cv.Mat(), tv = new cv.Mat();
  let rvec: number[], tvec: number[];
  try {
    let ok = cv.solvePnP(obj, img, K, dist0, rv, tv, false, cv.SOLVEPNP_IPPE);
    if (!ok) ok = cv.solvePnP(obj, img, K, dist0, rv, tv, false, cv.SOLVEPNP_ITERATIVE);
    rvec = Array.from(rv.data64F as Float64Array).slice(0, 3);
    tvec = Array.from(tv.data64F as Float64Array).slice(0, 3);
  } finally {
    for (const m of [obj, img, K, dist0, rv, tv]) m.delete();
  }
  if (spans.points.length < MIN_SPANS) {
    return { rvec, tvec, curve: [0, 0], width: sheetW, height: sheetH, box: [0, 0, sheetW, sheetH], residual: 0, unrolled: false };
  }
  // where each sample lies on the plane of the outline, to start from
  const src = cv.matFromArray(4, 1, cv.CV_32FC2, corners.flat());
  const dst = cv.matFromArray(4, 1, cv.CV_32FC2, plane.map(p => [p[0], p[1]]).flat());
  const M = cv.getPerspectiveTransform(src, dst);
  const h = Array.from(M.data64F as Float64Array);
  src.delete(); dst.delete(); M.delete();
  const observed: number[] = [], spanOf: number[] = [], startX: number[] = [], startY: number[] = [];
  spans.points.forEach((points, index) => {
    let sy = 0;
    const n = points.length / 2;
    for (let i = 0; i < n; i++) {
      const nx = (points[2 * i] - cx) * scale, ny = (points[2 * i + 1] - cy) * scale;
      observed.push(nx, ny);
      spanOf.push(index);
      const fx = Math.fround(nx), fy = Math.fround(ny);
      const wq = h[6] * fx + h[7] * fy + h[8];
      startX.push((h[0] * fx + h[1] * fy + h[2]) / wq);
      sy += (h[3] * fx + h[4] * fy + h[5]) / wq;
    }
    startY.push(sy / n);
  });
  const S = spans.points.length;
  const start = Float64Array.from([...rvec, ...tvec, 0, 0, ...startY, ...startX]);
  const pitch = spans.pitch * scale;
  const fit = fitSheet({ observed: Float64Array.from(observed), spanOf: Int32Array.from(spanOf), spans: S, width: sheetW, scale: pitch * 0.15 }, start, 200);
  const p = fit.params;
  const xs = Array.from(p.subarray(8 + S)), ys = Array.from(p.subarray(8, 8 + S));
  const abs = Array.from(fit.residuals, Math.abs);
  const residual = medianOf(abs) / 2;
  const box: [number, number, number, number] = [percentile(xs, 1), Math.min(...ys) - pitch * 0.5, percentile(xs, 99), Math.max(...ys) + pitch * 0.5];
  return { rvec: Array.from(p.subarray(0, 3)), tvec: Array.from(p.subarray(3, 6)), curve: [p[6], p[7]], width: sheetW, height: sheetH, box, residual, unrolled: true };
}

/** render: the page unrolled flat, the print and `pad` (sheet units) around it, at about the resolution it has in the frame. */
export function render(cv: CV, image: Mat, fit: PageFit, pad: [number, number]): Mat {
  const width = image.cols, height = image.rows, scale = 2.0 / width, cx = width / 2, cy = height / 2;
  const [bx0, by0, bx1, by1] = fit.box;
  const x0 = bx0 - pad[0], y0 = by0 - pad[1], x1 = bx1 + pad[0], y1 = by1 + pad[1];
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  const mid = project([mx, mx], [my, my + 0.01], [sheetZ(mx, fit.width, ...fit.curve), sheetZ(mx, fit.width, ...fit.curve)], fit.rvec, fit.tvec, new Float64Array(4));
  const perUnit = Math.hypot(mid[2] - mid[0], mid[3] - mid[1]) / 0.01 / scale;
  const outW = Math.max(16, Math.min(Math.round((x1 - x0) * perUnit), 6000));
  const outH = Math.max(16, Math.min(Math.round((y1 - y0) * perUnit), 8000));
  const gw = Math.max(2, Math.floor(outW / 16)), gh = Math.max(2, Math.floor(outH / 16));
  const gx: number[] = [], gy: number[] = [], gz: number[] = [];
  for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
    const x = x0 + (x1 - x0) * i / (gw - 1), y = y0 + (y1 - y0) * j / (gh - 1);
    gx.push(x); gy.push(y); gz.push(sheetZ(x, fit.width, ...fit.curve));
  }
  const pts = project(gx, gy, gz, fit.rvec, fit.tvec, new Float64Array(2 * gx.length));
  const coarseX = new cv.Mat(gh, gw, cv.CV_32FC1), coarseY = new cv.Mat(gh, gw, cv.CV_32FC1);
  const cxd = coarseX.data32F as Float32Array, cyd = coarseY.data32F as Float32Array;
  for (let k = 0; k < gx.length; k++) { cxd[k] = pts[2 * k] / scale + cx; cyd[k] = pts[2 * k + 1] / scale + cy; }
  const mapX = new cv.Mat(), mapY = new cv.Mat(), out = new cv.Mat();
  try {
    cv.resize(coarseX, mapX, new cv.Size(outW, outH), 0, 0, cv.INTER_CUBIC);
    cv.resize(coarseY, mapY, new cv.Size(outW, outH), 0, 0, cv.INTER_CUBIC);
    cv.remap(image, out, mapX, mapY, cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
  } finally {
    for (const m of [coarseX, coarseY, mapX, mapY]) m.delete();
  }
  return out;
}

/** crop_to_print: a flattened page cut to its print, with the running head, page number and notes near it, and a margin. */
export function cropToPrint(cv: CV, page: Mat, pitch: number, core: [number, number, number, number], margin = MARGIN): Mat {
  const gray = new cv.Mat(), grown = new cv.Mat(), labels = new cv.Mat(), stats = new cv.Mat(), centroids = new cv.Mat();
  return using([gray, grown, labels, stats, centroids], () => {
    cv.cvtColor(page, gray, cv.COLOR_RGBA2GRAY);
    const ink = inkMask(cv, gray, pitch);
    const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(Math.max(3, Math.trunc(pitch * 0.6)), Math.max(3, Math.trunc(pitch * 0.3))));
    cv.dilate(ink, grown, k);
    ink.delete(); k.delete();
    const count = cv.connectedComponentsWithStats(grown, labels, stats, centroids);
    const st = stats.data32S as Int32Array;
    let [x0, y0, x1, y1] = core;
    const reach = 3 * pitch, height = gray.rows, width = gray.cols;
    for (let index = 1; index < count; index++) {
      const x = st[index * 5], y = st[index * 5 + 1], w = st[index * 5 + 2], h = st[index * 5 + 3];
      if (x <= 1 || y <= 1 || x + w >= width - 1 || y + h >= height - 1) continue;
      if (h > 2 * pitch) continue;
      if (x + w < x0 - reach || x > x1 + reach || y + h < y0 - reach || y > y1 + reach) continue;
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + w); y1 = Math.max(y1, y + h);
    }
    const padding = Math.round(margin * (core[3] - core[1]));
    x0 = Math.max(0, x0 - padding); y0 = Math.max(0, y0 - padding);
    x1 = Math.min(width, x1 + padding); y1 = Math.min(height, y1 + padding);
    const roi = page.roi(new cv.Rect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)));
    const out = roi.clone();
    roi.delete();
    return out;
  });
}

/** even_paper: the page with its paper made evenly light (lightness divided by a closing of it); colour kept. */
export function evenPaper(cv: CV, page: Mat, pitch: number): Mat {
  const rgb = new cv.Mat(), lab = new cv.Mat(), planes = new cv.MatVector(), light = new cv.Mat(), paper = new cv.Mat(), blurred = new cv.Mat();
  return using([rgb, lab, planes, light, paper, blurred], () => {
    cv.cvtColor(page, rgb, cv.COLOR_RGBA2RGB);
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    cv.split(lab, planes);
    const L = planes.get(0);
    L.convertTo(light, cv.CV_32F);
    const size = Math.max(7, Math.trunc(pitch * 1.5)) | 1;
    const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));
    cv.morphologyEx(light, paper, cv.MORPH_CLOSE, k);
    k.delete();
    cv.GaussianBlur(paper, blurred, new cv.Size(0, 0), size);
    const l = light.data32F as Float32Array, pp = blurred.data32F as Float32Array, out = L.data as Uint8Array;
    for (let i = 0; i < out.length; i++) out[i] = Math.min(255, Math.max(0, Math.trunc(l[i] / Math.max(pp[i], 1.0) * 245.0)));
    planes.set(0, L);
    L.delete();
    cv.merge(planes, lab);
    const back = new cv.Mat();
    cv.cvtColor(lab, rgb, cv.COLOR_Lab2RGB);
    cv.cvtColor(rgb, back, cv.COLOR_RGB2RGBA);
    return back;
  });
}

/** _plain_crop: the part of `region` between columns x0 and x1, cut out along its tilted outline; null without enough region there. */
export function plainCrop(cv: CV, image: Mat, x0: number, x1: number, region: Mat | null): Mat | null {
  if (!region) return null;
  const W = region.cols, H = region.rows, d = region.data;
  const pts: number[] = [];
  for (let y = 0; y < H; y++) for (let x = x0; x < x1; x++) if (d[y * W + x]) pts.push(x, y);
  if (pts.length / 2 < 0.02 * W * H) return null;
  // the outline of a region: its convex hull is enough for the smallest rectangle
  const mat = cv.matFromArray(pts.length / 2, 1, cv.CV_32FC2, pts);
  const r = cv.minAreaRect(mat);
  mat.delete();
  let w = r.size.width, h = r.size.height, angle = r.angle;
  if (w < h * 0.2 || h < w * 0.2) return null;
  if (angle > 45) { angle -= 90; [w, h] = [h, w]; }
  const turn = cv.getRotationMatrix2D(new cv.Point(r.center.x, r.center.y), angle, 1.0);
  const t = turn.data64F as Float64Array;
  t[2] += w / 2 - r.center.x;
  t[5] += h / 2 - r.center.y;
  const out = new cv.Mat();
  cv.warpAffine(image, out, turn, new cv.Size(Math.trunc(w), Math.trunc(h)), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar());
  turn.delete();
  return out;
}

/** _render_with_room: the page flattened with four lines' room around the print; where the print lies in it, and the pitch there. */
function renderWithRoom(cv: CV, image: Mat, fit: PageFit, pitchPx: number): { flat: Mat; core: [number, number, number, number]; pitch: number } {
  const scale = 2.0 / image.cols;
  const [x0, y0, x1, y1] = fit.box;
  const room = 4 * pitchPx * scale;
  const flat = render(cv, image, fit, [room, room]);
  const perX = flat.cols / (x1 - x0 + 2 * room), perY = flat.rows / (y1 - y0 + 2 * room);
  return {
    flat,
    core: [Math.trunc(room * perX), Math.trunc(room * perY), Math.trunc(flat.cols - room * perX), Math.trunc(flat.rows - room * perY)],
    pitch: pitchPx * scale * perY,
  };
}

/**
 * page_images: the two pages of a spread (an RGBA Mat), each flat and cropped;
 * fewer if a half holds no print. `region` is where the print is in the
 * frame (the 480x270 print map, enlarged), or null.
 */
export function pageImages(cv: CV, image: Mat, region: { data: Uint8Array; width: number; height: number } | null, even = true): Page[] {
  const gray = new cv.Mat();
  const full = region ? regionMat(cv, region.data, region.width, region.height, image.cols, image.rows) : null;
  return using([gray, full], () => {
    cv.cvtColor(image, gray, cv.COLOR_RGBA2GRAY);
    const fold = findFold(cv, gray, full);
    const pages: Page[] = [];
    for (const [side, x0, x1] of [['left', 0, fold], ['right', fold, image.cols]] as ['left' | 'right', number, number][]) {
      if (x1 - x0 < 16) continue;
      const spans = textSpans(cv, gray, x0, x1, full);
      const outline = printOutline(spans);
      if (!outline) {
        const plain = plainCrop(cv, image, x0, x1, full);
        if (plain) {
          const page = even ? evenPaper(cv, plain, 30.0) : plain;
          if (even) plain.delete();
          pages.push({ image: page, side, lines: 0, unrolled: false, residual: 0 });
        }
        continue;
      }
      let fit = fitPage(cv, spans, outline, image.cols, image.rows);
      if (fit.unrolled && fit.residual > MAX_RESIDUAL) fit = fitPage(cv, { points: [], pitch: spans.pitch }, outline, image.cols, image.rows);
      const { flat, core, pitch } = renderWithRoom(cv, image, fit, spans.pitch);
      const cropped = cropToPrint(cv, flat, pitch, core);
      flat.delete();
      let page = cropped;
      if (even) { page = evenPaper(cv, cropped, pitch); cropped.delete(); }
      pages.push({ image: page, side, lines: spans.points.length, unrolled: fit.unrolled, residual: fit.residual });
    }
    return pages;
  });
}
