// Port of the print map of auto_camera/hands.py (calm_frames, print_maps,
// _print_map, fold_column) and of the zones measure() derives from it.
//
// Where print is on this book's pages is learned per spot from the frames in
// which the book lies still and nothing covers the spot: fine detail that
// changes from spread to spread. Python keeps every such frame; here each is
// added to running sums as it is decoded (PrintSums), which is the same
// arithmetic, in float32 like NumPy's.

import {
  GRID_W, HANDS_PRINT_LEVEL, MIN_SEEN, PRINT_CALM, PRINT_CHANGE, PRINT_INSIDE, PRINT_JOIN, PRINT_NEAR, PRINT_REACH, PRINT_SHARE,
  PRINT_SHARED, PROXY_HEIGHT, PROXY_WIDTH,
} from './constants.ts';
import { PIXELS, type Analysis } from './analysis.ts';
import {
  boxPoints, close, connectedComponents, dilate, ellipseKernel, erode, fillConvexPolyClipped, minAreaRect, open, rectKernel, roundHalfEven,
} from './imgproc.ts';

const W = PROXY_WIDTH, H = PROXY_HEIGHT;
const f32 = Math.fround;

/** calm_frames: up to `count` frames, spread over the video, in which the book lies still (or, in a video that never rests, its calmest). */
export function calmFrames(a: Pick<Analysis, 'pageMotion'>, count: number): number[] {
  const m = a.pageMotion, n = m.length;
  const moving = new Float32Array(n);
  for (let i = 0; i < n; i++) moving[i] = Math.max(m[i], i + 1 < n ? m[i + 1] : m[n - 1]);
  let calm: number[] = [];
  const limit = f32(PRINT_CALM);
  for (let i = 0; i < n; i++) if (moving[i] <= limit) calm.push(i);
  if (calm.length < Math.floor(count / 3)) {
    const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => moving[x] - moving[y] || x - y);
    calm = order.slice(0, count).sort((x, y) => x - y);
  }
  const L = calm.length, k = Math.min(count, L);
  if (!k) return [];
  // np.linspace(0, L - 1, k).astype(int)
  const picks = new Set<number>();
  const step = k > 1 ? (L - 1) / (k - 1) : 0;
  for (let j = 0; j < k; j++) picks.add(calm[Math.trunc(j === k - 1 && k > 1 ? L - 1 : j * step)]);
  return [...picks].sort((x, y) => x - y);
}

/** Median of the uint8 values of `values` where `where` is set (np.median of a uint8 selection). */
export function maskedMedianU8(values: Uint8Array, where: Uint8Array): number {
  const counts = new Uint32Array(256);
  let n = 0;
  for (let i = 0; i < values.length; i++) if (where[i]) { counts[values[i]]++; n++; }
  if (!n) return NaN;
  const k1 = Math.floor((n - 1) / 2), k2 = Math.floor(n / 2);
  let seen = 0, a = -1, b = -1;
  for (let v = 0; v < 256; v++) {
    seen += counts[v];
    if (a < 0 && seen > k1) a = v;
    if (seen > k2) { b = v; break; }
  }
  return (a + b) / 2;
}

/** The sums _print_map takes over the frames at rest: per spot, in how many it was seen, showed print, and its brightness. */
export class PrintSums {
  readonly seen = new Int32Array(PIXELS);
  readonly printed = new Int32Array(PIXELS);
  readonly light = new Float32Array(PIXELS);
  readonly light2 = new Float32Array(PIXELS);
  frames = 0;

  /** One frame at rest: its gray image, its fine detail, where nothing covers it, and the detail from which a spot shows print. */
  add(gray: Uint8Array, detail: Uint8Array, visible: Uint8Array, level: number): void {
    for (let i = 0; i < PIXELS; i++) {
      if (!visible[i]) continue;
      this.seen[i]++;
      if (detail[i] >= level) this.printed[i]++;
      const g = gray[i];
      this.light[i] = f32(this.light[i] + g);
      this.light2[i] = f32(this.light2[i] + g * g);
    }
    this.frames++;
  }

  /** Add another stretch's sums (frames later in the video). */
  merge(other: { seen: Int32Array; printed: Int32Array; light: Float32Array; light2: Float32Array; frames: number }): void {
    for (let i = 0; i < PIXELS; i++) {
      this.seen[i] += other.seen[i];
      this.printed[i] += other.printed[i];
      this.light[i] = f32(this.light[i] + other.light[i]);
      this.light2[i] = f32(this.light2[i] + other.light2[i]);
    }
    this.frames += other.frames;
  }
}

/** fold_column: column of the fold in an image `width` wide, midway between the page cells of the left page and those of the right. */
export function foldColumn(a: Pick<Analysis, 'gridLeft' | 'gridRight'>, width: number): number {
  let leftMax = -1, rightMin = -1;
  for (let x = 0; x < GRID_W; x++) {
    let l = false, r = false;
    for (let c = x; c < a.gridLeft.length; c += GRID_W) { if (a.gridLeft[c]) l = true; if (a.gridRight[c]) r = true; }
    if (l) leftMax = x;
    if (r && rightMin < 0) rightMin = x;
  }
  if (leftMax >= 0 && rightMin >= 0) return roundHalfEven((leftMax + 1 + rightMin) / 2 * width / GRID_W);
  return Math.floor(width / 2);
}

/** _print_map from the sums over the frames at rest; `cells` is print_area. Returns the map, filled out to a rectangle per page. */
export function printMap(a: Pick<Analysis, 'mask' | 'gridLeft' | 'gridRight'>, cells: Uint8Array, sums: PrintSums): Uint8Array {
  const n = sums.frames;
  const change = new Float32Array(PIXELS);
  const share = new Float64Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) {
    const count = Math.max(sums.seen[i], 1);
    share[i] = sums.printed[i] / count;
    const mean = f32(sums.light[i] / count);
    change[i] = f32(Math.sqrt(Math.max(f32(f32(sums.light2[i] / count) - f32(mean * mean)), 0)));
  }
  const atCells: number[] = [];
  for (let i = 0; i < PIXELS; i++) if (cells[i]) atCells.push(change[i]);
  atCells.sort((x, y) => x - y);
  const m = atCells.length;
  const usualChange = m % 2 ? atCells[(m - 1) / 2] : f32(f32(atCells[m / 2 - 1] + atCells[m / 2]) / 2);
  const changeLevel = f32(PRINT_CHANGE * usualChange);
  const minSeen = Math.min(MIN_SEEN, Math.floor(n / 3));
  const reach = 2 * PRINT_REACH + 1;
  const near = dilate(a.mask, W, H, rectKernel(reach, reach));
  let found: Uint8Array = new Uint8Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) {
    found[i] = share[i] >= PRINT_SHARE && sums.seen[i] >= minSeen && change[i] >= changeLevel && near[i] ? 1 : 0;
  }
  found = open(found, W, H, rectKernel(3, 3));
  // A running head or a folio stands a line or two clear of the body of the page; the stack of page edges is a patch of its own.
  found = close(found, W, H, rectKernel(3, PRINT_JOIN));
  found = close(found, W, H, rectKernel(5, 5));
  const cc = connectedComponents(found, W, H, 8);
  if (cc.count > 1) {
    const shared = new Int32Array(cc.count), area = new Int32Array(cc.count);
    for (let i = 0; i < PIXELS; i++) { area[cc.labels[i]]++; if (cells[i]) shared[cc.labels[i]]++; }
    const keep = new Uint8Array(cc.count);
    for (let l = 1; l < cc.count; l++) keep[l] = shared[l] >= PRINT_SHARED * area[l] ? 1 : 0;
    for (let i = 0; i < PIXELS; i++) found[i] = keep[cc.labels[i]];
  }
  let foundSum = 0, cellsSum = 0;
  for (let i = 0; i < PIXELS; i++) { foundSum += found[i]; cellsSum += cells[i] ? 1 : 0; }
  if (foundSum < 0.5 * cellsSum) return cells.slice();
  // A page of print is a rectangle, seen a little askew: filled out page by page, told apart at the fold.
  const fold = foldColumn(a, W);
  const filled = found.slice();
  for (const [lo, hi] of [[0, fold], [fold, W]]) {
    const xs: number[] = [], ys: number[] = [];
    for (let y = 0; y < H; y++) for (let x = lo; x < hi; x++) if (found[y * W + x]) { xs.push(x); ys.push(y); }
    if (xs.length < 50) continue;
    const box = boxPoints(minAreaRect(xs, ys)).map(([x, y]) => [roundHalfEven(x), roundHalfEven(y)] as [number, number]);
    const block = new Uint8Array(PIXELS);
    fillConvexPolyClipped(block, W, H, box);
    for (let y = 0; y < H; y++) for (let x = lo; x < hi; x++) if (block[y * W + x]) filled[y * W + x] = 1;
  }
  return filled;
}

/** The level of fine detail from which a spot shows print: PRINT_LEVEL times the median over the frames at rest of each one's median on the print cells. */
export function printLevel(perFrameMedians: number[]): number {
  if (!perFrameMedians.length) return 0;
  const s = [...perFrameMedians].sort((x, y) => x - y), k = s.length;
  const usual = k % 2 ? s[(k - 1) / 2] : (s[k / 2 - 1] + s[k / 2]) / 2;
  return HANDS_PRINT_LEVEL * usual;
}

/** The areas of measure(): the print and what lies around it, and the page's parts. */
export interface Zones {
  textAll: Uint8Array;   // the print map, or where print usually is (``text_all``)
  inside: Uint8Array;    // PRINT_INSIDE in from its edge: what the review list counts
  near: Uint8Array;      // within PRINT_NEAR of it: what frame choice counts
  region: Uint8Array;    // the page region and the map (the strict colour model's reach)
  wellInside: Uint8Array;// the map, PRINT_INSIDE in from its edge (the strict colour model's)
  textArea: number;
  insideArea: number;
  mapArea: number;
  pageArea: number;
}

export function zones(a: Pick<Analysis, 'mask' | 'inner'>, cells: Uint8Array, map: Uint8Array): Zones {
  const edge = 2 * PRINT_INSIDE + 1;
  const textAll = new Uint8Array(PIXELS), region = new Uint8Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) { textAll[i] = map[i] || cells[i] ? 1 : 0; region[i] = a.inner[i] || map[i] ? 1 : 0; }
  const inside = erode(textAll, W, H, rectKernel(edge, edge));
  const wellInside = erode(map, W, H, rectKernel(edge, edge));
  const near = dilate(textAll, W, H, ellipseKernel(2 * PRINT_NEAR + 1, 2 * PRINT_NEAR + 1));
  const sum = (x: Uint8Array) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] ? 1 : 0; return s; };
  return {
    textAll, inside, near, region, wellInside,
    textArea: Math.max(1, sum(textAll)), insideArea: Math.max(1, sum(inside)), mapArea: Math.max(1, sum(map)), pageArea: Math.max(1, sum(a.mask)),
  };
}
