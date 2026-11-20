// The handful of OpenCV and NumPy operations the selection logic uses,
// written to give the same results as the Python side (OpenCV 4.11, NumPy
// 1.26). Binary images are Uint8Array with 0 or 1 per pixel, row-major.
// test/imgproc.test.ts checks each one against fixtures made by OpenCV.

/** OpenCV's cvRound: round half to even. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x - Math.trunc(x)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

const f32 = Math.fround;

/** cv2.resize(..., INTER_AREA) for uint8 when both factors are whole numbers. */
export function resizeAreaInt(src: Uint8Array, sw: number, sh: number, dw: number, dh: number, out?: Uint8Array): Uint8Array {
  const fx = sw / dw, fy = sh / dh;
  if (!Number.isInteger(fx) || !Number.isInteger(fy)) throw new Error('resizeAreaInt needs whole-number factors');
  const dst = out ?? new Uint8Array(dw * dh);
  if (fx === 2 && fy === 2) {
    // OpenCV's SIMD path for 2x2: (a + b + c + d + 2) >> 2
    for (let y = 0; y < dh; y++) {
      const r0 = 2 * y * sw, r1 = r0 + sw, o = y * dw;
      for (let x = 0; x < dw; x++) {
        const s = 2 * x;
        dst[o + x] = (src[r0 + s] + src[r0 + s + 1] + src[r1 + s] + src[r1 + s + 1] + 2) >> 2;
      }
    }
    return dst;
  }
  const scale = f32(1 / (fx * fy));
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      let sum = 0;
      for (let j = 0; j < fy; j++) {
        const row = (y * fy + j) * sw + x * fx;
        for (let i = 0; i < fx; i++) sum += src[row + i];
      }
      const v = roundHalfEven(f32(sum * scale));
      dst[y * dw + x] = v > 255 ? 255 : v;
    }
  }
  return dst;
}

interface AreaTab { di: Int32Array; si: Int32Array; alpha: Float32Array; count: number }

function areaTab(ssize: number, dsize: number, scale: number): AreaTab {
  const di = new Int32Array(ssize * 2 + 2), si = new Int32Array(ssize * 2 + 2), alpha = new Float32Array(ssize * 2 + 2);
  let k = 0;
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale, fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) { di[k] = dx; si[k] = sx1 - 1; alpha[k++] = (sx1 - fsx1) / cellWidth; }
    for (let sx = sx1; sx < sx2; sx++) { di[k] = dx; si[k] = sx; alpha[k++] = 1.0 / cellWidth; }
    if (fsx2 - sx2 > 1e-3) { di[k] = dx; si[k] = sx2; alpha[k++] = Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth; }
  }
  return { di, si, alpha, count: k };
}

/** cv2.resize(..., INTER_AREA) for uint8 with any factor (OpenCV's generic area table). */
export function resizeArea(src: Uint8Array, sw: number, sh: number, dw: number, dh: number): Uint8Array {
  if (Number.isInteger(sw / dw) && Number.isInteger(sh / dh)) return resizeAreaInt(src, sw, sh, dw, dh);
  const xt = areaTab(sw, dw, sw / dw), yt = areaTab(sh, dh, sh / dh);
  const dst = new Uint8Array(dw * dh);
  const buf = new Float32Array(dw), sum = new Float32Array(dw);
  // group the vertical table by destination row
  let k = 0;
  for (let dy = 0; dy < dh; dy++) {
    sum.fill(0);
    while (k < yt.count && yt.di[k] === dy) {
      const sy = yt.si[k], beta = yt.alpha[k];
      buf.fill(0);
      const row = sy * sw;
      for (let j = 0; j < xt.count; j++) buf[xt.di[j]] = f32(buf[xt.di[j]] + f32(src[row + xt.si[j]] * xt.alpha[j]));
      for (let x = 0; x < dw; x++) sum[x] = f32(sum[x] + f32(beta * buf[x]));
      k++;
    }
    for (let x = 0; x < dw; x++) {
      const v = roundHalfEven(sum[x]);
      dst[dy * dw + x] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return dst;
}

/**
 * A plane (offset, stride) mapped through `lut` and shrunk by INTER_AREA to
 * dw x dh in one pass, without an intermediate full-size copy: the gray image
 * the keypoints are found on. Whole-number factors give exactly
 * resizeAreaInt of the mapped plane; other factors sum in double precision
 * where OpenCV sums in single, which can move a pixel by one level.
 */
export function lutAreaFromPlane(src: Uint8Array, offset: number, stride: number, sw: number, sh: number, lut: Uint8Array, dw: number, dh: number): Uint8Array {
  const dst = new Uint8Array(dw * dh);
  if (sw === 2 * dw && sh === 2 * dh) {
    for (let y = 0; y < dh; y++) {
      const r0 = offset + 2 * y * stride, r1 = r0 + stride, o = y * dw;
      for (let x = 0; x < dw; x++) {
        const s = 2 * x;
        dst[o + x] = (lut[src[r0 + s]] + lut[src[r0 + s + 1]] + lut[src[r1 + s]] + lut[src[r1 + s + 1]] + 2) >> 2;
      }
    }
    return dst;
  }
  const xt = areaTab(sw, dw, sw / dw), yt = areaTab(sh, dh, sh / dh);
  const row = new Float64Array(dw), acc = new Float64Array(dw);
  let k = 0;
  for (let dy = 0; dy < dh; dy++) {
    acc.fill(0);
    while (k < yt.count && yt.di[k] === dy) {
      const base = offset + yt.si[k] * stride, beta = yt.alpha[k];
      row.fill(0);
      for (let j = 0; j < xt.count; j++) row[xt.di[j]] += lut[src[base + xt.si[j]]] * xt.alpha[j];
      for (let x = 0; x < dw; x++) acc[x] += beta * row[x];
      k++;
    }
    for (let x = 0; x < dw; x++) {
      const v = roundHalfEven(acc[x]);
      dst[dy * dw + x] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return dst;
}

/** cv2.resize(..., INTER_NEAREST) for uint8. */
export function resizeNearest(src: Uint8Array, sw: number, sh: number, dw: number, dh: number): Uint8Array {
  const dst = new Uint8Array(dw * dh);
  const ifx = sw / dw, ify = sh / dh;
  const xs = new Int32Array(dw);
  for (let x = 0; x < dw; x++) xs[x] = Math.min(Math.floor(x * ifx), sw - 1);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(Math.floor(y * ify), sh - 1);
    const s = sy * sw, o = y * dw;
    for (let x = 0; x < dw; x++) dst[o + x] = src[s + xs[x]];
  }
  return dst;
}

/** BORDER_REFLECT_101 index. */
function reflect101(i: number, n: number): number {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

/** cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S, ksize=1)): |4-neighbour Laplacian|, saturated. */
export function laplacianAbs(src: Uint8Array, w: number, h: number, out?: Uint8Array): Uint8Array {
  const dst = out ?? new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const up = reflect101(y - 1, h) * w, row = y * w, down = reflect101(y + 1, h) * w;
    for (let x = 0; x < w; x++) {
      const l = x === 0 ? (w > 1 ? 1 : 0) : x - 1;
      const r = x === w - 1 ? (w > 1 ? w - 2 : 0) : x + 1;
      let v = src[up + x] + src[down + x] + src[row + l] + src[row + r] - 4 * src[row + x];
      if (v < 0) v = -v;
      dst[row + x] = v > 255 ? 255 : v;
    }
  }
  return dst;
}

/** cv2.blur(src, (3, 3)) for uint8 (normalised box filter, BORDER_REFLECT_101). */
export function boxBlur3(src: Uint8Array, w: number, h: number): Uint8Array {
  const dst = new Uint8Array(w * h);
  const rows = new Int32Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) rows[o + x] = src[o + reflect101(x - 1, w)] + src[o + x] + src[o + reflect101(x + 1, w)];
  }
  const scale = 1 / 9;
  for (let y = 0; y < h; y++) {
    const a = reflect101(y - 1, h) * w, b = y * w, c = reflect101(y + 1, h) * w;
    for (let x = 0; x < w; x++) dst[b + x] = roundHalfEven((rows[a + x] + rows[b + x] + rows[c + x]) * scale);
  }
  return dst;
}

/** cv2.magnitude(Sobel x, Sobel y) with ksize=3 on uint8, float32 output. */
export function sobelMagnitude(src: Uint8Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const ym = reflect101(y - 1, h) * w, y0 = y * w, yp = reflect101(y + 1, h) * w;
    for (let x = 0; x < w; x++) {
      const xm = reflect101(x - 1, w), xp = reflect101(x + 1, w);
      const gx = (src[ym + xp] - src[ym + xm]) + 2 * (src[y0 + xp] - src[y0 + xm]) + (src[yp + xp] - src[yp + xm]);
      const gy = (src[yp + xm] + 2 * src[yp + x] + src[yp + xp]) - (src[ym + xm] + 2 * src[ym + x] + src[ym + xp]);
      out[y0 + x] = f32(Math.sqrt(f32(f32(gx * gx) + f32(gy * gy))));
    }
  }
  return out;
}

// ---- binary morphology ----------------------------------------------------

/** A structuring element as, per kernel row, the [start, end) of its ones. */
export interface Kernel { width: number; height: number; start: Int32Array; end: Int32Array }

export function rectKernel(width: number, height: number): Kernel {
  return { width, height, start: new Int32Array(height), end: new Int32Array(height).fill(width) };
}

/** cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, height)). */
export function ellipseKernel(width: number, height: number): Kernel {
  const r = Math.floor(height / 2), c = Math.floor(width / 2);
  const invR2 = r ? 1 / (r * r) : 0;
  const start = new Int32Array(height), end = new Int32Array(height);
  for (let i = 0; i < height; i++) {
    const dy = i - r;
    let j1 = 0, j2 = 0;
    if (Math.abs(dy) <= r) {
      const dx = roundHalfEven(c * Math.sqrt((r * r - dy * dy) * invR2));
      j1 = Math.max(c - dx, 0);
      j2 = Math.min(c + dx + 1, width);
    }
    start[i] = j1;
    end[i] = j2;
  }
  return { width, height, start, end };
}

function rowPrefix(src: Uint8Array, w: number, h: number): Int32Array {
  const p = new Int32Array((w + 1) * h);
  for (let y = 0; y < h; y++) {
    let s = 0;
    const o = y * (w + 1), r = y * w;
    p[o] = 0;
    for (let x = 0; x < w; x++) { s += src[r + x] ? 1 : 0; p[o + x + 1] = s; }
  }
  return p;
}

/**
 * Binary erode or dilate with OpenCV's default border (outside pixels never
 * erode and never dilate), anchor at the kernel centre.
 */
function morph(src: Uint8Array, w: number, h: number, k: Kernel, erode: boolean): Uint8Array {
  const dst = new Uint8Array(w * h);
  const p = rowPrefix(src, w, h);
  const ax = Math.floor(k.width / 2), ay = Math.floor(k.height / 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let result = erode ? 1 : 0;
      for (let i = 0; i < k.height; i++) {
        if (k.end[i] <= k.start[i]) continue;
        const yy = y + i - ay;
        if (yy < 0 || yy >= h) continue;
        let x0 = x + k.start[i] - ax, x1 = x + k.end[i] - ax; // [x0, x1)
        if (x0 < 0) x0 = 0;
        if (x1 > w) x1 = w;
        if (x1 <= x0) continue;
        const o = yy * (w + 1);
        const ones = p[o + x1] - p[o + x0];
        if (erode) {
          if (ones !== x1 - x0) { result = 0; break; }
        } else if (ones > 0) { result = 1; break; }
      }
      dst[y * w + x] = result;
    }
  }
  return dst;
}

export const erode = (src: Uint8Array, w: number, h: number, k: Kernel) => morph(src, w, h, k, true);
export const dilate = (src: Uint8Array, w: number, h: number, k: Kernel) => morph(src, w, h, k, false);
export const open = (src: Uint8Array, w: number, h: number, k: Kernel) => dilate(erode(src, w, h, k), w, h, k);
export const close = (src: Uint8Array, w: number, h: number, k: Kernel) => erode(dilate(src, w, h, k), w, h, k);

// ---- connected components ------------------------------------------------

export interface Components { count: number; labels: Int32Array; areas: Int32Array }

/**
 * cv2.connectedComponentsWithStats on a binary image: label 0 is the
 * background, labels 1..count-1 the components. Labels are numbered in
 * raster order of each component's first pixel; only areas are used.
 */
export function connectedComponents(src: Uint8Array, w: number, h: number, connectivity: 4 | 8 = 8): Components {
  const labels = new Int32Array(w * h);
  const parent: number[] = [0];
  const find = (a: number): number => {
    while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  const union = (a: number, b: number): number => {
    a = find(a); b = find(b);
    if (a === b) return a;
    if (a < b) { parent[b] = a; return a; }
    parent[a] = b;
    return b;
  };
  let next = 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!src[i]) continue;
      let label = 0;
      const consider = (j: number) => {
        const l = labels[j];
        if (l) label = label ? union(label, l) : find(l);
      };
      if (x > 0) consider(i - 1);
      if (y > 0) {
        consider(i - w);
        if (connectivity === 8) {
          if (x > 0) consider(i - w - 1);
          if (x < w - 1) consider(i - w + 1);
        }
      }
      if (!label) { label = next++; parent.push(label); }
      labels[i] = label;
    }
  }
  const remap = new Int32Array(next);
  let count = 1;
  for (let l = 1; l < next; l++) {
    const root = find(l);
    if (root === l) remap[l] = count++;
  }
  for (let l = 1; l < next; l++) remap[l] = remap[find(l)];
  const areas = new Int32Array(count);
  for (let i = 0; i < labels.length; i++) {
    if (labels[i]) { labels[i] = remap[labels[i]]; areas[labels[i]]++; }
  }
  return { count, labels, areas };
}

// ---- convex hull and polygon fill ----------------------------------------

/** Convex hull of integer points (Andrew's monotone chain), counter-clockwise, no repeated end. */
export function convexHull(xs: ArrayLike<number>, ys: ArrayLike<number>): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i < xs.length; i++) pts.push([xs[i], ys[i]]);
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const unique: [number, number][] = [];
  for (const p of pts) if (!unique.length || unique[unique.length - 1][0] !== p[0] || unique[unique.length - 1][1] !== p[1]) unique.push(p);
  if (unique.length < 3) return unique;
  const cross = (o: [number, number], a: [number, number], b: [number, number]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: [number, number][] = [];
  for (const p of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: [number, number][] = [];
  for (let i = unique.length - 1; i >= 0; i--) {
    const p = unique[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

/** OpenCV's clipLine: the part of a segment inside a w x h image, or null when none is. */
function clipLine(w: number, h: number, x1: number, y1: number, x2: number, y2: number): [number, number, number, number] | null {
  const right = w - 1, bottom = h - 1;
  let c1 = Number(x1 < 0) + Number(x1 > right) * 2 + Number(y1 < 0) * 4 + Number(y1 > bottom) * 8;
  let c2 = Number(x2 < 0) + Number(x2 > right) * 2 + Number(y2 < 0) * 4 + Number(y2 > bottom) * 8;
  if ((c1 & c2) === 0 && (c1 | c2) !== 0) {
    let a: number;
    if (c1 & 12) {
      a = c1 < 8 ? 0 : bottom;
      x1 += Math.trunc((a - y1) * (x2 - x1) / (y2 - y1));
      y1 = a;
      c1 = Number(x1 < 0) + Number(x1 > right) * 2;
    }
    if (c2 & 12) {
      a = c2 < 8 ? 0 : bottom;
      x2 += Math.trunc((a - y2) * (x2 - x1) / (y2 - y1));
      y2 = a;
      c2 = Number(x2 < 0) + Number(x2 > right) * 2;
    }
    if ((c1 & c2) === 0 && (c1 | c2) !== 0) {
      if (c1) {
        a = c1 === 1 ? 0 : right;
        y1 += Math.trunc((a - x1) * (y2 - y1) / (x2 - x1));
        x1 = a;
        c1 = 0;
      }
      if (c2) {
        a = c2 === 1 ? 0 : right;
        y2 += Math.trunc((a - x2) * (y2 - y1) / (x2 - x1));
        x2 = a;
        c2 = 0;
      }
    }
  }
  return (c1 | c2) === 0 ? [x1, y1, x2, y2] : null;
}

/** OpenCV's Line() with an 8-connected LineIterator (left to right); a segment reaching outside the image is clipped first, as OpenCV does. */
function drawLine8(dst: Uint8Array, w: number, x1: number, y1: number, x2: number, y2: number, h = Infinity): void {
  if (x1 < 0 || x2 < 0 || y1 < 0 || y2 < 0 || x1 >= w || x2 >= w || y1 >= h || y2 >= h) {
    const clipped = clipLine(w, h, x1, y1, x2, y2);
    if (!clipped) return;
    [x1, y1, x2, y2] = clipped;
  }
  let dx = x2 - x1, dy = y2 - y1;
  let deltaX = 1, deltaY = 1;
  if (dx < 0) { dx = -dx; dy = -dy; [x1, x2] = [x2, x1]; [y1, y2] = [y2, y1]; }
  if (dy < 0) { dy = -dy; deltaY = -1; }
  const vert = dy > dx;
  if (vert) { [dx, dy] = [dy, dx]; [deltaX, deltaY] = [deltaY, deltaX]; }
  let err = dx - (dy + dy);
  const plusDelta = dx + dx, minusDelta = -(dy + dy);
  let minusShift = deltaX, plusShift = 0, minusStep = 0, plusStep = deltaY;
  if (vert) { [plusStep, plusShift] = [plusShift, plusStep]; [minusStep, minusShift] = [minusShift, minusStep]; }
  let x = x1, y = y1;
  for (let i = 0; i < dx + 1; i++) {
    dst[y * w + x] = 1;
    const mask = err < 0 ? -1 : 0;
    err += minusDelta + (plusDelta & mask);
    x += minusShift + (plusShift & mask);
    y += minusStep + (plusStep & mask);
  }
}

/** cv2.fillConvexPoly(img, points, 1) for integer vertices (shift 0, LINE_8), as OpenCV's FillConvexPoly; vertices may lie outside the image. */
export function fillConvexPoly(dst: Uint8Array, w: number, h: number, v: [number, number][]): void {
  const npts = v.length;
  if (npts === 0) return;
  const XY_SHIFT = 16, XY_ONE = 1 << XY_SHIFT;
  const delta1 = XY_ONE >> 1, delta2 = XY_ONE >> 1;
  let imin = 0, xmin = v[0][0], xmax = v[0][0], ymin = v[0][1], ymax = v[0][1];
  let p0 = v[npts - 1];
  for (let i = 0; i < npts; i++) {
    const p = v[i];
    if (p[1] < ymin) { ymin = p[1]; imin = i; }
    ymax = Math.max(ymax, p[1]);
    xmax = Math.max(xmax, p[0]);
    xmin = Math.min(xmin, p[0]);
    drawLine8(dst, w, p0[0], p0[1], p[0], p[1], h);
    p0 = p;
  }
  if (npts < 3 || xmax < 0 || ymax < 0 || xmin >= w || ymin >= h) return;
  ymax = Math.min(ymax, h - 1);
  // x positions in 16.16 fixed point; JavaScript numbers hold them exactly at these sizes
  const edge = [
    { idx: imin, di: 1, x: -XY_ONE, dx: 0, ye: ymin },
    { idx: imin, di: npts - 1, x: -XY_ONE, dx: 0, ye: ymin },
  ];
  let edges = npts;
  let y = ymin;
  do {
    for (let i = 0; i < 2; i++) {
      if (y >= edge[i].ye) {
        let idx0 = edge[i].idx;
        const di = edge[i].di;
        let idx = idx0 + di;
        if (idx >= npts) idx -= npts;
        for (; edges-- > 0;) {
          const ty = v[idx][1];
          if (ty > y) {
            const xs = v[idx0][0] * XY_ONE, xe = v[idx][0] * XY_ONE;
            edge[i].ye = ty;
            edge[i].dx = Math.trunc(((xe - xs) * 2 + (ty - y)) / (2 * (ty - y)));
            edge[i].x = xs;
            edge[i].idx = idx;
            break;
          }
          idx0 = idx;
          idx += di;
          if (idx >= npts) idx -= npts;
        }
      }
    }
    if (edges < 0) break;
    if (y >= 0) {
      let left = 0, right = 1;
      if (edge[0].x > edge[1].x) { left = 1; right = 0; }
      let xx1 = Math.floor((edge[left].x + delta1) / XY_ONE);
      let xx2 = Math.floor((edge[right].x + delta2) / XY_ONE);
      if (xx2 >= 0 && xx1 < w) {
        if (xx1 < 0) xx1 = 0;
        if (xx2 >= w) xx2 = w - 1;
        for (let x = xx1; x <= xx2; x++) dst[y * w + x] = 1;
      }
    }
    edge[0].x += edge[0].dx;
    edge[1].x += edge[1].dx;
  } while (++y <= ymax);
}

/** cv2.fillConvexPoly for vertices that may lie outside the image (fillConvexPoly clips as OpenCV does). */
export const fillConvexPolyClipped = fillConvexPoly;

// ---- smallest rotated rectangle (cv2.minAreaRect, cv2.boxPoints) ----------

export interface RotatedRect { cx: number; cy: number; width: number; height: number; angle: number }

/**
 * cv2.minAreaRect of points with whole-number coordinates: OpenCV's rotating
 * calipers (imgproc/src/rotcalipers.cpp of 4.11), step for step in float32,
 * over a convex hull. The hull's starting vertex is not OpenCV's, which can
 * matter only when two different rectangles have exactly the same area.
 */
export function minAreaRect(xs: ArrayLike<number>, ys: ArrayLike<number>): RotatedRect {
  const hull = convexHull(xs, ys);
  const n = hull.length;
  if (n === 0) return { cx: 0, cy: 0, width: 0, height: 0, angle: 0 };
  if (n === 1) return { cx: hull[0][0], cy: hull[0][1], width: 0, height: 0, angle: 0 };
  if (n === 2) {
    const dx = hull[1][0] - hull[0][0], dy = hull[1][1] - hull[0][1];
    return {
      cx: f32(f32(hull[0][0] + hull[1][0]) * 0.5), cy: f32(f32(hull[0][1] + hull[1][1]) * 0.5),
      width: f32(Math.sqrt(dx * dx + dy * dy)), height: 0, angle: f32(f32(f32(Math.atan2(dy, dx)) * 180) / Math.PI),
    };
  }
  const px = hull.map(p => f32(p[0])), py = hull.map(p => f32(p[1]));
  const vx = new Float32Array(n), vy = new Float32Array(n), inv = new Float32Array(n);
  let left = 0, bottom = 0, right = 0, top = 0;
  let leftX = px[0], rightX = px[0], topY = py[0], bottomY = py[0];
  let p0x = px[0], p0y = py[0];
  for (let i = 0; i < n; i++) {
    if (p0x < leftX) { leftX = p0x; left = i; }
    if (p0x > rightX) { rightX = p0x; right = i; }
    if (p0y > topY) { topY = p0y; top = i; }
    if (p0y < bottomY) { bottomY = p0y; bottom = i; }
    const j = i + 1 < n ? i + 1 : 0;
    const dx = px[j] - p0x, dy = py[j] - p0y; // in double, as the C++
    vx[i] = dx; vy[i] = dy;
    inv[i] = 1 / Math.sqrt(dx * dx + dy * dy);
    p0x = px[j]; p0y = py[j];
  }
  let orientation = 0;
  {
    let ax = vx[n - 1], ay = vy[n - 1];
    for (let i = 0; i < n; i++) {
      const bx = vx[i], by = vy[i];
      const convexity = ax * by - ay * bx;
      if (convexity !== 0) { orientation = convexity > 0 ? 1 : -1; break; }
      ax = bx; ay = by;
    }
    if (orientation === 0) throw new Error('minAreaRect: degenerate hull');
  }
  let baseA = orientation, baseB = 0;
  const seq = [bottom, right, top, left];
  let minarea = 3.4028234663852886e38;
  let bufLeft = 0, bufA = 0, bufWidth = 0, bufB = 0, bufHeight = 0, bufBottom = 0;
  const rx = new Float32Array(4), ry = new Float32Array(4);
  for (let k = 0; k < n; k++) {
    rx[0] = vx[seq[0]]; ry[0] = vy[seq[0]];
    rx[1] = vy[seq[1]]; ry[1] = -vx[seq[1]];          // rotate90CW
    rx[2] = -vx[seq[2]]; ry[2] = -vy[seq[2]];         // rotate180
    rx[3] = -vy[seq[3]]; ry[3] = vx[seq[3]];          // rotate90CCW
    let main = 0;
    for (let i = 1; i < 4; i++) {
      // firstVecIsRight(rot[i], rot[main]): rotate90CW(rot[i]) . rot[main] < 0, in float
      const tx = ry[i], ty = -rx[i];
      if (f32(f32(tx * rx[main]) + f32(ty * ry[main])) < 0) main = i;
    }
    const pindex = seq[main];
    const leadX = f32(vx[pindex] * inv[pindex]), leadY = f32(vy[pindex] * inv[pindex]);
    switch (main) {
      case 0: baseA = leadX; baseB = leadY; break;
      case 1: baseA = leadY; baseB = -leadX; break;
      case 2: baseA = -leadX; baseB = -leadY; break;
      default: baseA = -leadY; baseB = leadX; break;
    }
    seq[main] = seq[main] + 1 === n ? 0 : seq[main] + 1;
    let dx = f32(px[seq[1]] - px[seq[3]]), dy = f32(py[seq[1]] - py[seq[3]]);
    const width = f32(f32(dx * baseA) + f32(dy * baseB));
    dx = f32(px[seq[2]] - px[seq[0]]); dy = f32(py[seq[2]] - py[seq[0]]);
    const height = f32(f32(-dx * baseB) + f32(dy * baseA));
    const area = f32(width * height);
    if (area <= minarea) {
      minarea = area;
      bufLeft = seq[3]; bufA = baseA; bufWidth = width; bufB = baseB; bufHeight = height; bufBottom = seq[0];
    }
  }
  const A1 = bufA, B1 = bufB, A2 = -bufB, B2 = bufA;
  const C1 = f32(f32(A1 * px[bufLeft]) + f32(py[bufLeft] * B1));
  const C2 = f32(f32(A2 * px[bufBottom]) + f32(py[bufBottom] * B2));
  const idet = f32(1 / f32(f32(A1 * B2) - f32(A2 * B1)));
  const cornerX = f32(f32(f32(C1 * B2) - f32(C2 * B1)) * idet);
  const cornerY = f32(f32(f32(A1 * C2) - f32(A2 * C1)) * idet);
  const v1x = f32(A1 * bufWidth), v1y = f32(B1 * bufWidth);
  const v2x = f32(A2 * bufHeight), v2y = f32(B2 * bufHeight);
  const angle = f32(Math.atan2(v1y, v1x));
  return {
    cx: f32(cornerX + f32(f32(v1x + v2x) * 0.5)),
    cy: f32(cornerY + f32(f32(v1y + v2y) * 0.5)),
    width: f32(Math.sqrt(v1x * v1x + v1y * v1y)),
    height: f32(Math.sqrt(v2x * v2x + v2y * v2y)),
    angle: f32(f32(angle * 180) / Math.PI),
  };
}

/** cv2.boxPoints: the four corners, as RotatedRect::points computes them in float32. */
export function boxPoints(r: RotatedRect): [number, number][] {
  const radians = r.angle * Math.PI / 180;
  const b = f32(f32(Math.cos(radians)) * 0.5), a = f32(f32(Math.sin(radians)) * 0.5);
  const p0x = f32(f32(r.cx - f32(a * r.height)) - f32(b * r.width));
  const p0y = f32(f32(r.cy + f32(b * r.height)) - f32(a * r.width));
  const p1x = f32(f32(r.cx + f32(a * r.height)) - f32(b * r.width));
  const p1y = f32(f32(r.cy - f32(b * r.height)) - f32(a * r.width));
  return [[p0x, p0y], [p1x, p1y], [f32(f32(2 * r.cx) - p0x), f32(f32(2 * r.cy) - p0y)], [f32(f32(2 * r.cx) - p1x), f32(f32(2 * r.cy) - p1y)]];
}

// ---- Gaussian blur of a float64 image ------------------------------------

/** cv2.GaussianBlur(img, (0, 0), sigma) for float64 (kernel size from sigma, BORDER_REFLECT_101). */
export function gaussianBlur64(src: Float64Array, w: number, h: number, sigma: number): Float64Array {
  const ksize = (Math.round(sigma * 4 * 2 + 1) | 1);
  const half = (ksize - 1) / 2;
  const kernel = new Float64Array(ksize);
  let total = 0;
  const scale2X = -0.5 / (sigma * sigma);
  for (let i = 0; i < ksize; i++) { const x = i - half; kernel[i] = Math.exp(scale2X * x * x); total += kernel[i]; }
  for (let i = 0; i < ksize; i++) kernel[i] /= total;
  const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = 0; k < ksize; k++) s += kernel[k] * src[y * w + reflect101(x + k - half, w)];
      tmp[y * w + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = 0; k < ksize; k++) s += kernel[k] * tmp[reflect101(y + k - half, h) * w + x];
      out[y * w + x] = s;
    }
  }
  return out;
}

// ---- NumPy statistics ----------------------------------------------------

/** np.percentile(values, q) with the default linear method. */
export function percentile(values: ArrayLike<number>, q: number): number {
  const n = values.length;
  if (n === 0) return NaN;
  const sorted = Float64Array.from(values).sort();
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos), hi = Math.min(lo + 1, n - 1);
  const t = pos - lo;
  // NumPy's _lerp: a + (b - a) * t, with the other form when t >= 0.5
  const a = sorted[lo], b = sorted[hi];
  const diff = b - a;
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

/** np.median of numbers (mean of the two middle values for an even count). */
export function median(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return NaN;
  const sorted = Float64Array.from(values).sort();
  return n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

/** Median of small non-negative integers (0..255) by counting. */
export function medianU8(counts: Uint32Array, n: number): number {
  // k-th order statistics for k = (n-1)/2 and n/2 (0-based)
  const k1 = Math.floor((n - 1) / 2), k2 = Math.floor(n / 2);
  let seen = 0, a = -1, b = -1;
  for (let v = 0; v < counts.length; v++) {
    seen += counts[v];
    if (a < 0 && seen > k1) a = v;
    if (seen > k2) { b = v; break; }
  }
  return (a + b) / 2;
}
