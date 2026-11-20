// agreeing_points() of repeats.py: 2-nearest-neighbour Hamming matching
// (WebAssembly SIMD, with a plain JavaScript fallback), the ratio test, one
// use per keypoint, and a RANSAC similarity fit ported from OpenCV's
// estimateAffinePartial2D (same random generator, same iteration rule).

import { HAMMING_WASM } from './hamming-wasm.ts';
import { MAX_ROTATION, MAX_SCALE, MIN_GOOD_MATCHES, MIN_KEYPOINTS, RANSAC_REPROJ_THRESHOLD, RATIO_TEST } from './constants.ts';

/** Keypoints of one page: positions and 32-byte ORB descriptors. */
export interface PageFeatures { xs: Float32Array; ys: Float32Array; desc: Uint8Array }

export const emptyPage = (): PageFeatures => ({ xs: new Float32Array(0), ys: new Float32Array(0), desc: new Uint8Array(0) });

type Knn = (a: Uint8Array, na: number, b: Uint8Array, nb: number) => Int32Array;

let knnImpl: Knn | null = null;

function popcount32(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24);
}

/** The JavaScript fallback. */
export const knnJs: Knn = (a, na, b, nb) => {
  const out = new Int32Array(na * 3);
  const a32 = new Int32Array(a.buffer, a.byteOffset, na * 8);
  const b32 = new Int32Array(b.buffer, b.byteOffset, nb * 8);
  for (let i = 0; i < na; i++) {
    let best = 1000, second = 1000, bestj = -1;
    const o = i * 8;
    for (let j = 0; j < nb; j++) {
      const q = j * 8;
      let d = 0;
      for (let k = 0; k < 8; k++) d += popcount32(a32[o + k] ^ b32[q + k]);
      if (d < best) { second = best; best = d; bestj = j; } else if (d < second) second = d;
    }
    out[3 * i] = bestj; out[3 * i + 1] = best; out[3 * i + 2] = second;
  }
  return out;
};

function makeWasmKnn(): Knn | null {
  try {
    const bytes = Uint8Array.from(atob(HAMMING_WASM), c => c.charCodeAt(0));
    if (!WebAssembly.validate(bytes)) return null;
    const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), {});
    const memory = instance.exports.memory as WebAssembly.Memory;
    const knn2 = instance.exports.knn2 as (a: number, na: number, b: number, nb: number, out: number) => void;
    return (a, na, b, nb) => {
      const need = na * 32 + nb * 32 + na * 12 + 64;
      if (memory.buffer.byteLength < need) memory.grow(Math.ceil((need - memory.buffer.byteLength) / 65536));
      const heap = new Uint8Array(memory.buffer);
      const bOff = na * 32, outOff = bOff + nb * 32;
      heap.set(a.subarray(0, na * 32), 0);
      heap.set(b.subarray(0, nb * 32), bOff);
      knn2(0, na, bOff, nb, outOff);
      return new Int32Array(memory.buffer.slice(outOff, outOff + na * 12));
    };
  } catch {
    return null;
  }
}

export function knn(a: Uint8Array, na: number, b: Uint8Array, nb: number): Int32Array {
  if (!knnImpl) knnImpl = makeWasmKnn() ?? knnJs;
  return knnImpl(a, na, b, nb);
}

export const usingWasm = () => { if (!knnImpl) knnImpl = makeWasmKnn() ?? knnJs; return knnImpl !== knnJs; };

// ---- OpenCV's RNG and RANSAC --------------------------------------------

const TWO32 = 4294967296;
const COEFF = 4164903690;

/** cv::RNG with its multiply-with-carry step, seeded as RANSAC seeds it (uint64)-1. */
class CvRng {
  private lo = 0xffffffff;
  private hi = 0xffffffff;
  next(): number {
    // state = (uint64)(unsigned)state * COEFF + (unsigned)(state >> 32)
    const a0 = this.lo & 0xffff, a1 = this.lo >>> 16;
    const c0 = COEFF & 0xffff, c1 = COEFF >>> 16;
    const p00 = a0 * c0, mid = a0 * c1 + a1 * c0, p11 = a1 * c1;
    let low = p00 + (mid % 65536) * 65536 + this.hi;
    const carry = Math.floor(low / TWO32);
    low -= carry * TWO32;
    const high = (p11 + Math.floor(mid / 65536) + carry) % TWO32;
    this.lo = low;
    this.hi = high;
    return low;
  }
  uniform(a: number, b: number): number {
    return a === b ? a : (this.next() % (b - a)) + a;
  }
}

function updateIters(p: number, ep: number, modelPoints: number, maxIters: number): number {
  p = Math.min(Math.max(p, 0), 1);
  ep = Math.min(Math.max(ep, 0), 1);
  let num = Math.max(1 - p, Number.MIN_VALUE);
  let denom = 1 - Math.pow(1 - ep, modelPoints);
  if (denom < 2.2250738585072014e-308) return 0;
  num = Math.log(num);
  denom = Math.log(denom);
  if (denom >= 0 || -num >= maxIters * -denom) return maxIters;
  const r = num / denom;
  const rr = Math.round(r);
  return Math.abs(r - Math.trunc(r)) === 0.5 && rr % 2 !== 0 ? rr - 1 : rr;
}

/** 2-point similarity: M = [[S0, -S1, S2], [S1, S0, S3]]. */
function kernel(x1: number, y1: number, x2: number, y2: number, X1: number, Y1: number, X2: number, Y2: number): Float64Array {
  const d = 1 / ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
  const S0 = d * ((X1 - X2) * (x1 - x2) + (Y1 - Y2) * (y1 - y2));
  const S1 = d * ((Y1 - Y2) * (x1 - x2) - (X1 - X2) * (y1 - y2));
  const S2 = d * ((Y1 - Y2) * (x1 * y2 - x2 * y1) - (X1 * y2 - X2 * y1) * (y1 - y2) - (X1 * x2 - X2 * x1) * (x1 - x2));
  const S3 = d * (-(X1 - X2) * (x1 * y2 - x2 * y1) - (Y1 * x2 - Y2 * x1) * (x1 - x2) - (Y1 * y2 - Y2 * y1) * (y1 - y2));
  return Float64Array.of(S0, -S1, S2, S1, S0, S3);
}

const f32 = Math.fround;

function countInliers(sx: Float32Array, sy: Float32Array, dx: Float32Array, dy: Float32Array, m: Float64Array, thresh: number, mask: Uint8Array): number {
  const F0 = f32(m[0]), F1 = f32(m[1]), F2 = f32(m[2]), F3 = f32(m[3]), F4 = f32(m[4]), F5 = f32(m[5]);
  const t = f32(thresh * thresh);
  let nz = 0;
  for (let i = 0; i < sx.length; i++) {
    const a = f32(f32(f32(f32(F0 * sx[i]) + f32(F1 * sy[i])) + F2) - dx[i]);
    const b = f32(f32(f32(f32(F3 * sx[i]) + f32(F4 * sy[i])) + F5) - dy[i]);
    const e = f32(f32(a * a) + f32(b * b));
    const ok = e <= t ? 1 : 0;
    mask[i] = ok;
    nz += ok;
  }
  return nz;
}

export interface Fit { model: Float64Array; inliers: number }

/** cv2.estimateAffinePartial2D(src, dst, method=RANSAC, ransacReprojThreshold) with its defaults. */
export function estimateSimilarity(sx: Float32Array, sy: Float32Array, dx: Float32Array, dy: Float32Array,
  threshold = RANSAC_REPROJ_THRESHOLD, maxIters = 2000, confidence = 0.99): Fit | null {
  const count = sx.length, modelPoints = 2;
  if (count < modelPoints) return null;
  const rng = new CvRng();
  let niters = Math.max(maxIters, 1);
  let maxGood = 0;
  let bestModel: Float64Array | null = null;
  let mask = new Uint8Array(count), bestMask = new Uint8Array(count);
  if (count === modelPoints) {
    bestModel = kernel(sx[0], sy[0], sx[1], sy[1], dx[0], dy[0], dx[1], dy[1]);
    bestMask.fill(1);
    maxGood = count;
  } else {
    for (let iter = 0; iter < niters; iter++) {
      const i0 = rng.uniform(0, count);
      let i1 = rng.uniform(0, count);
      while (i1 === i0) i1 = rng.uniform(0, count);
      // checkSubset for two points never rejects
      const model = kernel(sx[i0], sy[i0], sx[i1], sy[i1], dx[i0], dy[i0], dx[i1], dy[i1]);
      const good = countInliers(sx, sy, dx, dy, model, threshold, mask);
      if (good > Math.max(maxGood, modelPoints - 1)) {
        [mask, bestMask] = [bestMask, mask];
        bestModel = model;
        maxGood = good;
        niters = updateIters(confidence, (count - good) / count, modelPoints, niters);
      }
    }
  }
  if (maxGood <= 0 || !bestModel) return null;
  // Refinement (OpenCV runs 10 Levenberg-Marquardt steps on the inliers; the
  // problem is linear, so the least-squares solution is where they go).
  if (count > 2) {
    let n = 0, mx = 0, my = 0, mX = 0, mY = 0;
    for (let i = 0; i < count; i++) if (bestMask[i]) { n++; mx += sx[i]; my += sy[i]; mX += dx[i]; mY += dy[i]; }
    if (n >= 2) {
      mx /= n; my /= n; mX /= n; mY /= n;
      let sxx = 0, sa = 0, sb = 0;
      for (let i = 0; i < count; i++) {
        if (!bestMask[i]) continue;
        const x = sx[i] - mx, y = sy[i] - my, X = dx[i] - mX, Y = dy[i] - mY;
        sxx += x * x + y * y;
        sa += x * X + y * Y;
        sb += x * Y - y * X;
      }
      if (sxx > 0) {
        const a = sa / sxx, b = sb / sxx;
        bestModel = Float64Array.of(a, -b, mX - a * mx + b * my, b, a, mY - b * mx - a * my);
      }
    }
  }
  return { model: bestModel, inliers: maxGood };
}

/** agreeing_points(a, b): matched keypoints of two pages that fit one rigid placement. */
export function agreeingPoints(a: PageFeatures, b: PageFeatures): number {
  const na = a.xs.length, nb = b.xs.length;
  if (na < MIN_KEYPOINTS || nb < MIN_KEYPOINTS) return 0;
  const res = knn(a.desc, na, b.desc, nb);
  // one match per train keypoint; Python's dict keeps the first insertion position
  const order: number[] = [];
  const bestQuery = new Map<number, number>();
  const bestDist = new Map<number, number>();
  for (let i = 0; i < na; i++) {
    const j = res[3 * i], d0 = res[3 * i + 1], d1 = res[3 * i + 2];
    if (j < 0 || nb < 2) continue;
    if (!(d0 < RATIO_TEST * d1)) continue;
    if (!bestQuery.has(j)) { order.push(j); bestQuery.set(j, i); bestDist.set(j, d0); }
    else if (d0 < bestDist.get(j)!) { bestQuery.set(j, i); bestDist.set(j, d0); }
  }
  if (order.length < MIN_GOOD_MATCHES) return 0;
  const n = order.length;
  const sx = new Float32Array(n), sy = new Float32Array(n), dx = new Float32Array(n), dy = new Float32Array(n);
  order.forEach((j, k) => {
    const i = bestQuery.get(j)!;
    sx[k] = a.xs[i]; sy[k] = a.ys[i]; dx[k] = b.xs[j]; dy[k] = b.ys[j];
  });
  const fit = estimateSimilarity(sx, sy, dx, dy);
  if (!fit) return 0;
  const scale = Math.hypot(fit.model[0], fit.model[3]);
  const turn = Math.abs(Math.atan2(fit.model[3], fit.model[0]) * 180 / Math.PI);
  if (!(1 / MAX_SCALE <= scale && scale <= MAX_SCALE) || turn > MAX_ROTATION) return 0;
  return fit.inliers;
}
