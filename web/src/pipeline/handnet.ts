// Port of auto_camera/handnet.py (Detections.mask, palms, fine_detail): which
// patches of skin are hands, from what the trained models found in a frame.

import { HANDNET_MIN_SCORE, MIN_SKIN, PROXY_HEIGHT, PROXY_WIDTH, SKIN_OPEN } from './constants.ts';
import { connectedComponents, convexHull, ellipseKernel, fillConvexPoly, open, roundHalfEven } from './imgproc.ts';

export const PALM = [0, 1, 5, 9, 13, 17];   // handnet.PALM: the wrist and the knuckles

const W = PROXY_WIDTH, H = PROXY_HEIGHT;
const skinKernel = ellipseKernel(SKIN_OPEN, SKIN_OPEN);

/**
 * A number rounded to the nearest float16, as NumPy's astype(np.float16)
 * (round to nearest, ties to even). Python keeps the landmarks and scores as
 * float16 (handdetect.save), so they are rounded the same way here.
 */
export function f16round(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  if (a >= 65520) return sign * Infinity;
  // exponent of a, clamped to float16's normal range; below it the spacing is that of the subnormals
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  else if (2 ** (e + 1) <= a) e++;
  const step = 2 ** (Math.max(e, -14) - 10);
  return sign * roundHalfEven(a / step) * step;
}

/**
 * fine_detail: per pixel, the weaker of the across and down fine detail,
 * each the 5x5 mean of absolute neighbour differences (cv2.blur, border
 * reflected without the edge). Done in whole numbers: a 5x5 mean of whole
 * numbers is k/25, which never lies halfway between two, so rounding it
 * gives what OpenCV's float32 blur and NumPy's round give.
 */
export function fineDetail(gray: Uint8Array, w = W, h = H, out = new Uint8Array(w * h)): Uint8Array {
  const across = new Int32Array(w * h), down = new Int32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x > 0) across[i] = Math.abs(gray[i] - gray[i - 1]);
      if (y > 0) down[i] = Math.abs(gray[i] - gray[i - w]);
    }
  }
  const sa = boxSum5(across, w, h), sd = boxSum5(down, w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.round(Math.min(sa[i], sd[i]) / 25);
    out[i] = v > 255 ? 255 : v;
  }
  return out;
}

const reflect101 = (p: number, n: number) => (p < 0 ? -p : p >= n ? 2 * n - 2 - p : p);

/** 5x5 sums with BORDER_REFLECT_101. */
function boxSum5(src: Int32Array, w: number, h: number): Int32Array {
  const rows = new Int32Array(w * h), out = new Int32Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) s += src[o + reflect101(x + k, w)];
      rows[o + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) s += rows[reflect101(y + k, h) * w + x];
      out[y * w + x] = s;
    }
  }
  return out;
}

/** The patches of skin of one frame (Detections.mask up to the palms). */
export interface SkinPatches {
  labels: Int32Array;
  keep: Uint8Array;      // per label: kept as a hand so far (reaches the edge of the picture)
  areas: Int32Array;
  inner: boolean;        // some patch large enough to count does not reach the edge: only a palm can make it a hand
  any: boolean;
}

export function skinPatches(skin: Uint8Array): SkinPatches | null {
  let any = false;
  for (let i = 0; i < skin.length; i++) if (skin[i]) { any = true; break; }
  if (!any) return null;
  const opened = open(skin, W, H, skinKernel);
  const cc = connectedComponents(opened, W, H, 8);
  const keep = new Uint8Array(cc.count);
  const mark = (i: number) => { keep[cc.labels[i]] = 1; };
  for (let x = 0; x < W; x++) { mark(x); mark((H - 1) * W + x); }
  for (let y = 0; y < H; y++) { mark(y * W); mark(y * W + W - 1); }
  keep[0] = 0;
  let inner = false;
  for (let l = 1; l < cc.count; l++) if (!keep[l] && cc.areas[l] >= MIN_SKIN) inner = true;
  return { labels: cc.labels, keep, areas: cc.areas, inner, any: cc.count > 1 };
}

/** palms: the palms of the hands the landmarker found, filled. Landmarks and scores as float16, as Python keeps them. */
export function palms(landmarks: Float32Array, scores: Float32Array): Uint8Array | null {
  let out: Uint8Array | null = null;
  for (let h = 0; h < scores.length; h++) {
    if (f16round(scores[h]) < HANDNET_MIN_SCORE) continue;
    const base = h * 21 * 2;
    let finite = true;
    for (let k = 0; k < 42; k++) if (!Number.isFinite(landmarks[base + k])) { finite = false; break; }
    if (!finite) continue;
    // (points[PALM] * [width, height]) in float32, rounded half to even
    const xs = PALM.map(p => roundHalfEven(Math.fround(f16round(landmarks[base + 2 * p]) * W)));
    const ys = PALM.map(p => roundHalfEven(Math.fround(f16round(landmarks[base + 2 * p + 1]) * H)));
    out ??= new Uint8Array(W * H);
    const hull = convexHull(xs, ys);
    fillConvexPoly(out, W, H, hull);
  }
  return out;
}

/** Detections.mask: skin in patches that reach the edge of the picture or hold a palm, at least MIN_SKIN pixels. */
export function handMask(patches: SkinPatches | null, palmMask: Uint8Array | null, out = new Uint8Array(W * H)): Uint8Array {
  out.fill(0);
  if (!patches || !patches.any) return out;
  const keep = patches.keep.slice();
  if (palmMask) for (let i = 0; i < W * H; i++) if (palmMask[i]) keep[patches.labels[i]] = 1;
  keep[0] = 0;
  for (let l = 1; l < keep.length; l++) if (patches.areas[l] < MIN_SKIN) keep[l] = 0;
  for (let i = 0; i < W * H; i++) out[i] = keep[patches.labels[i]];
  return out;
}
