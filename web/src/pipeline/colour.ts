// Colour conversions as the Python side does them: OpenCV's 8-bit
// BGR -> Lab (fixed point, sRGB, D65), BGR -> gray, and FFmpeg's default
// YUV -> BGR for an untagged limited-range video.

const LAB_SHIFT = 12;
const GAMMA_SHIFT = 3;
const LAB_SHIFT2 = LAB_SHIFT + GAMMA_SHIFT;
const LAB_CBRT_TAB_SIZE_B = 256 * 3 / 2 * (1 << GAMMA_SHIFT);

const f32 = Math.fround;

function cvRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x - Math.trunc(x)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** The largest float32 not above x (x positive). */
function f32Down(x: number): number {
  const f = f32(x);
  if (f <= x) return f;
  const buf = new Float32Array([f]);
  new Uint32Array(buf.buffer)[0] -= 1;
  return buf[0];
}

const sRGBGammaTab = new Uint16Array(256);
const labCbrtTab = new Uint16Array(LAB_CBRT_TAB_SIZE_B);
const coeffs = new Int32Array(9);
(function init() {
  const f255 = f32(255);
  for (let i = 0; i < 256; i++) {
    const x = f32(i / f255);
    const g = x <= f32(0.04045) ? f32(x / f32(12.92)) : f32(Math.pow(f32(f32(x + f32(0.055)) / f32(1.055)), 2.4));
    sRGBGammaTab[i] = cvRound(f32(f32(f255 * g) * (1 << GAMMA_SHIFT)));
  }
  const lthresh = f32(216 / 24389), lscale = f32(841 / 108), lbias = f32(16 / 116);
  const cbTabScale = f32(1 / (255 * (1 << GAMMA_SHIFT)));
  const lshift2 = 1 << LAB_SHIFT2;
  for (let i = 0; i < LAB_CBRT_TAB_SIZE_B; i++) {
    const x = f32(cbTabScale * i);
    // OpenCV's softfloat cube root comes out below the correctly rounded value
    // where they differ; rounding toward zero reproduces its table.
    const v = x < lthresh ? f32(f32(x * lscale) + lbias) : f32Down(Math.cbrt(x));
    labCbrtTab[i] = cvRound(lshift2 * v);
  }
  const sRGB2XYZ = [0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227];
  const D65 = [0.950456, 1.0, 1.088754];
  const lshift = 1 << LAB_SHIFT;
  // blueIdx = 0 (BGR input): coefficient order is B, G, R
  for (let i = 0; i < 3; i++) {
    coeffs[i * 3 + 2] = cvRound(lshift * sRGB2XYZ[i * 3] / D65[i]);
    coeffs[i * 3 + 1] = cvRound(lshift * sRGB2XYZ[i * 3 + 1] / D65[i]);
    coeffs[i * 3 + 0] = cvRound(lshift * sRGB2XYZ[i * 3 + 2] / D65[i]);
  }
})();

const descale = (x: number, n: number) => (x + (1 << (n - 1))) >> n;
const sat8 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v);
const LSCALE = Math.floor((116 * 255 + 50) / 100);
const LSHIFT = -Math.floor((16 * 255 * (1 << LAB_SHIFT2) + 50) / 100);

/** cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB) for one 8-bit pixel; writes L, a, b into out[o..o+2]. */
export function bgrToLab(b: number, g: number, r: number, out: Uint8Array | Int32Array, o: number): void {
  const B = sRGBGammaTab[b], G = sRGBGammaTab[g], R = sRGBGammaTab[r];
  const fX = labCbrtTab[descale(B * coeffs[0] + G * coeffs[1] + R * coeffs[2], LAB_SHIFT)];
  const fY = labCbrtTab[descale(B * coeffs[3] + G * coeffs[4] + R * coeffs[5], LAB_SHIFT)];
  const fZ = labCbrtTab[descale(B * coeffs[6] + G * coeffs[7] + R * coeffs[8], LAB_SHIFT)];
  out[o] = sat8(descale(LSCALE * fY + LSHIFT, LAB_SHIFT2));
  out[o + 1] = sat8(descale(500 * (fX - fY) + 128 * (1 << LAB_SHIFT2), LAB_SHIFT2));
  out[o + 2] = sat8(descale(200 * (fY - fZ) + 128 * (1 << LAB_SHIFT2), LAB_SHIFT2));
}

/** Whole-image BGR (interleaved) -> Lab (interleaved). */
export function bgrImageToLab(bgr: Uint8Array, out?: Uint8Array): Uint8Array {
  const lab = out ?? new Uint8Array(bgr.length);
  for (let i = 0; i < bgr.length; i += 3) bgrToLab(bgr[i], bgr[i + 1], bgr[i + 2], lab, i);
  return lab;
}

/** cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) for 8-bit. */
export function bgrToGray(bgr: Uint8Array): Uint8Array {
  const n = bgr.length / 3;
  const out = new Uint8Array(n);
  // OpenCV 4 RGB2Gray<uchar>: 15-bit coefficients RY15, GY15, BY15
  for (let i = 0; i < n; i++) out[i] = (bgr[3 * i] * 3735 + bgr[3 * i + 1] * 19235 + bgr[3 * i + 2] * 9798 + 16384) >> 15;
  return out;
}

/** HSV value channel of cv2.cvtColor(..., COLOR_BGR2HSV) for 8-bit: max(B, G, R). */
export function hsvValue(bgr: Uint8Array): Uint8Array {
  const n = bgr.length / 3;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.max(bgr[3 * i], bgr[3 * i + 1], bgr[3 * i + 2]);
  return out;
}

// FFmpeg's yuv420p -> bgr24 for untagged video: BT.601 limited range, each
// chroma sample shared by its 2x2 block of pixels. Measured against FFmpeg 6.1
// on a proxy frame: within 2 levels everywhere, 97% within 1.
const CY = 255 / 219;
const CRV = 104597 / 65536;
const CBU = 132201 / 65536;
const CGU = 25675 / 65536;
const CGV = 53279 / 65536;

/** Planar 4:2:0 (Y, U, V planes) -> interleaved BGR, as FFmpeg's -pix_fmt bgr24 does it. */
export function yuv420ToBgr(y: Uint8Array, u: Uint8Array, v: Uint8Array, w: number, h: number, out?: Uint8Array): Uint8Array {
  const bgr = out ?? new Uint8Array(w * h * 3);
  const cw = w >> 1;
  for (let row = 0; row < h; row++) {
    const crow = (row >> 1) * cw;
    for (let x = 0; x < w; x++) {
      const yy = CY * (y[row * w + x] - 16);
      const cu = u[crow + (x >> 1)] - 128, cv = v[crow + (x >> 1)] - 128;
      const o = (row * w + x) * 3;
      bgr[o] = sat8(Math.floor(yy + CBU * cu));
      bgr[o + 1] = sat8(Math.floor(yy - CGU * cu - CGV * cv));
      bgr[o + 2] = sat8(Math.floor(yy + CRV * cv));
    }
  }
  return bgr;
}

/** yuv420ToBgr for rows [row0, row1) only, written to out from its start. */
export function yuv420RowsToBgr(y: Uint8Array, u: Uint8Array, v: Uint8Array, w: number, row0: number, row1: number, out: Uint8Array): void {
  const cw = w >> 1;
  for (let row = row0; row < row1; row++) {
    const crow = (row >> 1) * cw;
    for (let x = 0; x < w; x++) {
      const yy = CY * (y[row * w + x] - 16);
      const cu = u[crow + (x >> 1)] - 128, cv = v[crow + (x >> 1)] - 128;
      const o = ((row - row0) * w + x) * 3;
      out[o] = sat8(Math.floor(yy + CBU * cu));
      out[o + 1] = sat8(Math.floor(yy - CGU * cu - CGV * cv));
      out[o + 2] = sat8(Math.floor(yy + CRV * cv));
    }
  }
}

/** FFmpeg's limited-range Y -> full-range gray (-pix_fmt gray): round((Y - 16) * 255 / 219). */
export const GRAY_FROM_Y = (() => {
  const t = new Uint8Array(256);
  for (let i = 0; i < 256; i++) t[i] = sat8(Math.round((i - 16) * 255 / 219));
  return t;
})();
