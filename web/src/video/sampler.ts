// Shrinking a decoded frame to the 480x270 analysis size the way the Python
// side's proxies were made: FFmpeg's scale_cuda with its default
// interpolation, which is bicubic with A = 0 on the two nearest source
// samples per axis (weights 1 - s, s with s = 3f^2 - 2f^3), no prefilter, and
// the result truncated. Measured against scale_cuda output on Einar frames:
// within one level everywhere, 95% identical (study in web/README.md).
//
// Y and the two chroma planes are shrunk separately, as scale_cuda does.

import { GRAY_FROM_Y } from '../pipeline/colour.ts';
import { PROXY_HEIGHT, PROXY_WIDTH } from '../pipeline/constants.ts';

interface Axis { i0: Int32Array; i1: Int32Array; w0: Float64Array; w1: Float64Array }

function axis(src: number, dst: number): Axis {
  const i0 = new Int32Array(dst), i1 = new Int32Array(dst), w0 = new Float64Array(dst), w1 = new Float64Array(dst);
  for (let o = 0; o < dst; o++) {
    const x = (o + 0.5) * src / dst - 0.5;
    const x0 = Math.floor(x);
    const f = x - x0;
    const s = 3 * f * f - 2 * f * f * f;
    i0[o] = Math.min(Math.max(x0, 0), src - 1);
    i1[o] = Math.min(Math.max(x0 + 1, 0), src - 1);
    w0[o] = 1 - s;
    w1[o] = s;
  }
  return { i0, i1, w0, w1 };
}

/** One plane: src (stride, width, height, element step for interleaved chroma) -> dst (dw x dh). */
function shrinkPlane(src: Uint8Array, offset: number, stride: number, step: number, ax: Axis, ay: Axis, dst: Uint8Array, dw: number, dh: number): void {
  for (let y = 0; y < dh; y++) {
    const r0 = offset + ay.i0[y] * stride, r1 = offset + ay.i1[y] * stride;
    const wy0 = ay.w0[y], wy1 = ay.w1[y];
    const o = y * dw;
    for (let x = 0; x < dw; x++) {
      const c0 = ax.i0[x] * step, c1 = ax.i1[x] * step;
      const wx0 = ax.w0[x], wx1 = ax.w1[x];
      const v = wy0 * (wx0 * src[r0 + c0] + wx1 * src[r0 + c1]) + wy1 * (wx0 * src[r1 + c0] + wx1 * src[r1 + c1]);
      dst[o + x] = v >= 255 ? 255 : v <= 0 ? 0 : Math.floor(v + 1e-9);
    }
  }
}

export interface PlaneLayout { offset: number; stride: number }

/** Shrinks frames of one size; holds the tap tables. */
export class Shrinker {
  readonly width: number;
  readonly height: number;
  private readonly lx: Axis;
  private readonly ly: Axis;
  private readonly cx: Axis;
  private readonly cy: Axis;
  /** 480x270 luma, 240x135 chroma: an I420 frame of the proxy size, in one buffer. */
  readonly i420 = new Uint8Array(PROXY_WIDTH * PROXY_HEIGHT * 3 / 2);
  readonly gray = new Uint8Array(PROXY_WIDTH * PROXY_HEIGHT);

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.lx = axis(width, PROXY_WIDTH);
    this.ly = axis(height, PROXY_HEIGHT);
    this.cx = axis(width >> 1, PROXY_WIDTH >> 1);
    this.cy = axis(height >> 1, PROXY_HEIGHT >> 1);
  }

  get y(): Uint8Array { return this.i420.subarray(0, PROXY_WIDTH * PROXY_HEIGHT); }
  get u(): Uint8Array { const n = PROXY_WIDTH * PROXY_HEIGHT; return this.i420.subarray(n, n + n / 4); }
  get v(): Uint8Array { const n = PROXY_WIDTH * PROXY_HEIGHT; return this.i420.subarray(n + n / 4, n + n / 2); }

  /** From a copied frame: format 'I420' (planes Y, U, V) or 'NV12' (planes Y, UV). */
  shrink(buf: Uint8Array, format: string, layout: PlaneLayout[], withChroma = true): void {
    const W = PROXY_WIDTH, H = PROXY_HEIGHT;
    shrinkPlane(buf, layout[0].offset, layout[0].stride, 1, this.lx, this.ly, this.y, W, H);
    if (withChroma) {
      if (format === 'NV12') {
        shrinkPlane(buf, layout[1].offset, layout[1].stride, 2, this.cx, this.cy, this.u, W / 2, H / 2);
        shrinkPlane(buf, layout[1].offset + 1, layout[1].stride, 2, this.cx, this.cy, this.v, W / 2, H / 2);
      } else {
        shrinkPlane(buf, layout[1].offset, layout[1].stride, 1, this.cx, this.cy, this.u, W / 2, H / 2);
        shrinkPlane(buf, layout[2].offset, layout[2].stride, 1, this.cx, this.cy, this.v, W / 2, H / 2);
      }
    }
    const y = this.y, g = this.gray;
    for (let i = 0; i < g.length; i++) g[i] = GRAY_FROM_Y[y[i]];
  }
}

/** Formats this page can read pixels from directly. */
export const READABLE_FORMATS = new Set(['I420', 'I420A', 'NV12']);

/** Copy a VideoFrame's planes into `buf` (grown as needed). */
export async function copyFrame(frame: VideoFrame, buf: { data: Uint8Array }): Promise<PlaneLayout[]> {
  const size = frame.allocationSize();
  if (buf.data.length < size) buf.data = new Uint8Array(size);
  const layout = await frame.copyTo(buf.data);
  return layout.map(l => ({ offset: l.offset, stride: l.stride }));
}

let rgbaCanvas: OffscreenCanvas | null = null;

/**
 * A frame's planes as I420 or NV12, whatever form the browser hands it out
 * in. Frames in other forms (an opaque GPU frame, RGB, 10-bit) are read as
 * RGBA and converted with the frame's colour matrix (BT.709 unless it says
 * BT.601); that is slower, and close to but not exactly the decoder's values.
 */
export async function readFrame(frame: VideoFrame, buf: { data: Uint8Array }): Promise<{ format: 'I420' | 'NV12'; layout: PlaneLayout[] }> {
  if (READABLE_FORMATS.has(frame.format ?? '')) {
    return { format: frame.format === 'NV12' ? 'NV12' : 'I420', layout: await copyFrame(frame, buf) };
  }
  const w = frame.visibleRect?.width ?? frame.codedWidth, h = frame.visibleRect?.height ?? frame.codedHeight;
  let rgba: Uint8Array | Uint8ClampedArray | null = null;
  try {
    const out = new Uint8Array(w * h * 4);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await frame.copyTo(out, { format: 'RGBA' } as any);
    rgba = out;
  } catch {
    if (!rgbaCanvas || rgbaCanvas.width !== w || rgbaCanvas.height !== h) rgbaCanvas = new OffscreenCanvas(w, h);
    const ctx = rgbaCanvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(frame, 0, 0);
    rgba = ctx.getImageData(0, 0, w, h).data;
  }
  const bt601 = frame.colorSpace?.matrix === 'smpte170m' || frame.colorSpace?.matrix === 'bt470bg';
  const kr = bt601 ? 0.299 : 0.2126, kb = bt601 ? 0.114 : 0.0722, kg = 1 - kr - kb;
  const need = w * h * 3 / 2;
  if (buf.data.length < need) buf.data = new Uint8Array(need);
  const d = buf.data, cw = w >> 1, ch = h >> 1, uOff = w * h, vOff = uOff + cw * ch;
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    d[i] = Math.round(16 + 219 / 255 * (kr * rgba[p] + kg * rgba[p + 1] + kb * rgba[p + 2]));
  }
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      let r = 0, g = 0, b = 0;
      for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
        const p = ((2 * y + dy) * w + 2 * x + dx) * 4;
        r += rgba[p]; g += rgba[p + 1]; b += rgba[p + 2];
      }
      r /= 4; g /= 4; b /= 4;
      const yy = kr * r + kg * g + kb * b;
      d[uOff + y * cw + x] = Math.round(128 + 224 / 255 * (b - yy) / (2 * (1 - kb)));
      d[vOff + y * cw + x] = Math.round(128 + 224 / 255 * (r - yy) / (2 * (1 - kr)));
    }
  }
  return { format: 'I420', layout: [{ offset: 0, stride: w }, { offset: uOff, stride: cw }, { offset: vOff, stride: cw }] };
}
