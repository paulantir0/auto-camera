// Port of auto_camera/ocr.py: read the digits printed in the head and foot
// bands of a still. Coordinates follow NumPy 2's float32 arithmetic where
// ocr.py works on the detector's float32 corner points.

import { BAND_IN, BAND_OUT, END_ASPECT, MAX_ASPECT, MIN_HEIGHT, MIN_SCORE } from '../pipeline/constants.ts';
import { pyRound } from '../pipeline/pyround.ts';
import type { Image, OcrEngine } from './engine.ts';

export interface Token {
  band: 'top' | 'bottom';
  x: number; // centre, as a share of the band's width, 3 decimals
  text: string;
  score: number; // 2 decimals
  line_end?: 'left' | 'right';
}

const NUMBER = /^\d{1,4}$/;
const LEADING = /^(\d{1,4})(?=\D)/;
const TRAILING = /(?<=\D)(\d{1,4})$/;
const f32 = Math.fround;

/** A rectangle of an image, clipped to it (NumPy slicing). */
export function crop(img: Image, y0: number, y1: number, x0: number, x1: number): Image {
  const ya = Math.max(0, Math.min(img.height, y0)), yb = Math.max(ya, Math.min(img.height, y1));
  const xa = Math.max(0, Math.min(img.width, x0)), xb = Math.max(xa, Math.min(img.width, x1));
  const w = xb - xa, h = yb - ya;
  const data = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) data.set(img.data.subarray(((ya + y) * img.width + xa) * 3, ((ya + y) * img.width + xb) * 3), y * w * 3);
  return { data, width: w, height: h };
}

/** ocr.bands: where the strips lie, in pixels of a frame of this size: [y0, y1, x0, x1] per band. */
export function bandRects(width: number, height: number, box: number[], rows: number[] | null): Record<'top' | 'bottom', [number, number, number, number]> {
  const [x0, x1, y0, y1] = box;
  const xa = Math.trunc(Math.max(0, x0 - 0.02) * width), xb = Math.trunc(Math.min(1, x1 + 0.02) * width);
  const pageH = y1 - y0;
  let top: [number, number], bottom: [number, number];
  if (rows === null) {
    top = [y0 - 0.03, y0 + 0.17 * pageH];
    bottom = [y1 - 0.14 * pageH, y1 + 0.03];
  } else {
    const [tf, tt, bf, bt] = rows.length === 4 ? rows : [rows[0], rows[0], rows[1], rows[1]];
    top = [tf - BAND_OUT * pageH, tt + BAND_IN * pageH];
    bottom = [bf - BAND_IN * pageH, bt + BAND_OUT * pageH];
  }
  const span = ([a, b]: [number, number]): [number, number, number, number] =>
    [Math.trunc(Math.max(0, a) * height), Math.trunc(Math.min(1, b) * height), xa, xb];
  return { top: span(top), bottom: span(bottom) };
}

/** Where the pixels of a still come from: its size, and any rectangle of it (y0, y1, x0, x1) as BGR. */
export interface Still {
  width: number;
  height: number;
  region(y0: number, y1: number, x0: number, x1: number): Promise<Image> | Image;
}

/** A still from an image already decoded in full. */
export const wholeImage = (image: Image): Still => ({ width: image.width, height: image.height, region: (y0, y1, x0, x1) => crop(image, y0, y1, x0, x1) });

/** ocr.read_still: every run of one to four digits found in the bands of one still. */
export async function readStill(engine: OcrEngine, still: Still, box: number[], rows: number[] | null): Promise<Token[]> {
  const tokens: Token[] = [];
  const rects = bandRects(still.width, still.height, box, rows);
  for (const name of ['top', 'bottom'] as const) {
    const [ya, yb, xa, xb] = rects[name];
    if (yb <= ya || xb <= xa) continue;
    const band = await still.region(ya, yb, xa, xb);
    if (!band.width || !band.height) continue;
    const found = await engine.detect(band);
    const lines: [number, number, number, number][] = [];
    for (const points of found) {
      const xs = points.map(p => f32(p[0])), ys = points.map(p => f32(p[1]));
      const x0 = Math.min(...xs), y0 = Math.min(...ys), x1 = Math.max(...xs), y1 = Math.max(...ys);
      const width = f32(x1 - x0), height = f32(y1 - y0);
      if (height < MIN_HEIGHT) continue;
      // lines are kept as Python floats (float64) from here on
      if (width > f32(MAX_ASPECT * height)) { lines.push([x0, y0, x1, y1]); continue; }
      const pad = Math.trunc(f32(0.15 * height)) + 2;
      const [text, score] = await engine.recognise(crop(band, Math.max(0, Math.trunc(y0) - pad), Math.trunc(y1) + pad, Math.max(0, Math.trunc(x0) - pad), Math.trunc(x1) + pad));
      if (NUMBER.test(text) && score >= MIN_SCORE) {
        tokens.push({ band: name, x: pyRound(f32(f32(f32(x0 + x1) / 2) / band.width), 3), text, score: pyRound(score, 2) });
      }
    }
    // the outermost line of each page: the head line in the top band, the foot line in the bottom one
    const middle = band.width / 2;
    const sides = [(l: number[]) => (l[0] + l[2]) / 2 < middle, (l: number[]) => (l[0] + l[2]) / 2 >= middle];
    for (const side of sides) {
      const pageLines = lines.filter(side);
      if (!pageLines.length) continue;
      // min/max keep the first of equals, as Python's do
      let pick = pageLines[0];
      for (const l of pageLines) if (name === 'top' ? l[1] < pick[1] : l[3] > pick[3]) pick = l;
      const [x0, y0, x1, y1] = pick;
      const height = y1 - y0;
      const pad = Math.trunc(0.15 * height) + 2;
      const reach = END_ASPECT * height;
      const ends: ['left' | 'right', number, number, RegExp][] = [
        ['left', x0, Math.min(x1, x0 + reach), LEADING],
        ['right', Math.max(x0, x1 - reach), x1, TRAILING],
      ];
      for (const [end, ea, eb, pattern] of ends) {
        const [text, score] = await engine.recognise(crop(band, Math.max(0, Math.trunc(y0) - pad), Math.trunc(y1) + pad, Math.max(0, Math.trunc(ea) - pad), Math.trunc(eb) + pad));
        const match = pattern.exec(text);
        if (match && score >= MIN_SCORE) {
          const centre = end === 'left' ? ea + 0.5 * height : eb - 0.5 * height;
          tokens.push({ band: name, x: pyRound(centre / band.width, 3), text: match[1], score: pyRound(score, 2), line_end: end });
        }
      }
    }
  }
  return tokens;
}
