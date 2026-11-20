// Port of auto_camera/hands.py: a per-video model of the colours that reach
// into the picture, and per-frame shares of page and print covered by them.

import { BINS, CHROMA_CHANGE, FOREGROUND, LIKELIHOOD_RATIO, LOW, MIN_BLOB, MIN_LIGHT, MIN_SAMPLES, PAPER_CHROMA_TOLERANCE, PAPER_VALUE_SHARE, PROXY_HEIGHT, PROXY_WIDTH, SPAN, GRID_H, GRID_W } from './constants.ts';
import { bgrImageToLab, bgrToLab } from './colour.ts';
import { dilate, gaussianBlur64, rectKernel, resizeNearest } from './imgproc.ts';
import { PIXELS, paperColour, printCells, type Analysis, type PaperColour } from './analysis.ts';

const W = PROXY_WIDTH, H = PROXY_HEIGHT;

export const binOf = (v: number) => {
  const b = Math.floor((v - LOW) * BINS / SPAN);
  return b < 0 ? 0 : b > BINS - 1 ? BINS - 1 : b;
};

export interface HandModel { lut: Uint8Array; samples: number; usable: boolean }

/**
 * build_model, fed one sampled proxy frame at a time.
 *
 * Plain (no `refined`): anything outside the page that differs from the
 * median frame is an intruder, and the page's colours are all that is ever
 * seen inside `inside` (the page region). With `refined` the stricter model
 * for the print map: an intruder also changes the colour of the background,
 * the page's colours leave out what is not the median frame's colour, and
 * every spot showing print counts as page, whatever its colour.
 */
export class ModelBuilder {
  readonly intruder = new Float64Array(BINS * BINS);   // what counts as intruder: all change (plain), coloured change (refined)
  readonly anyChange = new Float64Array(BINS * BINS);  // all change, which a refined model falls back on
  readonly page = new Float64Array(BINS * BINS);
  private readonly outside: Uint8Array;
  private readonly lab = new Uint8Array(PIXELS * 3);
  private readonly median: Uint8Array;
  private readonly medianLab: Uint8Array;
  private readonly inside: Uint8Array;
  private readonly refined: { printLevel: number } | null;

  constructor(median: Uint8Array, mask: Uint8Array, inside: Uint8Array, refined: { printLevel: number } | null = null) {
    this.median = median;
    this.inside = inside;
    this.refined = refined;
    this.medianLab = bgrImageToLab(median);
    const grown = dilate(mask, W, H, rectKernel(25, 25));
    this.outside = new Uint8Array(PIXELS);
    for (let i = 0; i < PIXELS; i++) this.outside[i] = grown[i] ? 0 : 1;
  }

  /** One sampled frame; `detail` (its fine detail) is needed only by a refined model with a print level. */
  add(frameBgr: Uint8Array, detail?: Uint8Array): void {
    const lab = this.lab, median = this.median, ml = this.medianLab, refined = this.refined;
    for (let i = 0; i < PIXELS; i++) {
      const o = 3 * i;
      bgrToLab(frameBgr[o], frameBgr[o + 1], frameBgr[o + 2], lab, o);
      const bin = binOf(lab[o + 1]) * BINS + binOf(lab[o + 2]);
      const moved = Math.max(Math.abs(frameBgr[o] - median[o]), Math.abs(frameBgr[o + 1] - median[o + 1]), Math.abs(frameBgr[o + 2] - median[o + 2])) > FOREGROUND;
      const chosen = moved && this.outside[i] === 1 && lab[o] > MIN_LIGHT;
      if (!refined) {
        if (chosen) { this.intruder[bin]++; this.anyChange[bin]++; }
        if (this.inside[i]) this.page[bin]++;
        continue;
      }
      if (chosen) this.anyChange[bin]++;
      const da = lab[o + 1] - ml[o + 1], db = lab[o + 2] - ml[o + 2];
      const coloured = da * da + db * db > CHROMA_CHANGE * CHROMA_CHANGE;
      if (chosen && coloured) this.intruder[bin]++;
      if (!this.inside[i]) continue;
      let own = !(moved && coloured);
      if (!own && refined.printLevel > 0 && detail && detail[i] >= refined.printLevel) own = true;
      if (own) this.page[bin]++;
    }
  }

  /** Add histograms counted elsewhere (another worker's share of the sample frames). */
  merge(intruder: Float64Array, page: Float64Array, anyChange?: Float64Array): void {
    for (let i = 0; i < BINS * BINS; i++) {
      this.intruder[i] += intruder[i]; this.page[i] += page[i];
      this.anyChange[i] += anyChange ? anyChange[i] : intruder[i];
    }
  }

  finish(): HandModel {
    return finishModel(this.intruder, this.page, this.refined ? this.anyChange : undefined);
  }
}

/** The model from the histograms; a refined model whose coloured change is too little falls back on all change (`anyChange`). */
export function finishModel(intruderCounts: Float64Array, pageCounts: Float64Array, anyChange?: Float64Array): HandModel {
  let samples = 0, pageTotal = 0;
  for (let i = 0; i < BINS * BINS; i++) { samples += intruderCounts[i]; pageTotal += pageCounts[i]; }
  if (anyChange && samples < MIN_SAMPLES) {
    intruderCounts = anyChange;
    samples = 0;
    for (let i = 0; i < BINS * BINS; i++) samples += intruderCounts[i];
  }
  const lut = new Uint8Array(BINS * BINS);
  if (samples < MIN_SAMPLES || pageTotal === 0) return { lut, samples, usable: false };
  const intruder = gaussianBlur64(intruderCounts.map(v => v / samples), BINS, BINS, 1.0);
  const page = gaussianBlur64(pageCounts.map(v => v / pageTotal), BINS, BINS, 1.0);
  for (let i = 0; i < BINS * BINS; i++) lut[i] = intruder[i] > LIKELIHOOD_RATIO * page[i] && intruder[i] > 1e-5 ? 1 : 0;
  return { lut, samples, usable: true };
}

/** print_area: where print usually is on this book's pages, in proxy pixels. */
export function printArea(a: Analysis): Uint8Array {
  const full = resizeNearest(printCells(a), GRID_W, GRID_H, W, H);
  for (let i = 0; i < PIXELS; i++) full[i] = full[i] && a.inner[i] ? 1 : 0;
  return full;
}

export interface HandSetup {
  model: HandModel;
  region: Uint8Array; // analysis.inner
  text: Uint8Array; // print_area
  colour: PaperColour | null;
}

export function handSetup(a: Analysis, model: HandModel): HandSetup {
  const text = printArea(a);
  let any = false;
  for (let i = 0; i < PIXELS; i++) if (text[i]) { any = true; break; }
  return { model, region: a.inner, text, colour: any ? paperColour(a.median, text) : null };
}

export interface HandReading { share: number; onPrint: number; paper: number }

/** What the stricter colour model needs for its cover of the print map (``on_map``). */
export interface MapSetup {
  model: HandModel;          // the refined model
  region: Uint8Array;        // page region and map: a patch must reach past it
  wellInside: Uint8Array;    // the map, PRINT_INSIDE in from its edge
  mapArea: number;
  level: number;             // fine detail from which a spot shows print
}

/**
 * covered() and the per-frame shares of measure(). Holds its scratch buffers,
 * so one instance per worker measures frame after frame without allocating.
 */
export class HandMeter {
  private readonly lab = new Uint8Array(PIXELS * 3);
  private readonly hit = new Uint8Array(PIXELS);
  private readonly eroded = new Uint8Array(PIXELS);
  private readonly opened = new Uint8Array(PIXELS);
  private readonly labels = new Int32Array(PIXELS);
  private parent = new Int32Array(4096);
  private readonly area: number;
  private readonly textArea: number;
  private readonly setup: HandSetup;
  private labDone = false;

  constructor(setup: HandSetup) {
    this.setup = setup;
    let area = 0, textArea = 0;
    for (let i = 0; i < PIXELS; i++) { area += setup.region[i]; textArea += setup.text[i]; }
    this.area = Math.max(1, area);
    this.textArea = Math.max(1, textArea);
  }

  measure(frameBgr: Uint8Array): HandReading {
    const { model, region, text, colour } = this.setup;
    const lab = this.lab;
    for (let i = 0, o = 0; i < PIXELS; i++, o += 3) bgrToLab(frameBgr[o], frameBgr[o + 1], frameBgr[o + 2], lab, o);
    this.labDone = true;
    let share = 0, onPrint = 0, paper = 1;
    if (model.usable) {
      const covered = this.covered(model.lut, region);
      let s = 0, t = 0;
      for (let i = 0; i < PIXELS; i++) if (covered[i]) { s++; if (text[i]) t++; }
      share = s / this.area;
      onPrint = t / this.textArea;
    }
    if (colour) {
      const vMin = PAPER_VALUE_SHARE * colour[0];
      const pa = colour[1], pb = colour[2];
      const limit = PAPER_CHROMA_TOLERANCE * PAPER_CHROMA_TOLERANCE;
      let p = 0;
      for (let i = 0, o = 0; i < PIXELS; i++, o += 3) {
        if (!region[i]) continue;
        const v = Math.max(frameBgr[o], frameBgr[o + 1], frameBgr[o + 2]);
        if (v < vMin) continue;
        const da = lab[o + 1] - pa, db = lab[o + 2] - pb;
        if (da * da + db * db < limit) p++;
      }
      paper = p / this.area;
    }
    return { share: Math.fround(share), onPrint: Math.fround(onPrint), paper: Math.fround(paper) };
  }

  /**
   * on_map of measure(), for the frame last given to measure(): the stricter
   * model's cover of the print map, well inside it, only where the frame
   * shows no print (paper under a lamp is the colour of skin, but smooth).
   * `detail` gives the frame's fine detail when it is needed.
   */
  onMap(map: MapSetup, detail: () => Uint8Array): number {
    if (!this.labDone) throw new Error('onMap before measure');
    if (!map.model.usable) return 0;
    const covered = this.covered(map.model.lut, map.region);
    let any = false;
    for (let i = 0; i < PIXELS; i++) if (covered[i] && map.wellInside[i]) { any = true; break; }
    if (!any) return 0;
    const d = detail();
    let n = 0;
    for (let i = 0; i < PIXELS; i++) if (covered[i] && map.wellInside[i] && d[i] < map.level) n++;
    return Math.fround(n / map.mapArea);
  }

  /** covered(): patches of the model's colours that reach past `region`, inside it. Uses this.lab. */
  private covered(lut: Uint8Array, region: Uint8Array): Uint8Array {
    const lab = this.lab, hit = this.hit;
    for (let i = 0, o = 0; i < PIXELS; i++, o += 3) {
      hit[i] = lab[o] > MIN_LIGHT && lut[binOf(lab[o + 1]) * BINS + binOf(lab[o + 2])] ? 1 : 0;
    }
    // morphological open with a 3x3 square (OpenCV border: outside never erodes or dilates)
    const er = this.eroded, op = this.opened;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let v = 1;
        for (let dy = -1; dy <= 1 && v; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= H) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            if (!hit[yy * W + xx]) { v = 0; break; }
          }
        }
        er[y * W + x] = v;
      }
    }
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let v = 0;
        for (let dy = -1; dy <= 1 && !v; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= H) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            if (er[yy * W + xx]) { v = 1; break; }
          }
        }
        op[y * W + x] = v;
      }
    }
    // 8-connected components, with area and "reaches outside the region"
    const labels = this.labels;
    let parent = this.parent;
    let next = 1;
    const find = (a: number) => {
      while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
      return a;
    };
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        labels[i] = 0;
        if (!op[i]) continue;
        let label = 0;
        for (let k = 0; k < 4; k++) {
          let j = -1;
          if (k === 0) { if (x > 0) j = i - 1; }
          else if (y > 0) {
            if (k === 1) j = i - W;
            else if (k === 2) { if (x > 0) j = i - W - 1; }
            else if (x < W - 1) j = i - W + 1;
          }
          if (j < 0) continue;
          const l = labels[j];
          if (!l) continue;
          if (!label) label = find(l);
          else {
            const a = find(label), b = find(l);
            if (a !== b) { if (a < b) parent[b] = a; else parent[a] = b; }
            label = a < b ? a : b;
          }
        }
        if (!label) {
          if (next >= parent.length) {
            const grown = new Int32Array(parent.length * 2);
            grown.set(parent);
            parent = this.parent = grown;
          }
          label = next;
          parent[next] = next;
          next++;
        }
        labels[i] = label;
      }
    }
    const area = new Int32Array(next), out = new Uint8Array(next);
    for (let i = 0; i < PIXELS; i++) {
      if (!labels[i]) continue;
      const root = find(labels[i]);
      labels[i] = root;
      area[root]++;
      if (!region[i]) out[root] = 1;
    }
    const covered = this.hit; // reuse
    for (let i = 0; i < PIXELS; i++) {
      const l = labels[i];
      covered[i] = l && region[i] && out[l] && area[l] >= MIN_BLOB ? 1 : 0;
    }
    return covered;
  }
}
