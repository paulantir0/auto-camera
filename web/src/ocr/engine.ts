// The OCR engine the Python audit uses (rapidocr-onnxruntime 1.4.4 with the
// PP-OCRv4 detector and recogniser), ported for the browser: the same
// models in onnxruntime-web, and rapidocr's own steps around them, with
// OpenCV.js for the image operations rapidocr does with OpenCV. Only the two
// ways ocr.py calls it are ported: detection alone on a band
// (RapidOCR(band, use_det=True, use_cls=False, use_rec=False), with
// det_limit_type "max") and recognition alone on a small crop
// (RapidOCR(crop, use_det=False, use_cls=False, use_rec=True)).
//
// Images are BGR, 3 bytes per pixel, row by row, as cv2.imread gives them.
// The Python side runs NumPy 2, where arithmetic on float32 arrays stays in
// float32; Math.fround marks the places where that can tip a rounding.

import { pyRound } from '../pipeline/pyround.ts';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CV = any;

export interface Image { data: Uint8Array; width: number; height: number }

/** What the engine needs of an onnxruntime session. */
export interface Session {
  run(input: Float32Array, dims: number[]): Promise<{ data: Float32Array; dims: readonly number[] }>;
}

// rapidocr's config.yaml and ocr.py's init_engine
const MAX_SIDE_LEN = 2000;
const MIN_SIDE_LEN = 30;
const LETTERBOX_MIN_HEIGHT = 30;
const WIDTH_HEIGHT_RATIO = 8;
const DET_THRESH = 0.3;
const BOX_THRESH = 0.5;
const MAX_CANDIDATES = 1000;
const UNCLIP_RATIO = 1.6;
const MIN_SIZE = 3;
const REC_HEIGHT = 48;
const REC_WIDTH = 320;
const ARC_TOLERANCE = 0.25; // pyclipper's default

/** NumPy's round to an integer (half to even). */
const npRound = (x: number) => pyRound(x, 0);

function resize(cv: CV, img: Image, width: number, height: number): Image {
  const src = new cv.Mat(img.height, img.width, cv.CV_8UC3);
  const dst = new cv.Mat();
  try {
    src.data.set(img.data);
    cv.resize(src, dst, new cv.Size(width, height), 0, 0, cv.INTER_LINEAR);
    return { data: new Uint8Array(dst.data), width, height };
  } finally {
    src.delete();
    dst.delete();
  }
}

/** main.RapidOCR.preprocess: reduce_max_side, then increase_min_side; the ratios of the last resize are kept. */
function preprocess(cv: CV, img: Image): { img: Image; ratioH: number; ratioW: number } {
  let ratioH = 1, ratioW = 1;
  const scaleTo = (im: Image, ratio: number) => {
    let rh = Math.trunc(im.height * ratio), rw = Math.trunc(im.width * ratio);
    rh = Math.trunc(npRound(rh / 32) * 32);
    rw = Math.trunc(npRound(rw / 32) * 32);
    if (rw <= 0 || rh <= 0) throw new Error('resize_w or resize_h is less than or equal to 0');
    const out = resize(cv, im, rw, rh);
    return { out, rh: im.height / rh, rw: im.width / rw };
  };
  if (Math.max(img.height, img.width) > MAX_SIDE_LEN) {
    const ratio = img.height > img.width ? MAX_SIDE_LEN / img.height : MAX_SIDE_LEN / img.width;
    const r = scaleTo(img, ratio);
    img = r.out; ratioH = r.rh; ratioW = r.rw;
  }
  if (Math.min(img.height, img.width) < MIN_SIDE_LEN) {
    const ratio = img.height < img.width ? MIN_SIDE_LEN / img.height : MIN_SIDE_LEN / img.width;
    const r = scaleTo(img, ratio);
    img = r.out; ratioH = r.rh; ratioW = r.rw;
  }
  return { img, ratioH, ratioW };
}

/** BGR bytes to a normalised float CHW tensor: (x / 255 - 0.5) / 0.5, channels kept in BGR order. */
function toTensor(img: Image, padWidth = img.width): Float32Array {
  const { width: w, height: h, data } = img;
  const out = new Float32Array(3 * h * padWidth);
  for (let c = 0; c < 3; c++) {
    const plane = c * h * padWidth;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) out[plane + y * padWidth + x] = (data[(y * w + x) * 3 + c] / 255 - 0.5) / 0.5;
    }
  }
  return out;
}

type Pt = [number, number];

/** DBPostProcess.get_mini_boxes: the min-area rectangle's corners in rapidocr's order, and its shorter side. */
function miniBox(cv: CV, rect: { center: { x: number; y: number }; size: { width: number; height: number }; angle: number }): [Pt[], number] {
  const pts: Pt[] = cv.RotatedRect.points(rect).map((p: { x: number; y: number }) => [Math.fround(p.x), Math.fround(p.y)] as Pt);
  pts.sort((a, b) => a[0] - b[0]);
  let i1 = 0, i2 = 1, i3 = 2, i4 = 3;
  if (pts[1][1] > pts[0][1]) { i1 = 0; i4 = 1; } else { i1 = 1; i4 = 0; }
  if (pts[3][1] > pts[2][1]) { i2 = 2; i3 = 3; } else { i2 = 3; i3 = 2; }
  return [[pts[i1], pts[i2], pts[i3], pts[i4]], Math.min(rect.size.width, rect.size.height)];
}

function minAreaRect(cv: CV, points: number[]): ReturnType<CV> {
  const m = cv.matFromArray(points.length / 2, 1, cv.CV_32SC2, points);
  try { return cv.minAreaRect(m); } finally { m.delete(); }
}

/** box_score_fast: mean of the prediction inside the box. */
function boxScoreFast(cv: CV, pred: Float32Array, w: number, h: number, box: Pt[]): number {
  const clip = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
  const xmin = clip(Math.floor(Math.min(...box.map(p => p[0]))), w - 1), xmax = clip(Math.ceil(Math.max(...box.map(p => p[0]))), w - 1);
  const ymin = clip(Math.floor(Math.min(...box.map(p => p[1]))), h - 1), ymax = clip(Math.ceil(Math.max(...box.map(p => p[1]))), h - 1);
  const mw = xmax - xmin + 1, mh = ymax - ymin + 1;
  const mask = cv.Mat.zeros(mh, mw, cv.CV_8UC1);
  const poly = cv.matFromArray(4, 1, cv.CV_32SC2, box.flatMap(p => [Math.trunc(Math.fround(p[0] - xmin)), Math.trunc(Math.fround(p[1] - ymin))]));
  const polys = new cv.MatVector();
  try {
    polys.push_back(poly);
    cv.fillPoly(mask, polys, new cv.Scalar(1));
    let sum = 0, n = 0;
    const m = mask.data as Uint8Array;
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        if (m[y * mw + x]) { sum += pred[(ymin + y) * w + xmin + x]; n++; }
      }
    }
    return n ? sum / n : 0;
  } finally {
    mask.delete(); poly.delete(); polys.delete();
  }
}

const clipperRound = (v: number) => (v < 0 ? Math.trunc(v - 0.5) : Math.trunc(v + 0.5));

/**
 * DBPostProcess.unclip: grow the box by area * ratio / perimeter, as
 * pyclipper's PyclipperOffset with round joins does (Clipper 6.4: integer
 * input, arcs of tolerance 0.25, rounded points); returns those points.
 */
function unclip(box: Pt[]): number[] {
  let area = 0, length = 0;
  for (let i = 0; i < 4; i++) {
    const [x1, y1] = box[i], [x2, y2] = box[(i + 1) % 4];
    area += x1 * y2 - x2 * y1;
    length += Math.hypot(x2 - x1, y2 - y1);
  }
  const delta = Math.abs(area / 2) * UNCLIP_RATIO / length;
  let src: Pt[] = box.map(p => [Math.trunc(p[0]), Math.trunc(p[1])] as Pt);
  // drop repeated points, as Clipper does
  src = src.filter((p, i) => i === 0 || p[0] !== src[i - 1][0] || p[1] !== src[i - 1][1]);
  while (src.length > 1 && src[src.length - 1][0] === src[0][0] && src[src.length - 1][1] === src[0][1]) src.pop();
  if (src.length < 3) return src.flat();
  // Clipper makes the outer polygon positively oriented (its Area >= 0) before offsetting;
  // its Area is minus half the trapezoid sum
  let a2 = 0;
  for (let i = 0, j = src.length - 1; i < src.length; j = i++) a2 += (src[j][0] + src[i][0]) * (src[j][1] - src[i][1]);
  if (!(-a2 / 2 >= 0)) src.reverse();
  const n = src.length;
  const normals: Pt[] = [];
  for (let j = 0; j < n; j++) {
    const [x1, y1] = src[j], [x2, y2] = src[(j + 1) % n];
    const dx = x2 - x1, dy = y2 - y1;
    if (dx === 0 && dy === 0) { normals.push([0, 0]); continue; }
    const f = 1 / Math.sqrt(dx * dx + dy * dy);
    normals.push([dy * f, -dx * f]);
  }
  const y = ARC_TOLERANCE > Math.abs(delta) * ARC_TOLERANCE ? Math.abs(delta) * ARC_TOLERANCE : ARC_TOLERANCE;
  let steps = Math.PI / Math.acos(1 - y / Math.abs(delta));
  if (steps > Math.abs(delta) * Math.PI) steps = Math.abs(delta) * Math.PI;
  const sin = Math.sin(2 * Math.PI / steps), cos = Math.cos(2 * Math.PI / steps);
  const stepsPerRad = steps / (2 * Math.PI);
  const out: number[] = [];
  const push = (x: number, yy: number) => out.push(clipperRound(x), clipperRound(yy));
  for (let j = 0, k = n - 1; j < n; k = j++) {
    const [px, py] = src[j];
    let sinA = normals[k][0] * normals[j][1] - normals[j][0] * normals[k][1];
    if (Math.abs(sinA * delta) < 1) {
      const cosA = normals[k][0] * normals[j][0] + normals[j][1] * normals[k][1];
      if (cosA > 0) { push(px + normals[k][0] * delta, py + normals[k][1] * delta); continue; }
    } else if (sinA > 1) sinA = 1;
    else if (sinA < -1) sinA = -1;
    if (sinA * delta < 0) {
      push(px + normals[k][0] * delta, py + normals[k][1] * delta);
      out.push(px, py);
      push(px + normals[j][0] * delta, py + normals[j][1] * delta);
      continue;
    }
    // DoRound
    const angle = Math.atan2(sinA, normals[k][0] * normals[j][0] + normals[k][1] * normals[j][1]);
    const count = Math.max(clipperRound(stepsPerRad * Math.abs(angle)), 1);
    let X = normals[k][0], Y = normals[k][1];
    for (let i = 0; i < count; i++) {
      push(px + X * delta, py + Y * delta);
      const X2 = X;
      X = X * cos - sin * Y;
      Y = X2 * sin + Y * cos;
    }
    push(px + normals[j][0] * delta, py + normals[j][1] * delta);
  }
  return out;
}

/** TextDetector.order_points_clockwise: tl, tr, br, bl. */
function orderClockwise(pts: Pt[]): Pt[] {
  const xs = [...pts].sort((a, b) => a[0] - b[0]);
  const left = xs.slice(0, 2).sort((a, b) => a[1] - b[1]);
  const right = xs.slice(2).sort((a, b) => a[1] - b[1]);
  return [left[0], right[0], right[1], left[1]];
}

export class OcrEngine {
  private readonly cv: CV;
  private readonly det: Session;
  private readonly rec: Session;
  private readonly chars: string[];
  /** Seconds in the two models and around them. */
  readonly spent = { detect: 0, recognise: 0, models: 0 };

  /** `characters`: the recogniser's dictionary (its "character" metadata), one per line. */
  constructor(cv: CV, det: Session, rec: Session, characters: string[]) {
    this.cv = cv;
    this.det = det;
    this.rec = rec;
    // CTCLabelDecode: "blank" first, a space last
    this.chars = ['blank', ...characters, ' '];
  }

  /** Text boxes in a band, as four corners each in the band's pixels (rapidocr with use_det only). */
  async detect(band: Image): Promise<Pt[][]> {
    const t0 = performance.now();
    const cv = this.cv;
    const rawH = band.height, rawW = band.width;
    const pre = preprocess(cv, band);
    let img = pre.img;
    // maybe_add_letterbox
    let top = 0;
    if (img.height <= LETTERBOX_MIN_HEIGHT || img.width / img.height > WIDTH_HEIGHT_RATIO) {
      const newH = Math.max(Math.trunc(img.width / WIDTH_HEIGHT_RATIO), LETTERBOX_MIN_HEIGHT) * 2;
      top = Math.trunc(Math.abs(newH - img.height) / 2);
      const padded = new Uint8Array((img.height + 2 * top) * img.width * 3);
      padded.set(img.data, top * img.width * 3);
      img = { data: padded, width: img.width, height: img.height + 2 * top };
    }
    // TextDetector: DetPreProcess with limit type "max"
    const maxWh = Math.max(img.height, img.width);
    const limit = maxWh < 960 ? 960 : maxWh < 1500 ? 1500 : 2000;
    const ratio = maxWh > limit ? (img.height > img.width ? limit / img.height : limit / img.width) : 1;
    const rh = Math.trunc(npRound(Math.trunc(img.height * ratio) / 32) * 32), rw = Math.trunc(npRound(Math.trunc(img.width * ratio) / 32) * 32);
    if (rh <= 0 || rw <= 0) return [];
    const resized = resize(cv, img, rw, rh);
    const t1 = performance.now();
    const pred = await this.det.run(toTensor(resized), [1, 3, rh, rw]);
    const t2 = performance.now();
    const ph = pred.dims[2], pw = pred.dims[3];
    const p = pred.data;
    // DBPostProcess: threshold, dilate with a 2x2 kernel, contours
    const seg = new cv.Mat(ph, pw, cv.CV_8UC1);
    const dil = new cv.Mat();
    const kernel = cv.Mat.ones(2, 2, cv.CV_8U);
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    const boxes: Pt[][] = [];
    try {
      const s = seg.data as Uint8Array;
      const thresh = Math.fround(DET_THRESH); // NumPy 2 compares float32 with float32
      for (let i = 0; i < ph * pw; i++) s[i] = p[i] > thresh ? 1 : 0;
      cv.dilate(seg, dil, kernel);
      const d = dil.data as Uint8Array;
      for (let i = 0; i < d.length; i++) d[i] = d[i] ? 255 : 0;
      cv.findContours(dil, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
      const destW = img.width, destH = img.height;
      const count = Math.min(contours.size(), MAX_CANDIDATES);
      for (let c = 0; c < count; c++) {
        const contour = contours.get(c);
        let points: Pt[], sside: number;
        try { [points, sside] = miniBox(cv, cv.minAreaRect(contour)); } finally { contour.delete(); }
        if (sside < MIN_SIZE) continue;
        const score = boxScoreFast(cv, p, pw, ph, points);
        if (BOX_THRESH > score) continue;
        const grown = unclip(points);
        if (grown.length < 6) continue;
        let box: Pt[];
        [box, sside] = miniBox(cv, minAreaRect(cv, grown));
        if (sside < MIN_SIZE + 2) continue;
        box = box.map(([x, y]) => [
          Math.trunc(Math.min(Math.max(npRound(Math.fround(Math.fround(x / pw) * destW)), 0), destW)),
          Math.trunc(Math.min(Math.max(npRound(Math.fround(Math.fround(y / ph) * destH)), 0), destH)),
        ] as Pt);
        // filter_tag_det_res
        let q = orderClockwise(box);
        q = q.map(([x, y]) => [Math.trunc(Math.min(Math.max(x, 0), destW - 1)), Math.trunc(Math.min(Math.max(y, 0), destH - 1))] as Pt);
        const rectW = Math.trunc(Math.hypot(q[0][0] - q[1][0], q[0][1] - q[1][1]));
        const rectH = Math.trunc(Math.hypot(q[0][0] - q[3][0], q[0][1] - q[3][1]));
        if (rectW <= 3 || rectH <= 3) continue;
        boxes.push(q);
      }
    } finally {
      seg.delete(); dil.delete(); kernel.delete(); contours.delete(); hierarchy.delete();
    }
    // _get_origin_points: undo the letterbox, then the preprocess resize, then clip
    const out = boxes.map(b => b.map(([x, y]) => {
      let X = Math.fround(x * pre.ratioW), Y = Math.fround((y - top) * pre.ratioH);
      if (X < 0) X = 0;
      if (Y < 0) Y = 0;
      if (X > rawW) X = rawW;
      if (Y > rawH) Y = rawH;
      return [X, Y] as Pt;
    }));
    this.spent.detect += (performance.now() - t0 - (t2 - t1)) / 1000;
    this.spent.models += (t2 - t1) / 1000;
    return out;
  }

  /** The text on a small crop and its score (rapidocr with use_rec only). */
  async recognise(crop: Image): Promise<[string, number]> {
    if (!crop.width || !crop.height) return ['', 0];
    const t0 = performance.now();
    const cv = this.cv;
    const img = preprocess(cv, crop).img;
    // TextRecognizer.resize_norm_img, one image
    const whRatio = img.width / img.height;
    const maxWhRatio = Math.max(REC_WIDTH / REC_HEIGHT, whRatio);
    const imgW = Math.trunc(REC_HEIGHT * maxWhRatio);
    const resizedW = Math.ceil(REC_HEIGHT * whRatio) > imgW ? imgW : Math.trunc(Math.ceil(REC_HEIGHT * whRatio));
    const resized = resize(cv, img, resizedW, REC_HEIGHT);
    const t1 = performance.now();
    const pred = await this.rec.run(toTensor(resized, imgW), [1, 3, REC_HEIGHT, imgW]);
    const t2 = performance.now();
    // CTCLabelDecode, greedy, duplicates and blanks removed; score is the mean probability kept
    const T = pred.dims[1], C = pred.dims[2], d = pred.data;
    let text = '', sum = 0, n = 0, previous = -1;
    for (let t = 0; t < T; t++) {
      let best = 0, prob = d[t * C];
      for (let c = 1; c < C; c++) if (d[t * C + c] > prob) { prob = d[t * C + c]; best = c; }
      if (best !== previous && best !== 0) { text += this.chars[best]; sum += prob; n++; }
      previous = best;
    }
    this.spent.recognise += (performance.now() - t0 - (t2 - t1)) / 1000;
    this.spent.models += (t2 - t1) / 1000;
    return [text.trim().replace(/ /g, ''), n ? sum / n : 0];
  }
}
