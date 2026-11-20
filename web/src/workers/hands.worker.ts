// A hands worker: owns one stretch of the reduced copy and decodes it once per
// pass (hands.measure of the Python pipeline, split into what can be done
// before the whole video's colour model and print map are known):
//
//   init      load Google's two hand models (MediaPipe), when the page asks for them
//   model     the sampled frames into the colour histograms; the frames at rest:
//             their fine detail on the print, and the hands the models find in them
//   printsums the frames at rest into the sums the print map is made from
//   strict    the sampled frames into the stricter colour model's histograms
//   measure   every frame: the colour models' cover, and on the frames that
//             matter (spreads.framesForHands) the trained models' hands

import { FrameDecoder } from '../video/decode.ts';
import { readFrame } from '../video/sampler.ts';
import { BINS, MIN_LIGHT, PROXY_HEIGHT, PROXY_WIDTH } from '../pipeline/constants.ts';
import { PIXELS } from '../pipeline/analysis.ts';
import { bgrToGray, bgrToLab, yuv420ToBgr } from '../pipeline/colour.ts';
import { HandMeter, ModelBuilder, binOf, type MapSetup } from '../pipeline/hands.ts';
import { HandDetector, bgrToImageData } from '../pipeline/handdetect.ts';
import { fineDetail, handMask, palms, skinPatches } from '../pipeline/handnet.ts';
import { PrintSums, maskedMedianU8, zones } from '../pipeline/printmap.ts';
import { dilate, rectKernel } from '../pipeline/imgproc.ts';
import { NET_WORK, type HandFiles, type PageSetup, type ProxyVideo } from '../messages.ts';

const post = (message: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(message, transfer);

let proxy: ProxyVideo | null = null;
let first = 0, last = 0;
let setup: PageSetup | null = null;
let samples: Set<number> = new Set();
let printFrames: number[] = [];
let detector: HandDetector | null = null;
const printHands = new Map<number, Uint8Array>(); // the hands found in the frames at rest, packed to bits

const W = PROXY_WIDTH, H = PROXY_HEIGHT;
const bgr = new Uint8Array(PIXELS * 3);
const planes = { y: new Uint8Array(PIXELS), u: new Uint8Array(PIXELS / 4), v: new Uint8Array(PIXELS / 4) };
const buf = { data: new Uint8Array(0) };
const image = new ImageData(W, H);
const grow = rectKernel(5, 5);

/** The decoded proxy frame as BGR, as FFmpeg's -pix_fmt bgr24 gives it. */
async function toBgr(frame: VideoFrame): Promise<Uint8Array> {
  const { format, layout } = await readFrame(frame, buf);
  const d = buf.data;
  for (let y = 0; y < H; y++) planes.y.set(d.subarray(layout[0].offset + y * layout[0].stride, layout[0].offset + y * layout[0].stride + W), y * W);
  if (format === 'NV12') {
    for (let y = 0; y < H / 2; y++) {
      const o = layout[1].offset + y * layout[1].stride;
      for (let x = 0; x < W / 2; x++) { planes.u[y * W / 2 + x] = d[o + 2 * x]; planes.v[y * W / 2 + x] = d[o + 2 * x + 1]; }
    }
  } else {
    for (let y = 0; y < H / 2; y++) {
      planes.u.set(d.subarray(layout[1].offset + y * layout[1].stride, layout[1].offset + y * layout[1].stride + W / 2), y * W / 2);
      planes.v.set(d.subarray(layout[2].offset + y * layout[2].stride, layout[2].offset + y * layout[2].stride + W / 2), y * W / 2);
    }
  }
  return yuv420ToBgr(planes.y, planes.u, planes.v, W, H, bgr);
}

/** Decode frames first..last of the proxy (keyframe-aligned) and call `each` with (index, BGR frame) for those `wanted` says. */
async function decodeStretch(each: (index: number, frame: Uint8Array) => Promise<void> | void, wanted: (index: number) => boolean = () => true): Promise<void> {
  const p = proxy!;
  const fps = p.fps;
  const decoder = new FrameDecoder(p.config, async frame => {
    const index = Math.round(frame.timestamp * fps / 1e6);
    if (index >= first && index <= last && wanted(index)) await each(index, await toBgr(frame));
  }, { maxQueue: 8, maxFrames: 4 });
  try {
    for (const c of p.chunks) await decoder.decode(c);
    await decoder.flush();
  } finally {
    decoder.close();
  }
}

const pack = (mask: Uint8Array) => {
  const out = new Uint8Array(Math.ceil(mask.length / 8));
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[i >> 3] |= 128 >> (i & 7);
  return out;
};
const unpack = (bits: Uint8Array, out = new Uint8Array(PIXELS)) => {
  for (let i = 0; i < PIXELS; i++) out[i] = (bits[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
};

/** What hands cover in a frame, by the trained models (handnet.Detections.mask); the landmarker only runs when a patch of skin needs a palm. */
function hands(frame: Uint8Array, out = new Uint8Array(PIXELS)): Uint8Array {
  const d = detector!;
  bgrToImageData(frame, image);
  const patches = skinPatches(d.skin(image));
  let palmMask: Uint8Array | null = null;
  if (patches?.inner) {
    const found = d.hands(image);
    palmMask = palms(found.landmarks, found.scores);
  }
  return handMask(patches, palmMask, out);
}

/** _hand_coloured: the colour model's colours, without looking at patches. */
function handColoured(frame: Uint8Array, lut: Uint8Array): Uint8Array {
  const out = new Uint8Array(PIXELS), lab = new Uint8Array(3);
  for (let i = 0, o = 0; i < PIXELS; i++, o += 3) {
    bgrToLab(frame[o], frame[o + 1], frame[o + 2], lab, 0);
    out[i] = lab[0] > MIN_LIGHT && lut[binOf(lab[1]) * BINS + binOf(lab[2])] ? 1 : 0;
  }
  return out;
}

async function init(files: HandFiles): Promise<void> {
  try {
    const load = async (name: string) => {
      const r = await fetch(`${files.base}/${name}`);
      if (!r.ok) throw new Error(`${name}: ${r.status}`);
      return new Uint8Array(await r.arrayBuffer());
    };
    const [landmarker, segmenter] = await Promise.all([load('hand_landmarker.task'), load('selfie_multiclass_256x256.tflite')]);
    detector = await HandDetector.create({ base: files.base, landmarker, segmenter });
    post({ kind: 'ready', net: true });
  } catch (err) {
    detector = null;
    post({ kind: 'ready', net: false, message: err instanceof Error ? err.message : String(err) });
  }
}

async function model(useNet: boolean): Promise<void> {
  const s = setup!;
  const builder = new ModelBuilder(s.median, s.mask, s.inner);
  const atRest = new Set(printFrames);
  const medians: number[] = [];
  let done = 0;
  await decodeStretch((index, frame) => {
    if (samples.has(index)) builder.add(frame);
    if (atRest.has(index)) {
      medians.push(maskedMedianU8(fineDetail(bgrToGray(frame)), s.text));
      if (useNet) {
        printHands.set(index, pack(hands(frame)));
        done += NET_WORK;
        if (done >= 200) { post({ kind: 'progress', done }); done = 0; }
      }
    }
  }, index => samples.has(index) || atRest.has(index));
  post({ kind: 'progress', done });
  post({ kind: 'model-done', intruder: builder.intruder, page: builder.page, frames: printFrames.filter(f => f >= first && f <= last), medians });
}

async function printSums(lut: Uint8Array, usable: boolean, level: number, useNet: boolean): Promise<void> {
  const sums = new PrintSums();
  const atRest = new Set(printFrames);
  const visible = new Uint8Array(PIXELS), hand = new Uint8Array(PIXELS);
  await decodeStretch((index, frame) => {
    const gray = bgrToGray(frame);
    visible.fill(1);
    if (usable) {
      const coloured = dilate(handColoured(frame, lut), W, H, grow);
      for (let i = 0; i < PIXELS; i++) if (coloured[i]) visible[i] = 0;
    }
    const bits = useNet ? printHands.get(index) : undefined;
    if (bits) {
      const grown = dilate(unpack(bits, hand), W, H, grow);
      for (let i = 0; i < PIXELS; i++) if (grown[i]) visible[i] = 0;
    }
    sums.add(gray, fineDetail(gray), visible, level);
  }, index => atRest.has(index));
  post({ kind: 'printsums-done', seen: sums.seen, printed: sums.printed, light: sums.light, light2: sums.light2, frames: sums.frames },
    [sums.seen.buffer, sums.printed.buffer, sums.light.buffer, sums.light2.buffer]);
}

async function strict(map: Uint8Array, level: number): Promise<void> {
  const s = setup!;
  const page = new Uint8Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) page[i] = s.inner[i] || map[i] ? 1 : 0;
  const builder = new ModelBuilder(s.median, s.mask, page, { printLevel: level });
  await decodeStretch((_, frame) => {
    builder.add(frame, level > 0 ? fineDetail(bgrToGray(frame)) : undefined);
  }, index => samples.has(index));
  post({ kind: 'strict-done', intruder: builder.intruder, anyChange: builder.anyChange, page: builder.page });
}

interface MeasureRequest {
  lut: Uint8Array; usable: boolean;
  strictLut: Uint8Array; strictUsable: boolean;
  map: Uint8Array; level: number;
  needed: Uint8Array; // frames first..last that the trained models must look at
  useNet: boolean;
}

async function measure(m: MeasureRequest): Promise<void> {
  const s = setup!;
  const meter = new HandMeter({ model: { lut: m.lut, samples: 0, usable: m.usable }, region: s.inner, text: s.text, colour: s.colour });
  const z = zones({ mask: s.mask, inner: s.inner }, s.text, m.map);
  const mapSetup: MapSetup = { model: { lut: m.strictLut, samples: 0, usable: m.strictUsable }, region: z.region, wellInside: z.wellInside, mapArea: z.mapArea, level: m.level };
  let anyText = false;
  for (let i = 0; i < PIXELS; i++) if (z.textAll[i]) { anyText = true; break; }
  const count = last - first + 1;
  const share = new Float32Array(count), onPrint = new Float32Array(count), paper = new Float32Array(count).fill(1), onMap = new Float32Array(count);
  const useNet = m.useNet && anyText;
  const net = useNet ? new Float32Array(count) : null, netPage = useNet ? new Float32Array(count) : null;
  const netNear = useNet ? new Float32Array(count) : null, netInside = useNet ? new Float32Array(count) : null;
  const hand = new Uint8Array(PIXELS);
  let done = 0;
  await decodeStretch((index, frame) => {
    const k = index - first;
    const r = meter.measure(frame);
    share[k] = r.share; onPrint[k] = r.onPrint; paper[k] = r.paper;
    let detail: Uint8Array | null = null;
    const fine = () => (detail ??= fineDetail(bgrToGray(frame)));
    onMap[k] = meter.onMap(mapSetup, fine);
    done++;
    if (useNet && m.needed[k]) {
      const bits = printHands.get(index);
      if (bits) unpack(bits, hand);
      else { hands(frame, hand); done += NET_WORK; }
      let t = 0, ins = 0, pg = 0, nr = 0;
      for (let i = 0; i < PIXELS; i++) if (hand[i]) { t += z.textAll[i]; ins += z.inside[i]; pg += s.mask[i]; nr += z.near[i]; }
      net![k] = t / z.textArea; netInside![k] = ins / z.insideArea; netPage![k] = pg / z.pageArea; netNear![k] = nr / z.textArea;
    }
    if (done >= 200) { post({ kind: 'progress', done }); done = 0; }
  });
  post({ kind: 'progress', done });
  const arrays = [share, onPrint, paper, onMap, net, netPage, netNear, netInside].filter(Boolean) as Float32Array[];
  post({ kind: 'measure-done', first, share, onPrint, paper, onMap, net, netPage, netNear, netInside }, arrays.map(a => a.buffer));
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.kind === 'init') {
      proxy = m.proxy; first = m.first; last = m.last; setup = m.setup;
      samples = new Set(m.samples); printFrames = m.printFrames;
      if (m.files) await init(m.files);
      else post({ kind: 'ready', net: false });
    } else if (m.kind === 'model') await model(m.useNet);
    else if (m.kind === 'printsums') await printSums(m.lut, m.usable, m.level, m.useNet);
    else if (m.kind === 'strict') await strict(m.map, m.level);
    else if (m.kind === 'measure') await measure(m);
  } catch (err) {
    post({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};

