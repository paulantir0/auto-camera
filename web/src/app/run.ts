// One scan of one video, start to finish, coordinating the workers. The page
// thread only passes messages and small arrays; every heavy step runs in a
// worker, and stopping terminates them all.

import { decodeStart, readVideoTrack, type VideoTrack } from '../video/mp4.ts';
import { finishModel } from '../pipeline/hands.ts';
import { regionCentreX, shownReasons, type Candidate, type Capture, type Selection } from '../pipeline/repeats.ts';
import type { Spread } from '../pipeline/spreads.ts';
import { BINS, SAMPLE_FRAMES } from '../pipeline/constants.ts';
import { printLevel } from '../pipeline/printmap.ts';
import { dilate, rectKernel } from '../pipeline/imgproc.ts';

/** pipeline.PRINT_REGION_GROW: proxy pixels the print map is enlarged by to find lines of text in a still (the book creeps). */
export const PRINT_REGION_GROW = 25;
import { NET_WORK, type HandArrays, type PageSetup, type ProxyVideo, type StillResult } from '../messages.ts';
import { softwareHevcPossible } from '../video/wasm-decoder.ts';
import { numberCheck, type Checked, type NumberCheck, type Readings } from '../ocr/audit.ts';
import type { Token } from '../ocr/read.ts';

export const STAGES = [
  { id: 'read', label: 'Reading the video' },
  { id: 'hands', label: 'Looking for hands over the pages' },
  { id: 'turns', label: 'Finding the page turns' },
  { id: 'pictures', label: 'Taking the pictures' },
  { id: 'repeats', label: 'Removing pictures of the same pages' },
] as const;
export type StageId = typeof STAGES[number]['id'];

export interface Progress { stage: StageId; fraction: number; detail?: string }

export interface Picture {
  frame: number;
  capture: Checked;
  page?: number | null; // the printed number of its left page, once the page numbers have been read
  number: number;
  name: string; // sequence_NNNN_tSSSS.SSs.jpg
  seconds: number; // time in the video
  flags: string[]; // plain-language reasons to look at it
  preview: Blob;
  full: Blob;
  size: number;
  crc: number;
  width: number;
  height: number;
}

/** What the page-number check needs that only the scan knows. */
export interface CheckSetup { box: number[]; rows: Record<number, number[] | null>; fps: number }

export interface ScanResult {
  pictures: Picture[];
  setup: CheckSetup;
  /** Where the print is in the frames (the print map, enlarged by PRINT_REGION_GROW as the Python pipeline does for its pages), 480x270; null without one. */
  region: { data: Uint8Array; width: number; height: number } | null;
  seconds: Record<string, number>;
  details: Record<string, string | number>;
}

export class UnsupportedVideo extends Error {}

const FLAG_TEXT: Record<string, string> = {
  hand_on_print: 'Hand over the text',
  never_at_rest: 'Page was moving',
};

const STORE_PREFIX = 'scan-';

/** Remove pictures left on this device by earlier scans. */
export async function clearStore(): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    for await (const name of (root as any).keys() as AsyncIterable<string>) {
      if (name.startsWith(STORE_PREFIX)) await root.removeEntry(name, { recursive: true });
    }
  } catch { /* no private storage here */ }
}

async function storeAvailable(): Promise<boolean> {
  try {
    const root = await navigator.storage.getDirectory();
    const probe = await root.getDirectoryHandle(`${STORE_PREFIX}probe`, { create: true });
    await root.removeEntry(probe.name, { recursive: true });
    return true;
  } catch {
    return false;
  }
}

/** The codec for the reduced copy of the video used by the hand measurement. */
async function chooseProxyCodec(fps: number): Promise<string> {
  for (const codec of ['avc1.42001f', 'avc1.42e01f', 'avc1.4d001f', 'vp8', 'vp09.00.10.08']) {
    try {
      const r = await VideoEncoder.isConfigSupported({ codec, width: 480, height: 270, bitrate: 600_000, framerate: fps, latencyMode: 'quality' });
      if (r.supported) return codec;
    } catch { /* try the next */ }
  }
  throw new UnsupportedVideo('This browser cannot encode video, which this page needs for a small working copy.');
}

/** How the video will be decoded: by the browser (WebCodecs), or by the page's own HEVC decoder in WebAssembly. */
export type DecoderKind = 'browser' | 'built-in';

export async function checkSupport(file: File): Promise<{ track: VideoTrack; hardware: boolean | null; decoder: DecoderKind }> {
  if (!('VideoDecoder' in self) || !('VideoEncoder' in self)) {
    throw new UnsupportedVideo('This browser does not have the video tools this page needs (WebCodecs). Try a recent Safari, Chrome or Edge.');
  }
  let track: VideoTrack;
  try {
    track = await readVideoTrack(file);
  } catch (e) {
    throw new UnsupportedVideo(e instanceof Error ? e.message : String(e));
  }
  const config: VideoDecoderConfig = { codec: track.codec, codedWidth: track.codedWidth, codedHeight: track.codedHeight, description: track.description };
  let ok = false;
  try { ok = !!(await VideoDecoder.isConfigSupported(config)).supported; } catch { ok = false; }
  const hevc = /^(hvc1|hev1)/.test(track.codec);
  if (!ok && hevc && softwareHevcPossible() && !new URLSearchParams(location.search).has('no-builtin')) {
    return { track, hardware: false, decoder: 'built-in' };
  }
  if (!ok) {
    throw new UnsupportedVideo(hevc
      ? 'This browser cannot play HEVC (H.265) video, the format iPhones record in. Open this page in Safari on an iPhone, iPad or Mac, or in Chrome or Edge on a computer whose graphics chip decodes HEVC.'
      : `This browser cannot decode this video (${track.codec}).`);
  }
  let hardware: boolean | null = null;
  try { hardware = !!(await VideoDecoder.isConfigSupported({ ...config, hardwareAcceleration: 'prefer-hardware' })).supported; } catch { hardware = null; }
  return { track, hardware, decoder: new URLSearchParams(location.search).has('builtin') && hevc ? 'built-in' : 'browser' };
}

interface Pool { size: number; still: number; frame: number; decodeThreads: number; stillThreads: number }

function poolSize(decoder: DecoderKind): Pool {
  const cores = navigator.hardwareConcurrency || 4;
  const memory = (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 4;
  const size = Math.max(1, Math.min(4, cores - 1, memory <= 2 ? 2 : 4));
  const still = Math.max(1, Math.min(3, Math.floor(cores / 2), memory <= 4 ? 2 : 3));
  if (decoder === 'browser') {
    const frame = Math.max(1, Math.min(3, cores - 2));
    return { size, still, frame, decodeThreads: 0, stillThreads: 0 };
  }
  // The built-in decoder is the slow part: one helper keeps up with it, and it gets the other cores.
  const decodeThreads = Math.max(1, Math.min(8, cores - 2));
  const stillThreads = Math.max(1, Math.min(4, Math.floor((cores - 1) / still)));
  return { size, still, frame: 1, decodeThreads, stillThreads };
}

export class Scan {
  private readonly workers = new Set<Worker>();
  private stopped = false;
  private readonly file: File;
  private readonly onProgress: (p: Progress) => void;
  private readonly started = performance.now();
  readonly seconds: Record<string, number> = {};
  readonly details: Record<string, string | number> = {};
  /** Per-frame measurements for numerical checks, kept only when the page is opened with ?debug. */
  debug: Record<string, unknown> | null = new URLSearchParams(location.search).has('debug') ? {} : null;

  private readonly decoder: DecoderKind;

  constructor(file: File, onProgress: (p: Progress) => void, decoder: DecoderKind = 'browser') {
    this.file = file;
    this.onProgress = onProgress;
    this.decoder = decoder;
  }

  private readonly pending = new Set<(e: Error) => void>();

  stop(): void {
    this.stopped = true;
    for (const w of this.workers) w.terminate();
    this.workers.clear();
    for (const reject of this.pending) reject(new Error('stopped'));
    this.pending.clear();
  }

  private spawn(name: string): Worker {
    if (this.stopped) throw new Error('stopped');
    const w = new Worker(`workers/${name}.js`);
    this.workers.add(w);
    return w;
  }

  private end(w: Worker): void {
    w.terminate();
    this.workers.delete(w);
  }

  /** Send a message and wait for the reply of kind `done`, passing progress messages on. */
  private ask<T>(w: Worker, message: unknown, done: string, transfer: Transferable[] = [], onMessage?: (m: { kind: string; [k: string]: unknown }) => void): Promise<T> {
    return new Promise((resolve, reject) => {
      if (this.stopped) { reject(new Error('stopped')); return; }
      this.pending.add(reject);
      const listener = (e: MessageEvent) => {
        const m = e.data;
        if (m.kind === 'error') { cleanup(); reject(new Error(m.message)); }
        else if (m.kind === done) { cleanup(); resolve(m as T); }
        else onMessage?.(m);
      };
      const failed = (e: ErrorEvent) => { cleanup(); reject(new Error(e.message || 'a worker failed')); };
      const cleanup = () => { this.pending.delete(reject); w.removeEventListener('message', listener); w.removeEventListener('error', failed); };
      w.addEventListener('message', listener);
      w.addEventListener('error', failed);
      w.postMessage(message, transfer);
    });
  }

  private mark(name: string, since: number): number {
    const now = performance.now();
    this.seconds[name] = Math.round((now - since) / 100) / 10;
    return now;
  }

  async run(track: VideoTrack): Promise<ScanResult> {
    const file = this.file;
    const pool = poolSize(this.decoder);
    const software = this.decoder === 'built-in';
    this.details.decoder = software ? `built into this page (WebAssembly, ${pool.decodeThreads} threads for reading, ${pool.stillThreads} per picture worker)` : 'the browser\'s own';
    this.details.video = `${track.codedWidth}x${track.codedHeight}, ${track.samples.length} frames, ${track.fps} fps, ${track.codec}`;
    this.details.workers = `${pool.frame} for reading, ${pool.size} for hands, ${pool.still} for pictures`;
    const useStore = await storeAvailable();
    this.details.storage = useStore ? 'on this device (private browser storage)' : 'in memory';
    await clearStore();
    const storeDir = `${STORE_PREFIX}${Date.now()}`;
    const proxyCodec = await chooseProxyCodec(track.fps || 11);
    this.details.workingCopy = proxyCodec;

    // still workers load OpenCV while the video is read
    const stillWorkers = Array.from({ length: pool.still }, () => this.spawn('still'));
    const loading = Promise.all(stillWorkers.map(w => this.ask(w, { kind: 'load' }, 'ready')));

    // 1. one pass over the whole video at reduced size
    let t = performance.now();
    const analysis = this.spawn('analysis');
    const total = track.samples.length;
    // frame helpers, each connected to the analysis worker by its own channel
    const helpers = Array.from({ length: pool.frame }, () => this.spawn('frame'));
    const ports = helpers.map(h => {
      const channel = new MessageChannel();
      h.postMessage({ kind: 'port', port: channel.port1 }, [channel.port1]);
      return channel.port2;
    });
    const p1 = await this.ask<{ frames: number; fps: number; proxy: ProxyVideo; page: PageSetup; region: Uint8Array; needed: Uint8Array; printFrames: number[]; seconds: Record<string, number>; decoderInfo: string; pageCells: number; debug: Record<string, Uint8Array> | null }>(
      analysis, { kind: 'pass1', file, track, encoderCodec: proxyCodec, ports, debug: this.debug !== null, software: software ? pool.decodeThreads : 0 }, 'pass1-done', ports,
      m => { if (m.kind === 'progress') this.onProgress({ stage: 'read', fraction: (m.done as number) / total }); });
    for (const h of helpers) this.end(h);
    if (this.debug && p1.debug) Object.assign(this.debug, p1.debug);
    t = this.mark('read', t);
    for (const [k, v] of Object.entries(p1.seconds)) this.details[`read: ${k} (s)`] = Math.round(v * 10) / 10;
    this.details.decodedFrames = p1.decoderInfo;
    this.details.pageCells = p1.pageCells;
    const frames = p1.frames, fps = p1.fps;
    this.onProgress({ stage: 'read', fraction: 1 });

    // 2. hands, in parallel stretches of the reduced copy (hands.measure): the colour model from sampled
    // frames and the hands in the frames at rest; the print map; the stricter colour model; then every frame
    const chunks = p1.proxy.chunks;
    // one chunk per frame, in frame order, with a keyframe every 11 frames as asked
    const aligned = chunks.length === frames && chunks.every((c, k) => (k === 0 || c.timestamp > chunks[k - 1].timestamp) && (k % 11 !== 0 || c.key));
    const keyAt = (k: number) => chunks[k]?.key;
    const parts = aligned ? pool.size : 1;
    const bounds: number[] = [0];
    for (let j = 1; j < parts; j++) {
      let b = Math.round(frames * j / parts / 11) * 11;
      while (b > bounds[bounds.length - 1] && !keyAt(b)) b -= 11;
      if (b > bounds[bounds.length - 1] && b < frames) bounds.push(b);
    }
    bounds.push(frames);
    const step = Math.max(1, Math.floor(frames / SAMPLE_FRAMES));
    const samples: number[] = [];
    for (let f = 0; f < frames; f += step) samples.push(f);
    const handWorkers = bounds.slice(0, -1).map(() => this.spawn('hands'));
    const wantNet = !new URLSearchParams(location.search).has('no-hand-models');
    const files = wantNet ? { base: new URL('hands', location.href).href } : null;
    const ready = await Promise.all(handWorkers.map((w, j) => {
      const first = bounds[j], last = bounds[j + 1] - 1;
      const mine = chunks.slice(first, last + 1);
      return this.ask<{ net: boolean; message?: string }>(w, {
        kind: 'init', proxy: { config: p1.proxy.config, chunks: mine, fps }, first, last, samples, printFrames: p1.printFrames, setup: p1.page, files,
      }, 'ready', mine.map(c => c.data.buffer as ArrayBuffer));
    }));
    const useNet = wantNet && ready.every(r => r.net);
    this.details.handModels = useNet ? 'Google MediaPipe hand landmarker and skin segmenter, version 0.10.21 (CPU)'
      : wantNet ? `not available in this browser (${ready.find(r => !r.net)?.message ?? 'unknown'}): colour only` : 'switched off';
    let neededCount = 0;
    for (const v of p1.needed) neededCount += v;
    const totalWork = frames + (useNet ? NET_WORK * (neededCount + p1.printFrames.length) : 0);
    let work = 0;
    const onWork = (m: { kind: string; [k: string]: unknown }) => {
      if (m.kind === 'progress') { work += m.done as number; this.onProgress({ stage: 'hands', fraction: Math.min(1, work / totalWork) }); }
    };
    const all = <T>(message: (j: number) => unknown, done: string) => Promise.all(handWorkers.map((w, j) => this.ask<T>(w, message(j), done, [], onWork)));
    // the colour model, and the frames at rest
    const histograms = await all<{ intruder: Float64Array; page: Float64Array; frames: number[]; medians: number[] }>(() => ({ kind: 'model', useNet }), 'model-done');
    const intruder = new Float64Array(BINS * BINS), pageHist = new Float64Array(BINS * BINS);
    for (const h of histograms) for (let i = 0; i < intruder.length; i++) { intruder[i] += h.intruder[i]; pageHist[i] += h.page[i]; }
    const model = finishModel(intruder, pageHist);
    this.details.handModel = model.usable ? `learned from ${model.samples} pixels` : `not usable (${model.samples} pixels)`;
    const medians = histograms.flatMap(h => h.medians);
    // the print map
    let map: Uint8Array = p1.page.text, level = 0;
    if (p1.printFrames.length && medians.length) {
      const sums = await all<{ seen: Int32Array; printed: Int32Array; light: Float32Array; light2: Float32Array; frames: number }>(
        () => ({ kind: 'printsums', lut: model.lut, usable: model.usable, level: printLevel(medians), useNet }), 'printsums-done');
      const made = await this.ask<{ map: Uint8Array; level: number }>(analysis, { kind: 'printmap', parts: sums, medians }, 'printmap-done');
      map = made.map; level = made.level;
    }
    // the stricter colour model, over the page and the map
    const refined = await all<{ intruder: Float64Array; anyChange: Float64Array; page: Float64Array }>(() => ({ kind: 'strict', map, level }), 'strict-done');
    const sIntruder = new Float64Array(BINS * BINS), sAny = new Float64Array(BINS * BINS), sPage = new Float64Array(BINS * BINS);
    for (const h of refined) for (let i = 0; i < sIntruder.length; i++) { sIntruder[i] += h.intruder[i]; sAny[i] += h.anyChange[i]; sPage[i] += h.page[i]; }
    const strict = finishModel(sIntruder, sPage, sAny);
    // every frame
    const share = new Float32Array(frames), onPrint = new Float32Array(frames), paper = new Float32Array(frames).fill(1), onMap = new Float32Array(frames);
    const netArrays = useNet ? { net: new Float32Array(frames), netPage: new Float32Array(frames), netNear: new Float32Array(frames), netInside: new Float32Array(frames) } : null;
    const measured = await all<HandArrays>(j => ({
      kind: 'measure', lut: model.lut, usable: model.usable, strictLut: strict.lut, strictUsable: strict.usable, map, level,
      needed: p1.needed.slice(bounds[j], bounds[j + 1]), useNet,
    }), 'measure-done');
    let anyNet = false;
    for (const r of measured) {
      share.set(r.share, r.first); onPrint.set(r.onPrint, r.first); paper.set(r.paper, r.first); onMap.set(r.onMap, r.first);
      if (netArrays && r.net) {
        anyNet = true;
        netArrays.net.set(r.net, r.first); netArrays.netPage.set(r.netPage!, r.first); netArrays.netNear.set(r.netNear!, r.first); netArrays.netInside.set(r.netInside!, r.first);
      }
    }
    for (const w of handWorkers) this.end(w);
    this.details.handFrames = useNet ? `${neededCount} of ${frames} frames looked at by the hand models` : 'none';
    t = this.mark('hands', t);
    this.onProgress({ stage: 'hands', fraction: 1 });

    // 3. turns and one candidate frame per spread
    const turns = await this.ask<{ spreads: Spread[]; landings: number; landingFrames: number[] }>(analysis, {
      kind: 'turns', hands: { usable: model.usable, samples: model.samples, share, onPrint, paper, onMap, ...(anyNet && netArrays ? netArrays : {}) },
    }, 'turns-done');
    this.details.pageTurns = turns.landings;
    if (this.debug) { this.debug.landings = turns.landingFrames; this.debug.spreads = turns.spreads.map(s => s.best); this.debug.hands = { share, onPrint, paper, onMap, ...(netArrays ?? {}) }; this.debug.printMap = map; }
    t = this.mark('turns', t);
    this.onProgress({ stage: 'turns', fraction: 1 });

    // 4. the chosen frames at full size
    await loading;
    const wanted = [...new Set(turns.spreads.map(s => s.best))].sort((a, b) => a - b);
    const stills = new Map<number, StillResult>();
    const spentPictures: Record<string, number> = {};
    const share2 = Math.ceil(wanted.length / stillWorkers.length);
    await Promise.all(stillWorkers.map(async (w, j) => {
      const mine = wanted.slice(j * share2, (j + 1) * share2);
      await this.ask(w, { kind: 'init', file, track, region: p1.region, jpegQuality: 0.92, previewWidth: 360, useStore, storeDir, software: software ? pool.stillThreads : 0 }, 'ready');
      if (!mine.length) return;
      const done = await this.ask<{ spent: Record<string, number> }>(w, { kind: 'frames', targets: mine.map(frame => ({ frame, start: decodeStart(track, frame) })) }, 'frames-done', [], m => {
        if (m.kind === 'still') {
          const r = m.result as StillResult;
          stills.set(r.frame, r);
          this.onProgress({ stage: 'pictures', fraction: stills.size / wanted.length });
        }
      });
      for (const [k, v] of Object.entries(done.spent)) spentPictures[k] = (spentPictures[k] ?? 0) + v;
    }));
    for (const [k, v] of Object.entries(spentPictures)) this.details[`pictures: ${k} (s, all workers)`] = Math.round(v * 10) / 10;
    for (const w of stillWorkers) this.end(w);
    t = this.mark('pictures', t);
    this.details.picturesTaken = wanted.length;

    // 5. the early merge, then the keypoint repeat check
    const grays: Record<number, Uint8Array> = {};
    for (const [f, r] of stills) grays[f] = r.gray;
    // the gray images and the keypoints are handed over, not copied: they are not needed here afterwards
    const merged = await this.ask<{ candidates: Candidate[]; rows: Record<number, number[] | null>; box: number[] }>(analysis, { kind: 'merge', grays }, 'merge-done',
      Object.values(grays).map(g => g.buffer as ArrayBuffer));
    if (this.debug) this.debug.candidates = merged.candidates.map(c => c.best);
    this.end(analysis);
    this.onProgress({ stage: 'repeats', fraction: 0.2 });
    const select = this.spawn('select');
    const features = merged.candidates.map(c => stills.get(c.best)!.features);
    const chosen = await this.ask<{ selection: Selection; seconds: number; comparisons: number; wasm: boolean }>(select, {
      candidates: merged.candidates, features, centreX: regionCentreX(p1.region), fps,
    }, 'select-done', [...new Set(features.flatMap(f => [f.xs.buffer, f.ys.buffer, f.desc.buffer]))] as ArrayBuffer[]);
    this.end(select);
    this.mark('repeats', t);
    this.details.candidates = merged.candidates.length;
    this.details.comparisons = chosen.comparisons;
    this.details.matcher = chosen.wasm ? 'WebAssembly SIMD' : 'JavaScript';
    this.details.removedAsRepeats = chosen.selection.repeatsMerged + chosen.selection.transitional.length + chosen.selection.shownElsewhere.length;
    this.details.notPages = chosen.selection.notPages.length;
    this.onProgress({ stage: 'repeats', fraction: 1 });

    // the pictures, and the files of candidates that were not kept removed from storage
    let dir: FileSystemDirectoryHandle | null = null;
    if (useStore) { try { dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(storeDir); } catch { dir = null; } }
    const kept = new Set(chosen.selection.captures.map(c => c.best));
    const pictures: Picture[] = [];
    for (const c of chosen.selection.captures as Capture[]) {
      const s = stills.get(c.best)!;
      let full: Blob | null = s.jpeg;
      if (!full && s.stored && dir) full = await (await dir.getFileHandle(s.stored)).getFile();
      if (!full) throw new Error('a picture went missing from storage');
      pictures.push({
        frame: c.best, capture: { best: c.best, hand_on_print: c.hand_on_print, detail: c.detail, best_motion: c.best_motion, review: c.review },
        number: c.index, name: c.file, seconds: c.best / fps, flags: shownReasons(c).map(r => FLAG_TEXT[r] ?? r),
        preview: s.preview, full, size: s.size, crc: s.crc, width: s.width, height: s.height,
      });
    }
    if (dir) for (const [f, s] of stills) if (!kept.has(f) && s.stored) { try { await dir.removeEntry(s.stored); } catch { /* ignore */ } }
    this.seconds.total = Math.round((performance.now() - this.started) / 100) / 10;
    const grown = dilate(map, 480, 270, rectKernel(2 * PRINT_REGION_GROW + 1, 2 * PRINT_REGION_GROW + 1));
    const region = grown.some(v => v) ? { data: grown, width: 480, height: 270 } : null;
    return { pictures, setup: { box: merged.box, rows: merged.rows, fps }, region, seconds: this.seconds, details: this.details };
  }
}

/**
 * Read the printed page numbers off the pictures (the Python audit's OCR,
 * src/ocr/) and run the printed-number check on them. The pictures are the
 * captures as selection chose them, in order.
 */
export class NumberReader {
  private worker: Worker | null = null;
  private stopped = false;
  private fail: ((e: Error) => void) | null = null;
  readonly seconds: Record<string, number> = {};

  stop(): void {
    this.stopped = true;
    this.worker?.terminate();
    this.worker = null;
    this.fail?.(new Error('stopped'));
  }

  async run(pictures: Picture[], setup: CheckSetup, onProgress: (done: number, total: number, loading: boolean) => void): Promise<{ check: NumberCheck; readings: Readings }> {
    const cores = navigator.hardwareConcurrency || 4;
    const worker = new Worker('workers/ocr.js');
    this.worker = worker;
    const readings: Readings = new Map();
    const started = performance.now();
    await new Promise<void>((resolve, reject) => {
      this.fail = reject;
      if (this.stopped) { reject(new Error('stopped')); return; }
      onProgress(0, pictures.length, true);
      worker.onmessage = (e: MessageEvent) => {
        const m = e.data;
        if (m.kind === 'ready') {
          this.seconds.load = (performance.now() - started) / 1000;
          onProgress(0, pictures.length, false);
          worker.postMessage({ kind: 'read', box: setup.box, jobs: pictures.map(p => ({ frame: p.frame, image: p.full, rows: setup.rows[p.frame] ?? null })) });
        } else if (m.kind === 'tokens') {
          readings.set(m.frame as number, m.tokens as Token[]);
          onProgress(readings.size, pictures.length, false);
        } else if (m.kind === 'read-done') {
          this.seconds.read = m.seconds as number;
          resolve();
        } else if (m.kind === 'error') reject(new Error(m.message));
      };
      worker.onerror = e => reject(new Error(e.message || 'the page-number reader failed'));
      worker.postMessage({ kind: 'init', threads: Math.max(1, Math.min(4, cores - 1)) });
    });
    worker.terminate();
    this.worker = null;
    // Python counts fixed places over every candidate's still; only the pictures kept are read here
    const check = numberCheck(readings, pictures.map(p => p.capture), pictures.map(p => p.frame));
    this.seconds.total = (performance.now() - started) / 1000;
    return { check, readings };
  }
}

export async function startScan(file: File, onProgress: (p: Progress) => void): Promise<{ scan: Scan; track: VideoTrack; hardware: boolean | null; decoder: DecoderKind }> {
  const { track, hardware, decoder } = await checkSupport(file);
  return { scan: new Scan(file, onProgress, decoder), track, hardware, decoder };
}
