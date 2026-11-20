// The analysis worker: decodes the whole video once at reduced size (pass 1),
// keeps the per-frame measurements, derives the page region, and later runs
// turn detection, frame choice and the early merge on request.

import { readSamples, sampleTimestamp, type VideoTrack } from '../video/mp4.ts';
import { FrameDecoder, type Decoder } from '../video/decode.ts';
import { WasmFrameDecoder, softwareThreads } from '../video/wasm-decoder.ts';
import { measureFrame, type FrameMeasure } from '../video/measure.ts';
import { CELL_THRESHOLD, MEDIAN_SAMPLE_FRAMES, PRINT_FRAMES, PRINT_PARTS, PROXY_HEIGHT, PROXY_WIDTH } from '../pipeline/constants.ts';
import { CELLS, PIXELS, derive, type Analysis } from '../pipeline/analysis.ts';
import { yuv420RowsToBgr } from '../pipeline/colour.ts';
import { handSetup, printArea } from '../pipeline/hands.ts';
import { detectTurns, framesForHands, groupSpreads, mergeRepeats, type Hands, type Spread, type Turns } from '../pipeline/spreads.ts';
import { PrintSums, calmFrames, printLevel, printMap } from '../pipeline/printmap.ts';
import { printRegion, toCandidate } from '../pipeline/repeats.ts';
import { bandRows, pageBox } from '../ocr/audit.ts';
import type { ProxyChunk } from '../messages.ts';

const post = (message: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(message, transfer);

let analysis: Analysis | null = null;
let grouped: Spread[] = [];
let turnsFound: Turns | null = null;
let lightSums: Uint16Array | null = null; // per frame and cell, four times the mean brightness (spreads.cellLight)
let printFrames: number[] = [];

/** Per-pixel median of the sampled frames, as np.median(stack, axis=0).astype(np.uint8), in bands of rows. */
function medianFrame(samples: Uint8Array[]): Uint8Array {
  const W = PROXY_WIDTH, H = PROXY_HEIGHT, n = samples.length;
  const out = new Uint8Array(PIXELS * 3);
  const band = 10;
  const rows = new Uint8Array(band * W * 3);
  const values = new Uint8Array(band * W * 3 * n);
  const counts = new Uint16Array(256);
  const k1 = Math.floor((n - 1) / 2), k2 = Math.floor(n / 2);
  const ySize = W * H, cSize = ySize / 4;
  for (let r0 = 0; r0 < H; r0 += band) {
    const r1 = Math.min(H, r0 + band);
    const positions = (r1 - r0) * W * 3;
    samples.forEach((s, k) => {
      yuv420RowsToBgr(s.subarray(0, ySize), s.subarray(ySize, ySize + cSize), s.subarray(ySize + cSize), W, r0, r1, rows);
      for (let p = 0; p < positions; p++) values[p * n + k] = rows[p];
    });
    for (let p = 0; p < positions; p++) {
      counts.fill(0);
      for (let k = 0; k < n; k++) counts[values[p * n + k]]++;
      let seen = 0, a = -1, b = -1;
      for (let v = 0; v < 256; v++) {
        seen += counts[v];
        if (a < 0 && seen > k1) a = v;
        if (seen > k2) { b = v; break; }
      }
      out[r0 * W * 3 + p] = Math.floor((a + b) / 2);
    }
  }
  return out;
}

let debug = false;

async function pass1(file: File, track: VideoTrack, encoderCodec: string, ports: MessagePort[], software: number): Promise<void> {
  const started = performance.now();
  const n = track.samples.length;
  const fps = track.fps || 11;
  const detail = new Uint8Array(n * CELLS);
  const moved = new Uint8Array(n * CELLS);
  const light = new Uint16Array(n * CELLS);
  const medStep = Math.max(1, Math.floor(n / MEDIAN_SAMPLE_FRAMES));
  const samples: Uint8Array[] = [];
  let previous: Float32Array = new Float32Array(CELLS);
  let count = 0;
  let decoderInfo = '';
  let copySeconds = 0, shrinkSeconds = 0, measureSeconds = 0;

  // the reduced copy that the hand measurement reads back
  const chunks: ProxyChunk[] = [];
  let proxyConfig: VideoDecoderConfig | null = null;
  let encodeError: Error | null = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      if (meta?.decoderConfig && !proxyConfig) proxyConfig = { ...meta.decoderConfig };
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      chunks.push({ key: chunk.type === 'key', timestamp: chunk.timestamp, data });
    },
    error: e => { encodeError = e instanceof Error ? e : new Error(String(e)); },
  });
  encoder.configure({ codec: encoderCodec, width: PROXY_WIDTH, height: PROXY_HEIGHT, bitrate: 600_000, framerate: fps, latencyMode: 'quality' });
  const frameUs = 1e6 / fps;

  // Results come back from the helpers in any order and are taken in frame order.
  let consumed = 0;
  let failure: Error | null = null;
  const ready = new Map<number, FrameMeasure>();
  let consuming: Promise<void> = Promise.resolve();
  const consume = async (r: FrameMeasure) => {
    const index = r.index;
    copySeconds += r.seconds[0]; shrinkSeconds += r.seconds[1]; measureSeconds += r.seconds[2];
    if (index > 0) {
      const o = index * CELLS;
      for (let c = 0; c < CELLS; c++) moved[o + c] = Math.abs(r.blocks[c] - previous[c]) > CELL_THRESHOLD ? 1 : 0;
    }
    previous = r.blocks;
    detail.set(r.detail, index * CELLS);
    for (let c = 0; c < CELLS; c++) light[index * CELLS + c] = Math.round(r.blocks[c] * 4);
    if (index % medStep === 0) samples.push(r.i420);
    const vf = new VideoFrame(r.i420, {
      format: 'I420', codedWidth: PROXY_WIDTH, codedHeight: PROXY_HEIGHT,
      timestamp: Math.round(index * frameUs), duration: Math.round(frameUs),
    });
    while (encoder.encodeQueueSize > 8) await new Promise(res => setTimeout(res, 1));
    if (encodeError) throw encodeError;
    encoder.encode(vf, { keyFrame: index % 11 === 0 });
    vf.close();
    if (index % 25 === 0) post({ kind: 'progress', done: index, total: n });
  };
  const arrived = (r: FrameMeasure) => {
    ready.set(r.index, r);
    consuming = consuming.then(async () => {
      while (!failure && ready.has(consumed)) {
        const next = ready.get(consumed)!;
        ready.delete(consumed);
        try { await consume(next); } catch (e) { failure = e instanceof Error ? e : new Error(String(e)); }
        consumed++;
      }
      wake();
    });
  };
  let inflight = 0;
  let waiter: (() => void) | null = null;
  const wake = () => { const w = waiter; waiter = null; w?.(); };
  const limit = Math.max(2, ports.length * 2);
  let local = ports.length === 0;
  for (const p of ports) {
    p.onmessage = (m: MessageEvent) => {
      inflight--;
      if (m.data.kind === 'error') { failure = new Error(m.data.message); wake(); return; }
      arrived(m.data.result as FrameMeasure);
      wake();
    };
  }

  const onFrame = async (frame: VideoFrame) => {
    if (failure) throw failure;
    const index = count++;
    if (index >= n) return;
    if (!decoderInfo || decoderInfo.endsWith('; ')) decoderInfo += `${frame.format} ${frame.visibleRect?.width ?? frame.codedWidth}x${frame.visibleRect?.height ?? frame.codedHeight}`;
    if (local) { arrived(await measureFrame(frame, index)); await consuming; return; }
    while (inflight >= limit || ready.size > 2 * limit) {
      await new Promise<void>(res => { waiter = res; setTimeout(res, 20); });
      if (failure) throw failure;
    }
    try {
      ports[index % ports.length].postMessage({ frame, index }, [frame as unknown as Transferable]);
      inflight++;
    } catch {
      // this browser cannot hand frames to another worker: measure them here
      local = true;
      while (inflight > 0 && !failure) await new Promise<void>(res => { waiter = res; setTimeout(res, 20); });
      arrived(await measureFrame(frame, index));
      await consuming;
    }
  };

  const decoder: Decoder = software
    ? await WasmFrameDecoder.open(track.description, softwareThreads(software), onFrame)
    : new FrameDecoder({
      codec: track.codec, codedWidth: track.codedWidth, codedHeight: track.codedHeight, description: track.description,
      hardwareAcceleration: 'no-preference',
    }, onFrame);
  if (software) decoderInfo = `built-in, ${softwareThreads(software)} threads; `;
  let readSeconds = 0;
  try {
    const BATCH = 64;
    for (let d = 0; d < n; d += BATCH) {
      const last = Math.min(n - 1, d + BATCH - 1);
      const t = performance.now();
      const datas = await readSamples(file, track, d, last);
      readSeconds += performance.now() - t;
      for (let k = d; k <= last; k++) {
        const s = track.samples[k];
        await decoder.decode({ key: s.sync, timestamp: sampleTimestamp(track, s), data: datas[k - d] });
      }
    }
    await decoder.flush();
    while (!failure && consumed < Math.min(count, n)) await new Promise<void>(res => { waiter = res; setTimeout(res, 20); });
    await consuming;
    if (failure) throw failure;
  } finally {
    decoder.close();
  }
  await encoder.flush();
  encoder.close();
  if (encodeError) throw encodeError;
  const frames = Math.min(count, n);
  const decodeSeconds = (performance.now() - started) / 1000;
  const t = performance.now();
  const median = medianFrame(samples);
  const medianSeconds = (performance.now() - t) / 1000;
  const t3 = performance.now();
  analysis = derive(fps, frames, detail.subarray(0, frames * CELLS), moved.subarray(0, frames * CELLS), median);
  const setup = handSetup(analysis, { lut: new Uint8Array(0), samples: 0, usable: false });
  const region = printRegion(analysis.mask, analysis.median);
  lightSums = light.subarray(0, frames * CELLS);
  // the page turns need no hands: found now, they say which frames the hand models must look at
  turnsFound = detectTurns(analysis);
  const needed = framesForHands(turnsFound, analysis);
  const cells = printArea(analysis);
  printFrames = cells.some(v => v) ? calmFrames(analysis, PRINT_FRAMES * PRINT_PARTS) : [];
  const deriveSeconds = (performance.now() - t3) / 1000;
  post({
    kind: 'pass1-done', frames, fps,
    debug: debug ? { detail: analysis.detail.slice(), moved: analysis.moved.slice(), median, mask: analysis.mask } : null,
    proxy: { config: proxyConfig, chunks, fps },
    page: { median, mask: analysis.mask, inner: analysis.inner, text: cells, colour: setup.colour },
    region, needed, printFrames,
    pageCells: analysis.gridPage.reduce((s, v) => s + v, 0),
    seconds: { decode: decodeSeconds, ...(decoder instanceof WasmFrameDecoder ? { 'built-in decoder': decoder.spent } : {}), read: readSeconds / 1000, copy: copySeconds, shrink: shrinkSeconds, measure: measureSeconds, median: medianSeconds, derive: deriveSeconds },
    decoderInfo, medianSamples: samples.length,
  }, chunks.map(c => c.data.buffer as ArrayBuffer));
}

/** print_maps: the print map from the hands workers' sums (in the order of their stretches) and the fine detail of each frame at rest. */
function makePrintMap(parts: { seen: Int32Array; printed: Int32Array; light: Float32Array; light2: Float32Array; frames: number }[], medians: number[]): void {
  if (!analysis) throw new Error('no analysis');
  const cells = printArea(analysis);
  if (!printFrames.length || !medians.length) { post({ kind: 'printmap-done', map: cells, level: 0 }); return; }
  const sums = new PrintSums();
  for (const part of parts) sums.merge(part);
  post({ kind: 'printmap-done', map: printMap(analysis, cells, sums), level: printLevel(medians) });
}

function turns(hands: Hands): void {
  if (!analysis || !turnsFound) throw new Error('no analysis');
  const t = turnsFound;
  grouped = groupSpreads(t, analysis, hands, lightSums);
  post({ kind: 'turns-done', spreads: grouped, landings: t.landings.length, landingFrames: t.landings });
}

function merge(grays: Record<number, Uint8Array>): void {
  if (!analysis) throw new Error('no analysis');
  const { merged, dropped } = mergeRepeats(grouped, analysis, f => {
    const g = grays[f];
    if (!g) throw new Error(`no picture for frame ${f}`);
    return g;
  });
  // where the page numbers would be read, should the visitor ask (export.band_rows and page_box, over the candidates)
  const rows = Object.fromEntries(bandRows(analysis, merged.map(s => s.best)));
  post({ kind: 'merge-done', spreads: merged, candidates: merged.map(s => toCandidate(s, analysis!.fps)), dropped, rows, box: pageBox(analysis.mask) });
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.kind === 'pass1') { debug = !!m.debug; await pass1(m.file, m.track, m.encoderCodec, m.ports ?? [], m.software ?? 0); }
    else if (m.kind === 'printmap') makePrintMap(m.parts, m.medians);
    else if (m.kind === 'turns') turns(m.hands);
    else if (m.kind === 'merge') merge(m.grays);
  } catch (err) {
    post({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
