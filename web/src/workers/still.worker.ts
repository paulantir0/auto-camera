// A still worker: seeks to the chosen frames in the original video, and for
// each one makes the full-size JPEG, a small preview, the 480x270 gray image
// the early merge compares, and the ORB keypoints the repeat check matches
// (with OpenCV.js, the same ORB as the Python side).

import { readSamples, sampleTimestamp, type VideoTrack } from '../video/mp4.ts';
import { FrameDecoder, type Decoder } from '../video/decode.ts';
import { WasmFrameDecoder, softwareThreads } from '../video/wasm-decoder.ts';
import { Shrinker, readFrame } from '../video/sampler.ts';
import { GRAY_FROM_Y } from '../pipeline/colour.ts';
import { ORB_FAST_THRESHOLD, ORB_FEATURES, PROXY_HEIGHT, PROXY_WIDTH } from '../pipeline/constants.ts';
import { lutAreaFromPlane, resizeNearest } from '../pipeline/imgproc.ts';
import { matchSize } from '../pipeline/repeats.ts';
import { crc32 } from '../app/crc32.ts';
import type { StillResult, StillTarget } from '../messages.ts';

declare function importScripts(...urls: string[]): void;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CV = any;

const post = (message: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(message, transfer);

let cv: CV = null;
let file: File;
let track: VideoTrack;
let jpegQuality = 0.92;
let previewWidth = 360;
let useStore = false;
let software = 0; // threads of the built-in decoder, or 0 for the browser's
let maskMat: CV = null;
let imgMat: CV = null;
let orb: CV = null;
let mw = 0, mh = 0;
let shrinker: Shrinker | null = null;
const buf = { data: new Uint8Array(0) };
let full: OffscreenCanvas | null = null;
let small: OffscreenCanvas | null = null;
let store: FileSystemDirectoryHandle | null = null;
// seconds spent per step, reported with frames-done
const spent = { decode: 0, draw: 0, planes: 0, gray: 0, keypoints: 0, jpeg: 0, store: 0 };
let lastMark = 0;
const lap = (name: keyof typeof spent) => { const now = performance.now(); spent[name] += (now - lastMark) / 1000; lastMark = now; };

async function loadOpenCv(): Promise<void> {
  importScripts('opencv.js');
  let m = (self as unknown as { cv: CV }).cv;
  if (m instanceof Promise) m = await m;
  else if (!m.Mat) await new Promise<void>(resolve => { m.onRuntimeInitialized = () => resolve(); });
  cv = m;
}

function features(gray: Uint8Array): { xs: Float32Array; ys: Float32Array; desc: Uint8Array } {
  imgMat.data.set(gray);
  const kp = new cv.KeyPointVector();
  const desc = new cv.Mat();
  try {
    orb.detectAndCompute(imgMat, maskMat, kp, desc);
    const n = kp.size();
    const xs = new Float32Array(n), ys = new Float32Array(n);
    for (let i = 0; i < n; i++) { const p = kp.get(i); xs[i] = p.pt.x; ys[i] = p.pt.y; }
    const d = n ? new Uint8Array(desc.data.slice(0, n * 32)) : new Uint8Array(0);
    return { xs, ys, desc: d };
  } finally {
    kp.delete();
    desc.delete();
  }
}

async function saveToStore(name: string, bytes: Uint8Array): Promise<string | null> {
  if (!store) return null;
  try {
    const handle = await store.getFileHandle(name, { create: true });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const access = await (handle as any).createSyncAccessHandle();
    access.truncate(0);
    access.write(bytes, { at: 0 });
    access.flush();
    access.close();
    return name;
  } catch {
    return null;
  }
}

async function onTarget(frame: VideoFrame, index: number): Promise<StillResult> {
  const width = frame.visibleRect?.width ?? frame.codedWidth, height = frame.visibleRect?.height ?? frame.codedHeight;
  if (!shrinker || shrinker.width !== width || shrinker.height !== height) shrinker = new Shrinker(width, height);
  // full-size and preview JPEGs, drawn before the planes are read so the frame can go soon after
  if (!full || full.width !== width || full.height !== height) full = new OffscreenCanvas(width, height);
  lap('decode');
  const fctx = full.getContext('2d')!;
  fctx.drawImage(frame, 0, 0);
  const jpegPromise = full.convertToBlob({ type: 'image/jpeg', quality: jpegQuality });
  const ph = Math.round(height * previewWidth / width);
  if (!small || small.width !== previewWidth || small.height !== ph) small = new OffscreenCanvas(previewWidth, ph);
  const sctx = small.getContext('2d')!;
  sctx.imageSmoothingQuality = 'high';
  sctx.drawImage(frame, 0, 0, previewWidth, ph);
  const previewPromise = small.convertToBlob({ type: 'image/jpeg', quality: 0.82 });
  // the planes: proxy-style gray for the merge, full gray for the keypoints
  lap('draw');
  const { format, layout } = await readFrame(frame, buf);
  lap('planes');
  shrinker.shrink(buf.data, format, layout, false);
  const gray = shrinker.gray.slice();
  const [w2, h2] = matchSize(width, height);
  if (w2 !== mw || h2 !== mh) throw new Error('frame size changed');
  const small2 = lutAreaFromPlane(buf.data, layout[0].offset, layout[0].stride, width, height, GRAY_FROM_Y, w2, h2);
  lap('gray');
  const feats = features(small2);
  lap('keypoints');
  const jpeg = await jpegPromise;
  const bytes = new Uint8Array(await jpeg.arrayBuffer());
  const crc = crc32(bytes);
  lap('jpeg');
  const stored = useStore ? await saveToStore(`f${String(index).padStart(6, '0')}.jpg`, bytes) : null;
  lap('store');
  return { frame: index, gray, features: feats, jpeg: stored ? null : jpeg, stored, size: bytes.length, crc, preview: await previewPromise, width, height };
}

async function run(targets: StillTarget[]): Promise<void> {
  lastMark = performance.now();
  const wanted = new Set(targets.map(t => t.frame));
  // presentation index of each sample's timestamp
  const indexOf = new Map<number, number>();
  const groups = new Map<number, number>(); // start -> last decode index needed
  for (const t of targets) {
    const d = track.presentation[t.frame];
    groups.set(t.start, Math.max(groups.get(t.start) ?? 0, d));
  }
  const handler = async (frame: VideoFrame) => {
    const index = indexOf.get(frame.timestamp);
    if (index === undefined || !wanted.has(index)) return;
    wanted.delete(index);
    const result = await onTarget(frame, index);
    post({ kind: 'still', result }, [result.gray.buffer, result.features.xs.buffer, result.features.ys.buffer, result.features.desc.buffer]);
  };
  const decoder: Decoder = software
    ? await WasmFrameDecoder.open(track.description, softwareThreads(software), handler)
    : new FrameDecoder({
      codec: track.codec, codedWidth: track.codedWidth, codedHeight: track.codedHeight, description: track.description,
      hardwareAcceleration: 'no-preference',
    }, handler, { maxQueue: 4, maxFrames: 2 });
  try {
    for (const [start, end] of [...groups.entries()].sort((a, b) => a[0] - b[0])) {
      const datas = await readSamples(file, track, start, end);
      for (let k = start; k <= end; k++) {
        const s = track.samples[k];
        const ts = sampleTimestamp(track, s);
        indexOf.set(ts, track.decodeToPresentation[k]);
        await decoder.decode({ key: s.sync, timestamp: ts, data: datas[k - start] });
      }
      await decoder.flush();
    }
  } finally {
    decoder.close();
  }
  if (wanted.size) throw new Error(`could not decode frames ${[...wanted].slice(0, 5).join(', ')}`);
  post({ kind: 'frames-done', spent });
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.kind === 'load') {
      await loadOpenCv();
      post({ kind: 'ready' });
    } else if (m.kind === 'init') {
      file = m.file; track = m.track; jpegQuality = m.jpegQuality; previewWidth = m.previewWidth; useStore = m.useStore; software = m.software ?? 0;
      if (!cv) await loadOpenCv();
      [mw, mh] = matchSize(track.codedWidth, track.codedHeight);
      const region = (m.region as Uint8Array).map(v => (v ? 255 : 0));
      maskMat?.delete(); imgMat?.delete(); orb?.delete();
      maskMat = new cv.Mat(mh, mw, cv.CV_8UC1);
      maskMat.data.set(resizeNearest(region, PROXY_WIDTH, PROXY_HEIGHT, mw, mh));
      imgMat = new cv.Mat(mh, mw, cv.CV_8UC1);
      orb = new cv.ORB(ORB_FEATURES, 1.2, 8, 31, 0, 2, cv.ORB_HARRIS_SCORE, 31, ORB_FAST_THRESHOLD);
      if (useStore) {
        try {
          const root = await navigator.storage.getDirectory();
          store = await root.getDirectoryHandle(m.storeDir, { create: true });
        } catch { store = null; }
      }
      post({ kind: 'ready' });
    } else if (m.kind === 'frames') await run(m.targets);
  } catch (err) {
    post({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
