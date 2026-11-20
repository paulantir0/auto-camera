// The OCR worker: reads the printed page numbers off the pictures with the
// same models and steps as the Python audit (src/ocr/). Loaded only when the
// visitor asks for the page-number check. Its files (onnxruntime-web, the two
// PP-OCRv4 models and the recogniser's dictionary) are served from ocr/.

import * as ort from 'onnxruntime-web/wasm';
import { OcrEngine, type Image, type Session } from '../ocr/engine.ts';
import { readStill, type Still } from '../ocr/read.ts';

declare function importScripts(...urls: string[]): void;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CV = any;

const post = (message: unknown) => (self as unknown as Worker).postMessage(message);
let engine: OcrEngine | null = null;

async function loadOpenCv(): Promise<CV> {
  importScripts('opencv.js');
  let m = (self as unknown as { cv: CV }).cv;
  if (m instanceof Promise) m = await m;
  else if (!m.Mat) await new Promise<void>(resolve => { m.onRuntimeInitialized = () => resolve(); });
  return m;
}

async function init(threads: number): Promise<void> {
  const base = new URL('../ocr/', self.location.href).href;
  ort.env.wasm.wasmPaths = base;
  ort.env.wasm.numThreads = (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated ? threads : 1;
  ort.env.wasm.proxy = false;
  const options: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  const [cv, det, rec, keys] = await Promise.all([
    loadOpenCv(),
    ort.InferenceSession.create(`${base}det.onnx`, options),
    ort.InferenceSession.create(`${base}rec.onnx`, options),
    fetch(`${base}keys.txt`).then(r => { if (!r.ok) throw new Error('could not load the OCR dictionary'); return r.text(); }),
  ]);
  const wrap = (s: ort.InferenceSession): Session => ({
    run: async (input, dims) => {
      const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', input, dims) });
      const t = out[s.outputNames[0]];
      return { data: t.data as Float32Array, dims: t.dims };
    },
  });
  const lines = keys.split('\n');
  if (lines[lines.length - 1] === '') lines.pop();
  engine = new OcrEngine(cv, wrap(det), wrap(rec), lines);
}

/** A JPEG still, decoded once; bands are copied out of it as BGR. */
async function stillFromBlob(blob: Blob): Promise<{ still: Still; close(): void }> {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const canvas = new OffscreenCanvas(1, 1);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  return {
    still: {
      width: bitmap.width,
      height: bitmap.height,
      region(y0, y1, x0, x1): Image {
        const w = x1 - x0, h = y1 - y0;
        canvas.width = w;
        canvas.height = h;
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(bitmap, x0, y0, w, h, 0, 0, w, h);
        const rgba = ctx.getImageData(0, 0, w, h).data;
        const bgr = new Uint8Array(w * h * 3);
        for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) { bgr[j] = rgba[i + 2]; bgr[j + 1] = rgba[i + 1]; bgr[j + 2] = rgba[i]; }
        return { data: bgr, width: w, height: h };
      },
    },
    close: () => bitmap.close(),
  };
}

export interface OcrJob { frame: number; image: Blob; rows: number[] | null }

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.kind === 'init') {
      await init(m.threads ?? 1);
      post({ kind: 'ready' });
    } else if (m.kind === 'read') {
      if (!engine) throw new Error('the reader is not ready');
      const started = performance.now();
      for (const job of m.jobs as OcrJob[]) {
        const { still, close } = await stillFromBlob(job.image);
        try {
          const tokens = await readStill(engine, still, m.box as number[], job.rows);
          post({ kind: 'tokens', frame: job.frame, tokens });
        } finally {
          close();
        }
      }
      post({ kind: 'read-done', seconds: (performance.now() - started) / 1000, spent: engine.spent });
    }
  } catch (err) {
    post({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
