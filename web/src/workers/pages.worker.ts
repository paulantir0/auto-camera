// A pages worker: from the full-size pictures to flat page images
// (src/pages/pages.ts, with OpenCV.js), and then each page set on the book's
// usual paper size (book.uniform_pages of the Python pipeline).

import { pageImages, type CV } from '../pages/pages.ts';
import { setJpegDpi } from '../pages/pdf.ts';

declare function importScripts(...urls: string[]): void;
const post = (message: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(message, transfer);

export const JPEG_QUALITY = 0.88;        // book.JPEG_QUALITY (88)
export const PAGE_HEIGHT_INCHES = 9.0;   // book.PAGE_HEIGHT_INCHES: the PDF shows a page this tall

let cv: CV = null;

async function loadOpenCv(): Promise<void> {
  importScripts('opencv.js');
  let m = (self as unknown as { cv: CV }).cv;
  if (m instanceof Promise) m = await m;
  else if (!m.Mat) await new Promise<void>(resolve => { m.onRuntimeInitialized = () => resolve(); });
  cv = m;
}

async function decode(blob: Blob): Promise<ImageData> {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

async function encode(image: ImageData, quality = JPEG_QUALITY): Promise<Blob> {
  const canvas = new OffscreenCanvas(image.width, image.height);
  canvas.getContext('2d')!.putImageData(image, 0, 0);
  return canvas.convertToBlob({ type: 'image/jpeg', quality });
}

interface Region { data: Uint8Array; width: number; height: number }

async function pages(jobs: { index: number; image: Blob }[], region: Region | null): Promise<void> {
  for (const job of jobs) {
    const pixels = await decode(job.image);
    const mat = cv.matFromImageData(pixels);
    try {
      const found = pageImages(cv, mat, region);
      for (const page of found) {
        const image = new ImageData(new Uint8ClampedArray(page.image.data), page.image.cols, page.image.rows);
        const width = page.image.cols, height = page.image.rows;
        page.image.delete();
        post({ kind: 'page', index: job.index, side: page.side, lines: page.lines, unrolled: page.unrolled, width, height, jpeg: await encode(image) });
      }
    } finally {
      mat.delete();
    }
    post({ kind: 'picture-done', index: job.index });
  }
  post({ kind: 'pages-done' });
}

/**
 * uniform_pages for one page: on paper (the colour of its top left corner)
 * of the book's usual shape, at least its usual size, the page at the top in
 * the middle; resolution such that it prints PAGE_HEIGHT_INCHES tall.
 */
async function uniform(jobs: { key: number; jpeg: Blob; width: number; height: number }[], usualW: number, usualH: number): Promise<void> {
  const shape = usualW / usualH;
  for (const job of jobs) {
    const { width: w, height: h } = job;
    let cw = Math.max(w, usualW), ch = Math.max(h, usualH);
    if (cw / ch < shape) cw = Math.round(ch * shape); else ch = Math.round(cw / shape);
    const page = await decode(job.jpeg);
    let sheet = page;
    if (cw !== w || ch !== h) {
      // the paper's colour: the median of the page's top left corner (8 x 8)
      const values: number[][] = [[], [], []];
      for (let y = 0; y < Math.min(8, h); y++) for (let x = 0; x < Math.min(8, w); x++) for (let c = 0; c < 3; c++) values[c].push(page.data[(y * w + x) * 4 + c]);
      const paper = values.map(v => { v.sort((a, b) => a - b); const n = v.length; return Math.trunc(n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2); });
      const canvas = new OffscreenCanvas(cw, ch);
      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = `rgb(${paper[0]}, ${paper[1]}, ${paper[2]})`;
      ctx.fillRect(0, 0, cw, ch);
      ctx.putImageData(page, Math.floor((cw - w) / 2), 0);
      sheet = ctx.getImageData(0, 0, cw, ch);
    }
    const blob = await encode(sheet);
    const jpeg = setJpegDpi(new Uint8Array(await blob.arrayBuffer()), ch / PAGE_HEIGHT_INCHES);
    post({ kind: 'uniform-page', key: job.key, jpeg: new Blob([jpeg as BlobPart], { type: 'image/jpeg' }), width: cw, height: ch });
  }
  post({ kind: 'uniform-done' });
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.kind === 'load') { await loadOpenCv(); post({ kind: 'ready' }); }
    else if (m.kind === 'pages') await pages(m.jobs, m.region);
    else if (m.kind === 'uniform') await uniform(m.jobs, m.usualW, m.usualH);
  } catch (err) {
    post({ kind: 'error', message: err instanceof Error ? `${err.message}` : String(err) });
  }
};
