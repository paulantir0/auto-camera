// The book from the pictures, on request: flat page images (pages workers,
// src/pages/pages.ts), each set on the book's usual paper size, read by
// Tesseract into one searchable PDF page each, and the pages joined into
// one PDF (auto_camera/book.py of the Python pipeline), with the text beside it.
// A page that shows the same printed page as one near it, and reads worse,
// is left out (src/pages/textrepeats.ts).

import { createWorker, OEM, PSM, type Worker as TesseractWorker } from 'tesseract.js';
import { joinPdfs } from '../pages/pdf.ts';
import { repeatedPages } from '../pages/textrepeats.ts';

/** book.OCR_LANGUAGES: the languages the books here are in. */
export const OCR_LANGUAGES = ['eng', 'fra', 'deu', 'lat', 'grc', 'ell', 'ita'];

export interface BookPage { name: string; picture: number; side: 'left' | 'right'; lines: number; unrolled: boolean; jpeg: Blob; width: number; height: number }
/** A page left out of the book: it shows the same printed page as `sameAs`, which reads better. */
export interface LeftOut { page: BookPage; sameAs: BookPage; share: number }
export interface Book { pages: BookPage[]; leftOut: LeftOut[]; pdf: Blob; text: Blob; seconds: Record<string, number> }
export type BookStep = 'loading' | 'pages' | 'reading' | 'joining';
export interface Region { data: Uint8Array; width: number; height: number }

export class BookMaker {
  private readonly workers = new Set<Worker>();
  private readonly readers: TesseractWorker[] = [];
  private stopped = false;
  private readonly pending = new Set<(e: Error) => void>();
  readonly seconds: Record<string, number> = {};

  stop(): void {
    this.stopped = true;
    for (const w of this.workers) w.terminate();
    this.workers.clear();
    for (const r of this.readers) void r.terminate().catch(() => undefined);
    this.readers.length = 0;
    for (const reject of this.pending) reject(new Error('stopped'));
    this.pending.clear();
  }

  private check(): void { if (this.stopped) throw new Error('stopped'); }

  /** Send a message and wait for the reply of kind `done`, passing others to `onMessage`. */
  private ask<T>(w: Worker, message: unknown, done: string, onMessage?: (m: { kind: string; [k: string]: unknown }) => void): Promise<T> {
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
      w.postMessage(message);
    });
  }

  async run(pictures: { full: Blob; flags?: string[] }[], region: Region | null, title: string, onProgress: (step: BookStep, done: number, total: number) => void): Promise<Book> {
    const started = performance.now();
    const cores = navigator.hardwareConcurrency || 4;
    const memory = (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 4;
    // 1. flat pages, a picture at a time from a shared queue
    onProgress('loading', 0, 1);
    const count = Math.max(1, Math.min(memory <= 4 ? 1 : 3, Math.floor(cores / 3)));
    const pageWorkers = Array.from({ length: count }, () => { const w = new Worker('workers/pages.js'); this.workers.add(w); return w; });
    await Promise.all(pageWorkers.map(w => this.ask(w, { kind: 'load' }, 'ready')));
    this.check();
    const found = new Map<number, Omit<BookPage, 'name'>[]>();
    let next = 0, done = 0;
    onProgress('pages', 0, pictures.length);
    await Promise.all(pageWorkers.map(async w => {
      while (next < pictures.length) {
        const index = next++;
        await this.ask(w, { kind: 'pages', jobs: [{ index, image: pictures[index].full }], region }, 'pages-done', m => {
          if (m.kind === 'page') {
            const list = found.get(index) ?? [];
            list.push({ picture: index, side: m.side as 'left' | 'right', lines: m.lines as number, unrolled: m.unrolled as boolean, jpeg: m.jpeg as Blob, width: m.width as number, height: m.height as number });
            found.set(index, list);
          }
        });
        this.check();
        onProgress('pages', ++done, pictures.length);
      }
    }));
    const raw = [...found.keys()].sort((a, b) => a - b).flatMap(k => found.get(k)!);
    if (!raw.length) throw new Error('no page could be cut out of the pictures');
    // 2. every page on the book's usual paper (uniform_pages)
    const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b), n = s.length; return Math.trunc(n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2); };
    const usualW = median(raw.map(p => p.width)), usualH = median(raw.map(p => p.height));
    const pages: BookPage[] = raw.map((p, k) => ({ ...p, name: `page_${String(k + 1).padStart(4, '0')}.jpg` }));
    let k2 = 0;
    await Promise.all(pageWorkers.map(async w => {
      while (k2 < pages.length) {
        const key = k2++;
        const p = pages[key];
        await this.ask(w, { kind: 'uniform', jobs: [{ key, jpeg: p.jpeg, width: p.width, height: p.height }], usualW, usualH }, 'uniform-done', m => {
          if (m.kind === 'uniform-page') { p.jpeg = m.jpeg as Blob; p.width = m.width as number; p.height = m.height as number; }
        });
        this.check();
      }
    }));
    for (const w of pageWorkers) { w.terminate(); this.workers.delete(w); }
    this.seconds.pages = (performance.now() - started) / 1000;
    // 3. each page read into a searchable one-page PDF, as `tesseract PAGE OUT -l LANGS pdf txt`
    const readStarted = performance.now();
    onProgress('loading', 0, 1);
    const readers = Math.max(1, Math.min(memory <= 4 ? 2 : 4, cores - 1, pages.length));
    const base = new URL('tesseract/', location.href).href;
    for (let r = 0; r < readers; r++) {
      const reader = await createWorker(OCR_LANGUAGES, OEM.LSTM_ONLY, {
        workerPath: `${base}worker.min.js`, corePath: base, langPath: `${base}lang`, gzip: false, cacheMethod: 'none', workerBlobURL: false,
      });
      this.check();
      // the command line reads a page with automatic page segmentation; the library's default is a single block
      await reader.setParameters({ tessedit_pageseg_mode: PSM.AUTO });
      this.readers.push(reader);
    }
    const pdfs: Blob[] = new Array(pages.length);
    const texts: string[] = new Array(pages.length);
    let n = 0, read = 0;
    onProgress('reading', 0, pages.length);
    await Promise.all(this.readers.map(async reader => {
      while (n < pages.length) {
        const k = n++;
        const result = await reader.recognize(pages[k].jpeg, {}, { pdf: true, text: true });
        this.check();
        const bytes = result.data.pdf as unknown as ArrayLike<number> | null;
        if (!bytes) throw new Error('Tesseract gave no PDF');
        pdfs[k] = new Blob([Uint8Array.from(bytes) as BlobPart], { type: 'application/pdf' });
        texts[k] = result.data.text ?? '';
        onProgress('reading', ++read, pages.length);
      }
    }));
    for (const r of this.readers.splice(0)) await r.terminate();
    this.seconds.reading = (performance.now() - readStarted) / 1000;
    // 4. pages that repeat a page near them left out, and one PDF
    onProgress('joining', 0, 1);
    const repeats = repeatedPages(texts, pages.map(p => p.picture), pages.map(p => pictures[p.picture].flags?.length ?? 0));
    const drop = new Set(repeats.map(r => r.page));
    const leftOut = repeats.map(r => ({ page: pages[r.page], sameAs: pages[r.kept], share: r.share }));
    const keep = (_: unknown, k: number) => !drop.has(k);
    const pdf = await joinPdfs(pdfs.filter(keep), title);
    const text = new Blob([texts.filter(keep).join('\f')], { type: 'text/plain;charset=utf-8' });
    this.seconds.total = (performance.now() - started) / 1000;
    return { pages: pages.filter(keep), leftOut, pdf, text, seconds: this.seconds };
  }
}
