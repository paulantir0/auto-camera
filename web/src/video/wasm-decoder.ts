// HEVC decoding in WebAssembly, for browsers whose WebCodecs has no HEVC
// decoder (Chrome and Firefox on Linux, for example). It is FFmpeg's own HEVC
// decoder (web/wasm/hevcdec.c, built by tools/build-hevc.sh), several times
// slower than a hardware decoder but exact. It hands out the same VideoFrame
// objects as WebCodecs does, so everything after it is unchanged.
//
// Calls into the decoder block this worker while a picture is decoded, which
// is why it only ever runs in workers. With threads > 1 FFmpeg decodes several
// pictures at once on WebAssembly threads (it needs a cross-origin-isolated
// page, which the site's headers give).

import type { Chunk, Decoder } from './decode.ts';

declare function importScripts(...urls: string[]): void;

interface HevcModule {
  _dec_open(extra: number, size: number, threads: number): number;
  _dec_send(d: number, data: number, size: number, pts: number, key: number): number;
  _dec_receive(d: number): number;
  _dec_reset(d: number): void;
  _dec_picture(d: number): number;
  _dec_width(d: number): number;
  _dec_height(d: number): number;
  _dec_pts(d: number): number;
  _dec_matrix(d: number): number;
  _dec_full_range(d: number): number;
  _dec_close(d: number): void;
  _dec_alloc(size: number): number;
  _dec_free(p: number): void;
  wasmMemory?: WebAssembly.Memory;
  HEAPU8: Uint8Array;
}

type Factory = (options: Record<string, unknown>) => Promise<HevcModule>;

/** Whether this page could decode HEVC in WebAssembly at all. */
export function softwareHevcPossible(): boolean {
  return typeof WebAssembly === 'object' && typeof VideoFrame === 'function';
}

/** Threads FFmpeg may use here: several only where shared memory is allowed. */
export function softwareThreads(wanted: number): number {
  return (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated && typeof SharedArrayBuffer === 'function'
    ? Math.max(1, Math.min(8, wanted)) : 1;
}

async function load(threads: number): Promise<HevcModule> {
  const g = self as unknown as { createHevcDecoder?: Factory };
  if (!g.createHevcDecoder) importScripts(threads > 1 ? 'hevc-mt.js' : 'hevc.js');
  return g.createHevcDecoder!({
    threads,
    // the threads' own workers load the same script as this one did
    mainScriptUrlOrBlob: threads > 1 ? 'hevc-mt.js' : 'hevc.js',
    print: () => { /* quiet */ },
    printErr: () => { /* quiet */ },
  });
}

const MATRIX: Record<number, VideoMatrixCoefficients> = { 1: 'bt709', 5: 'bt470bg', 6: 'smpte170m', 9: 'bt2020-ncl' as VideoMatrixCoefficients };

export class WasmFrameDecoder implements Decoder {
  private mod: HevcModule | null = null;
  private dec = 0;
  private input = 0;
  private inputSize = 0;
  private failure: Error | null = null;
  private readonly handler: (frame: VideoFrame) => Promise<void> | void;
  /** Seconds spent inside the decoder. */
  spent = 0;

  private constructor(handler: (frame: VideoFrame) => Promise<void> | void) {
    this.handler = handler;
  }

  static async open(description: Uint8Array | undefined, threads: number, handler: (frame: VideoFrame) => Promise<void> | void): Promise<WasmFrameDecoder> {
    const d = new WasmFrameDecoder(handler);
    const mod = await load(threads);
    d.mod = mod;
    const extra = description ?? new Uint8Array(0);
    const p = extra.length ? mod._dec_alloc(extra.length) : 0;
    if (extra.length) d.heap().set(extra, p);
    d.dec = mod._dec_open(p, extra.length, threads);
    if (p) mod._dec_free(p);
    if (!d.dec) throw new Error('the built-in HEVC decoder could not start');
    return d;
  }

  /** A view of the module's memory as it is now (it may have grown since). */
  private heap(): Uint8Array {
    const m = this.mod!;
    return m.wasmMemory ? new Uint8Array(m.wasmMemory.buffer) : m.HEAPU8;
  }

  private check(): void {
    if (this.failure) throw this.failure;
  }

  /** Hand out every picture the decoder has ready; true when it has said it has no more (after a flush). */
  private async drain(): Promise<boolean> {
    const m = this.mod!;
    for (;;) {
      const t = performance.now();
      const r = m._dec_receive(this.dec);
      this.spent += (performance.now() - t) / 1000;
      if (r === 1) return false;
      if (r === 2) return true;
      if (r < 0) throw new Error(`the built-in HEVC decoder failed (${r})`);
      const w = m._dec_width(this.dec), h = m._dec_height(this.dec);
      const size = w * h + 2 * ((w + 1) >> 1) * ((h + 1) >> 1);
      const p = m._dec_picture(this.dec);
      // VideoFrame wants memory of its own, not the decoder's shared memory
      const copy = new Uint8Array(size);
      copy.set(this.heap().subarray(p, p + size));
      const matrix = MATRIX[m._dec_matrix(this.dec)];
      const frame = new VideoFrame(copy, {
        format: 'I420', codedWidth: w, codedHeight: h, timestamp: m._dec_pts(this.dec),
        colorSpace: { matrix: matrix ?? 'bt709', primaries: 'bt709', transfer: 'bt709', fullRange: m._dec_full_range(this.dec) === 1 },
      });
      try {
        await this.handler(frame);
      } finally {
        frame.close();
      }
    }
  }

  async decode(chunk: Chunk): Promise<void> {
    this.check();
    const m = this.mod!;
    try {
      if (chunk.data.length > this.inputSize) {
        if (this.input) m._dec_free(this.input);
        this.inputSize = Math.max(chunk.data.length, 2 * this.inputSize);
        this.input = m._dec_alloc(this.inputSize);
        if (!this.input) throw new Error('out of memory');
      }
      this.heap().set(chunk.data, this.input);
      for (;;) {
        const t = performance.now();
        const r = m._dec_send(this.dec, this.input, chunk.data.length, chunk.timestamp, chunk.key ? 1 : 0);
        this.spent += (performance.now() - t) / 1000;
        if (r < 0) throw new Error(`the built-in HEVC decoder rejected a frame (${r})`);
        await this.drain();
        if (r === 0) break;
      }
    } catch (e) {
      this.failure = e instanceof Error ? e : new Error(String(e));
      throw this.failure;
    }
  }

  /** Every picture of what was sent so far comes out; then the decoder is ready to start again at a keyframe. */
  async flush(): Promise<void> {
    this.check();
    const m = this.mod!;
    try {
      const r = m._dec_send(this.dec, 0, 0, 0, 0);
      if (r < 0) throw new Error(`the built-in HEVC decoder failed to finish (${r})`);
      // after a flush the decoder gives every picture and then says it has no more
      for (let i = 0; !(await this.drain()); i++) if (i > 64) throw new Error('the built-in HEVC decoder did not finish');
      m._dec_reset(this.dec);
    } catch (e) {
      this.failure = e instanceof Error ? e : new Error(String(e));
      throw this.failure;
    }
  }

  close(): void {
    const m = this.mod;
    if (!m) return;
    if (this.dec) m._dec_close(this.dec);
    if (this.input) m._dec_free(this.input);
    this.dec = 0;
    this.input = 0;
    this.mod = null;
  }
}
