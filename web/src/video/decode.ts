// Drive a WebCodecs VideoDecoder: feed chunks with back-pressure and hand the
// decoded frames, in output order, to an async handler one at a time. At most
// a few frames are alive at once, so memory stays flat however long the video.

/** One compressed frame, as stored in the file. */
export interface Chunk { key: boolean; timestamp: number; data: Uint8Array }

/** What the workers need of a decoder, whichever kind it is. */
export interface Decoder {
  decode(chunk: Chunk): Promise<void>;
  flush(): Promise<void>;
  close(): void;
}

export interface DecodeOptions {
  maxQueue?: number; // chunks waiting inside the decoder
  maxFrames?: number; // decoded frames waiting for the handler
}

export class FrameDecoder implements Decoder {
  private readonly decoder: VideoDecoder;
  private readonly frames: VideoFrame[] = [];
  private failure: Error | null = null;
  private wake: (() => void) | null = null;
  private handling: Promise<void> = Promise.resolve();
  private busy = false;
  private readonly maxQueue: number;
  private readonly maxFrames: number;
  private readonly handler: (frame: VideoFrame) => Promise<void> | void;

  constructor(config: VideoDecoderConfig, handler: (frame: VideoFrame) => Promise<void> | void, options: DecodeOptions = {}) {
    this.handler = handler;
    this.maxQueue = options.maxQueue ?? 6;
    this.maxFrames = options.maxFrames ?? 3;
    this.decoder = new VideoDecoder({
      output: frame => { this.frames.push(frame); this.pump(); this.poke(); },
      error: e => { this.failure = e instanceof Error ? e : new Error(String(e)); this.poke(); },
    });
    this.decoder.configure(config);
    // 'dequeue' tells us when there is room again
    (this.decoder as unknown as EventTarget).addEventListener?.('dequeue', () => this.poke());
  }

  private poke(): void {
    const w = this.wake;
    this.wake = null;
    w?.();
  }

  private waitForChange(): Promise<void> {
    return new Promise(resolve => {
      this.wake = resolve;
      // some browsers have no dequeue event; look again soon
      setTimeout(() => { if (this.wake === resolve) this.poke(); }, 4);
    });
  }

  private pump(): void {
    if (this.busy) return;
    this.busy = true;
    this.handling = (async () => {
      try {
        while (this.frames.length) {
          const frame = this.frames.shift()!;
          try {
            if (!this.failure) await this.handler(frame);
          } catch (e) {
            this.failure = e instanceof Error ? e : new Error(String(e));
          } finally {
            frame.close();
          }
          this.poke();
        }
      } finally {
        this.busy = false;
      }
    })();
  }

  private check(): void {
    if (this.failure) throw this.failure;
  }

  async decode(chunk: Chunk): Promise<void> {
    this.check();
    while (this.decoder.decodeQueueSize >= this.maxQueue || this.frames.length >= this.maxFrames) {
      await this.waitForChange();
      this.check();
    }
    this.decoder.decode(new EncodedVideoChunk({ type: chunk.key ? 'key' : 'delta', timestamp: chunk.timestamp, data: chunk.data }));
  }

  /** Wait until every chunk fed so far has been decoded and handled. */
  async flush(): Promise<void> {
    this.check();
    await this.decoder.flush();
    while (this.frames.length || this.busy) {
      await this.handling;
      if (this.frames.length && !this.busy) this.pump();
    }
    this.check();
  }

  close(): void {
    for (const f of this.frames) f.close();
    this.frames.length = 0;
    if (this.decoder.state !== 'closed') this.decoder.close();
  }
}
