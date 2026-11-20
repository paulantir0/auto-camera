// Messages between the page and its workers.
import type { VideoTrack } from './video/mp4.ts';
import type { PaperColour } from './pipeline/analysis.ts';
import type { PageFeatures } from './pipeline/matching.ts';

export interface ProxyChunk { key: boolean; timestamp: number; data: Uint8Array }

export interface ProxyVideo {
  config: VideoDecoderConfig;
  chunks: ProxyChunk[];
  fps: number;
}

/** What every proxy worker needs to know about the page. */
export interface PageSetup {
  median: Uint8Array;
  mask: Uint8Array;
  inner: Uint8Array;
  text: Uint8Array; // print_area
  colour: PaperColour | null;
}

export type ToDecode =
  | { kind: 'pass1'; file: File; track: VideoTrack; encoderCodec: string; software: number };

export type FromDecode =
  | { kind: 'progress'; done: number; total: number }
  | { kind: 'pass1-done'; frames: number; fps: number; proxy: ProxyVideo; detail: Uint8Array; moved: Uint8Array; medianSamples: number;
      seconds: Record<string, number>; decoderInfo: string }
  | { kind: 'error'; message: string };

/** Progress units for a frame the trained hand models look at, against one for a frame that is only decoded and measured. */
export const NET_WORK = 20;

/** The folder (a URL) with MediaPipe's engine and the two hand models, as the build lays it out (dist/hands/). */
export interface HandFiles { base: string }

export type ToHands =
  | { kind: 'init'; proxy: ProxyVideo; first: number; last: number; samples: number[]; printFrames: number[]; setup: PageSetup; files: HandFiles | null }
  | { kind: 'model'; useNet: boolean }
  | { kind: 'printsums'; lut: Uint8Array; usable: boolean; level: number; useNet: boolean }
  | { kind: 'strict'; map: Uint8Array; level: number }
  | { kind: 'measure'; lut: Uint8Array; usable: boolean; strictLut: Uint8Array; strictUsable: boolean; map: Uint8Array; level: number; needed: Uint8Array; useNet: boolean };

export interface HandArrays {
  first: number;
  share: Float32Array; onPrint: Float32Array; paper: Float32Array; onMap: Float32Array;
  net: Float32Array | null; netPage: Float32Array | null; netNear: Float32Array | null; netInside: Float32Array | null;
}

export type FromHands =
  | { kind: 'progress'; done: number }
  | { kind: 'ready'; net: boolean; message?: string }
  | { kind: 'model-done'; intruder: Float64Array; page: Float64Array; frames: number[]; medians: number[] }
  | { kind: 'printsums-done'; seen: Int32Array; printed: Int32Array; light: Float32Array; light2: Float32Array; frames: number }
  | { kind: 'strict-done'; intruder: Float64Array; anyChange: Float64Array; page: Float64Array }
  | ({ kind: 'measure-done' } & HandArrays)
  | { kind: 'error'; message: string };

export interface StillTarget { frame: number; start: number } // presentation index, decode index to start from

export type ToStill =
  | { kind: 'init'; file: File; track: VideoTrack; region: Uint8Array; jpegQuality: number; previewWidth: number; useStore: boolean; software: number }
  | { kind: 'frames'; targets: StillTarget[] };

export interface StillResult {
  frame: number;
  gray: Uint8Array; // 480x270, proxy-style
  features: PageFeatures;
  jpeg: Blob | null; // when not stored on disk
  stored: string | null; // OPFS file name
  size: number;
  crc: number;
  preview: Blob;
  width: number;
  height: number;
}

export type FromStill =
  | { kind: 'ready' }
  | { kind: 'still'; result: StillResult }
  | { kind: 'frames-done' }
  | { kind: 'error'; message: string };
