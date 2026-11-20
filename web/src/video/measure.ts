// Shrink one decoded full-size frame to the analysis size and measure it.
// Used by the frame helpers of pass 1 (and by the analysis worker itself when
// it has none).

import { Shrinker, readFrame } from '../video/sampler.ts';
import { GRID_H, GRID_W, PROXY_HEIGHT, PROXY_WIDTH, THUMB_H, THUMB_W } from '../pipeline/constants.ts';
import { laplacianAbs, resizeAreaInt } from '../pipeline/imgproc.ts';
import { CELLS, PIXELS } from '../pipeline/analysis.ts';

let shrinker: Shrinker | null = null;
const buf = { data: new Uint8Array(0) };
const thumbs = new Uint8Array(THUMB_W * THUMB_H);
const lap = new Uint8Array(PIXELS);

export interface FrameMeasure {
  index: number;
  blocks: Float32Array; // 10x10 block means (from 5x5 thumbnails), for the motion cells
  detail: Uint8Array; // per-cell mean |Laplacian|
  i420: Uint8Array; // the 480x270 frame, for the working copy and the median
  seconds: [number, number, number]; // copy, shrink, measure
}

export async function measureFrame(frame: VideoFrame, index: number): Promise<FrameMeasure> {
  try {
    const width = frame.visibleRect?.width ?? frame.codedWidth, height = frame.visibleRect?.height ?? frame.codedHeight;
    if (!shrinker || shrinker.width !== width || shrinker.height !== height) shrinker = new Shrinker(width, height);
    const t0 = performance.now();
    const { format, layout } = await readFrame(frame, buf);
    frame.close();
    const t1 = performance.now();
    shrinker.shrink(buf.data, format, layout, true);
    const t2 = performance.now();
    const gray = shrinker.gray;
    resizeAreaInt(gray, PROXY_WIDTH, PROXY_HEIGHT, THUMB_W, THUMB_H, thumbs);
    const blocks = new Float32Array(CELLS);
    for (let gy = 0; gy < GRID_H; gy++) {
      for (let gx = 0; gx < GRID_W; gx++) {
        const a = 2 * gy * THUMB_W + 2 * gx;
        blocks[gy * GRID_W + gx] = (thumbs[a] + thumbs[a + 1] + thumbs[a + THUMB_W] + thumbs[a + THUMB_W + 1]) / 4;
      }
    }
    laplacianAbs(gray, PROXY_WIDTH, PROXY_HEIGHT, lap);
    const detail = resizeAreaInt(lap, PROXY_WIDTH, PROXY_HEIGHT, GRID_W, GRID_H);
    const t3 = performance.now();
    return { index, blocks, detail, i420: shrinker.i420.slice(), seconds: [(t1 - t0) / 1000, (t2 - t1) / 1000, (t3 - t2) / 1000] };
  } finally {
    try { frame.close(); } catch { /* closed already */ }
  }
}

