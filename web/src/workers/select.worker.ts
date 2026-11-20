// The repeat check: keypoint matching between candidates and the final choice.
import { memoAgree, selectCaptures, splitPages, type Candidate } from '../pipeline/repeats.ts';
import type { PageFeatures } from '../pipeline/matching.ts';
import { usingWasm } from '../pipeline/matching.ts';

const post = (message: unknown) => (self as unknown as Worker).postMessage(message);

self.onmessage = (e: MessageEvent) => {
  const m = e.data as { candidates: Candidate[]; features: PageFeatures[]; centreX: number; fps: number };
  try {
    const started = performance.now();
    const pages = m.features.map(f => splitPages(f, m.centreX));
    let calls = 0;
    const agree = memoAgree();
    const counted = (a: PageFeatures, b: PageFeatures) => { calls++; return agree(a, b); };
    const selection = selectCaptures(m.candidates, pages, m.fps, counted);
    post({ kind: 'select-done', selection, seconds: (performance.now() - started) / 1000, comparisons: calls, wasm: usingWasm() });
  } catch (err) {
    post({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
