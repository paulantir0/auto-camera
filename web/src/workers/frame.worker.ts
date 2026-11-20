// A frame helper for pass 1: given decoded full-size frames (transferred,
// not copied), shrink each to the analysis size and measure it. Several run
// side by side so the decoder, not the arithmetic, sets the pace. Frames are
// handled one at a time, in the order they arrive.

import { measureFrame } from '../video/measure.ts';

let port: MessagePort | null = null;
let chain: Promise<void> = Promise.resolve();

self.onmessage = (e: MessageEvent) => {
  if (e.data.kind !== 'port') return;
  port = e.data.port as MessagePort;
  port.onmessage = (m: MessageEvent) => {
    const { frame, index } = m.data as { frame: VideoFrame; index: number };
    chain = chain.then(async () => {
      try {
        const r = await measureFrame(frame, index);
        port!.postMessage({ kind: 'measured', result: r }, [r.blocks.buffer, r.detail.buffer, r.i420.buffer]);
      } catch (err) {
        port!.postMessage({ kind: 'error', message: err instanceof Error ? err.message : String(err) });
      }
    });
  };
};
