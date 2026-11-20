// The OCR port (src/ocr/engine.ts) against rapidocr on synthetic bands of
// drawn text: the same boxes, and the same
// text read from each, with onnxruntime-web and OpenCV.js in Node.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import * as ort from 'onnxruntime-web';
import { OcrEngine } from '../src/ocr/engine.ts';
import { crop } from '../src/ocr/read.ts';

const require = createRequire(import.meta.url);
const dir = new URL('./fixtures/synthetic/', import.meta.url).pathname;
const models = new URL('../ocr-models/', import.meta.url).pathname;

test('text boxes and readings as rapidocr (synthetic bands)', async () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let cv: any = require('@techstark/opencv-js/dist/opencv.js');
  if (cv instanceof Promise) cv = await cv;
  else if (!cv.Mat) await new Promise<void>(r => { cv.onRuntimeInitialized = () => r(); });
  ort.env.wasm.numThreads = 1;
  const det = await ort.InferenceSession.create(models + 'ch_PP-OCRv4_det_infer.onnx');
  const rec = await ort.InferenceSession.create(models + 'ch_PP-OCRv4_rec_infer.onnx');
  const wrap = (s: ort.InferenceSession) => ({ run: async (input: Float32Array, dims: number[]) => {
    const o = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', input, dims) });
    const t = o[s.outputNames[0]];
    return { data: t.data as Float32Array, dims: t.dims };
  } });
  const keys = readFileSync(models + 'ppocr_keys.txt', 'utf8').split('\n');
  keys.pop();
  const engine = new OcrEngine(cv, wrap(det), wrap(rec), keys);
  const bands = JSON.parse(readFileSync(dir + 'ocr.json', 'utf8'));
  for (const b of bands) {
    const image = { data: new Uint8Array(readFileSync(dir + b.file)), width: b.width, height: b.height };
    const boxes = await engine.detect(image);
    const key = (q: number[][]) => q.map(p => p.join(',')).join(' ');
    assert.deepEqual(boxes.map(key).sort(), b.boxes.map(key).sort(), `${b.file} boxes`);
    for (const r of b.reads) {
      const [text, score] = await engine.recognise(crop(image, r.crop[0], r.crop[1], r.crop[2], r.crop[3]));
      assert.equal(text, r.text, `${b.file} text`);
      assert.ok(Math.abs(score - r.score) < 0.01, `${b.file} score ${score} vs ${r.score}`);
    }
  }
});
