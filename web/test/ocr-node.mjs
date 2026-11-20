// Run the browser OCR engine (src/ocr/engine.ts) in Node on raw BGR bands, for checking it against
// rapidocr step by step: node test/ocr-node.mjs OPENCV_JS BAND.bin HEIGHT WIDTH [rec]
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import * as ort from 'onnxruntime-web';
import { OcrEngine } from '../src/ocr/engine.ts';

const [cvPath, bandPath, h, w, mode] = process.argv.slice(2);
const require = createRequire(import.meta.url);
let cv = require(cvPath);
if (cv instanceof Promise) cv = await cv; else if (!cv.Mat) await new Promise(r => { cv.onRuntimeInitialized = r; });
ort.env.wasm.numThreads = 1;
const models = new URL('../ocr-models/', import.meta.url).pathname;
const det = await ort.InferenceSession.create(models + 'ch_PP-OCRv4_det_infer.onnx');
const rec = await ort.InferenceSession.create(models + 'ch_PP-OCRv4_rec_infer.onnx');
const wrap = s => ({ run: async (input, dims) => { const o = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', input, dims) }); const t = o[s.outputNames[0]]; return { data: t.data, dims: t.dims }; } });
const keys = readFileSync(models + 'ppocr_keys.txt', 'utf8').split('\n'); keys.pop();
const engine = new OcrEngine(cv, wrap(det), wrap(rec), keys);
const band = { data: new Uint8Array(readFileSync(bandPath)), width: Number(w), height: Number(h) };
if (mode === 'rec') console.log(JSON.stringify(await engine.recognise(band)));
else {
  const boxes = await engine.detect(band);
  console.log(`${boxes.length} boxes`);
  for (const b of boxes.sort((p, q) => p[0][1] - q[0][1] || p[0][0] - q[0][0])) console.log('  ', JSON.stringify(b.map(p => p.map(v => Math.round(v * 100) / 100))));
}
