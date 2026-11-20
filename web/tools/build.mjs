#!/usr/bin/env node
// Production build: bundle the page and its workers into dist/, copy the
// static files and the self-hosted OpenCV.js, and check that nothing in
// dist/ points outside the site.
import { build } from 'esbuild';
import { cp, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { patchOpenCv } from './patch-opencv.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const dist = join(root, 'dist');
const watch = process.argv.includes('--dev');

execFileSync(process.execPath, [join(root, 'tools/build-wasm.mjs')], { stdio: 'inherit' });
// empty dist/, but keep .vercel/ (the link to the Vercel project made by the first deploy)
await mkdir(dist, { recursive: true });
for (const e of await readdir(dist)) if (e !== '.vercel') await rm(join(dist, e), { recursive: true, force: true });
await mkdir(join(dist, 'workers'), { recursive: true });

const common = { bundle: true, minify: !watch, sourcemap: false, target: ['es2022', 'safari16.4', 'chrome110', 'firefox130'], legalComments: 'eof', logLevel: 'warning' };
await build({ ...common, entryPoints: [join(root, 'src/app/main.ts')], outfile: join(dist, 'app.js'), format: 'esm' });
for (const name of ['analysis', 'frame', 'hands', 'still', 'select', 'ocr', 'pages']) {
  await build({ ...common, entryPoints: [join(root, `src/workers/${name}.worker.ts`)], outfile: join(dist, 'workers', `${name}.js`), format: 'iife' });
}
for (const f of ['index.html', 'app.css', 'icon.svg']) await cp(join(root, 'src', f), join(dist, f));
// OpenCV.js, rewritten to need no run-time code generation (see tools/patch-opencv.mjs)
await writeFile(join(dist, 'workers/opencv.js'), patchOpenCv(await readFile(join(root, 'node_modules/@techstark/opencv-js/dist/opencv.js'), 'utf8')));
await cp(join(root, 'vercel.json'), join(dist, 'vercel.json'));
// never upload anything the Vercel CLI may leave here (it writes a token to .env.local when linking)
await writeFile(join(dist, '.vercelignore'), '.env*\n.gitignore\n.vercelignore\n');
// the page-number reader: onnxruntime-web's WebAssembly engine, the PP-OCRv4 models and the recogniser's dictionary
await mkdir(join(dist, 'ocr'), { recursive: true });
for (const f of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) await cp(join(root, 'node_modules/onnxruntime-web/dist', f), join(dist, 'ocr', f));
await cp(join(root, 'ocr-models/ch_PP-OCRv4_det_infer.onnx'), join(dist, 'ocr/det.onnx'));
await cp(join(root, 'ocr-models/ch_PP-OCRv4_rec_infer.onnx'), join(dist, 'ocr/rec.onnx'));
await cp(join(root, 'ocr-models/ppocr_keys.txt'), join(dist, 'ocr/keys.txt'));
// onnxruntime-web's package carries no licence file; it is MIT, as its package.json says
const ortLicense = `MIT License\n\nCopyright (c) Microsoft Corporation\n\nPermission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n`;
// the model files, checked against their pinned SHA-256 (fetched if missing): tools/fetch-models.mjs
execFileSync(process.execPath, [join(root, 'tools/fetch-models.mjs')], { stdio: 'inherit' });
// Google's MediaPipe: its WebAssembly engine (from the pinned npm package) and the two hand models the Python pipeline uses
await mkdir(join(dist, 'hands'), { recursive: true });
for (const f of ['vision_wasm_internal.js', 'vision_wasm_internal.wasm', 'vision_wasm_nosimd_internal.js', 'vision_wasm_nosimd_internal.wasm']) {
  await cp(join(root, 'node_modules/@mediapipe/tasks-vision/wasm', f), join(dist, 'hands', f));
}
for (const f of ['hand_landmarker.task', 'selfie_multiclass_256x256.tflite']) await cp(join(root, 'hand-models', f), join(dist, 'hands', f));
// Tesseract.js for the searchable PDF: its worker, its LSTM engines (one per kind of WebAssembly support; each browser
// downloads one) and the language data Python's PDF uses (tessdata_fast 4.1.0)
await mkdir(join(dist, 'tesseract', 'lang'), { recursive: true });
await cp(join(root, 'node_modules/tesseract.js/dist/worker.min.js'), join(dist, 'tesseract', 'worker.min.js'));
for (const f of ['tesseract-core-lstm.wasm.js', 'tesseract-core-simd-lstm.wasm.js', 'tesseract-core-relaxedsimd-lstm.wasm.js']) {
  await cp(join(root, 'node_modules/tesseract.js-core', f), join(dist, 'tesseract', f));
}
for (const lang of ['eng', 'fra', 'deu', 'lat', 'grc', 'ell', 'ita']) await cp(join(root, 'tess-lang', `${lang}.traineddata`), join(dist, 'tesseract', 'lang', `${lang}.traineddata`));
// the built-in HEVC decoder (tools/build-hevc.sh), for browsers that have none
const hevcFiles = ['hevc.js', 'hevc.wasm', 'hevc-mt.js', 'hevc-mt.wasm'];
for (const f of hevcFiles) await cp(join(root, 'wasm', f), join(dist, 'workers', f));
// and its source, as the LGPL asks: FFmpeg's release tarball (kept by tools/build-hevc.sh) and the lines that connect it
await mkdir(join(dist, 'source'), { recursive: true });
await cp(join(root, 'wasm/hevcdec.c'), join(dist, 'source/hevcdec.c'));
await cp(join(root, 'tools/build-hevc.sh'), join(dist, 'source/build-hevc.sh'));
await cp(join(root, 'wasm/.build/ffmpeg-7.1.2.tar.xz'), join(dist, 'source/ffmpeg-7.1.2.tar.xz')).catch(async () => {
  throw new Error('wasm/.build/ffmpeg-7.1.2.tar.xz is missing: run tools/build-hevc.sh once (it downloads and checks it)');
});
const opencvLicense = await readFile(join(root, 'node_modules/@techstark/opencv-js/LICENSE'), 'utf8').catch(() => 'Apache License 2.0');
const lgpl = await readFile(join(root, 'wasm/COPYING.LGPLv2.1'), 'utf8');
await writeFile(join(dist, 'licenses.txt'),
  'This site bundles OpenCV.js (https://opencv.org), Apache License 2.0, via the @techstark/opencv-js package:\n\n' + opencvLicense +
  '\n\n----------------------------------------------------------------------\n\n' +
  'ocr/det.onnx, ocr/rec.onnx and ocr/keys.txt are the PP-OCRv4 text detection and recognition models and the recogniser\'s\n' +
  'character list from PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR), as converted to ONNX and packaged by\n' +
  'RapidOCR (https://github.com/RapidAI/RapidOCR), both under the Apache License 2.0 (text above). The steps around\n' +
  'the models in workers/ocr.js follow RapidOCR\'s rapidocr_onnxruntime 1.4.4 (Apache License 2.0).\n' +
  '\n----------------------------------------------------------------------\n\n' +
  'workers/ocr.js and ocr/ort-wasm-simd-threaded.* include ONNX Runtime Web (https://onnxruntime.ai):\n\n' + ortLicense +
  '\n\n----------------------------------------------------------------------\n\n' +
  'hands/vision_wasm_internal.* and hands/vision_wasm_nosimd_internal.*, and the code in workers/hands.js that drives them,\n' +
  'are MediaPipe Tasks Vision 0.10.21 (https://ai.google.dev/edge/mediapipe), Apache License 2.0 (text above), from the\n' +
  '@mediapipe/tasks-vision package. hands/hand_landmarker.task (hand landmarker, float16) and\n' +
  'hands/selfie_multiclass_256x256.tflite (multiclass selfie segmenter) are Google\'s MediaPipe models, which their model\n' +
  'cards publish under the Apache License 2.0, unmodified.\n' +
  '\n----------------------------------------------------------------------\n\n' +
  'tesseract/ is Tesseract.js 7.0.0 and Tesseract.js-core 7.0.0 (https://github.com/naptha/tesseract.js), Apache License 2.0,\n' +
  'which include Tesseract OCR (https://github.com/tesseract-ocr/tesseract, Apache License 2.0) and Leptonica\n' +
  '(http://leptonica.org, BSD 2-clause licence). tesseract/lang/*.traineddata are Tesseract\'s tessdata_fast 4.1.0 language\n' +
  'models (https://github.com/tesseract-ocr/tessdata_fast), Apache License 2.0.\n' +
  '\n----------------------------------------------------------------------\n\n' +
  'workers/hevc.wasm and workers/hevc-mt.wasm (with their .js loaders) are FFmpeg 7.1.2 (https://ffmpeg.org), its libavcodec\n' +
  'HEVC decoder and libavutil only, unmodified, built with Emscripten 4.0.10 for WebAssembly with no GPL parts.\n' +
  'FFmpeg is licensed under the GNU Lesser General Public License, version 2.1 or later, reproduced below.\n' +
  'Its complete source is served here as source/ffmpeg-7.1.2.tar.xz (the official release), with source/hevcdec.c (the few\n' +
  'lines that connect it to this page) and source/build-hevc.sh (every option used). You may rebuild the decoder from\n' +
  'them and replace these files.\n\n' + lgpl);

// Nothing served may reach another host at run time, and nothing personal may be in it.
const files = [];
async function walk(dir) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) await walk(p); else files.push(p);
  }
}
await walk(dist);
for (const f of files) console.log(`  ${f.slice(dist.length + 1)}  ${((await stat(f)).size / 1e3).toFixed(0)} kB`);
execFileSync(process.execPath, [join(root, 'tools/privacy-check.mjs')], { stdio: 'inherit' });
