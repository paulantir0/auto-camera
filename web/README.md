# Book Scanner in the browser

Live at **https://video.paulantir.com/**.

A static web page that does the whole book-scanner job inside the visitor's
browser: choose a video of a book being paged through, and it gives one
full-size picture per two-page spread, in page order, as a ZIP. On request it
then reads the printed page numbers, removes pictures that show the same
pages twice and says whether any pages are missing; and, on request, cuts
every picture into its two pages, flattens them and makes a searchable PDF of
the book. No server does any work and the video never leaves the device: the
page only ever downloads its own files, and its Content-Security-Policy
forbids connections to anything else.

It is a port of the Python pipeline in [`../auto_camera/`](../auto_camera):
the same steps, the same thresholds (`src/pipeline/constants.ts`, under the
Python names), the same OpenCV operations, the same two hand models (Google's
MediaPipe hand landmarker and skin segmenter), the same OCR models for the
page numbers, and OpenCV.js and Tesseract.js for the pages and the PDF. The
[main README](../README.md) says how the pipeline works.

## Run it locally

```bash
cd web
npm ci
npm run build          # dist/: the whole site; fetches the model files and checks them by SHA-256
npm run serve          # http://127.0.0.1:8765/ with the production headers
```

The build needs FFmpeg's release tarball once, for the built-in video
decoder's source: `tools/build-hevc.sh` downloads and checks it.

WebCodecs needs a secure context, so another device on the network cannot use
the `http://` address; use the deployed site there.

## Test it

```bash
npm test               # the port against fixtures made by the Python code
npm run typecheck
```

`test/fixtures/synthetic/` holds random images, drawn scenes, scripted motion,
made-up page-number sequences, bands of drawn text and two tiny test-pattern
videos, all run through the Python and OpenCV functions; nothing from any
recording. Tests that need fixtures made from real videos
(`test/fixtures/real/`, never in git) run only where those exist.

## Which browsers can read the videos

The recordings are HEVC (H.265). Where the browser has HEVC (Safari on
iPhone, iPad and Mac; Chrome and Edge on Windows and Mac with a GPU that
decodes it; many Android phones) the page uses it through WebCodecs. Where it
has not, the page uses its own decoder: FFmpeg 7.1.2's HEVC decoder compiled
to WebAssembly (`wasm/`, built by `tools/build-hevc.sh`, LGPL; its source is
served beside it under `source/`). It is exact: on the first 240 frames of a
recording every picture is bit-identical to native FFmpeg's, in Chrome and in
Firefox. At 3072 x 1728 it decodes 10.7 frames a second on one thread and
27.8 on four. Threads need the site's cross-origin-isolation headers, which
`vercel.json` sets.

## Deploy

The site is the `dist/` folder, deployed as static files with the headers in
`vercel.json`. Only `dist/` is uploaded.

```bash
npm ci
npm run deploy         # build (with the privacy check), then: vercel deploy dist --prod
```

Every build ends with `tools/privacy-check.mjs`, which fails if a file to be
served holds a local path, a user name or e-mail address, a word from a
video's file name, a reference to another host, or is of a kind that is not
expected.

## How it works

| Stage | Where | What |
| --- | --- | --- |
| Reading the video | analysis worker, 1-3 frame helpers | The MP4/MOV sample table is read by byte ranges (`src/video/mp4.ts`); every frame is decoded once (WebCodecs, or the built-in decoder, `src/video/wasm-decoder.ts`), shrunk to 480 x 270 and measured for motion and print sharpness in 10 x 10 cells; a small H.264 working copy is encoded on the way. |
| Hands | 1-4 hands workers | The working copy in parallel stretches (`src/workers/hands.worker.ts`): the print map from the frames at rest (`src/pipeline/printmap.ts`), then the hand models on the frames that frame choice reads (MediaPipe in WebAssembly, `src/pipeline/handdetect.ts`, `handnet.ts`). |
| Page turns | analysis worker | Turn detection and one candidate frame per spread. |
| Pictures | 1-3 still workers | For each candidate, seek to the keyframe before it and decode up to it: a full-size JPEG, a preview, and ORB keypoints (OpenCV.js). |
| Repeats | analysis and select workers | The early same-spread merge, then the keypoint repeat check. |
| Page numbers (on request) | OCR worker | Two bands of each picture read by PP-OCRv4 in onnxruntime-web; the number check removes repeats and reports gaps. |
| The book (on request) | 1-3 pages workers, 1-4 Tesseract workers | Each picture cut at the fold, its lines of text found, the curved-sheet model fitted (`src/pages/pages.ts`, `sheet.ts`) and the page unrolled; each page read by Tesseract.js in seven languages; the pages joined into one PDF (`src/pages/pdf.ts`). |

Memory stays flat with video length: a few full-size frames are alive at a
time, the pictures go to the origin private file system on the device, and
the ZIP is assembled from them without copying.

## Files served

`index.html`, `app.css`, `app.js`, `icon.svg`, `licenses.txt`,
`workers/{analysis,frame,hands,still,select,ocr,pages}.js`,
`workers/opencv.js`, `workers/hevc{,-mt}.{js,wasm}`, the OCR models and
runtime under `ocr/`, the hand models and MediaPipe under `hands/`,
Tesseract and its language data under `tesseract/`, and
`source/{ffmpeg-7.1.2.tar.xz,hevcdec.c,build-hevc.sh}`. The model and language
files are checked against pinned SHA-256 sums by `tools/fetch-models.mjs`.
