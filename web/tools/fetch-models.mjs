#!/usr/bin/env node
// The model files the build serves, each checked against the SHA-256 the
// Python pipeline uses:
// - hand-models/: Google's two MediaPipe hand models, copied from the Python
//   side's data/models when it has them, else from Google's model storage
//   (the same files the Windows build pins);
// - tess-lang/: Tesseract's language data for the PDF (tessdata_fast 4.1.0,
//   what Ubuntu's tesseract-ocr-* packages install and Python's PDF used),
//   copied from /usr/share/tesseract-ocr when installed, else from GitHub.
//
//   node tools/fetch-models.mjs
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const models = {
  'hand_landmarker.task': {
    sha: 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
    url: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
  },
  'selfie_multiclass_256x256.tflite': {
    sha: 'c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0',
    url: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
  },
};
const languages = {
  eng: '7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2',
  fra: 'ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca',
  deu: '19d219bbb6672c869d20a9636c6816a81eb9a71796cb93ebe0cb1530e2cdb22d',
  lat: '3859d8ba60404f4b79830622625bbc76fb4ee2808eac1ad360ffa77f0a533328',
  grc: '57b735557eeb67503803cf9f99786dd0b62417931352601952a0b1a23b6c4759',
  ell: '4fba8a0b461038d51f1c20d043d4f2ac38c4e778f1b90830847f7bd8fa3ba726',
  ita: 'b8f89e1e785118dac4d51ae042c029a64edb5c3ee42ef73027a6d412748d8827',
};
const files = [
  ...Object.entries(models).map(([name, { sha, url }]) => ({ dir: 'hand-models', name, sha, url, local: [join(root, '..', 'data', 'models', name)] })),
  ...Object.entries(languages).map(([lang, sha]) => ({
    dir: 'tess-lang', name: `${lang}.traineddata`, sha, url: `https://github.com/tesseract-ocr/tessdata_fast/raw/4.1.0/${lang}.traineddata`,
    local: ['5', '4.00'].map(v => `/usr/share/tesseract-ocr/${v}/tessdata/${lang}.traineddata`),
  })),
];
for (const { dir, name, sha, url, local } of files) {
  await mkdir(join(root, dir), { recursive: true });
  let data = null, from = '';
  for (const path of [join(root, dir, name), ...local]) {
    data = await readFile(path).catch(() => null);
    if (data && createHash('sha256').update(data).digest('hex') === sha) { from = path; break; }
    data = null;
  }
  if (!data) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    data = Buffer.from(await r.arrayBuffer());
    from = url;
  }
  const actual = createHash('sha256').update(data).digest('hex');
  if (actual !== sha) throw new Error(`${name} from ${from} has SHA-256 ${actual}, not ${sha}`);
  if (from !== join(root, dir, name)) {
    await writeFile(join(root, dir, name), data);
    console.log(`${name}  ${(data.length / 1e6).toFixed(1)} MB  from ${from}`);
  }
}
