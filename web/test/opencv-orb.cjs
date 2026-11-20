// Child process for test/opencv.test.ts: ORB through OpenCV.js the way the
// still worker uses it, on a synthetic image; prints the keypoints' digest.
const fs = require('fs');
const crypto = require('crypto');
const [, , cvPath, imagePath] = process.argv;
let cv = require(cvPath);
(async () => {
  if (cv instanceof Promise) cv = await cv;
  else if (!cv.Mat) await new Promise(r => { cv.onRuntimeInitialized = r; });
  const w = 480, h = 270;
  const pixels = new Uint8Array(fs.readFileSync(imagePath));
  const img = new cv.Mat(h, w, cv.CV_8UC1);
  img.data.set(pixels);
  const mask = new cv.Mat(h, w, cv.CV_8UC1);
  mask.data.fill(255);
  const orb = new cv.ORB(500, 1.2, 8, 31, 0, 2, cv.ORB_HARRIS_SCORE, 31, 12);
  const kp = new cv.KeyPointVector();
  const desc = new cv.Mat();
  orb.detectAndCompute(img, mask, kp, desc);
  const pts = [];
  for (let i = 0; i < kp.size(); i++) { const p = kp.get(i); pts.push([p.pt.x, p.pt.y, p.octave]); }
  const digest = crypto.createHash('sha256').update(JSON.stringify(pts)).update(Buffer.from(desc.data)).digest('hex');
  kp.delete(); desc.delete(); img.delete(); mask.delete(); orb.delete();
  console.log(JSON.stringify({ keypoints: pts.length, digest }));
})().catch(e => { console.log(JSON.stringify({ error: String(e) })); process.exit(1); });
