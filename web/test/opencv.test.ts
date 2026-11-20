// The OpenCV.js the site serves is patched to need no run-time code
// generation (so the Content-Security-Policy can forbid it). The patched
// copy, run with code generation from strings disallowed, must give exactly
// the ORB keypoints the original gives.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRequire } from 'node:module';
// @ts-expect-error plain JavaScript tool
import { patchOpenCv } from '../tools/patch-opencv.mjs';

const require = createRequire(import.meta.url);
const original = require.resolve('@techstark/opencv-js/dist/opencv.js');
const child = new URL('./opencv-orb.cjs', import.meta.url).pathname;
const image = new URL('./fixtures/synthetic/smooth.bin', import.meta.url).pathname;

test('patched OpenCV.js runs ORB without eval and gives the same keypoints', () => {
  const dir = mkdtempSync(join(tmpdir(), 'opencv-'));
  const patched = join(dir, 'opencv.js');
  writeFileSync(patched, patchOpenCv(readFileSync(original, 'utf8')));
  const run = (path: string, flags: string[]) => JSON.parse(execFileSync(process.execPath, [...flags, child, path, image], { encoding: 'utf8', maxBuffer: 1 << 24 }).trim().split('\n').pop()!);
  const before = run(original, []);
  const after = run(patched, ['--disallow-code-generation-from-strings']);
  assert.ok(before.keypoints > 50, JSON.stringify(before));
  assert.deepEqual(after, before);
  // and the original really does need code generation
  assert.throws(() => run(original, ['--disallow-code-generation-from-strings']));
});
