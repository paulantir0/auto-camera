// The fold between the two pages, against Python's pages.find_fold on real
// captures: tightly bound books, front matter, plain ones.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { findFold } from '../src/pages/pages.ts';

const dir = new URL('./fixtures/real/folds/', import.meta.url).pathname;
const here = existsSync(dir + 'folds.json');

test('the fold is the column Python finds, with and without the print region', { skip: here ? false : 'not exported on this machine' }, async () => {
  const require = createRequire(import.meta.url);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let cv: any = require('@techstark/opencv-js/dist/opencv.js');
  if (cv instanceof Promise) cv = await cv;
  else if (!cv.Mat) await new Promise<void>(r => { cv.onRuntimeInitialized = () => r(); });
  const cases: { name: string; width: number; height: number; region: string; region_width: number; region_height: number; fold: number; fold_without_region: number }[] =
    JSON.parse(readFileSync(dir + 'folds.json', 'utf8'));
  const wrong: string[] = [];
  for (const c of cases) {
    const gray = cv.matFromArray(c.height, c.width, cv.CV_8UC1, new Uint8Array(readFileSync(dir + c.name + '.gray')));
    const small = cv.matFromArray(c.region_height, c.region_width, cv.CV_8UC1, new Uint8Array(Buffer.from(c.region, 'base64')));
    const full = new cv.Mat();
    cv.resize(small, full, new cv.Size(c.width, c.height), 0, 0, cv.INTER_NEAREST);
    const fold = findFold(cv, gray, full), plain = findFold(cv, gray, null);
    if (fold !== c.fold || plain !== c.fold_without_region) wrong.push(`${c.name}: ${fold}/${plain} for ${c.fold}/${c.fold_without_region}`);
    gray.delete(); small.delete(); full.delete();
  }
  assert.ok(cases.length >= 40);
  assert.deepEqual(wrong, []);
});
