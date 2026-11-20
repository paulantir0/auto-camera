// The printed-number check (src/ocr/pagenumbers.ts, audit.ts) against
// fixtures made by the Python code.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { auditSequence, decodePages, noRoomBetween, repeatsByNumber, sequenceReport } from '../src/ocr/pagenumbers.ts';
import { numberCheck, type Checked, type Readings } from '../src/ocr/audit.ts';
import type { Token } from '../src/ocr/read.ts';

const synthetic = new URL('./fixtures/synthetic/numbers.json', import.meta.url).pathname;
const realDir = new URL('./fixtures/real/', import.meta.url).pathname;

test('page-number decoding, reports, repeats and gaps as pagenumbers.py (synthetic sequences)', () => {
  const cases = JSON.parse(readFileSync(synthetic, 'utf8'));
  for (const [n, c] of cases.entries()) {
    for (const step of [1, 2]) {
      assert.deepEqual(decodePages(c.candidates, step), c.decoded[step], `case ${n} decode step ${step}`);
      assert.deepEqual(sequenceReport(decodePages(c.candidates, step), step), c.report[step], `case ${n} report step ${step}`);
    }
    const report = auditSequence(c.candidates);
    assert.deepEqual(report, c.audit, `case ${n} audit`);
    const pages = decodePages(c.candidates, report.step);
    assert.deepEqual(repeatsByNumber(pages, report.step, (i, j) => c.quality[i] > c.quality[j]), c.repeats, `case ${n} repeats`);
    assert.deepEqual(noRoomBetween(pages, report.step, c.flagged), c.no_room, `case ${n} no room`);
  }
});

const real = existsSync(realDir) ? readdirSync(realDir).filter(s => existsSync(`${realDir}${s}/numbers.json`)) : [];
test('the number check on real readings, as audit.audit_one', { skip: real.length ? false : 'no real fixtures on this machine' }, () => {
  for (const slug of real) {
    const f = JSON.parse(readFileSync(`${realDir}${slug}/numbers.json`, 'utf8'));
    const readings: Readings = new Map(Object.entries(f.readings as Record<string, Token[]>).map(([k, v]) => [Number(k), v]));
    const r = numberCheck(readings, f.captures as Checked[], f.frames);
    assert.deepEqual(r.removed, f.removed, `${slug} removed`);
    assert.deepEqual(r.pages, f.pages, `${slug} pages`);
    assert.deepEqual(r.report, f.report, `${slug} report`);
    assert.deepEqual(r.before, f.before, `${slug} before`);
  }
});
