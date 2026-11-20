// The repeated-page rule against the Python code.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { repeatedPages, type Repeat } from '../src/pages/textrepeats.ts';

interface Case { slug?: string; texts: string[]; pictures: unknown[]; flags: number[]; expected: Repeat[] }

const load = (path: string): Case[] => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const same = (c: Case) => {
  const found = repeatedPages(c.texts, c.pictures, c.flags);
  assert.deepEqual(found.map(r => [r.page, r.kept]), c.expected.map(r => [r.page, r.kept]));
  found.forEach((r, k) => assert.ok(Math.abs(r.share - c.expected[k].share) < 1e-9, `share ${r.share} vs ${c.expected[k].share}`));
};

test('made-up books: the same pages left out as by Python, Greek and superscripts included', () => {
  const cases = load('./fixtures/synthetic/textrepeats.json');
  assert.ok(cases.some(c => c.expected.length) && cases.some(c => !c.expected.length));
  for (const c of cases) same(c);
});

const real = new URL('./fixtures/real/textrepeats.json', import.meta.url);
test('real books of run_20261112_corpus_v9: the same pages left out as by Python', { skip: existsSync(real) ? false : 'not exported on this machine' }, () => {
  for (const c of load(real.pathname)) same(c);
});
