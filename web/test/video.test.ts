// Demuxer, ZIP writer and constants, on synthetic data.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { decodeStart, readSamples, readVideoTrack } from '../src/video/mp4.ts';
import { buildZip } from '../src/app/zip.ts';
import { crc32 } from '../src/app/crc32.ts';
// @ts-expect-error plain JavaScript tool
import { unzipStored } from '../tools/unzip.mjs';
import * as C from '../src/pipeline/constants.ts';

const dir = new URL('./fixtures/video/', import.meta.url).pathname;
const blob = (name: string) => new Blob([readFileSync(dir + name)]);

test('MP4 with B-frames: sample table, presentation order, codec', async () => {
  const t = await readVideoTrack(blob('h264-bframes.mp4'));
  assert.equal(t.samples.length, 33);
  assert.equal(t.codedWidth, 320);
  assert.equal(t.codedHeight, 180);
  assert.equal(t.fps, 11);
  assert.match(t.codec, /^avc1\.64/);
  assert.ok(t.description && t.description.length > 10);
  // presentation order is by composition time and is a permutation
  const cts = Array.from(t.presentation, d => t.samples[d].cts);
  assert.deepEqual(cts, [...cts].sort((a, b) => a - b));
  assert.deepEqual([...t.presentation].sort((a, b) => a - b), [...Array(33).keys()]);
  assert.ok(t.samples[0].sync);
  assert.ok(t.samples.filter(s => s.sync).length >= 4);
  // the bytes of each sample are where the table says
  const data = await readSamples(blob('h264-bframes.mp4'), t, 0, 32);
  assert.equal(data.length, 33);
  data.forEach((d, k) => assert.equal(d.length, t.samples[k].size));
});

test('HEVC in QuickTime: hvc1 codec string, leading pictures seek one group back', async () => {
  const t = await readVideoTrack(blob('hevc.mov'));
  assert.equal(t.samples.length, 33);
  assert.match(t.codec, /^hvc1\.1\.6\.L\d+\./);
  // frame 7 is decoded after the keyframe at frame 8 (a leading picture), so decoding starts at the keyframe before
  const k8 = t.presentation[8];
  assert.ok(t.samples[k8].sync);
  assert.equal(decodeStart(t, 8), k8);
  assert.ok(decodeStart(t, 7) < k8);
  assert.equal(decodeStart(t, 0), 0);
});

test('ZIP: stored entries with the right CRCs, read back by an independent reader', async () => {
  const entries = [0, 1, 2].map(k => {
    const bytes = new Uint8Array(1000 + k * 77).map((_, i) => (i * 31 + k) & 255);
    return { name: `sequence_000${k + 1}_t0001.00s.jpg`, data: new Blob([bytes]), size: bytes.length, crc: crc32(bytes) };
  });
  const zip = buildZip(entries);
  const tmp = mkdtempSync(join(tmpdir(), 'zip-'));
  writeFileSync(join(tmp, 'a.zip'), new Uint8Array(await zip.arrayBuffer()));
  const names = await unzipStored(join(tmp, 'a.zip'), join(tmp, 'out'));
  assert.deepEqual(names, entries.map(e => e.name));
  assert.deepEqual(readdirSync(join(tmp, 'out')).sort(), entries.map(e => e.name));
});

test('every threshold has the value the Python code has', () => {
  const py = (module: string) => readFileSync(new URL(`../../auto_camera/${module}.py`, import.meta.url), 'utf8');
  const constants = new Map<string, number>();
  for (const module of ['analysis', 'hands', 'spreads', 'repeats', 'export', 'pagenumbers', 'audit', 'ocr', 'handnet', 'handdetect']) {
    for (const m of py(module).matchAll(/^([A-Z][A-Z0-9_]+) = ([-0-9.e]+)\b/gm)) constants.set(`${module}.${m[1]}`, Number(m[2]));
  }
  const renamed: Record<string, string> = {
    'repeats.FAR_APART': 'REPEATS_FAR_APART', 'export.UNSETTLED_MOTION': 'EXPORT_UNSETTLED_MOTION',
    'hands.PRINT_LEVEL': 'HANDS_PRINT_LEVEL', 'handnet.MIN_SCORE': 'HANDNET_MIN_SCORE',
  };
  // spreads.TWO_FRAMES (a flag, not a number) is False in Python, and the port has no two-frame spreads
  assert.match(py('spreads'), /^TWO_FRAMES = False\b/m);
  const ours = C as unknown as Record<string, number>;
  let checked = 0;
  for (const [key, value] of constants) {
    const name = renamed[key] ?? key.split('.')[1];
    if (name === 'DETAIL_SCALE' || !(name in ours)) { assert.ok(name in ours || ['GRID_H', 'GRID_W'].includes(name), `${key} is missing from constants.ts`); continue; }
    assert.equal(ours[name], value, `${key}`);
    checked++;
  }
  assert.ok(checked >= 45, `only ${checked} constants checked`);
});
