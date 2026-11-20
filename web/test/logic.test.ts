// Turn detection, frame choice and review flags against the Python code, on
// scripted synthetic motion (no video data).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Fixture, differences } from './load.ts';
import { CELLS, pageMaskFromBrightness, regionMotion, textFill, type Analysis } from '../src/pipeline/analysis.ts';
import { detectTurns, groupSpreads } from '../src/pipeline/spreads.ts';
import { reviewFlags, toCandidate } from '../src/pipeline/repeats.ts';
import { pyRound, pyFixed } from '../src/pipeline/pyround.ts';
import { captureName } from '../src/pipeline/repeats.ts';

const fx = new Fixture(new URL('./fixtures/synthetic', import.meta.url).pathname);

function synthetic(): Analysis {
  const moved = fx.u8('turn_moved'), detail = fx.u8('turn_detail');
  const left = fx.u8('turn_left'), right = fx.u8('turn_right');
  const frames = moved.length / CELLS;
  const page = left.map((v, i) => v | right[i]);
  const pageDetail = new Float32Array(frames);
  const cells = [...page.keys()].filter(c => page[c]);
  for (let f = 0; f < frames; f++) {
    let s = 0;
    for (const c of cells) s += detail[f * CELLS + c];
    pageDetail[f] = s / cells.length;
  }
  return {
    fps: 11, frames, detail, moved, median: new Uint8Array(0), text: new Uint8Array(CELLS), mask: new Uint8Array(0), inner: new Uint8Array(0),
    gridPage: page, gridLeft: left, gridRight: right,
    pageMotion: regionMotion(moved, frames, page), leftMotion: regionMotion(moved, frames, left), rightMotion: regionMotion(moved, frames, right), pageDetail,
  };
}

test('detect_turns finds the same landings', () => {
  const t = detectTurns(synthetic());
  const ref = fx.value<{ landings: number[]; confirmed: number[]; weak: boolean[]; landing_cover: number[] }>('turns');
  assert.deepEqual(t.landings, ref.landings);
  assert.deepEqual(t.confirmed, ref.confirmed);
  assert.deepEqual(t.weak, ref.weak);
  t.landingCover.forEach((v, i) => assert.ok(Math.abs(v - ref.landing_cover[i]) < 1e-6));
});

test('group_spreads chooses the same frames, with the same values', () => {
  const a = synthetic();
  const t = detectTurns(a);
  const hands = { usable: true, samples: 5000, share: fx.f32('hand_share'), onPrint: fx.f32('hand_on_print'), paper: fx.f32('hand_paper') };
  // the synthetic analysis has blank thumbnails, as Python's (cell_light all 0)
  const spreads = groupSpreads(t, a, hands, new Uint16Array(a.frames * CELLS));
  const ref = fx.value<Record<string, unknown>[]>('spreads');
  assert.equal(spreads.length, ref.length);
  const ours = spreads.map(s => {
    const c = toCandidate(s, 11) as unknown as Record<string, unknown>;
    delete c.review;
    return c;
  });
  assert.deepEqual(ours, ref);
  const flags = Object.fromEntries(spreads.map(s => [String(s.index), reviewFlags(s)]).filter(([, f]) => (f as string[]).length));
  assert.deepEqual(flags, fx.value('flags'));
});

test('brightness page mask and text fill (fallback for recordings with little print)', () => {
  const mask = pageMaskFromBrightness(fx.u8('bright_scene'), fx.array('bright_activity') as Float64Array);
  assert.equal(differences(mask, fx.u8('bright_mask')), 0);
  const fill = fx.value<{ blocks: number; scattered: number }>('text_fill');
  const left = fx.u8('turn_left'), right = fx.u8('turn_right');
  assert.ok(Math.abs(textFill(left.map((v, i) => v | right[i])) - fill.blocks) < 1e-12);
  assert.ok(Math.abs(textFill(fx.u8('scattered')) - fill.scattered) < 1e-12);
});

test('Python rounding and file names', () => {
  assert.equal(pyRound(0.125, 2), 0.12);
  assert.equal(pyRound(0.375, 2), 0.38);
  assert.equal(pyRound(2.675, 2), 2.67);
  assert.equal(pyRound(1.0005, 3), 1.0);
  assert.equal(pyRound(-0.00004, 4), -0);
  assert.equal(pyFixed(12.345, 7, 2), '0012.35');
  assert.equal(pyFixed(0.125, 7, 2), '0000.12');
  assert.equal(captureName(7, 1234, 11), 'sequence_0007_t0112.18s.jpg');
});
