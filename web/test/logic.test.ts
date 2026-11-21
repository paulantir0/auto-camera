// Turn detection, frame choice and review flags against the Python code, on
// scripted synthetic motion (no video data).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Fixture, differences } from './load.ts';
import { CELLS, pageMaskFromBrightness, regionMotion, textFill, type Analysis } from '../src/pipeline/analysis.ts';
import { detectTurns, groupSpreads, oneLandingPerFall } from '../src/pipeline/spreads.ts';
import { dropTransitional, reviewFlags, toCandidate, type Candidate } from '../src/pipeline/repeats.ts';
import type { PageFeatures } from '../src/pipeline/matching.ts';
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
  const ref = fx.value<{ landings: number[]; confirmed: number[]; weak: boolean[]; landing_cover: number[]; settled: number[] }>('turns');
  assert.deepEqual(t.landings, ref.landings);
  assert.deepEqual(t.settled ?? t.landings, ref.settled);
  assert.deepEqual(t.confirmed, ref.confirmed);
  assert.deepEqual(t.weak, ref.weak);
  t.landingCover.forEach((v, i) => assert.ok(Math.abs(v - ref.landing_cover[i]) < 1e-6));
});

test('a leaf that slows down in the air lands once', () => {
  // as tests/test_spreads.py: the left page swept (6-8), slowed to a quarter (9), swept again by the same leaf (10-13)
  const leftMotion = Float32Array.from([0, 0, 0, 0, 0, 0, 1, 1, 1, 0.2857, 1, 1, 1, 1, 0, 0, 0, 0, 0]);
  const cover = new Float32Array(leftMotion.length);
  const turns = (landings: number[], confirmed: number[]) =>
    ({ landings, confirmed, landingCover: landings.map((_, k) => k + 1), weak: landings.map(() => false), coverLeft: cover, coverRight: cover });
  const once = oneLandingPerFall(turns([6, 10], [9, 14]), leftMotion);
  assert.deepEqual([once.landings, once.settled, once.confirmed, once.landingCover], [[6], [10], [14], [2]]);
  // a sweep more than three frames after the slow-down is a landing of its own
  const apart = oneLandingPerFall(turns([6, 14], [9, 18]), leftMotion);
  assert.deepEqual([apart.landings, apart.settled], [[6, 14], undefined]);
  // and so is one right after the page was at rest
  const rested = Float32Array.from([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]);
  assert.deepEqual(oneLandingPerFall(turns([6, 10], [9, 13]), rested).landings, [6, 10]);
  // three sweeps of one fall are one landing, and a later turn stays its own
  const twice = Float32Array.from([0, 0, 1, 1, 1, 0.3, 1, 1, 1, 0.3, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]);
  const chain = oneLandingPerFall(turns([2, 6, 10, 18], [5, 9, 13, 20]), twice);
  assert.deepEqual([chain.landings, chain.settled, chain.confirmed], [[2, 18], [10, 18], [13, 20]]);
});

test('a better later view takes the place of a first view that never came to rest', () => {
  // as TransitionalTests in tests/test_repeats.py: pages are a keypoint count and a name; two pages agree when named alike
  const names = new Map<PageFeatures, string>();
  const page = (keypoints: number, name: string): PageFeatures => {
    const f = { xs: new Float32Array(keypoints), ys: new Float32Array(keypoints), desc: new Uint8Array(0) };
    names.set(f, name);
    return f;
  };
  const agree = (a: PageFeatures, b: PageFeatures) => (names.get(a) === names.get(b) ? 100 : 0);
  const capture = (motion: number, review: string[] = [], hand = 0): Candidate =>
    ({ best_motion: motion, detail: 50, hand_on_print: hand, review } as unknown as Candidate);
  const MOVING = ['never_at_rest'], COVERED = ['hand_on_print', 'never_at_rest'];
  const run = (pages: [PageFeatures, PageFeatures][], candidates: Candidate[]) => {
    const [kept, dropped] = dropTransitional(pages.map((_, i) => [i]), candidates, pages, agree);
    return { kept: kept.map(g => g[0]), dropped: Object.fromEntries(dropped) };
  };
  const settled = (l: string, r: string): [PageFeatures, PageFeatures] => [page(800, l), page(800, r)];

  let r = run([settled('L1', 'R1'), [page(800, 'L1'), page(800, 'R2')], settled('L2', 'R2')], [capture(0), capture(0.5, MOVING), capture(0)]);
  assert.deepEqual(r, { kept: [0, 2], dropped: { 1: 'left page matches the previous spread' } });

  const firstView: [PageFeatures, PageFeatures] = [page(100, 'leaf'), page(2400, 'R2')];
  r = run([settled('L1', 'R1'), firstView, [page(1400, 'L2'), page(1100, 'R2')], settled('L3', 'R3')],
    [capture(0), capture(0.65, COVERED, 0.09), capture(0.2, MOVING), capture(0)]);
  assert.deepEqual(r, { kept: [0, 2, 3], dropped: { 1: 'a better view of its right page follows' } });

  r = run([settled('L1', 'R1'), [page(500, 'L2'), page(2000, 'R2')], [page(1900, 'L2'), page(600, 'R2')], settled('L3', 'R3')],
    [capture(0), capture(0.56, COVERED, 0.23), capture(0.18, MOVING), capture(0)]);
  assert.deepEqual(r, { kept: [0, 2, 3], dropped: { 1: 'a better view of its left page follows' } });

  r = run([settled('L1', 'R1'), [page(100, 'leaf'), page(2400, 'R2')], [page(1400, 'L2'), page(1100, 'R2')], settled('L3', 'R3')],
    [capture(0), capture(0.3, MOVING), capture(0.3, COVERED, 0.05), capture(0)]);
  assert.deepEqual(r, { kept: [0, 1, 3], dropped: { 2: 'right page matches the previous spread' } });

  r = run([settled('L1', 'R1'), [page(1300, 'L2'), page(1200, 'R2')], [page(1300, 'L2'), page(1200, 'R9')], settled('L3', 'R3')],
    [capture(0), capture(0.3, COVERED, 0.02), capture(0.2, MOVING), capture(0)]);
  assert.deepEqual(r, { kept: [0, 1, 3], dropped: { 2: 'left page matches the previous spread' } });

  r = run([settled('L1', 'R1'), [page(100, 'leaf'), page(2400, 'R2')], [page(1400, 'L2'), page(1100, 'R2')], settled('L2', 'R2')],
    [capture(0), capture(0.65, COVERED, 0.09), capture(0.2, MOVING), capture(0)]);
  assert.deepEqual(r, { kept: [0, 3], dropped: { 1: 'right page matches the next spread', 2: 'left page matches the next spread' } });

  r = run([settled('L1', 'R1'), [page(100, 'leaf'), page(2400, 'R2')], [page(1400, 'L1'), page(1100, 'R2')], settled('L3', 'R3')],
    [capture(0), capture(0.65, COVERED, 0.09), capture(0.2, MOVING), capture(0)]);
  assert.deepEqual(r, { kept: [0, 1, 3], dropped: { 2: 'right page matches the previous spread' } });
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
