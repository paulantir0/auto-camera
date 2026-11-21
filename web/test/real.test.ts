// The whole selection logic against the Python code on real development
// videos. The fixtures are made by the Python code and stay on the
// machine that made them (test/fixtures/real is not in git and not deployed);
// without them these tests are skipped.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, existsSync } from 'node:fs';
import { Fixture, differences, maxAbsDiff } from './load.ts';
import { CELLS, derive, printCells, usualTextRows } from '../src/pipeline/analysis.ts';
import { HandMeter, ModelBuilder, finishModel, handSetup, printArea } from '../src/pipeline/hands.ts';
import { detectTurns, groupSpreads, mergeRepeats, type Hands } from '../src/pipeline/spreads.ts';
import { PrintSums, calmFrames, maskedMedianU8, printLevel, printMap, zones } from '../src/pipeline/printmap.ts';
import { fineDetail, handMask, palms, skinPatches } from '../src/pipeline/handnet.ts';
import { bgrToGray } from '../src/pipeline/colour.ts';
import { HANDS_PRINT_LEVEL } from '../src/pipeline/constants.ts';
import { memoAgree, printRegion, regionCentreX, selectCaptures, splitPages, toCandidate, type Candidate } from '../src/pipeline/repeats.ts';
import { agreeingPoints } from '../src/pipeline/matching.ts';

const root = new URL('./fixtures/real/', import.meta.url).pathname;
const slugs = existsSync(root) ? readdirSync(root).filter(s => Fixture.exists(root + s)) : [];

if (!slugs.length) test('real-video fixtures', { skip: 'none exported on this machine' }, () => {});

for (const slug of slugs) {
  const fx = new Fixture(root + slug);
  const name = slug.slice(0, 12);
  const frames = fx.value<number>('frames');
  const a = derive(fx.value<number>('fps'), frames, fx.u8('detail'), fx.u8('moved'), fx.u8('median'));

  test(`${name}: page region, halves and motion`, () => {
    assert.equal(differences(a.text, fx.u8('text')), 0, 'text cells');
    assert.equal(differences(a.mask, fx.u8('mask')), 0, 'page mask');
    assert.equal(differences(a.gridPage, fx.u8('grid_page')), 0, 'grid page');
    assert.equal(differences(a.gridLeft, fx.u8('grid_left')), 0, 'grid left');
    assert.equal(differences(a.gridRight, fx.u8('grid_right')), 0, 'grid right');
    assert.ok(maxAbsDiff(a.pageMotion, fx.f32('page_motion')) < 1e-6);
    assert.ok(maxAbsDiff(a.leftMotion, fx.f32('left_motion')) < 1e-6);
    assert.ok(maxAbsDiff(a.pageDetail, fx.f32('page_detail')) < 1e-4);
    assert.equal(differences(printCells(a), fx.u8('print_cells')), 0, 'print cells');
    assert.deepEqual(usualTextRows(a), fx.value('usual_text_rows'));
  });

  test(`${name}: hand model and per-frame hand shares`, () => {
    assert.equal(differences(printArea(a), fx.u8('print_area')), 0, 'print area');
    const builder = new ModelBuilder(a.median, a.mask, a.inner);
    const probe = fx.u8('probe_bgr');
    const px = 270 * 480 * 3;
    const probeFrames = fx.value<number[]>('probe_frames');
    for (let k = 0; k < probeFrames.length; k++) builder.add(probe.subarray(k * px, (k + 1) * px));
    assert.ok(maxAbsDiff(builder.intruder, fx.array('probe_intruder')) === 0, 'intruder histogram');
    assert.ok(maxAbsDiff(builder.page, fx.array('probe_page')) === 0, 'page histogram');
    if (fx.manifest.probe_lut) assert.equal(differences(builder.finish().lut, fx.u8('probe_lut')), 0, 'model from the probe frames');
    const model = { lut: fx.u8('hand_lut'), samples: fx.value<number>('hand_samples'), usable: fx.value<boolean>('hand_usable') };
    const setup = handSetup(a, model);
    assert.deepEqual(setup.colour, fx.value('paper_colour'));
    const meter = new HandMeter(setup);
    const share = fx.f32('hand_share'), onPrint = fx.f32('hand_on_print'), paper = fx.f32('hand_paper');
    probeFrames.forEach((f, k) => {
      const r = meter.measure(probe.subarray(k * px, (k + 1) * px));
      assert.equal(r.share, share[f], `share at ${f}`);
      assert.equal(r.onPrint, onPrint[f], `on print at ${f}`);
      assert.equal(r.paper, paper[f], `paper at ${f}`);
    });
    void finishModel;
  });

  const cells = printArea(a);
  const map = fx.u8('print_map');

  test(`${name}: frames at rest, print level and print map`, () => {
    assert.deepEqual(calmFrames(a, 300), fx.value('print_frames'));
    const level = printLevel(fx.value<number[]>('print_medians'));
    assert.equal(level, fx.value<number>('print_level'));
    const gray = fx.u8('print_probe_gray'), detail = fx.u8('print_probe_detail');
    const medians = fx.value<number[]>('print_medians');
    for (let k = 0; k < 3; k++) {
      const d = fineDetail(gray.subarray(k * 129600, (k + 1) * 129600));
      assert.equal(differences(d, detail.subarray(k * 129600, (k + 1) * 129600)), 0, `fine detail of print frame ${k}`);
      assert.equal(maskedMedianU8(d, cells), medians[k], `median on the print of frame ${k}`);
    }
    // sums added frame by frame, as _print_map adds them, on three frames
    const three = new PrintSums();
    const visible = fx.u8('print_probe_visible');
    for (let k = 0; k < 3; k++) three.add(gray.subarray(k * 129600, (k + 1) * 129600), detail.subarray(k * 129600, (k + 1) * 129600), visible.subarray(k * 129600, (k + 1) * 129600), level);
    let seen = 0;
    for (let i = 0; i < 3 * 129600; i++) seen += visible[i];
    assert.equal(three.seen.reduce((x, y) => x + y, 0), seen);
    const sums = new PrintSums();
    sums.seen.set(fx.array('sums_seen') as Int32Array); sums.printed.set(fx.array('sums_printed') as Int32Array);
    sums.light.set(fx.f32('sums_light')); sums.light2.set(fx.f32('sums_light2'));
    sums.frames = fx.value<number>('sums_frames');
    assert.equal(differences(printMap(a, cells, sums), map), 0, 'print map');
    void HANDS_PRINT_LEVEL;
  });

  test(`${name}: hands from the trained models' skin and landmarks, and their cover of the print`, () => {
    const frames = fx.manifest.net_frames ? fx.value<number[]>('net_frames') : [];
    if (!frames.length) return;
    const skin = fx.u8('net_skin'), landmarks = fx.f32('net_landmarks'), scores = fx.f32('net_scores'), masks = fx.u8('net_masks');
    const z = zones(a, cells, map);
    const net = fx.f32('hand_net'), inside = fx.f32('hand_net_inside'), page = fx.f32('hand_net_page'), near = fx.f32('hand_net_near');
    frames.forEach((f, k) => {
      const lm = landmarks.subarray(k * 168, (k + 1) * 168), sc = scores.subarray(k * 4, (k + 1) * 4);
      const mask = handMask(skinPatches(skin.subarray(k * 129600, (k + 1) * 129600)), palms(lm, sc));
      assert.equal(differences(mask, masks.subarray(k * 129600, (k + 1) * 129600)), 0, `hand mask of frame ${f}`);
      let t = 0, ins = 0, pg = 0, nr = 0;
      for (let i = 0; i < 129600; i++) if (mask[i]) { t += z.textAll[i]; ins += z.inside[i]; pg += a.mask[i]; nr += z.near[i]; }
      assert.equal(Math.fround(t / z.textArea), net[f], `net at ${f}`);
      assert.equal(Math.fround(ins / z.insideArea), inside[f], `net_inside at ${f}`);
      assert.equal(Math.fround(pg / z.pageArea), page[f], `net_page at ${f}`);
      assert.equal(Math.fround(nr / z.textArea), near[f], `net_near at ${f}`);
    });
  });

  test(`${name}: the stricter colour model and its cover of the print map`, () => {
    const probe = fx.u8('probe_bgr');
    const px = 270 * 480 * 3;
    const probeFrames = fx.value<number[]>('probe_frames');
    const level = fx.value<number>('print_level');
    const page = new Uint8Array(129600);
    for (let i = 0; i < page.length; i++) page[i] = a.inner[i] || map[i] ? 1 : 0;
    const builder = new ModelBuilder(a.median, a.mask, page, { printLevel: level });
    const fine = fx.u8('probe_fine');
    for (let k = 0; k < probeFrames.length; k++) {
      const frame = probe.subarray(k * px, (k + 1) * px);
      const d = fineDetail(bgrToGray(frame));
      assert.equal(differences(d, fine.subarray(k * 129600, (k + 1) * 129600)), 0, `fine detail of probe ${k}`);
      builder.add(frame, d);
    }
    assert.equal(maxAbsDiff(builder.intruder, fx.array('strict_probe_intruder')), 0, 'coloured intruder histogram');
    assert.equal(maxAbsDiff(builder.anyChange, fx.array('strict_probe_any')), 0, 'all-change histogram');
    assert.equal(maxAbsDiff(builder.page, fx.array('strict_probe_page')), 0, 'page histogram');
    const plain = { lut: fx.u8('hand_lut'), samples: fx.value<number>('hand_samples'), usable: fx.value<boolean>('hand_usable') };
    const meter = new HandMeter(handSetup(a, plain));
    const z = zones(a, cells, map);
    const strict = { model: { lut: fx.u8('strict_lut'), samples: 0, usable: fx.value<boolean>('strict_usable') }, region: z.region, wellInside: z.wellInside, mapArea: z.mapArea, level };
    const onMap = fx.f32('hand_on_map');
    probeFrames.forEach((f, k) => {
      const frame = probe.subarray(k * px, (k + 1) * px);
      meter.measure(frame);
      assert.equal(meter.onMap(strict, () => fineDetail(bgrToGray(frame))), onMap[f], `on_map at ${f}`);
    });
  });

  const optional = (key: string) => (fx.manifest[key] ? fx.f32(key) : null);
  const hands: Hands = {
    usable: fx.value<boolean>('hand_usable'), samples: 0, share: fx.f32('hand_share'), onPrint: fx.f32('hand_on_print'), paper: fx.f32('hand_paper'),
    onMap: fx.f32('hand_on_map'), net: optional('hand_net'), netPage: optional('hand_net_page'), netNear: optional('hand_net_near'), netInside: optional('hand_net_inside'),
  };
  // the per-cell brightness sums of the first pass, from the stored thumbnails
  const thumbs = fx.u8('thumbs');
  const lightSums = new Uint16Array(frames * CELLS);
  for (let f = 0; f < frames; f++) {
    for (let gy = 0; gy < 27; gy++) for (let gx = 0; gx < 48; gx++) {
      const t = f * 54 * 96 + 2 * gy * 96 + 2 * gx;
      lightSums[f * CELLS + gy * 48 + gx] = thumbs[t] + thumbs[t + 1] + thumbs[t + 96] + thumbs[t + 97];
    }
  }
  const turns = detectTurns(a);
  const grouped = groupSpreads(turns, a, hands, lightSums);

  test(`${name}: page turns and candidate frames`, () => {
    const ref = fx.value<{ landings: number[]; confirmed: number[]; weak: boolean[]; settled?: number[] }>('turns');
    assert.deepEqual(turns.landings, ref.landings);
    assert.deepEqual(turns.settled ?? turns.landings, ref.settled ?? ref.landings);
    assert.deepEqual(turns.confirmed, ref.confirmed);
    assert.deepEqual(turns.weak, ref.weak);
    const ours = grouped.map(s => { const c = toCandidate(s, a.fps) as unknown as Record<string, unknown>; delete c.review; return c; });
    assert.deepEqual(ours, fx.value('spreads_grouped'));
  });

  const mergeFrames = fx.value<number[]>('merge_frames');
  const gray = fx.u8('merge_gray');
  const { merged, dropped } = mergeRepeats(grouped, a, f => {
    const k = mergeFrames.indexOf(f);
    return gray.subarray(k * 129600, (k + 1) * 129600);
  });
  const candidates: Candidate[] = merged.map(s => toCandidate(s, a.fps));

  test(`${name}: early same-spread merge`, () => {
    assert.deepEqual(dropped.map(d => [d[0], d[1]]), fx.value<number[][]>('merge_dropped').map(d => [d[0], d[1]]));
    // Where a later spread's frame won the merge, Python keeps left_frame, right_frame and hand_inside of the frame that
    // lost (its capture image is that frame); the port follows the frame that won. Everything else is compared as it is.
    const ref = fx.value<Record<string, unknown>[]>('candidates').map(c => {
      if (c.left_frame === c.best) return c;
      return { ...c, left_frame: c.best, right_frame: c.best, hand_inside: (candidates.find(o => o.best === c.best) as unknown as Record<string, unknown>)?.hand_inside };
    });
    assert.deepEqual(candidates, ref);
  });

  test(`${name}: final captures from the same keypoints`, () => {
    assert.equal(differences(printRegion(a.mask, a.median), fx.u8('print_region')), 0, 'print region');
    assert.ok(Math.abs(regionCentreX(fx.u8('print_region')) - fx.value<number>('centre_x')) < 1e-9);
    const xs = fx.f32('feat_x'), ys = fx.f32('feat_y'), desc = fx.u8('feat_desc');
    const offsets = fx.value<number[]>('feat_offsets');
    const centre = fx.value<number>('centre_x');
    const pages = candidates.map((_, k) => {
      const [o0, o1] = [offsets[k], offsets[k + 1]];
      return splitPages({ xs: xs.slice(o0, o1), ys: ys.slice(o0, o1), desc: desc.slice(o0 * 32, o1 * 32) }, centre);
    });
    for (const [i, j, side, value] of fx.value<number[][]>('agree_log')) {
      assert.equal(agreeingPoints(pages[i][side], pages[j][side]), value, `agreeing points ${i}-${j} side ${side}`);
    }
    const sel = selectCaptures(candidates, pages, a.fps, memoAgree());
    const ref = fx.value<{ fixed: number[]; captures: { best: number; repeats: number[]; file: string }[] }>('selection');
    assert.deepEqual(sel.fixed, ref.fixed);
    assert.deepEqual(sel.captures.map(c => ({ best: c.best, repeats: c.repeats, file: c.file })), ref.captures.map(c => ({ best: c.best, repeats: c.repeats, file: c.file })));
  });
  void CELLS;
}
