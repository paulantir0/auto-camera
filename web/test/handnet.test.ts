// The trained-model helpers and the print map's geometry against the Python
// and OpenCV code, on synthetic inputs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Fixture, differences } from './load.ts';
import { f16round, fineDetail } from '../src/pipeline/handnet.ts';
import { boxPoints, fillConvexPolyClipped, minAreaRect, roundHalfEven } from '../src/pipeline/imgproc.ts';

const fx = new Fixture(new URL('./fixtures/synthetic/', import.meta.url).pathname);

test('fine detail both ways, as handnet.fine_detail', () => {
  assert.equal(differences(fineDetail(fx.u8('img')), fx.u8('fine_img')), 0, 'random image');
  assert.equal(differences(fineDetail(fx.u8('smooth')), fx.u8('fine_smooth')), 0, 'smooth image');
});

test('float16 rounding, as NumPy astype(np.float16)', () => {
  const input = fx.array('f16_in'), out = fx.array('f16_out');
  for (let i = 0; i < input.length; i++) assert.equal(f16round(input[i]), out[i], `${input[i]}`);
});

test('smallest rotated rectangle, its corners and the filled box, as cv2', () => {
  const [count, h, w] = fx.shape('rect_masks');
  const masks = fx.u8('rect_masks'), fills = fx.u8('rect_fills');
  const params = fx.f32('rect_params'), boxes = fx.f32('rect_boxes');
  let exact = 0, filledDiff = 0;
  for (let k = 0; k < count; k++) {
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i < w * h; i++) if (masks[k * w * h + i]) { xs.push(i % w); ys.push(Math.floor(i / w)); }
    const r = minAreaRect(xs, ys);
    const box = boxPoints(r);
    // the same rectangle: its corners as a set (OpenCV may start from another corner when two edges give the same area)
    const ref = Array.from({ length: 4 }, (_, j) => [boxes[(k * 4 + j) * 2], boxes[(k * 4 + j) * 2 + 1]]);
    const near = box.every(([x, y]) => ref.some(([rx, ry]) => Math.abs(rx - x) < 1e-3 && Math.abs(ry - y) < 1e-3));
    assert.ok(near, `blob ${k}: corners ${JSON.stringify(box)} vs ${JSON.stringify(ref)} (params ${Array.from(params.subarray(k * 5, k * 5 + 5))})`);
    if (box.every(([x, y], j) => x === ref[j][0] && y === ref[j][1])) exact++;
    const fill = new Uint8Array(w * h);
    fillConvexPolyClipped(fill, w, h, box.map(([x, y]) => [roundHalfEven(x), roundHalfEven(y)] as [number, number]));
    filledDiff += differences(fill, fills.subarray(k * w * h, (k + 1) * w * h));
  }
  assert.equal(filledDiff, 0, 'filled boxes');
  assert.ok(exact >= count * 0.5, `only ${exact} of ${count} with the very same corner order`);
});
