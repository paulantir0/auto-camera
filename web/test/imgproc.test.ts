// The primitives against OpenCV and NumPy, on synthetic images (no video data).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Fixture, differences, maxAbsDiff } from './load.ts';
import * as ip from '../src/pipeline/imgproc.ts';
import { bgrToGray, bgrImageToLab } from '../src/pipeline/colour.ts';
import { findValleys } from '../src/pipeline/analysis.ts';

const fx = new Fixture(new URL('./fixtures/synthetic', import.meta.url).pathname);
const W = 480, H = 270;
const img = fx.u8('img'), smooth = fx.u8('smooth'), binary = fx.u8('binary');

test('INTER_AREA with whole-number factors is exact', () => {
  assert.equal(differences(ip.resizeAreaInt(img, W, H, 96, 54), fx.u8('area_5')), 0);
  assert.equal(differences(ip.resizeAreaInt(img, W, H, 48, 27), fx.u8('area_10')), 0);
  assert.equal(differences(ip.resizeAreaInt(img, W, H, 240, 135), fx.u8('area_2')), 0);
});

test('INTER_AREA with a fractional factor is within one level', () => {
  const out = ip.resizeArea(fx.u8('big'), 384, 216, 154, 86);
  assert.ok(maxAbsDiff(out, fx.u8('area_frac')) <= 1, `max diff ${maxAbsDiff(out, fx.u8('area_frac'))}`);
  assert.ok(differences(out, fx.u8('area_frac')) / out.length < 0.01);
});

test('Laplacian, box blur and Sobel magnitude', () => {
  assert.equal(differences(ip.laplacianAbs(img, W, H), fx.u8('laplacian')), 0);
  assert.equal(differences(ip.laplacianAbs(smooth, W, H), fx.u8('laplacian_smooth')), 0);
  assert.equal(differences(ip.boxBlur3(img, W, H), fx.u8('blur3')), 0);
  assert.ok(maxAbsDiff(ip.sobelMagnitude(smooth, W, H), fx.f32('sobel_mag')) < 1e-4);
});

test('binary morphology with square kernels', () => {
  for (const k of [3, 5, 7, 21, 25]) {
    const kern = ip.rectKernel(k, k);
    assert.equal(differences(ip.erode(binary, W, H, kern), fx.u8(`erode_${k}`)), 0, `erode ${k}`);
    assert.equal(differences(ip.dilate(binary, W, H, kern), fx.u8(`dilate_${k}`)), 0, `dilate ${k}`);
    assert.equal(differences(ip.open(binary, W, H, kern), fx.u8(`open_${k}`)), 0, `open ${k}`);
    assert.equal(differences(ip.close(binary, W, H, kern), fx.u8(`close_${k}`)), 0, `close ${k}`);
  }
});

test('elliptic structuring elements and dilation with them', () => {
  for (const [kw, kh] of [[61, 51], [21, 35], [7, 7]]) {
    const k = ip.ellipseKernel(kw, kh);
    const dense = new Uint8Array(kw * kh);
    for (let i = 0; i < kh; i++) for (let j = k.start[i]; j < k.end[i]; j++) dense[i * kw + j] = 1;
    assert.equal(differences(dense, fx.u8(`ellipse_${kw}x${kh}`)), 0, `kernel ${kw}x${kh}`);
    assert.equal(differences(ip.dilate(binary, W, H, k), fx.u8(`dilate_ellipse_${kw}x${kh}`)), 0, `dilate ${kw}x${kh}`);
  }
});

test('connected components have the same areas', () => {
  for (const conn of [4, 8] as const) {
    const cc = ip.connectedComponents(binary, W, H, conn);
    const areas = Array.from(cc.areas.slice(1)).sort((a, b) => a - b);
    assert.deepEqual(areas, fx.value<number[]>(`cc_${conn}_areas_sorted`));
  }
});

test('convex hull fill is the same as fillConvexPoly', () => {
  const pts = fx.array('hull_points') as Int32Array;
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < pts.length; i += 2) { xs.push(pts[i]); ys.push(pts[i + 1]); }
  const out = new Uint8Array(W * H);
  ip.fillConvexPoly(out, W, H, ip.convexHull(xs, ys));
  const ref = fx.u8('hull_fill');
  let area = 0;
  for (const v of ref) area += v;
  const diff = differences(out, ref);
  assert.equal(diff, 0, `${diff} pixels differ of ${area}`);
});

test('Gaussian blur of a float histogram', () => {
  assert.ok(maxAbsDiff(ip.gaussianBlur64(fx.array('hist') as Float64Array, 64, 64, 1.0), fx.array('hist_blur')) < 1e-12);
});

test('nearest-neighbour scaling', () => {
  assert.equal(differences(ip.resizeNearest(fx.u8('grid'), 48, 27, 480, 270), fx.u8('grid_up')), 0);
  const region = fx.u8('region').map(v => v * 255);
  assert.equal(differences(ip.resizeNearest(region, 480, 270, 1536, 864), fx.u8('region_up')), 0);
});

test('BGR to Lab and to gray as OpenCV', () => {
  const colours = fx.u8('colours');
  const lab = bgrImageToLab(colours);
  const ref = fx.u8('colours_lab');
  const diff = differences(lab, ref);
  assert.ok(maxAbsDiff(lab, ref) <= 1, `max ${maxAbsDiff(lab, ref)}`);
  assert.equal(diff, 0, `${diff} of ${ref.length} Lab values differ`);
  assert.equal(differences(bgrToGray(colours), fx.u8('colours_gray')), 0);
});

test('percentile as NumPy', () => {
  const vals = fx.f32('pct_values');
  for (const [q, v] of Object.entries(fx.value<Record<string, number>>('percentiles'))) {
    assert.ok(Math.abs(ip.percentile(vals, Number(q)) - v) < 1e-6, `q=${q}`);
  }
});

test('find_valleys', () => {
  assert.deepEqual(findValleys(fx.f32('valley_signal')), fx.value('valleys'));
});

test('gray-and-shrink in one pass, straight from a strided plane', () => {
  const identity = Uint8Array.from({ length: 256 }, (_, i) => i);
  // the plane sits at an offset with a wider stride, as a decoded frame's luma does
  const padded = (src: Uint8Array, w: number, h: number) => {
    const out = new Uint8Array(7 + (w + 5) * h);
    for (let y = 0; y < h; y++) out.set(src.subarray(y * w, (y + 1) * w), 7 + y * (w + 5));
    return out;
  };
  assert.equal(differences(ip.lutAreaFromPlane(padded(img, W, H), 7, W + 5, W, H, identity, 240, 135), fx.u8('area_2')), 0);
  const frac = ip.lutAreaFromPlane(padded(fx.u8('big'), 384, 216), 7, 389, 384, 216, identity, 154, 86);
  assert.ok(maxAbsDiff(frac, fx.u8('area_frac')) <= 1);
  assert.ok(differences(frac, fx.u8('area_frac')) / frac.length < 0.01);
});
