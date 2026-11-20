// Port of the parts of auto_camera/audit.py and export.py that turn the
// numbers read off the pictures into the printed-number check: which
// readings can be page numbers, the decoded sequence, the captures that
// repeat a number, and the gaps.

import { captureRank } from '../pipeline/spreads.ts';
import { LINE_END_GAIN, MIN_READ, MIN_READ_SHARE, PLACE_WIDTH, PROXY_HEIGHT, PROXY_WIDTH, ROW_SLACK, STATIC_MIN, STATIC_SHARE } from '../pipeline/constants.ts';
import { textRows, usualTextRows, type Analysis } from '../pipeline/analysis.ts';
import { pyRound } from '../pipeline/pyround.ts';
import { auditSequence, decodePages, noRoomBetween, repeatsByNumber, sequenceReport, type Anomaly, type SequenceReport } from './pagenumbers.ts';
import type { Token } from './read.ts';

/** Readings per frame. */
export type Readings = Map<number, Token[]>;

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

/** export.band_rows: per frame, where the head and the foot of the print may lie (fractions of the frame height). */
export function bandRows(a: Analysis, frames: number[]): Map<number, number[] | null> {
  const here = new Map(frames.map(f => [f, textRows(a, f, null)] as const));
  const known = [...here.values()].filter((r): r is [number, number] => r !== null);
  const out = new Map<number, number[] | null>();
  if (!known.length) {
    const usual = usualTextRows(a);
    for (const f of frames) out.set(f, usual ? [usual[0], usual[0], usual[1], usual[1]] : null);
    return out;
  }
  const top = median(known.map(r => r[0])), bottom = median(known.map(r => r[1]));
  const slack = ROW_SLACK * (bottom - top);
  for (const [f, rows] of here) {
    const first = rows && rows[0] <= top + slack ? rows[0] : top;
    const last = rows && rows[1] >= bottom - slack ? rows[1] : bottom;
    out.set(f, [Math.min(first, top), Math.max(first, top), Math.min(last, bottom), Math.max(last, bottom)]);
  }
  return out;
}

/** export.page_box: bounding box of the page region as fractions of the frame (x0, x1, y0, y1). */
export function pageBox(mask: Uint8Array): number[] {
  let r0 = -1, r1 = -1, c0 = PROXY_WIDTH, c1 = -1;
  for (let y = 0; y < PROXY_HEIGHT; y++) {
    for (let x = 0; x < PROXY_WIDTH; x++) {
      if (!mask[y * PROXY_WIDTH + x]) continue;
      if (r0 < 0) r0 = y;
      r1 = y;
      if (x < c0) c0 = x;
      if (x > c1) c1 = x;
    }
  }
  if (r0 < 0) return [0, 1, 0, 1];
  return [c0 / PROXY_WIDTH, c1 / PROXY_WIDTH, r0 / PROXY_HEIGHT, r1 / PROXY_HEIGHT];
}

const placeKey = (t: Token) => `${parseInt(t.text, 10)}|${t.band}|${pyRound(t.x / PLACE_WIDTH, 0)}|${t.line_end ? 1 : 0}`;

/** _fixed_places: (number, band, place, line end) read on a large share of the stills. */
function fixedPlaces(readings: Readings, frames: number[]): Set<string> {
  const seen = new Map<string, number>();
  for (const f of frames) for (const key of new Set((readings.get(f) ?? []).map(placeKey))) seen.set(key, (seen.get(key) ?? 0) + 1);
  const limit = Math.max(STATIC_MIN, STATIC_SHARE * frames.length);
  return new Set([...seen].filter(([, n]) => n > limit).map(([k]) => k));
}

/** usable_tokens: the readings of each still that could be its page number. */
function usableTokens(readings: Readings, frames: number[], lineEnds = true): Map<number, Token[]> {
  const fixed = fixedPlaces(readings, frames);
  return new Map(frames.map(f => [f, (readings.get(f) ?? []).filter(t => !fixed.has(placeKey(t)) && (lineEnds || !t.line_end))]));
}

/** page_number_tokens: with or without the numbers at the ends of head and foot lines, whichever explains clearly more. */
function pageNumberTokens(readings: Readings, frames: number[], order: number[]): Map<number, Token[]> {
  const choices = [false, true].map(lineEnds => {
    const tokens = usableTokens(readings, frames, lineEnds);
    const report = auditSequence(order.map(f => (tokens.get(f) ?? []).map(t => parseInt(t.text, 10))));
    return { value: report.read - report.anomalies.length, tokens };
  });
  return choices[1].value > LINE_END_GAIN * choices[0].value ? choices[1].tokens : choices[0].tokens;
}

/** What the check needs of a capture. */
export interface Checked {
  best: number;
  hand_on_print: number;
  detail: number;
  best_motion: number;
  review: string[];
}

/** audit.capture_quality (spreads.capture_rank), compared as Python compares tuples. */
function better(a: Checked, b: Checked): boolean {
  const qa = captureRank(a), qb = captureRank(b);
  return qa[0] !== qb[0] ? qa[0] > qb[0] : qa[1] > qb[1];
}

export interface NumberCheck {
  /** Positions (in the list given) of the captures the check removes. */
  removed: number[];
  /** The printed number of each kept capture's left page (or of both, when they share one), or null. */
  pages: (number | null)[];
  /** The kept captures' sequence: gaps, extra captures, restarts. Positions are among the kept ones. */
  report: SequenceReport;
  /** The sequence as selection left it, before anything was removed. */
  before: SequenceReport;
  /** Whether enough numbers were read to vouch for the sequence (audit.auditable). */
  vouched: boolean;
}

/**
 * audit.audit_one without the files: the readings of the captures' stills,
 * the captures as selection chose them (in order), and the frames whose
 * readings decide which places are fixed (Python uses every candidate's).
 */
export function numberCheck(readings: Readings, captures: Checked[], frames: number[], numberCheckOn = true): NumberCheck {
  const tokens = pageNumberTokens(readings, frames, captures.map(c => c.best));
  const digits = (c: Checked) => [...new Set((tokens.get(c.best) ?? []).map(t => parseInt(t.text, 10)))].sort((a, b) => a - b);
  const sameNumber = (i: number, j: number) => (tokens.get(captures[i].best) ?? []).some(a => (tokens.get(captures[j].best) ?? []).some(b =>
    a.text === b.text && a.band === b.band && Math.abs(a.x - b.x) <= 1.5 * PLACE_WIDTH));
  const selection = auditSequence(captures.map(digits));
  let drop = new Set<number>();
  if (numberCheckOn && selection.read >= 10) {
    const pages = decodePages(captures.map(digits), selection.step);
    drop = new Set(repeatsByNumber(pages, selection.step, (i, j) => better(captures[i], captures[j]), sameNumber));
    for (const i of noRoomBetween(pages, selection.step, captures.map(c => c.review.length > 0))) drop.add(i);
  }
  const kept = captures.filter((_, k) => !drop.has(k));
  const decoded = kept.length ? decodePages(kept.map(digits), selection.step) : [];
  const report = kept.length ? sequenceReport(decoded, selection.step) : selection;
  return {
    removed: [...drop].sort((a, b) => a - b),
    pages: decoded,
    report,
    before: selection,
    vouched: report.read >= MIN_READ && report.read >= MIN_READ_SHARE * kept.length,
  };
}

export type { Anomaly, SequenceReport };
