// Port of auto_camera/pagenumbers.py: check a sequence of captures against
// the page numbers printed on them, and find captures that repeat a number.
// Same names, same thresholds, same order of operations.

import { SHIFT_COST, MIN_SUPPORT, JUMP_COST, RESTART_SPREADS, NUMBER_LOOK_BACK } from '../pipeline/constants.ts';

/** The number that identifies the spread a printed number belongs to. */
export function spreadPage(number: number, step = 2): number {
  return step === 2 ? number - (((number % 2) + 2) % 2) : number;
}

/** decode_pages: pick, per capture, the candidate number that fits the sequence. */
export function decodePages(candidates: number[][], step = 2): (number | null)[] {
  const count = candidates.length;
  const options = candidates.map(c => [...new Set(c.map(n => spreadPage(n, step)))].sort((a, b) => a - b));
  const support = new Map<number, number>();
  options.forEach((pages, i) => {
    for (const page of pages) {
      const o = page - step * i;
      support.set(o, (support.get(o) ?? 0) + 1);
    }
  });
  const offsets = [...support.entries()].filter(([, n]) => n >= MIN_SUPPORT).map(([o]) => o).sort((a, b) => a - b);
  if (!offsets.length) return new Array(count).fill(null);
  const K = offsets.length;
  const index = new Map(offsets.map((o, k) => [o, k]));
  // shift[from * K + to]
  const shift = new Float64Array(K * K);
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) shift[a * K + b] = Math.min(SHIFT_COST * Math.abs(offsets[a] - offsets[b]) / step, JUMP_COST);
  let score = new Float64Array(K);
  const back = new Int32Array(count * K);
  const hits = new Uint8Array(count * K);
  for (let i = 0; i < count; i++) {
    for (const page of options[i]) {
      const k = index.get(page - step * i);
      if (k !== undefined) hits[i * K + k] = 1;
    }
    if (i) {
      // total[from, to] = score[from] - shift[from, to]; np.argmax takes the first maximum
      const next = new Float64Array(K);
      for (let b = 0; b < K; b++) {
        let best = -Infinity, arg = 0;
        for (let a = 0; a < K; a++) {
          const v = score[a] - shift[a * K + b];
          if (v > best) { best = v; arg = a; }
        }
        back[i * K + b] = arg;
        next[b] = best;
      }
      score = next;
    }
    for (let k = 0; k < K; k++) score[k] += hits[i * K + k];
  }
  const result: (number | null)[] = new Array(count).fill(null);
  let k = 0;
  for (let j = 1; j < K; j++) if (score[j] > score[k]) k = j;
  for (let i = count - 1; i >= 0; i--) {
    if (hits[i * K + k]) result[i] = offsets[k] + step * i;
    if (i) k = back[i * K + k];
  }
  return result;
}

export interface Anomaly {
  after: number;
  before: number;
  page_after: number;
  page_before: number;
  missing: number;
  extra: number;
  unclear: boolean;
}

export interface SequenceReport {
  captures: number;
  read: number;
  step: number;
  discarded_misreads: number[];
  first_page: number | null;
  last_page: number | null;
  span_captures: number;
  span_expected: number;
  missing: number;
  extra: number;
  unclear: number;
  anomalies: Anomaly[];
  restarts: Anomaly[];
}

const anomaly = (after: number, before: number, page_after: number, page_before: number, more: Partial<Anomaly> = {}): Anomaly =>
  ({ after, before, page_after, page_before, missing: 0, extra: 0, unclear: false, ...more });

const floorDiv = (a: number, b: number) => Math.floor(a / b);

/** sequence_report: count missing and extra spreads in a sequence of spread numbers. */
export function sequenceReport(pages: (number | null)[], step = 2): SequenceReport {
  const report: SequenceReport = {
    captures: pages.length, read: 0, step, discarded_misreads: [], first_page: null, last_page: null,
    span_captures: 0, span_expected: 0, missing: 0, extra: 0, unclear: 0, anomalies: [], restarts: [],
  };
  const known: [number, number][] = [];
  pages.forEach((p, i) => { if (p !== null) known.push([i, p]); });
  const offsets = known.map(([i, p]) => p - step * i);
  const kept: [number, number][] = [];
  known.forEach(([index, page], k) => {
    if (k > 0 && k < known.length - 1 && offsets[k - 1] === offsets[k + 1] && offsets[k + 1] !== offsets[k]) {
      report.discarded_misreads.push(index);
      return;
    }
    kept.push([index, page]);
  });
  report.read = kept.length;
  if (!kept.length) return report;
  report.first_page = kept[0][1];
  report.last_page = kept[kept.length - 1][1];
  report.span_captures = kept[kept.length - 1][0] - kept[0][0] + 1;
  let section = [kept[0][1]];
  const sections = [section];
  for (let n = 0; n + 1 < kept.length; n++) {
    const [i, p] = kept[n], [j, q] = kept[n + 1];
    const steps = j - i;
    const spreads = floorDiv(q - p, step);
    if (Math.abs(spreads - steps) > RESTART_SPREADS) {
      report.restarts.push(anomaly(i, j, p, q, { missing: Math.max(0, spreads - steps), extra: Math.max(0, steps - spreads) }));
      section = [q];
      sections.push(section);
      continue;
    }
    section.push(q);
    if (((q - p) % step + step) % step) {
      report.anomalies.push(anomaly(i, j, p, q, { unclear: true }));
      report.unclear += 1;
      continue;
    }
    if (spreads > steps) {
      report.anomalies.push(anomaly(i, j, p, q, { missing: spreads - steps }));
      report.missing += spreads - steps;
    } else if (spreads < steps) {
      report.anomalies.push(anomaly(i, j, p, q, { extra: steps - spreads }));
      report.extra += steps - spreads;
    }
  }
  report.span_expected = sections.reduce((s, ps) => s + floorDiv(Math.max(...ps) - Math.min(...ps), step) + 1, 0);
  return report;
}

/** audit_sequence: decode and check, trying both numberings (two per spread, then one). */
export function auditSequence(candidates: number[][]): SequenceReport {
  let best: SequenceReport | null = null;
  for (const step of [2, 1]) {
    const report = sequenceReport(decodePages(candidates, step), step);
    if (best === null || report.read - report.anomalies.length > best.read - best.anomalies.length) best = report;
  }
  return best!;
}

/** repeats_by_number: positions of captures that repeat a printed page number seen just before. */
export function repeatsByNumber(pages: (number | null)[], step: number, better: (i: number, j: number) => boolean,
  sameNumber?: (i: number, j: number) => boolean): number[] {
  const count = pages.length;
  const dropped = new Set<number>();
  const numbered = pages.map((p, i) => (p !== null ? i : -1)).filter(i => i >= 0);
  for (let n = 0; n + 1 < numbered.length; n++) {
    const i = numbered[n];
    const later = numbered.slice(n + 1, n + 1 + NUMBER_LOOK_BACK);
    if (pages[later[0]]! < pages[i]! && later.some(j => pages[j] === pages[i] && (!sameNumber || sameNumber(i, j)))) dropped.add(i);
  }
  const known: number[] = [];
  for (let i = 0; i < count; i++) {
    const page = pages[i];
    if (page === null || dropped.has(i)) continue;
    const recent = known.slice(-NUMBER_LOOK_BACK).reverse();
    const twin = recent.find(j => pages[j] === page);
    if (twin === undefined || (sameNumber && !sameNumber(i, twin))) {
      known.push(i);
      continue;
    }
    const highest = Math.max(...known.map(j => pages[j]!));
    const before = known.filter(j => j < twin).map(j => pages[j]!);
    let after: number | null = null;
    for (let k = i + 1; k < count; k++) if (pages[k] !== null) { after = pages[k]; break; }
    const gapBefore = before.length > 0 && page - Math.max(...before) > step;
    const gapAfter = after !== null && after - highest > step;
    if (gapBefore || gapAfter) {
      known.push(i);
      continue;
    }
    let neighbours = true;
    for (let k = twin + 1; k < i; k++) if (!(pages[k] === null || dropped.has(k))) { neighbours = false; break; }
    if (neighbours && better(i, twin)) {
      dropped.add(twin);
      known[known.indexOf(twin)] = i;
    } else {
      dropped.add(i);
    }
  }
  return [...dropped].sort((a, b) => a - b);
}

/** no_room_between: flagged, unnumbered captures where the numbering leaves no room for a spread. */
export function noRoomBetween(pages: (number | null)[], step: number, flagged: boolean[]): number[] {
  const dropped: number[] = [];
  const numbered = pages.map((p, i) => (p !== null ? i : -1)).filter(i => i >= 0);
  for (let n = 0; n + 1 < numbered.length; n++) {
    const a = numbered[n], b = numbered[n + 1];
    const d = pages[b]! - pages[a]!;
    if (b - a > 1 && d >= 0 && d <= step) for (let i = a + 1; i < b; i++) if (flagged[i]) dropped.push(i);
  }
  return dropped;
}
