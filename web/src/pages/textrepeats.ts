// Pages of a book that show the same printed page as a page shortly before or
// after, found by their text: auto_camera/textrepeats.py of the Python
// pipeline, on the same numbers (see there for why each is what it is).
//
// After OCR, two views of one printed page (a glimpse during a fumbled turn,
// a leaf in the air, a hand over the text, or two calm views the keypoints
// could not tell apart) share most of their uncommon three-word phrases,
// however badly one of them reads. A page is left out only when a page it
// matches reads better on both counts (words read right, and their share)
// and comes from a picture flagged for no more reasons; otherwise both stay.

/** Python's [^\W\d_]{3,}: letters and the numerals that are not decimal digits (superscripts), three or more. */
const WORD = /[\p{L}\p{Nl}\p{No}]{3,}/gu;
export const REACH = 6;     // pages either way
export const COMMON = 4;    // a phrase on more pages than this says nothing about which page
export const LEAST = 50;    // phrases a page must have to be compared (not a title page)
export const SHARE = 0.3;   // share of the smaller page's phrases found in the other
export const KNOWN = 3;     // pages a word must be on to count as read right

export interface Repeat { page: number; kept: number; share: number }

/**
 * The pages to leave out (0-based, in page order) and the page kept for
 * each. `pictures[i]` names the picture page i was cut from (the two pages of
 * one picture are never the same page); `flags[i]` is how many reasons that
 * picture is flagged for.
 */
export function repeatedPages(texts: string[], pictures?: readonly unknown[], flags?: readonly number[]): Repeat[] {
  const n = texts.length;
  const words = texts.map(t => t.toLowerCase().match(WORD) ?? []);
  const onPages = new Map<string, number>();
  for (const ws of words) for (const w of new Set(ws)) onPages.set(w, (onPages.get(w) ?? 0) + 1);
  const known = words.map(ws => ws.reduce((s, w) => s + ((onPages.get(w) ?? 0) >= KNOWN ? 1 : 0), 0));
  const legible = words.map((ws, i) => (ws.length ? known[i] / ws.length : 0));
  const triples = words.map(ws => { const s = new Set<string>(); for (let i = 0; i + 2 < ws.length; i++) s.add(`${ws[i]}\u0001${ws[i + 1]}\u0001${ws[i + 2]}`); return s; });
  const phrasePages = new Map<string, number>();
  for (const p of triples) for (const x of p) phrasePages.set(x, (phrasePages.get(x) ?? 0) + 1);
  const phrases = triples.map(p => new Set([...p].filter(x => phrasePages.get(x)! <= COMMON)));
  const flag = flags ?? new Array<number>(n).fill(0);

  // pairs of pages that show the same printed page; never through a third
  // page, which may be a view of two pages during a turn
  const same = Array.from({ length: n }, () => new Map<number, number>());
  for (let i = 0; i < n; i++) {
    const a = phrases[i];
    if (a.size < LEAST) continue;
    for (let j = Math.max(0, i - REACH); j < i; j++) {
      const b = phrases[j];
      if (b.size < LEAST || (pictures && pictures[i] === pictures[j])) continue;
      const [small, large] = a.size <= b.size ? [a, b] : [b, a];
      let shared = 0;
      for (const x of small) if (large.has(x)) shared++;
      const s = shared / Math.min(a.size, b.size);
      if (s >= SHARE) { same[i].set(j, s); same[j].set(i, s); }
    }
  }
  const beats = (o: number, m: number) =>
    known[o] >= known[m] && legible[o] >= legible[m] && flag[o] <= flag[m] && (known[o] > known[m] || legible[o] > legible[m] || o < m);
  const rank = (a: number, b: number) => known[b] - known[a] || legible[b] - legible[a] || a - b;

  // from the best read down: a page gives way to a page kept before it that shows the same page and beats it
  const kept: number[] = [];
  const out: Repeat[] = [];
  for (const m of [...Array(n).keys()].sort(rank)) {
    const better = kept.filter(o => same[m].has(o) && beats(o, m));
    if (better.length) {
      const o = better.sort(rank)[0];
      out.push({ page: m, kept: o, share: Math.round(same[m].get(o)! * 1000) / 1000 });
    } else {
      kept.push(m);
    }
  }
  return out.sort((a, b) => a.page - b.page);
}
