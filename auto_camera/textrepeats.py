"""Pages of a book that show the same printed page as a page shortly before or after, found by their text.

Selection and the printed-number check leave a second view of some spreads in
the book: a glimpse during a fumbled turn, a leaf in the air, a hand over the
text, or two calm views the keypoints could not tell apart; where the page
numbers cannot be read, the number check cannot remove them. After OCR each
page has its text, and two views of one printed page share most of their
uncommon phrases, however badly one of them reads. A page of such a group is
left out only when another page of the group reads better on both counts (at
least as many words read right, and at least as large a share of its words
read right) and comes from a picture flagged for no more reasons (a hand over
the print, never at rest). Neither count alone picks the better view (more
words is also what a mid-turn view showing two pages has; a larger share is
also what a view with its first lines under the turning leaf has), and the
text does not see a hand beside the lines it covers; leaving out the better
view would make the book worse than keeping both, so where no page is better
on all three, all stay. On the 63 pages the two counts alone left out of
run_20261112_corpus_v9's books, the checkers called the page kept the better
copy on 49 and the page left out on 12; of those 12, four had a hand or a
lifted leaf on the page kept, by my own look, and the flags keep three of
them.

The measure, set on the 48 books of run_20261112_corpus_v9 (every pair at or
above the threshold that was read was the same page; below it, front matter
and appendices of different pages begin to share their phrases):

- words: runs of three or more letters, lower-cased;
- phrases: three words in a row, of those found on at most ``COMMON`` pages of
  the book (a running head, a formula or a recurring name is not evidence);
- two pages within ``REACH`` places of each other, from different pictures,
  each with at least ``LEAST`` such phrases, are the same page when the
  phrases they share are at least ``SHARE`` of the smaller set;
- words read right: words found on at least ``KNOWN`` pages of the book
  (misread words are found nowhere else).

The browser version (web/src/pages/textrepeats.ts) does the same, on the same
numbers.
"""

from __future__ import annotations

import re
from collections import Counter

WORD = re.compile(r"[^\W\d_]{3,}")
REACH = 6        # pages either way: a spread shown again after up to two others
COMMON = 4       # a phrase on more pages than this says nothing about which page
LEAST = 50       # phrases a page must have to be compared (not a title page)
SHARE = 0.3      # share of the smaller page's phrases found in the other
KNOWN = 3        # pages a word must be on to count as read right


def repeated_pages(texts: list[str], pictures: list[object] | None = None, flags: list[int] | None = None) -> list[dict]:
    """The pages to leave out, as ``{"page": i, "kept": k, "share": s}`` (0-based, in page order).

    ``pictures[i]`` names the picture page ``i`` was cut from: the two pages of
    one picture are never the same page. ``flags[i]`` is how many reasons
    that picture is flagged for (none when not given).
    """
    flags = flags if flags is not None else [0] * len(texts)
    words = [WORD.findall(t.lower()) for t in texts]
    on_pages = Counter(w for ws in words for w in set(ws))
    known = [sum(on_pages[w] >= KNOWN for w in ws) for ws in words]
    legible = [k / len(ws) if ws else 0.0 for k, ws in zip(known, words)]
    triples = [{tuple(ws[i:i + 3]) for i in range(len(ws) - 2)} for ws in words]
    phrase_pages = Counter(x for p in triples for x in p)
    phrases = [{x for x in p if phrase_pages[x] <= COMMON} for p in triples]

    # pairs of pages that show the same printed page; never through a third page,
    # which may be a view of two pages during a turn (it would join two different pages)
    same: dict[int, dict[int, float]] = {i: {} for i in range(len(texts))}
    for i, a in enumerate(phrases):
        if len(a) < LEAST:
            continue
        for j in range(max(0, i - REACH), i):
            b = phrases[j]
            if len(b) < LEAST or (pictures is not None and pictures[i] == pictures[j]):
                continue
            s = len(a & b) / min(len(a), len(b))
            if s >= SHARE:
                same[i][j] = same[j][i] = s

    def beats(o: int, m: int) -> bool:
        """Page o reads better than page m on both counts (on a tie in both, the earlier page), from a picture flagged no more."""
        return (known[o] >= known[m] and legible[o] >= legible[m] and flags[o] <= flags[m]
                and (known[o] > known[m] or legible[o] > legible[m] or o < m))

    # from the best read down: a page gives way to a page kept before it that shows the same page and beats it
    kept: list[int] = []
    out = []
    for m in sorted(range(len(texts)), key=lambda k: (-known[k], -legible[k], k)):
        better = [o for o in kept if o in same[m] and beats(o, m)]
        if better:
            o = max(better, key=lambda k: (known[k], legible[k], -k))
            out.append({"page": m, "kept": o, "share": round(same[m][o], 3)})
        else:
            kept.append(m)
    return sorted(out, key=lambda r: r["page"])
