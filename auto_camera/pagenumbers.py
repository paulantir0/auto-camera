"""Check a sequence of captures against the page numbers printed on them.

This is an audit, not part of frame selection. Each capture contributes the
number of its left-hand page, or nothing if no number could be read. In a
complete sequence that number rises by exactly two per capture, so a jump of
four means a spread is missing and a repeat means a spread was captured twice.
Unread captures are bridged: only the count between two read numbers matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def spread_page(number: int, step: int = 2) -> int:
    """The number that identifies the spread a printed number belongs to.

    In an ordinary book (``step`` 2) left pages carry even numbers and right
    pages odd ones, so both numbers of a spread map to the even one. In a
    facing-page translation (``step`` 1) both pages carry the same number and
    it rises by one per spread.
    """
    return number - (number % 2) if step == 2 else number


SHIFT_COST = 0.6         # cost of one missing spread or one capture too many, in units of one read number
MIN_SUPPORT = 2          # an offset needs this many captures behind it to be considered
JUMP_COST = 3.0          # cost of the numbering starting afresh, however far it jumps
RESTART_SPREADS = 5      # a break of more spreads than this is the numbering starting afresh, not pages lost


def decode_pages(candidates: list[list[int]], step: int = 2) -> list[int | None]:
    """Pick, per capture, the candidate number that fits the sequence.

    A header band often holds other digits besides the page number: section
    ranges in a running head, note numbers, dates. Taken one capture at a time
    they cannot be told apart, but only the real page numbers advance by
    ``step`` from one capture to the next.

    Each candidate implies an offset, ``page - step * position``. In a
    faultless sequence every real page number has the same offset; a missing
    spread raises it by one step from there on, a capture too many (a repeat,
    or a junk frame with no number) lowers it by one. Stray digits have
    offsets of their own. The decoder finds the path of offsets through the
    captures that collects the most candidates while paying ``SHIFT_COST`` for
    every step it shifts, and returns for each capture the page on that path,
    or ``None`` where the capture has no candidate on it.

    The price of a shift stops rising at ``JUMP_COST``. A volume of reprinted
    essays, a book bound with another, or a reader who moves to a later
    chapter all make the numbering start afresh somewhere else, and the run of
    numbers after such a jump is as real as the run before it. Four numbers in
    a row on the new offset pay for the jump; a stray digit or two do not.
    """
    import numpy as np

    count = len(candidates)
    options = [sorted({spread_page(c, step) for c in cands}) for cands in candidates]
    support: dict[int, int] = {}
    for i, pages in enumerate(options):
        for page in pages:
            offset = page - step * i
            support[offset] = support.get(offset, 0) + 1
    offsets = sorted(o for o, n in support.items() if n >= MIN_SUPPORT)
    if not offsets:
        return [None] * count
    index = {o: k for k, o in enumerate(offsets)}
    values = np.array(offsets, dtype=np.float64)
    shift = np.minimum(SHIFT_COST * np.abs(values[:, None] - values[None, :]) / step, JUMP_COST)      # [from, to]
    score = np.zeros(len(offsets))
    back = np.zeros((count, len(offsets)), dtype=np.int32)
    hits = np.zeros((count, len(offsets)), dtype=bool)
    for i, pages in enumerate(options):
        for page in pages:
            k = index.get(page - step * i)
            if k is not None:
                hits[i, k] = True
        if i:
            total = score[:, None] - shift
            back[i] = np.argmax(total, axis=0)
            score = total[back[i], np.arange(len(offsets))]
        score = score + hits[i]
    result: list[int | None] = [None] * count
    k = int(np.argmax(score))
    for i in range(count - 1, -1, -1):
        if hits[i, k]:
            result[i] = offsets[k] + step * i
        if i:
            k = int(back[i, k])
    return result


@dataclass
class Anomaly:
    after: int            # index of the last capture before the break
    before: int           # index of the first capture after it
    page_after: int
    page_before: int
    missing: int = 0      # spreads absent between the two captures
    extra: int = 0        # captures too many between them (a repeat or a backward flip)
    unclear: bool = False # the numbers differ by an odd amount, so one is probably misread

    def as_dict(self) -> dict:
        return {
            "after": self.after, "before": self.before,
            "page_after": self.page_after, "page_before": self.page_before,
            "missing": self.missing, "extra": self.extra, "unclear": self.unclear,
        }


@dataclass
class SequenceReport:
    captures: int
    read: int                       # captures with a usable page number
    step: int = 2                   # 2 for an ordinary book, 1 when both pages share a number
    discarded_misreads: list[int] = field(default_factory=list)
    first_page: int | None = None
    last_page: int | None = None
    span_captures: int = 0          # captures between the first and last read number, inclusive
    span_expected: int = 0          # spreads the page numbers say that span should hold
    missing: int = 0
    extra: int = 0
    unclear: int = 0
    anomalies: list[Anomaly] = field(default_factory=list)
    restarts: list[Anomaly] = field(default_factory=list)   # places where the numbering starts afresh

    def as_dict(self) -> dict:
        return {
            "captures": self.captures, "read": self.read, "step": self.step,
            "discarded_misreads": self.discarded_misreads,
            "first_page": self.first_page, "last_page": self.last_page,
            "span_captures": self.span_captures, "span_expected": self.span_expected,
            "missing": self.missing, "extra": self.extra, "unclear": self.unclear,
            "anomalies": [a.as_dict() for a in self.anomalies],
            "restarts": [a.as_dict() for a in self.restarts],
        }


def sequence_report(pages: list[int | None], step: int = 2) -> SequenceReport:
    """Count missing and extra spreads in a sequence of spread numbers."""
    report = SequenceReport(captures=len(pages), read=0, step=step)
    known = [(i, p) for i, p in enumerate(pages) if p is not None]
    # A lone number that disagrees with two neighbours that agree with each
    # other is a misread, not a skip followed by an equal and opposite repeat.
    offsets = [p - step * i for i, p in known]
    kept: list[tuple[int, int]] = []
    for k, (index, page) in enumerate(known):
        if 0 < k < len(known) - 1 and offsets[k - 1] == offsets[k + 1] != offsets[k]:
            report.discarded_misreads.append(index)
            continue
        kept.append((index, page))
    report.read = len(kept)
    if not kept:
        return report
    report.first_page, report.last_page = kept[0][1], kept[-1][1]
    report.span_captures = kept[-1][0] - kept[0][0] + 1
    section = [kept[0][1]]         # pages since the numbering last started afresh
    sections = [section]
    for (i, p), (j, q) in zip(kept[:-1], kept[1:]):
        steps = j - i
        spreads = (q - p) // step
        if abs(spreads - steps) > RESTART_SPREADS:
            # The numbering starts afresh here. What lies between the two
            # captures is not counted either way; the place is listed.
            report.restarts.append(Anomaly(i, j, p, q, missing=max(0, spreads - steps), extra=max(0, steps - spreads)))
            section = [q]
            sections.append(section)
            continue
        section.append(q)
        if (q - p) % step:
            report.anomalies.append(Anomaly(i, j, p, q, unclear=True))
            report.unclear += 1
            continue
        if spreads > steps:
            report.anomalies.append(Anomaly(i, j, p, q, missing=spreads - steps))
            report.missing += spreads - steps
        elif spreads < steps:
            report.anomalies.append(Anomaly(i, j, p, q, extra=steps - spreads))
            report.extra += steps - spreads
    report.span_expected = sum((max(pages) - min(pages)) // step + 1 for pages in sections)
    return report


def audit_sequence(candidates: list[list[int]]) -> SequenceReport:
    """Decode and check a sequence, working out the book's numbering by itself.

    Both numberings are tried: two per spread (ordinary) and one per spread
    (facing-page translation). The one that explains more captures wins; a tie
    goes to the ordinary one.
    """
    best: SequenceReport | None = None
    for step in (2, 1):
        report = sequence_report(decode_pages(candidates, step), step)
        consistent = report.read - len(report.anomalies)
        if best is None or consistent > best.read - len(best.anomalies):
            best = report
    assert best is not None
    return best


NUMBER_LOOK_BACK = 4     # a repeat is looked for among this many earlier numbered captures


def repeats_by_number(pages: list[int | None], step: int, better, same_number=None) -> list[int]:
    """Positions of captures that repeat a printed page number seen just before.

    ``pages`` are the decoded spread numbers of consecutive captures and
    ``better(i, j)`` says whether capture ``i`` is the better image of the two.
    A capture whose number already appeared among the last few numbered
    captures is a repeat: a disturbed second view of the same spread, or the
    reader flipping back. If the two are neighbours (nothing numbered between
    them) the better image is kept; if the sequence had moved on in between,
    the earlier one is kept, because it sits at the right place. The reverse
    case, a spread glimpsed too early and then shown again in its place, is
    handled first: there the early glimpse is the one dropped.

    ``same_number(i, j)``, when given, must confirm that captures ``i`` and
    ``j`` really show the same printed number in the same place; decoding two
    captures to one spread is not enough by itself, because a stray digit can
    decode to a neighbour's page (a chapter number in a running head did, and
    a clean capture of a different spread was removed as its "repeat").

    Captures without a readable number are never dropped. And nothing is
    dropped next to a gap: if the pages on either side of the pair do not run
    on in single steps, one of the two may be a different spread whose number
    was misread (6 and 8 are easily confused), and removing it would hide a
    missing page.
    """
    count = len(pages)
    dropped: set[int] = set()
    numbered = [i for i in range(count) if pages[i] is not None]
    # A reader who turns two leaves at once sees the wrong spread, flips back,
    # and comes to that spread again in its place. The first, out-of-order
    # sight of it is dropped: its number is higher than the next capture's and
    # turns up again within the next few.
    for n, i in enumerate(numbered[:-1]):
        later = numbered[n + 1:n + 1 + NUMBER_LOOK_BACK]
        if pages[later[0]] < pages[i] and any(pages[j] == pages[i] and (same_number is None or same_number(i, j)) for j in later):
            dropped.add(i)
    known: list[int] = []            # positions of numbered captures kept so far
    for i in range(count):
        page = pages[i]
        if page is None or i in dropped:
            continue
        twin = next((j for j in reversed(known[-NUMBER_LOOK_BACK:]) if pages[j] == page), None)
        if twin is None or (same_number is not None and not same_number(i, twin)):
            known.append(i)
            continue
        highest = max(pages[j] for j in known)
        before = [pages[j] for j in known if j < twin]
        after = next((pages[k] for k in range(i + 1, count) if pages[k] is not None), None)
        gap_before = bool(before) and page - max(before) > step
        gap_after = after is not None and after - highest > step
        if gap_before or gap_after:
            known.append(i)
            continue
        neighbours = all(pages[k] is None or k in dropped for k in range(twin + 1, i))
        if neighbours and better(i, twin):
            dropped.add(twin)
            known[known.index(twin)] = i
        else:
            dropped.add(i)
    return sorted(dropped)


def no_room_between(pages: list[int | None], step: int, flagged: list[bool]) -> list[int]:
    """Positions of flagged, unnumbered captures that sit where the numbering leaves no room for a spread.

    When one numbered capture shows spread ``p`` and the next numbered one
    shows ``p + step`` (or ``p`` again), nothing lies between them in the
    book. A capture between them is a leaf in the air, a hand across the page
    or a glimpse of some other spread. Only captures that are flagged (never
    at rest, or a hand on the print) are returned: one that came to rest clean
    and still has no number is more likely a plate or an inserted leaf, which
    is a page of the book, and stays.
    """
    dropped: list[int] = []
    numbered = [i for i, page in enumerate(pages) if page is not None]
    for a, b in zip(numbered[:-1], numbered[1:]):
        if b - a > 1 and 0 <= pages[b] - pages[a] <= step:
            dropped.extend(i for i in range(a + 1, b) if flagged[i])
    return dropped
