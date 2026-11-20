"""A run folder turned into what the page shows.

Everything here is read from the files the pipeline leaves in
``<run>/<video>/``: ``spreads.json`` (the pictures, in page order, with their
flags), ``audit.json`` (the page-number check) and, for runs made through
``run_video``, ``result.json``. Older run folders have no ``result.json`` and
some have no ``audit.json``; the view says what it can and no more.

The wording of the verdict lives here, in one place, so it can be tested.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import date
from pathlib import Path

from . import paths
from .paths import Refused

FLAG_WORDS = {
    "hand_on_print": "hand over the text",
    "never_at_rest": "page was moving",
}

# The same rule as auto_camera.audit.auditable, repeated here so that the server
# does not have to import the pipeline (and with it numpy and OpenCV).
# tests/test_webapp_runs.py checks the two agree.
MIN_READ = 30
MIN_READ_SHARE = 0.6

JOB_MARKER = "web_job.json"       # written by jobs.py beside a scan started from the page
MAX_JSON_BYTES = 20 * 1024 * 1024


def can_vouch(audit: dict) -> bool:
    """Whether enough page numbers were read for the check to speak for the whole video."""
    final = audit.get("final") or {}
    read = final.get("read") or 0
    return read >= MIN_READ and read >= MIN_READ_SHARE * (audit.get("captures") or 0)


def read_json(path: Path) -> dict | None:
    """A JSON object from a file, or None when it is missing, too large or not an object."""
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


RUN_NAME = re.compile(r"run_(\d{4})(\d{2})(\d{2})_(.+)\Z")


def run_label(name: str) -> str:
    """'run_20261111_corpus' as '11 November 2026 · corpus'."""
    match = RUN_NAME.match(name)
    if not match:
        return name
    try:
        day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return name
    what = match.group(4).replace("_", " ")
    if what == "web":
        what = "scanned from this page"
    return f"{day.day} {day:%B %Y} · {what}"


def _plural(count: int, one: str, many: str | None = None) -> str:
    return f"{count} {one if count == 1 else (many or one + 's')}"


def _page(number) -> int:
    """A spread is known by its left page; the first spread of a book has only page 1, on the right."""
    return 1 if number == 0 else number


def _title(spreads: dict, result: dict | None, slug: str) -> str:
    for source in (spreads.get("video"), (result or {}).get("video")):
        if isinstance(source, str) and source.strip():
            name = re.split(r"[\\/]", source.strip())[-1]
            stem = name.rsplit(".", 1)[0] if "." in name else name
            if stem:
                return stem[:200]
    return slug


def _mark(anomaly: dict, step: int, after: int, before: int, captures: list[dict]) -> dict:
    """What to say between two pictures where the page numbers do not run on."""
    page_after, page_before = _page(anomaly.get("page_after")), _page(anomaly.get("page_before"))
    missing, extra = int(anomaly.get("missing") or 0), int(anomaly.get("extra") or 0)
    first, last = captures[after].get("index", after + 1), captures[before].get("index", before + 1)
    adjacent = before == after + 1
    mark = {"kind": "note", "missing": missing, "extra": extra, "page_after": page_after, "page_before": page_before,
            "after_number": first, "before_number": last}
    if missing:
        mark["kind"] = "gap"
        mark["title"] = f"{_plural(missing, 'spread')} missing"
        if adjacent:
            mark["text"] = f"The page numbers jump from {page_after} to {page_before} here."
            raw_after, raw_before = anomaly.get("page_after"), anomaly.get("page_before")
            if step == 2 and isinstance(raw_after, int) and isinstance(raw_before, int) and raw_before - raw_after == 2 * (missing + 1):
                low, high = raw_after + 2, raw_before - 1
                mark["title"] = f"Missing: pages {low} and {high}" if missing == 1 else f"Missing: pages {low} to {high}"
        else:
            mark["text"] = (f"Somewhere between picture {first} (page {page_after}) and picture {last} (page {page_before}). "
                            "The pictures in between have no number that could be read.")
    elif extra:
        count = before - after + 1
        which = "one is" if extra == 1 else f"{extra} are"
        mark["kind"] = "repeat"
        mark["title"] = "Possible repeat" if extra == 1 else f"{extra} possible repeats"
        span = f"Pictures {first} and {last}" if count == 2 else f"Pictures {first} to {last}"
        if page_after == page_before:
            mark["text"] = f"{span} {'both' if count == 2 else 'all'} show page {page_after}, so {which} probably a repeat."
        else:
            mark["text"] = (f"{span} are {count} pictures for {_plural(count - extra, 'spread')} "
                            f"(pages {page_after} to {page_before}), so {which} probably a repeat.")
    else:
        mark["title"] = "Page numbers unclear"
        mark["text"] = f"The numbers read here do not fit together ({page_after} then {page_before}). One was probably misread."
    return mark


def _restart_mark(restart: dict, after: int, before: int, captures: list[dict]) -> dict:
    """What to say between two pictures where the printed numbering starts afresh.

    A volume of reprinted essays, or a jump to another chapter: not a missing
    page and not a repeat, so it carries no count of either.
    """
    page_after, page_before = _page(restart.get("page_after")), _page(restart.get("page_before"))
    first, last = captures[after].get("index", after + 1), captures[before].get("index", before + 1)
    where = "here" if before == after + 1 else f"between picture {first} and picture {last}"
    return {"kind": "restart", "missing": 0, "extra": 0, "page_after": page_after, "page_before": page_before,
            "after_number": first, "before_number": last, "title": "Numbering starts again",
            "text": f"The page numbering starts again {where}: {page_after}, then {page_before}."}


def page_label(page, step) -> str | None:
    """The printed pages a picture shows: 'p. 46-47', or 'p. 46' when both facing pages carry the same number.

    ``page`` is the number of the left-hand page as the pipeline read it, or
    None when it could not be read; ``step`` is 2 for an ordinary book and 1
    for a facing-page translation. Nothing is said when either is unknown.
    """
    if not isinstance(page, int) or isinstance(page, bool) or page < 0:
        return None
    if step == 1:
        return f"p. {page}"
    if step == 2:
        return "p. 1" if page == 0 else f"p. {page}-{page + 1}"     # the first spread has only page 1, on the right
    return None


def _times(count: int) -> str:
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def left_out(spreads: dict) -> dict | None:
    """Frames the pipeline looked at and did not keep as a page of their own, in one sentence.

    ``not_pages`` is what comes before the book is opened and after it is
    closed; ``shown_elsewhere`` and ``transitional`` are views caught during a
    page turn whose pages are shown properly close by. Older runs have only
    ``transitional`` or none of the three.
    """
    def frames(key: str) -> set:
        entries = spreads.get(key)
        entries = [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
        return {e["frame"] if isinstance(e.get("frame"), int) else (key, i) for i, e in enumerate(entries)}

    closed = frames("not_pages")
    glimpses = (frames("shown_elsewhere") | frames("transitional")) - closed
    if not closed and not glimpses:
        return None
    closed_words = "from before the book was opened or after it was closed"
    glimpse_words = "glimpse during a page turn" if len(glimpses) == 1 else "glimpses during page turns"
    if closed and glimpses:
        text = f"{len(closed) + len(glimpses)} frames were left out: {len(closed)} {closed_words}, {len(glimpses)} {glimpse_words}."
    elif closed:
        text = f"{_plural(len(closed), 'frame')} {closed_words} {'was' if len(closed) == 1 else 'were'} left out."
    else:
        text = f"{len(glimpses)} {glimpse_words} {'was' if len(glimpses) == 1 else 'were'} left out."
    return {"total": len(closed) + len(glimpses), "not_open": len(closed), "glimpses": len(glimpses), "text": text}


def verdict(final: dict | None, vouched: bool, pictures: int, *, gaps: list[dict] = (), restarts: int = 0, note: str = "",
            stale: bool = False, job_state: str = "") -> dict:
    """The page-number verdict in plain words.

    ``state`` is one of ``complete``, ``missing``, ``unreadable`` (the check ran
    but read too few numbers to vouch), ``not_checked`` (no check with this
    run), ``unfinished``, ``running`` and ``empty``. ``short`` is the same
    verdict for a row in a list.

    ``restarts`` is how many times the printed numbering starts afresh. That
    is neither a missing page nor a repeat, but "from 10 to 232" no longer
    describes such a book, and the numbers cannot vouch for the place itself.
    """
    if job_state == "running":
        return {"state": "running", "headline": "This video is being scanned right now.",
                "details": ["The pictures below are not final yet."], "short": "Being scanned now"}
    if job_state == "unfinished":
        return {"state": "unfinished", "headline": "This scan did not finish, so pictures may be missing.",
                "details": ["Scan the video again to get a complete result."], "short": "Did not finish"}
    if pictures == 0:
        return {"state": "empty", "headline": "No pictures were made from this video.", "details": [], "short": "No pictures"}
    if final is None:
        if stale:
            why = "The page-number check saved with this run belongs to a different set of pictures."
        else:
            why = note or "There is no page-number check saved with this run."
        return {"state": "not_checked", "headline": "Page numbers were not checked, so nothing vouches for the order.",
                "details": [why], "short": "Page numbers not checked"}

    read = int(final.get("read") or 0)
    missing, extra = int(final.get("missing") or 0), int(final.get("extra") or 0)
    first, last = _page(final.get("first_page")), _page(final.get("last_page"))
    repeats = (f"{_plural(extra, 'picture')} may show a spread a second time; "
               f"{'it is' if extra == 1 else 'they are'} marked below.") if extra else ""
    again = f"The page numbering starts again {_times(restarts)}, as in a collection of separate pieces." if restarts else ""
    were_read = f"Page numbers were read on {read} of {_plural(pictures, 'picture')}."
    if not vouched:
        details = [f"A page number was read on only {read} of {_plural(pictures, 'picture')}." if read
                   else f"No page number could be read on any of the {_plural(pictures, 'picture')}."]
        if gaps:
            details.append(f"Where numbers were read, {_plural(len(gaps), 'possible gap')} "
                           f"{'is' if len(gaps) == 1 else 'are'} marked below.")
        if restarts:
            details.append(f"Where numbers were read, the page numbering starts again {_times(restarts)}; "
                           f"{'the place is' if restarts == 1 else 'each place is'} marked below.")
        return {"state": "unreadable", "headline": "Page numbers could not be read on this book, so the order is not checked.",
                "details": details, "short": "Order not checked"}
    if not missing and restarts:
        details = [again,
                   f"{'The place is' if restarts == 1 else 'Each place is'} marked below, because just there the numbers "
                   "cannot tell whether a page is missing.",
                   were_read]
        if repeats:
            details.append(repeats)
        return {"state": "complete", "headline": "No page is missing.", "details": details, "short": "No page missing"}
    if not missing:
        details = [f"Page numbers were read on {read} of {_plural(pictures, 'picture')} and they run on without a gap."]
        if repeats:
            details.append(repeats)
        return {"state": "complete", "headline": f"Every page number from {first} to {last} is there.",
                "details": details, "short": f"All pages there ({first} to {last})"}

    places = [g for g in gaps if g.get("missing")]
    total = _plural(missing, "spread")
    if len(places) == 1 and places[0]["missing"] == missing:
        headline = f"{total} missing after page {places[0]['page_after']}"
    elif 1 < len(places) <= 3 and sum(g["missing"] for g in places) == missing:
        parts = [f"after page {g['page_after']}" if g["missing"] == 1 else f"{g['missing']} after page {g['page_after']}" for g in places]
        headline = f"{total} missing: {', '.join(parts[:-1])} and {parts[-1]}"
    else:
        headline = f"{total} missing in {_plural(len(places), 'place')}" if places else f"{total} missing"
    details = ["Each place is marked among the pictures below."]
    if restarts:
        details += [f"{again} {'That place is' if restarts == 1 else 'Those places are'} marked too.", were_read]
    else:
        details.append(f"The page numbers run from {first} to {last}; they were read on {read} of {_plural(pictures, 'picture')}.")
    if repeats:
        details.append(repeats)
    return {"state": "missing", "headline": headline, "details": details, "short": f"{total} missing"}


def _job_state(marker: dict | None, live: bool) -> str:
    """'' for a finished scan (or one not made from the page), else 'running' or 'unfinished'."""
    if not marker:
        return ""
    status = marker.get("status")
    if status == "done":
        return ""
    if status == "running" and live:
        return "running"
    return "unfinished"


def _note(result: dict | None) -> str:
    """Why the page numbers were not checked, when the pipeline said so."""
    text = (result or {}).get("audit_note")
    if not isinstance(text, str) or not text:
        return ""
    if ".venv-ocr" in text or "OCR" in text:
        return "The part that reads page numbers is not installed on this computer."
    return text[:300]


def video_view(root: Path, run: str, slug: str, *, live: bool = False) -> dict:
    """Everything the result page shows for one video of one run.

    ``items`` is the pictures in page order with a mark wherever the printed
    page numbers do not run on (a gap, a possible repeat, or the numbering
    starting afresh), placed after the picture it follows. Each picture says
    which printed pages it shows when the pipeline recorded that.
    ``live`` says a scan of this video into this run is going on right now.

    Newer runs carry more than older ones (``final.restarts`` in
    ``audit.json``; ``page`` on each capture and the lists of left-out frames
    in ``spreads.json``). Every one of those is optional here.
    """
    folder = paths.video_dir(root, run, slug)
    spreads = read_json(folder / "spreads.json")
    if spreads is None:
        raise Refused(404, "There are no results for that video in this run.")
    audit = read_json(folder / "audit.json")
    result = read_json(folder / "result.json")
    job_state = _job_state(read_json(folder / JOB_MARKER), live)

    listed = spreads.get("captures")
    captures = [c for c in listed if isinstance(c, dict)] if isinstance(listed, list) else []
    try:
        on_disk = {entry.name for entry in os.scandir(folder / "captures")}
    except OSError:
        on_disk = set()

    final = audit.get("final") if audit else None
    stale = False
    if not isinstance(final, dict):
        final = None
    elif isinstance(audit.get("captures"), int) and audit["captures"] != len(captures):
        final, stale = None, True          # the check was made for another list of pictures
    vouched = False
    if final is not None:
        from_pipeline = (result or {}).get("audit")
        if isinstance(from_pipeline, dict) and "vouched" in from_pipeline and result.get("captures") == len(captures):
            vouched = bool(from_pipeline["vouched"])
        else:
            vouched = can_vouch(audit)

    position = {c.get("file"): i for i, c in enumerate(captures) if isinstance(c.get("file"), str)}
    marks_after: dict[int, list[dict]] = {}
    beside: dict[int, set[str]] = {}
    marks: list[dict] = []
    restarts = 0
    numbering = None           # 2: a picture shows pages n and n + 1; 1: both pages carry n; None: not known
    if final is not None and not job_state:
        step = final.get("step") if final.get("step") in (1, 2) else 2
        numbering = final.get("step") if final.get("step") in (1, 2) else None
        # ``restarts`` came later than ``anomalies``; a run without the key has none.
        for is_restart, entries in ((False, final.get("anomalies")), (True, final.get("restarts"))):
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict):
                    continue
                restarts += is_restart
                after = position.get(entry.get("after_file"), entry.get("after"))
                before = position.get(entry.get("before_file"), entry.get("before"))
                if not (isinstance(after, int) and isinstance(before, int) and 0 <= after < before < len(captures)):
                    continue
                mark = _restart_mark(entry, after, before, captures) if is_restart else _mark(entry, step, after, before, captures)
                marks.append(mark)
                marks_after.setdefault(after, []).append(mark)
                for i in range(after, before + 1):
                    beside.setdefault(i, set()).add(mark["kind"])

    items: list[dict] = []
    flagged = hand = moving = 0
    for i, capture in enumerate(captures):
        name = capture.get("file")
        name = name if isinstance(name, str) and paths.IMAGE_NAME.match(name) else None
        review = capture.get("review")
        flags = [f for f in review if isinstance(f, str)] if isinstance(review, list) else []
        flagged += bool(flags)
        hand += "hand_on_print" in flags
        moving += "never_at_rest" in flags
        seconds = capture.get("best_seconds")
        number = capture.get("index")
        items.append({
            "kind": "picture",
            "number": number if isinstance(number, int) else i + 1,
            "file": name,
            "on_disk": bool(name) and name in on_disk,
            "seconds": round(seconds, 1) if isinstance(seconds, (int, float)) else None,
            "pages": page_label(capture.get("page"), numbering),
            "flags": flags,
            "labels": [FLAG_WORDS.get(f, f.replace("_", " ")) for f in flags],
            "beside_mark": i in beside,
            "beside_kinds": sorted(beside.get(i, ())),
        })
        items.extend(marks_after.get(i, []))

    gaps = [m for m in marks if m["kind"] == "gap"]
    said = verdict(final, vouched, len(captures), gaps=gaps, restarts=restarts, note=_note(result), stale=stale,
                   job_state=job_state)
    video_seconds = (result or {}).get("video_seconds")
    if not isinstance(video_seconds, (int, float)):
        frames, fps = spreads.get("frames"), spreads.get("fps")
        video_seconds = round(frames / fps, 1) if isinstance(frames, int) and isinstance(fps, (int, float)) and fps > 0 else None
    try:
        version = int((folder / "spreads.json").stat().st_mtime)
    except OSError:
        version = 0
    return {
        "run": run,
        "run_label": run_label(run),
        "slug": slug,
        "title": _title(spreads, result, slug),
        "verdict": said,
        "counts": {
            "pictures": len(captures),
            "needs_look": flagged,
            "hand": hand,
            "moving": moving,
            "gaps": len(gaps),
            "missing": int(final.get("missing") or 0) if final is not None and not job_state else None,
            "repeats": int(final.get("extra") or 0) if final is not None and not job_state else None,
            "restarts": restarts if final is not None and not job_state else None,
            "numbers_read": int(final.get("read") or 0) if final is not None else None,
            "files_missing": sum(1 for item in items if item["kind"] == "picture" and not item["on_disk"]),
        },
        "video_seconds": video_seconds,
        "left_out": None if job_state else left_out(spreads),
        "folder": _for_show(folder / "captures"),
        "version": version,
        "items": items,
    }


def _for_show(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


# Summaries are small and the JSON behind them is not (up to a megabyte per
# video), so a run page of 48 videos is read once and then remembered until a
# file changes.
_summaries: dict[tuple, dict] = {}
_summaries_lock = threading.Lock()
SUMMARY_FILES = ("spreads.json", "audit.json", "result.json", JOB_MARKER, "captures")


def _stamp(folder: Path) -> tuple:
    stamps = []
    for name in SUMMARY_FILES:
        try:
            stamps.append((folder / name).stat().st_mtime_ns)
        except OSError:
            stamps.append(0)
    return tuple(stamps)


def video_summary(root: Path, run: str, slug: str, *, live: bool = False) -> dict:
    """``video_view`` without the pictures, for a row in a list."""
    folder = paths.video_dir(root, run, slug)
    key = (str(folder), _stamp(folder), live)
    with _summaries_lock:
        if key in _summaries:
            return _summaries[key]
    view = video_view(root, run, slug, live=live)
    summary = {k: v for k, v in view.items() if k != "items"}
    with _summaries_lock:
        if len(_summaries) > 4000:
            _summaries.clear()
        _summaries[key] = summary
    return summary


def _video_slugs(folder: Path) -> list[str]:
    slugs = []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return []
    for entry in entries:
        if paths.SAFE_NAME.match(entry.name) and entry.is_dir() and os.path.isfile(os.path.join(entry.path, "spreads.json")):
            slugs.append(entry.name)
    return sorted(slugs)


def run_view(root: Path, run: str, *, live_slug: str | None = None) -> dict:
    """The videos of one run folder, each with its verdict, and totals."""
    folder = paths.run_dir(root, run)
    videos = []
    for slug in _video_slugs(folder):
        try:
            videos.append(video_summary(root, run, slug, live=slug == live_slug))
        except Refused:
            continue
    videos.sort(key=lambda v: v["title"].lower())
    states: dict[str, int] = {}
    for video in videos:
        states[video["verdict"]["state"]] = states.get(video["verdict"]["state"], 0) + 1
    return {
        "run": run,
        "run_label": run_label(run),
        "videos": videos,
        "totals": {"videos": len(videos), "pictures": sum(v["counts"]["pictures"] for v in videos), "states": states},
    }


def run_names(root: Path) -> list[str]:
    """Run folders under ``root``, newest first."""
    try:
        entries = [e for e in os.scandir(root) if paths.SAFE_NAME.match(e.name) and e.is_dir()]
    except OSError:
        return []

    def order(entry) -> tuple:
        match = RUN_NAME.match(entry.name)
        try:
            changed = entry.stat().st_mtime
        except OSError:
            changed = 0.0
        return (match.group(1) + match.group(2) + match.group(3) if match else "", changed)

    return [e.name for e in sorted(entries, key=order, reverse=True)]


def runs_view(root: Path, *, live: tuple[str, str] | None = None) -> list[dict]:
    """Every run folder that holds at least one video, newest first."""
    out = []
    for name in run_names(root):
        try:
            view = run_view(root, name, live_slug=live[1] if live and live[0] == name else None)
        except Refused:
            continue
        if view["totals"]["videos"]:
            out.append({"run": name, "run_label": view["run_label"], "totals": view["totals"]})
    return out


def latest_result(root: Path, slug: str, names: list[str] | None = None) -> dict | None:
    """The newest run that holds this video, for the 'earlier result' link beside a video."""
    if not paths.SAFE_NAME.match(slug):
        return None
    for name in names if names is not None else run_names(root):
        if os.path.isfile(os.path.join(root, name, slug, "spreads.json")):
            try:
                summary = video_summary(root, name, slug)
            except Refused:
                continue
            return {"run": name, "run_label": summary["run_label"], "slug": slug,
                    "state": summary["verdict"]["state"], "short": summary["verdict"]["short"]}
    return None
