"""Check one video's captures against printed page numbers, and write its final folder.

The OCR itself runs in its own environment (``.venv-ocr``, see ``ocr.py``) and
its readings are cached per video and frame under ``data/cache/ocr``. Everything
here works from that cache.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .export import capture_still, still_name
from .pagenumbers import audit_sequence, decode_pages, no_room_between, repeats_by_number, sequence_report
from .paths import CACHE_ROOT
from .proxy import REPO_ROOT
from .spreads import capture_rank

OCR_CACHE = CACHE_ROOT / "ocr_v4"     # v4: bands span from each still's own head and foot of the print to their usual place in the video; head and foot lines are read at their ends
OCR_PYTHON = REPO_ROOT / ".venv-ocr" / "bin" / "python"
MIN_READ = 30             # numbers read, below which a video's sequence is not vouched for
MIN_READ_SHARE = 0.6      # and the share of its captures with a number
STATIC_SHARE = 0.08       # a number read on more than this share of a video's stills is not a page number
STATIC_MIN = 6            # ... and on more than this many
PLACE_WIDTH = 0.06        # two readings are "in the same place" within this share of the band's width
LINE_END_GAIN = 1.15      # numbers at line ends are used when they explain this many times more captures


def tokens_for(slug: str) -> dict[str, list[dict]]:
    path = OCR_CACHE / f"{slug}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def capture_quality(capture: dict) -> tuple[float, float, float]:
    """Higher is better: see ``spreads.capture_rank``."""
    return capture_rank(capture)


def ocr_jobs(run: Path, slugs: list[str]) -> list[dict]:
    """The candidate stills of these videos that have not been read yet."""
    jobs = []
    for slug in slugs:
        path = run / slug / "spreads.json"
        summary = json.loads(path.read_text())
        have = tokens_for(slug)
        if not have or any(len(c.get("text_rows") or []) != 4 for c in summary["candidates"]):
            # Written by an earlier version: work the bands out again from the analysis.
            from .analysis import analyse
            from .export import band_rows
            fresh = band_rows(analyse(slug), [c["best"] for c in summary["candidates"]])
            for key in ("candidates", "captures", "captures_before_number_check"):
                for capture in summary.get(key, []):
                    capture["text_rows"] = fresh.get(capture["best"])
            path.write_text(json.dumps(summary, indent=1))
        rows = {c["best"]: c.get("text_rows") for c in summary["candidates"]}
        for frame in sorted({c["best"] for c in summary["candidates"]}):
            if str(frame) not in have:
                jobs.append({"slug": slug, "frame": frame, "box": summary["page_box"], "rows": rows.get(frame),
                             "path": str(Path(summary["stills_dir"]) / still_name(frame))})
    return jobs


def _fixed_places(cache: dict[str, list[dict]], frames: list[int]) -> set[tuple[int, str, int, bool]]:
    """(number, band, place, at a line end) combinations read on a large share of a video's stills.

    The same number in the same place on many stills is something fixed in the
    picture or in the running head: a calendar on the wall behind the book, a
    chapter number. It is never a page number. The place is part of the key,
    so a real page 8 is not thrown out because chapter 8 has a long running
    head.
    """
    seen: dict[tuple[int, str, int, bool], int] = {}
    for frame in frames:
        for key in {_place(t) for t in cache.get(str(frame), [])}:
            seen[key] = seen.get(key, 0) + 1
    limit = max(STATIC_MIN, STATIC_SHARE * len(frames))
    return {key for key, count in seen.items() if count > limit}


def _place(token: dict) -> tuple[int, str, int, bool]:
    return int(token["text"]), token["band"], int(round(token["x"] / PLACE_WIDTH)), bool(token.get("line_end"))


def usable_tokens(cache: dict[str, list[dict]], frames: list[int], line_ends: bool = True) -> dict[int, list[dict]]:
    """The tokens of each still that could be its page number."""
    fixed = _fixed_places(cache, frames)
    return {frame: [t for t in cache.get(str(frame), []) if _place(t) not in fixed and (line_ends or not t.get("line_end"))] for frame in frames}


def page_number_tokens(cache: dict[str, list[dict]], frames: list[int], captures: list[dict]) -> dict[int, list[dict]]:
    """Tokens per still, with or without the numbers read at the ends of head and foot lines.

    A page number printed beside a running title is only found at a line end
    ("46  REFORMATION THOUGHT"). But the ends of running heads also carry
    chapter and verse numbers, which mislead the decoder. So the sequence is
    decoded both ways over ``captures`` and the line ends are used only when
    they explain clearly more of them.
    """
    order = [c["best"] for c in captures]
    choices = []
    for line_ends in (False, True):
        tokens = usable_tokens(cache, frames, line_ends)
        report = audit_sequence([[int(t["text"]) for t in tokens.get(frame, [])] for frame in order])
        choices.append((report.read - len(report.anomalies), tokens))
    (plain, without), (more, with_ends) = choices
    return with_ends if more > LINE_END_GAIN * plain else without


def page_number_candidates(cache: dict[str, list[dict]], frames: list[int], captures: list[dict] | None = None) -> dict[int, list[int]]:
    """The numbers read on each of these stills that could be its page number."""
    tokens = page_number_tokens(cache, frames, captures) if captures is not None else usable_tokens(cache, frames)
    return {frame: sorted({int(t["text"]) for t in found}) for frame, found in tokens.items()}


def run_ocr(jobs: list[dict], run: Path, workers: int = 1) -> None:
    """Read the stills in ``jobs`` in the OCR environment and add them to the cache."""
    if not jobs:
        return
    job_file = run / "ocr_jobs.json"
    job_file.write_text(json.dumps(jobs))
    subprocess.run([str(OCR_PYTHON), str(REPO_ROOT / "tools" / "ocr_stills.py"), "--jobs", str(job_file),
                    "--cache-dir", str(OCR_CACHE), "--workers", str(workers)], check=True)


def auditable(audit: dict) -> bool:
    """Whether enough numbers were read for the audit to vouch for this video's sequence."""
    final = audit["final"]
    return final["read"] >= MIN_READ and final["read"] >= MIN_READ_SHARE * audit["captures"]


def audit_one(run: Path, slug: str, number_check: bool = True) -> dict:
    """Check one video's captures against printed page numbers.

    With ``number_check``, captures that repeat their neighbour's printed
    number are then removed (see ``pagenumbers.repeats_by_number``), and so
    are flagged captures without a number that sit between two numbers with
    no room for a spread between them (``pagenumbers.no_room_between``). The
    run folder is rewritten and the sequence is checked again. Both results
    are kept: ``before_number_check`` is what selection alone produced.
    """
    path = run / slug / "spreads.json"
    summary = json.loads(path.read_text())
    captures = summary.get("captures_before_number_check", summary["captures"])
    tokens = page_number_tokens(tokens_for(slug), [c["best"] for c in summary["candidates"]], captures)
    digits = lambda item: sorted({int(t["text"]) for t in tokens.get(item["best"], [])})

    def same_number(i: int, j: int) -> bool:
        """Both captures show one number in the same place: the same printed page number."""
        return any(a["text"] == b["text"] and a["band"] == b["band"] and abs(a["x"] - b["x"]) <= 1.5 * PLACE_WIDTH
                   for a in tokens.get(captures[i]["best"], []) for b in tokens.get(captures[j]["best"], []))

    out = {"slug": slug, "candidates": len(summary["candidates"]), "hand_model": summary.get("hand_model")}
    selection = audit_sequence([digits(i) for i in captures])
    out["before_number_check"] = selection.as_dict()
    out["captures_before_number_check"] = len(captures)
    kept = captures
    dropped: list[dict] = []
    if number_check and selection.read >= 10:
        pages = decode_pages([digits(i) for i in captures], selection.step)
        drop = set(repeats_by_number(pages, selection.step, lambda i, j: capture_quality(captures[i]) > capture_quality(captures[j]), same_number))
        drop |= set(no_room_between(pages, selection.step, [bool(c.get("review")) for c in captures]))
        kept = [c for k, c in enumerate(captures) if k not in drop]
        dropped = [c for k, c in enumerate(captures) if k in drop]
    final = [dict(c) for c in kept]
    for number, capture in enumerate(final, 1):
        capture["index"] = number
        capture["file"] = f"sequence_{number:04d}_t{capture['best'] / summary['fps']:07.2f}s.jpg"
    decoded = decode_pages([digits(i) for i in final], selection.step) if final else []
    report = sequence_report(decoded, selection.step) if final else selection
    for capture, page in zip(final, decoded):
        capture["page"] = page          # the printed number of its left page (of either page when both carry the same), or None
    out["final"] = report.as_dict()
    for anomaly in out["final"]["anomalies"] + out["final"]["restarts"]:
        anomaly["after_file"] = final[anomaly["after"]]["file"]
        anomaly["before_file"] = final[anomaly["before"]]["file"]
        anomaly["seconds_between"] = round((final[anomaly["before"]]["best"] - final[anomaly["after"]]["best"]) / summary["fps"], 1)
    out["captures"] = len(final)
    out["removed_by_number"] = len(dropped)
    out["hand_on_print"] = sum(1 for c in final if "hand_on_print" in shown_reasons(c))
    out["never_at_rest"] = sum(1 for c in final if "never_at_rest" in c.get("review", []))
    summary["captures_before_number_check"] = captures
    summary["captures"] = final
    summary["removed_by_number"] = [c["best"] for c in dropped]
    summary["spreads"] = len(final)
    path.write_text(json.dumps(summary, indent=1))
    (run / slug / "audit.json").write_text(json.dumps(out, indent=1))
    return out


def shown_reasons(capture: dict) -> list[str]:
    """The review reasons a person is told about: "hand_on_print" only for a hand well inside the print (``hand_inside``)."""
    from .export import HAND_ON_PRINT
    reasons = list(capture.get("review", []))
    if "hand_on_print" in reasons and capture.get("hand_inside", capture.get("hand_on_print", 0.0)) <= HAND_ON_PRINT:
        reasons.remove("hand_on_print")
    return reasons


def write_final(video_dir: Path, stills: Path) -> int:
    """Rebuild ``captures/`` and ``needs_review.txt`` of one video from its ``spreads.json``.

    ``stills`` is the folder of decoded frames (``f000123.jpg``); captures are
    hard links into it. Returns how many captures need a look.
    """
    summary = json.loads((video_dir / "spreads.json").read_text())
    captures = summary["captures"]
    target = video_dir / "captures"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for capture in captures:
        link = target / capture["file"]
        source = capture_still(capture, stills)
        try:
            link.hardlink_to(source)
        except OSError:
            shutil.copy2(source, link)
    flagged = [c for c in captures if shown_reasons(c)]
    lines = [f"{c['file']}\t{', '.join(shown_reasons(c))}" for c in flagged]
    audit_path = video_dir / "audit.json"
    if audit_path.exists():
        final = json.loads(audit_path.read_text())["final"]
        for anomaly in final["anomalies"]:
            if anomaly["missing"]:
                lines.append(f"{anomaly['after_file']}\t{anomaly['missing']} spread(s) missing after this one "
                             f"(printed numbers go {anomaly['page_after']} then {anomaly['page_before']})")
            elif anomaly["extra"]:
                lines.append(f"{anomaly['after_file']}\t{anomaly['extra']} capture(s) too many between here and {anomaly['before_file']}")
        for restart in final.get("restarts", []):
            lines.append(f"{restart['after_file']}\tthe page numbering starts afresh after this one "
                         f"(printed numbers go {restart['page_after']} then {restart['page_before']}, {restart['seconds_between']} s apart)")
    checks_path = video_dir / "eye_check.json"
    if checks_path.exists():
        # What vision-model checkers said of each capture, when a run has been shown to them; a second opinion, not truth.
        checks = json.loads(checks_path.read_text())
        named = {c["file"] for c in captures}
        for name, check in checks.items():
            if name in named and check["verdict"] != "clean":
                lines.append(f"{name}\tchecker by eye: {check['verdict'].replace('_', ' ')}" + (f" ({check['reason']})" if check.get("reason") else ""))
    (video_dir / "needs_review.txt").write_text("\n".join(sorted(lines)) + ("\n" if lines else ""))
    return len(flagged)
