"""The whole fast pipeline on one video, behind one call.

    from auto_camera.pipeline import run_video
    result = run_video(Path("book.mov"), Path("~/Pictures/auto-camera-production/run_X").expanduser())

``run_video`` is the entry point for anything that drives the pipeline from
outside (a command-line tool, the browser app). The stages it calls are
described in the README; what they do inside may change, this call should not.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import audit as audit_module
from . import book, handdetect
from .analysis import analyse
from .export import DEFAULT_STILLS_DIR, contact_sheets, write_run
from .hands import measure
from .handnet import DEFAULT_HANDNET_DIR
from .proxy import DEFAULT_PROXY_DIR, build_proxy, slugify
from .spreads import segment

STAGES = ["proxy", "analysis", "hands", "turns", "stills", "numbers", "finish", "pages"]
HAND_MODELS = [handdetect.MODEL, handdetect.SEGMENTER]
PRINT_REGION_GROW = 25      # proxy pixels the print map is enlarged by to find lines of text in a still (the book creeps)


def find_hands(slug: str, workers: int) -> str:
    """Run the trained hand models on every frame of the proxy (``auto_camera/handdetect.py``), if they and MediaPipe are here.

    The proxy is split into ``workers`` stretches, each run in a fresh worker
    process (started, not forked, so that it works the same in the Windows
    app and never inherits this process's OpenCV threads).

    Returns a note when they could not be run; the pipeline then sees hands
    by colour only.
    """
    if (DEFAULT_HANDNET_DIR / f"{slug}.npz").exists():
        return ""
    if not all(model.exists() for model in HAND_MODELS):
        return f"hand models not run: {HAND_MODELS[0].parent} is missing them (see the README, quick start)"
    if not handdetect.mediapipe_available():
        return "hand models not run: MediaPipe is not installed in this environment"
    parts = max(1, workers)
    os.environ.setdefault("GLOG_minloglevel", "2")       # read by the workers when they start
    with ProcessPoolExecutor(parts, mp_context=multiprocessing.get_context("spawn"), initializer=handdetect.quiet_worker) as pool:
        handdetect.detect(slug, DEFAULT_PROXY_DIR, DEFAULT_HANDNET_DIR, parts, pool)
    return ""


def run_video(video: Path, run: Path, *, use_gpu: bool = True, audit: bool = True, ocr_workers: int = 2,
              progress: Callable[[str, str], None] | None = None, pages: bool = True, workers: int = 4) -> dict:
    """Turn one video into one still per spread under ``run/<slug>/``.

    ``progress(stage, detail)`` is called as each stage in ``STAGES`` starts.
    With ``audit``, printed page numbers are read and the sequence is checked;
    that needs the OCR environment and is skipped, with a note in the result,
    when it is not installed.

    The folder gets ``captures/`` (the images, in page order), ``spreads.json``,
    ``needs_review.txt``, ``review/contact_sheet_NNN.jpg`` and, with the audit,
    ``audit.json``. With ``pages`` it also gets ``pages/`` (each page flat and
    cropped, two per spread) and ``book.pdf`` (searchable when Tesseract is
    installed, with ``book.txt``). The returned dict is also written there as
    ``result.json``. ``workers`` is how many processes the hand models and
    the page flattening may use.
    """
    say = progress or (lambda stage, detail="": None)
    started = time.time()
    seconds: dict[str, float] = {}

    def stage(name: str, detail: str = "") -> None:
        now = time.time()
        if seconds:
            last = next(reversed(seconds))
            seconds[last] = round(now - seconds[last], 1)
        seconds[name] = now
        say(name, detail)

    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"no such video: {video}")
    slug = slugify(video.name)
    out = Path(run) / slug
    out.mkdir(parents=True, exist_ok=True)

    stage("proxy", "making a small copy of the video")
    proxy = DEFAULT_PROXY_DIR / f"{slug}.mp4"
    if not proxy.exists() or proxy.stat().st_mtime < video.stat().st_mtime:
        build_proxy(video, proxy, use_gpu=use_gpu, threads=2 if use_gpu else 4)

    stage("analysis", "measuring motion and sharpness in every frame")
    analysis = analyse(slug)
    stage("hands", "finding hands over the page")
    hand_note = find_hands(slug, workers)
    hands = measure(analysis)
    stage("turns", "finding page turns and choosing a frame for each spread")
    spreads = segment(analysis, hands)
    stage("stills", f"saving {len(spreads)} full-size frames and removing repeats")
    stills = DEFAULT_STILLS_DIR / slug
    summary = write_run(analysis, spreads, video, out, use_gpu=use_gpu, hand_model=hands.usable, sheets=False)

    audit_result = None
    audit_note = ""
    if audit and not audit_module.OCR_PYTHON.exists():
        audit_note = "page numbers not checked: the OCR environment (.venv-ocr) is not installed"
    elif audit:
        stage("numbers", "reading printed page numbers")
        audit_module.run_ocr(audit_module.ocr_jobs(Path(run), [slug]), Path(run), workers=ocr_workers)
        audit_result = audit_module.audit_one(Path(run), slug)
        summary = json.loads((out / "spreads.json").read_text())

    stage("finish", "writing the final folder")
    to_review = audit_module.write_final(out, stills)
    contact_sheets(analysis, summary["captures"], out / "review")
    book_result = None
    if pages:
        stage("pages", "flattening each page and binding the book")
        region = None
        if hands.where is not None and hands.where.any():
            region = cv2.dilate(hands.where.astype(np.uint8), np.ones((2 * PRINT_REGION_GROW + 1,) * 2, np.uint8)) > 0
        files = book.write_pages(out, region, workers=workers)
        book_result = book.write_pdf(files, out / book.PDF_NAME, workers=workers)
    stage("done")
    seconds.pop("done")

    result = {
        "slug": slug,
        "video": str(video),
        "folder": str(out),
        "video_seconds": round(analysis.frames / analysis.fps, 1),
        "candidates": len(summary["candidates"]),
        "captures": len(summary["captures"]),
        "to_review": to_review,
        "hand_model": hands.usable,
        "seconds": seconds,
        "total_seconds": round(time.time() - started, 1),
        "audit": None,
        "audit_note": audit_note,
        "hand_note": hand_note,
        "book": book_result,
    }
    if audit_result is not None:
        final = audit_result["final"]
        result["audit"] = {
            "vouched": audit_module.auditable(audit_result),
            "numbers_read": final["read"],
            "first_page": final["first_page"],
            "last_page": final["last_page"],
            "missing": final["missing"],
            "extra": final["extra"],
            "unclear": final["unclear"],
            "gaps": [{"after_file": a["after_file"], "before_file": a["before_file"], "page_after": a["page_after"],
                      "page_before": a["page_before"], "missing": a["missing"], "extra": a["extra"]} for a in final["anomalies"]],
            "removed_as_repeats_by_number": audit_result["removed_by_number"],
        }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result
