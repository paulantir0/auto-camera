#!/usr/bin/env python3
"""OCR the page-number bands of many stills and store the result per video and frame.

Runs in .venv-ocr. Input is a JSON list of {"slug", "frame", "path", "box"}.
Results are merged into CACHE_DIR/<slug>.json, keyed by frame number, so a
frame is only ever read once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_camera import ocr  # noqa: E402


def _work(job: dict) -> tuple[str, int, list[dict]]:
    return job["slug"], job["frame"], ocr.read_still(job["path"], job["box"], rows=job.get("rows"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict[str, dict[str, list[dict]]] = defaultdict(dict)

    def flush() -> None:
        for slug, frames in results.items():
            path = args.cache_dir / f"{slug}.json"
            merged = json.loads(path.read_text()) if path.exists() else {}
            merged.update(frames)
            path.write_text(json.dumps(merged))
        results.clear()

    with Pool(args.workers, initializer=ocr.init_engine, initargs=(1,)) as pool:
        for done, (slug, frame, tokens) in enumerate(pool.imap_unordered(_work, jobs, chunksize=8), 1):
            results[slug][str(frame)] = tokens
            if done % 500 == 0:
                flush()
                print(f"  ocr {done}/{len(jobs)}  {time.time() - started:.0f}s", flush=True)
    flush()
    print(f"ocr: {len(jobs)} stills in {time.time() - started:.0f}s with {args.workers} workers", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
