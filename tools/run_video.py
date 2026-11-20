#!/usr/bin/env python3
"""The whole pipeline on one video file, with the page-number check.

    tools/run_capped.sh .venv/bin/python tools/run_video.py VIDEO --run ~/Pictures/auto-camera-production/run_YYYYMMDD_name
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_camera.pipeline import run_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--run", type=Path, required=True, help="run folder; a subfolder for the video is written")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-audit", action="store_true", help="skip reading the printed page numbers")
    parser.add_argument("--ocr-workers", type=int, default=2)
    args = parser.parse_args()
    result = run_video(args.video, args.run.expanduser(), use_gpu=not args.no_gpu, audit=not args.no_audit, ocr_workers=args.ocr_workers,
                       progress=lambda stage, detail="": print(f"[{stage}] {detail}", flush=True))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
