"""Small pictures for the grid, made when first asked for and kept on disk.

A capture is 3072x1728 and about 700 KB; a grid of several hundred of them
must not be sent or decoded at that size. The JPEG decoder is asked for a
reduced version directly (``draft``), so making a thumbnail holds about a
megabyte, and at most two are made at once.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from . import Config

THUMB_WIDTH = 480
_slots = threading.BoundedSemaphore(2)


def thumb_path(config: Config, run: str, slug: str, name: str) -> Path:
    """Where the thumbnail of a capture is kept. The three names have been checked by ``paths``."""
    return config.thumbs_dir / run / slug / f"{name}.w{THUMB_WIDTH}.jpg"


def _fresh(target: Path, source: Path) -> bool:
    try:
        return target.stat().st_mtime_ns >= source.stat().st_mtime_ns
    except OSError:
        return False


def thumbnail(config: Config, source: Path, run: str, slug: str, name: str) -> Path:
    """The cached thumbnail of ``source``, made now if it is missing or older than the picture."""
    target = thumb_path(config, run, slug, name)
    if _fresh(target, source):
        return target
    with _slots:
        if not _fresh(target, source):
            _make(source, target)
    return target


def _make(source: Path, target: Path) -> None:
    from PIL import Image

    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with Image.open(source) as image:
            image.draft("RGB", (THUMB_WIDTH, 1))          # decode at 1/2, 1/4 or 1/8 size, no smaller than the width asked for
            small = image.convert("RGB")
            small.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 8), Image.Resampling.LANCZOS)
            small.save(scratch, "JPEG", quality=82)
        os.replace(scratch, target)
    finally:
        try:
            scratch.unlink()
        except OSError:
            pass
