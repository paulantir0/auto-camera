"""What the server may read, and nothing else.

Every name that arrives in a URL is one path component checked against a
strict pattern, joined below a fixed root, and then resolved and checked to
still be inside that root (so a symbolic link cannot lead out). Nothing in the
server builds a path from a request any other way.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

# One path component: starts with a letter or digit, so never ".", ".." or a hidden file.
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
IMAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.jpe?g\Z", re.IGNORECASE)


class Refused(Exception):
    """A request that will not be served. ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def split_url_path(raw: str) -> list[str]:
    """The components of a URL path, percent-decoded one by one.

    Splitting happens before decoding, so an encoded slash cannot add a level;
    a component that decodes to something containing a separator is refused.
    """
    path = raw.split("?", 1)[0].split("#", 1)[0]
    parts = []
    for piece in path.split("/"):
        if piece == "":
            continue
        try:
            part = unquote(piece, errors="strict") if "%" in piece else piece
        except UnicodeDecodeError:
            raise Refused(400, "That address is not valid.") from None
        if part in (".", "..") or "/" in part or "\\" in part or "\0" in part:
            raise Refused(400, "That address is not valid.")
        parts.append(part)
    return parts


def safe_name(text: str) -> str:
    if not isinstance(text, str) or not SAFE_NAME.match(text):
        raise Refused(400, "That address is not valid.")
    return text


def _inside(path: Path, root: Path) -> Path:
    """``path`` resolved, if it exists and is still below ``root``."""
    try:
        real = path.resolve(strict=True)
        base = root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise Refused(404, "That was not found.") from None
    if real == base or not real.is_relative_to(base):
        raise Refused(404, "That was not found.")
    return real


def run_dir(root: Path, run: str) -> Path:
    """A run folder: a directory directly under ``root``."""
    real = _inside(root / safe_name(run), root)
    if not real.is_dir():
        raise Refused(404, "That was not found.")
    return real


def video_dir(root: Path, run: str, slug: str) -> Path:
    """One video's folder inside a run folder."""
    real = _inside(run_dir(root, run) / safe_name(slug), root)
    if not real.is_dir():
        raise Refused(404, "That was not found.")
    return real


def capture_file(root: Path, run: str, slug: str, name: str) -> Path:
    """One full-size picture: a JPEG in the ``captures`` folder of a video."""
    if not isinstance(name, str) or not IMAGE_NAME.match(name):
        raise Refused(400, "That address is not valid.")
    real = _inside(video_dir(root, run, slug) / "captures" / name, root)
    if not real.is_file():
        raise Refused(404, "That was not found.")
    return real


STATIC_FILES = {
    "index.html": "text/html; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}


def static_file(static_dir: Path, name: str) -> tuple[Path, str]:
    """One of the page's own files, from a fixed list."""
    if name not in STATIC_FILES:
        raise Refused(404, "That was not found.")
    return static_dir / name, STATIC_FILES[name]
