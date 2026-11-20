"""The videos that can be scanned: the library folder, and files dropped into the page.

A dropped file is written to ``data/uploads/`` a megabyte at a time; it is
never held in memory. Lengths come from the corpus inventory where the file is
listed there, and otherwise from ``ffprobe``, which reads only the file's
header; they are remembered in a small file so each video is probed once.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import threading
import unicodedata
from pathlib import Path

from . import Config
from .jobs import slugify
from .paths import Refused

VIDEO_SUFFIXES = {".mov", ".m4v", ".mp4"}
CHUNK = 1024 * 1024
MAX_UPLOAD_BYTES = 8 * 1024 ** 3        # the largest recording so far is 1.75 GB
KEEP_FREE_BYTES = 5 * 1024 ** 3         # never fill the disk to less than this
SOURCES = ("added", "library")


def clean_upload_name(name: str) -> str:
    """A safe file name for a dropped video, keeping as much of the original as is harmless."""
    if not isinstance(name, str):
        raise Refused(400, "That file has no name.")
    name = unicodedata.normalize("NFC", re.split(r"[\\/]", name)[-1])
    stem, dot, suffix = name.rpartition(".")
    suffix = "." + suffix.lower()
    if not dot or suffix not in VIDEO_SUFFIXES:
        raise Refused(415, "Only MOV, M4V and MP4 videos can be scanned.")
    stem = re.sub(r"[^\w .,'()&+-]", "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")[:120].rstrip(" .")
    if not slugify(stem):
        stem = f"video {stem}".strip()
    return stem + suffix


class Videos:
    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.Lock()
        self._uploading = threading.Lock()
        self._inventory: dict[tuple[str, int], float] | None = None
        self._lengths: dict[str, float | None] | None = None

    # ---- listing -----------------------------------------------------------

    def _folder(self, source: str) -> Path:
        if source == "library":
            return self.config.video_dir
        if source == "added":
            return self.config.uploads_dir
        raise Refused(400, "That video was not found.")

    def _files(self, source: str) -> list[os.DirEntry]:
        try:
            entries = list(os.scandir(self._folder(source)))
        except OSError:
            return []
        keep = [e for e in entries if not e.name.startswith(".") and os.path.splitext(e.name)[1].lower() in VIDEO_SUFFIXES
                and e.is_file()]
        return sorted(keep, key=lambda e: e.name.lower())

    def listing(self) -> list[dict]:
        """Every video that can be scanned: the ones dropped into the page first, then the library."""
        out = []
        for source in SOURCES:
            for entry in self._files(source):
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                out.append({"source": source, "name": entry.name, "bytes": stat.st_size, "slug": slugify(entry.name),
                            "seconds": self._length(Path(entry.path), stat)})
        return out

    def find(self, source: str, name: str) -> Path:
        """The path of a listed video. The name must be exactly one in the folder's listing."""
        for entry in self._files(source):
            if entry.name == name:
                return Path(entry.path)
        raise Refused(404, "That video was not found.")

    # ---- lengths -----------------------------------------------------------

    def _length(self, path: Path, stat: os.stat_result) -> float | None:
        with self._lock:
            if self._inventory is None:
                self._inventory = self._read_inventory()
            listed = self._inventory.get((path.name, stat.st_size))
            if listed is not None:
                return listed
            if self._lengths is None:
                self._lengths = self._read_lengths()
            key = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
            if key not in self._lengths:
                self._lengths[key] = probe_seconds(path)
                self._write_lengths()
            return self._lengths[key]

    def _read_inventory(self) -> dict[tuple[str, int], float]:
        found = {}
        try:
            with open(self.config.inventory, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        found[(row["file"], int(row["size_bytes"]))] = round(float(row["duration_seconds"]), 1)
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            pass
        return found

    def _lengths_file(self) -> Path:
        return self.config.state_dir / "video_lengths.json"

    def _read_lengths(self) -> dict[str, float | None]:
        try:
            data = json.loads(self._lengths_file().read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_lengths(self) -> None:
        try:
            self.config.state_dir.mkdir(parents=True, exist_ok=True)
            self._lengths_file().write_text(json.dumps(self._lengths, indent=1))
        except OSError:
            pass

    # ---- receiving a dropped file -------------------------------------------

    def check_upload(self, name: str, size) -> str:
        """The cleaned name if a file of this name and size can be taken; otherwise says why not."""
        clean = clean_upload_name(name)
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise Refused(400, "That file is empty.")
        if size > MAX_UPLOAD_BYTES:
            raise Refused(413, "That video is too large to copy here (the limit is 8 GB).")
        self.config.uploads_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.config.uploads_dir).free < size + KEEP_FREE_BYTES:
            raise Refused(507, "There is not enough free space on this computer for that video.")
        return clean

    def _taken_slugs(self) -> set[str]:
        taken = {slugify(e.name) for source in SOURCES for e in self._files(source)}
        try:
            taken |= {Path(e.name).stem for e in os.scandir(self.config.proxies_dir)}
        except OSError:
            pass
        return taken

    def _free_name(self, clean: str) -> str:
        """``clean``, or ``clean`` with -2, -3 ... added.

        The pipeline keeps its caches by a name derived from the file name, so
        a new file must not share that name with any video it has seen.
        """
        taken = self._taken_slugs()
        stem, suffix = os.path.splitext(clean)
        candidate, number = clean, 2
        folder = self.config.uploads_dir
        while slugify(candidate) in taken or (folder / candidate).exists() or (folder / (candidate + ".part")).exists():
            candidate = f"{stem}-{number}{suffix}"
            number += 1
        return candidate

    def receive(self, name: str, size: int, stream) -> dict:
        """Write ``size`` bytes from ``stream`` to the uploads folder, a chunk at a time."""
        if not self._uploading.acquire(blocking=False):
            raise Refused(409, "Another video is being copied right now. Wait for it to finish.")
        try:
            final = self._free_name(self.check_upload(name, size))
            target = self.config.uploads_dir / final
            part = self.config.uploads_dir / (final + ".part")
            try:
                with open(part, "xb") as out:
                    remaining = size
                    while remaining:
                        chunk = stream.read(min(CHUNK, remaining))
                        if not chunk:
                            raise Refused(400, "The video did not arrive completely. Try again.")
                        out.write(chunk)
                        remaining -= len(chunk)
                os.replace(part, target)
            except BaseException:
                try:
                    part.unlink()
                except OSError:
                    pass
                raise
            stat = target.stat()
            return {"source": "added", "name": final, "bytes": stat.st_size, "slug": slugify(final),
                    "seconds": self._length(target, stat)}
        finally:
            self._uploading.release()


def probe_seconds(path: Path) -> float | None:
    """A video's length from its header, or None when it cannot be read."""
    try:
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
        return round(float(done.stdout.strip().splitlines()[0]), 1)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
