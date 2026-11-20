"""The book scanner as a page in the browser, served from this PC only.

    .venv/bin/python tools/web_app.py        # then open http://127.0.0.1:8770
    .venv/bin/python tools/web_app.py --lan  # also for the home network (no login; see network.py)

The server is small on purpose. It never decodes video and never imports the
pipeline: each scan is a subprocess started through ``tools/run_capped.sh``,
one at a time, and everything shown afterwards is read from the files that
``auto_camera.pipeline.run_video`` leaves in the run folder.

    paths.py    what may be served, and nothing else
    runs.py     a run folder turned into what the page shows (verdict, counts, pictures, gaps)
    jobs.py     starting, following and cancelling one scan
    videos.py   the list of videos, and receiving a dropped file in chunks
    thumbs.py   small pictures for the grid, made on demand and kept on disk
    network.py  who is answered when the app is opened to the home network
    server.py   the HTTP side
    static/     the page itself (plain HTML, CSS and JavaScript, no build step)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8770


@dataclass(frozen=True)
class Config:
    """Every folder the app reads or writes. Tests point these at temporary folders."""

    repo: Path
    runs_root: Path          # run folders live directly under this; the only place images are served from
    video_dir: Path          # the video library; read only
    uploads_dir: Path        # videos dropped into the page
    thumbs_dir: Path         # cached grid pictures
    state_dir: Path          # small files of the app's own (video lengths, the running job's pid)
    static_dir: Path
    inventory: Path          # data/corpus/video_inventory.csv: lengths of the library videos
    proxies_dir: Path        # only looked at to keep an uploaded video's name from colliding with a cached one

    @classmethod
    def default(cls) -> "Config":
        home = Path.home()
        return cls(
            repo=REPO_ROOT,
            runs_root=home / "Pictures" / "auto-camera-production",
            video_dir=home / "Videos" / "auto-camera-data",
            uploads_dir=REPO_ROOT / "data" / "uploads",
            thumbs_dir=REPO_ROOT / "data" / "cache" / "thumbs",
            state_dir=REPO_ROOT / "data" / "cache" / "webapp",
            static_dir=REPO_ROOT / "webapp" / "static",
            inventory=REPO_ROOT / "data" / "corpus" / "video_inventory.csv",
            proxies_dir=REPO_ROOT / "data" / "cache" / "proxies",
        )

    @classmethod
    def under(cls, base: Path) -> "Config":
        """A self-contained layout below ``base``, for tests."""
        base = Path(base)
        return cls(
            repo=REPO_ROOT,
            runs_root=base / "runs",
            video_dir=base / "videos",
            uploads_dir=base / "uploads",
            thumbs_dir=base / "thumbs",
            state_dir=base / "state",
            static_dir=REPO_ROOT / "webapp" / "static",
            inventory=base / "video_inventory.csv",
            proxies_dir=base / "proxies",
        )
