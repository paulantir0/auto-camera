"""Where the pipeline keeps its caches, finds its models and, in the Windows app, its programs.

From source nothing changes: caches under the repository's ``data/cache``,
models in ``data/models``, ffmpeg from the system. In the Windows app built by
PyInstaller (``sys.frozen``) the caches go to
``%LOCALAPPDATA%\\AutoCameraBookScanner\\cache`` and the models and ffmpeg are
the copies bundled in the app (``sys._MEIPASS``).

Two environment variables override the defaults, for tests and for runs that
must leave the usual caches alone:

    AUTO_CAMERA_CACHE    the cache root (proxies/, analysis/, hands/, handnet/, stills/, ...)
    AUTO_CAMERA_MODELS   the folder holding hand_landmarker.task and selfie_multiclass_256x256.tflite

Only the standard library is imported here: ``auto_camera.handdetect`` uses
this module in the separate hand-model environment as well.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "AutoCameraBookScanner"
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ENV = "AUTO_CAMERA_CACHE"
MODELS_ENV = "AUTO_CAMERA_MODELS"


def frozen() -> bool:
    """True inside an app built by PyInstaller."""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """The folder the app's bundled files were unpacked to (PyInstaller's ``_MEIPASS``); the repository from source."""
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return REPO_ROOT


def user_data_root() -> Path:
    """The app's own folder for data that may be thrown away (frozen only)."""
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / APP_NAME
    if sys.platform == "win32":
        return Path.home() / "AppData" / "Local" / APP_NAME
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / APP_NAME


def cache_root() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    if frozen():
        return user_data_root() / "cache"
    return REPO_ROOT / "data" / "cache"


def models_dir() -> Path:
    override = os.environ.get(MODELS_ENV)
    if override:
        return Path(override).expanduser()
    if frozen():
        return bundle_root() / "models"
    return REPO_ROOT / "data" / "models"


def bundled_tools_dir() -> Path | None:
    """The folder holding the bundled ffmpeg and ffprobe, frozen only; from source the system's are used."""
    return bundle_root() / "ffmpeg" if frozen() else None


# Read once, when the pipeline is imported. A process that wants another
# cache root sets AUTO_CAMERA_CACHE before importing any of the pipeline.
CACHE_ROOT = cache_root()
MODELS_DIR = models_dir()
