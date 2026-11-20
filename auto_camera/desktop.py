"""What the desktop app (``auto_camera_gui.py``) does behind its window.

- ``prepare_environment``: the bundled ffmpeg first on PATH (Windows app), and
  Tesseract added when it is installed in its usual place, so the PDF is
  searchable.
- ``check_gpu``: whether FFmpeg can decode and shrink on an NVIDIA card.
- ``ScanJob``: one scan (``pipeline.run_video``) in a child process, its
  progress coming back through a queue, and Stop ending it and everything it
  started.
- ``summarize``: the result in a few plain sentences.
- ``self_test``: the checks the Windows build runs on itself.

Nothing of the pipeline is imported when this module is: in the Windows app
every helper process starts by running the app's script, and the scan's
process must be free to choose its cache folder before the pipeline reads it
(``auto_camera/paths.py``).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from webapp.jobs import STAGE_WORDS, slugify

from . import paths

VIDEO_SUFFIXES = (".mov", ".mp4", ".m4v", ".avi", ".mkv")
KEEP_VIDEO_CACHES = 3          # the Windows app keeps the working files of this many videos
GPU_CHECK_SECONDS = 60
TESSERACT_HINT_WINDOWS = ("Installing Tesseract makes it searchable: the UB Mannheim build for Windows, "
                          "with the English, French, German, Latin and Greek languages (eng, fra, deu, lat, grc, ell) ticked.")
TESSERACT_HINT_OTHER = "Installing Tesseract, with the eng, fra, deu, lat, grc and ell languages, makes it searchable."
LANGUAGE_NAMES = {"eng": "English", "fra": "French", "deu": "German", "lat": "Latin", "grc": "ancient Greek",
                  "ell": "modern Greek", "ita": "Italian"}


def default_workers(cores: int | None = None) -> int:
    """Worker processes for the hand models and the page flattening: up to four, leaving one core free."""
    cores = cores if cores is not None else (os.cpu_count() or 2)
    return max(1, min(4, cores - 1))


def stage_words(key: str, detail: str = "") -> str:
    """A stage of ``pipeline.STAGES`` in the words the app shows."""
    if key in STAGE_WORDS:
        return STAGE_WORDS[key]
    return (detail[:1].upper() + detail[1:]) if detail else key


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {sec:02d} s"
    return f"{sec} s"


# ---- folders and programs --------------------------------------------------------------

def pictures_dir() -> Path:
    """The user's Pictures folder (on Windows wherever it was moved to, OneDrive included)."""
    if sys.platform == "win32":
        try:
            import ctypes

            class GUID(ctypes.Structure):
                _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                            ("Data4", ctypes.c_ubyte * 8)]

            # FOLDERID_Pictures, {33E28130-4E1E-4676-835A-98395C3BC3BB}
            folder_id = GUID(0x33E28130, 0x4E1E, 0x4676, (ctypes.c_ubyte * 8)(0x83, 0x5A, 0x98, 0x39, 0x5C, 0x3B, 0xC3, 0xBB))
            found = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(found)) == 0:
                value = found.value
                ctypes.windll.ole32.CoTaskMemFree(found)
                if value:
                    return Path(value)
        except Exception:
            pass
    return Path.home() / "Pictures"


def default_output_dir(today: date | None = None) -> Path:
    """Pictures/AutoCamera/YYYYMMDD: where a day's scans go unless another folder is chosen."""
    return pictures_dir() / "AutoCamera" / f"{(today or date.today()):%Y%m%d}"


def _exe(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def find_tesseract(environ: dict | None = None) -> Path | None:
    """Tesseract on PATH or, on Windows, where its installers put it."""
    environ = os.environ if environ is None else environ
    found = shutil.which("tesseract", path=environ.get("PATH"))
    if found:
        return Path(found)
    if sys.platform != "win32":
        return None
    bases = [environ.get("ProgramFiles") or r"C:\Program Files", environ.get("ProgramFiles(x86)"),
             str(Path(environ["LOCALAPPDATA"]) / "Programs") if environ.get("LOCALAPPDATA") else None]
    for base in bases:
        if base:
            candidate = Path(base) / "Tesseract-OCR" / "tesseract.exe"
            if candidate.is_file():
                return candidate
    return None


def _add_to_path(environ: dict, folder: Path, *, first: bool) -> None:
    parts = [p for p in environ.get("PATH", "").split(os.pathsep) if p]
    if str(folder) in parts:
        parts.remove(str(folder))
    parts = [str(folder), *parts] if first else [*parts, str(folder)]
    environ["PATH"] = os.pathsep.join(parts)


def prepare_environment(environ: dict | None = None) -> dict:
    """Put the programs the pipeline calls on PATH; returns where each was found (None: not found).

    In the Windows app the bundled ffmpeg and ffprobe go first, so that a
    different FFmpeg installed on the computer is not used by mistake. From
    source the system's are used. Tesseract is added where it was installed.
    """
    environ = os.environ if environ is None else environ
    tools = paths.bundled_tools_dir()
    if tools is not None and (tools / _exe("ffmpeg")).is_file():
        _add_to_path(environ, tools, first=True)
    tesseract = find_tesseract(environ)
    if tesseract is not None and shutil.which("tesseract", path=environ.get("PATH")) is None:
        _add_to_path(environ, tesseract.parent, first=False)
    where = {name: shutil.which(name, path=environ.get("PATH")) for name in ("ffmpeg", "ffprobe", "tesseract")}
    return where


def hide_console_windows() -> None:
    """On Windows, start every helper program (ffmpeg, Tesseract) without a console window of its own.

    The app has no console, so without this each call of ffmpeg would flash
    a black window. Every process of the Windows app runs this first.
    """
    if sys.platform != "win32" or getattr(subprocess.Popen, "_no_console_window", False):
        return
    original = subprocess.Popen

    class QuietPopen(original):  # type: ignore[misc, valid-type]
        _no_console_window = True

        def __init__(self, *args, **kwargs):
            kwargs["creationflags"] = (kwargs.get("creationflags") or 0) | subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    subprocess.Popen = QuietPopen  # type: ignore[misc]


def ensure_std_streams(log_name: str) -> None:
    """A windowed app has no standard output; give it a log file so that nothing that writes there fails."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        folder = paths.user_data_root() / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        stream = open(folder / log_name, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        stream = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


# ---- the NVIDIA check ------------------------------------------------------------------

def check_gpu(ffmpeg: str = "ffmpeg", timeout: float = GPU_CHECK_SECONDS) -> tuple[bool, str]:
    """(usable, why): FFmpeg lists the ``cuda`` hardware decoder and a tiny clip decodes and shrinks on it.

    The clip is made here with FFmpeg's own test pattern and decoded the way
    the pipeline makes its proxy (``-hwaccel cuda``, frames kept on the card,
    ``scale_cuda``), so a computer that passes can run the GPU path.
    """
    def run(command: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)

    try:
        listed = run([ffmpeg, "-hide_banner", "-hwaccels"])
    except FileNotFoundError:
        return False, "ffmpeg was not found"
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"ffmpeg could not be asked ({error.__class__.__name__})"
    if listed.returncode != 0 or "cuda" not in listed.stdout.split():
        return False, "this ffmpeg has no NVIDIA (CUDA) decoder"
    try:
        with tempfile.TemporaryDirectory(prefix="gpu-check-") as tmp:
            clip = Path(tmp) / "pattern.mp4"
            made = run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=11", "-frames:v", "4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)])
            if made.returncode != 0 or not clip.is_file():
                return False, "ffmpeg could not make a test clip"
            decoded = run([ffmpeg, "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(clip),
                           "-vf", "scale_cuda=160:90,hwdownload,format=nv12", "-f", "null", "-"])
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"the NVIDIA test did not finish ({error.__class__.__name__})"
    if decoded.returncode != 0:
        return False, "no NVIDIA card or driver that ffmpeg can use"
    return True, "NVIDIA decoding works"


def detect_gpu(ffmpeg: str = "ffmpeg") -> bool:
    return check_gpu(ffmpeg)[0]


# ---- one video ---------------------------------------------------------------------------

def video_info(path: Path) -> dict:
    """Size, frame rate, length and frame count of a video, from ffprobe."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout
    data = json.loads(out)
    if not data.get("streams"):
        raise ValueError("this file has no video in it")
    stream, fmt = data["streams"][0], data.get("format", {})
    num, den = (stream.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(fmt.get("duration") or 0.0)
    frames = int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else int(round(duration * fps))
    return {"width": int(stream.get("width") or 0), "height": int(stream.get("height") or 0), "fps": fps, "frames": frames,
            "duration": duration, "codec": stream.get("codec_name", ""), "size": int(fmt.get("size") or path.stat().st_size)}


def first_frame_png(path: Path, width: int = 640) -> bytes:
    """The first frame of a video as PNG bytes, ``width`` pixels across."""
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:-2",
                           "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout


def video_cache_name(video: Path) -> str:
    """A cache folder name for one video file: its pipeline name and a fingerprint of the file.

    The pipeline names its caches after the file name alone; phones and
    cameras reuse names (IMG_0001.MOV), so the Windows app gives each file
    its own cache folder.
    """
    stat = video.stat()
    key = f"{video.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8", "replace")
    return f"{(slugify(video.name) or 'video')[:40]}-{hashlib.sha1(key).hexdigest()[:10]}"


def prune_caches(base: Path, keep: int, current: Path | None = None) -> list[Path]:
    """Delete all but the ``keep`` most recently used video cache folders under ``base``; returns those deleted."""
    if not base.is_dir():
        return []
    folders = [p for p in base.iterdir() if p.is_dir() and len(p.name) > 11 and p.name[-11] == "-"
               and all(c in "0123456789abcdef" for c in p.name[-10:]) and p != current]
    folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for folder in folders[max(0, keep - (1 if current is not None else 0)):]:
        shutil.rmtree(folder, ignore_errors=True)
        removed.append(folder)
    return removed


# ---- a scan ------------------------------------------------------------------------------

@dataclass
class ScanSettings:
    video: str
    run_dir: str                 # the video's results go to run_dir/<its name>/
    use_gpu: bool = False
    workers: int = 2
    pages: bool = True
    audit: bool | None = None    # None: only from source, when the OCR environment (.venv-ocr) is there


def _plain_error(error: BaseException) -> str:
    if isinstance(error, FileNotFoundError) and getattr(error, "filename", None) in ("ffmpeg", "ffprobe", "tesseract"):
        return f"{error.filename} was not found. The app needs it to read the video."
    if isinstance(error, MemoryError):
        return "The computer ran out of memory."
    if isinstance(error, subprocess.CalledProcessError):
        program = Path(str(error.cmd[0])).name if error.cmd else "a helper program"
        said = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        said = " ".join(said.strip().splitlines()[-2:])[:300]
        return f"{program} failed (exit code {error.returncode})" + (f": {said}" if said else ".")
    text = str(error).strip()
    return f"{error.__class__.__name__}: {text}" if text else error.__class__.__name__


def _gpu_failure(error: BaseException) -> bool:
    return isinstance(error, subprocess.CalledProcessError) and "cuda" in " ".join(str(part) for part in (error.cmd or []))


def run_scan(settings: ScanSettings, emit: Callable[[tuple], None]) -> dict:
    """Scan one video, telling ``emit`` each stage; returns ``summarize`` of the result.

    Called in the scan's own process (``scan_process``). In the Windows app
    each video gets its own cache folder (see ``video_cache_name``) unless
    AUTO_CAMERA_CACHE says where; that has to be settled before the pipeline
    is imported, which is why it is imported here and not at the top.
    """
    video = Path(settings.video)
    if not video.is_file():
        raise FileNotFoundError(f"The video is not there any more: {video}")
    if paths.frozen() and not os.environ.get(paths.CACHE_ENV):
        if "auto_camera.proxy" in sys.modules:
            raise RuntimeError("the pipeline was imported before its cache folder was chosen")
        base = paths.cache_root()
        folder = base / video_cache_name(video)
        folder.mkdir(parents=True, exist_ok=True)
        os.utime(folder)
        prune_caches(base, KEEP_VIDEO_CACHES, current=folder)
        os.environ[paths.CACHE_ENV] = str(folder)      # for the worker processes the pipeline starts
        paths.CACHE_ROOT = folder

    from . import audit as audit_module
    from .pipeline import run_video

    audit = settings.audit if settings.audit is not None else (not paths.frozen() and audit_module.OCR_PYTHON.exists())
    notes: list[str] = []

    def progress(stage: str, detail: str = "") -> None:
        emit(("stage", stage, detail, time.time()))

    def scan(use_gpu: bool) -> dict:
        return run_video(video, Path(settings.run_dir), use_gpu=use_gpu, audit=audit, progress=progress,
                         pages=settings.pages, workers=max(1, int(settings.workers)))

    try:
        result = scan(settings.use_gpu)
    except Exception as error:
        if not (settings.use_gpu and _gpu_failure(error)):
            raise
        note = "The NVIDIA card could not decode this video, so it was read on the CPU instead."
        notes.append(note)
        emit(("note", note))
        result = scan(False)
    return summarize(result, notes)


def scan_process(settings: dict, events) -> None:
    """The scan's child process: runs ``run_scan`` and sends ("stage", ...), ("note", ...) and one ("done", summary) or ("error", message, details)."""
    hide_console_windows()
    ensure_std_streams("scan.log")
    prepare_environment()
    stage = {"key": ""}

    def emit(event: tuple) -> None:
        if event[0] == "stage":
            stage["key"] = event[1]
        events.put(event)

    try:
        summary = run_scan(ScanSettings(**settings), emit)
    except BaseException as error:      # the window must hear of every way this ends
        where = f" while {stage_words(stage['key']).lower()}" if stage["key"] else ""
        events.put(("error", f"The scan stopped with an error{where}. {_plain_error(error)}", traceback.format_exc()))
    else:
        events.put(("done", summary))


def stop_tree(pid: int, wait: float = 5.0) -> None:
    """End a process and every process it started (the hand models' workers, ffmpeg, Tesseract)."""
    import psutil

    try:
        top = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        top.suspend()               # so that it starts nothing new while its children are being listed
    except psutil.Error:
        pass
    try:
        family = [top, *top.children(recursive=True)]
    except psutil.Error:
        family = [top]
    for process in family:
        try:
            process.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(family, timeout=wait)


class ScanJob:
    """One scan in a child process, as the window sees it."""

    def __init__(self, settings: ScanSettings):
        context = multiprocessing.get_context("spawn")
        self.settings = settings
        self.queue = context.Queue()
        self.process = context.Process(target=scan_process, args=(asdict(settings), self.queue), name="book-scan")
        self.started = 0.0
        self.ended: float | None = None
        self.stopped = False
        self.outcome: tuple | None = None    # the ("done", ...) or ("error", ...) event

    @property
    def folder(self) -> Path:
        """Where this video's results are written."""
        return Path(self.settings.run_dir) / slugify(Path(self.settings.video).name)

    @property
    def running(self) -> bool:
        return self.ended is None

    def start(self) -> "ScanJob":
        self.started = time.time()
        self.process.start()
        return self

    def poll(self) -> list[tuple]:
        """The events that have come in since the last call; the last is ("done"/"error"/"stopped", ...) once it has ended."""
        out = self._drain()
        if self.ended is None and self.outcome is None and not self.process.is_alive():
            out += self._drain()
            if self.outcome is None:
                self.process.join(1)
                code = self.process.exitcode
                self.outcome = ("error", f"The scan ended unexpectedly (exit code {code}). The computer may have run out of memory.", "")
                out.append(self.outcome)
        if self.outcome is not None and self.ended is None:
            self.ended = time.time()
            self.process.join(2)
        return out

    def _drain(self) -> list[tuple]:
        out = []
        while True:
            try:
                event = self.queue.get_nowait()
            except queue.Empty:
                return out
            except (EOFError, OSError):
                return out
            if event[0] in ("done", "error"):
                self.outcome = event
            out.append(event)

    def stop(self) -> None:
        """End the scan now. Its folder is left as far as it got."""
        if self.ended is not None:
            return
        self.stopped = True
        if self.process.pid is not None:
            stop_tree(self.process.pid)
        self.process.join(5)
        self.outcome = ("stopped",)
        self.ended = time.time()


def run_in_child(settings: ScanSettings, on_event: Callable[[tuple], None] | None = None, poll_seconds: float = 0.5) -> tuple:
    """Run a scan in a child process exactly as the window does and wait for it; returns its last event."""
    job = ScanJob(settings).start()
    while True:
        for event in job.poll():
            if on_event:
                on_event(event)
        if not job.running:
            return job.outcome
        time.sleep(poll_seconds)


# ---- the result in words -----------------------------------------------------------------

def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    except OSError:
        return 0


def summarize(result: dict, notes: list[str] | None = None) -> dict:
    """What the window shows when a scan is done, from ``run_video``'s result."""
    folder = Path(result["folder"])
    book = result.get("book") or {}
    pdf = folder / "book.pdf"
    review = folder / "needs_review.txt"
    spreads = int(result.get("captures", 0))
    pages = int(book.get("pages", 0))
    searchable = bool(book.get("searchable"))
    to_review = int(result.get("to_review", 0))
    review_lines = _count_lines(review)
    lines = [f"{spreads} spread{'s' if spreads != 1 else ''} found in {format_seconds(result.get('video_seconds', 0))} of video "
             f"(took {format_seconds(result.get('total_seconds', 0))})."]
    if book:
        if searchable:
            names = [LANGUAGE_NAMES.get(code, code) for code in book.get("languages", [])]
            lines.append(f"{pages} page images and a searchable PDF (text read as {', '.join(names) or 'the installed languages'}).")
        else:
            hint = TESSERACT_HINT_WINDOWS if sys.platform == "win32" else TESSERACT_HINT_OTHER
            lines.append(f"{pages} page images and a PDF of them. The PDF is not searchable. {hint}")
    if to_review:
        lines.append(f"{to_review} spread{'s' if to_review != 1 else ''} may need a look (a hand over the text, "
                     "or a page that was still moving): see the review list.")
    elif review_lines:
        lines.append("The review list has notes on the page numbering.")
    else:
        lines.append("Nothing was flagged for review.")
    numbers = result.get("audit")
    if numbers:
        if numbers.get("vouched"):
            missing = int(numbers.get("missing") or 0)
            lines.append(f"Printed page numbers run from {numbers.get('first_page')} to {numbers.get('last_page')}"
                         + (f"; {missing} spread{'s' if missing != 1 else ''} seem to be missing." if missing else " with no gap."))
        else:
            lines.append("Too few printed page numbers could be read to check that no page is missing.")
    for note in [*(notes or []), result.get("hand_note") or ""]:
        if note:
            lines.append(note[:1].upper() + note[1:] + ("" if note.endswith(".") else "."))
    return {
        "folder": str(folder),
        "captures_dir": str(folder / "captures"),
        "pdf": str(pdf) if pdf.is_file() else None,
        "review": str(review) if review.is_file() and review_lines else None,
        "spreads": spreads,
        "pages": pages,
        "searchable": searchable,
        "to_review": to_review,
        "lines": lines,
        "result": result,
    }


# ---- the Windows build's check of itself -------------------------------------------------

def self_test(out: Path | None = None) -> dict:
    """Check what the app needs, as it will be used: programs, models, imports, and the hand models in worker processes.

    The Windows build runs this on itself in CI (``AutoCameraBookScanner.exe
    --self-test report.json``). Returns the report; ``report["ok"]`` is True
    when everything required works.
    """
    report: dict = {"frozen": paths.frozen(), "python": sys.version.split()[0], "platform": sys.platform, "checks": {}}
    required = {"ffmpeg", "ffprobe", "libx264", "models", "imports", "hand models"}

    def check(name: str, test: Callable[[], object]) -> None:
        try:
            report["checks"][name] = {"ok": True, "detail": str(test())[:500]}
        except BaseException as error:
            report["checks"][name] = {"ok": False, "detail": f"{error.__class__.__name__}: {error}"[:500],
                                      "trace": traceback.format_exc()[-2000:]}

    where = prepare_environment()
    report["found"] = where
    run = lambda command: subprocess.run(command, capture_output=True, text=True, check=True, timeout=120,  # noqa: E731
                                         stdin=subprocess.DEVNULL).stdout
    check("ffmpeg", lambda: run(["ffmpeg", "-version"]).splitlines()[0])
    check("ffprobe", lambda: run(["ffprobe", "-version"]).splitlines()[0])

    def libx264() -> str:
        if "libx264" not in run(["ffmpeg", "-hide_banner", "-encoders"]):
            raise RuntimeError("ffmpeg has no libx264 encoder (the proxy needs it)")
        return "libx264 encoder present"

    check("libx264", libx264)

    def models() -> str:
        found = [paths.models_dir() / "hand_landmarker.task", paths.models_dir() / "selfie_multiclass_256x256.tflite"]
        missing = [str(p) for p in found if not p.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        return ", ".join(f"{p.name} {p.stat().st_size // 1024} KB" for p in found)

    check("models", models)

    def imports() -> str:
        import scipy

        from . import book, handnet, pages, pipeline  # noqa: F401
        return f"pipeline stages {', '.join(pipeline.STAGES)}; scipy {scipy.__version__}"

    check("imports", imports)

    def hand_models() -> str:
        from concurrent.futures import ProcessPoolExecutor

        from . import handdetect, handnet

        with tempfile.TemporaryDirectory(prefix="self-test-") as tmp:
            folder = Path(tmp)
            run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=480x270:rate=11", "-frames:v", "8",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(folder / "pattern.mp4")])
            with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn"), initializer=handdetect.quiet_worker) as pool:
                handdetect.detect("pattern", folder, folder, 2, pool)
            found = handnet.load("pattern", folder)
            if found is None or len(found) != 8:
                raise RuntimeError(f"expected 8 frames of detections, got {None if found is None else len(found)}")
            return f"MediaPipe {handdetect.import_mediapipe().__version__}: 8 frames in two worker processes"

    check("hand models", hand_models)
    report["gpu"] = check_gpu()
    report["tesseract"] = where.get("tesseract")
    report["ok"] = all(report["checks"].get(name, {}).get("ok") for name in required)
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=1))
    return report
