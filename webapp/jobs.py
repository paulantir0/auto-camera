"""Starting, following and cancelling one scan.

A scan is the command the README gives for one video,

    tools/run_capped.sh .venv/bin/python tools/run_video.py VIDEO --run RUN_DIR

run as a subprocess in its own process group. ``run_capped.sh`` puts the
memory and CPU ceilings on it and refuses to start when the machine is busy.
The command prints one ``[stage] detail`` line on standard output as each
stage starts; those lines are all the progress there is. Only one scan runs at a time: a second
request is refused while the first is going.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import date, datetime
from pathlib import Path

from . import Config
from .runs import JOB_MARKER, read_json

# The stages of auto_camera.pipeline.STAGES, in order, in the words the page
# shows. tests/test_webapp_jobs.py checks the keys still match the pipeline's;
# a stage the pipeline adds later is still shown, under its own description.
STAGE_WORDS = {
    "proxy": "Making a small copy of the video",
    "analysis": "Looking at every frame",
    "hands": "Finding hands over the page",
    "turns": "Finding the page turns",
    "stills": "Saving a picture of each spread",
    "numbers": "Reading the printed page numbers",
    "finish": "Putting the results together",
    "pages": "Flattening the pages and making the PDF",
}

PROGRESS_LINE = re.compile(r"\[([a-z][a-z0-9_]*)\][ \t]?(.*)\Z")
BUSY_EXIT = 75                 # run_capped.sh: the machine is too busy to start
ACTIVE = ("starting", "running")
KILL_AFTER_SECONDS = 5.0


def parse_progress(line: str) -> tuple[str, str] | None:
    """``("stills", "saving 127 full-size frames ...")`` from a ``[stills] saving ...`` line, else None."""
    match = PROGRESS_LINE.match(line.rstrip("\r\n"))
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def slugify(name: str) -> str:
    """The pipeline's folder name for a video (the same rule as auto_camera.proxy.slugify)."""
    return re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")


def default_command(config: Config, video: Path, run_dir: Path) -> list[str]:
    return [str(config.repo / "tools" / "run_capped.sh"), str(config.repo / ".venv" / "bin" / "python"),
            str(config.repo / "tools" / "run_video.py"), str(video), "--run", str(run_dir)]


def web_run_dir(config: Config, today: date | None = None) -> Path:
    """Where scans started from the page go: one folder per day."""
    return config.runs_root / f"run_{(today or date.today()):%Y%m%d}_web"


class Busy(Exception):
    """A scan is already going. ``job`` is it, when it is one of ours."""

    def __init__(self, message: str, job: "Job | None" = None):
        super().__init__(message)
        self.message = message
        self.job = job


class Job:
    """One scan and what is known about how far it has got."""

    def __init__(self, name: str, source: str, video: Path, run_dir: Path):
        self.id = secrets.token_hex(6)
        self.name = name
        self.source = source
        self.video = video
        self.run_dir = run_dir
        self.slug = slugify(video.name)
        self.status = "starting"          # starting, running, done, cancelled, failed, busy
        self.message = ""
        self.stages = [{"key": key, "label": label, "state": "waiting", "detail": "", "seconds": None}
                       for key, label in STAGE_WORDS.items()]
        self.tail: deque[str] = deque(maxlen=60)
        self.started = time.time()
        self.finished: float | None = None
        self.cancel_requested = False
        self.process: subprocess.Popen | None = None
        self.reader_done = threading.Event()
        self.marker_written = False
        self._stage_started: float | None = None
        self.lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    def feed(self, line: str, now: float | None = None) -> bool:
        """Take one line of the command's output. True when it was a progress line."""
        now = time.time() if now is None else now
        line = line.rstrip("\r\n")
        progress = parse_progress(line)
        with self.lock:
            if progress is None:
                if line.strip():
                    self.tail.append(line[:500])
                return False
            key, detail = progress
            self._close_stage(now)
            if key == "done":
                return True
            self.status = "running"
            stage = next((s for s in self.stages if s["key"] == key), None)
            if stage is None:
                stage = {"key": key, "label": (detail[:1].upper() + detail[1:]) if detail else key, "state": "waiting",
                         "detail": "", "seconds": None}
                self.stages.append(stage)
            for earlier in self.stages[:self.stages.index(stage)]:
                if earlier["state"] == "waiting":
                    earlier["state"] = "skipped"
            stage["state"] = "running"
            stage["detail"] = detail[:300]
            self._stage_started = now
            return True

    def note(self, line: str) -> None:
        """Keep a line that is not progress (the command's error output) for the error report."""
        line = line.rstrip("\r\n")
        if line.strip():
            with self.lock:
                self.tail.append(line[:500])

    def _close_stage(self, now: float, state: str = "done") -> None:
        for stage in self.stages:
            if stage["state"] == "running":
                stage["state"] = state
                stage["seconds"] = round(now - (self._stage_started or now), 1)

    def finish(self, status: str, message: str = "", now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self.lock:
            self._close_stage(now, "done" if status == "done" else "stopped")
            self.status = status
            self.message = message
            self.finished = now

    def current_stage(self) -> dict | None:
        """The stage the scan is in, or was in when it ended."""
        reached = [s for s in self.stages if s["state"] not in ("waiting", "skipped")]
        return reached[-1] if reached else None

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self.lock:
            stages = []
            for stage in self.stages:
                shown = dict(stage)
                if stage["state"] == "running" and self._stage_started is not None:
                    shown["seconds"] = round(now - self._stage_started, 1)
                stages.append(shown)
            out = {
                "id": self.id, "name": self.name, "source": self.source, "run": self.run_dir.name, "slug": self.slug,
                "status": self.status, "active": self.active, "cancelling": self.cancel_requested and self.active,
                "message": self.message, "stages": stages,
                "elapsed": round((self.finished or now) - self.started, 1),
            }
            if self.status == "failed":
                out["output"] = list(self.tail)[-30:]
            return out


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class JobRunner:
    """Runs one scan at a time and remembers the last few."""

    def __init__(self, config: Config, command=default_command):
        self.config = config
        self.command = command
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._latest: Job | None = None

    # ---- what the page asks ------------------------------------------------

    def latest(self) -> Job | None:
        return self._latest

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def live(self) -> tuple[str, str] | None:
        """(run folder name, video folder name) of the scan going on now, if any."""
        job = self._latest
        return (job.run_dir.name, job.slug) if job and job.active else None

    def start(self, video: Path, name: str, source: str) -> Job:
        """Start scanning ``video``. Raises ``Busy`` when a scan is already going."""
        if not slugify(video.name):
            raise ValueError("That file's name has no letters or digits in it. Rename it and try again.")
        with self._lock:
            if self._latest and self._latest.active:
                raise Busy("Another video is being scanned. Wait for it to finish, or stop it first.", self._latest)
            left_over = self._left_over()
            if left_over:
                raise Busy(f"A scan started earlier ({left_over}) is still finishing. Try again in a minute.")
            job = Job(name, source, video, web_run_dir(self.config))
            self._jobs[job.id] = job
            while len(self._jobs) > 20:
                self._jobs.pop(next(iter(self._jobs)))
            self._latest = job
        threading.Thread(target=self._run, args=(job,), name=f"scan-{job.id}", daemon=True).start()
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None or not job.active:
            return job
        job.cancel_requested = True
        process = job.process
        if process is not None:
            _signal_group(process.pid, signal.SIGTERM)
            timer = threading.Timer(KILL_AFTER_SECONDS, self._kill, args=(job,))
            timer.daemon = True
            timer.start()
        return job

    def shutdown(self, wait: float = 8.0) -> None:
        """Stop the running scan, if any, before the server goes away."""
        job = self._latest
        if job is None or not job.active:
            return
        self.cancel(job.id)
        if not job.reader_done.wait(wait):
            self._kill(job)
            job.reader_done.wait(2.0)

    # ---- the scan itself ---------------------------------------------------

    @staticmethod
    def _kill(job: Job) -> None:
        process = job.process
        if process is not None and (process.poll() is None or not job.reader_done.is_set()):
            _signal_group(process.pid, signal.SIGKILL)

    def _pid_file(self) -> Path:
        return self.config.state_dir / "running_job.json"

    def _left_over(self) -> str:
        """The video name of a scan that outlived an earlier server and is still running, else ''."""
        note = read_json(self._pid_file())
        if not note:
            return ""
        pid = note.get("pid")
        try:
            cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        except (OSError, TypeError, ValueError):
            cmdline = b""
        if b"run_video.py" in cmdline:
            return str(note.get("video", "another video"))[:200]
        self._forget_pid()
        return ""

    def _forget_pid(self) -> None:
        try:
            self._pid_file().unlink()
        except OSError:
            pass

    def _write_marker(self, job: Job, status: str) -> None:
        """Note beside the results whether this scan is going, finished or was cut short."""
        folder = job.run_dir / job.slug
        try:
            folder.mkdir(parents=True, exist_ok=True)
            note = {"status": status, "job": job.id, "video": job.name,
                    "started": datetime.fromtimestamp(job.started).isoformat(timespec="seconds"),
                    "updated": datetime.now().isoformat(timespec="seconds")}
            (folder / JOB_MARKER).write_text(json.dumps(note, indent=1))
            job.marker_written = True
        except OSError:
            pass

    def _run(self, job: Job) -> None:
        try:
            self._follow(job)
        except Exception as error:      # a scan must never be left looking as if it were still running
            if job.active:
                job.finish("failed", f"The scan could not be followed: {error}")
        finally:
            job.reader_done.set()

    def _follow(self, job: Job) -> None:
        command = self.command(self.config, job.video, job.run_dir)
        try:
            process = subprocess.Popen(command, cwd=self.config.repo, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1,
                                       start_new_session=True)
        except OSError as error:
            job.finish("failed", f"The scan could not be started: {error}")
            return
        job.process = process
        try:
            self.config.state_dir.mkdir(parents=True, exist_ok=True)
            self._pid_file().write_text(json.dumps({"pid": process.pid, "video": job.name}))
        except OSError:
            pass
        if job.cancel_requested:
            _signal_group(process.pid, signal.SIGTERM)

        # Progress is only ever read from standard output, where run_video.py prints it;
        # whatever the programs underneath write to standard error is kept for the error report.
        def keep_errors() -> None:
            for line in iter(lambda: process.stderr.readline(8192), ""):
                job.note(line)

        errors = threading.Thread(target=keep_errors, name=f"scan-errors-{job.id}", daemon=True)
        errors.start()
        for line in iter(lambda: process.stdout.readline(8192), ""):
            if job.feed(line) and not job.marker_written and not job.cancel_requested:
                self._write_marker(job, "running")
        code = process.wait()
        errors.join(2.0)
        process.stdout.close()
        if not errors.is_alive():
            process.stderr.close()
        self._forget_pid()

        stage = job.current_stage()
        if job.cancel_requested:
            status, message = "cancelled", "Stopped. The scan was not finished."
        elif code == 0 and read_json(job.run_dir / job.slug / "result.json") is not None:
            status, message = "done", ""
        elif code == 0:
            status, message = "failed", "The scan ended without leaving a result."
        elif code == BUSY_EXIT and any(line.startswith("run_capped:") for line in job.tail):
            status = "busy"
            message = "The computer is busy with other work right now, so the scan was not started. Try again in a few minutes."
        elif code in (-signal.SIGKILL, 128 + signal.SIGKILL):
            status = "failed"
            message = "The scan was stopped by the computer, most likely because it needed more memory than it is allowed."
        else:
            where = f"while {stage['label'][:1].lower() + stage['label'][1:]}" if stage else "before it got going"
            status, message = "failed", f"The scan stopped with an error {where}."
        if job.marker_written:
            self._write_marker(job, status)
        job.finish(status, message)
