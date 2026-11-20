#!/usr/bin/env python3
"""Auto Camera Book Scanner: the desktop app.

Film a book while paging through it; this turns the video into one picture
per two-page spread, in page order, plus each page flattened and cropped and
a PDF of the book.

    python auto_camera_gui.py                          # the window
    python auto_camera_gui.py --self-test REPORT.json  # check programs, models and imports; no window
    python auto_camera_gui.py --scan VIDEO FOLDER      # one scan as the window runs it, progress printed; no window

The scan itself is ``auto_camera.pipeline.run_video``, run in a child process
(``auto_camera.desktop.ScanJob``) so that Stop can end it. The Windows build
(``.github/workflows/windows-exe.yml``) bundles ffmpeg and the hand models.
"""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from tkinter import BooleanVar, IntVar, StringVar, filedialog, messagebox, ttk
import tkinter as tk

from auto_camera import desktop

try:  # native file drops; bundled in the Windows build
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:  # pragma: no cover - depends on the platform
    DND_FILES = None
    TkinterDnD = None

APP_TITLE = "Auto Camera Book Scanner"
VIDEO_FILETYPES = [("Videos", " ".join(f"*{s} *{s.upper()}" for s in desktop.VIDEO_SUFFIXES)), ("All files", "*.*")]
PREVIEW_SIZE = (288, 162)
THUMB_SIZE = (160, 90)
CELL_WIDTH = THUMB_SIZE[0] + 8      # a thumbnail and the space around it
DARK = "#15181c"
LIGHT_TEXT = "#d8dde5"
MARKS = {"waiting": "○", "running": "▶", "done": "✓", "skipped": "–", "stopped": "■"}


def thumb_caption(index: int, name: str) -> str:
    """"3  at 0:10" for the third capture, taken 10 s into the video (its file is sequence_0003_t0010.09s.jpg)."""
    found = re.search(r"_t(\d+(?:\.\d+)?)s", name)
    if not found:
        return f"{index + 1}"
    seconds = int(float(found.group(1)))
    return f"{index + 1}  at {seconds // 60}:{seconds % 60:02d}"


def open_path(path: Path) -> None:
    """Open a file or folder with whatever the computer opens it with."""
    if hasattr(os, "startfile"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class ScannerApp:
    def __init__(self, root: tk.Tk):
        from PIL import Image, ImageTk   # noqa: F401  (imported here: the helper processes never need it)

        self.root = root
        self.root.title(APP_TITLE)
        self.root.minsize(1060, 700)
        width, height = min(1320, self.root.winfo_screenwidth() - 40), min(900, self.root.winfo_screenheight() - 80)
        self.root.geometry(f"{max(1060, width)}x{max(700, height)}")
        self.root.report_callback_exception = self._callback_failed
        self.ui: queue.Queue[tuple] = queue.Queue()      # from this process's helper threads
        self.job: desktop.ScanJob | None = None
        self.video_info: dict | None = None
        self.photos: dict[str, object] = {}              # Tk drops images nobody holds
        self.thumb_names: list[str] = []
        self.thumb_cells: dict[int, ttk.Frame] = {}
        self.columns = 3
        self.thumb_queue: queue.Queue[Path | None] = queue.Queue()
        self.stage_rows: dict[str, dict] = {}
        self.stage_started: dict[str, float] = {}
        self.summary: dict | None = None
        self.details: tk.Toplevel | None = None
        self.log: tk.Text | None = None
        self.log_lines: list[str] = []
        self.last_folder_check = 0.0
        self.captures_ready = False

        self.video_path = StringVar()
        self.output_dir = StringVar(value=str(desktop.default_output_dir()))
        self.workers = IntVar(value=desktop.default_workers())
        self.use_gpu = BooleanVar(value=False)
        self.make_pages = BooleanVar(value=True)
        self.gpu_text = StringVar(value="checking the graphics card…")
        self.stats_text = StringVar(value="No video chosen yet.")
        self.target_text = StringVar()
        self.status_text = StringVar(value="Choose a video to start.")
        self.elapsed_text = StringVar()
        self.resource_text = StringVar()
        self.summary_text = StringVar(value="When a scan is done, what it found is shown here.")
        self.drop_enabled = bool(TkinterDnD and DND_FILES)

        self._style()
        self._build()
        self.output_dir.trace_add("write", lambda *_: self._refresh_target())
        self.video_path.trace_add("write", lambda *_: self._refresh_target())
        self._refresh_target()
        self._reset_stages()

        self.tools = desktop.prepare_environment()
        missing = [name for name in ("ffmpeg", "ffprobe") if not self.tools.get(name)]
        if missing:
            where = ("The app's own copy is missing; download the app again." if getattr(sys, "frozen", False)
                     else "Install FFmpeg and make sure it is on PATH.")
            self.start_button.configure(state="disabled")
            self.status_text.set(f"{' and '.join(missing)} not found. {where}")
            self.gpu_text.set("not checked (no ffmpeg)")
        else:
            threading.Thread(target=self._check_gpu, daemon=True).start()
        threading.Thread(target=self._thumb_worker, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._log(f"{APP_TITLE} ready. ffmpeg: {self.tools.get('ffmpeg') or 'not found'}; "
                  f"Tesseract: {self.tools.get('tesseract') or 'not installed (the PDF will not be searchable)'}.")
        self._tick()

    # ---- layout ------------------------------------------------------------------------

    def _style(self) -> None:
        style = ttk.Style(self.root)
        for theme in (("vista",) if sys.platform == "win32" else ()) + ("clam",):
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                continue
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Muted.TLabel", foreground="#5d6673")
        style.configure("Stage.TLabel", font=("Segoe UI", 10))
        style.configure("Now.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Error.TLabel", foreground="#b3261e")
        style.configure("Start.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 8))
        style.configure("Stop.TButton", padding=(12, 8))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=5, uniform="halves")
        outer.columnconfigure(1, weight=6, uniform="halves")
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="One picture per two-page spread, flat page images and a PDF, from a video of a book being paged through.",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(header, textvariable=self.resource_text, style="Muted.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        left = ttk.Frame(outer)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        right = ttk.Frame(outer)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_video(left)
        self._build_output(left)
        self._build_settings(left)
        self._build_buttons(left)
        self._build_progress(right)
        self._build_gallery(right)
        self._build_result(right)
        self._register_drop(self.root)

    def _build_video(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="1. The video", padding=(10, 6))
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        entry = ttk.Entry(panel, textvariable=self.video_path)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        entry.bind("<Return>", lambda _e: self._set_video(Path(self.video_path.get())))
        self.video_button = ttk.Button(panel, text="Choose video…", command=self._choose_video)
        self.video_button.grid(row=0, column=1)
        hint = "or drop a video file anywhere on this window" if self.drop_enabled else "type its path and press Enter, or choose it"
        ttk.Label(panel, text=hint, style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 6))
        body = ttk.Frame(panel)
        body.grid(row=2, column=0, columnspan=2, sticky="ew")
        body.columnconfigure(1, weight=1)
        self.preview = tk.Label(body, text="No video", bg=DARK, fg=LIGHT_TEXT, width=36, height=9)
        self.preview.grid(row=0, column=0, sticky="nw")
        ttk.Label(body, textvariable=self.stats_text, justify="left", wraplength=150).grid(row=0, column=1, sticky="nw", padx=(10, 0))
        self._register_drop(entry)
        self._register_drop(self.preview)

    def _build_output(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="2. Where the results go", padding=(10, 6))
        panel.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        panel.columnconfigure(0, weight=1)
        ttk.Entry(panel, textvariable=self.output_dir).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.output_button = ttk.Button(panel, text="Choose folder…", command=self._choose_output)
        self.output_button.grid(row=0, column=1)
        ttk.Label(panel, textvariable=self.target_text, style="Muted.TLabel", wraplength=480, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_settings(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="3. Settings", padding=(10, 6))
        panel.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        panel.columnconfigure(2, weight=1)
        ttk.Label(panel, text="CPU workers").grid(row=0, column=0, sticky="w")
        self.workers_box = ttk.Spinbox(panel, from_=1, to=max(1, os.cpu_count() or 1), textvariable=self.workers, width=5)
        self.workers_box.grid(row=0, column=1, sticky="w", padx=(8, 8))
        ttk.Label(panel, text=f"this computer has {os.cpu_count() or '?'} cores",
                  style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        self.gpu_box = ttk.Checkbutton(panel, text="Use the NVIDIA graphics card", variable=self.use_gpu, state="disabled")
        self.gpu_box.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(panel, textvariable=self.gpu_text, style="Muted.TLabel").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.pages_box = ttk.Checkbutton(panel, text="Flatten the pages and make a PDF", variable=self.make_pages)
        self.pages_box.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _build_buttons(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        bar.columnconfigure(2, weight=1)
        self.start_button = ttk.Button(bar, text="Start", style="Start.TButton", command=self._start)
        self.start_button.grid(row=0, column=0)
        self.stop_button = ttk.Button(bar, text="Stop", style="Stop.TButton", command=self._stop, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(8, 0))
        ttk.Label(bar, textvariable=self.status_text, wraplength=250, justify="left").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Button(bar, text="Details", command=self._show_details).grid(row=0, column=3, sticky="e")

    def _show_details(self) -> None:
        """The log, in a window of its own (opened by the Details button, and by itself when a scan fails)."""
        if self.details is not None and self.details.winfo_exists():
            self.details.deiconify()
            self.details.lift()
            return
        window = tk.Toplevel(self.root)
        window.title(f"{APP_TITLE}: details")
        window.geometry("820x360")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        self.log = tk.Text(window, wrap="word", state="normal", bg="#101317", fg="#e6e9ee", relief="flat", font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(window, command=self.log.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=bar.set)
        self.log.insert("end", "".join(self.log_lines))
        self.log.see("end")
        self.log.configure(state="disabled")
        self.details = window

    def _build_progress(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Progress", padding=(10, 6))
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self.stage_frame = ttk.Frame(panel)
        self.stage_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.stage_frame.columnconfigure(1, weight=1)
        self.bar = ttk.Progressbar(panel, mode="indeterminate")
        self.bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.bar.grid_remove()
        ttk.Label(panel, textvariable=self.elapsed_text, style="Muted.TLabel").grid(row=1, column=2, sticky="e", padx=(10, 0), pady=(8, 0))

    def _build_gallery(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Spreads", padding=(10, 6))
        panel.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(panel, highlightthickness=0, height=260)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(panel, command=self.canvas.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=bar.set)
        self.gallery = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.gallery, anchor="nw")
        self.gallery.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda _e: self._layout_gallery())
        for widget in (self.canvas, self.gallery):
            widget.bind("<MouseWheel>", self._wheel)
            widget.bind("<Button-4>", lambda _e: self.canvas.yview_scroll(-2, "units"))
            widget.bind("<Button-5>", lambda _e: self.canvas.yview_scroll(2, "units"))
        self.gallery_hint = ttk.Label(self.gallery, text="A picture of each spread appears here once it is saved. Click one to open it.",
                                      style="Muted.TLabel")
        self.gallery_hint.grid(row=0, column=0, columnspan=8, sticky="w")

    def _build_result(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Result", padding=(10, 6))
        panel.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        panel.columnconfigure(0, weight=1)
        self.summary_label = ttk.Label(panel, textvariable=self.summary_text, wraplength=560, justify="left")
        self.summary_label.grid(row=0, column=0, sticky="ew")
        bar = ttk.Frame(panel)
        bar.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.open_folder_button = ttk.Button(bar, text="Open folder", command=lambda: self._open("folder"), state="disabled")
        self.open_folder_button.grid(row=0, column=0)
        self.open_pdf_button = ttk.Button(bar, text="Open PDF", command=lambda: self._open("pdf"), state="disabled")
        self.open_pdf_button.grid(row=0, column=1, padx=(8, 0))
        self.open_review_button = ttk.Button(bar, text="Open review list", command=lambda: self._open("review"), state="disabled")
        self.open_review_button.grid(row=0, column=2, padx=(8, 0))

    def _register_drop(self, widget: tk.Widget) -> None:
        if not self.drop_enabled:
            return
        try:
            widget.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
            widget.dnd_bind("<<Drop>>", self._dropped)  # type: ignore[attr-defined]
        except (tk.TclError, AttributeError):
            pass

    # ---- the video and the folder ------------------------------------------------------

    def _dropped(self, event: object) -> None:
        data = getattr(event, "data", "")
        try:
            names = self.root.tk.splitlist(data)
        except tk.TclError:
            names = [data]
        if names and not self._busy():
            self._set_video(Path(names[0]))

    def _choose_video(self) -> None:
        name = filedialog.askopenfilename(title="Choose the video of the book", filetypes=VIDEO_FILETYPES)
        if name:
            self._set_video(Path(name))

    def _choose_output(self) -> None:
        name = filedialog.askdirectory(title="Choose where the results go", initialdir=self._existing_parent(Path(self.output_dir.get())))
        if name:
            self.output_dir.set(name)

    @staticmethod
    def _existing_parent(path: Path) -> str:
        path = path.expanduser()
        while not path.exists() and path.parent != path:
            path = path.parent
        return str(path)

    def _set_video(self, path: Path) -> None:
        path = path.expanduser()
        self.video_path.set(str(path))
        self.video_info = None
        if not path.is_file():
            self.stats_text.set("That file is not there.")
            self.preview.configure(image="", text="No video")
            return
        self.stats_text.set("Reading the video…")
        self.preview.configure(image="", text="Loading…")
        threading.Thread(target=self._read_video, args=(path,), daemon=True).start()

    def _read_video(self, path: Path) -> None:
        try:
            info = desktop.video_info(path)
        except Exception as error:
            self.ui.put(("video_error", path, desktop._plain_error(error)))
            return
        try:
            png = desktop.first_frame_png(path, PREVIEW_SIZE[0])
        except Exception:
            png = b""
        self.ui.put(("video_info", path, info, png))

    def _show_video(self, path: Path, info: dict, png: bytes) -> None:
        from PIL import Image, ImageTk

        if Path(self.video_path.get()).expanduser() != path:
            return
        self.video_info = info
        size_mb = info["size"] / 1e6
        self.stats_text.set("\n".join([
            path.name,
            f"{info['width']} x {info['height']}, {info['codec'].upper()}",
            f"{info['fps']:.2f} frames a second",
            f"{desktop.format_seconds(info['duration'])} long, {info['frames']:,} frames",
            f"{size_mb / 1000:.2f} GB" if size_mb >= 1000 else f"{size_mb:.0f} MB",
        ]))
        if png:
            image = Image.open(io.BytesIO(png))
            image.thumbnail(PREVIEW_SIZE)
            photo = ImageTk.PhotoImage(image)
            self.photos["preview"] = photo
            self.preview.configure(image=photo, text="", width=image.width, height=image.height)
        else:
            self.preview.configure(image="", text="No preview")
        if not self._busy():
            self.status_text.set("Ready. Press Start.")

    def _refresh_target(self) -> None:
        video = self.video_path.get().strip()
        folder = Path(self.output_dir.get().strip() or ".").expanduser()
        if video:
            name = desktop.slugify(Path(video).name) or "?"
            self.target_text.set(f"This video's results will be in {folder / name}")
        else:
            self.target_text.set(f"Each video gets a folder of its own in {folder}")

    # ---- a scan ------------------------------------------------------------------------

    def _busy(self) -> bool:
        return self.job is not None and self.job.running

    def _start(self) -> None:
        if self._busy():
            return
        video = Path(self.video_path.get().strip()).expanduser()
        if not video.is_file():
            messagebox.showerror(APP_TITLE, "Choose a video first: the file given is not there.")
            return
        if not desktop.slugify(video.name):
            messagebox.showerror(APP_TITLE, "That file's name has no letters or digits in it. Rename it and try again.")
            return
        folder = Path(self.output_dir.get().strip()).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            messagebox.showerror(APP_TITLE, f"The results folder cannot be made there:\n{folder}\n\n{error}")
            return
        try:
            workers = max(1, int(self.workers.get()))
        except (tk.TclError, ValueError):
            workers = desktop.default_workers()
            self.workers.set(workers)
        settings = desktop.ScanSettings(video=str(video), run_dir=str(folder), use_gpu=bool(self.use_gpu.get()), workers=workers,
                                        pages=bool(self.make_pages.get()))
        self._reset_stages()
        self._clear_gallery()
        self.summary = None
        self.summary_text.set("Working…")
        self.summary_label.configure(style="TLabel")
        for button in (self.open_folder_button, self.open_pdf_button, self.open_review_button):
            button.configure(state="disabled")
        self._log(f"Starting {video.name}: results in {folder}; {workers} worker(s); "
                  f"{'NVIDIA card' if settings.use_gpu else 'CPU only'}; {'with' if settings.pages else 'without'} pages and PDF.")
        try:
            self.job = desktop.ScanJob(settings).start()
        except Exception as error:
            self.job = None
            self._log(traceback.format_exc())
            messagebox.showerror(APP_TITLE, f"The scan could not be started: {error}")
            return
        self._set_running(True)
        self.status_text.set("Starting…")

    def _stop(self) -> None:
        if not self._busy():
            return
        if not messagebox.askyesno(APP_TITLE, "Stop the scan now?\n\nWhat it has saved so far stays in the folder, unfinished."):
            return
        self.status_text.set("Stopping…")
        self.root.update_idletasks()
        self.job.stop()
        self._finish_stages("stopped")
        self._set_running(False)
        self.elapsed_text.set(f"stopped after {desktop.format_seconds(self.job.ended - self.job.started)}")
        self.status_text.set("Stopped. The scan was not finished.")
        self.summary_text.set("Stopped before the end. The folder holds what was done so far.")
        self.open_folder_button.configure(state="normal" if self.job.folder.exists() else "disabled")
        self._log("Stopped.")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for widget in (self.start_button, self.video_button, self.output_button, self.workers_box, self.pages_box):
            widget.configure(state=state)
        if not running and self.gpu_text.get().startswith("checking"):
            self.gpu_box.configure(state="disabled")
        else:
            self.gpu_box.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")
        if running:
            self.bar.grid()
            self.bar.start(12)
        else:
            self.bar.stop()
            self.bar.grid_remove()

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "stage":
            _, key, detail, when = event
            self._enter_stage(key, detail, when)
            # From here on captures/ holds this scan's pictures (before, it may hold an earlier scan's).
            if key in ("finish", "pages", "done"):
                self.captures_ready = True
        elif kind == "note":
            self._log(event[1])
        elif kind == "done":
            self._done(event[1])
        elif kind == "error":
            self._failed(event[1], event[2] if len(event) > 2 else "")

    def _done(self, summary: dict) -> None:
        self.summary = summary
        self._finish_stages("done")
        self._set_running(False)
        self.status_text.set("Done.")
        self.summary_text.set("\n".join(summary["lines"]))
        self.open_folder_button.configure(state="normal")
        self.open_pdf_button.configure(state="normal" if summary.get("pdf") else "disabled")
        self.open_review_button.configure(state="normal" if summary.get("review") else "disabled")
        self.captures_ready = True
        self._refresh_gallery(force=True)
        for line in summary["lines"]:
            self._log(line)
        self._log(f"Folder: {summary['folder']}")
        self.root.bell()

    def _failed(self, message: str, details: str) -> None:
        self._finish_stages("stopped")
        self._set_running(False)
        self.status_text.set("The scan failed.")
        self.summary_text.set(message)
        self.summary_label.configure(style="Error.TLabel")
        if self.job is not None and self.job.folder.exists():
            self.open_folder_button.configure(state="normal")
        self._log(message)
        if details:
            self._log(details.rstrip())
            self._show_details()
        messagebox.showerror(APP_TITLE, message)

    # ---- stages ------------------------------------------------------------------------

    def _reset_stages(self) -> None:
        for child in self.stage_frame.winfo_children():
            child.destroy()
        self.stage_rows.clear()
        self.stage_started.clear()
        for key, words in desktop.STAGE_WORDS.items():
            self._add_stage_row(key, words)
        self.elapsed_text.set("")

    def _add_stage_row(self, key: str, words: str) -> dict:
        row = len(self.stage_rows)
        mark = ttk.Label(self.stage_frame, text=MARKS["waiting"], width=2, style="Stage.TLabel")
        mark.grid(row=row, column=0, sticky="w")
        label = ttk.Label(self.stage_frame, text=words, style="Stage.TLabel", foreground="#8a919b")
        label.grid(row=row, column=1, sticky="w")
        seconds = ttk.Label(self.stage_frame, text="", style="Muted.TLabel")
        seconds.grid(row=row, column=2, sticky="e")
        entry = {"state": "waiting", "mark": mark, "label": label, "seconds": seconds, "detail": ""}
        self.stage_rows[key] = entry
        return entry

    def _set_stage(self, key: str, state: str, seconds: float | None = None) -> None:
        entry = self.stage_rows[key]
        entry["state"] = state
        entry["mark"].configure(text=MARKS[state])
        entry["label"].configure(style="Now.TLabel" if state == "running" else "Stage.TLabel",
                                 foreground="" if state in ("running", "done") else "#8a919b")
        if seconds is not None:
            entry["seconds"].configure(text=desktop.format_seconds(seconds))

    def _enter_stage(self, key: str, detail: str, when: float) -> None:
        for other, entry in self.stage_rows.items():
            if entry["state"] == "running":
                self._set_stage(other, "done", when - self.stage_started.get(other, when))
        if key == "done":
            return
        if key not in self.stage_rows:
            self._add_stage_row(key, desktop.stage_words(key, detail))
        keys = list(self.stage_rows)
        for earlier in keys[:keys.index(key)]:
            if self.stage_rows[earlier]["state"] == "waiting":
                self._set_stage(earlier, "skipped")
        self.stage_started[key] = when
        self._set_stage(key, "running", 0)
        self.status_text.set(f"{desktop.stage_words(key, detail)}: {detail}" if detail else desktop.stage_words(key))
        self._log(f"{desktop.stage_words(key, detail)}" + (f" ({detail})" if detail else ""))

    def _finish_stages(self, how: str) -> None:
        now = time.time()
        for key, entry in self.stage_rows.items():
            if entry["state"] == "running":
                self._set_stage(key, "done" if how == "done" else "stopped", now - self.stage_started.get(key, now))
            elif entry["state"] == "waiting" and how == "done":
                self._set_stage(key, "skipped")

    # ---- the pictures ------------------------------------------------------------------

    def _clear_gallery(self) -> None:
        for child in self.gallery.winfo_children():
            if child is not self.gallery_hint:
                child.destroy()
        self.thumb_names = []
        self.thumb_cells.clear()
        for key in [k for k in self.photos if k != "preview"]:
            del self.photos[key]
        while True:
            try:
                self.thumb_queue.get_nowait()
            except queue.Empty:
                break
        self.gallery_hint.grid()
        self.captures_ready = False
        self.canvas.yview_moveto(0)

    def _refresh_gallery(self, force: bool = False) -> None:
        """Show the pictures in the video's captures/ folder that are not shown yet."""
        if self.job is None or not (self.captures_ready or force):
            return
        folder = self.job.folder / "captures"
        try:
            names = sorted(p.name for p in folder.iterdir() if p.suffix.lower() == ".jpg")
        except OSError:
            return
        if names[:len(self.thumb_names)] != self.thumb_names:      # the folder was rebuilt with other pictures
            self._clear_gallery()
            self.captures_ready = True
        for name in names[len(self.thumb_names):]:
            self.thumb_names.append(name)
            self.thumb_queue.put(folder / name)

    def _thumb_worker(self) -> None:
        from PIL import Image

        while True:
            path = self.thumb_queue.get()
            if path is None:
                return
            try:
                with Image.open(path) as image:
                    image.draft("RGB", (THUMB_SIZE[0] * 2, THUMB_SIZE[1] * 2))
                    small = image.convert("RGB")
                    small.thumbnail(THUMB_SIZE)
            except Exception:
                continue
            self.ui.put(("thumb", path, small))

    def _place_thumb(self, path: Path, image) -> None:
        from PIL import ImageTk

        if path.name not in self.thumb_names:
            return
        self.gallery_hint.grid_remove()
        index = self.thumb_names.index(path.name)
        photo = ImageTk.PhotoImage(image)
        self.photos[path.name] = photo
        cell = ttk.Frame(self.gallery, padding=3)
        cell.grid(row=1 + index // self.columns, column=index % self.columns, sticky="nw")
        self.thumb_cells[index] = cell
        picture = tk.Label(cell, image=photo, bg=DARK, cursor="hand2")
        picture.pack()
        caption = ttk.Label(cell, text=thumb_caption(index, path.name), style="Muted.TLabel")
        caption.pack(anchor="w")
        for widget in (cell, picture, caption):
            widget.bind("<Button-1>", lambda _e, p=path: self._open_file(p))
            widget.bind("<MouseWheel>", self._wheel)
            widget.bind("<Button-4>", lambda _e: self.canvas.yview_scroll(-2, "units"))
            widget.bind("<Button-5>", lambda _e: self.canvas.yview_scroll(2, "units"))

    def _layout_gallery(self) -> None:
        """As many thumbnails to a row as the gallery is wide."""
        columns = max(1, self.canvas.winfo_width() // CELL_WIDTH)
        if columns == self.columns:
            return
        self.columns = columns
        for index, cell in self.thumb_cells.items():
            cell.grid_configure(row=1 + index // columns, column=index % columns)

    def _wheel(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.delta > 0 else 1), "units")

    # ---- the loop ----------------------------------------------------------------------

    def _tick(self) -> None:
        try:
            while True:
                event = self.ui.get_nowait()
                kind = event[0]
                if kind == "video_info":
                    self._show_video(event[1], event[2], event[3])
                elif kind == "video_error":
                    if Path(self.video_path.get()).expanduser() == event[1]:
                        self.stats_text.set(f"This file cannot be read as a video.\n{event[2]}")
                        self.preview.configure(image="", text="No preview")
                elif kind == "gpu":
                    self._show_gpu(event[1], event[2])
                elif kind == "thumb":
                    self._place_thumb(event[1], event[2])
        except queue.Empty:
            pass
        if self.job is not None and self.job.running:
            for event in self.job.poll():
                self._handle(event)
            now = time.time()
            if self.job is not None and self.job.running:
                self.elapsed_text.set(f"{desktop.format_seconds(now - self.job.started)} so far")
                for key, entry in self.stage_rows.items():
                    if entry["state"] == "running":
                        entry["seconds"].configure(text=desktop.format_seconds(now - self.stage_started.get(key, now)))
                if now - self.last_folder_check > 1.5:
                    self.last_folder_check = now
                    self._refresh_gallery()
                    self._show_resources()
            elif self.job is not None and self.job.ended:
                self.elapsed_text.set(f"{desktop.format_seconds(self.job.ended - self.job.started)} in all")
                self.resource_text.set("")
        self.root.after(150, self._tick)

    def _show_resources(self) -> None:
        try:
            import psutil

            top = psutil.Process(self.job.process.pid)
            family = [top, *top.children(recursive=True)]
            cpu = sum(p.cpu_percent(interval=None) for p in family)
            memory = sum(p.memory_info().rss for p in family) / 1e9
            self.resource_text.set(f"Scan: {len(family)} processes, CPU {cpu:.0f}%, memory {memory:.1f} GB")
        except Exception:
            self.resource_text.set("")

    # ---- the graphics card -------------------------------------------------------------

    def _check_gpu(self) -> None:
        try:
            usable, why = desktop.check_gpu()
        except Exception as error:      # never leave the box waiting
            usable, why = False, str(error)
        self.ui.put(("gpu", usable, why))

    def _show_gpu(self, usable: bool, why: str) -> None:
        self.use_gpu.set(usable)
        self.gpu_text.set("found; decoding is faster with it" if usable else f"not used: {why}")
        if not self._busy():
            self.gpu_box.configure(state="normal")
        self._log(f"Graphics card: {why}.")

    # ---- opening, logging, closing -----------------------------------------------------

    def _open(self, what: str) -> None:
        if what == "folder":
            target = Path(self.summary["folder"]) if self.summary else (self.job.folder if self.job else None)
        else:
            target = Path(self.summary[what]) if self.summary and self.summary.get(what) else None
        if target is None or not target.exists():
            messagebox.showinfo(APP_TITLE, "That is not there (yet).")
            return
        self._open_file(target)

    def _open_file(self, path: Path) -> None:
        try:
            open_path(path)
        except Exception as error:
            messagebox.showerror(APP_TITLE, f"Could not open {path}:\n{error}")

    def _log(self, text: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {text}\n"
        self.log_lines.append(line)
        del self.log_lines[:-2000]
        if self.log is not None and self.log.winfo_exists():
            self.log.configure(state="normal")
            self.log.insert("end", line)
            self.log.see("end")
            self.log.configure(state="disabled")

    def _callback_failed(self, kind, value, trace) -> None:
        details = "".join(traceback.format_exception(kind, value, trace))
        try:
            self._log(details.rstrip())
        except Exception:
            pass
        messagebox.showerror(APP_TITLE, f"Something went wrong in the window: {value}")

    def _close(self) -> None:
        if self._busy():
            if not messagebox.askyesno(APP_TITLE, "A scan is running. Stop it and close?"):
                return
            self.job.stop()
        self.thumb_queue.put(None)
        self.root.destroy()


def scan_without_window(video: Path, folder: Path) -> int:
    """One scan exactly as the window runs it (child process, default settings), progress printed; for checking a build."""
    desktop.prepare_environment()
    settings = desktop.ScanSettings(video=str(video), run_dir=str(folder), use_gpu=desktop.detect_gpu(),
                                    workers=desktop.default_workers())
    print(f"scanning with {settings}", flush=True)

    def show(event: tuple) -> None:
        if event[0] == "done":
            print("\n".join(event[1]["lines"]), f"\nFolder: {event[1]['folder']}", flush=True)
        elif event[0] == "error":
            print(event[1], event[2], sep="\n", flush=True)
        else:
            print(*event[1:3], flush=True)

    return 0 if desktop.run_in_child(settings, show)[0] == "done" else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--self-test"]:
        out = Path(argv[1]) if len(argv) > 1 else None
        report = desktop.self_test(out)
        if out is None:
            print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 1
    if argv[:1] == ["--scan"] and len(argv) >= 3:
        return scan_without_window(Path(argv[1]), Path(argv[2]))
    if sys.platform == "win32":
        try:      # sharp text on high-resolution screens instead of a blurred, stretched window
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = None
    if TkinterDnD is not None:
        try:
            root = TkinterDnD.Tk()
        except Exception:      # the drop library could not be loaded here
            root = None
    if root is None:
        globals()["DND_FILES"] = None
        root = tk.Tk()
    app = ScannerApp(root)
    if argv[:1] == ["--close-after"] and len(argv) > 1:      # for checking that the window opens; closes itself
        root.after(int(float(argv[1]) * 1000), app._close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    desktop.hide_console_windows()        # in the Windows app every helper process starts here too
    multiprocessing.freeze_support()
    desktop.ensure_std_streams("app.log")
    raise SystemExit(main())
