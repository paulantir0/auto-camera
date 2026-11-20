# The desktop app

**Auto Camera Book Scanner** (`auto_camera_gui.py`) is a window around the
pipeline (`auto_camera.pipeline.run_video`).

1. Choose the video of the book (the button, or drop the file on the window).
   The window shows its first frame, size, frame rate and length.
2. Choose where the results go. By default that is `Pictures\AutoCamera\`
   and today's date; each video gets a folder of its own inside.
3. Press Start. Each step is listed as it runs, with the time it took, and a
   picture of each spread appears as soon as it is saved. Stop ends the scan
   at once; what it saved so far stays in the folder.

When it is done the window says how many spreads and pages it found, whether
the PDF is searchable, and how many spreads may need a look, with buttons to
open the folder, the PDF and the review list. The folder holds `captures\`
(one picture per spread, in page order), `pages\` (each page flattened and
cropped), `book.pdf`, `needs_review.txt` and `review\` contact sheets.

## Settings

How many CPU worker processes to use (four at most, by default one fewer than
the computer has cores), and whether to use an NVIDIA graphics card. The app
checks at start whether FFmpeg can decode on the card and ticks the box only
if it can; without one the CPU gives the same results, more slowly. If the
card fails on a particular video, the scan starts again on the CPU by itself.

## A searchable PDF

The PDF is searchable only when Tesseract is installed. On Windows install the
[UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) in its
default place (`C:\Program Files\Tesseract-OCR`) and tick the English, French,
German, Latin and Greek languages (`eng`, `fra`, `deu`, `lat`, `grc`, `ell`);
the app finds it there. Without it the PDF holds the page images only.

The printed-page-number check is not done in the built app (it needs the
separate OCR environment); run from source, the app does it when `.venv-ocr`
exists.

## Working files

The app keeps its working files (a small copy of the video, measurements, full
size frames) in `%LOCALAPPDATA%\AutoCameraBookScanner\cache`, one folder per
video file, and keeps those of the last three videos only. Run from source it
uses `data/cache` like the command-line tools. `AUTO_CAMERA_CACHE` and
`AUTO_CAMERA_MODELS` move the cache and the hand models
(`auto_camera/paths.py`).

## From source

Python with Tkinter, the packages in `requirements.txt`, FFmpeg on PATH and
the two hand models in `data/models` (see the README's quick start):

```bash
python auto_camera_gui.py
python auto_camera_gui.py --self-test report.json   # checks ffmpeg, the models and the hand models, no window
python auto_camera_gui.py --scan VIDEO FOLDER        # one scan as the window runs it, progress printed, no window
```

## The Windows build

GitHub Actions (`.github/workflows/windows-exe.yml`) builds
`AutoCameraBookScanner-windows-x64.exe` on manual workflow runs and version
tags. It bundles a static FFmpeg (the GPL build from BtbN/FFmpeg-Builds) and
Google's two MediaPipe models, checks the models against the ones the
pipeline was measured with, and runs the built app's `--self-test` before
publishing it. The exe needs no Python, FFmpeg or internet access on the
Windows computer. It is large (FFmpeg, OpenCV, SciPy and MediaPipe are inside)
and takes a few seconds to start.

The app has been run from source on Linux end to end. The Windows build has
not been run yet.
