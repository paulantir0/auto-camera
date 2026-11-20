"""Write full-resolution stills, the final capture sequence and review sheets.

Stills are cached per video and named by frame number, so a change to turn
detection or to the merge step only decodes the frames it has not seen before.
A run folder holds hard links to the stills it uses, under sequence names.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .analysis import ROW_SLACK, Analysis, grab_frames, text_rows, usual_text_rows
from .paths import CACHE_ROOT
from .proxy import DEFAULT_PROXY_DIR, REPO_ROOT
from .spreads import Spread

DEFAULT_STILLS_DIR = CACHE_ROOT / "stills"
UNSETTLED_MOTION = 0.15   # chosen frame still had this much of the page moving
HAND_ON_PRINT = 0.003     # chosen frame has a hand over this share of the print area


def still_name(frame: int) -> str:
    return f"f{frame:06d}.jpg"


def capture_frames(capture: dict) -> list[int]:
    """The frames a capture is made from: its chosen frame, and the frames of its two pages when they differ."""
    best = capture["best"]
    return sorted({best, capture.get("left_frame", best), capture.get("right_frame", best)})


def two_frames(capture: dict) -> bool:
    """True when the capture's pages come from different frames (``spreads.TWO_FRAMES_STEPS``)."""
    best = capture["best"]
    return capture.get("left_frame", best) != best or capture.get("right_frame", best) != best


def capture_still(capture: dict, stills: Path) -> Path:
    """The image of a capture: its frame's still, or the left page of one frame beside the right page of another.

    The two halves meet at the fold, where the gutter's shadow hides the
    seam. The image is made once and kept beside the stills.
    """
    if not two_frames(capture):
        return stills / still_name(capture["best"])
    left, right = capture.get("left_frame", capture["best"]), capture.get("right_frame", capture["best"])
    path = stills / f"p{left:06d}_{right:06d}.jpg"
    if not path.exists():
        left_image = cv2.imread(str(stills / still_name(left)))
        right_image = cv2.imread(str(stills / still_name(right)))
        x = int(round(capture.get("fold", 0.5) * left_image.shape[1]))
        joined = right_image.copy()
        joined[:, :x] = left_image[:, :x]
        cv2.imwrite(str(path), joined, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def export_frames(video: Path, frames: list[int], out_dir: Path, names: list[str], *, use_gpu: bool = True,
                  quality: int = 2, threads: int = 4) -> None:
    """Decode ``video`` once and save the given frame indices as JPEGs.

    ``frames`` must be strictly ascending; ``names`` are the output file names
    in the same order. Decoding stops after the last wanted frame.
    """
    if not frames:
        return
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("frames must be strictly ascending")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        # A long select expression is passed through a filter script file, not the command line.
        script = Path(tmp) / "select.txt"
        script.write_text("select='" + "+".join(f"eq(n\\,{frame})" for frame in frames) + "'")
        decode = ["-hwaccel", "cuda"] if use_gpu else ["-threads", str(threads)]
        cmd = [
            "ffmpeg", "-v", "error", "-y", *decode, "-i", str(video),
            "-filter_script:v", str(script), "-fps_mode", "vfr", "-frames:v", str(len(frames)),
            "-q:v", str(quality), "-pix_fmt", "yuvj420p", "-threads", "2", str(Path(tmp) / "%06d.jpg"),
        ]
        subprocess.run(cmd, check=True)
        written = sorted(Path(tmp).glob("*.jpg"))
        if len(written) != len(frames):
            raise RuntimeError(f"asked for {len(frames)} frames, ffmpeg wrote {len(written)}")
        for source, name in zip(written, names):
            shutil.move(str(source), str(out_dir / name))


def ensure_stills(video: Path, frames: list[int], stills_dir: Path, *, use_gpu: bool = True, threads: int = 4) -> int:
    """Make sure a still exists for every frame; returns how many were decoded."""
    missing = sorted({f for f in frames if not (stills_dir / still_name(f)).exists()})
    export_frames(video, missing, stills_dir, [still_name(f) for f in missing], use_gpu=use_gpu, threads=threads)
    return len(missing)


def review_flags(spreads: list[Spread]) -> dict[int, list[str]]:
    """Reasons a person should look at a capture, keyed by spread index."""
    flags: dict[int, list[str]] = {}
    for spread in spreads:
        reasons = []
        if spread.hand_on_print > HAND_ON_PRINT:
            reasons.append("hand_on_print")
        if spread.best_motion > UNSETTLED_MOTION:
            reasons.append("never_at_rest")
        if reasons:
            flags[spread.index] = reasons
    return flags


def band_rows(analysis: Analysis, frames: list[int]) -> dict[int, list[float] | None]:
    """Per frame, where the head and the foot of the print may lie: ``[top_from, top_to, bottom_from, bottom_to]``.

    Fractions of the frame height, for the page-number reader. The first and
    last lines are measured in each frame, because the book creeps on the
    stand. They are set against where print usually starts and ends in this
    video (the median over the frames): a page that
    starts low or ends early, or ends in small-print notes that do not register
    as print, has its real head or foot at the usual place, not at its first
    or last strong line. The range always includes the usual place, and the
    frame's own line too when that lies outside it or close to it.
    """
    here = {frame: text_rows(analysis, frame) for frame in frames}
    known = [rows for rows in here.values() if rows]
    if not known:
        usual = usual_text_rows(analysis)
        return {frame: ([usual[0], usual[0], usual[1], usual[1]] if usual else None) for frame in frames}
    top, bottom = float(np.median([r[0] for r in known])), float(np.median([r[1] for r in known]))
    slack = ROW_SLACK * (bottom - top)
    out: dict[int, list[float] | None] = {}
    for frame, rows in here.items():
        first = rows[0] if rows and rows[0] <= top + slack else top
        last = rows[1] if rows and rows[1] >= bottom - slack else bottom
        out[frame] = [min(first, top), max(first, top), min(last, bottom), max(last, bottom)]
    return out


def page_box(mask: np.ndarray) -> list[float]:
    """Bounding box of the page region as fractions of the frame: x0, x1, y0, y1."""
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return [0.0, 1.0, 0.0, 1.0]
    h, w = mask.shape
    return [float(cols.min() / w), float(cols.max() / w), float(rows.min() / h), float(rows.max() / h)]


def contact_sheets(analysis: Analysis, captures: list[dict], out_dir: Path, *, proxy_dir: Path = DEFAULT_PROXY_DIR,
                   columns: int = 6, rows: int = 8) -> list[Path]:
    """Sheets of proxy-resolution thumbnails of the final captures, labelled."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    frames = grab_frames(proxy_dir / f"{analysis.slug}.mp4", [c["best"] for c in captures])
    tiles = []
    for capture in captures:
        tile = cv2.resize(frames[capture["best"]], (320, 180), interpolation=cv2.INTER_AREA)
        flagged = bool(capture.get("review"))
        label = f"{capture['index']} {capture['best'] / analysis.fps:.1f}s m{capture['best_motion']:.2f} h{capture.get('hand_on_print', 0) * 100:.1f}"
        colour = (0, 0, 255) if flagged else (0, 255, 255)
        cv2.putText(tile, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(tile, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        if flagged:
            cv2.rectangle(tile, (0, 0), (319, 179), (0, 0, 255), 2)
        tiles.append(tile)
    paths = []
    per_sheet = columns * rows
    for number, start in enumerate(range(0, len(tiles), per_sheet), 1):
        chunk = tiles[start:start + per_sheet]
        while len(chunk) % columns:
            chunk.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(chunk[r:r + columns]) for r in range(0, len(chunk), columns)])
        path = out_dir / f"contact_sheet_{number:03d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    return paths


def write_run(analysis: Analysis, spreads: list[Spread], video: Path, out_dir: Path, *, use_gpu: bool = True,
              stills_dir: Path | None = None, threads: int = 4, sheets: bool = True, hand_model: bool | None = None) -> dict:
    """Decode the candidate stills, choose the final captures and write the run folder.

    ``out_dir`` gets ``spreads.json`` (candidates and final captures),
    ``captures/`` (the final sequence, hard links to cached stills) and
    ``review/`` (contact sheets).
    """
    from .repeats import print_region, select_captures

    out_dir.mkdir(parents=True, exist_ok=True)
    stills = (stills_dir or DEFAULT_STILLS_DIR) / analysis.slug
    decoded = ensure_stills(video, sorted({f for s in spreads for f in capture_frames(s.as_dict(analysis.fps))}), stills,
                            use_gpu=use_gpu, threads=threads)
    flags = review_flags(spreads)
    rows = band_rows(analysis, [s.best for s in spreads])
    summary = {
        "slug": analysis.slug,
        "video": str(video),
        "fps": analysis.fps,
        "frames": analysis.frames,
        "page_box": page_box(analysis.mask),
        "hand_model": hand_model,
        "stills_dir": str(stills),
        "stills_decoded": decoded,
        "candidates": [dict(s.as_dict(analysis.fps), still=still_name(s.best), review=flags.get(s.index, []),
                            text_rows=rows.get(s.best)) for s in spreads],
    }
    (out_dir / "spreads.json").write_text(json.dumps(summary, indent=1))
    summary = select_captures(out_dir, print_region(analysis.mask, analysis.median))
    if sheets:
        contact_sheets(analysis, summary["captures"], out_dir / "review")
    return summary
