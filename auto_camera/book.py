"""From the captures of one video to flat page images and a PDF of the book.

``write_pages`` turns every capture (a still of a two-page spread) into two
page images with ``pages.page_images`` and writes them in reading order;
``write_pdf`` binds them into a PDF. With Tesseract installed the PDF is
searchable (the recognised text lies invisibly over each page image, as in a
scanned book) and a plain-text copy is written beside it; without it the PDF
holds the images only. With the text, a page that shows the same printed page
as one near it (``textrepeats``) is left out of the book, and its image is
moved to ``pages/repeated/``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .pages import page_images
from .textrepeats import repeated_pages

PAGE_DIR = "pages"
REPEAT_DIR = "repeated"      # under pages/: the images of pages left out as repeats
PDF_NAME = "book.pdf"
TEXT_NAME = "book.txt"
PAGE_HEIGHT_INCHES = 9.0     # the PDF shows a page this tall, whatever its pixel size
JPEG_QUALITY = 88
# Languages tried, of those Tesseract has installed: the books here are in
# English, French, German, Latin and Greek (ancient and modern).
OCR_LANGUAGES = ("eng", "fra", "deu", "lat", "grc", "ell", "ita")


def _one(job: tuple[str, str, np.ndarray | None, bool]) -> list[tuple[str, np.ndarray, int, bool]]:
    still, name, region, even = job
    image = cv2.imread(still)
    if image is None:
        return []
    return [(page.side, page.image, page.lines, page.unrolled) for page in page_images(image, region, even=even)]


def write_pages(video_dir: Path, region: np.ndarray | None = None, *, workers: int = 1, even: bool = True) -> list[Path]:
    """Page images for every capture of one video's run folder, in reading order, under ``video_dir/pages``.

    ``region`` is where print is in this video's frames (``hands.Hands.where``,
    a little enlarged); it keeps the edge of the desk and the like out of
    the lines of text. Returns the page files; ``pages.json`` beside them
    says which capture and side each came from.
    """
    summary = json.loads((video_dir / "spreads.json").read_text())
    captures = summary["captures"]
    out = video_dir / PAGE_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    jobs = [(str(video_dir / "captures" / c["file"]), c["file"], region, even) for c in captures]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_one, jobs, chunksize=2))
    else:
        results = [_one(job) for job in jobs]
    files, listing = [], []
    for capture, pages in zip(captures, results):
        for side, image, lines, unrolled in pages:
            path = out / f"page_{len(files) + 1:04d}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            files.append(path)
            listing.append({"page": path.name, "capture": capture["file"], "side": side, "lines": lines, "unrolled": unrolled,
                            "review": capture.get("review", [])})
    (out / "pages.json").write_text(json.dumps(listing, indent=1))
    uniform_pages(files)
    return files


def uniform_pages(files: list[Path]) -> None:
    """Give every page of a book the same shape, and a resolution that prints it PAGE_HEIGHT_INCHES tall.

    Each page is cut to its own print, so a page that ends a chapter comes
    out short. Shown at the height of a page it would be blown up. Instead
    every page is set on white paper at least as large as the book's usual
    page, of the usual shape, at the top and in the middle, so the print
    keeps one size throughout.
    """
    if not files:
        return
    sizes = []
    for path in files:
        with Image.open(path) as image:
            sizes.append(image.size)
    widths, heights = np.array([w for w, _ in sizes]), np.array([h for _, h in sizes])
    usual_w, usual_h = int(np.median(widths)), int(np.median(heights))
    shape = usual_w / usual_h
    for path, (w, h) in zip(files, sizes):
        canvas_w, canvas_h = max(w, usual_w), max(h, usual_h)
        if canvas_w / canvas_h < shape:
            canvas_w = int(round(canvas_h * shape))
        else:
            canvas_h = int(round(canvas_w / shape))
        with Image.open(path) as image:
            page = image.convert("RGB")
        if (canvas_w, canvas_h) != (w, h):
            corner = page.crop((0, 0, min(8, w), min(8, h)))
            paper = tuple(int(v) for v in np.median(np.asarray(corner).reshape(-1, 3), axis=0))
            sheet = Image.new("RGB", (canvas_w, canvas_h), paper)
            sheet.paste(page, ((canvas_w - w) // 2, 0))
            page = sheet
        dpi = canvas_h / PAGE_HEIGHT_INCHES
        page.save(path, quality=JPEG_QUALITY, dpi=(dpi, dpi))


def tesseract_languages() -> list[str]:
    """The languages of ``OCR_LANGUAGES`` that Tesseract has installed; empty without Tesseract."""
    if shutil.which("tesseract") is None:
        return []
    out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout.split()
    return [lang for lang in OCR_LANGUAGES if lang in out]


def write_pdf(pages: list[Path], out: Path, *, ocr: bool = True, workers: int = 4) -> dict:
    """Bind ``pages`` into ``out``; searchable if Tesseract is there and ``ocr``. Returns what was done.

    Tesseract reads a page in a few seconds with all the book languages on,
    so with ``pypdf`` installed the pages are read ``workers`` at a time,
    each into a one-page PDF, and joined; without it, one after another.
    """
    if not pages:
        return {"pages": 0, "searchable": False}
    with Image.open(pages[len(pages) // 2]) as middle:
        dpi = max(72, int(round(middle.size[1] / PAGE_HEIGHT_INCHES)))
    languages = tesseract_languages() if ocr else []
    if not languages:
        _jpeg_pdf(pages, out, dpi)
        return {"pages": len(pages), "searchable": False, "dpi": dpi}
    language = "+".join(languages)
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    try:
        from pypdf import PdfWriter
    except ImportError:
        PdfWriter = None
    with tempfile.TemporaryDirectory() as tmp:
        if PdfWriter is None or workers <= 1:
            listing = Path(tmp) / "pages.txt"
            listing.write_text("\n".join(str(p.resolve()) for p in pages) + "\n")
            base = Path(tmp) / "book"
            # Each page carries its own resolution (``uniform_pages``), which Tesseract uses for the page's size.
            subprocess.run(["tesseract", str(listing), str(base), "-l", language, "pdf", "txt"], check=True, capture_output=True, env=env)
            texts = base.with_suffix(".txt").read_text(errors="replace").split("\f")
            if len(texts) == len(pages) + 1 and not texts[-1].strip():
                texts = texts[:-1]       # the separator after the last page
            repeats = repeated_pages(texts, *_pictures(pages)) if len(texts) == len(pages) and PdfWriter is not None else []
            if repeats:
                from pypdf import PdfReader
                drop = {r["page"] for r in repeats}
                writer = PdfWriter()
                for k, page in enumerate(PdfReader(str(base.with_suffix(".pdf"))).pages):
                    if k not in drop:
                        writer.add_page(page)
                with open(out, "wb") as f:
                    writer.write(f)
                out.with_name(TEXT_NAME).write_text("\f".join(t for k, t in enumerate(texts) if k not in drop))
            else:
                shutil.move(str(base.with_suffix(".pdf")), out)
                shutil.move(str(base.with_suffix(".txt")), out.with_name(TEXT_NAME))
        else:
            def read(item: tuple[int, Path]) -> Path:
                index, page = item
                base = Path(tmp) / f"{index:05d}"
                subprocess.run(["tesseract", str(page.resolve()), str(base), "-l", language, "pdf", "txt"], check=True, capture_output=True, env=env)
                return base
            with ThreadPoolExecutor(workers) as pool:
                bases = list(pool.map(read, enumerate(pages)))
            texts = [base.with_suffix(".txt").read_text(errors="replace") for base in bases]
            repeats = repeated_pages(texts, *_pictures(pages))
            drop = {r["page"] for r in repeats}
            writer = PdfWriter()
            for k, base in enumerate(bases):
                if k not in drop:
                    writer.append(str(base.with_suffix(".pdf")))
            partial = out.with_suffix(".partial.pdf")
            with open(partial, "wb") as f:
                writer.write(f)
            partial.replace(out)
            out.with_name(TEXT_NAME).write_text("\f".join(t for k, t in enumerate(texts) if k not in drop))
    left_out = set_aside(pages, repeats)
    return {"pages": len(pages) - len(left_out), "searchable": True, "languages": languages, "dpi": dpi, "repeats": left_out}


def _pictures(pages: list[Path]) -> tuple[list[str] | None, list[int] | None]:
    """The capture each page was cut from and how many reasons it is flagged for, from the pages.json beside them."""
    listing = pages[0].parent / "pages.json"
    if not listing.exists():
        return None, None
    entries = {entry["page"]: entry for entry in json.loads(listing.read_text())}
    return ([entries[p.name]["capture"] if p.name in entries else p.name for p in pages],
            [len(entries[p.name].get("review", [])) if p.name in entries else 0 for p in pages])


def set_aside(pages: list[Path], repeats: list[dict]) -> list[dict]:
    """Move the images of the pages left out to pages/repeated/ and note them in pages.json; the notes."""
    notes = [{"page": pages[r["page"]].name, "same_as": pages[r["kept"]].name, "share": r["share"]} for r in repeats]
    if not notes:
        return notes
    folder = pages[0].parent
    (folder / REPEAT_DIR).mkdir(exist_ok=True)
    for note in notes:
        source = folder / note["page"]
        if source.exists():
            source.replace(folder / REPEAT_DIR / note["page"])
    listing = folder / "pages.json"
    if listing.exists():
        same_as = {note["page"]: note for note in notes}
        entries = json.loads(listing.read_text())
        for entry in entries:
            if entry["page"] in same_as:
                entry["left_out"] = {"same_as": same_as[entry["page"]]["same_as"], "share": same_as[entry["page"]]["share"]}
        partial = listing.with_suffix(".partial")      # a new file, not the old one rewritten (it may be a hard link)
        partial.write_text(json.dumps(entries, indent=1))
        partial.replace(listing)
    return notes


def _jpeg_pdf(pages: list[Path], out: Path, dpi: int) -> None:
    """A PDF of JPEG files, one per page, each embedded as it is (no re-encoding)."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    page_ids = []
    catalog_id = add(b"")            # filled in at the end
    pages_id = add(b"")
    for path in pages:
        data = path.read_bytes()
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        colour = b"/DeviceGray" if channels == 1 else b"/DeviceRGB"
        image_id = add(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace %s /BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n"
                       % (width, height, colour, len(data)) + data + b"\nendstream")
        w_pt, h_pt = width * PAGE_HEIGHT_INCHES * 72.0 / height, PAGE_HEIGHT_INCHES * 72.0
        content = b"q %.2f 0 0 %.2f 0 0 cm /Im0 Do Q" % (w_pt, h_pt)
        content_id = add(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        page_ids.append(add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /XObject << /Im0 %d 0 R >> >> /Contents %d 0 R >>"
                            % (pages_id, w_pt, h_pt, image_id, content_id)))
    objects[catalog_id - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id
    objects[pages_id - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % i for i in page_ids), len(page_ids))
    with open(out, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for number, body in enumerate(objects, start=1):
            offsets.append(f.tell())
            f.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
        xref = f.tell()
        f.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
        for offset in offsets:
            f.write(b"%010d 00000 n \n" % offset)
        f.write(b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog_id, xref))
