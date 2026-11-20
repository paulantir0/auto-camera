"""Page flattening, the PDF writer and the CPU proxy's sampling, on drawn images (no video needed)."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from auto_camera import book, pages
from auto_camera.proxy import shrink_plane


def text_page(width=900, height=1300, pitch=34, margin=80, seed=1) -> np.ndarray:
    """A white page with lines of dark "words" at a regular pitch, as print looks to the line finder."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width), 235, np.uint8)
    for y in range(margin, height - margin, pitch):
        x = margin
        while x < width - margin - 40:
            w = int(rng.integers(25, 90))
            cv2.rectangle(page, (x, y), (min(x + w, width - margin), y + int(pitch * 0.45)), 30, -1)
            x += w + int(rng.integers(10, 18))
    return page


def spread(tilt_left=0.0, tilt_right=0.0, gap=120) -> np.ndarray:
    """Two text pages side by side on a dark background, each turned by a few degrees."""
    left, right = text_page(seed=1), text_page(seed=2)
    h, w = left.shape
    frame = np.full((h + 200, 2 * w + gap + 200), 40, np.uint8)
    for page, x0, tilt in ((left, 100, tilt_left), (right, 100 + w + gap, tilt_right)):
        turn = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
        turned = cv2.warpAffine(page, turn, (w, h), borderValue=235)
        frame[100:100 + h, x0:x0 + w] = turned
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


class ShrinkTests(unittest.TestCase):
    def test_a_flat_plane_stays_flat(self):
        plane = np.full((1728, 3072), 117, np.uint8)
        self.assertTrue((shrink_plane(plane, 480, 270) == 117).all())

    def test_each_target_sample_mixes_only_the_two_nearest_source_samples(self):
        # 8 columns to 2: centres at source x 1.5 and 5.5, so columns 1, 2 and 5, 6 at half weight each.
        plane = np.zeros((2, 8), np.uint8)
        plane[:, 1], plane[:, 2], plane[:, 5], plane[:, 6] = 100, 200, 10, 20
        plane[:, 0] = plane[:, 3] = plane[:, 4] = plane[:, 7] = 255      # never sampled: no prefilter
        out = shrink_plane(plane, 2, 2)
        self.assertEqual(out[0].tolist(), [150, 15])


class FoldTests(unittest.TestCase):
    def test_the_fold_is_the_gap_between_the_two_blocks_of_print(self):
        image = spread()
        fold = pages.find_fold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.assertTrue(100 + 900 - 80 <= fold <= 100 + 900 + 120 + 80, fold)

    def test_an_empty_outer_margin_is_not_taken_for_the_fold(self):
        # A wide empty band at the left of the left page: emptier than the fold, but with all the print on one side.
        image = spread()
        image[:, :700] = np.where(image[:, :700] < 100, 235, image[:, :700])
        image[:, :300] = 40
        fold = pages.find_fold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.assertGreater(fold, 900)


    def test_with_one_page_blank_the_fold_is_the_gap_nearest_the_middle_of_the_book(self):
        image = spread()
        image[:, :1000] = np.where(image[:, :1000] < 100, 235, image[:, :1000])     # the left page blank
        region = np.zeros(image.shape[:2], bool)
        region[100:-100, 100:-100] = True
        fold = pages.find_fold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), region)
        self.assertTrue(950 <= fold <= 1150, fold)


    def test_in_a_tightly_bound_book_the_narrow_gutter_is_the_fold(self):
        image = spread(gap=24)
        image[:, 100 + 900 - 80:100 + 900] = np.where(image[:, 100 + 900 - 80:100 + 900] < 100, 30, 235)[:, :80]     # print to the fold
        image[:, 100 + 900 + 24:100 + 900 + 24 + 80] = np.where(image[:, 100 + 900 + 24:100 + 900 + 24 + 80] < 100, 30, 235)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # the right page's print runs to its inner edge too
        gray[100 + 80:100 + 1300 - 80:34, 100 + 900 + 24:100 + 900 + 24 + 80] = 30
        gray[100 + 80:100 + 1300 - 80:34, 100 + 900 - 80:100 + 900] = 30
        region = np.zeros(gray.shape, bool)
        region[100:-100, 100:-100] = True
        fold = pages.find_fold(gray, region)
        self.assertTrue(100 + 900 - 30 <= fold <= 100 + 900 + 24 + 30, fold)

    def test_beside_a_sparse_title_page_the_fold_is_not_the_edge_of_the_book(self):
        image = spread()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray[100:1400, 100:2020] = 235                         # both pages nearly blank:
        gray[300:330, 300:700] = 30                            # a few lines on the left page,
        gray[500:530, 1300:1800] = 30                          # a title on the right one
        gray[700:760, 1250:1850] = 30
        gray[1100:1130, 1400:1700] = 30
        region = np.zeros(gray.shape, bool)
        region[100:-100, 100:-100] = True
        fold = pages.find_fold(gray, region)
        self.assertTrue(700 <= fold <= 1250, fold)      # between the left page's print and the right page's


class LineTests(unittest.TestCase):
    def test_pitch_is_found_on_a_level_and_a_turned_page(self):
        for tilt in (0.0, 8.0):
            page = cv2.warpAffine(text_page(), cv2.getRotationMatrix2D((450, 650), tilt, 1.0), (900, 1300), borderValue=235)
            self.assertAlmostEqual(pages.estimate_pitch(page), 34, delta=3)

    def test_every_line_is_found_once(self):
        image = cv2.cvtColor(text_page(), cv2.COLOR_GRAY2BGR)
        spans = pages.text_spans(image, 0, image.shape[1])
        rows = sorted({int(round(float(np.median(p[:, 1])) / 34)) for p in spans.points})
        expected = len(range(80, 1300 - 80, 34))
        self.assertGreaterEqual(len(rows), expected - 1)
        self.assertLessEqual(len(rows), expected)

    def test_a_long_isolated_stripe_is_not_a_line(self):
        image = text_page()
        image[1000:] = 235                       # the print ends here...
        image[1250:1262, 50:850] = 20            # ...and the edge of the desk lies well below it
        spans = pages.text_spans(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0, 900)
        self.assertFalse(any(float(np.median(p[:, 1])) > 1240 for p in spans.points))


class PageTests(unittest.TestCase):
    def test_turned_pages_come_out_level(self):
        out = pages.page_images(spread(tilt_left=5.0, tilt_right=-7.0), even=False)
        self.assertEqual([p.side for p in out], ["left", "right"])
        for page in out:
            spans = pages.text_spans(page.image, 0, page.image.shape[1])
            slopes = [abs(np.polyfit(p[:, 0], p[:, 1], 1)[0]) for p in spans.points if len(p) >= 5]
            self.assertGreater(len(slopes), 20)
            self.assertLess(float(np.median(slopes)), 0.01, page.side)

    def test_a_half_without_print_gives_no_page_without_a_region(self):
        image = spread()
        image[:, 1100:] = np.where(image[:, 1100:] < 100, 235, image[:, 1100:])
        self.assertEqual([p.side for p in pages.page_images(image, even=False)], ["left"])

    def test_even_paper_makes_shaded_paper_white_and_keeps_the_print_dark(self):
        page = cv2.cvtColor(text_page(), cv2.COLOR_GRAY2BGR).astype(np.float32)
        shade = np.linspace(0.55, 1.0, page.shape[1], dtype=np.float32)[None, :, None]
        out = pages.even_paper((page * shade).astype(np.uint8), 34)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        paper = gray[40:70, :]                        # above the first line
        self.assertLess(int(paper.max()) - int(paper.min()), 25)
        self.assertLess(int(np.median(gray[80:95, 100:800])), 120)


class PdfTests(unittest.TestCase):
    def test_jpeg_pages_are_bound_without_tesseract(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for index, size in enumerate(((300, 200), (320, 210))):
                path = Path(tmp) / f"page_{index}.jpg"
                cv2.imwrite(str(path), np.full((size[0], size[1], 3), 200, np.uint8))
                files.append(path)
            out = Path(tmp) / "book.pdf"
            result = book.write_pdf(files, out, ocr=False)
            data = out.read_bytes()
            self.assertEqual((result["pages"], result["searchable"]), (2, False))
            self.assertTrue(data.startswith(b"%PDF-1.4"))
            self.assertIn(b"/Count 2", data)
            self.assertEqual(data.count(b"/Subtype /Image"), 2)
            # Every object starts where the cross-reference table says it does.
            start = int(data.rsplit(b"startxref\n", 1)[1].split(b"\n")[0])
            table = data[start:].split(b"\n")
            count = int(table[1].split()[1])
            for number, row in enumerate(table[3:3 + count - 1], start=1):
                offset = int(row.split()[0])
                self.assertTrue(data[offset:].startswith(b"%d 0 obj" % number))


if __name__ == "__main__":
    unittest.main()


class TwoFrameCaptureTests(unittest.TestCase):
    def test_the_left_page_comes_from_one_frame_and_the_right_from_the_other(self):
        from auto_camera.export import capture_frames, capture_still, still_name
        with tempfile.TemporaryDirectory() as tmp:
            stills = Path(tmp)
            cv2.imwrite(str(stills / still_name(10)), np.full((40, 100, 3), 50, np.uint8))
            cv2.imwrite(str(stills / still_name(14)), np.full((40, 100, 3), 200, np.uint8))
            capture = {"best": 14, "left_frame": 10, "right_frame": 14, "fold": 0.4}
            self.assertEqual(capture_frames(capture), [10, 14])
            joined = cv2.imread(str(capture_still(capture, stills)))
            self.assertLess(abs(int(joined[20, 20, 0]) - 50), 4)
            self.assertLess(abs(int(joined[20, 80, 0]) - 200), 4)
            self.assertEqual(capture_still({"best": 14}, stills), stills / still_name(14))
