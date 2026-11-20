import unittest

import cv2
import numpy as np

from auto_camera import hands
from auto_camera.analysis import GRID_H, GRID_W, Analysis

HEIGHT, WIDTH = 270, 480


def analysis_with(motion: np.ndarray, mask: np.ndarray | None = None) -> Analysis:
    count = len(motion)
    page = np.zeros((GRID_H, GRID_W), dtype=bool)
    page[3:24, 4:44] = True
    return Analysis(
        slug="synthetic", fps=11.0,
        thumbs=np.zeros((count, 54, 96), dtype=np.uint8), detail=np.zeros((count, GRID_H, GRID_W), dtype=np.uint8),
        moved=np.zeros((count, GRID_H, GRID_W), dtype=bool), median=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
        mask=np.zeros((HEIGHT, WIDTH), dtype=bool) if mask is None else mask,
        grid_page=page, grid_left=page, grid_right=page,
        page_motion=motion.astype(np.float32), left_motion=motion.astype(np.float32), right_motion=motion.astype(np.float32),
        page_detail=np.zeros(count, dtype=np.float32),
    )


class FineDetailTests(unittest.TestCase):
    """Print has strokes in every direction; the edge of a book has them in one."""

    def picture(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        gray = np.full((HEIGHT, WIDTH), 200, dtype=np.uint8)
        gray[40:120, 40:200] = rng.integers(40, 220, size=(80, 160), dtype=np.uint8)     # print: detail both ways
        gray[40:120, 240:300:2] = 60                                                     # a stack of page edges: vertical lines
        gray[150:200:2, 40:200] = 60                                                     # the foot of the book block: horizontal lines
        return gray

    def test_print_scores_and_lines_one_way_do_not(self):
        detail = hands._fine_detail(self.picture())
        print_level = float(np.median(detail[50:110, 50:190]))
        self.assertGreater(print_level, 20)
        self.assertLess(float(detail[50:110, 250:290].max()), 0.2 * print_level)
        self.assertLess(float(detail[160:190, 50:190].max()), 0.2 * print_level)

    def test_plain_paper_and_skin_are_flat(self):
        detail = hands._fine_detail(self.picture())
        self.assertEqual(int(detail[220:260, 300:460].max()), 0)


class PrintMapTests(unittest.TestCase):
    """One stretch's map from its frames at rest (``_print_map``)."""

    def setUp(self):
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[20:250, 30:450] = True
        self.analysis = analysis_with(np.zeros(40), mask)
        self.cells = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.cells[60:200, 60:220] = True               # where print usually is, by the coarse cell measure
        rng = np.random.default_rng(1)
        frames = 40
        self.gray = np.full((frames, HEIGHT, WIDTH), 200, dtype=np.uint8)
        self.detail = np.zeros((frames, HEIGHT, WIDTH), dtype=np.uint8)
        for index in range(frames):                     # a different page of print in every frame
            self.gray[index, 60:200, 60:220] = rng.integers(40, 220, size=(140, 160), dtype=np.uint8)
            self.gray[index, 44:50, 60:220] = rng.integers(40, 220, size=(6, 160), dtype=np.uint8)      # running head, a few pixels clear of the body
            self.gray[index, 60:200, 240:252] = rng.integers(40, 220, size=(140, 12), dtype=np.uint8)   # a strip across a margin: page edges
            self.detail[index] = hands._fine_detail(self.gray[index])
        self.visible = np.ones((frames, HEIGHT, WIDTH), dtype=bool)
        self.usual = float(np.median(self.detail[:, 80:180, 80:200]))

    def test_the_body_and_its_running_head_are_print_and_a_strip_across_the_margin_is_not(self):
        found = hands._print_map(self.analysis, self.cells, self.detail, self.visible, self.gray, self.usual)
        self.assertTrue(found[120, 140])
        self.assertTrue(found[47, 140], "the running head belongs to the page")
        self.assertFalse(found[120, 246], "page edges beside the print are not print")
        self.assertFalse(found[230, 400])

    def test_print_under_a_hand_that_is_there_most_of_the_time_is_still_print(self):
        # A hand lies on the top outer corner in 30 of the 40 frames: no print to see there, and the model knows it is a hand.
        for index in range(30):
            self.gray[index, 60:110, 170:220] = 150
            self.detail[index] = hands._fine_detail(self.gray[index])
            self.visible[index, 56:114, 166:224] = False
        found = hands._print_map(self.analysis, self.cells, self.detail, self.visible, self.gray, self.usual)
        self.assertTrue(found[85, 195])

    def test_what_never_changes_is_not_print(self):
        self.gray[:, 215:225, 300:420] = self.gray[0, 215:225, 300:420] = np.tile(np.array([40, 220], dtype=np.uint8), (10, 60))
        for index in range(len(self.gray)):
            self.detail[index] = hands._fine_detail(self.gray[index])
        found = hands._print_map(self.analysis, self.cells, self.detail, self.visible, self.gray, self.usual)
        self.assertFalse(found[220, 360])


class CoveredTests(unittest.TestCase):
    """A patch of hand colour counts only if it reaches past the page (unchanged behaviour, pinned here)."""

    def setUp(self):
        lut = np.zeros((hands.BINS, hands.BINS), dtype=bool)
        skin = cv2.cvtColor(np.full((1, 1, 3), (120, 150, 210), dtype=np.uint8), cv2.COLOR_BGR2LAB)
        a, b = hands._bins(skin)
        lut[int(a[0, 0]), int(b[0, 0])] = True
        self.model = hands.HandModel(lut, 10000, True)
        self.region = np.zeros((HEIGHT, WIDTH), dtype=bool)
        self.region[40:230, 60:420] = True
        self.frame = np.full((HEIGHT, WIDTH, 3), 230, dtype=np.uint8)

    def test_a_hand_coming_in_from_the_side_counts(self):
        self.frame[100:140, 380:480] = (120, 150, 210)
        hit = hands.covered(self.frame, self.model, self.region)
        self.assertGreater(int(hit.sum()), 1000)
        self.assertFalse(hit[~self.region].any())

    def test_a_plate_inside_the_page_does_not(self):
        self.frame[100:160, 200:300] = (120, 150, 210)
        self.assertEqual(int(hands.covered(self.frame, self.model, self.region).sum()), 0)


if __name__ == "__main__":
    unittest.main()
