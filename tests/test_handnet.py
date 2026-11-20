import unittest

import numpy as np

from auto_camera import handnet

SHAPE = (270, 480)


def hand_points(wrist=(0.80, 0.60), size=0.08):
    """21 landmarks of an open hand pointing up and to the left, as shares of width and height."""
    wx, wy = wrist
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = (wx, wy)
    knuckles = [(-0.9, -0.4), (-0.5, -1.0), (-0.1, -1.1), (0.25, -1.0), (0.55, -0.85)]   # thumb base, then index .. little
    for finger, (dx, dy) in enumerate(knuckles):
        base = 1 + 4 * finger
        for joint in range(4):
            reach = 1.0 + 0.35 * joint
            points[base + joint] = (wx + dx * size * reach * 0.56, wy + dy * size * reach)
    return points


class SilhouetteTests(unittest.TestCase):
    def test_a_hand_covers_its_palm_and_fingers_and_not_far_beyond(self):
        points = hand_points()
        mask = handnet.silhouette(points[None], np.array([0.9]), SHAPE)
        wrist = (points[0] * [480, 270]).astype(int)
        palm_middle = ((points[0] + points[9]) / 2 * [480, 270]).astype(int)
        self.assertTrue(mask[palm_middle[1], palm_middle[0]])
        self.assertTrue(mask[wrist[1], wrist[0]])
        self.assertFalse(mask[20, 20])
        self.assertLess(mask.mean(), 0.05)

    def test_a_doubtful_detection_draws_nothing(self):
        self.assertFalse(handnet.silhouette(hand_points()[None], np.array([0.2]), SHAPE).any())

    def test_no_hand_draws_nothing(self):
        self.assertFalse(handnet.silhouette(np.full((2, 21, 2), np.nan, np.float32), np.zeros(2), SHAPE).any())


class DetectionsTests(unittest.TestCase):
    def detections(self, skin: np.ndarray) -> handnet.Detections:
        bits = np.packbits(skin.reshape(1, -1), axis=1)
        return handnet.Detections(np.full((1, 1, 21, 2), np.nan, np.float32), np.zeros((1, 1), np.float32), bits, SHAPE)

    def setUp(self):
        self.page = np.zeros(SHAPE, dtype=bool)
        self.page[30:240, 60:420] = True

    def test_skin_that_reaches_in_from_outside_the_page_is_a_hand(self):
        skin = np.zeros(SHAPE, dtype=bool)
        skin[50:90, 380:480] = True                # fingers on the right page, arm off the right edge
        mask = self.detections(skin).mask(0, self.page)
        self.assertTrue(mask[60, 400])

    def test_skin_wholly_inside_the_page_is_printed_there(self):
        skin = np.zeros(SHAPE, dtype=bool)
        skin[100:150, 200:260] = True              # a sepia portrait
        self.assertFalse(self.detections(skin).mask(0, self.page).any())

    def test_specks_are_ignored(self):
        skin = np.zeros(SHAPE, dtype=bool)
        skin[100:102, 418:422] = True               # 8 pixels across the page edge
        self.assertFalse(self.detections(skin).mask(0, self.page).any())

    def test_cover_is_the_share_of_the_print_under_a_hand(self):
        skin = np.zeros(SHAPE, dtype=bool)
        skin[40:80, 300:480] = True
        where = np.zeros(SHAPE, dtype=bool)
        where[40:240, 280:400] = True
        share = handnet.cover(self.detections(skin), where, self.page, 1)
        self.assertAlmostEqual(float(share[0]), (40 * 100) / (200 * 120), places=2)    # the corners of the patch are rounded off


class SkinRulesTests(unittest.TestCase):
    """A patch of skin is a hand when it reaches the edge of the picture or holds a palm the landmarker found."""

    def detections(self, skin: np.ndarray, landmarks=None, scores=None) -> handnet.Detections:
        landmarks = np.full((1, 1, 21, 2), np.nan, np.float32) if landmarks is None else landmarks
        scores = np.zeros((1, 1), np.float32) if scores is None else scores
        return handnet.Detections(landmarks, scores, np.packbits(skin.reshape(1, -1), axis=1), SHAPE)

    def test_bands_thinner_than_a_finger_are_dropped(self):
        skin = np.zeros(SHAPE, dtype=bool)
        skin[40:90, 360:480] = True                # a real hand, arm off the right edge
        for row in range(120, 160, 6):
            skin[row:row + 3, 300:480] = True      # bands three pixels high, also reaching the edge
        mask = self.detections(skin).mask(0)
        self.assertTrue(mask[60, 400])
        self.assertFalse(mask[120:160, 300:470].any())

    def test_a_patch_in_the_middle_of_the_picture_is_a_hand_only_with_a_palm_in_it(self):
        skin = np.zeros(SHAPE, dtype=bool)
        skin[100:170, 200:300] = True              # fingers that come over the top of the book, the arm hidden behind it
        self.assertFalse(self.detections(skin).mask(0).any())
        points = np.full((1, 1, 21, 2), np.nan, np.float32)
        points[0, 0] = [[0.5, 0.55]] * 21
        for joint, (x, y) in zip(handnet.PALM, [(0.45, 0.6), (0.43, 0.5), (0.47, 0.45), (0.5, 0.44), (0.53, 0.45), (0.56, 0.5)]):
            points[0, 0, joint] = (x, y)
        mask = self.detections(skin, points, np.array([[0.9]], np.float32)).mask(0)
        self.assertTrue(mask[130, 250])

    def test_the_landmarks_alone_draw_nothing(self):
        # The landmarker places the fingers of the holding hand behind the page; without skin there, nothing is covered.
        points = hand_points()[None, None]
        self.assertFalse(self.detections(np.zeros(SHAPE, dtype=bool), points, np.array([[0.9]], np.float32)).mask(0).any())


if __name__ == "__main__":
    unittest.main()
