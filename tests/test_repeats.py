import unittest
from unittest import mock

import numpy as np

from auto_camera import repeats


def page(keypoints: int, name: str):
    """A stand-in for one page's features: ``keypoints`` points, and a name the fake matcher goes by."""
    return (np.zeros((keypoints, 2), dtype=np.float32), name)


def capture(motion: float, review: list[str] | None = None) -> dict:
    return {"best_motion": motion, "detail": 50.0, "hand_on_print": 0.0, "review": review or []}


class ShownElsewhereTests(unittest.TestCase):
    """A capture that never came to rest goes only when everything legible in it is shown properly close by."""

    def run_rule(self, pages, candidates, same: set[frozenset]):
        def agreeing(a, b) -> int:
            return 40 if a[1] == b[1] or frozenset((a[1], b[1])) in same else 0

        groups = [[i] for i in range(len(pages))]
        with mock.patch.object(repeats, "agreeing_points", agreeing):
            kept, dropped = repeats.drop_shown_elsewhere(groups, candidates, pages)
        return [g[0] for g in kept], [p for p, _ in dropped]

    def test_a_glimpse_of_a_spread_shown_properly_next_goes(self):
        pages = [(page(800, "L1"), page(800, "R1")), (page(700, "L2"), page(100, "blur")), (page(800, "L2"), page(800, "R2"))]
        candidates = [capture(0.0), capture(0.4, ["never_at_rest"]), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates, set())
        self.assertEqual(kept, [0, 2])
        self.assertEqual(dropped, [1])

    def test_a_capture_with_a_page_of_its_own_stays(self):
        pages = [(page(800, "L1"), page(800, "R1")), (page(800, "L2"), page(800, "R-only-here")), (page(800, "L2"), page(800, "R2"))]
        candidates = [capture(0.0), capture(0.4, ["never_at_rest"]), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates, set())
        self.assertEqual(kept, [0, 1, 2])

    def test_a_plain_page_in_full_view_beside_a_dense_one_is_legible(self):
        # The left page takes 82% of the capture's keypoints and happens to agree with the next spread's left page;
        # the right page has 460, about what a right page has in this video, and is shown nowhere else.
        pages = [(page(2000, "L1"), page(500, "R1")), (page(2040, "L2"), page(460, "R2")), (page(1600, "L3"), page(900, "R3")), (page(1750, "L4"), page(750, "R4"))]
        candidates = [capture(0.05), capture(0.16, ["never_at_rest"]), capture(0.1), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates, {frozenset(("L2", "L3"))})
        self.assertEqual(kept, [0, 1, 2, 3])
        self.assertEqual(dropped, [])

    def test_a_page_mostly_under_a_hand_is_still_not_legible(self):
        pages = [(page(2000, "L1"), page(700, "R1")), (page(2300, "L2"), page(160, "hand")), (page(1600, "L2"), page(900, "R2")), (page(1750, "L4"), page(750, "R4"))]
        candidates = [capture(0.05), capture(0.5, ["hand_on_print", "never_at_rest"]), capture(0.1), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates, set())
        self.assertEqual(dropped, [1])

    def test_a_capture_that_came_to_rest_is_never_dropped(self):
        pages = [(page(800, "L1"), page(800, "R1")), (page(700, "L2"), page(100, "blur")), (page(800, "L2"), page(800, "R2"))]
        candidates = [capture(0.0), capture(0.05, ["hand_on_print"]), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates, set())
        self.assertEqual(kept, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
