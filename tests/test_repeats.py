import unittest
from unittest import mock

import numpy as np

from auto_camera import repeats


def page(keypoints: int, name: str):
    """A stand-in for one page's features: ``keypoints`` points, and a name the fake matcher goes by."""
    return (np.zeros((keypoints, 2), dtype=np.float32), name)


def capture(motion: float, review: list[str] | None = None, hand: float = 0.0) -> dict:
    return {"best_motion": motion, "detail": 50.0, "hand_on_print": hand, "review": review or []}


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


class TransitionalTests(unittest.TestCase):
    """A frame that never came to rest and shares a page with a neighbour is a view of that neighbour's spread."""

    MOVING = ["never_at_rest"]
    COVERED = ["hand_on_print", "never_at_rest"]

    def run_rule(self, pages, candidates):
        def agreeing(a, b) -> int:
            return 100 if a[1] == b[1] else 0

        groups = [[i] for i in range(len(pages))]
        with mock.patch.object(repeats, "agreeing_points", agreeing):
            kept, dropped = repeats.drop_transitional(groups, candidates, pages)
        return [g[0] for g in kept], dict(dropped)

    def test_a_frame_between_two_settled_spreads_goes(self):
        pages = [(page(800, "L1"), page(800, "R1")), (page(800, "L1"), page(800, "R2")), (page(800, "L2"), page(800, "R2"))]
        kept, dropped = self.run_rule(pages, [capture(0.0), capture(0.5, self.MOVING), capture(0.0)])
        self.assertEqual(kept, [0, 2])
        self.assertEqual(dropped, {1: "left page matches the previous spread"})

    def test_a_calmer_view_takes_the_place_of_a_first_view_with_the_leaf_still_coming_down(self):
        # The spread 2 is never at rest. Its first view has the left page under the falling leaf and a hand on the
        # print; the second shows both pages and nearly rests.
        pages = [(page(800, "L1"), page(800, "R1")), (page(100, "leaf"), page(2400, "R2")), (page(1400, "L2"), page(1100, "R2")), (page(800, "L3"), page(800, "R3"))]
        candidates = [capture(0.0), capture(0.65, self.COVERED, hand=0.09), capture(0.2, self.MOVING), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates)
        self.assertEqual(kept, [0, 2, 3])
        self.assertEqual(dropped, {1: "a better view of its right page follows"})

    def test_a_clear_view_takes_the_place_of_one_with_a_hand_across_the_same_two_pages(self):
        # Both pages agree; the first view was not grouped with the second because a third lay between them.
        pages = [(page(800, "L1"), page(800, "R1")), (page(500, "L2"), page(2000, "R2")), (page(1900, "L2"), page(600, "R2")), (page(800, "L3"), page(800, "R3"))]
        candidates = [capture(0.0), capture(0.56, self.COVERED, hand=0.23), capture(0.18, self.MOVING), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates)
        self.assertEqual(kept, [0, 2, 3])
        self.assertEqual(dropped, {1: "a better view of its left page follows"})

    def test_the_first_view_stays_when_the_later_one_ranks_no_higher(self):
        pages = [(page(800, "L1"), page(800, "R1")), (page(100, "leaf"), page(2400, "R2")), (page(1400, "L2"), page(1100, "R2")), (page(800, "L3"), page(800, "R3"))]
        candidates = [capture(0.0), capture(0.3, self.MOVING), capture(0.3, self.COVERED, hand=0.05), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates)
        self.assertEqual(kept, [0, 1, 3])
        self.assertEqual(dropped, {2: "right page matches the previous spread"})

    def test_a_frame_from_the_next_turn_does_not_take_the_place_of_a_view_of_both_pages(self):
        # Spread 2 is never at rest but both its pages are in view; the next frame has its left page and the next
        # spread's right page, and less hand. Spread 2's right page is in the earlier view only.
        pages = [(page(800, "L1"), page(800, "R1")), (page(1300, "L2"), page(1200, "R2")), (page(1300, "L2"), page(1200, "R9")), (page(800, "L3"), page(800, "R3"))]
        candidates = [capture(0.0), capture(0.3, self.COVERED, hand=0.02), capture(0.2, self.MOVING), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates)
        self.assertEqual(kept, [0, 1, 3])
        self.assertEqual(dropped, {2: "left page matches the previous spread"})

    def test_a_view_of_the_next_settled_spread_takes_nobodys_place(self):
        pages = [(page(800, "L1"), page(800, "R1")), (page(100, "leaf"), page(2400, "R2")), (page(1400, "L2"), page(1100, "R2")), (page(800, "L2"), page(800, "R2"))]
        candidates = [capture(0.0), capture(0.65, self.COVERED, hand=0.09), capture(0.2, self.MOVING), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates)
        # both are views of the spread that then comes to rest, and both go
        self.assertEqual(kept, [0, 3])
        self.assertEqual(dropped, {1: "right page matches the next spread", 2: "left page matches the next spread"})

    def test_a_view_of_a_spread_that_settles_four_captures_on_takes_nobodys_place(self):
        # The settled view is too far ahead to vouch against the first view, but next to the second.
        pages = [(page(800, "L1"), page(800, "R1")), (page(100, "leaf"), page(2400, "R2")), (page(90, "blur"), page(80, "blur2")), (page(95, "blur3"), page(85, "blur4")),
                 (page(1400, "L2"), page(1100, "R2")), (page(800, "L2"), page(800, "R2"))]
        candidates = [capture(0.0), capture(0.65, self.COVERED, hand=0.09), capture(0.5, self.MOVING), capture(0.5, self.MOVING), capture(0.2, self.MOVING), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates)
        self.assertEqual(dropped.get(4), "left page matches the next spread")
        self.assertIn(1, kept)

    def test_a_view_that_would_only_be_a_disturbed_view_of_the_capture_before_takes_nobodys_place(self):
        # Without the first view, the second would be dropped for sharing its left page with the settled spread 1.
        pages = [(page(800, "L1"), page(800, "R1")), (page(100, "leaf"), page(2400, "R2")), (page(1400, "L1"), page(1100, "R2")), (page(800, "L3"), page(800, "R3"))]
        candidates = [capture(0.0), capture(0.65, self.COVERED, hand=0.09), capture(0.2, self.MOVING), capture(0.0)]
        kept, dropped = self.run_rule(pages, candidates)
        self.assertEqual(kept, [0, 1, 3])
        self.assertEqual(dropped, {2: "right page matches the previous spread"})


if __name__ == "__main__":
    unittest.main()
