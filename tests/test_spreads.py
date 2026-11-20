import unittest

import numpy as np

from auto_camera.analysis import GRID_H, GRID_W, Analysis, find_valleys
from auto_camera.spreads import detect_turns, group_spreads, hidden_print

LEFT = np.zeros((GRID_H, GRID_W), dtype=bool)
LEFT[3:24, 4:22] = True
RIGHT = np.zeros((GRID_H, GRID_W), dtype=bool)
RIGHT[3:24, 26:44] = True


def synthetic(events: list[str], detail: dict[int, float] | None = None) -> Analysis:
    """Build an Analysis from a per-frame script.

    "." nothing moves, "R" the whole right page moves (leaf lifting), "L" the
    whole left page moves (leaf landing), "r" the outer third of the right page
    moves (a hand at the edge), "l" the lower half of the left page moves (a
    hand pressing the page).
    """
    count = len(events)
    moved = np.zeros((count, GRID_H, GRID_W), dtype=bool)
    for index, event in enumerate(events):
        if event == "R":
            moved[index][RIGHT] = True
        elif event == "L":
            moved[index][LEFT] = True
        elif event == "r":
            moved[index, 3:24, 38:44] = True
        elif event == "l":
            moved[index, 14:24, 4:22] = True
    page = LEFT | RIGHT

    def share(region: np.ndarray) -> np.ndarray:
        return moved[:, region].mean(axis=1).astype(np.float32)

    page_detail = np.full(count, 10.0, dtype=np.float32)
    for index, value in (detail or {}).items():
        page_detail[index] = value
    return Analysis(
        slug="synthetic", fps=11.0,
        thumbs=np.zeros((count, 54, 96), dtype=np.uint8), detail=np.zeros((count, GRID_H, GRID_W), dtype=np.uint8),
        moved=moved, median=np.zeros((270, 480, 3), dtype=np.uint8), mask=np.zeros((270, 480), dtype=bool),
        grid_page=page, grid_left=LEFT, grid_right=RIGHT,
        page_motion=share(page), left_motion=share(LEFT), right_motion=share(RIGHT), page_detail=page_detail,
    )


class FindValleysTests(unittest.TestCase):
    def test_flat_low_stretch_gives_one_valley_per_window(self):
        motion = np.array([0.9] * 5 + [0.0] * 3 + [0.9] * 5, dtype=np.float32)
        self.assertEqual(len(find_valleys(motion)), 1)

    def test_shallow_dip_inside_a_burst_is_ignored(self):
        motion = np.array([0.9, 0.9, 0.8, 0.9, 0.9], dtype=np.float32)
        self.assertEqual(find_valleys(motion), [])

    def test_clear_dip_that_never_reaches_rest_still_counts(self):
        motion = np.array([0.9] * 4 + [0.3] + [0.9] * 4, dtype=np.float32)
        self.assertEqual(find_valleys(motion), [4])


class DetectTurnsTests(unittest.TestCase):
    TURN = list("RRLL")

    def test_each_turn_is_one_landing(self):
        events = list("....") + self.TURN + list(".....") + self.TURN + list(".....")
        turns = detect_turns(synthetic(events))
        self.assertEqual(len(turns.landings), 2)

    def test_hand_at_the_page_edge_is_not_a_turn(self):
        events = list("....") + list("rrr.rrr.rr") + list("....")
        self.assertEqual(detect_turns(synthetic(events)).landings, [])

    def test_hand_pressing_half_the_left_page_is_not_a_turn(self):
        events = list("....") + self.TURN + list("..") + list("lll") + list("....")
        self.assertEqual(len(detect_turns(synthetic(events)).landings), 1)

    def test_leaf_lifted_and_laid_back_is_not_a_turn(self):
        events = list("....") + list("RR..RR") + list("....")
        self.assertEqual(detect_turns(synthetic(events)).landings, [])

    def test_leaf_held_upright_then_landed_is_one_turn(self):
        events = list("....") + list("RR") + list("......") + list("LL") + list("....")
        self.assertEqual(len(detect_turns(synthetic(events)).landings), 1)


class GroupSpreadsTests(unittest.TestCase):
    def test_picks_sharpest_still_frame_between_landings(self):
        events = list("....") + list("RRLL") + list("........") + list("RRLL") + list("....")
        analysis = synthetic(events, detail={10: 30.0, 11: 50.0, 12: 30.0})
        spreads = group_spreads(detect_turns(analysis), analysis)
        self.assertEqual(len(spreads), 3)
        self.assertEqual(spreads[1].best, 11)

    def test_moving_frame_loses_to_a_still_one_of_similar_detail(self):
        events = list("....") + list("RRLL") + list("....rrr.") + list("RRLL") + list("....")
        analysis = synthetic(events, detail={9: 48.0, 13: 50.0})
        spreads = group_spreads(detect_turns(analysis), analysis)
        self.assertEqual(spreads[1].best, 9)


class FakeHands:
    usable = True

    def __init__(self, count: int, on_print: dict[int, float]):
        self.share = np.zeros(count, dtype=np.float32)
        self.on_print = np.zeros(count, dtype=np.float32)
        self.paper = np.ones(count, dtype=np.float32)
        for index, value in on_print.items():
            self.on_print[index] = self.share[index] = value


class HandAwareChoiceTests(unittest.TestCase):
    EVENTS = list("....") + list("RRLL") + list("........") + list("RRLL") + list("....")

    def test_hand_free_frame_beats_a_sharper_one_with_a_hand_on_the_print(self):
        analysis = synthetic(self.EVENTS, detail={10: 40.0, 11: 50.0})
        hands = FakeHands(len(self.EVENTS), {11: 0.05})
        spreads = group_spreads(detect_turns(analysis), analysis, hands)
        self.assertEqual(spreads[1].best, 10)
        self.assertTrue(spreads[1].hand_free_exists)

    def test_a_much_softer_hand_free_frame_is_not_taken(self):
        analysis = synthetic(self.EVENTS, detail={10: 20.0, 11: 50.0})
        hands = FakeHands(len(self.EVENTS), {n: 0.05 for n in range(8, 16) if n != 10})
        spreads = group_spreads(detect_turns(analysis), analysis, hands)
        self.assertEqual(spreads[1].best, 11)
        self.assertFalse(spreads[1].hand_free_exists)
        self.assertGreater(spreads[1].hand_on_print, 0)

    def test_without_a_hand_model_the_sharpest_frame_wins(self):
        analysis = synthetic(self.EVENTS, detail={10: 40.0, 11: 50.0})
        spreads = group_spreads(detect_turns(analysis), analysis, None)
        self.assertEqual(spreads[1].best, 11)


if __name__ == "__main__":
    unittest.main()


class CalmFramesTests(unittest.TestCase):
    """A frame taken while a leaf is in the air must not be chosen when the spread was shown at rest."""

    EVENTS = list("....") + list("RRLL") + list("........") + list("RRLL") + list("....")

    def test_a_moving_frame_with_no_hand_loses_to_still_frames_with_one(self):
        # Something reads as a hand on every still frame (a colour plate did, once);
        # frame 16 is the next leaf lifting, sharp and "hand-free".
        analysis = synthetic(self.EVENTS, detail={n: 50.0 for n in range(8, 18)})
        hands = FakeHands(len(self.EVENTS), {n: 0.05 for n in range(8, 16)})
        spreads = group_spreads(detect_turns(analysis), analysis, hands)
        self.assertIn(spreads[1].best, range(8, 15))
        self.assertFalse(spreads[1].hand_free_exists)

    def test_a_spread_never_at_rest_still_gets_its_calmest_frame(self):
        events = list("....") + list("RRLL") + list("rrrrlrrr") + list("RRLL") + list("....")
        analysis = synthetic(events, detail={n: 50.0 for n in range(8, 16)})
        spreads = group_spreads(detect_turns(analysis), analysis, FakeHands(len(events), {}))
        self.assertIn(spreads[1].best, range(8, 16))
        self.assertNotEqual(spreads[1].best, 12)        # the frame where half the left page moved


class WeakLandingTests(unittest.TestCase):
    def test_a_weak_sweep_in_an_interval_twice_too_long_is_a_landing(self):
        from auto_camera.spreads import WEAK_LAND_COVERAGE
        turn, rest = list("RRLL"), list("........")
        events = list("....") + (turn + rest) * 4 + list("RRWW") + rest + (turn + rest) * 4 + list("....")
        count = len(events)
        analysis = synthetic(["L" if e == "W" else e for e in events])
        weak_at = [i for i, e in enumerate(events) if e == "W"]
        # the weak landing sweeps only the upper 80% of the left page
        for index in weak_at:
            analysis.moved[index] = False
            analysis.moved[index, 3:20, 4:22] = True
        for name, region in (("page_motion", LEFT | RIGHT), ("left_motion", LEFT), ("right_motion", RIGHT)):
            setattr(analysis, name, analysis.moved[:, region].mean(axis=1).astype(np.float32))
        self.assertGreater(17 / 21, WEAK_LAND_COVERAGE)
        turns = detect_turns(analysis)
        self.assertEqual(len(turns.landings), 9)
        self.assertEqual(sum(turns.weak), 1)

    def test_a_weak_sweep_in_an_ordinary_interval_is_not(self):
        turn, rest = list("RRLL"), list("........")
        events = list("....") + (turn + rest) * 4 + list("RRLL") + list("..ll....") + (turn + rest) * 4 + list("....")
        turns = detect_turns(synthetic(events))
        self.assertEqual(len(turns.landings), 9)
        self.assertEqual(sum(turns.weak), 0)


class HiddenPrintTests(unittest.TestCase):
    """Print that the calm frames of a spread show and one frame does not."""

    def frames(self, count: int = 6) -> np.ndarray:
        detail = np.zeros((count, GRID_H, GRID_W), dtype=np.uint8)
        detail[:, LEFT | RIGHT] = 60
        return detail

    def test_the_flat_of_a_hand_over_the_print_is_seen(self):
        detail = self.frames()
        detail[4, 5:11, 30:37] = 5                       # a hand on the top of the right page in frame 4
        hidden = hidden_print(detail, np.ones(6, dtype=bool), LEFT | RIGHT, 20.0)
        self.assertGreater(hidden[4], 0.01)
        self.assertEqual(float(hidden[[0, 1, 2, 3, 5]].max()), 0.0)

    def test_one_cell_is_not_a_hand(self):
        # A held book shifts by a few pixels between frames, and a cell on the edge of a line loses its print.
        detail = self.frames()
        detail[4, 8, 30] = 5
        detail[4, 15, 12:14] = 5
        self.assertEqual(float(hidden_print(detail, np.ones(6, dtype=bool), LEFT | RIGHT, 20.0).max()), 0.0)

    def test_a_frame_that_is_softer_all_over_hides_nothing(self):
        detail = self.frames()
        detail[3] = detail[3] // 3
        self.assertEqual(float(hidden_print(detail, np.ones(6, dtype=bool), LEFT | RIGHT, 20.0).max()), 0.0)

    def test_a_blank_half_page_hides_nothing(self):
        detail = self.frames()
        detail[:, 14:24, 26:44] = 4                      # the right page ends half way down, in every frame
        self.assertEqual(float(hidden_print(detail, np.ones(6, dtype=bool), LEFT | RIGHT, 20.0).max()), 0.0)

    def test_the_spread_is_what_its_reference_frames_show(self):
        # Frames 4 and 5 show the next right page, printed where this one is blank; they are not reference frames.
        detail = self.frames()
        detail[:4, 14:24, 26:44] = 4
        reference = np.array([True, True, True, True, False, False])
        self.assertEqual(float(hidden_print(detail, reference, LEFT | RIGHT, 20.0)[:4].max()), 0.0)

    def test_without_a_frame_at_rest_nothing_is_said(self):
        detail = self.frames()
        detail[4, 5:11, 30:37] = 5
        self.assertEqual(float(hidden_print(detail, np.zeros(6, dtype=bool), LEFT | RIGHT, 20.0).max()), 0.0)

    def test_one_odd_frame_at_rest_does_not_say_what_the_spread_looks_like(self):
        # Frame 0 has a hand lifting the next leaf: fine detail of its own where this spread's right page is blank.
        detail = self.frames()
        detail[1:, 14:24, 26:44] = 4
        self.assertEqual(float(hidden_print(detail, np.ones(6, dtype=bool), LEFT | RIGHT, 20.0)[1:].max()), 0.0)

    def test_a_blank_stretch_of_page_hides_nothing_whatever_the_other_frames_show(self):
        # Half the frames at rest show another leaf, printed where this spread's right page ends half way down.
        detail = self.frames()
        detail[3:, 14:24, 26:44] = 4
        light = np.full(detail.shape, 150.0)             # print is grey
        light[:, ~(LEFT | RIGHT)] = 200.0
        light[3:, 14:24, 26:44] = 200.0                  # bare paper
        self.assertGreater(float(hidden_print(detail, np.ones(6, dtype=bool), LEFT | RIGHT, 20.0)[3:].min()), 0.05)
        self.assertEqual(float(hidden_print(detail, np.ones(6, dtype=bool), LEFT | RIGHT, 20.0, light)[3:].max()), 0.0)

    def test_a_hand_is_darker_than_paper_and_still_counts(self):
        detail = self.frames()
        detail[4, 5:11, 30:37] = 5
        light = np.full(detail.shape, 150.0)
        light[:, :3, :] = 200.0                          # the top margin: bare paper somewhere in the page cells
        light[4, 5:11, 30:37] = 140.0
        cells = LEFT | RIGHT
        cells = cells.copy(); cells[1:3, 4:44] = True
        self.assertGreater(float(hidden_print(detail, np.ones(6, dtype=bool), cells, 20.0, light)[4]), 0.01)


class HiddenPrintChoiceTests(unittest.TestCase):
    EVENTS = list("....") + list("RRLL") + list("........") + list("RRLL") + list("....")

    def analysis(self, sharp: dict[int, float]):
        analysis = synthetic(self.EVENTS, detail=sharp)
        analysis.detail[:, LEFT | RIGHT] = 60
        analysis.thumbs[:] = 200                         # paper; print and hands are darker
        analysis.thumbs[:, 6:48, 8:88] = 150
        return analysis

    def hand(self, analysis, frames: slice) -> None:
        analysis.detail[frames, 5:11, 30:37] = 5
        analysis.thumbs[frames, 10:22, 60:74] = 140

    def test_a_frame_with_a_hand_no_colour_model_sees_loses_to_a_clear_one(self):
        analysis = self.analysis({10: 40.0, 13: 50.0})
        self.hand(analysis, slice(13, 16))               # from frame 13 on a pale hand lies on the right page
        spreads = group_spreads(detect_turns(analysis), analysis, None)
        self.assertEqual(spreads[1].best, 10)
        self.assertEqual(spreads[1].print_hidden, 0.0)

    def test_a_hand_in_the_only_frame_sharp_enough_is_recorded(self):
        analysis = self.analysis({10: 20.0, 12: 50.0})
        self.hand(analysis, slice(12, 16))               # the frames before it are clear but far too soft to take
        spreads = group_spreads(detect_turns(analysis), analysis, None)
        self.assertEqual(spreads[1].best, 12)
        self.assertGreater(spreads[1].print_hidden, 0.01)
        self.assertGreater(spreads[1].hand_on_print, 0.01)
        self.assertFalse(spreads[1].hand_free_exists)


class PrintMapChoiceTests(unittest.TestCase):
    """The share of the print map that is covered counts only beyond what the spread's cleanest calm frame has."""

    EVENTS = list("....") + list("RRLL") + list("........") + list("RRLL") + list("....")

    def hands(self, on_map: dict[int, float]) -> FakeHands:
        hands = FakeHands(len(self.EVENTS), {})
        hands.on_map = np.zeros(len(self.EVENTS), dtype=np.float32)
        for index, value in on_map.items():
            hands.on_map[index] = value
        return hands

    def test_a_hand_on_the_corner_that_the_print_cells_miss_is_avoided(self):
        analysis = synthetic(self.EVENTS, detail={10: 45.0, 11: 50.0})
        spreads = group_spreads(detect_turns(analysis), analysis, self.hands({11: 0.02, 12: 0.02}))
        self.assertEqual(spreads[1].best, 10)
        self.assertEqual(spreads[1].hand_on_print, 0.0)

    def test_a_few_tenths_of_a_percent_are_not_a_hand(self):
        # The map's share wavers between frames that look alike; the still, sharp frame keeps its place.
        analysis = synthetic(self.EVENTS, detail={10: 45.0, 11: 50.0})
        spreads = group_spreads(detect_turns(analysis), analysis, self.hands({11: 0.004, 12: 0.003}))
        self.assertEqual(spreads[1].best, 11)
        self.assertEqual(spreads[1].hand_on_print, 0.0)

    def test_what_is_there_in_every_frame_of_the_spread_changes_nothing(self):
        # A lit margin the map takes for print: 4% in every frame. The sharpest frame is taken, and nothing is recorded.
        analysis = synthetic(self.EVENTS, detail={10: 45.0, 11: 50.0})
        spreads = group_spreads(detect_turns(analysis), analysis, self.hands({n: 0.04 for n in range(len(self.EVENTS))}))
        self.assertEqual(spreads[1].best, 11)
        self.assertEqual(spreads[1].hand_on_print, 0.0)


class TrainedHandChoiceTests(unittest.TestCase):
    """Hands the trained models found (``hands.net``): avoided, and so is a frame right next to one."""

    EVENTS = list("....") + list("RRLL") + list("..........") + list("RRLL") + list("....")

    def hands(self, net: dict[int, float]) -> FakeHands:
        hands = FakeHands(len(self.EVENTS), {})
        hands.net = np.zeros(len(self.EVENTS), dtype=np.float32)
        for index, value in net.items():
            hands.net[index] = value
        return hands

    def test_a_frame_with_a_hand_found_by_the_models_loses_to_one_without(self):
        analysis = synthetic(self.EVENTS, detail={9: 45.0, 12: 50.0})
        spreads = group_spreads(detect_turns(analysis), analysis, self.hands({12: 0.03}))
        self.assertEqual(spreads[1].best, 9)

    def test_a_moving_frame_between_two_with_a_hand_is_no_better_than_they_are(self):
        # Frame 13 shows no hand only because the hand moving over it is a blur; frames 12 and 14 show it.
        analysis = synthetic(self.EVENTS, detail={8: 44.0, 13: 50.0})
        analysis.page_motion[13] = 0.12
        spreads = group_spreads(detect_turns(analysis), analysis, self.hands({12: 0.03, 14: 0.03}))
        self.assertEqual(spreads[1].best, 8)

    def test_a_moving_frame_with_a_hand_on_one_side_only_is_taken_for_what_it_shows(self):
        # The hand is arriving: frames 14 and 15 show it, 12 does not.
        analysis = synthetic(self.EVENTS, detail={8: 44.0, 13: 50.0})
        analysis.page_motion[13] = 0.12
        spreads = group_spreads(detect_turns(analysis), analysis, self.hands({14: 0.03, 15: 0.03}))
        self.assertEqual(spreads[1].best, 13)

    def test_a_hand_over_a_few_line_ends_counts_beyond_the_thumb_that_is_always_there(self):
        analysis = synthetic(self.EVENTS, detail={9: 45.0, 12: 50.0})
        net = {n: 0.004 for n in range(len(self.EVENTS))}       # the holding thumb on two letters, always
        net[12] = 0.007                                         # and fingers on the ends of three lines
        spreads = group_spreads(detect_turns(analysis), analysis, self.hands(net))
        self.assertEqual(spreads[1].best, 9)

    def test_a_still_frame_next_to_one_with_a_hand_is_taken_for_what_it_shows(self):
        analysis = synthetic(self.EVENTS, detail={8: 44.0, 13: 50.0})
        spreads = group_spreads(detect_turns(analysis), analysis, self.hands({12: 0.03, 14: 0.03}))
        self.assertEqual(spreads[1].best, 13)

    def test_a_turning_hand_on_the_margin_is_avoided_when_the_print_is_clear_either_way(self):
        analysis = synthetic(self.EVENTS, detail={9: 45.0, 12: 50.0})
        hands = self.hands({})
        hands.net_page = np.full(len(self.EVENTS), 0.02, dtype=np.float32)     # the holding hand, always
        hands.net_page[12] = 0.05                                               # and the other hand on the corner
        spreads = group_spreads(detect_turns(analysis), analysis, hands)
        self.assertEqual(spreads[1].best, 9)

    def test_a_hand_lifting_the_corner_beside_the_print_costs_more_than_a_thumb_on_its_last_line(self):
        # Frame 9: the thumb that holds the book on the last line of the left page (on the print).
        # Frame 12: no thumb there, but the other hand lifting the right page's corner just beside its print.
        analysis = synthetic(self.EVENTS, detail={9: 45.0, 12: 50.0})
        hands = self.hands({9: 0.005})
        hands.net_near = hands.net.copy()
        hands.net_near[12] = 0.02
        spreads = group_spreads(detect_turns(analysis), analysis, hands)
        self.assertEqual(spreads[1].best, 9)
        self.assertAlmostEqual(spreads[1].hand_on_print, 0.005, places=4)    # the flag still counts only what is on the print

    def test_with_a_hand_on_each_page_in_turn_each_page_comes_from_its_own_frame(self):
        # The turning hand on the right page early in the spread, the holding hand on the left page late.
        import auto_camera.spreads as spreads_module
        self.addCleanup(setattr, spreads_module, "TWO_FRAMES", spreads_module.TWO_FRAMES)
        spreads_module.TWO_FRAMES = True
        analysis = synthetic(self.EVENTS, detail={n: 50.0 for n in range(8, 18)})
        hands = self.hands({n: 0.02 for n in range(8, 18)})
        hands.net_left = np.zeros(len(self.EVENTS), dtype=np.float32)
        hands.net_right = np.zeros(len(self.EVENTS), dtype=np.float32)
        hands.net_right[8:12] = 0.04
        hands.net_left[12:18] = 0.04
        spread = group_spreads(detect_turns(analysis), analysis, hands)[1]
        self.assertTrue(12 <= spread.right_frame <= 17, spread.right_frame)
        self.assertTrue(8 <= spread.left_frame <= 11, spread.left_frame)
        self.assertLess(spread.hand_on_print, 0.02)

    def test_one_frame_clear_on_both_pages_needs_no_second(self):
        analysis = synthetic(self.EVENTS, detail={n: 50.0 for n in range(8, 18)})
        hands = self.hands({12: 0.03})
        hands.net_left = np.zeros(len(self.EVENTS), dtype=np.float32)
        hands.net_right = np.zeros(len(self.EVENTS), dtype=np.float32)
        hands.net_right[12] = 0.03
        spread = group_spreads(detect_turns(analysis), analysis, hands)[1]
        self.assertEqual((spread.left_frame, spread.right_frame), (spread.best, spread.best))
        self.assertEqual(spread.as_dict(10.0)["left_frame"], spread.best)

    def test_the_flag_records_what_the_chosen_frame_itself_shows(self):
        analysis = synthetic(self.EVENTS, detail={n: 50.0 for n in range(8, 18)})
        spreads = group_spreads(detect_turns(analysis), analysis, self.hands({n: 0.02 for n in range(8, 18)}))
        self.assertAlmostEqual(spreads[1].hand_on_print, 0.02, places=4)
        self.assertFalse(spreads[1].hand_free_exists)


class CaptureRankTests(unittest.TestCase):
    """Between captures of one spread: still before mid-turn, then less hand on the page, then sharpness."""

    def test_a_still_capture_with_a_thumb_on_two_letters_beats_a_mid_turn_one_without(self):
        from auto_camera.spreads import capture_rank
        still = {"hand_on_print": 0.004, "hand_on_page": 0.03, "best_motion": 0.05, "detail": 50.0}
        moving = {"hand_on_print": 0.0, "hand_on_page": 0.03, "best_motion": 0.45, "detail": 52.0}
        self.assertGreater(capture_rank(still), capture_rank(moving))

    def test_a_hand_across_the_print_loses_even_to_a_capture_that_never_came_to_rest(self):
        from auto_camera.spreads import capture_rank
        still = {"hand_on_print": 0.06, "hand_on_page": 0.10, "best_motion": 0.05, "detail": 50.0}
        moving = {"hand_on_print": 0.0, "hand_on_page": 0.03, "best_motion": 0.20, "detail": 45.0}
        self.assertGreater(capture_rank(moving), capture_rank(still))

    def test_with_the_print_clear_either_way_the_sharper_still_capture_wins(self):
        from auto_camera.spreads import capture_rank
        a = {"hand_on_print": 0.0, "hand_on_page": 0.06, "best_motion": 0.03, "detail": 55.0}
        b = {"hand_on_print": 0.0, "hand_on_page": 0.02, "best_motion": 0.08, "detail": 50.0}
        self.assertGreater(capture_rank(a), capture_rank(b))
