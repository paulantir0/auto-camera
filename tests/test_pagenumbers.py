import unittest

from auto_camera.pagenumbers import audit_sequence, decode_pages, no_room_between, repeats_by_number, sequence_report, spread_page


class SpreadPageTests(unittest.TestCase):
    def test_both_numbers_of_a_spread_map_to_the_even_one(self):
        self.assertEqual(spread_page(8), 8)
        self.assertEqual(spread_page(9), 8)


class DecodePagesTests(unittest.TestCase):
    def test_picks_the_number_that_advances_by_two(self):
        # running heads contribute stray digits; only the page numbers form a chain
        candidates = [[144, 7], [146, 15], [19, 149], [36], [152, 44], [154]]
        self.assertEqual(decode_pages(candidates), [144, 146, 148, None, 152, 154])

    def test_uses_either_page_of_the_spread(self):
        self.assertEqual(decode_pages([[8], [11], [12, 13], [15]]), [8, 10, 12, 14])

    def test_keeps_a_chain_across_a_skip(self):
        self.assertEqual(decode_pages([[4], [6], [12], [14]]), [4, 6, 12, 14])

    def test_keeps_a_repeat(self):
        self.assertEqual(decode_pages([[4], [6], [6], [8]]), [4, 6, 6, 8])

    def test_no_candidates(self):
        self.assertEqual(decode_pages([[], []]), [None, None])

    def test_chain_survives_several_unnumbered_frames_between_consecutive_pages(self):
        # four junk frames with no readable number sit between page 50 and page 52
        candidates = [[46], [48], [50], [], [], [], [], [52], [54], [56]]
        self.assertEqual(decode_pages(candidates), [46, 48, 50, None, None, None, None, 52, 54, 56])

    def test_facing_page_translation_rises_by_one(self):
        self.assertEqual(decode_pages([[21], [22, 69], [23], [24, 70]], step=1), [21, 22, 23, 24])


class AuditSequenceTests(unittest.TestCase):
    def test_ordinary_book_is_recognised(self):
        report = audit_sequence([[4, 5], [6], [8, 9], [11], [12]])
        self.assertEqual((report.step, report.missing, report.extra), (2, 0, 0))

    def test_facing_page_translation_is_recognised(self):
        report = audit_sequence([[21, 21], [22], [23, 23], [24], [25], [26]])
        self.assertEqual((report.step, report.missing, report.extra), (1, 0, 0))

    def test_skip_in_a_facing_page_translation(self):
        report = audit_sequence([[21], [22], [23], [25], [26], [27]])
        self.assertEqual((report.step, report.missing), (1, 1))


class SequenceReportTests(unittest.TestCase):
    def test_complete_sequence_has_no_anomalies(self):
        report = sequence_report([4, 6, 8, 10, 12])
        self.assertEqual((report.missing, report.extra, report.unclear), (0, 0, 0))
        self.assertEqual((report.span_captures, report.span_expected), (5, 5))

    def test_unread_captures_are_bridged(self):
        report = sequence_report([4, None, None, 10, 12])
        self.assertEqual((report.missing, report.extra), (0, 0))
        self.assertEqual(report.read, 3)

    def test_skipped_spread_is_counted(self):
        report = sequence_report([4, 6, 10, 12])
        self.assertEqual((report.missing, report.extra), (1, 0))
        self.assertEqual((report.anomalies[0].after, report.anomalies[0].before), (1, 2))

    def test_two_skipped_spreads_across_an_unread_capture(self):
        report = sequence_report([4, None, 12])
        self.assertEqual(report.missing, 2)

    def test_repeated_spread_is_counted_as_extra(self):
        report = sequence_report([4, 6, 6, 8])
        self.assertEqual((report.missing, report.extra), (0, 1))

    def test_lone_misread_is_discarded(self):
        report = sequence_report([4, 6, 88, 10, 12])
        self.assertEqual((report.missing, report.extra), (0, 0))
        self.assertEqual(report.discarded_misreads, [2])

    def test_real_skip_is_not_mistaken_for_a_misread(self):
        report = sequence_report([4, 6, 12, 14, 16])
        self.assertEqual(report.missing, 2)
        self.assertEqual(report.discarded_misreads, [])

    def test_odd_difference_is_unclear(self):
        report = sequence_report([4, 6, 9, 11])
        self.assertEqual(report.unclear, 1)

    def test_nothing_read(self):
        report = sequence_report([None, None])
        self.assertEqual(report.read, 0)
        self.assertIsNone(report.first_page)


class RepeatsByNumberTests(unittest.TestCase):
    @staticmethod
    def prefer_later(i, j):
        return i > j

    @staticmethod
    def prefer_earlier(i, j):
        return i < j

    def test_keeps_the_better_of_neighbouring_repeats(self):
        self.assertEqual(repeats_by_number([4, 6, 6, 6, 8], 2, self.prefer_later), [1, 2])
        self.assertEqual(repeats_by_number([4, 6, 6, 6, 8], 2, self.prefer_earlier), [2, 3])

    def test_unread_captures_are_never_dropped(self):
        self.assertEqual(repeats_by_number([4, None, None, 6], 2, self.prefer_later), [])

    def test_repeat_across_an_unread_capture(self):
        self.assertEqual(repeats_by_number([4, 6, None, 6, 8], 2, self.prefer_earlier), [3])

    def test_going_back_keeps_the_earlier_capture(self):
        # 4, 6, 4 again, 8: the second 4 is a revisit, even if it is the better image
        self.assertEqual(repeats_by_number([4, 6, 4, 8], 2, self.prefer_later), [2])

    def test_pair_followed_by_a_gap_is_left_alone(self):
        # 6, 6, 10: one of the sixes may really be 8 with a misread number
        self.assertEqual(repeats_by_number([4, 6, 6, 10], 2, self.prefer_later), [])

    def test_pair_after_a_gap_is_left_alone(self):
        self.assertEqual(repeats_by_number([4, 8, 8, 10], 2, self.prefer_later), [])

    def test_step_one_books(self):
        self.assertEqual(repeats_by_number([21, 22, 22, 23], 1, self.prefer_later), [1])

    def test_early_glimpse_of_a_later_spread_is_dropped(self):
        # 140, a glimpse of 144, back to 142, then 144 in its place
        self.assertEqual(repeats_by_number([138, 140, 144, 142, 144, 146], 2, self.prefer_earlier), [2])

    def test_skip_without_going_back_is_kept(self):
        self.assertEqual(repeats_by_number([138, 140, 144, 146], 2, self.prefer_earlier), [])

    def test_far_apart_equal_numbers_are_not_repeats(self):
        self.assertEqual(repeats_by_number([4, 6, 8, 10, 12, 14, 4], 2, self.prefer_later), [])


if __name__ == "__main__":
    unittest.main()


class RestartTests(unittest.TestCase):
    """A volume of reprinted essays: the numbering starts afresh with each one."""

    def essays(self):
        first = [[p, p + 1] for p in range(280, 300, 2)]       # 10 spreads, 280..298
        second = [[p, p + 1] for p in range(2, 22, 2)]         # 10 spreads, 2..20
        third = [[p, p + 1] for p in range(162, 178, 2)]       # 8 spreads, 162..176
        return first + second + third

    def test_each_run_is_decoded(self):
        pages = decode_pages(self.essays())
        self.assertEqual(pages, list(range(280, 300, 2)) + list(range(2, 22, 2)) + list(range(162, 178, 2)))

    def test_restarts_are_listed_and_not_counted_as_missing_or_extra(self):
        report = sequence_report(decode_pages(self.essays()))
        self.assertEqual((report.missing, report.extra, report.unclear), (0, 0, 0))
        self.assertEqual([(r.page_after, r.page_before) for r in report.restarts], [(298, 2), (20, 162)])
        self.assertEqual(report.span_expected, 28)

    def test_a_gap_inside_a_run_is_still_missing(self):
        candidates = self.essays()
        del candidates[14]                                     # the spread 10/11 of the second essay
        report = sequence_report(decode_pages(candidates))
        self.assertEqual(report.missing, 1)
        self.assertEqual(len(report.restarts), 2)

    def test_two_stray_numbers_do_not_start_a_run(self):
        candidates = [[p] for p in range(100, 140, 2)]
        candidates[8] = [116, 502]
        candidates[9] = [118, 504]
        self.assertEqual(decode_pages(candidates), list(range(100, 140, 2)))

    def test_flipping_back_does_not_inflate_the_expected_count(self):
        report = sequence_report([100, 104, 102, 104, 106])
        self.assertEqual(report.span_expected, 4)
        self.assertEqual(report.restarts, [])


class NoRoomBetweenTests(unittest.TestCase):
    def test_a_flagged_capture_between_consecutive_spreads_goes(self):
        self.assertEqual(no_room_between([10, None, 12, 14], 2, [False, True, False, False]), [1])

    def test_an_unflagged_one_stays(self):
        # a clean, settled capture with no number: a plate or an inserted leaf
        self.assertEqual(no_room_between([10, None, 12, 14], 2, [False, False, False, False]), [])

    def test_nothing_goes_where_a_spread_is_absent(self):
        # 10 then 14: the capture between them may be spread 12
        self.assertEqual(no_room_between([10, None, 14], 2, [False, True, False]), [])

    def test_several_flagged_captures_in_one_place(self):
        self.assertEqual(no_room_between([10, None, None, 12], 2, [False, True, True, False]), [1, 2])

    def test_between_two_views_of_the_same_spread(self):
        self.assertEqual(no_room_between([10, None, 10, 12], 2, [False, True, False, False]), [1])

    def test_not_across_a_backward_step(self):
        self.assertEqual(no_room_between([14, None, 10], 2, [False, True, False]), [])


class RepeatNeedsTheSameNumberTests(unittest.TestCase):
    def test_a_repeat_is_dropped_when_confirmed(self):
        self.assertEqual(repeats_by_number([10, 12, 12, 14], 2, lambda i, j: False, lambda i, j: True), [2])

    def test_and_kept_when_the_two_do_not_show_the_same_number(self):
        # the second 12 was decoded from a stray digit; the capture is another spread
        self.assertEqual(repeats_by_number([10, 12, 12, 14], 2, lambda i, j: False, lambda i, j: False), [])
