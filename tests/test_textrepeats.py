import random
import unittest

from auto_camera.textrepeats import repeated_pages


def book(pages: int = 40, words: int = 260, seed: int = 5) -> list[str]:
    """Pages of made-up prose: every page different, a running head on each."""
    rng = random.Random(seed)
    vocabulary = ["".join(rng.choice("abcdefghiklmnoprstuvwy") for _ in range(rng.randint(3, 9))) for _ in range(900)]
    return [f"THE MADE UP BOOK {k + 1}\n" + " ".join(rng.choice(vocabulary) for _ in range(words)) for k in range(pages)]


def garble(text: str, share: float, seed: int = 9) -> str:
    """The text as OCR reads a page with a hand over part of it: a stretch of words misread, most of them."""
    rng = random.Random(seed)
    words = text.split()
    start = rng.randrange(0, int(len(words) * (1 - share)))
    stop = start + int(len(words) * share)
    return " ".join("".join(rng.choice("xqzj") for _ in w) if start <= k < stop and rng.random() < 0.8 else w
                    for k, w in enumerate(words))


class TextRepeatTests(unittest.TestCase):
    def test_different_pages_are_all_kept(self):
        self.assertEqual(repeated_pages(book()), [])

    def test_a_garbled_second_view_is_left_out_and_the_clean_one_kept(self):
        pages = book()
        pages.insert(13, garble(pages[11], 0.35))     # a glimpse of p.12 during the next turn
        found = repeated_pages(pages)
        self.assertEqual([(r["page"], r["kept"]) for r in found], [(13, 11)])

    def test_the_clean_view_is_kept_when_it_comes_second(self):
        pages = book()
        pages[11] = garble(pages[13], 0.35)
        found = repeated_pages(pages)
        self.assertEqual([(r["page"], r["kept"]) for r in found], [(11, 13)])

    def test_the_two_pages_of_one_picture_are_never_the_same_page(self):
        pages = book()
        pages[21] = pages[20]
        pictures = [k // 2 for k in range(len(pages))]
        self.assertEqual(repeated_pages(pages, pictures), [])
        self.assertEqual([r["page"] for r in repeated_pages(pages)], [21])

    def test_a_page_seen_again_far_away_is_kept(self):
        pages = book()
        pages[30] = pages[10]
        self.assertEqual(repeated_pages(pages), [])

    def test_short_pages_are_not_compared(self):
        pages = book()
        pages[1] = "The Made Up Book. Edited by Somebody Else, with an introduction by Another One"
        pages[2] = "The Made Up Book. Edited by Somebody Else, with an introduction by Another One. Copyright 2026"
        self.assertEqual(repeated_pages(pages), [])

    def test_three_views_of_a_page_keep_the_most_legible(self):
        pages = book()
        pages.insert(16, garble(pages[14], 0.3, seed=1))
        pages.insert(18, garble(pages[14], 0.15, seed=2))
        found = repeated_pages(pages)
        self.assertEqual(sorted((r["page"], r["kept"]) for r in found), [(16, 14), (18, 14)])


    def test_a_view_missing_its_first_lines_gives_way_to_the_whole_page(self):
        pages = book()
        words = pages[11].split()
        pages.insert(13, " ".join(words[len(words) // 4:]))
        self.assertEqual([(r["page"], r["kept"]) for r in repeated_pages(pages)], [(13, 11)])

    def test_when_neither_view_reads_better_on_both_counts_both_stay(self):
        pages = book()
        # a view of p.12 during the turn: its text and half the next page's, a stretch of it misread
        pages.insert(13, garble(pages[11] + " " + " ".join(pages[12].split()[:150]), 0.25, seed=4))
        self.assertEqual(repeated_pages(pages), [])

    def test_a_better_read_from_a_flagged_picture_does_not_replace_an_unflagged_one(self):
        pages = book()
        words = pages[11].split()
        pages.insert(13, " ".join(words[len(words) // 4:]))     # p.12 again, its first lines hidden
        flags = [0] * len(pages)
        flags[11] = 2                                            # the whole view was taken with a hand on the print
        self.assertEqual(repeated_pages(pages, flags=flags), [])
        flags[13] = 2
        self.assertEqual([(r["page"], r["kept"]) for r in repeated_pages(pages, flags=flags)], [(13, 11)])


if __name__ == "__main__":
    unittest.main()
