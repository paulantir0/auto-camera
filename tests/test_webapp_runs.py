"""A run folder turned into what the browser app shows. No video needed."""

import http.client
import io
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

from webapp import Config, runs
from webapp.paths import Refused
from webapp.server import App, Server


NO_PAGE = object()      # a capture written before the pipeline recorded page numbers has no "page" key at all


def capture(number: int, seconds: float, review: list[str] | None = None, page=NO_PAGE) -> dict:
    made = {"index": number, "file": f"sequence_{number:04d}_t{seconds:07.2f}s.jpg", "best_seconds": seconds,
            "review": review or [], "hand_on_print": 0.02 if review and "hand_on_print" in review else 0.0}
    if page is not NO_PAGE:
        made["page"] = page
    return made


def final(captures: int, read: int, first, last, anomalies: list[dict] | None = None, step: int = 2,
          restarts: list[dict] | None = None) -> dict:
    """The ``final`` part of audit.json. Without ``restarts`` the key is left out, as in runs made before it existed.

    As in the pipeline, what lies across a restart is counted neither as missing nor as extra.
    """
    anomalies = anomalies or []
    made = {"captures": captures, "read": read, "step": step, "discarded_misreads": [], "first_page": first, "last_page": last,
            "missing": sum(a["missing"] for a in anomalies), "extra": sum(a["extra"] for a in anomalies), "unclear": 0,
            "anomalies": anomalies}
    if restarts is not None:
        made["restarts"] = restarts
    return made


def restart(captures: list[dict], after: int, before: int, page_after: int, page_before: int, missing: int = 0, extra: int = 0) -> dict:
    """A place where the numbering starts afresh, shaped as audit.json has it."""
    return {**anomaly(captures, after, before, page_after, page_before, missing, extra), "seconds_between": 3.4}


def anomaly(captures: list[dict], after: int, before: int, page_after: int, page_before: int, missing: int = 0, extra: int = 0) -> dict:
    """A break between the captures at positions ``after`` and ``before`` (counted from 0, as the audit does)."""
    return {"after": after, "before": before, "page_after": page_after, "page_before": page_before, "missing": missing,
            "extra": extra, "unclear": False, "after_file": captures[after]["file"], "before_file": captures[before]["file"]}


def write_video(root: Path, run: str, slug: str, captures: list[dict], *, audit_final: dict | None = None,
                result: dict | None = None, video: str = "/home/ubuntu/videos/A Book IMG-0001.M4V", files: bool = True,
                more: dict | None = None) -> Path:
    """One video's folder in a run. ``more`` adds further keys to spreads.json."""
    folder = root / run / slug
    (folder / "captures").mkdir(parents=True, exist_ok=True)
    if files:
        for item in captures:
            (folder / "captures" / item["file"]).write_bytes(b"jpeg " + item["file"].encode())
    (folder / "spreads.json").write_text(json.dumps({"slug": slug, "video": video, "fps": 11.0, "frames": 2751,
                                                      "candidates": [], "captures": captures, **(more or {})}))
    if audit_final is not None:
        (folder / "audit.json").write_text(json.dumps({"slug": slug, "final": audit_final, "captures": len(captures)}))
    if result is not None:
        (folder / "result.json").write_text(json.dumps(result))
    return folder


class RunFolderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def numbered(self, count: int) -> list[dict]:
        return [capture(n, round(2.2 * n, 2)) for n in range(1, count + 1)]


class VerdictTests(RunFolderCase):
    def test_every_page_there(self):
        captures = self.numbered(118)
        write_video(self.root, "run_20261111_corpus", "einar", captures, audit_final=final(118, 108, 10, 232))
        view = runs.video_view(self.root, "run_20261111_corpus", "einar")
        self.assertEqual(view["verdict"]["state"], "complete")
        self.assertEqual(view["verdict"]["headline"], "Every page number from 10 to 232 is there.")
        self.assertEqual(view["counts"]["missing"], 0)
        self.assertEqual(view["counts"]["pictures"], 118)
        self.assertEqual([item["kind"] for item in view["items"]], ["picture"] * 118)

    def test_one_gap_is_named_and_marked_between_its_two_pictures(self):
        captures = self.numbered(120)
        gap = anomaly(captures, 97, 98, 176, 180, missing=1)
        write_video(self.root, "run_x", "img-0052", captures, audit_final=final(120, 110, 10, 250, [gap]))
        view = runs.video_view(self.root, "run_x", "img-0052")
        self.assertEqual(view["verdict"]["state"], "missing")
        self.assertEqual(view["verdict"]["headline"], "1 spread missing after page 176")
        self.assertEqual(view["verdict"]["short"], "1 spread missing")
        kinds = [item["kind"] for item in view["items"]]
        at = kinds.index("gap")
        self.assertEqual(kinds.count("gap"), 1)
        self.assertEqual(view["items"][at - 1]["file"], captures[97]["file"])
        self.assertEqual(view["items"][at + 1]["file"], captures[98]["file"])
        mark = view["items"][at]
        self.assertEqual(mark["title"], "Missing: pages 178 and 179")
        self.assertIn("176", mark["text"])
        self.assertIn("180", mark["text"])
        self.assertTrue(view["items"][at - 1]["beside_mark"] and view["items"][at + 1]["beside_mark"])
        self.assertFalse(view["items"][0]["beside_mark"])
        self.assertEqual(view["counts"]["gaps"], 1)

    def test_several_gaps_are_each_named(self):
        captures = self.numbered(200)
        gaps = [anomaly(captures, 24, 25, 26, 30, missing=1), anomaly(captures, 60, 61, 100, 104, missing=1),
                anomaly(captures, 97, 98, 176, 180, missing=1)]
        write_video(self.root, "run_x", "book", captures, audit_final=final(200, 190, 10, 420, gaps))
        said = runs.video_view(self.root, "run_x", "book")["verdict"]
        self.assertEqual(said["headline"], "3 spreads missing: after page 26, after page 100 and after page 176")

    def test_many_gaps_are_counted_not_listed(self):
        captures = self.numbered(200)
        gaps = [anomaly(captures, n, n + 1, 10 * n, 10 * n + 4, missing=1) for n in (5, 15, 25, 35, 45)]
        write_video(self.root, "run_x", "book", captures, audit_final=final(200, 190, 10, 420, gaps))
        said = runs.video_view(self.root, "run_x", "book")["verdict"]
        self.assertEqual(said["headline"], "5 spreads missing in 5 places")

    def test_a_gap_across_unread_pictures_says_where_it_lies(self):
        captures = self.numbered(60)
        gap = anomaly(captures, 10, 13, 30, 38, missing=1)
        write_video(self.root, "run_x", "book", captures, audit_final=final(60, 50, 10, 130, [gap]))
        view = runs.video_view(self.root, "run_x", "book")
        mark = next(item for item in view["items"] if item["kind"] == "gap")
        self.assertEqual(mark["title"], "1 spread missing")
        self.assertIn("between picture 11 (page 30) and picture 14 (page 38)", mark["text"])
        self.assertEqual([i["number"] for i in view["items"] if i["kind"] == "picture" and i["beside_mark"]], [11, 12, 13, 14])

    def test_a_video_the_check_cannot_vouch_for(self):
        captures = self.numbered(240)
        write_video(self.root, "run_x", "img-0042", captures, audit_final=final(240, 2, 212, 214))
        view = runs.video_view(self.root, "run_x", "img-0042")
        self.assertEqual(view["verdict"]["state"], "unreadable")
        self.assertEqual(view["verdict"]["headline"], "Page numbers could not be read on this book, so the order is not checked.")
        self.assertIn("only 2 of 240 pictures", view["verdict"]["details"][0])

    def test_no_numbers_read_at_all(self):
        write_video(self.root, "run_x", "closed-book", self.numbered(12), audit_final=final(12, 0, None, None))
        said = runs.video_view(self.root, "run_x", "closed-book")["verdict"]
        self.assertEqual(said["state"], "unreadable")
        self.assertIn("No page number could be read", said["details"][0])

    def test_a_possible_gap_in_a_video_that_cannot_be_vouched_for_is_still_marked(self):
        captures = self.numbered(100)
        gap = anomaly(captures, 20, 21, 50, 54, missing=1)
        write_video(self.root, "run_x", "book", captures, audit_final=final(100, 35, 10, 90, [gap]))
        view = runs.video_view(self.root, "run_x", "book")
        self.assertEqual(view["verdict"]["state"], "unreadable")
        self.assertEqual(sum(item["kind"] == "gap" for item in view["items"]), 1)
        self.assertIn("1 possible gap is marked below", " ".join(view["verdict"]["details"]))

    def test_an_older_run_without_a_check(self):
        write_video(self.root, "run_20261111_fast_v0", "book", self.numbered(20))
        view = runs.video_view(self.root, "run_20261111_fast_v0", "book")
        self.assertEqual(view["verdict"]["state"], "not_checked")
        self.assertIsNone(view["counts"]["missing"])

    def test_repeats_are_marked_but_do_not_make_a_page_missing(self):
        captures = self.numbered(100)
        extra = anomaly(captures, 40, 42, 90, 92, extra=1)
        write_video(self.root, "run_x", "book", captures, audit_final=final(100, 90, 10, 206, [extra]))
        view = runs.video_view(self.root, "run_x", "book")
        self.assertEqual(view["verdict"]["state"], "complete")
        self.assertIn("1 picture may show a spread a second time", " ".join(view["verdict"]["details"]))
        mark = next(item for item in view["items"] if item["kind"] == "repeat")
        self.assertEqual(mark["text"], "Pictures 41 to 43 are 3 pictures for 2 spreads (pages 90 to 92), so one is probably a repeat.")

    def test_two_pictures_of_the_same_page(self):
        captures = self.numbered(100)
        extra = anomaly(captures, 40, 41, 90, 90, extra=1)
        write_video(self.root, "run_x", "book", captures, audit_final=final(100, 90, 10, 206, [extra]))
        mark = next(item for item in runs.video_view(self.root, "run_x", "book")["items"] if item["kind"] == "repeat")
        self.assertEqual(mark["text"], "Pictures 41 and 42 both show page 90, so one is probably a repeat.")

    def test_the_first_spread_is_called_page_one(self):
        captures = self.numbered(100)
        write_video(self.root, "run_x", "book", captures, audit_final=final(100, 90, 0, 198))
        self.assertEqual(runs.video_view(self.root, "run_x", "book")["verdict"]["headline"], "Every page number from 1 to 198 is there.")

    def test_no_pictures(self):
        write_video(self.root, "run_x", "book", [])
        self.assertEqual(runs.video_view(self.root, "run_x", "book")["verdict"]["state"], "empty")


class WhoVouchesTests(RunFolderCase):
    def test_the_rule_matches_the_pipelines(self):
        from auto_camera.audit import auditable
        for read, captures in ((108, 118), (30, 50), (29, 29), (30, 51), (64, 122), (76, 216), (0, 12), (300, 500), (299, 500)):
            with self.subTest(read=read, captures=captures):
                audit = {"final": {"read": read}, "captures": captures}
                self.assertEqual(runs.can_vouch(audit), auditable(audit))

    def test_the_pipelines_own_word_is_used_when_a_result_is_saved(self):
        captures = self.numbered(118)
        result = {"slug": "book", "captures": 118, "audit": {"vouched": False}, "audit_note": ""}
        write_video(self.root, "run_x", "book", captures, audit_final=final(118, 108, 10, 232), result=result)
        self.assertEqual(runs.video_view(self.root, "run_x", "book")["verdict"]["state"], "unreadable")

    def test_a_result_saying_the_reader_is_not_installed(self):
        result = {"slug": "book", "captures": 20, "audit": None,
                  "audit_note": "page numbers not checked: the OCR environment (.venv-ocr) is not installed"}
        write_video(self.root, "run_x", "book", self.numbered(20), result=result)
        said = runs.video_view(self.root, "run_x", "book")["verdict"]
        self.assertEqual(said["state"], "not_checked")
        self.assertEqual(said["details"], ["The part that reads page numbers is not installed on this computer."])

    def test_a_check_made_for_another_list_of_pictures_is_not_believed(self):
        folder = write_video(self.root, "run_x", "book", self.numbered(118), audit_final=final(118, 108, 10, 232))
        spreads = json.loads((folder / "spreads.json").read_text())
        spreads["captures"] = spreads["captures"][:100]
        (folder / "spreads.json").write_text(json.dumps(spreads))
        said = runs.video_view(self.root, "run_x", "book")["verdict"]
        self.assertEqual(said["state"], "not_checked")

    def test_a_scan_that_was_cut_short_says_so(self):
        folder = write_video(self.root, "run_x", "book", self.numbered(118), audit_final=final(118, 108, 10, 232))
        (folder / runs.JOB_MARKER).write_text(json.dumps({"status": "cancelled"}))
        view = runs.video_view(self.root, "run_x", "book")
        self.assertEqual(view["verdict"]["state"], "unfinished")
        (folder / runs.JOB_MARKER).write_text(json.dumps({"status": "running"}))
        self.assertEqual(runs.video_view(self.root, "run_x", "book", live=True)["verdict"]["state"], "running")
        self.assertEqual(runs.video_view(self.root, "run_x", "book", live=False)["verdict"]["state"], "unfinished")
        (folder / runs.JOB_MARKER).write_text(json.dumps({"status": "done"}))
        self.assertEqual(runs.video_view(self.root, "run_x", "book")["verdict"]["state"], "complete")


class PictureListTests(RunFolderCase):
    def test_flags_are_put_in_plain_words_and_counted(self):
        captures = [capture(1, 0.7), capture(2, 5.3, ["hand_on_print"]), capture(3, 10.1, ["never_at_rest"]),
                    capture(4, 12.0, ["hand_on_print", "never_at_rest"]), capture(5, 15.0, ["something_new"])]
        write_video(self.root, "run_x", "book", captures)
        view = runs.video_view(self.root, "run_x", "book")
        labels = [item["labels"] for item in view["items"]]
        self.assertEqual(labels, [[], ["hand over the text"], ["page was moving"], ["hand over the text", "page was moving"], ["something new"]])
        self.assertEqual((view["counts"]["needs_look"], view["counts"]["hand"], view["counts"]["moving"]), (4, 2, 2))
        self.assertEqual([item["number"] for item in view["items"]], [1, 2, 3, 4, 5])
        self.assertEqual(view["items"][1]["seconds"], 5.3)

    def test_the_title_is_the_video_name_without_its_folder_or_ending(self):
        write_video(self.root, "run_x", "book", self.numbered(3), video="/home/ubuntu/videos/Hagner, Donald The Use IMG-0017.M4V")
        self.assertEqual(runs.video_view(self.root, "run_x", "book")["title"], "Hagner, Donald The Use IMG-0017")

    def test_names_from_the_files_are_passed_on_as_plain_data(self):
        write_video(self.root, "run_x", "book", self.numbered(2), video="/v/<img src=x onerror=alert(1)>.mp4")
        self.assertEqual(runs.video_view(self.root, "run_x", "book")["title"], "<img src=x onerror=alert(1)>")

    def test_a_listed_file_with_an_unsafe_name_is_not_offered(self):
        captures = self.numbered(2) + [{"index": 3, "file": "../../../etc/passwd", "best_seconds": 9.0, "review": []}]
        write_video(self.root, "run_x", "book", captures, files=False)
        view = runs.video_view(self.root, "run_x", "book")
        self.assertIsNone(view["items"][2]["file"])
        self.assertFalse(view["items"][2]["on_disk"])
        self.assertEqual(view["counts"]["files_missing"], 3)

    def test_a_folder_without_results_is_not_a_video(self):
        (self.root / "run_x" / "_audit_old").mkdir(parents=True)
        (self.root / "run_x" / "empty").mkdir()
        with self.assertRaises(Refused):
            runs.video_view(self.root, "run_x", "empty")
        self.assertEqual(runs.run_view(self.root, "run_x")["videos"], [])

    def test_broken_json_does_not_break_the_page(self):
        folder = write_video(self.root, "run_x", "book", self.numbered(3), audit_final=final(3, 0, None, None))
        (folder / "audit.json").write_text("{ not json")
        self.assertEqual(runs.video_view(self.root, "run_x", "book")["verdict"]["state"], "not_checked")
        (folder / "spreads.json").write_text("[1, 2, 3]")
        with self.assertRaises(Refused):
            runs.video_view(self.root, "run_x", "book")


class RunListTests(RunFolderCase):
    def test_run_names_in_words(self):
        self.assertEqual(runs.run_label("run_20261111_corpus"), "11 November 2026 · corpus")
        self.assertEqual(runs.run_label("run_20261111_hands_v1"), "11 November 2026 · hands v1")
        self.assertEqual(runs.run_label("run_20261111_web"), "11 November 2026 · scanned from this page")
        self.assertEqual(runs.run_label("run_20261341_x"), "run_20261341_x")
        self.assertEqual(runs.run_label("something-else"), "something-else")

    def test_a_run_lists_its_videos_with_totals(self):
        captures = self.numbered(120)
        write_video(self.root, "run_20261111_corpus", "b-book", self.numbered(118), audit_final=final(118, 108, 10, 232), video="/v/B book.mp4")
        write_video(self.root, "run_20261111_corpus", "a-book", captures,
                    audit_final=final(120, 110, 10, 250, [anomaly(captures, 97, 98, 176, 180, missing=1)]), video="/v/A book.mp4")
        write_video(self.root, "run_20261111_corpus", "c-book", self.numbered(40), audit_final=final(40, 2, 12, 14), video="/v/C book.mp4")
        (self.root / "run_20261111_corpus" / "summary_dev.json").write_text("{}")
        view = runs.run_view(self.root, "run_20261111_corpus")
        self.assertEqual([v["title"] for v in view["videos"]], ["A book", "B book", "C book"])
        self.assertEqual([v["verdict"]["state"] for v in view["videos"]], ["missing", "complete", "unreadable"])
        self.assertEqual(view["totals"], {"videos": 3, "pictures": 278, "states": {"missing": 1, "complete": 1, "unreadable": 1}})
        self.assertNotIn("items", view["videos"][0])

    def test_runs_are_listed_newest_first_and_empty_ones_left_out(self):
        write_video(self.root, "run_20261109_old", "book", self.numbered(3))
        write_video(self.root, "run_20261111_web", "book", self.numbered(4))
        (self.root / "run_20261112_empty").mkdir()
        (self.root / ".hidden").mkdir()
        self.assertEqual([r["run"] for r in runs.runs_view(self.root)], ["run_20261111_web", "run_20261109_old"])
        latest = runs.latest_result(self.root, "book")
        self.assertEqual((latest["run"], latest["state"]), ("run_20261111_web", "not_checked"))
        self.assertIsNone(runs.latest_result(self.root, "another-book"))

    def test_a_summary_follows_a_changed_file(self):
        folder = write_video(self.root, "run_x", "book", self.numbered(118), audit_final=final(118, 108, 10, 232))
        self.assertEqual(runs.video_summary(self.root, "run_x", "book")["verdict"]["state"], "complete")
        (folder / runs.JOB_MARKER).write_text(json.dumps({"status": "failed"}))
        self.assertEqual(runs.video_summary(self.root, "run_x", "book")["verdict"]["state"], "unfinished")


class NumberingStartsAgainTests(RunFolderCase):
    """``final.restarts``: the printed numbering starts afresh. Not a missing page, not a repeat."""

    def collection(self, places: int, count: int = 200) -> tuple[list[dict], list[dict]]:
        captures = self.numbered(count)
        restarts = [restart(captures, 10 * n, 10 * n + 1, 302 + 2 * n, 280 + 2 * n, extra=12) for n in range(1, places + 1)]
        return captures, restarts

    def test_the_verdict_says_no_page_is_missing_and_how_often_the_numbering_starts_again(self):
        captures, restarts = self.collection(12)
        write_video(self.root, "run_x", "essays", captures, audit_final=final(200, 180, 1, 281, restarts=restarts))
        view = runs.video_view(self.root, "run_x", "essays")
        said = view["verdict"]
        self.assertEqual(said["state"], "complete")
        self.assertEqual(said["headline"], "No page is missing.")
        self.assertEqual(said["details"][0], "The page numbering starts again 12 times, as in a collection of separate pieces.")
        self.assertIn("Each place is marked below", said["details"][1])
        self.assertEqual(said["short"], "No page missing")
        self.assertNotIn("from 1 to 281", " ".join([said["headline"], *said["details"]]))

    def test_it_is_counted_as_neither_missing_nor_repeated(self):
        captures, restarts = self.collection(12)
        write_video(self.root, "run_x", "essays", captures, audit_final=final(200, 180, 1, 281, restarts=restarts))
        view = runs.video_view(self.root, "run_x", "essays")
        self.assertEqual((view["counts"]["missing"], view["counts"]["repeats"], view["counts"]["gaps"]), (0, 0, 0))
        self.assertEqual(view["counts"]["restarts"], 12)
        marks = [item for item in view["items"] if item["kind"] != "picture"]
        self.assertEqual([m["kind"] for m in marks], ["restart"] * 12)
        self.assertTrue(all(m["missing"] == 0 and m["extra"] == 0 for m in marks))
        self.assertNotIn("repeat", " ".join(view["verdict"]["details"]))

    def test_each_place_is_marked_between_its_two_pictures(self):
        captures = self.numbered(120)
        place = restart(captures, 60, 61, 302, 280, extra=12)
        write_video(self.root, "run_x", "essays", captures, audit_final=final(120, 100, 10, 398, restarts=[place]))
        view = runs.video_view(self.root, "run_x", "essays")
        kinds = [item["kind"] for item in view["items"]]
        at = kinds.index("restart")
        self.assertEqual(kinds.count("restart"), 1)
        self.assertEqual(view["items"][at - 1]["file"], captures[60]["file"])
        self.assertEqual(view["items"][at + 1]["file"], captures[61]["file"])
        mark = view["items"][at]
        self.assertEqual(mark["title"], "Numbering starts again")
        self.assertEqual(mark["text"], "The page numbering starts again here: 302, then 280.")
        self.assertEqual(view["items"][at - 1]["beside_kinds"], ["restart"])
        self.assertEqual(view["items"][at + 1]["beside_kinds"], ["restart"])
        self.assertEqual(view["items"][0]["beside_kinds"], [])

    def test_the_words_for_once_and_twice(self):
        for places, words in ((1, "starts again once"), (2, "starts again twice"), (3, "starts again 3 times")):
            with self.subTest(places=places):
                captures, restarts = self.collection(places)
                write_video(self.root, "run_x", f"essays-{places}", captures, audit_final=final(200, 180, 1, 281, restarts=restarts))
                said = runs.video_view(self.root, "run_x", f"essays-{places}")["verdict"]
                self.assertEqual(said["headline"], "No page is missing.")
                self.assertIn(words, said["details"][0])
                self.assertTrue(said["details"][1].startswith("The place is" if places == 1 else "Each place is"))

    def test_a_facing_page_translation_keeps_the_numbers_as_read(self):
        captures = self.numbered(120)
        place = restart(captures, 60, 61, 303, 281, extra=20)
        write_video(self.root, "run_x", "essays", captures, audit_final=final(120, 100, 10, 340, step=1, restarts=[place]))
        mark = next(item for item in runs.video_view(self.root, "run_x", "essays")["items"] if item["kind"] == "restart")
        self.assertEqual(mark["text"], "The page numbering starts again here: 303, then 281.")

    def test_a_restart_across_unread_pictures_says_between_which(self):
        captures = self.numbered(120)
        place = restart(captures, 60, 63, 302, 12, extra=140)
        write_video(self.root, "run_x", "essays", captures, audit_final=final(120, 100, 10, 120, restarts=[place]))
        view = runs.video_view(self.root, "run_x", "essays")
        mark = next(item for item in view["items"] if item["kind"] == "restart")
        self.assertEqual(mark["text"], "The page numbering starts again between picture 61 and picture 64: 302, then 12.")
        self.assertEqual([i["number"] for i in view["items"] if i["kind"] == "picture" and "restart" in i["beside_kinds"]], [61, 62, 63, 64])

    def test_a_real_gap_beside_restarts_is_still_the_headline(self):
        captures, restarts = self.collection(3)
        gap = anomaly(captures, 97, 98, 176, 180, missing=1)
        write_video(self.root, "run_x", "essays", captures, audit_final=final(200, 180, 1, 281, [gap], restarts=restarts))
        view = runs.video_view(self.root, "run_x", "essays")
        said = view["verdict"]
        self.assertEqual((said["state"], said["headline"]), ("missing", "1 spread missing after page 176"))
        self.assertIn("The page numbering starts again 3 times, as in a collection of separate pieces.", said["details"][1])
        self.assertNotIn("run from", " ".join(said["details"]))
        self.assertEqual((view["counts"]["missing"], view["counts"]["restarts"], view["counts"]["gaps"]), (1, 3, 1))
        self.assertEqual(sorted({item["kind"] for item in view["items"]}), ["gap", "picture", "restart"])

    def test_restarts_in_a_video_the_check_cannot_vouch_for(self):
        captures, restarts = self.collection(2)
        write_video(self.root, "run_x", "essays", captures, audit_final=final(200, 40, 1, 281, restarts=restarts))
        said = runs.video_view(self.root, "run_x", "essays")["verdict"]
        self.assertEqual(said["state"], "unreadable")
        self.assertIn("the page numbering starts again twice", " ".join(said["details"]))

    def test_an_empty_list_and_a_missing_key_both_mean_none(self):
        captures = self.numbered(118)
        write_video(self.root, "run_x", "new", captures, audit_final=final(118, 108, 10, 232, restarts=[]))
        write_video(self.root, "run_x", "old", captures, audit_final=final(118, 108, 10, 232))
        self.assertNotIn("restarts", json.loads((self.root / "run_x" / "old" / "audit.json").read_text())["final"])
        for slug in ("new", "old"):
            with self.subTest(slug=slug):
                view = runs.video_view(self.root, "run_x", slug)
                self.assertEqual(view["verdict"]["headline"], "Every page number from 10 to 232 is there.")
                self.assertEqual(view["counts"]["restarts"], 0)
                self.assertEqual({item["kind"] for item in view["items"]}, {"picture"})


class PageLabelTests(RunFolderCase):
    """``page`` on each capture: the printed number of its left-hand page, or null when unread."""

    def test_an_ordinary_book_shows_two_pages_a_picture(self):
        self.assertEqual(runs.page_label(46, 2), "p. 46-47")
        self.assertEqual(runs.page_label(302, 2), "p. 302-303")

    def test_a_facing_page_translation_shows_one_number_a_picture(self):
        self.assertEqual(runs.page_label(46, 1), "p. 46")

    def test_the_first_spread_has_only_page_one(self):
        self.assertEqual(runs.page_label(0, 2), "p. 1")

    def test_nothing_is_said_when_the_page_or_the_numbering_is_not_known(self):
        for page, step in ((None, 2), (None, 1), (46, None), (46, 3), ("46", 2), (True, 2), (-2, 2), (4.0, 2)):
            with self.subTest(page=page, step=step):
                self.assertIsNone(runs.page_label(page, step))

    def pictures(self, step: int) -> list[dict]:
        pages = [None, 10, 12, None, 16] if step == 2 else [None, 21, 22, None, 24]
        captures = [capture(n, 2.0 * n, page=page) for n, page in enumerate(pages, 1)]
        write_video(self.root, "run_x", "book", captures, audit_final=final(5, 3, pages[1], pages[-1], step=step))
        return [item for item in runs.video_view(self.root, "run_x", "book")["items"] if item["kind"] == "picture"]

    def test_each_picture_is_labelled_in_an_ordinary_book(self):
        self.assertEqual([item["pages"] for item in self.pictures(2)], [None, "p. 10-11", "p. 12-13", None, "p. 16-17"])

    def test_each_picture_is_labelled_in_a_facing_page_translation(self):
        self.assertEqual([item["pages"] for item in self.pictures(1)], [None, "p. 21", "p. 22", None, "p. 24"])

    def test_an_older_run_has_no_page_on_its_captures(self):
        captures = self.numbered(5)
        self.assertNotIn("page", captures[0])
        write_video(self.root, "run_x", "book", captures, audit_final=final(5, 3, 10, 16))
        self.assertEqual([item["pages"] for item in runs.video_view(self.root, "run_x", "book")["items"]], [None] * 5)

    def test_pages_without_a_check_to_say_how_the_book_is_numbered_are_not_labelled(self):
        captures = [capture(n, 2.0 * n, page=8 + 2 * n) for n in range(1, 6)]
        write_video(self.root, "run_x", "book", captures)
        self.assertEqual([item["pages"] for item in runs.video_view(self.root, "run_x", "book")["items"]], [None] * 5)


class LeftOutTests(RunFolderCase):
    """``not_pages``, ``shown_elsewhere`` and ``transitional``: frames that were not kept as a page of their own."""

    @staticmethod
    def frames(start: int, count: int, reason: str) -> list[dict]:
        return [{"frame": start + 10 * n, "reason": reason} for n in range(count)]

    def test_one_sentence_with_the_counts(self):
        more = {"not_pages": self.frames(7900, 2, "no paper in the page region: not an open book"),
                "shown_elsewhere": self.frames(100, 12, "right page is shown 1 capture(s) later"), "transitional": []}
        write_video(self.root, "run_x", "book", self.numbered(20), more=more)
        told = runs.video_view(self.root, "run_x", "book")["left_out"]
        self.assertEqual(told["text"], "14 frames were left out: 2 from before the book was opened or after it was closed, "
                                       "12 glimpses during page turns.")
        self.assertEqual((told["total"], told["not_open"], told["glimpses"]), (14, 2, 12))

    def test_both_kinds_of_glimpse_are_counted_together(self):
        more = {"not_pages": self.frames(7900, 2, "no paper"), "shown_elsewhere": self.frames(100, 12, "shown later"),
                "transitional": self.frames(105, 55, "right page matches the next spread")}
        told = runs.left_out(more)
        self.assertEqual(told["text"], "69 frames were left out: 2 from before the book was opened or after it was closed, "
                                       "67 glimpses during page turns.")

    def test_one_kind_alone_and_single_frames(self):
        self.assertEqual(runs.left_out({"transitional": self.frames(5, 3, "x")})["text"], "3 glimpses during page turns were left out.")
        self.assertEqual(runs.left_out({"shown_elsewhere": self.frames(5, 1, "x")})["text"], "1 glimpse during a page turn was left out.")
        self.assertEqual(runs.left_out({"not_pages": self.frames(5, 1, "x")})["text"],
                         "1 frame from before the book was opened or after it was closed was left out.")
        self.assertEqual(runs.left_out({"not_pages": self.frames(5, 2, "x"), "transitional": self.frames(50, 1, "x")})["text"],
                         "3 frames were left out: 2 from before the book was opened or after it was closed, 1 glimpse during a page turn.")

    def test_nothing_is_said_when_nothing_was_left_out_or_the_lists_are_not_there(self):
        self.assertIsNone(runs.left_out({}))
        self.assertIsNone(runs.left_out({"not_pages": [], "shown_elsewhere": [], "transitional": []}))
        self.assertIsNone(runs.left_out({"not_pages": None, "shown_elsewhere": "many", "transitional": 3}))
        write_video(self.root, "run_x", "book", self.numbered(20))
        self.assertIsNone(runs.video_view(self.root, "run_x", "book")["left_out"])


class OlderRunTests(RunFolderCase):
    """A run from before any of the newer fields, as run_20261111_corpus is, reads exactly as it did."""

    def test_none_of_the_newer_keys(self):
        captures = self.numbered(118)
        folder = write_video(self.root, "run_20261111_corpus", "einar", captures, audit_final=final(118, 108, 10, 232))
        spreads = json.loads((folder / "spreads.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        self.assertFalse({"not_pages", "shown_elsewhere", "transitional"} & set(spreads))
        self.assertNotIn("page", spreads["captures"][0])
        self.assertNotIn("restarts", audit["final"])
        view = runs.video_view(self.root, "run_20261111_corpus", "einar")
        self.assertEqual(view["verdict"]["headline"], "Every page number from 10 to 232 is there.")
        self.assertEqual(view["verdict"]["details"], ["Page numbers were read on 108 of 118 pictures and they run on without a gap."])
        self.assertEqual(view["counts"]["restarts"], 0)
        self.assertIsNone(view["left_out"])
        self.assertTrue(all(item["pages"] is None and item["beside_kinds"] == [] for item in view["items"]))
        self.assertEqual(runs.video_summary(self.root, "run_20261111_corpus", "einar")["verdict"]["short"], "All pages there (10 to 232)")

    def test_a_run_without_any_check_has_no_count_of_restarts(self):
        write_video(self.root, "run_20261111_fast_v0", "book", self.numbered(20))
        view = runs.video_view(self.root, "run_20261111_fast_v0", "book")
        self.assertIsNone(view["counts"]["restarts"])
        self.assertIsNone(view["left_out"])


class PicturesOverHttpTests(unittest.TestCase):
    """Thumbnails, full-size pictures and the ZIP, from a small made-up run."""

    @classmethod
    def setUpClass(cls):
        from PIL import Image

        cls.tmp = tempfile.TemporaryDirectory()
        cls.config = Config.under(Path(cls.tmp.name))
        cls.captures = [capture(n, 2.0 * n) for n in (1, 2, 3)]
        folder = write_video(cls.config.runs_root, "run_a", "book", cls.captures, files=False)
        for shade, item in zip((40, 120, 200), cls.captures):
            Image.new("RGB", (1536, 864), (shade, shade, shade)).save(folder / "captures" / item["file"], "JPEG", quality=85)
        (folder / "captures" / "left-over.jpg").write_bytes(b"not listed in spreads.json")
        cls.folder = folder
        cls.config.video_dir.mkdir()
        cls.server = Server(App(cls.config), 0, quiet=True)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def get(self, path: str):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def test_a_thumbnail_is_small_and_kept_on_disk(self):
        from PIL import Image

        name = self.captures[0]["file"]
        status, body, headers = self.get(f"/thumb/run_a/book/{name}")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        with Image.open(io.BytesIO(body)) as image:
            self.assertEqual(image.size, (480, 270))
        kept = list((self.config.thumbs_dir / "run_a" / "book").iterdir())
        self.assertEqual([p.name for p in kept], [f"{name}.w480.jpg"])
        self.assertLess(len(body), (self.folder / "captures" / name).stat().st_size)
        made = kept[0].stat().st_mtime_ns
        self.assertEqual(self.get(f"/thumb/run_a/book/{name}")[1], body)
        self.assertEqual(kept[0].stat().st_mtime_ns, made)

    def test_the_full_size_picture_is_the_file_itself(self):
        name = self.captures[1]["file"]
        status, body, _ = self.get(f"/image/run_a/book/{name}")
        self.assertEqual(status, 200)
        self.assertEqual(body, (self.folder / "captures" / name).read_bytes())

    def test_the_zip_holds_the_listed_pictures_in_page_order(self):
        status, body, headers = self.get("/zip/run_a/book")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        self.assertEqual(headers.get("Transfer-Encoding"), "chunked")
        self.assertIn('filename="book.zip"', headers["Content-Disposition"])
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.namelist(), [f"book/{item['file']}" for item in self.captures])
            for item in self.captures:
                self.assertEqual(archive.read(f"book/{item['file']}"), (self.folder / "captures" / item["file"]).read_bytes())
                self.assertEqual(archive.getinfo(f"book/{item['file']}").compress_type, zipfile.ZIP_STORED)

    def test_the_result_as_the_page_asks_for_it(self):
        status, body, headers = self.get("/api/runs/run_a/book")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        view = json.loads(body)
        self.assertEqual([item["file"] for item in view["items"]], [item["file"] for item in self.captures])
        self.assertTrue(all(item["on_disk"] for item in view["items"]))
        status, body, _ = self.get("/api/runs")
        self.assertEqual([run["run"] for run in json.loads(body)["runs"]], ["run_a"])

    def test_a_video_without_pictures_has_no_zip(self):
        write_video(self.config.runs_root, "run_a", "empty-book", [])
        self.assertEqual(self.get("/zip/run_a/empty-book")[0], 404)


if __name__ == "__main__":
    unittest.main()
