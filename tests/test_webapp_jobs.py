"""Following a scan from the lines it prints, and running one at a time. No video needed.

The scans here are tiny stand-in commands, not the pipeline.
"""

import ast
import json
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path

from webapp import REPO_ROOT, Config, runs
from webapp.jobs import STAGE_WORDS, Busy, Job, JobRunner, default_command, parse_progress, slugify, web_run_dir

STAND_IN = r"""
import json, sys, time
from pathlib import Path
mode, run, slug = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
def say(text):
    print(text, flush=True)
if mode == "busy":
    print("run_capped: load is 14.2 (over 12); not starting. Wait, or set FORCE=1.", file=sys.stderr, flush=True)
    sys.exit(75)
say("[proxy] making a small copy of the video")
if mode == "slow":
    time.sleep(60)
if mode == "fail":
    print("[warning] this line is on standard error and is not a stage", file=sys.stderr)
    print("Traceback (most recent call last):", file=sys.stderr)
    print("RuntimeError: ffmpeg said no", file=sys.stderr, flush=True)
    sys.exit(1)
say("[analysis] measuring motion and sharpness in every frame")
say("[hands] finding hands over the page")
say("[turns] finding page turns and choosing a frame for each spread")
say("[stills] saving 3 full-size frames and removing repeats")
say("[finish] writing the final folder")
say("[pages] flattening each page and binding the book")
out = run / slug
out.mkdir(parents=True, exist_ok=True)
(out / "spreads.json").write_text(json.dumps({"captures": []}))
(out / "result.json").write_text(json.dumps({"slug": slug, "captures": 0, "audit": None, "audit_note": ""}))
say("[done] ")
say(json.dumps({"slug": slug}, indent=1))
"""


def wait_until(test, seconds: float = 10.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if test():
            return True
        time.sleep(0.02)
    return test()


class ProgressLineTests(unittest.TestCase):
    def test_a_stage_line(self):
        self.assertEqual(parse_progress("[proxy] making a small copy of the video\n"), ("proxy", "making a small copy of the video"))
        self.assertEqual(parse_progress("[stills] saving 127 full-size frames and removing repeats"),
                         ("stills", "saving 127 full-size frames and removing repeats"))

    def test_a_stage_line_without_detail(self):
        self.assertEqual(parse_progress("[done] "), ("done", ""))
        self.assertEqual(parse_progress("[done]"), ("done", ""))
        self.assertEqual(parse_progress("[finish]\r\n"), ("finish", ""))

    def test_other_output_is_not_progress(self):
        for line in ("", "{", ' "slug": "einar",', ' "to_review": [', " ]", "frame= 2751 fps=310", "run_capped: load is 14 (over 12)",
                     "  [proxy] indented", "[Proxy] capital", "[12] number", "[] empty", "text [proxy] later",
                     "[h264 @ 0x55d0c0] error while decoding"):
            with self.subTest(line=line):
                self.assertIsNone(parse_progress(line))

    def test_the_stage_names_are_the_pipelines(self):
        tree = ast.parse((REPO_ROOT / "auto_camera" / "pipeline.py").read_text())
        stages = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "STAGES" for t in node.targets))
        self.assertEqual(list(STAGE_WORDS), stages)

    def test_the_folder_name_is_the_pipelines(self):
        from auto_camera.proxy import slugify as pipeline_slugify
        for name in ("Einar Thomasson Canon and Canonicity IMG-0014.M4V", "Ignatius, Lettres SC - IMG-0023.M4V",
                     "Olson, Mark Wesley's Doctrine of Justification - IMG-0024.M4V", "IMG-0042.M4V", "new book-2.mp4", "---.mp4"):
            with self.subTest(name=name):
                self.assertEqual(slugify(name), pipeline_slugify(name))


class JobProgressTests(unittest.TestCase):
    def job(self) -> Job:
        return Job("Book.mp4", "library", Path("/videos/Book.mp4"), Path("/runs/run_20261111_web"))

    def states(self, job: Job) -> dict:
        return {stage["key"]: stage["state"] for stage in job.snapshot()["stages"]}

    def test_starts_with_every_stage_waiting_in_the_pipelines_order(self):
        job = self.job()
        shot = job.snapshot()
        self.assertEqual([stage["key"] for stage in shot["stages"]], ["proxy", "analysis", "hands", "turns", "stills", "numbers", "finish", "pages"])
        self.assertEqual(set(self.states(job).values()), {"waiting"})
        self.assertEqual((shot["status"], shot["active"], shot["slug"], shot["run"]), ("starting", True, "book", "run_20261111_web"))

    def test_each_line_moves_one_stage_on_and_times_the_last(self):
        job = self.job()
        self.assertTrue(job.feed("[proxy] making a small copy of the video", now=100.0))
        self.assertEqual(self.states(job)["proxy"], "running")
        self.assertEqual(job.snapshot(now=103.0)["stages"][0]["seconds"], 3.0)
        self.assertTrue(job.feed("[analysis] measuring motion and sharpness in every frame", now=112.5))
        shot = job.snapshot(now=113.0)
        self.assertEqual((shot["stages"][0]["state"], shot["stages"][0]["seconds"]), ("done", 12.5))
        self.assertEqual((shot["stages"][1]["state"], shot["stages"][1]["detail"]), ("running", "measuring motion and sharpness in every frame"))
        self.assertEqual(shot["status"], "running")

    def test_output_that_is_not_progress_changes_nothing_but_is_kept_for_the_error_report(self):
        job = self.job()
        job.feed("[proxy] making a small copy of the video", now=1.0)
        before = self.states(job)
        self.assertFalse(job.feed('{"slug": "book"}', now=2.0))
        self.assertFalse(job.feed("frame= 100 fps= 30", now=2.0))
        self.assertEqual(self.states(job), before)
        job.finish("failed", "The scan stopped with an error.", now=3.0)
        self.assertEqual(job.snapshot()["output"], ['{"slug": "book"}', "frame= 100 fps= 30"])

    def test_a_stage_that_is_passed_over_is_marked_not_needed(self):
        job = self.job()
        for now, line in enumerate(("[proxy] a", "[analysis] b", "[hands] c", "[turns] d", "[stills] e", "[finish] f", "[pages] g", "[done] ")):
            job.feed(line, now=float(now))
        self.assertEqual(self.states(job), {"proxy": "done", "analysis": "done", "hands": "done", "turns": "done", "stills": "done",
                                            "numbers": "skipped", "finish": "done", "pages": "done"})

    def test_a_stage_the_page_does_not_know_is_shown_under_its_own_words(self):
        job = self.job()
        job.feed("[proxy] a", now=0.0)
        job.feed("[deskew] straightening the pages", now=1.0)
        last = job.snapshot()["stages"][-1]
        self.assertEqual((last["key"], last["label"], last["state"]), ("deskew", "Straightening the pages", "running"))

    def test_finishing_closes_the_stage_in_hand(self):
        job = self.job()
        job.feed("[proxy] a", now=0.0)
        job.feed("[analysis] b", now=4.0)
        job.finish("cancelled", "Stopped.", now=9.0)
        shot = job.snapshot()
        self.assertEqual((shot["stages"][1]["state"], shot["stages"][1]["seconds"]), ("stopped", 5.0))
        self.assertEqual((shot["status"], shot["active"]), ("cancelled", False))
        self.assertNotIn("output", shot)


class CommandTests(unittest.TestCase):
    def test_a_scan_is_the_capped_command_from_the_project_notes(self):
        config = Config.default()
        command = default_command(config, Path("/videos/A Book.M4V"), Path("/pictures/run_20261111_web"))
        self.assertEqual(command, [str(REPO_ROOT / "tools" / "run_capped.sh"), str(REPO_ROOT / ".venv" / "bin" / "python"),
                                   str(REPO_ROOT / "tools" / "run_video.py"), "/videos/A Book.M4V", "--run", "/pictures/run_20261111_web"])

    def test_scans_from_the_page_go_to_a_dated_web_folder(self):
        config = Config.default()
        self.assertEqual(web_run_dir(config, date(2026, 11, 11)), Path.home() / "Pictures" / "auto-camera-production" / "run_20261111_web")
        self.assertEqual(web_run_dir(config, date(2027, 1, 5)).name, "run_20270105_web")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = Config.under(Path(self.tmp.name))
        self.config.video_dir.mkdir(parents=True)
        self.video = self.config.video_dir / "A Book.mp4"
        self.video.write_bytes(b"not really a video")
        self.runner: JobRunner | None = None

    def tearDown(self):
        if self.runner is not None:
            self.runner.shutdown(wait=5.0)
        self.tmp.cleanup()

    def start(self, mode: str):
        def command(config, video, run_dir):
            return [sys.executable, "-c", STAND_IN, mode, str(run_dir), slugify(video.name)]
        self.runner = JobRunner(self.config, command)
        return self.runner.start(self.video, self.video.name, "library")

    def test_a_scan_that_finishes(self):
        job = self.start("ok")
        self.assertTrue(wait_until(lambda: not job.active))
        shot = job.snapshot()
        self.assertEqual((shot["status"], shot["message"]), ("done", ""))
        self.assertEqual([s["state"] for s in shot["stages"]], ["done", "done", "done", "done", "done", "skipped", "done", "done"])
        folder = web_run_dir(self.config) / "a-book"
        self.assertEqual(json.loads((folder / runs.JOB_MARKER).read_text())["status"], "done")
        self.assertEqual((shot["run"], shot["slug"]), (web_run_dir(self.config).name, "a-book"))
        self.assertIsNone(self.runner.live())
        self.assertFalse((self.config.state_dir / "running_job.json").exists())

    def test_a_second_scan_is_refused_while_one_is_going(self):
        job = self.start("slow")
        self.assertTrue(wait_until(lambda: job.snapshot()["status"] == "running"))
        with self.assertRaises(Busy) as refused:
            self.runner.start(self.video, self.video.name, "library")
        self.assertIs(refused.exception.job, job)
        self.assertEqual(self.runner.live(), (web_run_dir(self.config).name, "a-book"))
        self.runner.cancel(job.id)
        self.assertTrue(wait_until(lambda: not job.active))
        again = self.runner.start(self.video, self.video.name, "library")
        self.assertIsNot(again, job)

    def test_stopping_a_scan_ends_its_process_and_marks_the_folder(self):
        job = self.start("slow")
        self.assertTrue(wait_until(lambda: job.snapshot()["stages"][0]["state"] == "running"))
        process = job.process
        self.runner.cancel(job.id)
        self.assertTrue(job.snapshot()["cancelling"])
        self.assertTrue(wait_until(lambda: not job.active, 8.0))
        self.assertIsNotNone(process.poll())
        shot = job.snapshot()
        self.assertEqual((shot["status"], shot["stages"][0]["state"]), ("cancelled", "stopped"))
        folder = web_run_dir(self.config) / "a-book"
        self.assertEqual(json.loads((folder / runs.JOB_MARKER).read_text())["status"], "cancelled")

    def test_a_busy_machine_is_reported_as_such_and_leaves_no_trace(self):
        job = self.start("busy")
        self.assertTrue(wait_until(lambda: not job.active))
        shot = job.snapshot()
        self.assertEqual(shot["status"], "busy")
        self.assertIn("busy with other work", shot["message"])
        self.assertFalse((web_run_dir(self.config) / "a-book").exists())

    def test_a_scan_that_fails_says_where_and_keeps_the_output(self):
        job = self.start("fail")
        self.assertTrue(wait_until(lambda: not job.active))
        shot = job.snapshot()
        self.assertEqual(shot["status"], "failed")
        self.assertEqual(shot["message"], "The scan stopped with an error while making a small copy of the video.")
        self.assertIn("RuntimeError: ffmpeg said no", shot["output"])
        self.assertEqual([s["key"] for s in shot["stages"]], list(STAGE_WORDS))
        self.assertEqual(json.loads((web_run_dir(self.config) / "a-book" / runs.JOB_MARKER).read_text())["status"], "failed")

    def test_a_command_that_cannot_be_started(self):
        self.runner = JobRunner(self.config, lambda config, video, run_dir: [str(Path(self.tmp.name) / "no-such-program")])
        job = self.runner.start(self.video, self.video.name, "library")
        self.assertTrue(wait_until(lambda: not job.active))
        self.assertEqual(job.snapshot()["status"], "failed")

    def test_a_scan_left_running_by_an_earlier_server_blocks_a_new_one(self):
        import subprocess
        left_over = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "tools/run_video.py"])
        try:
            self.config.state_dir.mkdir(parents=True)
            note = self.config.state_dir / "running_job.json"
            note.write_text(json.dumps({"pid": left_over.pid, "video": "Earlier Book.mp4"}))
            self.runner = JobRunner(self.config, lambda *_: [sys.executable, "-c", "pass"])
            with self.assertRaises(Busy) as refused:
                self.runner.start(self.video, self.video.name, "library")
            self.assertIsNone(refused.exception.job)
            self.assertIn("Earlier Book.mp4", refused.exception.message)
        finally:
            left_over.kill()
            left_over.wait()
        # once that process is gone the note is stale and no longer blocks
        job = self.runner.start(self.video, self.video.name, "library")
        self.assertTrue(wait_until(lambda: not job.active))
        self.assertFalse(note.exists())

    def test_a_file_name_with_nothing_to_name_a_folder_by_is_refused(self):
        self.runner = JobRunner(self.config, lambda *_: [sys.executable, "-c", "pass"])
        with self.assertRaises(ValueError):
            self.runner.start(self.config.video_dir / "---.mp4", "---.mp4", "library")


if __name__ == "__main__":
    unittest.main()
