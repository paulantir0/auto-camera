import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from auto_camera import desktop, paths
from auto_camera.pipeline import STAGES


def completed(command, code=0, out=""):
    return subprocess.CompletedProcess(command, code, stdout=out, stderr="")


class FakeFFmpeg:
    """Stands in for subprocess.run: answers ``-hwaccels``, makes the test clip, and decodes it (or not)."""

    def __init__(self, hwaccels="Hardware acceleration methods:\nvdpau\ncuda\nvaapi\n", decode_code=0, clip_code=0):
        self.hwaccels = hwaccels
        self.decode_code = decode_code
        self.clip_code = clip_code
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if "-hwaccels" in command:
            return completed(command, out=self.hwaccels)
        if "lavfi" in command:
            if self.clip_code == 0:
                Path(command[-1]).write_bytes(b"clip")
            return completed(command, self.clip_code)
        if "-hwaccel" in command:
            return completed(command, self.decode_code)
        raise AssertionError(f"unexpected command {command}")


class GpuCheckTests(unittest.TestCase):
    def check(self, fake):
        with mock.patch.object(desktop.subprocess, "run", side_effect=fake):
            return desktop.check_gpu()

    def test_cuda_listed_and_decoding_works(self):
        fake = FakeFFmpeg()
        usable, _ = self.check(fake)
        self.assertTrue(usable)
        decode = fake.commands[-1]
        # The check decodes the way the pipeline's proxy does: on the card, shrunk there.
        self.assertIn("cuda", decode[decode.index("-hwaccel") + 1])
        self.assertIn("-hwaccel_output_format", decode)
        self.assertTrue(any("scale_cuda" in part for part in decode))

    def test_no_cuda_in_ffmpeg_means_cpu(self):
        fake = FakeFFmpeg(hwaccels="Hardware acceleration methods:\nvaapi\nqsv\n")
        usable, why = self.check(fake)
        self.assertFalse(usable)
        self.assertIn("CUDA", why)
        self.assertEqual(len(fake.commands), 1)

    def test_cuda_listed_but_no_card_means_cpu(self):
        usable, why = self.check(FakeFFmpeg(decode_code=1))
        self.assertFalse(usable)
        self.assertIn("NVIDIA", why)

    def test_test_clip_not_made_means_cpu(self):
        self.assertFalse(self.check(FakeFFmpeg(clip_code=1))[0])

    def test_no_ffmpeg_means_cpu(self):
        usable, why = self.check(FileNotFoundError(2, "No such file", "ffmpeg"))
        self.assertFalse(usable)
        self.assertIn("not found", why)

    def test_a_hung_ffmpeg_means_cpu(self):
        self.assertFalse(self.check(subprocess.TimeoutExpired(["ffmpeg"], 60))[0])

    def test_detect_gpu_is_the_yes_or_no(self):
        with mock.patch.object(desktop.subprocess, "run", side_effect=FakeFFmpeg(decode_code=1)):
            self.assertFalse(desktop.detect_gpu())
        with mock.patch.object(desktop.subprocess, "run", side_effect=FakeFFmpeg()):
            self.assertTrue(desktop.detect_gpu())

    def test_only_a_cuda_command_failing_is_taken_for_the_card(self):
        self.assertTrue(desktop._gpu_failure(subprocess.CalledProcessError(1, ["ffmpeg", "-hwaccel", "cuda", "-i", "x"])))
        self.assertFalse(desktop._gpu_failure(subprocess.CalledProcessError(1, ["ffmpeg", "-i", "x"])))
        self.assertFalse(desktop._gpu_failure(RuntimeError("cuda")))


class SettingsTests(unittest.TestCase):
    def test_workers_leave_a_core_free_and_stop_at_four(self):
        self.assertEqual(desktop.default_workers(1), 1)
        self.assertEqual(desktop.default_workers(2), 1)
        self.assertEqual(desktop.default_workers(4), 3)
        self.assertEqual(desktop.default_workers(12), 4)

    def test_output_goes_to_pictures_by_day(self):
        with mock.patch.object(desktop, "pictures_dir", return_value=Path("/home/reader/Pictures")):
            self.assertEqual(desktop.default_output_dir(date(2026, 11, 12)), Path("/home/reader/Pictures/AutoCamera/20261112"))

    def test_every_pipeline_stage_has_words(self):
        for stage in STAGES:
            self.assertIn(stage, desktop.STAGE_WORDS)
        self.assertEqual(desktop.stage_words("later", "doing something new"), "Doing something new")


class EnvironmentTests(unittest.TestCase):
    def test_bundled_ffmpeg_goes_first_when_frozen(self):
        with tempfile.TemporaryDirectory() as bundle:
            tools = Path(bundle) / "ffmpeg"
            tools.mkdir()
            (tools / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")).write_text("")
            environ = {"PATH": os.pathsep.join(["/usr/bin", "/bin"])}
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(sys, "_MEIPASS", bundle, create=True):
                desktop.prepare_environment(environ)
            self.assertEqual(environ["PATH"].split(os.pathsep)[0], str(tools))

    def test_from_source_path_is_left_alone(self):
        environ = {"PATH": os.pathsep.join(["/usr/bin", "/bin"])}
        with mock.patch.object(desktop, "find_tesseract", return_value=None):
            desktop.prepare_environment(environ)
        self.assertEqual(environ["PATH"], os.pathsep.join(["/usr/bin", "/bin"]))

    def test_tesseract_is_found_where_its_windows_installer_puts_it(self):
        with tempfile.TemporaryDirectory() as programs:
            exe = Path(programs) / "Tesseract-OCR" / "tesseract.exe"
            exe.parent.mkdir()
            exe.write_text("")
            environ = {"PATH": "", "ProgramFiles": programs}
            with mock.patch.object(desktop.sys, "platform", "win32"), mock.patch.object(desktop.shutil, "which", return_value=None):
                self.assertEqual(desktop.find_tesseract(environ), exe)
            with mock.patch.object(desktop, "find_tesseract", return_value=exe):
                desktop.prepare_environment(environ)
            self.assertIn(str(exe.parent), environ["PATH"].split(os.pathsep))


class CacheTests(unittest.TestCase):
    def test_each_file_gets_its_own_cache_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a" / "IMG_0001.MOV"
            second = Path(tmp) / "b" / "IMG_0001.MOV"
            for path in (first, second):
                path.parent.mkdir()
                path.write_bytes(b"video")
            self.assertTrue(desktop.video_cache_name(first).startswith("img-0001-"))
            self.assertNotEqual(desktop.video_cache_name(first), desktop.video_cache_name(second))
            before = desktop.video_cache_name(first)
            os.utime(first, (time.time() + 100, time.time() + 100))
            self.assertNotEqual(desktop.video_cache_name(first), before)

    def test_old_caches_are_pruned_and_others_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folders = []
            for index in range(5):
                folder = base / f"book-{index:010x}"
                folder.mkdir()
                os.utime(folder, (1000 + index, 1000 + index))
                folders.append(folder)
            other = base / "not-a-video-cache"
            other.mkdir()
            current = folders[0]
            removed = desktop.prune_caches(base, 3, current=current)
            self.assertTrue(current.exists())
            self.assertTrue(other.exists())
            self.assertEqual(sorted(removed), sorted(folders[1:3]))
            self.assertTrue(folders[3].exists() and folders[4].exists())


class SummaryTests(unittest.TestCase):
    def result(self, folder: Path, **extra):
        out = {"folder": str(folder), "captures": 12, "to_review": 2, "video_seconds": 250.0, "total_seconds": 70.0,
               "audit": None, "hand_note": "", "book": {"pages": 24, "searchable": False, "dpi": 300}}
        out.update(extra)
        return out

    def test_summary_without_tesseract_says_how_to_get_a_searchable_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "book.pdf").write_bytes(b"%PDF")
            (folder / "needs_review.txt").write_text("spread_0003.jpg\thand_on_print\nspread_0007.jpg\tnever_at_rest\n")
            summary = desktop.summarize(self.result(folder))
        text = "\n".join(summary["lines"])
        self.assertIn("12 spreads", text)
        self.assertIn("24 page images", text)
        self.assertIn("not searchable", text)
        self.assertIn("Tesseract", text)
        self.assertIn("2 spreads may need a look", text)
        self.assertEqual(summary["pdf"], str(folder / "book.pdf"))
        self.assertEqual(summary["review"], str(folder / "needs_review.txt"))
        self.assertFalse(summary["searchable"])

    def test_summary_with_tesseract_and_nothing_to_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "needs_review.txt").write_text("")
            summary = desktop.summarize(self.result(folder, to_review=0, book={"pages": 24, "searchable": True, "languages": ["eng", "lat"]}),
                                        ["The NVIDIA card could not decode this video, so it was read on the CPU instead."])
        text = "\n".join(summary["lines"])
        self.assertIn("searchable PDF", text)
        self.assertIn("English, Latin", text)
        self.assertIn("Nothing was flagged", text)
        self.assertIn("read on the CPU", text)
        self.assertIsNone(summary["pdf"])
        self.assertIsNone(summary["review"])

    def test_errors_are_put_plainly(self):
        error = subprocess.CalledProcessError(1, ["/usr/bin/ffmpeg", "-i", "x"], stderr=b"line one\nInvalid data found\n")
        self.assertEqual(desktop._plain_error(error), "ffmpeg failed (exit code 1): line one Invalid data found")
        self.assertEqual(desktop._plain_error(MemoryError()), "The computer ran out of memory.")


class ChildProcessTests(unittest.TestCase):
    def test_a_scan_that_fails_reports_it_through_the_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = desktop.ScanSettings(video=str(Path(tmp) / "missing.mov"), run_dir=tmp)
            events = []
            outcome = desktop.run_in_child(settings, events.append, poll_seconds=0.05)
        self.assertEqual(outcome[0], "error")
        self.assertIn("not there", outcome[1])
        self.assertEqual(events[-1], outcome)

    def test_stop_ends_the_process_and_what_it_started(self):
        import psutil

        code = ("import subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "print(child.pid, flush=True)\n"
                "time.sleep(60)\n")
        def alive(pid: int) -> bool:
            try:
                return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False

        top = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            grandchild = int(top.stdout.readline())
            self.assertTrue(alive(grandchild))
            desktop.stop_tree(top.pid, wait=5)
            top.wait(5)
            self.assertFalse(alive(grandchild))
        finally:
            if top.poll() is None:
                top.kill()
                top.wait(5)
            top.stdout.close()


if __name__ == "__main__":
    unittest.main()
