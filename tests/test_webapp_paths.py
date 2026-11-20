"""What the browser app will and will not serve. No video needed."""

import http.client
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from webapp import Config, paths
from webapp.paths import Refused
from webapp.server import App, Server
from webapp.videos import clean_upload_name


def refused_with(case: unittest.TestCase, status: int, call, *args):
    with case.assertRaises(Refused) as caught:
        call(*args)
    case.assertEqual(caught.exception.status, status)


class UrlPathTests(unittest.TestCase):
    def test_plain_components(self):
        self.assertEqual(paths.split_url_path("/image/run_1/some-book/sequence_0001_t0000.73s.jpg?v=3"),
                         ["image", "run_1", "some-book", "sequence_0001_t0000.73s.jpg"])

    def test_percent_encoding_is_decoded_per_component(self):
        self.assertEqual(paths.split_url_path("/a%20b/c"), ["a b", "c"])

    def test_dot_segments_are_refused_plain_and_encoded(self):
        for raw in ("/image/../x", "/image/./x", "/image/%2e%2e/x", "/image/%2E%2E/x", "/image/.%2e/x"):
            with self.subTest(raw=raw):
                refused_with(self, 400, paths.split_url_path, raw)

    def test_an_encoded_separator_cannot_add_a_level(self):
        for raw in ("/image/a%2fb", "/image/a%5cb", "/image/..%2f..%2fetc%2fpasswd", "/image/a%00b"):
            with self.subTest(raw=raw):
                refused_with(self, 400, paths.split_url_path, raw)

    def test_bytes_that_are_not_text_are_refused(self):
        refused_with(self, 400, paths.split_url_path, "/image/%ff%fe")


class SafeNameTests(unittest.TestCase):
    def test_ordinary_names_pass(self):
        for name in ("run_20261111_corpus", "img-0052", "sequence_0001_t0000.73s.jpg", "a"):
            self.assertEqual(paths.safe_name(name), name)

    def test_anything_that_could_leave_the_folder_is_refused(self):
        for name in ("", ".", "..", ".hidden", "a/b", "a\\b", "/etc", "-rf", "a b", "caf\u00e9", "a\0b", "x" * 300, None, 7):
            with self.subTest(name=name):
                refused_with(self, 400, paths.safe_name, name)


class RunFolderPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "runs"
        captures = self.root / "run_a" / "book" / "captures"
        captures.mkdir(parents=True)
        (captures / "sequence_0001_t0000.73s.jpg").write_bytes(b"picture")
        (captures / "notes.txt").write_text("not a picture")
        self.secret = base / "secret.jpg"
        self.secret.write_bytes(b"secret")
        self.outside = base / "outside"
        (self.outside / "book" / "captures").mkdir(parents=True)
        (self.outside / "book" / "captures" / "x.jpg").write_bytes(b"outside")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_capture_resolves_inside_the_root(self):
        found = paths.capture_file(self.root, "run_a", "book", "sequence_0001_t0000.73s.jpg")
        self.assertEqual(found.read_bytes(), b"picture")
        self.assertTrue(found.is_relative_to(self.root.resolve()))

    def test_only_jpegs_are_captures(self):
        refused_with(self, 400, paths.capture_file, self.root, "run_a", "book", "notes.txt")
        refused_with(self, 400, paths.capture_file, self.root, "run_a", "book", "spreads.json")

    def test_traversal_in_any_position_is_refused(self):
        refused_with(self, 400, paths.capture_file, self.root, "..", "book", "x.jpg")
        refused_with(self, 400, paths.capture_file, self.root, "run_a", "..", "x.jpg")
        refused_with(self, 400, paths.capture_file, self.root, "run_a", "book", "../../../secret.jpg")
        refused_with(self, 400, paths.capture_file, self.root, "run_a", "book", "..")

    def test_absolute_names_are_refused(self):
        refused_with(self, 400, paths.capture_file, self.root, "run_a", "book", str(self.secret))
        refused_with(self, 400, paths.run_dir, self.root, str(self.outside))

    def test_missing_things_are_not_found(self):
        refused_with(self, 404, paths.run_dir, self.root, "run_b")
        refused_with(self, 404, paths.video_dir, self.root, "run_a", "other-book")
        refused_with(self, 404, paths.capture_file, self.root, "run_a", "book", "sequence_0002_t0001.00s.jpg")

    def test_a_link_to_a_file_outside_the_root_is_not_served(self):
        os.symlink(self.secret, self.root / "run_a" / "book" / "captures" / "linked.jpg")
        refused_with(self, 404, paths.capture_file, self.root, "run_a", "book", "linked.jpg")

    def test_a_link_to_a_folder_outside_the_root_is_not_a_run(self):
        os.symlink(self.outside, self.root / "run_link")
        refused_with(self, 404, paths.run_dir, self.root, "run_link")
        refused_with(self, 404, paths.capture_file, self.root, "run_link", "book", "x.jpg")

    def test_a_hard_link_as_the_pipeline_makes_is_served(self):
        target = self.root / "run_a" / "book" / "captures" / "sequence_0002_t0001.00s.jpg"
        os.link(self.root / "run_a" / "book" / "captures" / "sequence_0001_t0000.73s.jpg", target)
        self.assertEqual(paths.capture_file(self.root, "run_a", "book", target.name).read_bytes(), b"picture")

    def test_the_root_itself_is_not_a_run(self):
        refused_with(self, 400, paths.run_dir, self.root, ".")


class StaticFileTests(unittest.TestCase):
    def test_only_the_listed_files(self):
        folder = Path("/somewhere/static")
        self.assertEqual(paths.static_file(folder, "app.js")[0], folder / "app.js")
        for name in ("server.py", "../server.py", "app.js.bak", "", "index.html/"):
            with self.subTest(name=name):
                refused_with(self, 404, paths.static_file, folder, name)

    def test_the_listed_files_exist(self):
        static = Config.default().static_dir
        for name in paths.STATIC_FILES:
            self.assertTrue((static / name).is_file(), name)


class UploadNameTests(unittest.TestCase):
    def test_keeps_an_ordinary_name(self):
        self.assertEqual(clean_upload_name("Einar Thomasson Canon and Canonicity IMG-0014.M4V"),
                         "Einar Thomasson Canon and Canonicity IMG-0014.m4v")

    def test_drops_any_folder_part(self):
        self.assertEqual(clean_upload_name("../../etc/cron.d/book.mp4"), "book.mp4")
        self.assertEqual(clean_upload_name("C:\\Users\\reader\\book.mov"), "book.mov")

    def test_replaces_characters_that_do_not_belong_in_a_file_name(self):
        cleaned = clean_upload_name('a<b>:"c|d?*\n.mp4')
        self.assertRegex(cleaned, r"\A[\w .,'()&+-]+\.mp4\Z")
        self.assertNotIn("\n", cleaned)

    def test_only_videos(self):
        for name in ("notes.txt", "book.mp4.exe", "book", ".mp4.sh", ""):
            with self.subTest(name=name):
                refused_with(self, 415, clean_upload_name, name)

    def test_a_name_without_letters_or_digits_still_gets_a_usable_one(self):
        self.assertEqual(clean_upload_name("---.mp4"), "video ---.mp4")


class ServerRefusalTests(unittest.TestCase):
    """The same rules, asked over HTTP the way a browser or curl would."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.config = Config.under(base)
        captures = cls.config.runs_root / "run_a" / "book" / "captures"
        captures.mkdir(parents=True)
        (captures / "sequence_0001_t0000.73s.jpg").write_bytes(b"picture")
        (cls.config.runs_root / "run_a" / "book" / "spreads.json").write_text(json.dumps(
            {"captures": [{"index": 1, "file": "sequence_0001_t0000.73s.jpg", "best_seconds": 0.73, "review": []}]}))
        (base / "secret.txt").write_text("secret")
        cls.config.video_dir.mkdir()
        cls.server = Server(App(cls.config), 0, quiet=True)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def ask(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def test_listens_on_this_machine_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_a_capture_is_served(self):
        status, body, headers = self.ask("GET", "/image/run_a/book/sequence_0001_t0000.73s.jpg")
        self.assertEqual((status, body), (200, b"picture"))
        self.assertEqual(headers["Content-Type"], "image/jpeg")

    def test_traversal_is_refused(self):
        attempts = [
            "/image/run_a/book/../../../secret.txt",
            "/image/run_a/book/..%2f..%2f..%2fsecret.txt",
            "/image/run_a/book/%2e%2e/%2e%2e/%2e%2e/secret.txt",
            "/image/..%2f..%2fsecret.txt/book/x.jpg",
            "/image/run_a/..%2f..%2fsecret.txt/x.jpg",
            "/thumb/run_a/book/..%2f..%2f..%2fsecret.txt",
            "/zip/run_a/..%2f..",
            "/zip/%2e%2e/book",
            "/static/..%2fserver.py",
            "/static/../server.py",
            "/api/runs/..%2f..",
            "/api/runs/run_a/..%2f..%2f..",
            "//etc/passwd",
            "/image/run_a/book/spreads.json",
        ]
        for path in attempts:
            with self.subTest(path=path):
                status, body, _ = self.ask("GET", path)
                self.assertIn(status, (400, 404))
                self.assertNotIn(b"secret", body)
                self.assertIn("error", json.loads(body))

    def test_other_files_of_the_project_are_not_served(self):
        for path in ("/webapp/server.py", "/static/server.py", "/CLAUDE.md", "/data/corpus/split.json", "/.git/config"):
            with self.subTest(path=path):
                self.assertEqual(self.ask("GET", path)[0], 404)

    def test_the_page_and_its_files_are_served(self):
        for path, kind in (("/", "text/html"), ("/static/app.js", "text/javascript"), ("/static/app.css", "text/css")):
            status, body, headers = self.ask("GET", path)
            self.assertEqual(status, 200)
            self.assertTrue(headers["Content-Type"].startswith(kind))
            self.assertTrue(body)

    def test_a_request_under_another_host_name_is_refused(self):
        status, _, _ = self.ask("GET", "/api/runs", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)

    def test_a_request_from_another_site_cannot_start_anything(self):
        body = json.dumps({"source": "library", "name": "x.mp4"}).encode()
        for headers in ({"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                status, _, _ = self.ask("POST", "/api/jobs", body, {"Content-Type": "application/json", **headers})
                self.assertEqual(status, 403)

    def test_a_video_that_is_not_listed_cannot_be_started(self):
        for name in ("../../etc/passwd", "/etc/passwd", "missing.mp4"):
            with self.subTest(name=name):
                body = json.dumps({"source": "library", "name": name}).encode()
                status, _, _ = self.ask("POST", "/api/jobs", body, {"Content-Type": "application/json"})
                self.assertEqual(status, 404)

    def test_an_upload_that_is_not_a_video_is_refused_and_leaves_nothing(self):
        status, _, _ = self.ask("POST", "/api/upload?name=notes.txt", b"hello")
        self.assertEqual(status, 415)
        self.assertEqual(list(self.config.uploads_dir.glob("notes*")), [])

    def test_an_upload_lands_in_the_uploads_folder_under_a_clean_name(self):
        status, body, _ = self.ask("POST", "/api/upload?name=..%2F..%2Fnew%20book.mp4", b"0123456789" * 1000)
        self.assertEqual(status, 201)
        name = json.loads(body)["video"]["name"]
        self.assertEqual(name, "new book.mp4")
        self.assertEqual((self.config.uploads_dir / name).stat().st_size, 10000)
        self.assertEqual([p.name for p in self.config.uploads_dir.iterdir()], [name])


if __name__ == "__main__":
    unittest.main()
