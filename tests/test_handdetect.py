import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

from auto_camera import handdetect, handnet

WIDTH, HEIGHT, FPS = 480, 270, 11.0


def fake_detections(frames: int, seed: int = 0):
    """What ``detect_range`` returns for ``frames`` frames: some with a hand, some without, skin in a corner."""
    rng = np.random.default_rng(seed)
    landmarks = np.full((frames, handdetect.MAX_HANDS, 21, 2), np.nan, dtype=np.float32)
    scores = np.zeros((frames, handdetect.MAX_HANDS), dtype=np.float32)
    skins = []
    for index in range(frames):
        if index % 2 == 0:
            landmarks[index, 0] = rng.uniform(0.1, 0.9, (21, 2))
            scores[index, 0] = rng.uniform(0.5, 1.0)
        skin = np.zeros((HEIGHT, WIDTH), dtype=bool)
        skin[: 10 + index, WIDTH - 40:] = True
        skins.append(np.packbits(skin))
    return landmarks, scores, np.array(skins, dtype=np.uint8), WIDTH, HEIGHT, FPS


def old_writer(out_dir: Path, slug: str, landmarks, scores, skins, width, height, fps) -> None:
    """How the file was written before the code moved to auto_camera/handdetect.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{slug}.partial.npz"
    np.savez_compressed(tmp, landmarks=landmarks.astype(np.float16), scores=scores.astype(np.float16),
                        skin=skins, width=width, height=height, fps=fps, version=2)
    tmp.replace(out_dir / f"{slug}.npz")


def members(path: Path) -> dict[str, bytes]:
    """The arrays inside an .npz, as bytes (the zip's own dates differ from one write to the next)."""
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_is_byte_for_byte_what_the_old_tool_wrote(self):
        data = fake_detections(7)
        with mock.patch.object(handdetect, "detect_range", return_value=data):
            handdetect.detect("book", self.dir / "proxies", self.dir / "new")
        old_writer(self.dir / "old", "book", *data)
        self.assertEqual(members(self.dir / "new" / "book.npz"), members(self.dir / "old" / "book.npz"))
        self.assertFalse((self.dir / "new" / "book.partial.npz").exists())

    def test_handnet_reads_what_is_written(self):
        landmarks, scores, skins, width, height, fps = fake_detections(5)
        with mock.patch.object(handdetect, "detect_range", return_value=(landmarks, scores, skins, width, height, fps)):
            handdetect.detect("book", self.dir, self.dir)
        found = handnet.load("book", self.dir)
        self.assertIsNotNone(found)
        self.assertEqual(len(found), 5)
        self.assertEqual(found.shape, (HEIGHT, WIDTH))
        np.testing.assert_allclose(found.landmarks, landmarks.astype(np.float16).astype(np.float32), equal_nan=True)
        np.testing.assert_allclose(found.scores, scores.astype(np.float16).astype(np.float32))
        for index in range(5):
            expected = np.unpackbits(skins[index])[: HEIGHT * WIDTH].reshape(HEIGHT, WIDTH).astype(bool)
            np.testing.assert_array_equal(found.skin(index), expected)
        with np.load(self.dir / "book.npz") as stored:
            self.assertEqual(sorted(stored.files), ["fps", "height", "landmarks", "scores", "skin", "version", "width"])
            self.assertEqual(stored["landmarks"].dtype, np.float16)
            self.assertEqual(stored["scores"].dtype, np.float16)
            self.assertEqual(stored["skin"].dtype, np.uint8)
            self.assertEqual(int(stored["version"]), handdetect.VERSION)

    def test_stretches_run_in_a_pool_come_back_in_order(self):
        whole = fake_detections(9, seed=3)

        def piece(slug, start, stop, proxy_dir):
            return whole[0][start:stop], whole[1][start:stop], whole[2][start:stop], WIDTH, HEIGHT, FPS

        with mock.patch.object(handdetect, "detect_range", side_effect=piece), \
                mock.patch.object(handdetect, "frame_count", return_value=9), ThreadPoolExecutor(3) as pool:
            handdetect.detect("book", self.dir, self.dir / "split", parts=3, pool=pool)
        old_writer(self.dir / "whole", "book", *whole)
        self.assertEqual(members(self.dir / "split" / "book.npz"), members(self.dir / "whole" / "book.npz"))

    def test_one_part_with_a_pool_runs_in_the_pool(self):
        data = fake_detections(3)
        pool = mock.MagicMock()
        pool.submit.return_value.result.return_value = data
        handdetect.detect("book", self.dir, self.dir, parts=1, pool=pool)
        pool.submit.assert_called_once()
        self.assertEqual(len(handnet.load("book", self.dir)), 3)


class IsolationTests(unittest.TestCase):
    def test_module_imports_nothing_else_of_the_project(self):
        import ast

        tree = ast.parse(Path(handdetect.__file__).read_text())
        relative = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level}
        self.assertEqual(relative, {"paths"})
        package = Path(handdetect.__file__).with_name("__init__.py").read_text()
        self.assertNotIn("import", "\n".join(line for line in package.splitlines() if not line.lstrip().startswith(("#", '"'))))


if __name__ == "__main__":
    unittest.main()
