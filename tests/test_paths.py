import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from auto_camera import paths

REPO = Path(__file__).resolve().parents[1]
MODULES = {
    "auto_camera.proxy": "DEFAULT_PROXY_DIR",
    "auto_camera.analysis": "DEFAULT_ANALYSIS_DIR",
    "auto_camera.hands": "DEFAULT_HANDS_DIR",
    "auto_camera.handnet": "DEFAULT_HANDNET_DIR",
    "auto_camera.export": "DEFAULT_STILLS_DIR",
    "auto_camera.audit": "OCR_CACHE",
    "auto_camera.handdetect": "OUT_DIR",
}


@contextmanager
def environment(**values):
    """os.environ with the two overrides removed and ``values`` set, for the length of the block."""
    with mock.patch.dict(os.environ):
        for name in (paths.CACHE_ENV, paths.MODELS_ENV, "LOCALAPPDATA"):
            os.environ.pop(name, None)
        os.environ.update(values)
        yield


def constants_in_fresh_process(**env) -> dict:
    """The cache constants as a new process sees them on import, under ``env``."""
    code = ("import json, importlib; names = %r; out = {}\n"
            "for module, name in names.items(): out[module] = str(getattr(importlib.import_module(module), name))\n"
            "import auto_camera.pipeline as p; out['models'] = [str(m) for m in p.HAND_MODELS]\n"
            "print(json.dumps(out))") % (MODULES,)
    clean = {k: v for k, v in os.environ.items() if k not in (paths.CACHE_ENV, paths.MODELS_ENV)}
    clean.update(env)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=clean, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


class FromSourceTests(unittest.TestCase):
    def test_defaults_are_the_repository_folders(self):
        with environment():
            self.assertEqual(paths.cache_root(), REPO / "data" / "cache")
            self.assertEqual(paths.models_dir(), REPO / "data" / "models")
            self.assertIsNone(paths.bundled_tools_dir())
            self.assertFalse(paths.frozen())

    def test_the_pipeline_keeps_its_old_cache_folders(self):
        found = constants_in_fresh_process()
        cache = REPO / "data" / "cache"
        self.assertEqual(found["auto_camera.proxy"], str(cache / "proxies"))
        self.assertEqual(found["auto_camera.analysis"], str(cache / "analysis"))
        self.assertEqual(found["auto_camera.hands"], str(cache / "hands"))
        self.assertEqual(found["auto_camera.handnet"], str(cache / "handnet"))
        self.assertEqual(found["auto_camera.export"], str(cache / "stills"))
        self.assertEqual(found["auto_camera.audit"], str(cache / "ocr_v4"))
        self.assertEqual(found["auto_camera.handdetect"], str(cache / "handnet"))
        self.assertEqual(found["models"], [str(REPO / "data" / "models" / "hand_landmarker.task"),
                                           str(REPO / "data" / "models" / "selfie_multiclass_256x256.tflite")])


class OverrideTests(unittest.TestCase):
    def test_environment_moves_cache_and_models(self):
        with environment(AUTO_CAMERA_CACHE="/somewhere/cache", AUTO_CAMERA_MODELS="/elsewhere/models"):
            self.assertEqual(paths.cache_root(), Path("/somewhere/cache"))
            self.assertEqual(paths.models_dir(), Path("/elsewhere/models"))

    def test_environment_wins_over_the_frozen_defaults(self):
        with environment(AUTO_CAMERA_CACHE="/somewhere/cache", AUTO_CAMERA_MODELS="/elsewhere/models"), \
                mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(sys, "_MEIPASS", "/bundle", create=True):
            self.assertEqual(paths.cache_root(), Path("/somewhere/cache"))
            self.assertEqual(paths.models_dir(), Path("/elsewhere/models"))

    def test_every_cache_constant_follows_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            found = constants_in_fresh_process(AUTO_CAMERA_CACHE=tmp, AUTO_CAMERA_MODELS=str(Path(tmp) / "m"))
        for module, value in found.items():
            if module == "models":
                self.assertEqual(value, [str(Path(tmp) / "m" / "hand_landmarker.task"), str(Path(tmp) / "m" / "selfie_multiclass_256x256.tflite")])
            else:
                self.assertTrue(value.startswith(tmp), (module, value))


class FrozenTests(unittest.TestCase):
    def test_frozen_defaults(self):
        with tempfile.TemporaryDirectory() as bundle, environment(LOCALAPPDATA=r"C:\Users\reader\AppData\Local"), \
                mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(sys, "_MEIPASS", bundle, create=True):
            self.assertTrue(paths.frozen())
            self.assertEqual(paths.cache_root(), Path(r"C:\Users\reader\AppData\Local") / "AutoCameraBookScanner" / "cache")
            self.assertEqual(paths.models_dir(), Path(bundle) / "models")
            self.assertEqual(paths.bundled_tools_dir(), Path(bundle) / "ffmpeg")

    def test_frozen_without_localappdata_still_has_a_cache(self):
        with environment(), mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(sys, "_MEIPASS", "/b", create=True):
            root = paths.cache_root()
        self.assertEqual(root.name, "cache")
        self.assertEqual(root.parent.name, "AutoCameraBookScanner")


if __name__ == "__main__":
    unittest.main()
