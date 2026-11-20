#!/usr/bin/env python3
"""The book scanner as a page in the browser, served from this PC only.

    .venv/bin/python tools/web_app.py            # then open http://127.0.0.1:8770/
    .venv/bin/python tools/web_app.py --port 8771
    .venv/bin/python tools/web_app.py --lan      # also for phones and computers on the home network; no login
    .venv/bin/python tools/web_app.py --lan --host-name books.example.net

Pick a video, watch it being scanned, look through the pictures and the
page-number verdict, and browse earlier runs. See ``webapp/__init__.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
