"""The HTTP side: the page, a small JSON API, and pictures out of run folders.

Listens on 127.0.0.1 only, unless it is started with ``--lan``; then it
listens on every interface and answers only clients with a private address
that ask for it by one of this machine's own names (see ``network.py``).
Pictures are served only from run folders under ``Config.runs_root`` (see
``paths.py``); the page's own files come from a fixed list. The server holds
no video or image data in memory: uploads are written, and pictures and ZIP
files sent, a chunk at a time.

    GET  /                                   the page
    GET  /api/videos                         videos that can be scanned
    POST /api/jobs {source, name}            start a scan (409 while one is going)
    GET  /api/job                            the latest scan and how far it has got
    POST /api/jobs/<id>/cancel               stop it
    POST /api/upload/check {name, size}      would this file be accepted?
    POST /api/upload?name=...                the file itself, as the request body
    GET  /api/runs                           run folders
    GET  /api/runs/<run>                     the videos of one run, with verdicts
    GET  /api/runs/<run>/<video>             one video's result: verdict, counts, pictures, gaps
    GET  /thumb/<run>/<video>/<file>         small picture for the grid
    GET  /image/<run>/<video>/<file>         the full-size picture
    GET  /zip/<run>/<video>                  all pictures of one video, as a ZIP
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import signal
import socket
import sys
import threading
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import DEFAULT_PORT, Config, network, paths, runs, thumbs
from .jobs import Busy, JobRunner, default_command
from .paths import Refused
from .videos import Videos

HOST = "127.0.0.1"
LOCAL_NAMES = {"127.0.0.1", "localhost", "[::1]"}
MAX_JSON_BODY = 64 * 1024
SEND_CHUNK = 256 * 1024
PAGE_POLICY = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
               "object-src 'none'; base-uri 'none'; form-action 'none'")


class App:
    """What the handlers share: the folders, the scan runner and the video list."""

    def __init__(self, config: Config, command=default_command):
        self.config = config
        self.jobs = JobRunner(config, command)
        self.videos = Videos(config)


class Server(ThreadingHTTPServer):
    """Listens on 127.0.0.1 and nothing else, unless ``lan`` is set.

    With ``lan`` it listens on every interface, IPv4 and IPv6, turns away
    every client whose address is not private, and answers only to this
    machine's own names and addresses and to ``host_names``.
    """

    daemon_threads = True

    def __init__(self, app: App, port: int, *, quiet: bool = False, lan: bool = False, host_names=(), bind: str | None = None):
        self.app = app
        self.quiet = quiet              # no request log; the tests use this
        self.lan = lan
        self.client_allowed = network.client_allowed
        self.hosts: network.HostRule | None = None
        self.turned_away = 0
        address = bind or ("::" if lan else HOST)       # ``bind`` lets the tests use home-network mode without leaving loopback
        self.address_family = socket.AF_INET6 if ":" in address else socket.AF_INET
        try:
            super().__init__((address, port), Handler)
        except OSError as error:
            if not lan or bind or error.errno not in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                raise
            self.address_family = socket.AF_INET        # a machine without IPv6
            super().__init__(("0.0.0.0", port), Handler)
        if lan:
            self.hosts = network.HostRule(self.server_address[1], host_names)

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)      # one socket for IPv4 clients too
        super().server_bind()


class ChunkedWriter:
    """Lets ``zipfile`` write straight into an HTTP response of unknown length.

    It has ``write`` and ``flush`` and deliberately no ``tell`` or ``seek``,
    which is how ``zipfile`` knows to write a stream. Small writes are
    gathered so the response is not sent a few bytes at a time.
    """

    def __init__(self, raw, chunked: bool):
        self.raw = raw
        self.chunked = chunked
        self.pending = bytearray()

    def write(self, data) -> int:
        self.pending += data
        if len(self.pending) >= SEND_CHUNK:
            self._send()
        return len(data)

    def flush(self) -> None:
        pass

    def _send(self) -> None:
        if not self.pending:
            return
        if self.chunked:
            self.raw.write(b"%x\r\n" % len(self.pending))
            self.raw.write(self.pending)
            self.raw.write(b"\r\n")
        else:
            self.raw.write(self.pending)
        self.pending = bytearray()

    def finish(self) -> None:
        self._send()
        if self.chunked:
            self.raw.write(b"0\r\n\r\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "BookScanner"
    sys_version = ""
    timeout = 120                      # an idle or stalled connection is dropped

    @property
    def app(self) -> App:
        return self.server.app

    # ---- plumbing ----------------------------------------------------------

    def handle(self) -> None:
        if self.server.lan and not self.server.client_allowed(self.client_address[0]):
            self._turn_away()
            return
        super().handle()

    def _turn_away(self) -> None:
        """Home-network mode: a client from outside gets 403 and nothing else. What it asked for is not looked at."""
        body = json.dumps({"error": "The book scanner only answers devices on the home network."}).encode("utf-8")
        try:
            self.connection.settimeout(5)
            for _ in range(200):        # let it finish saying what it wants, so that it sees the answer and not a reset
                if self.rfile.readline(65537) in (b"\r\n", b"\n", b""):
                    break
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Type: application/json; charset=utf-8\r\n"
                             b"Content-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body))
        except OSError:
            pass
        self._linger()
        self.server.turned_away += 1
        if not self.server.quiet and (self.server.turned_away <= 5 or self.server.turned_away % 100 == 0):
            print(f"turned away a client outside the home network: {self.client_address[0]} "
                  f"({self.server.turned_away} so far)", file=sys.stderr, flush=True)

    def do_GET(self) -> None:
        self._dispatch(self._get)

    def do_POST(self) -> None:
        self._dispatch(self._post)

    def _dispatch(self, route) -> None:
        self._begun = False
        self._body_pending = self.command == "POST"
        try:
            self._check_host()
            parts = paths.split_url_path(self.path)
            query = parse_qs(urlsplit(self.path).query)
            route(parts, query)
        except Refused as refusal:
            self._refuse(refusal.status, refusal.message)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:
            traceback.print_exc(file=sys.stderr)
            self._refuse(500, "Something went wrong on this computer. Try again.")
        if self._body_pending and self.close_connection:
            self._linger()

    def _linger(self) -> None:
        """Before hanging up on a request whose body was not read, let the answer reach the client.

        Closing a socket that still holds unread data makes the system send a
        reset, and a reset can overtake the answer. So stop sending, then read
        and drop a bounded amount of what is still arriving.
        """
        try:
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_WR)
            self.connection.settimeout(0.5)
            for _ in range(32):
                if not self.connection.recv(65536):
                    break
        except OSError:
            pass

    def _check_host(self) -> None:
        """Answer only when addressed as this machine (a page elsewhere must not reach this through a made-up name)."""
        host = self.headers.get("Host", "")
        if self.server.lan:
            if not self.server.hosts.allows(host):
                raise Refused(403, "The book scanner does not answer to that address. "
                                   "Use one of the addresses it showed when it was started.")
            return
        name = host[:host.index("]") + 1] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0]
        if name.lower() not in LOCAL_NAMES:
            raise Refused(403, "This page only works on this PC, at http://127.0.0.1.")

    def _check_same_site(self) -> None:
        """A request that changes something must come from this page, not from another site open in the browser."""
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{self.headers.get('Host', '')}":
            raise Refused(403, "That request did not come from this page.")
        if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            raise Refused(403, "That request did not come from this page.")

    def _begin(self, status: int, content_type: str, length: int | None, *, cache: str = "no-store", extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Referrer-Policy", "no-referrer")
        if content_type.startswith("text/html"):
            self.send_header("Content-Security-Policy", PAGE_POLICY)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        if self._body_pending:
            # the request body was not read, so the connection cannot be used again
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self._begun = True

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._begin(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _refuse(self, status: int, message: str) -> None:
        if self._begun:                 # part of an answer has gone out; all that is left is to hang up
            self.close_connection = True
            return
        try:
            self._json({"error": message}, status)
        except OSError:
            self.close_connection = True

    def _file(self, path: Path, content_type: str, cache: str) -> None:
        try:
            handle = open(path, "rb")
        except OSError:
            raise Refused(404, "That was not found.") from None
        with handle:
            self._begin(200, content_type, os.fstat(handle.fileno()).st_size, cache=cache)
            shutil.copyfileobj(handle, self.wfile, SEND_CHUNK)

    def _body_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise Refused(411, "That request was not complete.") from None
        if length < 0 or length > MAX_JSON_BODY:
            raise Refused(413, "That request was too large.")
        raw = self.rfile.read(length)
        self._body_pending = False
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            raise Refused(400, "That request could not be read.") from None
        if not isinstance(data, dict):
            raise Refused(400, "That request could not be read.")
        return data

    def log_request(self, code="-", size="-") -> None:
        """Log what changes something, what is refused and ZIP downloads; not every picture and progress check."""
        status = getattr(code, "value", code)
        routine = self.command == "GET" and isinstance(status, int) and status < 400 and not self.path.startswith("/zip/")
        if not routine:
            super().log_request(code, size)

    def log_message(self, format, *args) -> None:
        if not self.server.quiet:
            super().log_message(format, *args)

    # ---- GET ---------------------------------------------------------------

    def _get(self, parts: list[str], query: dict) -> None:
        if not parts:
            return self._static("index.html")
        head, rest = parts[0], parts[1:]
        if head == "static" and len(rest) == 1:
            return self._static(rest[0])
        if head == "api":
            return self._api_get(rest)
        root = self.app.config.runs_root
        lasting = "private, max-age=604800, immutable" if "v" in query else "private, max-age=60"
        if head == "image" and len(rest) == 3:
            return self._file(paths.capture_file(root, *rest), "image/jpeg", lasting)
        if head == "thumb" and len(rest) == 3:
            source = paths.capture_file(root, *rest)
            try:
                small = thumbs.thumbnail(self.app.config, source, *rest)
            except Exception as error:
                print(f"thumbnail failed for {source}: {error}", file=sys.stderr)
                raise Refused(500, "That picture could not be made smaller.") from None
            return self._file(small, "image/jpeg", lasting)
        if head == "zip" and len(rest) == 2:
            return self._zip(*rest)
        raise Refused(404, "That was not found.")

    def _static(self, name: str) -> None:
        path, content_type = paths.static_file(self.app.config.static_dir, name)
        self._file(path, content_type, "no-cache")

    def _api_get(self, rest: list[str]) -> None:
        app, root = self.app, self.app.config.runs_root
        live = app.jobs.live()
        if rest == ["videos"]:
            names = runs.run_names(root)
            videos = app.videos.listing()
            for video in videos:
                video["earlier"] = runs.latest_result(root, video["slug"], names) if video["slug"] else None
            return self._json({"videos": videos, "job": self._job_snapshot()})
        if rest == ["job"]:
            return self._json({"job": self._job_snapshot()})
        if len(rest) == 2 and rest[0] == "jobs":
            job = app.jobs.get(rest[1])
            if job is None:
                raise Refused(404, "That scan is not known any more.")
            return self._json({"job": job.snapshot()})
        if rest == ["runs"]:
            return self._json({"runs": runs.runs_view(root, live=live)})
        if len(rest) == 2 and rest[0] == "runs":
            return self._json(runs.run_view(root, rest[1], live_slug=live[1] if live and live[0] == rest[1] else None))
        if len(rest) == 3 and rest[0] == "runs":
            return self._json(runs.video_view(root, rest[1], rest[2], live=live == (rest[1], rest[2])))
        raise Refused(404, "That was not found.")

    def _job_snapshot(self) -> dict | None:
        job = self.app.jobs.latest()
        return job.snapshot() if job else None

    def _zip(self, run: str, slug: str) -> None:
        """All pictures of one video in page order, stored (they are JPEGs already) and sent as they are read."""
        root = self.app.config.runs_root
        folder = paths.video_dir(root, run, slug)
        spreads = runs.read_json(folder / "spreads.json") or {}
        listed = spreads.get("captures") if isinstance(spreads.get("captures"), list) else []
        files = []
        for capture in listed:
            try:
                files.append(paths.capture_file(root, run, slug, capture.get("file") if isinstance(capture, dict) else None))
            except Refused:
                continue
        if not files:
            raise Refused(404, "There are no pictures to download for that video.")
        chunked = self.request_version != "HTTP/1.0"
        extra = {"Content-Disposition": f'attachment; filename="{slug}.zip"'}
        if chunked:
            extra["Transfer-Encoding"] = "chunked"
        else:
            extra["Connection"] = "close"
            self.close_connection = True
        self._begin(200, "application/zip", None, extra=extra)
        out = ChunkedWriter(self.wfile, chunked)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
            for path in files:
                info = zipfile.ZipInfo.from_file(path, f"{slug}/{path.name}")
                info.compress_type = zipfile.ZIP_STORED
                with open(path, "rb") as source, archive.open(info, "w") as target:
                    shutil.copyfileobj(source, target, SEND_CHUNK)
        out.finish()

    # ---- POST --------------------------------------------------------------

    def _post(self, parts: list[str], query: dict) -> None:
        self._check_same_site()
        app = self.app
        if parts == ["api", "jobs"]:
            body = self._body_json()
            video = app.videos.find(str(body.get("source", "")), str(body.get("name", "")))
            try:
                job = app.jobs.start(video, video.name, str(body.get("source")))
            except Busy as busy:
                return self._json({"error": busy.message, "job": busy.job.snapshot() if busy.job else None}, 409)
            except ValueError as error:
                raise Refused(400, str(error)) from None
            return self._json({"job": job.snapshot()}, 201)
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
            self._body_json()
            job = app.jobs.cancel(parts[2])
            if job is None:
                raise Refused(404, "That scan is not known any more.")
            return self._json({"job": job.snapshot()})
        if parts == ["api", "upload", "check"]:
            body = self._body_json()
            return self._json({"ok": True, "name": app.videos.check_upload(body.get("name"), body.get("size"))})
        if parts == ["api", "upload"]:
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise Refused(411, "The size of the video was not given.") from None
            self.connection.settimeout(60)
            video = app.videos.receive(query.get("name", [""])[0], length, self.rfile)
            self._body_pending = False
            return self._json({"video": video}, 201)
        raise Refused(404, "That was not found.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The book scanner as a page in the browser. On this PC only, unless --lan is given.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--lan", action="store_true",
                        help="also answer phones and computers on the home network. There is no login: "
                             "anyone on that network can see the scans and start one")
    parser.add_argument("--host-name", action="append", default=[], metavar="NAME",
                        help="with --lan: one more name this PC is reached by, such as a DNS name for its home address "
                             "(may be given more than once)")
    args = parser.parse_args(argv)
    if args.host_name and not args.lan:
        parser.error("--host-name only applies together with --lan")
    try:
        names = [network.clean_host_name(name) for name in args.host_name]
    except ValueError as error:
        parser.error(str(error))
    app = App(Config.default())
    try:
        server = Server(app, args.port, lan=args.lan, host_names=names)
    except OSError as error:
        print(f"Could not start on port {args.port}: {error.strerror or error}. "
              f"If the book scanner is already running, use that one; otherwise try --port with another number.", file=sys.stderr)
        return 1
    if args.lan:
        print("\n".join(network.startup_lines(args.port, server.hosts.refresh(), names)), flush=True)
    else:
        print(f"Book scanner is running. Open http://{HOST}:{args.port}/ in a browser on this PC. Ctrl+C stops it.", flush=True)

    def stop(*_) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    finally:
        app.jobs.shutdown()
        server.server_close()
    print("Book scanner stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
