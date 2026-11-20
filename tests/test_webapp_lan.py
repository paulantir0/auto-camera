"""Home-network mode of the browser app: which clients and which Host headers are answered.

The servers here stay on loopback (``bind="127.0.0.1"``); only the rules of
home-network mode are switched on.
"""

import contextlib
import http.client
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

from webapp import Config, network
from webapp.server import App, Server, main


class ClientAddressTests(unittest.TestCase):
    def test_this_machine_and_private_networks_are_served(self):
        for address in ("127.0.0.1", "127.8.9.10", "10.0.0.7", "10.255.255.254", "172.16.0.1", "172.31.255.254",
                        "192.168.1.87", "192.168.255.1", "169.254.10.10", "100.64.0.1", "100.101.102.103", "100.127.255.254"):
            with self.subTest(address=address):
                self.assertTrue(network.client_allowed(address))

    def test_everything_else_on_ipv4_is_refused(self):
        for address in ("8.8.8.8", "1.1.1.1", "9.255.255.255", "11.0.0.1", "172.15.255.255", "172.32.0.1", "192.167.255.255",
                        "192.169.0.1", "169.253.1.1", "169.255.0.1", "100.63.255.255", "100.128.0.1", "126.255.255.255",
                        "128.0.0.1", "0.0.0.0", "255.255.255.255", "224.0.0.1", "203.0.113.9"):
            with self.subTest(address=address):
                self.assertFalse(network.client_allowed(address))

    def test_ipv6_this_machine_unique_local_and_link_local_are_served(self):
        for address in ("::1", "fc00::1", "fd7a:115c:a1e0::1234:5678", "fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", "fe80::1",
                        "fe80::fdda:808c:e4ab:a264", "febf::1"):
            with self.subTest(address=address):
                self.assertTrue(network.client_allowed(address))

    def test_a_zone_after_a_link_local_address_does_not_matter(self):
        self.assertTrue(network.client_allowed("fe80::fdda:808c:e4ab:a264%enp42s0"))
        self.assertTrue(network.client_allowed("fe80::1%3"))

    def test_public_and_other_ipv6_is_refused(self):
        for address in ("2606:4700:4700::1111", "2001:4860:4860::8888", "::", "fbff::1", "fe00::1", "fec0::1", "ff02::1",
                        "2001:db8::1", "::2"):
            with self.subTest(address=address):
                self.assertFalse(network.client_allowed(address))

    def test_an_ipv4_client_seen_through_an_ipv6_socket_is_judged_as_ipv4(self):
        for address in ("::ffff:192.168.1.5", "::ffff:127.0.0.1", "::ffff:10.1.2.3", "::ffff:100.101.102.103", "::ffff:c0a8:0105"):
            with self.subTest(address=address):
                self.assertTrue(network.client_allowed(address))
        for address in ("::ffff:8.8.8.8", "::ffff:172.32.0.1", "::ffff:100.128.0.1", "::ffff:0808:0808"):
            with self.subTest(address=address):
                self.assertFalse(network.client_allowed(address))

    def test_a_private_ipv4_address_wrapped_any_other_way_is_refused(self):
        # 6to4, NAT64 and the old "IPv4-compatible" form of 192.168.1.1: none is how a local client appears
        for address in ("2002:c0a8:0101::1", "64:ff9b::c0a8:101", "::192.168.1.1", "::c0a8:101"):
            with self.subTest(address=address):
                self.assertFalse(network.client_allowed(address))

    def test_what_is_not_an_address_is_refused(self):
        for address in ("", "localhost", "192.168.1", "192.168.1.256", "192.168.1.5/24", "192.168.1.5:8770", "fe80::zz", None, 7,
                        ("192.168.1.5", 8770), b"192.168.1.5"):
            with self.subTest(address=address):
                self.assertFalse(network.client_allowed(address))


class HostHeaderTests(unittest.TestCase):
    def rule(self, addresses=("192.168.1.166", "192.168.1.87", "100.101.102.103", "fd7a:115c:a1e0::1234:5678"), names=("books.example.net",),
             **more) -> network.HostRule:
        return network.HostRule(8770, names, hostname="reader-desk", discover=lambda: list(addresses), **more)

    def test_this_machine_by_name_and_by_each_of_its_addresses(self):
        rule = self.rule()
        for host in ("localhost:8770", "localhost", "127.0.0.1:8770", "[::1]:8770", "reader-desk:8770", "reader-desk.local:8770",
                     "192.168.1.166:8770", "192.168.1.87:8770", "100.101.102.103:8770", "[fd7a:115c:a1e0::1234:5678]:8770",
                     "[FD7A:115C:A1E0:0:0:0:1234:5678]:8770", "books.example.net:8770", "books.example.net",
                     "Reader-Desk.Local:8770", "BOOKS.example.net.:8770"):
            with self.subTest(host=host):
                self.assertTrue(rule.allows(host))

    def test_any_other_name_or_address_is_refused(self):
        rule = self.rule()
        for host in ("evil.example", "evil.example:8770", "192.168.1.166.evil.example:8770", "reader-desk.evil.example:8770",
                     "books.example.net.evil.example", "localhost.evil.example:8770", "192.168.1.77:8770", "10.0.0.1:8770",
                     "8.8.8.8:8770", "[2606:4700:4700::1111]:8770", "[fd00::9]:8770", "reader-desk.lan:8770", "example.net:8770",
                     "", " ", "192.168.1.166:8770:1", "192.168.1.166:port", "[fd7a:115c:a1e0::1234:5678", "fd7a:115c:a1e0::1234:5678",
                     "http://192.168.1.166:8770", "192.168.1.166:8770/", "192.168.1.166 :8770", "user@192.168.1.166:8770", None):
            with self.subTest(host=host):
                self.assertFalse(rule.allows(host))

    def test_a_port_in_the_header_must_be_the_servers(self):
        rule = self.rule()
        self.assertTrue(rule.allows("192.168.1.166:8770"))
        self.assertFalse(rule.allows("192.168.1.166:9999"))
        self.assertFalse(rule.allows("localhost:80"))
        self.assertFalse(rule.allows("books.example.net:443"))

    def test_without_extra_names_only_the_machines_own(self):
        rule = self.rule(names=())
        self.assertFalse(rule.allows("books.example.net:8770"))
        self.assertTrue(rule.allows("reader-desk:8770"))

    def test_a_fully_qualified_host_name_also_gives_the_short_one(self):
        rule = network.HostRule(8770, hostname="reader-desk.home.example", discover=lambda: [])
        for host in ("reader-desk.home.example:8770", "reader-desk:8770", "reader-desk.local:8770"):
            self.assertTrue(rule.allows(host), host)
        self.assertFalse(rule.allows("home.example:8770"))

    def test_an_address_the_machine_gets_later_is_found_when_it_is_first_asked_for(self):
        addresses = ["192.168.1.166"]
        rule = network.HostRule(8770, hostname="reader-desk", discover=lambda: list(addresses), refresh_seconds=0.0)
        self.assertFalse(rule.allows("192.168.1.90:8770"))
        addresses.append("192.168.1.90")
        self.assertTrue(rule.allows("192.168.1.90:8770"))
        addresses.remove("192.168.1.166")
        rule.refresh()
        self.assertFalse(rule.allows("192.168.1.166:8770"))

    def test_addresses_are_not_looked_up_again_for_every_unknown_host(self):
        calls = []
        rule = network.HostRule(8770, hostname="reader-desk", discover=lambda: calls.append(1) or ["192.168.1.166"], refresh_seconds=60.0)
        for _ in range(20):
            self.assertFalse(rule.allows("192.168.1.90:8770"))
            self.assertFalse(rule.allows("evil.example:8770"))
        self.assertEqual(len(calls), 1)

    def test_a_public_address_of_this_machine_is_never_a_host(self):
        rule = self.rule(addresses=("192.168.1.166", "2606:4700:4700::1111", "203.0.113.9"))
        self.assertFalse(rule.allows("[2606:4700:4700::1111]:8770"))
        self.assertFalse(rule.allows("203.0.113.9:8770"))

    def test_names_given_on_the_command_line_must_be_plain_names(self):
        self.assertEqual(network.clean_host_name(" Books.Example.NET. "), "books.example.net")
        for name in ("http://books.example.net", "books.example.net:8770", "books.example.net/path", "", "a b", "*.example.net", "-x.net"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    network.clean_host_name(name)

    def test_this_machines_own_addresses_are_private_and_not_loopback(self):
        for text in network.own_addresses():
            address = network.parse_address(text)
            self.assertTrue(network.is_private(address), text)
            self.assertFalse(address.is_loopback, text)


class StartUpTextTests(unittest.TestCase):
    def test_the_addresses_to_open_one_per_line_and_that_there_is_no_login(self):
        lines = network.startup_lines(8770, ["192.168.1.166", "192.168.1.87", "100.101.102.103", "fe80::1", "fd7a:115c:a1e0::1234:5678"], [])
        self.assertIn("  http://192.168.1.166:8770/", lines)
        self.assertIn("  http://192.168.1.87:8770/", lines)
        self.assertIn("  http://100.101.102.103:8770/   (over Tailscale)", lines)
        self.assertEqual(sum(line.startswith("  http://") for line in lines), 3)
        self.assertEqual(lines[-1], "There is no login in this mode: anyone on the home network can see the scans and start one.")
        self.assertIn("On this PC itself: http://127.0.0.1:8770/", lines)

    def test_extra_names_are_listed_too(self):
        lines = network.startup_lines(8771, ["192.168.1.166"], ["books.example.net"])
        self.assertEqual([line for line in lines if line.startswith("  ")], ["  http://192.168.1.166:8771/", "  http://books.example.net:8771/"])

    def test_a_machine_with_only_ipv6_at_home(self):
        lines = network.startup_lines(8770, ["fe80::1", "fd00::5"], [])
        self.assertEqual([line for line in lines if line.startswith("  ")], ["  http://[fd00::5]:8770/"])

    def test_no_address_at_all(self):
        lines = network.startup_lines(8770, [], [])
        self.assertIn("No home-network address was found on this PC. Is it connected to the network?", lines)
        self.assertEqual(lines[-1], "There is no login in this mode: anyone on the home network can see the scans and start one.")


class HomeNetworkServerTests(unittest.TestCase):
    """The rules as the server applies them, asked over HTTP."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.config = Config.under(Path(cls.tmp.name))
        cls.config.video_dir.mkdir(parents=True)
        cls.config.runs_root.mkdir()
        cls.app = App(cls.config)
        cls.server = Server(cls.app, 0, quiet=True, lan=True, host_names=["books.example.net"], bind="127.0.0.1")
        cls.port = cls.server.server_address[1]
        cls.server.hosts = network.HostRule(cls.port, ["books.example.net"], hostname="reader-desk", discover=lambda: ["192.168.1.166"])
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.plain = Server(App(cls.config), 0, quiet=True)
        cls.plain_port = cls.plain.server_address[1]
        threading.Thread(target=cls.plain.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.server, cls.plain):
            server.shutdown()
            server.server_close()
        cls.tmp.cleanup()

    def tearDown(self):
        self.server.client_allowed = network.client_allowed

    def ask(self, method: str, path: str, *, host: str | None = None, body: bytes | None = None, headers: dict | None = None, port=None):
        connection = http.client.HTTPConnection("127.0.0.1", port or self.port, timeout=10)
        try:
            sent = dict(headers or {})
            if host is not None:
                sent["Host"] = host
            connection.request(method, path, body=body, headers=sent)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_without_the_mode_the_server_is_on_loopback_and_knows_no_other_host(self):
        self.assertEqual(self.plain.server_address[0], "127.0.0.1")
        self.assertFalse(self.plain.lan)
        self.assertIsNone(self.plain.hosts)
        self.assertEqual(self.ask("GET", "/api/videos", port=self.plain_port)[0], 200)
        self.assertEqual(self.ask("GET", "/api/videos", host=f"192.168.1.166:{self.plain_port}", port=self.plain_port)[0], 403)
        self.assertEqual(self.ask("GET", "/api/videos", host=f"reader-desk:{self.plain_port}", port=self.plain_port)[0], 403)

    def test_an_accepted_host_is_answered(self):
        for host in (f"192.168.1.166:{self.port}", f"reader-desk.local:{self.port}", f"localhost:{self.port}", f"books.example.net:{self.port}"):
            with self.subTest(host=host):
                status, body = self.ask("GET", "/api/videos", host=host)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["videos"], [])

    def test_a_foreign_host_is_refused(self):
        for host in ("evil.example", f"evil.example:{self.port}", f"192.168.1.77:{self.port}", "192.168.1.166:1"):
            with self.subTest(host=host):
                status, body = self.ask("GET", "/api/videos", host=host)
                self.assertEqual(status, 403)
                self.assertIn("error", json.loads(body))

    def test_a_change_must_come_from_the_page_at_the_host_that_was_accepted(self):
        host = f"192.168.1.166:{self.port}"
        check = json.dumps({"name": "book.mp4", "size": 1000}).encode()
        json_type = {"Content-Type": "application/json"}
        self.assertEqual(self.ask("POST", "/api/upload/check", host=host, body=check, headers={**json_type, "Origin": f"http://{host}"})[0], 200)
        self.assertEqual(self.ask("POST", "/api/upload/check", host=host, body=check, headers=json_type)[0], 200)
        for origin in ("http://evil.example", f"http://localhost:{self.port}", f"http://books.example.net:{self.port}", f"https://{host}", "null"):
            with self.subTest(origin=origin):
                status, _ = self.ask("POST", "/api/upload/check", host=host, body=check, headers={**json_type, "Origin": origin})
                self.assertEqual(status, 403)

    def test_a_client_from_outside_gets_403_and_nothing_else_happens(self):
        seen = []
        self.server.client_allowed = lambda address: seen.append(address) and False
        start = json.dumps({"source": "library", "name": "x.mp4"}).encode()
        for method, path, body in (("GET", "/", None), ("GET", "/api/videos", None), ("GET", "/no/such/thing", None),
                                   ("POST", "/api/jobs", start), ("POST", "/api/upload?name=book.mp4", b"0123456789"),
                                   ("DELETE", "/api/runs", None)):
            with self.subTest(method=method, path=path):
                status, answer = self.ask(method, path, host=f"192.168.1.166:{self.port}", body=body)
                self.assertEqual(status, 403)
                self.assertEqual(json.loads(answer), {"error": "The book scanner only answers devices on the home network."})
        self.assertEqual(seen, ["127.0.0.1"] * 6)
        self.assertIsNone(self.app.jobs.latest())
        self.assertEqual(list(self.config.uploads_dir.glob("*")), [])

    def test_the_client_rule_is_the_one_for_private_addresses(self):
        self.assertIs(self.server.client_allowed, network.client_allowed)
        self.assertEqual(self.ask("GET", "/api/job", host=f"localhost:{self.port}")[0], 200)


class CommandLineTests(unittest.TestCase):
    def refused(self, *arguments: str) -> str:
        said = io.StringIO()
        with contextlib.redirect_stderr(said), self.assertRaises(SystemExit) as stopped:
            main(list(arguments))
        self.assertEqual(stopped.exception.code, 2)
        return said.getvalue()

    def test_a_host_name_needs_the_home_network_mode(self):
        self.assertIn("--host-name only applies together with --lan", self.refused("--host-name", "books.example.net"))

    def test_a_host_name_must_be_a_name(self):
        self.assertIn("is not a host name", self.refused("--lan", "--host-name", "http://books.example.net:8770/"))


if __name__ == "__main__":
    unittest.main()
