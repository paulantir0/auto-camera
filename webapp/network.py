"""Who may reach the app when it is opened to the home network (``--lan``).

Without ``--lan`` none of this is used: the server listens on 127.0.0.1 and
nothing else can connect. With it the server listens on every interface, and
two rules stand between it and the wider world:

* the *client's address* must be this machine or a private one. A request
  from anywhere else is answered 403 before it is looked at, so a port
  forwarded by accident does not publish the scans;
* the *Host header* must be a name or address this machine is really known by
  at home. That stops a web page elsewhere from reaching the app through a
  name of its own that it points at this PC.

There is no login. Anyone on the home network can see the scans and start one.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
import time

ALLOWED_V4 = tuple(ipaddress.ip_network(net) for net in (
    "127.0.0.0/8",        # this machine
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",   # private networks
    "169.254.0.0/16",     # link-local
    "100.64.0.0/10",      # shared address space, which Tailscale uses
))
ALLOWED_V6 = tuple(ipaddress.ip_network(net) for net in (
    "::1/128",            # this machine
    "fc00::/7",           # unique local addresses (Tailscale's are among them)
    "fe80::/10",          # link-local
))
TAILSCALE = ipaddress.ip_network("100.64.0.0/10")
DNS_NAME = re.compile(r"[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?\Z")


def parse_address(text) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """An address as the socket reports it, or None when it is not one.

    A zone ("fe80::1%eth0") is dropped. An IPv4 client of a socket that
    listens on both families is reported as ``::ffff:a.b.c.d`` and is returned
    as the IPv4 address it is. No other way of wrapping an IPv4 address in an
    IPv6 one is unwrapped.
    """
    if not isinstance(text, str):
        return None
    try:
        address = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return None
    if address.version == 6 and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def is_private(address) -> bool:
    if address is None:
        return False
    return any(address in net for net in (ALLOWED_V4 if address.version == 4 else ALLOWED_V6))


def client_allowed(text) -> bool:
    """Whether a client at this address is served in home-network mode."""
    return is_private(parse_address(text))


def own_addresses() -> list[str]:
    """This machine's own private addresses on interfaces that are up, IPv4 first; never loopback."""
    found: list[str] = []
    try:
        import psutil

        stats = psutil.net_if_stats()
        for name, addresses in psutil.net_if_addrs().items():
            if name in stats and not stats[name].isup:
                continue
            found += [a.address for a in addresses if a.family in (socket.AF_INET, socket.AF_INET6)]
    except Exception:
        # Without psutil: the address this machine would use to reach the home network, and what its own name resolves to.
        for family, target in ((socket.AF_INET, "192.168.255.255"), (socket.AF_INET6, "fd00::1")):
            try:
                with socket.socket(family, socket.SOCK_DGRAM) as probe:
                    probe.connect((target, 9))        # a UDP socket: nothing is sent
                    found.append(probe.getsockname()[0])
            except OSError:
                pass
        try:
            found += [info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None)]
        except OSError:
            pass
    kept: dict[str, ipaddress.IPv4Address | ipaddress.IPv6Address] = {}
    for text in found:
        address = parse_address(text)
        if is_private(address) and not address.is_loopback:
            kept[str(address)] = address
    return [text for text, address in sorted(kept.items(), key=lambda item: item[1].version)]      # stable: interface order kept


def split_host(header) -> tuple[str, int | None] | None:
    """``("192.168.1.166", 8770)`` from a Host header; the port is None when none is given. None if it is not a Host."""
    if not isinstance(header, str):
        return None
    text = header.strip().lower()
    port = None
    if text.startswith("["):
        end = text.find("]")
        if end < 0:
            return None
        name, rest = text[:end + 1], text[end + 1:]
        if rest:
            if not rest.startswith(":"):
                return None
            text_port = rest[1:]
        else:
            text_port = None
    elif ":" in text:
        name, _, text_port = text.rpartition(":")
        if ":" in name:
            return None
    else:
        name, text_port = text, None
    if text_port is not None:
        if not text_port.isascii() or not text_port.isdigit() or len(text_port) > 5:
            return None
        port = int(text_port)
    if not name.startswith("["):
        name = name[:-1] if name.endswith(".") else name
        if not DNS_NAME.match(name):
            return None
    return name, port


def clean_host_name(text: str) -> str:
    """A name given with ``--host-name``: a plain DNS name, without scheme, port or path."""
    name = text.strip().lower().rstrip(".")
    if not DNS_NAME.match(name):
        raise ValueError(f"{text!r} is not a host name (give just the name, such as books.example.net)")
    return name


class HostRule:
    """The Host headers answered in home-network mode.

    ``localhost`` and loopback addresses; this machine's host name, with and
    without ``.local``; each of its own private addresses; and the names given
    with ``--host-name``. A port, when the header has one, must be the port the
    server listens on. Addresses come and go (wifi, Tailscale), so they are
    looked up again, at most every few seconds, when an unknown one turns up.
    """

    def __init__(self, port: int, names=(), *, hostname: str | None = None, discover=own_addresses, refresh_seconds: float = 5.0):
        self.port = port
        hostname = (socket.gethostname() if hostname is None else hostname).strip().lower().rstrip(".")
        self.names = {"localhost"} | {clean_host_name(name) for name in names}
        if hostname and DNS_NAME.match(hostname):
            short = hostname.split(".", 1)[0]
            self.names |= {hostname, short, f"{short}.local"}
        self.extra_names = [clean_host_name(name) for name in names]
        self._discover = discover
        self._refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        self._looked = 0.0
        self._own: set = set()
        self.refresh()

    def refresh(self) -> list[str]:
        """Look up this machine's addresses again; returns them as text."""
        try:
            texts = list(self._discover())
        except Exception:
            texts = []
        with self._lock:
            self._own = {address for address in map(parse_address, texts) if address is not None}
            self._looked = time.monotonic()
        return texts

    def allows(self, header) -> bool:
        parsed = split_host(header)
        if parsed is None:
            return False
        name, port = parsed
        if port is not None and port != self.port:
            return False
        if name.startswith("["):
            address = parse_address(name[1:-1])
            if address is None:
                return False
        else:
            address = parse_address(name)
            if address is None:
                return name in self.names
        if address.is_loopback:
            return True
        if not is_private(address):
            return False
        with self._lock:
            known, stale = address in self._own, time.monotonic() - self._looked >= self._refresh_seconds
        if not known and stale:
            self.refresh()
            with self._lock:
                known = address in self._own
        return known


def url_for(host: str, port: int) -> str:
    return f"http://[{host}]:{port}/" if ":" in host else f"http://{host}:{port}/"


def startup_lines(port: int, addresses: list[str], extra_names: list[str]) -> list[str]:
    """What home-network mode prints when it starts: the addresses to open, one per line, and that there is no login."""
    parsed = [(text, parse_address(text)) for text in addresses]
    shown = [(text, address) for text, address in parsed if address is not None and address.version == 4 and not address.is_link_local]
    if not shown:        # a machine with no private IPv4 address: its unique-local IPv6 ones
        shown = [(text, address) for text, address in parsed if address is not None and address.version == 6 and not address.is_link_local]
    lines = ["Book scanner is running for the home network. Ctrl+C stops it."]
    if shown or extra_names:
        lines.append("Open one of these in a browser on a phone or computer at home:")
        for text, address in shown:
            lines.append(f"  {url_for(text, port)}" + ("   (over Tailscale)" if address.version == 4 and address in TAILSCALE else ""))
        lines += [f"  {url_for(name, port)}" for name in extra_names]
    else:
        lines.append("No home-network address was found on this PC. Is it connected to the network?")
    lines.append(f"On this PC itself: http://127.0.0.1:{port}/")
    lines.append("There is no login in this mode: anyone on the home network can see the scans and start one.")
    return lines
