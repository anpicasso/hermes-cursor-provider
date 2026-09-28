"""TLS socket setup for Cursor's fixed HTTP/2 origin."""
from __future__ import annotations

import base64
import os
import socket
import ssl
from urllib.parse import unquote, urlsplit
from urllib.request import getproxies, proxy_bypass

from .constants import CURSOR_API_URL
from .framing import CursorProtocolError

_ALLOWED_HOST = "api2.cursor.sh"


def _ssl_context() -> ssl.SSLContext:
    cafile = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE") or None
    capath = os.getenv("SSL_CERT_DIR") or None
    context = ssl.create_default_context(cafile=cafile, capath=capath)
    context.set_alpn_protocols(["h2"])
    return context


def _connect_tunnel(proxy_url: str, host: str, port: int, timeout: float) -> socket.socket:
    proxy = urlsplit(proxy_url)
    if proxy.scheme not in {"http", ""} or not proxy.hostname:
        raise CursorProtocolError("Cursor supports HTTP CONNECT proxies only")
    sock = socket.create_connection((proxy.hostname, proxy.port or 80), timeout=timeout)
    auth = ""
    if proxy.username is not None:
        raw = f"{unquote(proxy.username)}:{unquote(proxy.password or '')}".encode()
        auth = f"Proxy-Authorization: Basic {base64.b64encode(raw).decode()}\r\n"
    request = (
        f"CONNECT {host}:{port} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"{auth}Connection: keep-alive\r\n\r\n"
    ).encode("ascii")
    sock.sendall(request)
    response = bytearray()
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response.extend(chunk)
        if len(response) > 65536:
            break
    status_line = bytes(response).split(b"\r\n", 1)[0]
    if b" 200 " not in status_line:
        sock.close()
        raise CursorProtocolError(f"Proxy CONNECT failed: {status_line.decode('ascii', 'replace')}")
    return sock


def open_h2_tls(base_url: str = CURSOR_API_URL, *, timeout: float = 30.0) -> tuple[ssl.SSLSocket, str, int]:
    parsed = urlsplit(base_url or CURSOR_API_URL)
    host, port = parsed.hostname or "", parsed.port or 443
    if parsed.scheme != "https" or host != _ALLOWED_HOST or port != 443:
        raise CursorProtocolError(f"Refusing non-Cursor endpoint: {base_url!r}")

    proxy_url = "" if proxy_bypass(host) else (getproxies().get("https") or "")
    raw = _connect_tunnel(proxy_url, host, port, timeout) if proxy_url else socket.create_connection((host, port), timeout=timeout)
    raw.settimeout(timeout)
    try:
        tls = _ssl_context().wrap_socket(raw, server_hostname=host)
    except Exception:
        raw.close()
        raise
    if tls.selected_alpn_protocol() != "h2":
        tls.close()
        raise CursorProtocolError("Cursor Agent API requires HTTP/2 (TLS ALPN h2)")
    return tls, host, port
