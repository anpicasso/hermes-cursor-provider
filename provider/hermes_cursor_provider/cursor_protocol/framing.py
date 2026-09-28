"""Connect protocol framing helpers."""
from __future__ import annotations

import json
from collections.abc import Iterator

_STATUS_BY_CODE = {
    "unauthenticated": 401,
    "permission_denied": 403,
    "not_found": 404,
    "resource_exhausted": 429,
    "deadline_exceeded": 504,
    "unavailable": 503,
}


class CursorProtocolError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, code: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def frame_connect_message(data: bytes, flags: int = 0) -> bytes:
    payload = bytes(data)
    return bytes([flags]) + len(payload).to_bytes(4, "big") + payload


def parse_connect_frames(buffer: bytes | bytearray) -> Iterator[tuple[int, bytes]]:
    offset = 0
    total = len(buffer)
    while total - offset >= 5:
        flags = buffer[offset]
        size = int.from_bytes(buffer[offset + 1 : offset + 5], "big")
        end = offset + 5 + size
        if total < end:
            break
        yield flags, bytes(buffer[offset + 5 : end])
        offset = end
    if isinstance(buffer, bytearray) and offset:
        del buffer[:offset]


def parse_connect_end_stream(data: bytes) -> CursorProtocolError | None:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        return CursorProtocolError("Failed to parse Connect end stream")
    error = payload.get("error")
    if not error:
        return None
    code = str(error.get("code") or "unknown") if isinstance(error, dict) else "unknown"
    message = str(error.get("message") or "Unknown error") if isinstance(error, dict) else str(error)
    return CursorProtocolError(
        f"Connect error {code}: {message}",
        status_code=_STATUS_BY_CODE.get(code),
        code=code,
    )
