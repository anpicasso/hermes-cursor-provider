from hermes_cursor_provider.cursor_protocol.framing import (
    CursorProtocolError,
    frame_connect_message,
    parse_connect_end_stream,
    parse_connect_frames,
)


def test_frames_preserve_partial_tail():
    first = frame_connect_message(b"one")
    second = frame_connect_message(b"two", flags=3)
    partial = frame_connect_message(b"three")[:6]
    buffer = bytearray(first + second + partial)
    assert list(parse_connect_frames(buffer)) == [(0, b"one"), (3, b"two")]
    assert buffer == partial


def test_connect_error_carries_http_status():
    error = parse_connect_end_stream(b'{"error":{"code":"resource_exhausted","message":"slow down"}}')
    assert isinstance(error, CursorProtocolError)
    assert error.status_code == 429
    assert "slow down" in str(error)
