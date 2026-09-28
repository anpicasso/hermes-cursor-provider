"""Cursor model discovery via GetUsableModels."""
from __future__ import annotations

import socket
import uuid
from typing import Any

from . import agent_pb2
from .constants import CURSOR_API_URL, CURSOR_CLIENT_VERSION, CURSOR_GET_USABLE_MODELS_PATH, normalize_cursor_model_id
from .framing import CursorProtocolError
from .network import open_h2_tls


def parse_usable_models_payload(payload: bytes) -> list[str]:
    response = agent_pb2.GetUsableModelsResponse()
    response.ParseFromString(payload)
    return sorted({normalize_cursor_model_id(item.model_id) for item in response.models if item.model_id})


def fetch_cursor_usable_models(
    *, api_key: str, base_url: str = CURSOR_API_URL, timeout: float = 12.0
) -> list[str]:
    token = (api_key or "").strip()
    if not token:
        return []
    try:
        import h2.config
        import h2.connection
        import h2.events
    except ImportError as exc:
        raise RuntimeError("Cursor provider requires h2>=4.2") from exc

    connection = h2.connection.H2Connection(config=h2.config.H2Configuration(header_encoding="utf-8"))
    tls = None
    body = bytearray()
    status: int | None = None
    grpc_status: int | None = None
    try:
        tls, host, _ = open_h2_tls(base_url, timeout=timeout)
        connection.initiate_connection()
        tls.sendall(connection.data_to_send())
        stream_id = connection.get_next_available_stream_id()
        connection.send_headers(
            stream_id,
            [
                (":method", "POST"), (":path", CURSOR_GET_USABLE_MODELS_PATH), (":scheme", "https"),
                (":authority", host), ("authorization", f"Bearer {token}"),
                ("content-type", "application/proto"), ("connect-protocol-version", "1"),
                ("te", "trailers"), ("x-ghost-mode", "true"),
                ("x-cursor-client-version", CURSOR_CLIENT_VERSION), ("x-cursor-client-type", "cli"),
                ("x-request-id", str(uuid.uuid4())),
            ],
            end_stream=False,
        )
        connection.send_data(stream_id, agent_pb2.GetUsableModelsRequest().SerializeToString(), end_stream=True)
        tls.sendall(connection.data_to_send())
        ended = False
        while not ended:
            try:
                data = tls.recv(65535)
            except socket.timeout as exc:
                raise CursorProtocolError("Cursor model discovery timed out", status_code=504) from exc
            if not data:
                break
            events = connection.receive_data(data)
            pending = connection.data_to_send()
            if pending:
                tls.sendall(pending)
            for event in events:
                if isinstance(event, h2.events.ResponseReceived):
                    raw = dict(event.headers).get(":status")
                    status = int(raw) if raw else None
                elif isinstance(event, h2.events.TrailersReceived):
                    grpc_status = int(dict(event.headers).get("grpc-status") or 0)
                elif isinstance(event, h2.events.DataReceived):
                    connection.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
                    pending = connection.data_to_send()
                    if pending:
                        tls.sendall(pending)
                    body.extend(event.data)
                elif isinstance(event, h2.events.StreamEnded):
                    ended = True
                elif isinstance(event, h2.events.StreamReset):
                    raise CursorProtocolError("Cursor model discovery stream reset", status_code=502)
        if status and status != 200:
            raise CursorProtocolError(f"Cursor model discovery failed with HTTP {status}", status_code=status)
        if grpc_status:
            raise CursorProtocolError(f"Cursor model discovery gRPC error {grpc_status}")
        return parse_usable_models_payload(bytes(body)) if body else []
    finally:
        if tls is not None:
            tls.close()
