"""HTTP/2 Connect client for Cursor's private Agent API.

Cursor-native execution is deliberately rejected. Only structured MCP calls are
returned to Hermes, which keeps approval, policy, and tool execution in Hermes.
"""
from __future__ import annotations

import json
import socket
import threading
import uuid
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote

from . import agent_pb2
from .constants import (
    CONNECT_END_STREAM_FLAG,
    CURSOR_AGENT_RUN_PATH,
    CURSOR_API_URL,
    CURSOR_CLIENT_VERSION,
    normalize_cursor_model_id,
)
from .framing import CursorProtocolError, frame_connect_message, parse_connect_end_stream, parse_connect_frames
from .network import open_h2_tls
from .request_builder import build_mcp_tool_definitions, build_run_request_bytes, encode_tool_input_schema

_REJECT_REASON = "Cursor-native execution is disabled; Hermes owns tool execution"
_GRPC_STATUS = {4: 504, 7: 403, 8: 429, 14: 503, 16: 401}


def _usage(prompt_tokens: int = 0, completion_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )


def _tool_call(call_id: str, name: str, arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments if arguments is not None else {})),
    )


def _response(state: dict[str, Any], model: str = "") -> SimpleNamespace:
    reasoning = "".join(state["reasoning_parts"])
    message = SimpleNamespace(
        role="assistant",
        content="".join(state["text_parts"]) or None,
        tool_calls=state["tool_calls"] or None,
        reasoning=reasoning or None,
        reasoning_content=reasoning or None,
        reasoning_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=message, finish_reason=state["finish_reason"])],
        usage=_usage(completion_tokens=state["completion_tokens"]),
        model=model,
    )


def _decode_mcp_args(mcp_args: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, raw in dict(mcp_args.args).items():
        try:
            result[key] = json.loads(raw.decode())
        except Exception:
            result[key] = raw.decode("utf-8", "replace")
    return result


def _request_context_result(tools: list[Any]) -> Any:
    context = agent_pb2.RequestContext(
        rules=[], repository_info=[], tools=tools, git_repos=[], project_layouts=[],
        mcp_instructions=[], file_contents={}, custom_subagents=[],
    )
    return agent_pb2.RequestContextResult(success=agent_pb2.RequestContextSuccess(request_context=context))


def _rejected_exec(exec_msg: Any, tools: list[Any]) -> Any:
    case = exec_msg.WhichOneof("message")
    client = agent_pb2.ExecClientMessage(id=exec_msg.id, exec_id=exec_msg.exec_id)
    if case == "request_context_args":
        client.request_context_result.CopyFrom(_request_context_result(tools))
    elif case == "read_args":
        client.read_result.rejected.path = exec_msg.read_args.path
        client.read_result.rejected.reason = _REJECT_REASON
    elif case == "write_args":
        client.write_result.rejected.path = exec_msg.write_args.path
        client.write_result.rejected.reason = _REJECT_REASON
    elif case == "delete_args":
        client.delete_result.rejected.path = exec_msg.delete_args.path
        client.delete_result.rejected.reason = _REJECT_REASON
    elif case == "diagnostics_args":
        client.diagnostics_result.rejected.path = exec_msg.diagnostics_args.path
        client.diagnostics_result.rejected.reason = _REJECT_REASON
    elif case in {"shell_args", "shell_stream_args"}:
        source = getattr(exec_msg, case)
        client.shell_result.rejected.command = source.command
        client.shell_result.rejected.working_directory = source.working_directory
        client.shell_result.rejected.reason = _REJECT_REASON
        client.shell_result.rejected.is_readonly = False
    elif case == "background_shell_spawn_args":
        source = exec_msg.background_shell_spawn_args
        client.background_shell_spawn_result.rejected.command = source.command
        client.background_shell_spawn_result.rejected.working_directory = source.working_directory
        client.background_shell_spawn_result.rejected.reason = _REJECT_REASON
        client.background_shell_spawn_result.rejected.is_readonly = False
    elif case == "write_shell_stdin_args":
        client.write_shell_stdin_result.error.error = _REJECT_REASON
    elif case == "fetch_args":
        client.fetch_result.error.url = exec_msg.fetch_args.url
        client.fetch_result.error.error = _REJECT_REASON
    elif case == "mcp_args":
        client.mcp_result.rejected.reason = _REJECT_REASON
        client.mcp_result.rejected.is_readonly = False
    elif case == "ls_args":
        client.ls_result.rejected.path = exec_msg.ls_args.path
        client.ls_result.rejected.reason = _REJECT_REASON
    elif case == "grep_args":
        client.grep_result.error.error = _REJECT_REASON
    elif case == "list_mcp_resources_exec_args":
        client.list_mcp_resources_exec_result.success.CopyFrom(agent_pb2.ListMcpResourcesSuccess(resources=[]))
    elif case == "read_mcp_resource_exec_args":
        client.read_mcp_resource_exec_result.not_found.uri = exec_msg.read_mcp_resource_exec_args.uri
    elif case == "record_screen_args":
        client.record_screen_result.failure.error = _REJECT_REASON
    elif case == "computer_use_args":
        client.computer_use_result.error.error = _REJECT_REASON
    else:
        raise CursorProtocolError(f"Unsupported Cursor exec request: {case or 'unknown'}", status_code=502)
    return client


def _kv_response(kv_msg: Any, blobs: dict[str, bytes]) -> Any:
    client = agent_pb2.KvClientMessage(id=kv_msg.id)
    case = kv_msg.WhichOneof("message")
    if case == "get_blob_args":
        data = blobs.get(kv_msg.get_blob_args.blob_id.hex())
        if data is not None:
            client.get_blob_result.blob_data = data
        else:
            client.get_blob_result.CopyFrom(agent_pb2.GetBlobResult())
    elif case == "set_blob_args":
        blobs[kv_msg.set_blob_args.blob_id.hex()] = kv_msg.set_blob_args.blob_data
        client.set_blob_result.CopyFrom(agent_pb2.SetBlobResult())
    else:
        raise CursorProtocolError(f"Unsupported Cursor KV request: {case or 'unknown'}", status_code=502)
    return client


def _rejected_interaction(query: Any) -> Any:
    """Answer server-side agent requests without granting Cursor execution."""
    case = query.WhichOneof("query")
    response = agent_pb2.InteractionResponse(id=query.id)
    direct = {
        "web_search_request_query": "web_search_request_response",
        "switch_mode_request_query": "switch_mode_request_response",
        "exa_search_request_query": "exa_search_request_response",
        "exa_fetch_request_query": "exa_fetch_request_response",
    }
    if case in direct:
        getattr(response, direct[case]).rejected.reason = _REJECT_REASON
    elif case == "ask_question_interaction_query":
        response.ask_question_interaction_response.result.rejected.reason = _REJECT_REASON
    elif case == "create_plan_request_query":
        response.create_plan_request_response.result.error.error = _REJECT_REASON
    elif case == "setup_vm_environment_args":
        # The recovered schema has no rejection/error branch for this response.
        # An explicitly present result with no success arm fails closed.
        response.setup_vm_environment_result.CopyFrom(agent_pb2.SetupVmEnvironmentResult())
    else:
        raise CursorProtocolError(f"Unsupported Cursor interaction request: {case or 'unknown'}", status_code=502)
    return agent_pb2.AgentClientMessage(interaction_response=response)


def _new_state() -> dict[str, Any]:
    return {
        "text_parts": [], "reasoning_parts": [], "tool_calls": [], "current_tool_call": None,
        "saw_token_delta": False, "completion_tokens": 0, "finish_reason": "stop", "turn_completed": False,
    }


def _apply_payload(
    payload: bytes,
    *,
    state: dict[str, Any],
    blobs: dict[str, bytes],
    tools: list[Any],
    send_message: Callable[[Any], None],
    on_text_delta: Callable[[str], None] | None = None,
    on_reasoning_delta: Callable[[str], None] | None = None,
) -> bool:
    server = agent_pb2.AgentServerMessage()
    server.ParseFromString(payload)
    case = server.WhichOneof("message")
    if case == "interaction_update":
        update = server.interaction_update
        update_case = update.WhichOneof("message")
        if update_case == "text_delta":
            delta = update.text_delta.text
            state["text_parts"].append(delta)
            if on_text_delta:
                on_text_delta(delta)
        elif update_case == "thinking_delta":
            delta = update.thinking_delta.text
            state["reasoning_parts"].append(delta)
            if on_reasoning_delta:
                on_reasoning_delta(delta)
        elif update_case == "tool_call_started":
            started = update.tool_call_started
            call = started.tool_call
            if call.WhichOneof("tool") == "mcp_tool_call":
                args = call.mcp_tool_call.args
                state["current_tool_call"] = {
                    "id": args.tool_call_id or started.call_id or str(uuid.uuid4()),
                    "name": args.tool_name or args.name,
                    "partial": "",
                    "arguments": {},
                }
        elif update_case in {"partial_tool_call", "tool_call_delta"}:
            current = state["current_tool_call"]
            if current is not None:
                delta = update.partial_tool_call.args_text_delta if update_case == "partial_tool_call" else ""
                current["partial"] += delta
                try:
                    current["arguments"] = json.loads(current["partial"])
                except Exception:
                    pass
        elif update_case == "tool_call_completed":
            completed = update.tool_call_completed.tool_call
            current = state["current_tool_call"]
            if current is not None and completed.WhichOneof("tool") == "mcp_tool_call":
                decoded = _decode_mcp_args(completed.mcp_tool_call.args)
                state["tool_calls"].append(_tool_call(current["id"], current["name"], decoded or current["arguments"]))
                state["current_tool_call"] = None
        elif update_case == "token_delta":
            state["saw_token_delta"] = True
            state["completion_tokens"] += update.token_delta.tokens
        elif update_case == "turn_ended":
            state["finish_reason"] = "tool_calls" if state["tool_calls"] else "stop"
            state["turn_completed"] = True
            return True
    elif case == "conversation_checkpoint_update" and not state["saw_token_delta"]:
        state["completion_tokens"] = server.conversation_checkpoint_update.token_details.used_tokens
    elif case == "kv_server_message":
        send_message(agent_pb2.AgentClientMessage(kv_client_message=_kv_response(server.kv_server_message, blobs)))
    elif case == "exec_server_message":
        send_message(agent_pb2.AgentClientMessage(exec_client_message=_rejected_exec(server.exec_server_message, tools)))
    elif case == "interaction_query":
        send_message(_rejected_interaction(server.interaction_query))
    return False


def _consume_connect_stream(
    chunks: Iterable[bytes],
    *,
    blob_store: dict[str, bytes],
    request_context_tools: list[Any],
    send_client_frame: Callable[[bytes], None] | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    on_reasoning_delta: Callable[[str], None] | None = None,
) -> SimpleNamespace:
    state = _new_state()
    buffer = bytearray()

    def send(message: Any) -> None:
        if send_client_frame:
            send_client_frame(frame_connect_message(message.SerializeToString()))

    for chunk in chunks:
        buffer.extend(chunk)
        for flags, payload in parse_connect_frames(buffer):
            if flags & CONNECT_END_STREAM_FLAG:
                error = parse_connect_end_stream(payload)
                if error and not state["turn_completed"]:
                    raise error
                continue
            if _apply_payload(
                payload, state=state, blobs=blob_store, tools=request_context_tools,
                send_message=send, on_text_delta=on_text_delta, on_reasoning_delta=on_reasoning_delta,
            ):
                break
    if not state["turn_completed"]:
        raise CursorProtocolError("Cursor stream ended before turn completion", status_code=502)
    return _response(state)


def _context_tools(tools: list[dict[str, Any]] | None) -> list[Any]:
    return [
        agent_pb2.McpToolDefinition(
            name=item["name"], description=item["description"],
            provider_identifier=item["providerIdentifier"], tool_name=item["toolName"],
            input_schema=encode_tool_input_schema(item["inputSchema"]),
        )
        for item in build_mcp_tool_definitions(tools)
    ]


def run_cursor_turn(
    *,
    api_key: str,
    model_id: str,
    messages: list[dict[str, Any]],
    system_prompt: list[str] | str | None,
    tools: list[dict[str, Any]] | None,
    conversation_id: str,
    blob_store: dict[str, bytes] | None = None,
    conversation_state: Any | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    on_reasoning_delta: Callable[[str], None] | None = None,
    interrupt_event: threading.Event | None = None,
    base_url: str = CURSOR_API_URL,
    custom_system_prompt: str | None = None,
    timeout: float = 60.0,
) -> SimpleNamespace:
    if not (api_key or "").strip():
        raise CursorProtocolError("Cursor access token is required", status_code=401)
    try:
        import h2.config
        import h2.connection
        import h2.events
    except ImportError as exc:
        raise RuntimeError("Cursor provider requires h2>=4.2") from exc

    blobs = blob_store if blob_store is not None else {}
    request, next_state, blobs = build_run_request_bytes(
        messages=messages, system_prompt=system_prompt, tools=tools,
        model_id=model_id, conversation_id=conversation_id, blob_store=blobs,
        conversation_state=conversation_state, custom_system_prompt=custom_system_prompt,
    )
    context_tools = _context_tools(tools)
    connection = h2.connection.H2Connection(config=h2.config.H2Configuration(header_encoding="utf-8"))
    tls = None
    send_lock = threading.RLock()
    heartbeat_stop = threading.Event()
    stream_id = 0

    def flush_locked() -> None:
        pending = connection.data_to_send()
        if pending and tls is not None:
            tls.sendall(pending)

    def send_message(message: Any) -> None:
        with send_lock:
            connection.send_data(stream_id, frame_connect_message(message.SerializeToString()))
            flush_locked()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(5.0):
            try:
                send_message(agent_pb2.AgentClientMessage(client_heartbeat=agent_pb2.ClientHeartbeat()))
            except Exception:
                return

    response_status: int | None = None
    grpc_status: int | None = None
    grpc_message = ""
    stream_state = _new_state()
    buffer = bytearray()
    model = normalize_cursor_model_id(model_id)

    try:
        tls, host, _ = open_h2_tls(base_url, timeout=timeout)
        with send_lock:
            connection.initiate_connection()
            flush_locked()
            stream_id = connection.get_next_available_stream_id()
            connection.send_headers(
                stream_id,
                [
                    (":method", "POST"), (":path", CURSOR_AGENT_RUN_PATH), (":scheme", "https"),
                    (":authority", host), ("authorization", f"Bearer {api_key}"),
                    ("content-type", "application/connect+proto"), ("connect-protocol-version", "1"),
                    ("te", "trailers"), ("x-ghost-mode", "true"),
                    ("x-cursor-client-version", CURSOR_CLIENT_VERSION), ("x-cursor-client-type", "cli"),
                    ("x-request-id", str(uuid.uuid4())),
                ],
                end_stream=False,
            )
            connection.send_data(stream_id, frame_connect_message(request), end_stream=False)
            flush_locked()
        threading.Thread(target=heartbeat, daemon=True).start()

        ended = False
        while not ended:
            if interrupt_event is not None and interrupt_event.is_set():
                raise InterruptedError("Cursor request aborted")
            try:
                data = tls.recv(65535)
            except socket.timeout as exc:
                raise CursorProtocolError("Cursor stream timed out", status_code=504) from exc
            if not data:
                break
            with send_lock:
                events = connection.receive_data(data)
                flush_locked()
            for event in events:
                if isinstance(event, h2.events.ResponseReceived):
                    raw = dict(event.headers).get(":status")
                    response_status = int(raw) if raw else None
                elif isinstance(event, h2.events.TrailersReceived):
                    trailers = dict(event.headers)
                    grpc_status = int(trailers.get("grpc-status") or 0)
                    grpc_message = str(trailers.get("grpc-message") or "")
                elif isinstance(event, h2.events.DataReceived):
                    with send_lock:
                        connection.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
                        flush_locked()
                    buffer.extend(event.data)
                    for flags, payload in parse_connect_frames(buffer):
                        if flags & CONNECT_END_STREAM_FLAG:
                            error = parse_connect_end_stream(payload)
                            if error and not stream_state["turn_completed"]:
                                raise error
                            continue
                        if _apply_payload(
                            payload, state=stream_state, blobs=blobs, tools=context_tools,
                            send_message=send_message, on_text_delta=on_text_delta,
                            on_reasoning_delta=on_reasoning_delta,
                        ):
                            ended = True
                            break
                elif isinstance(event, h2.events.StreamEnded):
                    ended = True
                elif isinstance(event, h2.events.StreamReset):
                    raise CursorProtocolError("Cursor stream reset", status_code=502)

        if response_status and response_status != 200:
            raise CursorProtocolError(f"Cursor request failed with HTTP {response_status}", status_code=response_status)
        if grpc_status:
            raise CursorProtocolError(
                f"Cursor gRPC error {grpc_status}: {unquote(grpc_message)}",
                status_code=_GRPC_STATUS.get(grpc_status), code=str(grpc_status),
            )
        if not stream_state["turn_completed"]:
            raise CursorProtocolError("Cursor stream ended before turn completion", status_code=502)
        result = _response(stream_state, model)
        result.conversation_state = next_state
        result.blob_store = blobs
        return result
    finally:
        heartbeat_stop.set()
        if tls is not None:
            tls.close()
