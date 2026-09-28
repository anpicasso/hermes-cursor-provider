import json

import pytest
import h2.config
import h2.connection

from hermes_cursor_provider.cursor_protocol import agent_pb2
from hermes_cursor_provider.cursor_protocol import stream as stream_module
from hermes_cursor_provider.cursor_protocol.framing import CursorProtocolError, frame_connect_message, parse_connect_frames
from hermes_cursor_provider.cursor_protocol.stream import (
    _consume_connect_stream,
    _rejected_exec,
    _rejected_interaction,
)


def _frame(**kwargs):
    message = agent_pb2.AgentServerMessage(interaction_update=agent_pb2.InteractionUpdate(**kwargs))
    return frame_connect_message(message.SerializeToString())


def test_stream_handles_split_frames_and_mcp_tool_call():
    frames = b"".join([
        _frame(text_delta=agent_pb2.TextDeltaUpdate(text="hello")),
        _frame(thinking_delta=agent_pb2.ThinkingDeltaUpdate(text="hmm")),
        _frame(tool_call_started=agent_pb2.ToolCallStartedUpdate(
            call_id="call-1",
            tool_call=agent_pb2.ToolCall(mcp_tool_call=agent_pb2.McpToolCall(args=agent_pb2.McpArgs(
                name="read_file", tool_name="read_file", tool_call_id="call-1",
            ))),
        )),
        _frame(tool_call_completed=agent_pb2.ToolCallCompletedUpdate(
            call_id="call-1",
            tool_call=agent_pb2.ToolCall(mcp_tool_call=agent_pb2.McpToolCall(args=agent_pb2.McpArgs(
                name="read_file", tool_name="read_file", tool_call_id="call-1", args={"path": b'"README.md"'},
            ))),
        )),
        _frame(turn_ended=agent_pb2.TurnEndedUpdate()),
    ])
    result = _consume_connect_stream(
        [frames[:7], frames[7:19], frames[19:]], blob_store={}, request_context_tools=[]
    )
    message = result.choices[0].message
    assert message.content == "hello"
    assert message.reasoning == "hmm"
    assert result.choices[0].finish_reason == "tool_calls"
    assert json.loads(message.tool_calls[0].function.arguments) == {"path": "README.md"}


def test_native_read_is_rejected_without_execution():
    writes = []
    server = agent_pb2.AgentServerMessage(exec_server_message=agent_pb2.ExecServerMessage(
        id=7, exec_id="exec-1", read_args=agent_pb2.ReadArgs(path="/etc/passwd"),
    ))
    _consume_connect_stream(
        [frame_connect_message(server.SerializeToString()), _frame(turn_ended=agent_pb2.TurnEndedUpdate())],
        blob_store={}, request_context_tools=[], send_client_frame=writes.append,
    )
    _, payload = list(parse_connect_frames(bytearray(writes[0])))[0]
    response = agent_pb2.AgentClientMessage()
    response.ParseFromString(payload)
    rejected = response.exec_client_message.read_result.rejected
    assert "Hermes owns" in rejected.reason


def test_every_native_exec_request_gets_a_response():
    fields = agent_pb2.ExecServerMessage.DESCRIPTOR.oneofs_by_name["message"].fields
    assert len(fields) == 17
    for field in fields:
        request = agent_pb2.ExecServerMessage(id=7, exec_id="exec-1")
        getattr(request, field.name).SetInParent()
        response = _rejected_exec(request, [])
        assert response.WhichOneof("message"), field.name
    assert "Hermes owns" in _rejected_exec(
        agent_pb2.ExecServerMessage(id=7, exec_id="exec-1", grep_args=agent_pb2.GrepArgs()), []
    ).grep_result.error.error


def test_every_interaction_query_gets_a_fail_closed_response():
    fields = agent_pb2.InteractionQuery.DESCRIPTOR.oneofs_by_name["query"].fields
    expected = {
        "web_search_request_query": "web_search_request_response",
        "ask_question_interaction_query": "ask_question_interaction_response",
        "switch_mode_request_query": "switch_mode_request_response",
        "exa_search_request_query": "exa_search_request_response",
        "exa_fetch_request_query": "exa_fetch_request_response",
        "create_plan_request_query": "create_plan_request_response",
        "setup_vm_environment_args": "setup_vm_environment_result",
    }
    assert {field.name for field in fields} == set(expected)
    for field in fields:
        query = agent_pb2.InteractionQuery(id=9)
        getattr(query, field.name).SetInParent()
        response = _rejected_interaction(query).interaction_response
        assert response.id == 9
        assert response.WhichOneof("result") == expected[field.name]


def test_step_completion_does_not_truncate_the_turn():
    result = _consume_connect_stream(
        [
            _frame(step_completed=agent_pb2.StepCompletedUpdate(step_id=1)),
            _frame(text_delta=agent_pb2.TextDeltaUpdate(text="after-step")),
            _frame(turn_ended=agent_pb2.TurnEndedUpdate()),
        ],
        blob_store={},
        request_context_tools=[],
    )
    assert result.choices[0].message.content == "after-step"


def test_incomplete_stream_is_not_reported_as_success():
    with pytest.raises(CursorProtocolError, match="before turn completion") as caught:
        _consume_connect_stream(
            [_frame(text_delta=agent_pb2.TextDeltaUpdate(text="cut off"))],
            blob_store={},
            request_context_tools=[],
        )
    assert caught.value.status_code == 502


def test_http2_stream_reset_fails_fast(monkeypatch):
    class ResetSocket:
        def __init__(self):
            self.sent = bytearray()
            self.replied = False

        def sendall(self, data):
            self.sent.extend(data)

        def recv(self, _size):
            if self.replied:
                return b""
            server = h2.connection.H2Connection(
                config=h2.config.H2Configuration(client_side=False, header_encoding="utf-8")
            )
            server.initiate_connection()
            server.receive_data(bytes(self.sent))
            server.reset_stream(1, error_code=8)
            self.replied = True
            return server.data_to_send()

        def close(self):
            pass

    sock = ResetSocket()
    monkeypatch.setattr(stream_module, "open_h2_tls", lambda *args, **kwargs: (sock, "api2.cursor.sh", 443))
    with pytest.raises(CursorProtocolError, match="stream reset") as caught:
        stream_module.run_cursor_turn(
            api_key="token",
            model_id="default",
            messages=[{"role": "user", "content": "hello"}],
            system_prompt=None,
            tools=None,
            conversation_id="conversation",
        )
    assert caught.value.status_code == 502
