"""Native tool-call interception, lenient parsing, corrective retry, and debug logging."""
from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cursor_provider import client as client_module
from hermes_cursor_provider.client import (
    CursorSDKClient,
    extract_tool_calls,
    match_allowed_name,
    render_prompt,
)
from test_client import (  # noqa: E402
    FakeAgent,
    FakeCursorClient,
    ScriptedRun,
    _assistant_event,
    _stream_calls,
    _stream_text,
    _tool,
    fake_sdk,  # autouse fixture: patches the Cursor SDK with fakes
)


def _native(name, args, status="running", call_id="c1", result=None):
    return SimpleNamespace(
        type="tool_call", agent_id="a", run_id="r", call_id=call_id,
        name=name, status=status, args=args, result=result,
    )


def _sequence(sdk, scripts):
    """Each agent.send() returns the next scripted run; returns (runs, agent)."""
    runs = [ScriptedRun(sdk.result, script) for script in scripts]
    agent = FakeAgent(sdk)
    queue = list(runs)

    def send(message):
        agent.messages.append(message)
        sdk.messages.append(message)
        return queue.pop(0)

    agent.send = send
    sdk.create_agent = lambda options: agent
    return runs, agent


def _stream(client, tools=None, scope=None):
    kwargs = {"_cursor_session_scope": scope} if scope else {}
    return list(
        client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "make a card"}],
            tools=tools if tools is not None else [_tool("kanban_create")],
            stream=True,
            **kwargs,
        )
    )


@pytest.mark.parametrize(
    "written,expected",
    [
        ("kanban_create", "kanban_create"),
        ("functions.kanban_create", "kanban_create"),
        ("mcp__hermes__kanban_create", "kanban_create"),
        ("mcp_kanban_create", "kanban_create"),
        ("KanbanCreate", "kanban_create"),
        ("create", None),
        ("shell", None),
        ("", None),
    ],
)
def test_match_allowed_name(written, expected):
    assert match_allowed_name(written, {"kanban_create", "kanban_link"}) == expected


def test_match_allowed_name_unique_suffix_only():
    allowed = {"mcp_playwright_browser_navigate", "mcp_playwright_browser_click"}
    assert match_allowed_name("browser_navigate", allowed) == "mcp_playwright_browser_navigate"
    assert match_allowed_name("navigate", allowed) is None  # single word: too generic
    both = {"a_browser_click", "b_browser_click"}
    assert match_allowed_name("browser_click", both) is None  # ambiguous


def test_lenient_payload_without_function_wrapper_and_object_arguments():
    text = 'ok <tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>'
    calls, content = extract_tool_calls(text, {"read_file"})
    assert [c.function.name for c in calls] == ["read_file"]
    assert json.loads(calls[0].function.arguments) == {"path": "a.txt"}
    assert content == "ok"


def test_parser_reuse_after_flush_resets_incomplete_block_state():
    allowed = {"read_file"}
    valid = (
        '<tool_call>{"id":"c9","type":"function","function":{"name":"read_file",'
        '"arguments":"{\\"path\\":\\"a.py\\"}"}}</tool_call>'
    )
    incomplete_blocks = [
        '<tool_call>{"function":{"name":"read_file","arguments":"{\\"path\\":\\"' + "A" * 200,
        '<tool_call>{"function":{"name":"read_file","arguments":"abc' + "\\",
        '<tool_call>{"function":{"name":"read_file","arguments":"{}"}<',
    ]
    for incomplete in incomplete_blocks:
        parser = client_module._ToolBlockStream(allowed)
        list(parser.feed(incomplete))
        list(parser.flush())
        list(parser.feed(valid))
        list(parser.flush())
        assert [call.function.name for call in parser.calls] == ["read_file"], incomplete[-20:]


def test_prompt_tells_model_native_tools_do_not_exist():
    prompt, _ = render_prompt([{"role": "user", "content": "x"}], tools=[_tool()])
    assert "Tool not found" in prompt
    assert prompt.rstrip().endswith("Continue from the final transcript entry.")
    assert "Reminder: when a tool is needed" in prompt


def test_native_call_is_intercepted_and_run_cancelled():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        runs, agent = _sequence(sdk, [[
            _native("kanban_create", {"title": "t", "assignee": "researcher"}),
            _native("kanban_create", {"title": "t", "assignee": "researcher"}, status="error",
                    result="Tool not found"),
            _assistant_event("I could not create the card"),
        ]])
        chunks = _stream(client)
        calls = _stream_calls(chunks)
        assert [c.function.name for c in calls] == ["kanban_create"]
        assert json.loads(calls[0].function.arguments)["assignee"] == "researcher"
        assert _stream_text(chunks) == ""  # the apology after the failed native call is dropped
        assert runs[0].cancelled
        finish = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
        assert finish == ["tool_calls"]
        assert len(agent.messages) == 1  # no retry needed
    finally:
        client.close()


def test_native_mcp_wrapper_is_unwrapped_and_prefixes_matched():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _sequence(sdk, [[
            _native("CallMcpTool", {"toolName": "mcp_kanban_create", "arguments": json.dumps({"title": "x"})},
                    status="error", result="not found"),
        ]])
        calls = _stream_calls(_stream(client))
        assert [c.function.name for c in calls] == ["kanban_create"]
        assert json.loads(calls[0].function.arguments) == {"title": "x"}
    finally:
        client.close()


def test_two_parallel_native_calls_both_forwarded():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _sequence(sdk, [[
            _native("kanban_create", {"title": "a"}, call_id="c1"),
            _native("kanban_create", {"title": "b"}, call_id="c2"),
            _native("kanban_create", {"title": "a"}, call_id="c1", status="error", result="Tool not found"),
        ]])
        calls = _stream_calls(_stream(client))
        assert [json.loads(c.function.arguments)["title"] for c in calls] == ["a", "b"]
    finally:
        client.close()


def test_completed_only_parallel_native_calls_are_all_forwarded():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        runs, _ = _sequence(sdk, [[
            _native("kanban_create", {"title": "a"}, call_id="c1", status="completed", result="ok"),
            _native("kanban_create", {"title": "b"}, call_id="c2", status="completed", result="ok"),
            _assistant_event("should not be forwarded"),
        ]])
        chunks = _stream(client)
        calls = _stream_calls(chunks)
        assert [json.loads(c.function.arguments)["title"] for c in calls] == ["a", "b"]
        assert _stream_text(chunks) == ""
        assert runs[0].cancelled  # the run is still cancelled once the batch is collected
    finally:
        client.close()


def test_native_batch_stops_before_a_follow_up_tool_call():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        runs, _ = _sequence(sdk, [[
            _native("kanban_create", {"title": "a"}, call_id="c1", status="error", result="Tool not found"),
            _assistant_event("trying another tool"),
            _native("browser_exec", {"cmd": "x"}, call_id="c2", status="error", result="Tool not found"),
        ]])
        calls = _stream_calls(_stream(client, tools=[_tool("kanban_create"), _tool("browser_exec")]))
        assert [c.function.name for c in calls] == ["kanban_create"]
        assert runs[0].cancelled
    finally:
        client.close()


def test_identical_native_calls_are_deduplicated():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        script = [
            _native("kanban_request_review", {"task_id": "t_1"}, call_id=f"c{i}", status="running")
            for i in range(20)
        ] + [
            _native(
                "kanban_request_review", {"task_id": "t_1"}, call_id=f"c{i}",
                status="error", result="Tool not found",
            )
            for i in range(20)
        ]
        _sequence(sdk, [script])
        calls = _stream_calls(_stream(client, tools=[_tool("kanban_request_review")]))
        assert [c.function.name for c in calls] == ["kanban_request_review"]
    finally:
        client.close()


def test_duplicate_running_only_native_calls_are_deduplicated_at_run_end():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _sequence(sdk, [[
            _native("kanban_request_review", {"task_id": "t_1"}, call_id=f"c{i}", status="running")
            for i in range(5)
        ]])
        calls = _stream_calls(_stream(client, tools=[_tool("kanban_request_review")]))
        assert [c.function.name for c in calls] == ["kanban_request_review"]
    finally:
        client.close()


def test_interception_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_CURSOR_INTERCEPT_NATIVE", "0")
    monkeypatch.setenv("HERMES_CURSOR_RETRY", "0")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _sequence(sdk, [[_native("kanban_create", {"title": "a"}, status="error"), _assistant_event("sorry")]])
        chunks = _stream(client)
        assert _stream_calls(chunks) == []
        assert _stream_text(chunks) == "sorry"
    finally:
        client.close()


def test_unknown_native_tool_triggers_one_corrective_retry_that_succeeds():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    block = '<tool_call>{"id":"c9","type":"function","function":{"name":"kanban_create","arguments":"{\\"title\\":\\"t\\"}"}}</tool_call>'
    try:
        runs, agent = _sequence(sdk, [
            [_native("browser_exec", {"cmd": "x"}, status="error", result="Tool not found"),
             _assistant_event("Tools are unavailable here.")],
            [_assistant_event(block)],
        ])
        calls = _stream_calls(_stream(client))
        assert [c.function.name for c in calls] == ["kanban_create"]
        assert len(agent.messages) == 2
        correction = agent.messages[1]["text"]
        assert "did not produce a valid Hermes tool call" in correction
        assert "browser_exec" in correction and "kanban_create" in correction
    finally:
        client.close()


def test_retry_after_incomplete_block_parses_the_corrected_call():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    rejected = '<tool_call>{"name":"kanban_create"}</tool_call>'
    incomplete = '<tool_call>{"function":{"name":"kanban_create","arguments":"{\\"title\\":\\"' + "A" * 200
    valid = (
        '<tool_call>{"id":"c9","type":"function","function":{"name":"kanban_create",'
        '"arguments":"{\\"title\\":\\"t\\"}"}}</tool_call>'
    )
    try:
        runs, agent = _sequence(sdk, [
            [_assistant_event(rejected + incomplete)],
            [_assistant_event(valid)],
        ])
        chunks = _stream(client)
        calls = _stream_calls(chunks)
        assert [c.function.name for c in calls] == ["kanban_create"]
        assert json.loads(calls[0].function.arguments) == {"title": "t"}
        finish = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
        assert finish == ["tool_calls"]
        assert len(agent.messages) == 2
    finally:
        client.close()


def test_retry_happens_at_most_once():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _, agent = _sequence(sdk, [
            [_native("browser_exec", {}, status="error"), _assistant_event("no")],
            [_native("browser_exec", {}, status="error"), _assistant_event("still no")],
        ])
        chunks = _stream(client)
        assert _stream_calls(chunks) == []
        assert len(agent.messages) == 2
    finally:
        client.close()


def test_plain_text_answer_without_tool_attempt_is_not_retried():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _, agent = _sequence(sdk, [[_assistant_event("All done.")]])
        chunks = _stream(client)
        assert _stream_text(chunks) == "All done."
        assert len(agent.messages) == 1
    finally:
        client.close()


def test_poisoned_session_agent_is_reset_after_failed_tool_turn():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _, agent = _sequence(sdk, [
            [_native("browser_exec", {}, status="error"), _assistant_event("no")],
            [_native("browser_exec", {}, status="error"), _assistant_event("no")],
        ])
        _stream(client, scope="scope-p")
        assert agent.closed  # retained agent dropped so the next turn starts clean
    finally:
        client.close()


def test_debug_events_are_opt_in_and_omit_raw_payloads(monkeypatch, tmp_path):
    log = Path(tmp_path / "hermes" / "logs" / "cursor-provider-events.jsonl")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    args = {"command": "curl -H 'Authorization: Bearer LEAK_ME_42'"}
    result = "AWS_SECRET=hunter2"
    try:
        _sequence(sdk, [[_native("kanban_create", args, status="error", result=result)]])
        _stream(client)
        assert not log.exists()
        monkeypatch.setenv("HERMES_CURSOR_DEBUG_EVENTS", "1")
        _sequence(sdk, [[_native("kanban_create", args, status="error", result=result)]])
        _stream(client)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        row = rows[0]
        assert row["kind"] == "tool_call" and row["name"] == "kanban_create"
        assert row["status"] == "error" and row["call_id"] == "c1"
        assert row["arg_keys"] == ["command"]
        assert "args" not in row and "result" not in row
        assert "hunter2" not in log.read_text()
        assert "LEAK_ME_42" not in log.read_text()
        assert stat.S_IMODE(log.stat().st_mode) == 0o600
        assert stat.S_IMODE(log.parent.stat().st_mode) == 0o700
    finally:
        client.close()


def test_rejected_block_debug_log_omits_raw_block(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CURSOR_DEBUG_EVENTS", "1")
    monkeypatch.setenv("HERMES_CURSOR_RETRY", "0")
    log = Path(tmp_path / "hermes" / "logs" / "cursor-provider-events.jsonl")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    block = '<tool_call>{"name":"kanban_create","arguments":"hunter2"}</tool_call>'
    try:
        _sequence(sdk, [[_assistant_event(block)]])
        _stream(client)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert any(row["kind"] == "rejected_tool_block" and row["chars"] == len(block) for row in rows)
        assert "hunter2" not in log.read_text()
    finally:
        client.close()


def test_debug_event_never_logs_raw_payload_field_names(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CURSOR_DEBUG_EVENTS", "1")
    log = Path(tmp_path / "hermes" / "logs" / "cursor-provider-events.jsonl")
    client_module._debug_event(
        "tool_call", name="x", args={"password": "hunter2"}, result="hunter2", block="hunter2"
    )
    row = json.loads(log.read_text())
    assert row["kind"] == "tool_call" and row["name"] == "x"
    assert "args" not in row and "result" not in row and "block" not in row
    assert "hunter2" not in log.read_text()


def test_incomplete_args_on_running_event_are_not_used():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        runs, _ = _sequence(sdk, [[
            _native("kanban_create", {}, status="running"),  # args still streaming
            _native("kanban_create", {"title": "final", "assignee": "researcher"}, status="error",
                    result="Tool not found"),
        ]])
        calls = _stream_calls(_stream(client))
        assert len(calls) == 1
        assert json.loads(calls[0].function.arguments) == {"title": "final", "assignee": "researcher"}
        assert runs[0].cancelled
    finally:
        client.close()


def test_final_event_without_args_falls_back_to_pending_args():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _sequence(sdk, [[
            _native("kanban_create", {"title": "t", "assignee": "qa"}, status="running"),
            _native("kanban_create", None, status="error", result="Tool not found"),
        ]])
        calls = _stream_calls(_stream(client))
        assert json.loads(calls[0].function.arguments) == {"title": "t", "assignee": "qa"}
    finally:
        client.close()


def test_running_only_call_is_converted_when_the_run_ends():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _sequence(sdk, [[_native("kanban_create", {"title": "t", "assignee": "qa"}, status="running")]])
        calls = _stream_calls(_stream(client))
        assert [c.function.name for c in calls] == ["kanban_create"]
    finally:
        client.close()


def test_debug_log_omits_typed_text_and_values(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CURSOR_DEBUG_EVENTS", "1")
    log = Path(tmp_path / "hermes" / "logs" / "cursor-provider-events.jsonl")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _sequence(sdk, [[_native("browser_type", {"ref": "e5", "text": "S3cretPw", "value": "S3cretPw"}, status="error")]])
        _stream(client, tools=[_tool("browser_type")])
        assert "S3cretPw" not in log.read_text()
    finally:
        client.close()
