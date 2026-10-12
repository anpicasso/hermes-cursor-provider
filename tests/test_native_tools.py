"""Native tool-call interception, lenient parsing, corrective retry, and debug logging."""
from __future__ import annotations

import json
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


def test_debug_events_logged_only_when_enabled_and_secrets_masked(monkeypatch, tmp_path):
    log = Path(tmp_path / "hermes" / "logs" / "cursor-provider-events.jsonl")
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    try:
        _sequence(sdk, [[_native("kanban_create", {"title": "t", "password": "hunter2"}, status="error")]])
        _stream(client)
        assert not log.exists()
        monkeypatch.setenv("HERMES_CURSOR_DEBUG_EVENTS", "1")
        _sequence(sdk, [[_native("kanban_create", {"title": "t", "password": "hunter2"}, status="error")]])
        _stream(client)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert rows and rows[0]["kind"] == "tool_call" and rows[0]["name"] == "kanban_create"
        assert "hunter2" not in log.read_text()
        assert rows[0]["args"]["password"] == "***"
    finally:
        client.close()


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


def test_debug_log_masks_typed_text_and_values(monkeypatch, tmp_path):
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
