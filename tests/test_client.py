from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from hermes_cursor_provider import client as client_module
from hermes_cursor_provider.client import (
    CURSOR_API_URL,
    CursorSDKClient,
    extract_images,
    extract_tool_calls,
    list_cursor_models,
    render_prompt,
)


@dataclass
class FakeResult:
    result: str
    model: object
    usage: object
    stream_text: str = ""
    status: str = "FINISHED"
    status_message: str = ""


class FakeRun:
    def __init__(self, result):
        self._result = result
        self.cancelled = False

    def wait(self):
        return self._result

    def stream(self):
        if self._result.stream_text:
            yield SimpleNamespace(
                type="assistant",
                message=SimpleNamespace(
                    content=[SimpleNamespace(text=self._result.stream_text)]
                ),
            )
        if self._result.status != "FINISHED":
            yield SimpleNamespace(
                type="status",
                message=SimpleNamespace(
                    status=self._result.status,
                    message=self._result.status_message,
                ),
            )

    def cancel(self):
        self.cancelled = True


class FailingRun(FakeRun):
    def wait(self):
        raise TimeoutError("run timed out")


class FakeAgent:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False
        self.messages = []

    def send(self, message):
        self.owner.messages.append(message)
        self.messages.append(message)
        return FakeRun(self.owner.result)

    def close(self):
        self.closed = True


class FakeCursorClient:
    launches = []
    instances = []
    models = [SimpleNamespace(id="composer-2.5"), SimpleNamespace(id="gpt-5.4")]

    def __init__(self):
        self.options = []
        self.messages = []
        self.agents = []
        self.closed = False
        self.result = FakeResult(
            result="hello",
            model=SimpleNamespace(id="composer-2.5"),
            usage=SimpleNamespace(
                input_tokens=4,
                output_tokens=2,
                cache_read_tokens=1,
                total_tokens=7,
            ),
        )
        self.__class__.instances.append(self)

    @classmethod
    def launch_bridge(cls, **kwargs):
        cls.launches.append(kwargs)
        return cls()

    def with_options(self, **kwargs):
        self.timeout_options = kwargs
        return self

    def create_agent(self, options):
        self.options.append(options)
        agent = FakeAgent(self)
        self.agents.append(agent)
        return agent

    def list_models(self, *, api_key):
        self.list_api_key = api_key
        return self.models

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch, tmp_path):
    client_module._reset_session_registry()
    FakeCursorClient.launches.clear()
    FakeCursorClient.instances.clear()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    module = SimpleNamespace(CursorClient=FakeCursorClient)
    monkeypatch.setattr(client_module, "importlib", SimpleNamespace(import_module=lambda name: module))
    yield
    client_module._reset_session_registry()


def _tool(name="read_file"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Read one file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }


def test_prompt_keeps_hermes_authoritative_and_filters_tool_choice():
    prompt, allowed = render_prompt(
        [{"role": "user", "content": "inspect it"}],
        tools=[_tool(), _tool("write_file")],
        tool_choice={"type": "function", "function": {"name": "read_file"}},
    )
    assert allowed == {"read_file"}
    assert "Hermes is the only tool control plane" in prompt
    assert '"name":"read_file"' in prompt
    assert '"name":"write_file"' not in prompt
    assert render_prompt([], tools=[_tool()], tool_choice="none")[1] == set()


def test_tool_call_parser_accepts_only_offered_valid_calls():
    valid = '<tool_call>{"id":"call_7","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"a.txt\\"}"}}</tool_call>'
    calls, content = extract_tool_calls("before\n" + valid + "\nafter", {"read_file"})
    assert len(calls) == 1
    assert calls[0].id == "call_7"
    assert calls[0].function.name == "read_file"
    assert calls[0].function.arguments == '{"path":"a.txt"}'
    assert content == "before\n\nafter"

    rejected = valid.replace("read_file", "shell")
    assert extract_tool_calls(rejected, {"read_file"}) == ([], rejected)
    malformed = valid.replace('{\\"path\\":\\"a.txt\\"}', "not-json")
    assert extract_tool_calls(malformed, {"read_file"}) == ([], malformed)


def test_images_are_bounded_and_reject_unsafe_urls():
    encoded = base64.b64encode(b"image").decode()
    content = [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        {"type": "image_url", "image_url": {"url": "http://example.com/no.png"}},
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
    ]
    assert extract_images([{"role": "user", "content": content}]) == [
        {"url": "https://example.com/a.png"},
        {"data": encoded, "mimeType": "image/png"},
    ]


def test_sync_completion_uses_official_bridge_with_no_cursor_tools():
    client = CursorSDKClient(api_key="crsr_test", base_url=CURSOR_API_URL)
    try:
        response = client.chat.completions.create(
            model="cursor/composer-2.5",
            messages=[{"role": "user", "content": "hello"}],
            tools=[_tool()],
            timeout=15,
        )
        sdk = FakeCursorClient.instances[0]
        options = sdk.options[0]
        assert options["api_key"] == "crsr_test"
        assert options["model"] == "composer-2.5"
        assert options["tools"] == []
        assert options["mcp_servers"] == {}
        assert "sandbox_options" not in options["local"]
        assert not Path(options["local"]["cwd"]).exists()
        assert sdk.messages[0]["text"].endswith("Continue from the final transcript entry.")
        assert response.choices[0].message.content == "hello"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens == 7
        assert sdk.timeout_options["timeout"] == 15
    finally:
        client.close()
    assert sdk.agents[0].closed
    assert not sdk.closed


def test_completion_uses_streamed_text_when_wait_result_is_empty():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    sdk.result.stream_text = "streamed answer"
    try:
        response = client.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "hello"}]
        )
    finally:
        client.close()
    assert response.choices[0].message.content == "streamed answer"
    assert response.usage.total_tokens == 7


def test_terminal_sdk_error_is_not_misreported_as_empty_success():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = ""
    sdk.result.status = "ERROR"
    sdk.result.status_message = "You're out of usage. Switch to Auto."
    try:
        with pytest.raises(RuntimeError, match="out of usage") as exc_info:
            client.chat.completions.create(
                model="composer-2.5", messages=[{"role": "user", "content": "hello"}]
            )
    finally:
        client.close()
    assert exc_info.value.status_code == 429


def test_tool_call_and_sync_stream_are_openai_shaped():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = '<tool_call>{"function":{"name":"read_file","arguments":{"path":"x"}}}</tool_call>'
    try:
        chunks = list(client.chat.completions.create(
            model="auto", messages=[], tools=[_tool()], stream=True
        ))
    finally:
        client.close()
    assert chunks[0].choices[0].delta.tool_calls[0].function.name == "read_file"
    assert chunks[0].choices[0].finish_reason == "tool_calls"
    assert sdk.options[0]["model"] == "auto"
    assert chunks[-1].choices == []
    assert chunks[-1].usage.total_tokens == 7


def test_async_completion_and_stream():
    client = CursorSDKClient(api_key="crsr_test")

    async def scenario():
        response = await client.chat.completions.create(model="auto", messages=[])
        stream = await client.chat.completions.create(model="auto", messages=[], stream=True)
        chunks = [chunk async for chunk in stream]
        return response, chunks

    try:
        response, chunks = asyncio.run(scenario())
    finally:
        client.close()
    assert response.choices[0].message.content == "hello"
    assert chunks[0].choices[0].delta.content == "hello"


def test_model_listing_uses_sdk_and_closes_bridge():
    assert list_cursor_models(api_key="crsr_test", timeout=4) == ["composer-2.5", "gpt-5.4"]
    sdk = FakeCursorClient.instances[0]
    assert sdk.list_api_key == "crsr_test"
    assert sdk.closed
    assert FakeCursorClient.launches[0]["allow_api_key_env_fallback"] is False


def test_missing_key_and_foreign_base_url_fail_before_launch():
    foreign = CursorSDKClient(api_key="secret", base_url="https://example.com")
    with pytest.raises(ValueError):
        foreign.chat.completions.create(model="auto", messages=[])
    foreign.close()
    client = CursorSDKClient(api_key="")
    try:
        with pytest.raises(RuntimeError, match="CURSOR_API_KEY"):
            client.chat.completions.create(model="auto", messages=[])
    finally:
        client.close()
    assert FakeCursorClient.launches == []


def test_session_reuses_agent_across_facades_and_sends_only_delta():
    first = CursorSDKClient(api_key="crsr_test")
    first.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first-user"}],
        _cursor_session_scope="scope-1",
    )
    first.close()

    second = CursorSDKClient(api_key="crsr_test")
    second.chat.completions.create(
        model="auto",
        messages=[
            {"role": "user", "content": "first-user"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "second-user"},
        ],
        _cursor_session_scope="scope-1",
    )
    second.close()

    sdk = FakeCursorClient.instances[0]
    assert len(FakeCursorClient.instances) == 1
    assert len(sdk.agents) == 1
    assert len(sdk.agents[0].messages) == 2
    assert '"content":"second-user"' in sdk.agents[0].messages[1]["text"]
    assert '"content":"first-user"' not in sdk.agents[0].messages[1]["text"]
    assert not sdk.agents[0].closed


def test_session_tool_continuation_reuses_agent_without_replaying_history():
    client = CursorSDKClient(api_key="crsr_test")
    sdk = client._sdk_client()
    sdk.result.result = '<tool_call>{"id":"call_1","function":{"name":"read_file","arguments":{"path":"x"}}}</tool_call>'
    first = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "inspect-x"}],
        tools=[_tool()],
        _cursor_session_scope="scope-tool",
    )
    call = first.choices[0].message.tool_calls[0]
    sdk.result.result = "done"
    client.chat.completions.create(
        model="auto",
        messages=[
            {"role": "user", "content": "inspect-x"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call.id, "content": "file-data"},
        ],
        tools=[_tool()],
        _cursor_session_scope="scope-tool",
    )
    client.close()

    assert len(sdk.agents) == 1
    assert len(sdk.agents[0].messages) == 2
    assert '"content":"file-data"' in sdk.agents[0].messages[1]["text"]
    assert "inspect-x" not in sdk.agents[0].messages[1]["text"]


def test_session_isolates_scope_account_and_model():
    cases = [
        ("crsr_a", "auto", "scope-1"),
        ("crsr_a", "auto", "scope-2"),
        ("crsr_b", "auto", "scope-1"),
        ("crsr_a", "composer-2.5", "scope-1"),
    ]
    for api_key, model, scope in cases:
        client = CursorSDKClient(api_key=api_key)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{api_key}-{model}-{scope}"}],
            _cursor_session_scope=scope,
        )
        client.close()

    sdk = FakeCursorClient.instances[0]
    assert len(FakeCursorClient.instances) == 1
    assert len(sdk.agents) == 4
    assert len(client_module._SESSIONS) == 4


def test_history_rewrite_replaces_agent_and_cancel_preserves_session():
    client = CursorSDKClient(api_key="crsr_test")
    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "original"}],
        _cursor_session_scope="scope-rewrite",
    )
    sdk = FakeCursorClient.instances[0]
    synthetic = FakeRun(sdk.result)
    client._runs.add(synthetic)
    client.cancel()
    client._runs.clear()
    assert synthetic.cancelled

    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "rewritten"}],
        _cursor_session_scope="scope-rewrite",
    )
    client.close()

    assert len(sdk.agents) == 2
    assert sdk.agents[0].closed
    assert not sdk.agents[1].closed


def test_lru_evicts_closed_agent_without_deleting_bridge(monkeypatch):
    monkeypatch.setenv("HERMES_CURSOR_MAX_SESSIONS", "1")
    client = CursorSDKClient(api_key="crsr_test")
    for scope in ("scope-old", "scope-new"):
        client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": scope}],
            _cursor_session_scope=scope,
        )
    client.close()

    sdk = FakeCursorClient.instances[0]
    assert len(client_module._SESSIONS) == 1
    assert sdk.agents[0].closed
    assert not sdk.agents[1].closed
    assert not sdk.closed


def test_busy_session_uses_closed_one_shot_agent():
    client = CursorSDKClient(api_key="crsr_test")
    client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "first"}],
        _cursor_session_scope="scope-busy",
    )
    sdk = FakeCursorClient.instances[0]
    session = next(iter(client_module._SESSIONS.values()))
    session.lock.acquire()
    try:
        client.chat.completions.create(
            model="auto",
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "concurrent"},
            ],
            _cursor_session_scope="scope-busy",
        )
    finally:
        session.lock.release()
        client.close()

    assert len(sdk.agents) == 2
    assert not sdk.agents[0].closed
    assert sdk.agents[1].closed


def test_eviction_cleanup_cannot_delete_replacement_workspace(monkeypatch):
    monkeypatch.setenv("HERMES_CURSOR_MAX_SESSIONS", "1")
    home = client_module._home_key()
    key_old = (home, "scope-old", "account", "auto")
    key_new = (home, "scope-new", "account", "auto")
    old = client_module._lease_session(key_old)
    assert old is not None
    old_cwd = old.cwd
    client_module._release_session(old)

    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    original_close = client_module._close_session

    def paused_close(session):
        if session is old:
            cleanup_started.set()
            assert allow_cleanup.wait(2)
        original_close(session)

    monkeypatch.setattr(client_module, "_close_session", paused_close)
    worker_result = []

    def evict_old():
        leased = client_module._lease_session(key_new)
        worker_result.append(leased)
        if leased is not None:
            client_module._release_session(leased)

    worker = threading.Thread(target=evict_old)
    worker.start()
    assert cleanup_started.wait(2)
    replacement = client_module._lease_session(key_old)
    assert replacement is not None
    try:
        assert replacement.cwd != old_cwd
        allow_cleanup.set()
        worker.join(2)
        assert not worker.is_alive()
        assert replacement.cwd.exists()
    finally:
        allow_cleanup.set()
        client_module._release_session(replacement)
    assert worker_result and worker_result[0] is not None


def test_failed_wait_cancels_in_flight_run():
    client = CursorSDKClient(api_key="crsr_test")
    agent = FakeAgent(FakeCursorClient())
    failed = FailingRun(agent.owner.result)
    agent.send = lambda message: failed
    with pytest.raises(TimeoutError, match="run timed out"):
        client._run_agent(agent, prompt="hello", images=[])
    assert failed.cancelled
    assert failed not in client._runs
