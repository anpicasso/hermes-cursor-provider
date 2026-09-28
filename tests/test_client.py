from __future__ import annotations

import asyncio
import base64
from pathlib import Path
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


class FakeRun:
    def __init__(self, result):
        self._result = result
        self.cancelled = False

    def wait(self):
        return self._result

    def cancel(self):
        self.cancelled = True


class FakeAgent:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False

    def send(self, message):
        self.owner.messages.append(message)
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
        self.closed = False
        self.result = SimpleNamespace(
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
        return FakeAgent(self)

    def list_models(self, *, api_key):
        self.list_api_key = api_key
        return self.models

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    FakeCursorClient.launches.clear()
    FakeCursorClient.instances.clear()
    module = SimpleNamespace(CursorClient=FakeCursorClient)
    monkeypatch.setattr(client_module, "importlib", SimpleNamespace(import_module=lambda name: module))


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
        assert options["local"]["sandbox_options"] == {"enabled": True}
        assert Path(options["local"]["cwd"]).is_dir()
        assert sdk.messages[0]["text"].endswith("Continue from the final transcript entry.")
        assert response.choices[0].message.content == "hello"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens == 7
        assert sdk.timeout_options["timeout"] == 15
    finally:
        client.close()
    assert sdk.closed


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
    with pytest.raises(ValueError):
        CursorSDKClient(api_key="secret", base_url="https://example.com")
    client = CursorSDKClient(api_key="")
    try:
        with pytest.raises(RuntimeError, match="CURSOR_API_KEY"):
            client.chat.completions.create(model="auto", messages=[])
    finally:
        client.close()
    assert FakeCursorClient.launches == []
