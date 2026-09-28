import sys
from types import ModuleType
from types import SimpleNamespace

from hermes_cursor_provider import client as client_module
from hermes_cursor_provider import credentials
from hermes_cursor_provider.client import CursorClient


def _completion(model, conversation_id, text="ok", tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=tool_calls, reasoning=None),
            finish_reason="tool_calls" if tool_calls else "stop",
        )],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=1, total_tokens=1),
        model=model,
        conversation_state=SimpleNamespace(id=conversation_id),
        blob_store={"x": b"y"},
    )


def test_client_preserves_session_and_streams(monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        if kwargs.get("on_text_delta"):
            kwargs["on_text_delta"]("ok")
        return _completion(kwargs["model_id"], kwargs["conversation_id"])

    monkeypatch.setattr(client_module, "run_cursor_turn", fake_run)
    client = CursorClient(api_key="token")
    first = client.chat.completions.create(
        model="default", messages=[{"role": "user", "content": "one"}],
        extra_body={"_cursor_session_id": "session"},
    )
    chunks = list(client.chat.completions.create(
        model="default", messages=[{"role": "user", "content": "two"}], stream=True,
        extra_body={"_cursor_session_id": "session"},
    ))
    assert first.choices[0].message.content == "ok"
    assert chunks[0].choices[0].delta.content == "ok"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert calls[0]["conversation_id"] == calls[1]["conversation_id"]
    assert calls[1]["conversation_state"].id == calls[0]["conversation_id"]


def test_client_honors_tool_choice_none(monkeypatch):
    seen = {}
    def fake_run(**kwargs):
        seen.update(kwargs)
        return _completion(kwargs["model_id"], kwargs["conversation_id"])
    monkeypatch.setattr(client_module, "run_cursor_turn", fake_run)
    CursorClient(api_key="token").chat.completions.create(
        model="default", messages=[], tools=[{"type": "function", "function": {"name": "x"}}], tool_choice="none",
    )
    assert seen["tools"] == []


def test_client_refreshes_expiring_pool_token(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return _completion(kwargs["model_id"], kwargs["conversation_id"])

    pool = SimpleNamespace(
        try_refresh_matching=lambda **kwargs: SimpleNamespace(runtime_api_key="fresh-token")
    )
    agent = ModuleType("agent")
    credential_pool = ModuleType("agent.credential_pool")
    setattr(credential_pool, "load_pool", lambda provider: pool)
    setattr(agent, "credential_pool", credential_pool)
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.credential_pool", credential_pool)
    monkeypatch.setattr(credentials, "token_expiry_ms", lambda _: 0)
    monkeypatch.setattr(client_module, "run_cursor_turn", fake_run)
    client = CursorClient(api_key="stale-token")
    client.chat.completions.create(model="default", messages=[])
    assert seen["api_key"] == "fresh-token"
    assert client.api_key == "fresh-token"
