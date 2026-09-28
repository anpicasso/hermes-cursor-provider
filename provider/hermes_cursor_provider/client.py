"""OpenAI-compatible client facade for Cursor's Agent protocol."""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from .cursor_protocol.constants import CURSOR_API_URL
from .cursor_protocol.stream import run_cursor_turn

_HERMES_AUTHORITY = (
    "Hermes is the control plane. Use only MCP tools supplied by provider 'hermes-agent'. "
    "Do not use Cursor-native shell, file, browser, computer-use, todo, or other execution tools. "
    "When a Hermes tool is needed, request it once and stop so Hermes can apply approvals and execute it."
)
_DONE = object()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def _secret(value: Any) -> str:
    getter = getattr(value, "get_secret_value", None)
    return str(getter() if callable(getter) else (value or ""))


def _timeout_seconds(value: Any, default: float = 60.0) -> float:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    for attr in ("read", "connect"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)
    return default


def _chunk(*, model: str, content: str | None = None, reasoning: str | None = None,
           tool_calls: list[Any] | None = None, finish_reason: str | None = None,
           usage: Any = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, reasoning=reasoning, reasoning_content=reasoning, tool_calls=tool_calls)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _tool_delta(call: Any, index: int) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=call.id,
        type="function",
        function=SimpleNamespace(name=call.function.name, arguments=call.function.arguments),
    )


@dataclass
class _SessionState:
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_state: Any = None
    blob_store: dict[str, bytes] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


class CursorClient:
    """Duck-typed OpenAI client used by Hermes' provider seam."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, **_: Any):
        self.api_key = _secret(api_key)
        self.base_url = str(base_url or CURSOR_API_URL).rstrip("/")
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self._states: dict[str, _SessionState] = {}
        self._states_lock = threading.Lock()
        self._interrupts: set[threading.Event] = set()
        self._interrupts_lock = threading.Lock()
        self._token_lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.is_closed = True
        self.cancel()

    def cancel(self) -> None:
        with self._interrupts_lock:
            for event in tuple(self._interrupts):
                event.set()

    def _state(self, session_id: str) -> _SessionState:
        with self._states_lock:
            return self._states.setdefault(session_id, _SessionState())

    def _fresh_api_key(self) -> str:
        """Refresh an expiring pool-backed token before opening the request."""
        with self._token_lock:
            from .credentials import token_expiry_ms

            expiry = token_expiry_ms(self.api_key)
            if expiry is None or expiry > int(time.time() * 1000) + 60_000:
                return self.api_key
            try:
                from agent.credential_pool import load_pool

                refreshed = load_pool("cursor").try_refresh_matching(api_key_hint=self.api_key)
                if refreshed is not None:
                    self.api_key = _secret(refreshed.runtime_api_key)
            except Exception:
                # The still-valid token may succeed; normal 401 recovery remains available.
                pass
            return self.api_key

    def _run(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str,
        timeout: float,
        on_text_delta=None,
        on_reasoning_delta=None,
    ) -> Any:
        if self.is_closed:
            raise RuntimeError("Cursor client is closed")
        state = self._state(session_id)
        interrupt = threading.Event()
        with self._interrupts_lock:
            self._interrupts.add(interrupt)
        normalized = [item for item in (_plain(value) for value in messages) if isinstance(item, dict)]
        system = [item.get("content", "") for item in normalized if item.get("role") == "system"]
        conversation = [item for item in normalized if item.get("role") != "system"]
        try:
            with state.lock:
                result = run_cursor_turn(
                    api_key=self._fresh_api_key(),
                    base_url=self.base_url,
                    model_id=model,
                    messages=conversation,
                    system_prompt=system,
                    tools=tools,
                    conversation_id=state.conversation_id,
                    blob_store=state.blob_store,
                    conversation_state=state.conversation_state,
                    custom_system_prompt=_HERMES_AUTHORITY,
                    on_text_delta=on_text_delta,
                    on_reasoning_delta=on_reasoning_delta,
                    interrupt_event=interrupt,
                    timeout=timeout,
                )
                state.conversation_state = result.conversation_state
                state.blob_store = result.blob_store
                return result
        finally:
            with self._interrupts_lock:
                self._interrupts.discard(interrupt)

    def _stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_id: str,
        timeout: float,
    ):
        events: queue.Queue[Any] = queue.Queue()
        result_box: list[Any] = []

        def worker() -> None:
            try:
                result_box.append(
                    self._run(
                        model=model, messages=messages, tools=tools, session_id=session_id, timeout=timeout,
                        on_text_delta=lambda value: events.put(("content", value)),
                        on_reasoning_delta=lambda value: events.put(("reasoning", value)),
                    )
                )
            except BaseException as exc:
                result_box.append(exc)
            finally:
                events.put(_DONE)

        threading.Thread(target=worker, name="cursor-provider-stream", daemon=True).start()
        while True:
            event = events.get()
            if event is _DONE:
                break
            kind, value = event
            yield _chunk(model=model, content=value if kind == "content" else None,
                         reasoning=value if kind == "reasoning" else None)
        result = result_box[0]
        if isinstance(result, BaseException):
            raise result
        for index, call in enumerate(result.choices[0].message.tool_calls or []):
            yield _chunk(model=model, tool_calls=[_tool_delta(call, index)])
        yield _chunk(
            model=model,
            finish_reason=result.choices[0].finish_reason,
            usage=result.usage,
        )

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[Any] | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        body = dict(extra_body or {})
        session_id = str(body.pop("_cursor_session_id", "") or "default")
        normalized_messages = [_plain(item) for item in (messages or [])]
        normalized_tools = [] if tool_choice == "none" else [_plain(item) for item in (tools or [])]
        kwargs = {
            "model": str(model or "default"),
            "messages": normalized_messages,
            "tools": normalized_tools,
            "session_id": session_id,
            "timeout": _timeout_seconds(timeout),
        }
        return self._stream(**kwargs) if stream else self._run(**kwargs)
