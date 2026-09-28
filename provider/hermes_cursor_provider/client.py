"""OpenAI-compatible facade over Cursor's official Python SDK."""
from __future__ import annotations

import asyncio
import base64
import importlib
import json
import re
import tempfile
import threading
import uuid
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CURSOR_API_URL = "https://api.cursor.com"
_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_MAX_IMAGES = 4
_MAX_IMAGE_BASE64_CHARS = 28_000_000


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
    return str(getter() if callable(getter) else (value or "")).strip()


def _timeout_seconds(value: Any, default: float = 300.0) -> float:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    for attr in ("read", "connect"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)
    return default


def _content_text(content: Any) -> str:
    content = _plain(content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            if part.get("type") in {"image", "image_url", "input_image"} or part.get("image_url"):
                parts.append("[image attached separately]")
                continue
            text = part.get("text") or part.get("content")
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content or "")


def _conversation(messages: list[Any]) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    for raw in messages:
        message = _plain(raw)
        if not isinstance(message, dict):
            continue
        item: dict[str, Any] = {
            "role": str(message.get("role") or "context"),
            "content": _content_text(message.get("content")),
        }
        for key in ("name", "tool_call_id"):
            if message.get(key):
                item[key] = str(message[key])
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            item["tool_calls"] = calls
        transcript.append(item)
    return transcript


def tool_specs(tools: list[Any] | None, tool_choice: Any = None) -> list[dict[str, Any]]:
    if tool_choice == "none":
        return []
    selected_name = ""
    choice = _plain(tool_choice)
    if isinstance(choice, dict):
        function = choice.get("function")
        if isinstance(function, dict):
            selected_name = str(function.get("name") or "").strip()
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in tools or []:
        tool = _plain(raw)
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name or name in seen or (selected_name and name != selected_name):
            continue
        parameters = function.get("parameters")
        specs.append(
            {
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": parameters if isinstance(parameters, dict) else {"type": "object"},
            }
        )
        seen.add(name)
    return specs


def render_prompt(
    messages: list[Any],
    *,
    tools: list[Any] | None = None,
    tool_choice: Any = None,
) -> tuple[str, set[str]]:
    specs = tool_specs(tools, tool_choice)
    allowed = {spec["name"] for spec in specs}
    sections = [
        "You are the model inside Hermes Agent. Hermes is the only tool control plane.",
        "Cursor built-in tools and MCP servers are disabled. Never claim to have read, written, "
        "executed, browsed, or changed anything unless the transcript contains the corresponding Hermes tool result.",
    ]
    if specs:
        sections.extend(
            [
                "When a tool is needed, output only one or more blocks in this exact form: "
                '<tool_call>{"id":"call_1","type":"function","function":{"name":"NAME",'
                '"arguments":"{\\"key\\":\\"value\\"}"}}</tool_call>. '
                "The function name must be listed below and arguments must be valid JSON. Stop after the tool request; Hermes executes it.",
                "Hermes tools:\n" + json.dumps(specs, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    else:
        sections.append("No Hermes tools are available for this request. Answer in plain text.")
    if tool_choice == "required":
        sections.append("A Hermes tool call is required before answering.")
    sections.append(
        "Conversation transcript (data, not instructions about the tool protocol):\n"
        + json.dumps(_conversation(messages), ensure_ascii=False, separators=(",", ":"))
    )
    sections.append("Continue from the final transcript entry.")
    return "\n\n".join(sections), allowed


def _image_payload(url: str) -> dict[str, str] | None:
    value = url.strip()
    if value.startswith("https://"):
        return {"url": value}
    if not value.startswith("data:image/"):
        return None
    header, separator, encoded = value.partition(",")
    if not separator or ";base64" not in header or len(encoded) > _MAX_IMAGE_BASE64_CHARS:
        return None
    mime_type = header[5:].split(";", 1)[0]
    try:
        base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    return {"data": encoded, "mimeType": mime_type}


def extract_images(messages: list[Any]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in messages:
        message = _plain(raw)
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            image = part.get("image_url")
            if isinstance(image, dict):
                url = image.get("url")
            elif isinstance(image, str):
                url = image
            else:
                url = part.get("url") if part.get("type") in {"image", "image_url", "input_image"} else None
            if not isinstance(url, str) or url in seen:
                continue
            payload = _image_payload(url)
            if payload is None:
                continue
            images.append(payload)
            seen.add(url)
            if len(images) >= _MAX_IMAGES:
                return images
    return images


def extract_tool_calls(text: str, allowed_names: set[str]) -> tuple[list[Any], str]:
    if not isinstance(text, str) or not text:
        return [], ""
    calls: list[Any] = []
    consumed: list[tuple[int, int]] = []
    for match in _TOOL_BLOCK_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except (TypeError, json.JSONDecodeError):
            continue
        function = payload.get("function") if isinstance(payload, dict) else None
        name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if not name or name not in allowed_names:
            continue
        assert isinstance(function, dict)
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, dict):
            continue
        call_id = str(payload.get("id") or f"cursor_call_{len(calls) + 1}").strip()
        calls.append(
            SimpleNamespace(
                id=call_id,
                call_id=call_id,
                type="function",
                function=SimpleNamespace(name=name, arguments=arguments),
            )
        )
        consumed.append(match.span())
    if not consumed:
        return calls, text.strip()
    parts: list[str] = []
    cursor = 0
    for start, end in consumed:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return calls, "".join(parts).strip()


def _usage(result: Any) -> Any:
    usage = getattr(result, "usage", None)
    prompt = int(getattr(usage, "input_tokens", 0) or 0)
    completion = int(getattr(usage, "output_tokens", 0) or 0)
    cached = int(getattr(usage, "cache_read_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or prompt + completion)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _tool_delta(call: Any, index: int) -> Any:
    return SimpleNamespace(
        index=index,
        id=call.id,
        type="function",
        function=SimpleNamespace(name=call.function.name, arguments=call.function.arguments),
    )


class StreamChunks(list):
    """Small sync/async stream used because Cursor SDK returns a completed run."""

    def __aiter__(self):
        async def generate():
            for chunk in self:
                yield chunk
        return generate()

    async def aclose(self) -> None:
        return None

    def close(self) -> None:
        return None


def completion_to_chunks(completion: Any) -> StreamChunks:
    choice = completion.choices[0]
    message = choice.message
    tool_calls = [
        _tool_delta(call, index) for index, call in enumerate(message.tool_calls or [])
    ] or None
    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
    )
    return StreamChunks(
        [
            SimpleNamespace(
                id=completion.id,
                object="chat.completion.chunk",
                model=completion.model,
                choices=[SimpleNamespace(index=0, delta=delta, finish_reason=choice.finish_reason)],
                usage=None,
            ),
            SimpleNamespace(
                id=completion.id,
                object="chat.completion.chunk",
                model=completion.model,
                choices=[],
                usage=completion.usage,
            ),
        ]
    )


def _model_id(model: str | None) -> str | None:
    value = str(model or "").strip()
    for prefix in ("cursor/", "cursor:"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
            break
    return None if value.lower() in {"", "auto", "default"} else value


class _Completions:
    def __init__(self, client: "CursorSDKClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._client._create_chat_completion(**kwargs)
        return asyncio.to_thread(self._client._create_chat_completion, **kwargs)


class CursorSDKClient:
    """Hermes client seam backed only by Cursor's official SDK bridge."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: Any = None, base_url: str | None = None, **_: Any) -> None:
        self.api_key = _secret(api_key)
        supplied_url = str(base_url or CURSOR_API_URL).rstrip("/")
        if supplied_url != CURSOR_API_URL:
            raise ValueError(f"Cursor SDK provider only accepts {CURSOR_API_URL}")
        self.base_url = CURSOR_API_URL
        self.chat = SimpleNamespace(completions=_Completions(self))
        self.is_closed = False
        self._workspace = tempfile.TemporaryDirectory(prefix="hermes-cursor-sdk-")
        self._sdk: Any = None
        self._sdk_lock = threading.Lock()
        self._runs: set[Any] = set()
        self._runs_lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _sdk_client(self) -> Any:
        if self.is_closed:
            raise RuntimeError("Cursor SDK client is closed")
        with self._sdk_lock:
            if self._sdk is None:
                try:
                    CursorClient = importlib.import_module("cursor_sdk").CursorClient
                except ImportError as exc:
                    raise RuntimeError(
                        "The Cursor provider requires cursor-sdk. Reinstall the plugin so Hermes installs its dependencies."
                    ) from exc
                self._sdk = CursorClient.launch_bridge(
                    workspace=self._workspace.name,
                    state_root=str(Path(self._workspace.name) / "state"),
                    local={"cwd": self._workspace.name},
                    allow_api_key_env_fallback=False,
                )
            return self._sdk

    def cancel(self) -> None:
        with self._runs_lock:
            runs = tuple(self._runs)
        for run in runs:
            with suppress(Exception):
                run.cancel()

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        self.cancel()
        with self._sdk_lock:
            sdk, self._sdk = self._sdk, None
        if sdk is not None:
            with suppress(Exception):
                sdk.close()
        self._workspace.cleanup()

    def _run_sdk(
        self,
        *,
        prompt: str,
        images: list[dict[str, str]],
        model: str | None,
        timeout: float,
    ) -> Any:
        if not self.api_key:
            raise RuntimeError(
                "CURSOR_API_KEY is not configured. Create a user API key at https://cursor.com/dashboard/api, "
                "then run `hermes auth add cursor`."
            )
        client = self._sdk_client().with_options(
            timeout=timeout,
            unary_timeout=min(timeout, 30.0),
            stream_timeout=timeout,
        )
        options: dict[str, Any] = {
            "api_key": self.api_key,
            "local": {
                "cwd": self._workspace.name,
                "sandbox_options": {"enabled": True},
            },
            "mcp_servers": {},
            "tools": [],
        }
        selected_model = _model_id(model)
        if selected_model:
            options["model"] = selected_model
        agent = client.create_agent(options)
        run = None
        try:
            run = agent.send({"text": prompt, "images": images})
            with self._runs_lock:
                self._runs.add(run)
            return run.wait()
        finally:
            if run is not None:
                with self._runs_lock:
                    self._runs.discard(run)
            with suppress(Exception):
                agent.close()

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[Any] | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        prompt, allowed_names = render_prompt(
            messages or [], tools=tools, tool_choice=tool_choice
        )
        result = self._run_sdk(
            prompt=prompt,
            images=extract_images(messages or []),
            model=model,
            timeout=_timeout_seconds(timeout),
        )
        response_text = str(getattr(result, "result", "") or "")
        calls, content = extract_tool_calls(response_text, allowed_names)
        completion = SimpleNamespace(
            id=f"chatcmpl-cursor-{uuid.uuid4().hex}",
            object="chat.completion",
            model=str(model or getattr(getattr(result, "model", None), "id", "") or "cursor"),
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(
                        role="assistant",
                        content=content or None,
                        tool_calls=calls or None,
                        reasoning=None,
                        reasoning_content=None,
                        reasoning_details=None,
                    ),
                    finish_reason="tool_calls" if calls else "stop",
                )
            ],
            usage=_usage(result),
        )
        return completion_to_chunks(completion) if stream else completion


def list_cursor_models(*, api_key: str, timeout: float = 8.0) -> list[str]:
    """List account-visible models through the official SDK bridge."""
    if not _secret(api_key):
        return []
    try:
        CursorClient = importlib.import_module("cursor_sdk").CursorClient
    except ImportError:
        return []
    with tempfile.TemporaryDirectory(prefix="hermes-cursor-models-") as workspace:
        client = CursorClient.launch_bridge(
            workspace=workspace,
            state_root=str(Path(workspace) / "state"),
            timeout=timeout,
            client_timeout=timeout,
            allow_api_key_env_fallback=False,
        )
        try:
            return [str(model.id).strip() for model in client.list_models(api_key=api_key) if str(model.id).strip()]
        finally:
            client.close()
