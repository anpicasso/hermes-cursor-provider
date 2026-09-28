"""Translate OpenAI chat history into Cursor Agent protobuf requests.

Request shaping is derived from the MIT-licensed oh-my-pi Cursor provider and
NousResearch/hermes-agent PR #40876. See NOTICE.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from . import agent_pb2
from .constants import normalize_cursor_model_id

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
CURSOR_NATIVE_TOOL_NAMES = {"bash", "read", "write", "delete", "ls", "grep", "lsp", "todo"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts).strip()


def build_cursor_system_prompt_jsons(system_prompt: list[str] | str | None) -> list[str]:
    prompts = [system_prompt] if isinstance(system_prompt, str) else list(system_prompt or [])
    prompts = [item.strip() for item in prompts if isinstance(item, str) and item.strip()]
    if not prompts:
        prompts = [DEFAULT_SYSTEM_PROMPT]
    return [json.dumps({"role": "system", "content": prompt}) for prompt in prompts]


def create_blob_id(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def store_cursor_blob(blob_store: dict[str, bytes], data: bytes) -> bytes:
    blob_id = create_blob_id(data)
    blob_store[blob_id.hex()] = data
    return blob_id


def deterministic_message_id(key: str) -> str:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _root_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content.strip()}] if content.strip() else []
    if not isinstance(content, list):
        return []
    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and str(item.get("text") or "").strip():
            parts.append({"type": "text", "text": str(item["text"]).strip()})
        elif item.get("type") == "image_url":
            image = item.get("image_url")
            url = image.get("url", "") if isinstance(image, dict) else str(image or "")
            media_type = image.get("mime_type", "image/png") if isinstance(image, dict) else "image/png"
            if url:
                parts.append({"type": "image", "image": url, "mediaType": media_type})
    return parts


def _assistant_text(message: dict[str, Any]) -> str:
    parts = [_text(message.get("content"))]
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        parts.append(
            "[Hermes Tool Call]\n"
            + json.dumps(
                {
                    "id": call.get("id") or "",
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "{}",
                },
                sort_keys=True,
            )
        )
    return "\n".join(part for part in parts if part)


def _tool_result_text(message: dict[str, Any]) -> str:
    body = _text(message.get("content"))
    call_id = str(message.get("tool_call_id") or "")
    name = str(message.get("name") or "")
    label = " ".join(bit for bit in (f"id={call_id}" if call_id else "", f"name={name}" if name else "") if bit)
    return f"[Hermes Tool Result{(' ' + label) if label else ''}]\n{body}"


def _history_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    if role in {"user", "developer"}:
        content = _root_content(message.get("content"))
        return {"role": "user", "content": content} if content else None
    if role == "assistant":
        text = _assistant_text(message)
        return {"role": "assistant", "content": [{"type": "text", "text": text}]} if text else None
    if role in {"tool", "tool_result"}:
        return {"role": "user", "content": [{"type": "text", "text": _tool_result_text(message)}]}
    return None


def build_root_prompt_messages_json(
    messages: list[dict[str, Any]],
    system_prompt_ids: list[bytes],
    blob_store: dict[str, bytes],
    history_limit: int | None = None,
) -> list[bytes]:
    limit = len(messages) if history_limit is None else history_limit
    entries = list(system_prompt_ids)
    for message in messages[:limit]:
        payload = _history_payload(message)
        if payload:
            entries.append(store_cursor_blob(blob_store, json.dumps(payload, sort_keys=True).encode()))
    return entries


def create_cursor_user_message(text: str, message_id: str | None = None):
    return agent_pb2.UserMessage(text=text, message_id=message_id or str(uuid.uuid4()))


def build_conversation_turns(
    messages: list[dict[str, Any]],
    blob_store: dict[str, bytes],
    history_limit: int | None = None,
) -> list[bytes]:
    limit = len(messages) if history_limit is None else history_limit
    history = messages[:limit]
    turns: list[bytes] = []
    index = 0
    while index < len(history):
        message = history[index]
        if message.get("role") not in {"user", "developer"}:
            index += 1
            continue
        user_text = _text(message.get("content"))
        if not user_text:
            index += 1
            continue
        turn_number = len(turns)
        user = create_cursor_user_message(
            user_text,
            deterministic_message_id(f"u:{turn_number}:{user_text}"),
        )
        user_blob = store_cursor_blob(blob_store, user.SerializeToString())
        steps: list[bytes] = []
        index += 1
        while index < len(history) and history[index].get("role") not in {"user", "developer"}:
            item = history[index]
            text = _assistant_text(item) if item.get("role") == "assistant" else (
                _tool_result_text(item) if item.get("role") in {"tool", "tool_result"} else ""
            )
            if text:
                step = agent_pb2.ConversationStep(assistant_message=agent_pb2.AssistantMessage(text=text))
                steps.append(store_cursor_blob(blob_store, step.SerializeToString()))
            index += 1
        turn = agent_pb2.ConversationTurnStructure(
            agent_conversation_turn=agent_pb2.AgentConversationTurnStructure(
                user_message=user_blob,
                steps=steps,
            )
        )
        turns.append(store_cursor_blob(blob_store, turn.SerializeToString()))
    return turns


def _active_action(messages: list[dict[str, Any]]) -> tuple[Any, int]:
    if not messages:
        return agent_pb2.ConversationAction(resume_action=agent_pb2.ResumeAction()), 0
    last = messages[-1]
    if last.get("role") in {"user", "developer"}:
        return agent_pb2.ConversationAction(
            user_message_action=agent_pb2.UserMessageAction(user_message=create_cursor_user_message(_text(last.get("content"))))
        ), len(messages) - 1
    if last.get("role") in {"tool", "tool_result"}:
        start = len(messages) - 1
        while start and messages[start - 1].get("role") in {"tool", "tool_result"}:
            start -= 1
        result_text = "\n\n".join(_tool_result_text(item) for item in messages[start:])
        return agent_pb2.ConversationAction(
            user_message_action=agent_pb2.UserMessageAction(user_message=create_cursor_user_message(result_text))
        ), start
    return agent_pb2.ConversationAction(resume_action=agent_pb2.ResumeAction()), len(messages)


def build_mcp_tool_definitions(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for raw in tools or []:
        tool = raw.get("function") if raw.get("type") == "function" else raw
        tool = tool if isinstance(tool, dict) else {}
        name = str(tool.get("name") or "")
        if not name or name in CURSOR_NATIVE_TOOL_NAMES:
            continue
        definitions.append(
            {
                "name": name,
                "description": str(tool.get("description") or ""),
                "providerIdentifier": "hermes-agent",
                "toolName": name,
                "inputSchema": tool.get("parameters") or {"type": "object", "properties": {}, "required": []},
            }
        )
    return definitions


def encode_tool_input_schema(schema: dict[str, Any] | None) -> bytes:
    from google.protobuf import json_format, struct_pb2

    value = struct_pb2.Value()
    json_format.ParseDict(schema or {"type": "object", "properties": {}, "required": []}, value)
    return value.SerializeToString()


def build_run_request_bytes(
    *,
    messages: list[dict[str, Any]],
    system_prompt: list[str] | str | None,
    tools: list[dict[str, Any]] | None,
    model_id: str,
    conversation_id: str,
    blob_store: dict[str, bytes] | None = None,
    conversation_state: Any | None = None,
    custom_system_prompt: str | None = None,
) -> tuple[bytes, Any, dict[str, bytes]]:
    del tools  # Tool definitions travel in RequestContext.
    blobs: dict[str, bytes] = {}
    system_ids = [store_cursor_blob(blobs, item.encode()) for item in build_cursor_system_prompt_jsons(system_prompt)]
    action, history_limit = _active_action(messages)
    roots = build_root_prompt_messages_json(messages, system_ids, blobs, history_limit)
    turns = build_conversation_turns(messages, blobs, history_limit)

    if conversation_state is not None:
        state = agent_pb2.ConversationStateStructure()
        state.CopyFrom(conversation_state)
        del state.root_prompt_messages_json[:]
        state.root_prompt_messages_json.extend(roots)
        del state.turns[:]
        state.turns.extend(turns)
    else:
        state = agent_pb2.ConversationStateStructure(
            root_prompt_messages_json=roots,
            turns=turns,
            todos=[], pending_tool_calls=[], previous_workspace_uris=[],
            file_states={}, file_states_v2={}, summary_archives=[], turn_timings=[],
            subagent_states={}, self_summary_count=0, read_paths=[],
        )

    model = normalize_cursor_model_id(model_id)
    request = agent_pb2.AgentRunRequest(
        conversation_state=state,
        action=action,
        model_details=agent_pb2.ModelDetails(model_id=model, display_model_id=model, display_name=model),
        conversation_id=conversation_id,
    )
    if custom_system_prompt:
        request.custom_system_prompt = custom_system_prompt
    payload = agent_pb2.AgentClientMessage(run_request=request).SerializeToString()
    if blob_store is not None:
        blob_store.clear()
        blob_store.update(blobs)
        blobs = blob_store
    return payload, state, blobs
