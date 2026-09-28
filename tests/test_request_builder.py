import json

from hermes_cursor_provider.cursor_protocol import agent_pb2
from hermes_cursor_provider.cursor_protocol.request_builder import (
    build_mcp_tool_definitions,
    build_root_prompt_messages_json,
    build_run_request_bytes,
)


def _decode(blobs, ids):
    return [json.loads(blobs[item.hex()].decode()) for item in ids]


def test_tool_result_becomes_active_user_action():
    messages = [
        {"role": "user", "content": "check"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"x"}'},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "contents"},
    ]
    payload, state, blobs = build_run_request_bytes(
        messages=messages, system_prompt="system", tools=[], model_id="default", conversation_id="conv",
    )
    envelope = agent_pb2.AgentClientMessage()
    envelope.ParseFromString(payload)
    action = envelope.run_request.action
    assert action.WhichOneof("action") == "user_message_action"
    assert "call-1" in action.user_message_action.user_message.text
    assert "contents" in action.user_message_action.user_message.text
    roots = _decode(blobs, list(state.root_prompt_messages_json))
    assert any("Hermes Tool Call" in json.dumps(item) for item in roots)
    assert not any("contents" in json.dumps(item) for item in roots)


def test_only_non_native_tools_are_advertised():
    tools = [
        {"type": "function", "function": {"name": "bash", "parameters": {}}},
        {"type": "function", "function": {"name": "read_file", "description": "read", "parameters": {"type": "object"}}},
    ]
    definitions = build_mcp_tool_definitions(tools)
    assert [item["name"] for item in definitions] == ["read_file"]
    assert definitions[0]["providerIdentifier"] == "hermes-agent"
