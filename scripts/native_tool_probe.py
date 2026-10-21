#!/usr/bin/env python3
"""Live probe (needs a real CURSOR_API_KEY): does the model call a Hermes tool natively or via <tool_call>?

Run on the Hermes server with the plugin installed:
  HERMES_CURSOR_DEBUG_EVENTS=1 HERMES_HOME=/tmp/probe-home CURSOR_API_KEY=... \
  python scripts/native_tool_probe.py [model]
Then read /tmp/probe-home/logs/cursor-provider-events.jsonl for the raw SDK events.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider"))

from hermes_cursor_provider.client import CursorSDKClient  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "kanban_create",
            "description": "Create a kanban card for a profile.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "assignee": {"type": "string"}},
                "required": ["title", "assignee"],
            },
        },
    }
]


def main() -> int:
    key = os.environ.get("CURSOR_API_KEY", "")
    if not key:
        print("Set CURSOR_API_KEY first.")
        return 2
    model = sys.argv[1] if len(sys.argv) > 1 else "auto"
    client = CursorSDKClient(api_key=key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Create a kanban card titled 'probe' assigned to researcher."}],
            tools=TOOLS,
        )
    finally:
        client.close()
    message = response.choices[0].message
    calls = [{"name": c.function.name, "arguments": c.function.arguments} for c in (message.tool_calls or [])]
    print(json.dumps({"finish_reason": response.choices[0].finish_reason, "text": message.content, "tool_calls": calls}, indent=2))
    return 0 if calls else 1


if __name__ == "__main__":
    raise SystemExit(main())
