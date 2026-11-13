"""Issue 11: a tool turn that ends in inert protocol text must recover or fail closed.

The upstream recovery loop only counted a rejected block while the stream was still
held. Once prose had been streamed, a malformed or unterminated ``<tool_call>`` body
was emitted as ordinary answer text, so Hermes executed nothing and saw ``finish=stop``.
These tests pin the attempt-level detection, the native-miss narration path, and the
configured one-retry bound.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_recovery import CALL, MALFORMED_CALL, TOOLS, OfflineRun, OfflineSDK


class NativeMissRun(OfflineRun):
    """One run that emits a native tool name the request never offered."""

    def stream(self):
        yield SimpleNamespace(
            type="tool_call", name="shell", status="completed",
            args={"command": "id"}, call_id="native-miss", result=None,
        )
        yield from super().stream()


class Issue11Tests(unittest.TestCase):
    def setUp(self):
        from hermes_cursor_provider import client as adapter

        self.adapter = adapter
        adapter._reset_session_registry()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.environment = patch.dict(os.environ, {"HERMES_HOME": self.directory.name})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.sdk = OfflineSDK()
        self.loader = patch.object(adapter, "_load_cursor_client", return_value=SimpleNamespace(
            launch_bridge=lambda **kwargs: self.sdk))
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.client = adapter.CursorSDKClient(api_key="offline-only")
        self.addCleanup(self.client.close)
        self.addCleanup(adapter._reset_session_registry)

    def request(self, **kwargs):
        return self.client.chat.completions.create(
            messages=[{"role": "user", "content": "Recap the payroll work."}],
            tools=TOOLS, **kwargs,
        )

    @staticmethod
    def _parts(chunks):
        content = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in chunks if chunk.choices and chunk.choices[0].delta.content
        )
        calls = [
            call for chunk in chunks if chunk.choices
            for call in (chunk.choices[0].delta.tool_calls or [])
        ]
        return content, calls

    def test_unterminated_block_after_streamed_prose_is_retried_and_hidden(self):
        leftover = (
            "The plan is ready. <tool_call>"
            + json.dumps({"id": "bad", "function": {
                "name": "skill_view", "arguments": {"name": "payroll"}}})
        )
        self.sdk.runs.extend([OfflineRun(leftover), OfflineRun(CALL)])
        content, calls = self._parts(list(self.request(stream=True)))
        self.assertEqual(content, "The plan is ready.")
        self.assertEqual([call.function.name for call in calls], ["skill_view"])
        self.assertEqual(len(self.sdk.messages), 2)
        self.assertNotIn("<tool_call>", content)

    def test_malformed_block_after_streamed_prose_is_retried_and_hidden(self):
        text = "The plan is ready. " + MALFORMED_CALL
        self.sdk.runs.extend([OfflineRun(text), OfflineRun(CALL)])
        content, calls = self._parts(list(self.request(stream=True)))
        self.assertEqual(content, "The plan is ready.")
        self.assertEqual([call.function.name for call in calls], ["skill_view"])
        self.assertEqual(len(self.sdk.messages), 2)

    def test_failed_tool_turn_warning_is_metadata_only(self):
        sentinel = "PROMPT-SENTINEL-DO-NOT-LOG"
        self.sdk.runs.extend([OfflineRun(sentinel + " " + MALFORMED_CALL), OfflineRun(CALL)])
        with self.assertLogs("hermes_cursor_provider", level="WARNING") as captured:
            _content, calls = self._parts(list(self.request(stream=True)))
        self.assertEqual([call.function.name for call in calls], ["skill_view"])
        warnings = [line for line in captured.output if "failed tool turn" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("rejected=1", warnings[0])
        self.assertIn("native_misses=0", warnings[0])
        self.assertNotIn(sentinel, warnings[0])

    def test_nonstream_leftover_also_recovers(self):
        leftover = (
            "The plan is ready. <tool_call>"
            + json.dumps({"id": "bad", "function": {
                "name": "skill_view", "arguments": {"name": "payroll"}}})
        )
        self.sdk.runs.extend([OfflineRun(leftover), OfflineRun(CALL)])
        response = self.request()
        self.assertIsNone(response.choices[0].message.content)
        self.assertEqual(
            [call.function.name for call in response.choices[0].message.tool_calls],
            ["skill_view"],
        )
        self.assertEqual(len(self.sdk.messages), 2)

    def test_native_miss_narration_gets_two_nudges(self):
        narration = "Native channel failed — switching to plain-text tool calls."
        self.sdk.runs.extend([
            NativeMissRun(narration), NativeMissRun(narration), OfflineRun(CALL),
        ])
        content, calls = self._parts(list(self.request(stream=True)))
        self.assertEqual(content, "")
        self.assertEqual([call.function.name for call in calls], ["skill_view"])
        self.assertEqual(len(self.sdk.messages), 3)

    def test_rejected_only_exhaustion_fails_closed_instead_of_returning_protocol_text(self):
        self.sdk.runs.extend(OfflineRun(MALFORMED_CALL) for _ in range(3))
        with self.assertRaisesRegex(self.adapter.CursorSDKRunError, "recovery exhausted") as caught:
            list(self.request(stream=True))
        self.assertEqual(caught.exception.error_code, "cursor_plan_only")
        self.assertEqual(len(self.sdk.messages), 3)

    def test_one_retry_env_restores_the_old_bound(self):
        narration = "Native channel failed — switching to plain-text tool calls."
        with patch.dict(os.environ, {"HERMES_CURSOR_MAX_TOOL_RETRIES": "1"}):
            self.sdk.runs.extend([NativeMissRun(narration), NativeMissRun(narration)])
            with self.assertRaisesRegex(self.adapter.CursorSDKRunError, "recovery exhausted"):
                list(self.request(stream=True))
        self.assertEqual(len(self.sdk.messages), 2)


if __name__ == "__main__":
    unittest.main()
