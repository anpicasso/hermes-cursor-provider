"""Offline recovery regressions, runnable with stdlib unittest (no SDK required)."""
from __future__ import annotations

import asyncio
from collections import deque
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provider"))
from hermes_cursor_provider import client as adapter


ANNOUNCEMENT = (
    "I'll load the payroll organization skill so the recap matches the work "
    "already done and the goal you described."
)
TOOLS = [{"type": "function", "function": {
    "name": "skill_view", "parameters": {"type": "object"},
}}]
CALL = '<tool_call>' + json.dumps({"id": "recovered", "function": {
    "name": "skill_view", "arguments": {"name": "payroll-record-organization"},
}}) + '</tool_call>'


class OfflineRun:
    def __init__(self, text, fragments=None):
        self.text = text
        self.fragments = fragments if fragments is not None else [text]
        self.cancelled = False

    def stream(self):
        for text in self.fragments:
            if callable(text):
                text = text()
            yield SimpleNamespace(type="assistant", message=SimpleNamespace(
                content=[SimpleNamespace(text=text)]))

    def wait(self):
        return SimpleNamespace(result=self.text, status="FINISHED", usage=SimpleNamespace(
            input_tokens=4, output_tokens=2), model=SimpleNamespace(id="auto"))

    def cancel(self):
        self.cancelled = True


class OfflineAgent:
    def __init__(self, sdk):
        self.sdk = sdk
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)
        self.sdk.messages.append(message)
        if not self.sdk.runs:
            raise AssertionError("Unexpected additional SDK send")
        return self.sdk.runs.popleft()

    def close(self):
        self.closed = True


class OfflineSDK:
    def __init__(self):
        self.runs = deque()
        self.agents = []
        self.options = []
        self.messages = []

    def with_options(self, **kwargs):
        return self

    def create_agent(self, options):
        self.options.append(options)
        agent = OfflineAgent(self)
        self.agents.append(agent)
        return agent

    def close(self):
        pass


class RecoveryTests(unittest.TestCase):
    def setUp(self):
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
            messages=[{"role": "user", "content": "Recap the payroll organization work."}],
            tools=TOOLS, **kwargs)

    def test_recovery_is_capped_at_two_nudges_and_invalidates_failed_session(self):
        self.sdk.runs.extend(OfflineRun(ANNOUNCEMENT) for _ in range(4))
        chunks = []
        with self.assertRaisesRegex(adapter.CursorSDKRunError, "recovery exhausted"):
            for chunk in self.request(stream=True, _cursor_session_scope="exhausted"):
                chunks.append(chunk)
        self.assertEqual(len(self.sdk.messages), 3)
        self.assertEqual(chunks, [])
        self.assertEqual(len(self.sdk.runs), 1)
        session = next(iter(adapter._SESSIONS.values()))
        self.assertIsNone(session.agent)
        self.assertEqual(session.history, [])
        self.assertFalse(session.lock.locked())
        self.assertTrue(self.sdk.agents[0].closed)
        self.assertFalse(self.client._runs)

    def test_required_and_named_choices_recover_even_without_action_announcement(self):
        for choice in ("required", {"type": "function", "function": {"name": "skill_view"}}):
            for streaming in (False, True):
                with self.subTest(choice=choice, streaming=streaming):
                    self.sdk.runs.extend([OfflineRun("A plain answer."), OfflineRun(CALL)])
                    response = self.request(tool_choice=choice, stream=streaming)
                    if streaming:
                        chunks = list(response)
                        self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks if c.choices), "")
                        calls = [call for c in chunks if c.choices for call in (c.choices[0].delta.tool_calls or [])]
                    else:
                        calls = response.choices[0].message.tool_calls
                        self.assertIsNone(response.choices[0].message.content)
                    self.assertEqual(calls[0].function.name, "skill_view")
                    self.assertFalse(self.sdk.runs)

    def test_recovery_requires_a_valid_call_not_a_completion_claim_or_unoffered_call(self):
        for failed_retry in ("Done.", CALL.replace("skill_view", "shell"), CALL.replace(
                '"arguments": {"name": "payroll-record-organization"}', '"arguments": "not-json"')):
            with self.subTest(failed_retry=failed_retry):
                self.sdk.runs.clear()
                self.sdk.runs.extend([OfflineRun(ANNOUNCEMENT), OfflineRun(failed_retry), OfflineRun(CALL)])
                chunks = list(self.request(stream=True))
                self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks if c.choices), "")
                calls = [call for c in chunks if c.choices for call in (c.choices[0].delta.tool_calls or [])]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].function.name, "skill_view")
                self.assertFalse(self.sdk.runs)

    def test_recovered_stream_preserves_session_history_and_continues_after_tool_result(self):
        self.sdk.runs.extend([
            OfflineRun(ANNOUNCEMENT, [ANNOUNCEMENT[:2], ANNOUNCEMENT[2:13], ANNOUNCEMENT[13:]]),
            OfflineRun(CALL, [CALL[:8], CALL[8:37], CALL[37:]]),
        ])
        chunks = list(self.request(stream=True, _cursor_session_scope="recovered-history"))
        self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks if c.choices), "")
        calls = [call for c in chunks if c.choices for call in (c.choices[0].delta.tool_calls or [])]
        self.assertEqual(len(calls), 1)
        session = next(iter(adapter._SESSIONS.values()))
        expected = {"role": "assistant", "content": "", "tool_calls": [{
            "id": "recovered", "type": "function", "function": {
                "name": "skill_view", "arguments": '{"name":"payroll-record-organization"}',
            }}]}
        self.assertEqual(session.history[-1], expected)
        self.assertNotIn(ANNOUNCEMENT, json.dumps(session.history))
        history = list(session.history) + [{"role": "tool", "tool_call_id": "recovered", "content": "skill-data"}]
        self.sdk.runs.append(OfflineRun("Here is the recap."))
        response = self.client.chat.completions.create(
            messages=history, tools=TOOLS, _cursor_session_scope="recovered-history")
        self.assertEqual(response.choices[0].message.content, "Here is the recap.")
        self.assertEqual(len(self.sdk.agents), 1)
        self.assertEqual(len(self.sdk.messages), 3)
        self.assertIn("skill-data", self.sdk.messages[-1]["text"])
        self.assertNotIn("Recap the payroll", self.sdk.messages[-1]["text"])

    def test_ordinary_answers_questions_and_conditional_offers_are_not_retried(self):
        examples = [
            "The files are organized by year.", "Which folder should I inspect?",
            "I can load the payroll skill if you want.", "I'll load it if you want.",
            "If needed, I'll check the files.", "I'll load it when you confirm.",
            "Would you like me to check the files?", "I'll explain how payroll filing works.",
            "I'll check the files. The existing records are already organized.",
            "I'll read the file.\nHere is an explanation, not an action request.",
            "I'll load it after you approve.", "I'll check the records should you need it.",
            "I'll load the skill upon approval.", "I'll run through the explanation.",
            "I'll check " + ("the payroll records " * 30).strip(),
        ]
        for text in examples:
            with self.subTest(text=text):
                self.sdk.runs.append(OfflineRun(text))
                count = len(self.sdk.messages)
                response = self.request()
                self.assertEqual(response.choices[0].message.content, text)
                self.assertEqual(response.choices[0].finish_reason, "stop")
                self.assertEqual(len(self.sdk.messages), count + 1)

    def test_no_tools_disabled_tools_and_title_requests_do_not_recover(self):
        for tools, choice in ((None, None), ([], "required"), (TOOLS, "none")):
            with self.subTest(tools=tools, choice=choice):
                self.sdk.runs.append(OfflineRun(ANNOUNCEMENT))
                count = len(self.sdk.messages)
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": "Generate a session title."}],
                    tools=tools, tool_choice=choice)
                self.assertEqual(response.choices[0].message.content, ANNOUNCEMENT)
                self.assertEqual(len(self.sdk.messages), count + 1)
        self.sdk.runs.append(OfflineRun("Payroll organization recap"))
        response = self.request()
        self.assertEqual(response.choices[0].message.content, "Payroll organization recap")

    def test_split_conditional_keyword_in_action_target_never_leaks(self):
        self.sdk.runs.extend([
            OfflineRun("I'll inspect the should_retry helper.",
                       ["I'll inspect the should", "_retry helper."]),
            OfflineRun(CALL),
        ])
        chunks = list(self.request(stream=True))
        self.assertEqual("".join(c.choices[0].delta.content or ""
                                 for c in chunks if c.choices), "")
        self.assertEqual(len(self.sdk.messages), 2)
        calls = [call for c in chunks if c.choices
                 for call in (c.choices[0].delta.tool_calls or [])]
        self.assertEqual(len(calls), 1)

    def test_recovery_candidates_stay_private_at_every_split_boundary(self):
        texts = ["I'll inspect the " + keyword + "_helper."
                 for keyword in ("if", "unless", "could", "would", "want", "once",
                                 "when", "after", "before", "should", "upon")]
        texts += ["I'll run through_helper.", "  I'll inspect the helper.\n\t  ",
                  "I'll inspect the helper! \n", "I’ll inspect the helper.  "]
        for text in texts:
            self.assertTrue(adapter._action_only(text))
            for split in range(1, len(text)):
                with self.subTest(text=text, split=split):
                    self.sdk.runs.extend([
                        OfflineRun(text, [text[:split], text[split:]]), OfflineRun(CALL),
                    ])
                    chunks = list(self.request(stream=True))
                    self.assertEqual("".join(c.choices[0].delta.content or ""
                                             for c in chunks if c.choices), "")
                    self.assertFalse(self.sdk.runs)
                    self.assertIn("tool_calls", [c.choices[0].finish_reason
                                                 for c in chunks if c.choices])

    def test_irrevocable_rejections_release_without_waiting_for_full_response(self):
        prefixes = ["I'll inspect it?", "I'll inspect it. More", "I'll inspect it! More",
                    "I'll inspect it.\nMore", "I'll inspect it\nMore",
                    "I'll inspect it should you", "I'll run through the"]
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                delivered = []

                def rest():
                    self.assertEqual("".join(delivered), prefix)
                    return " explanation."

                text = prefix + " explanation."
                self.sdk.runs.append(OfflineRun(text, [prefix, rest]))
                count = len(self.sdk.messages)
                for chunk in self.request(stream=True):
                    if chunk.choices and chunk.choices[0].delta.content:
                        delivered.append(chunk.choices[0].delta.content)
                self.assertEqual("".join(delivered), text)
                self.assertEqual(len(self.sdk.messages), count + 1)

    def test_cancellation_during_corrective_run_cleans_up_without_leaking(self):
        started = threading.Event()
        cancelled = threading.Event()

        class BlockingRetry(OfflineRun):
            def stream(self):
                started.set()
                if not cancelled.wait(5):
                    raise AssertionError("Corrective run was not cancelled")
                raise adapter.CursorSDKRunError("Corrective run cancelled")
                yield  # Keep the SDK stream lazy.

            def cancel(self):
                super().cancel()
                cancelled.set()

        retry = BlockingRetry("")
        self.sdk.runs.extend([OfflineRun(ANNOUNCEMENT), retry, OfflineRun(CALL)])
        stream = self.request(stream=True, _cursor_session_scope="cancel-corrective")
        chunks, errors = [], []

        def consume():
            try:
                chunks.extend(stream)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        try:
            self.assertTrue(started.wait(5), "Corrective run did not start")
            stream.close()
        finally:
            cancelled.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertTrue(retry.cancelled)
        self.assertEqual(chunks, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], adapter.CursorSDKRunError)
        self.assertEqual(len(self.sdk.messages), 2)
        self.assertEqual(len(self.sdk.runs), 1)
        self.assertFalse(self.client._runs)
        session = next(iter(adapter._SESSIONS.values()))
        self.assertIsNone(session.agent)
        self.assertEqual(session.history, [])
        self.assertFalse(session.lock.locked())
        self.assertTrue(self.sdk.agents[0].closed)

    def test_tools_available_ordinary_text_still_streams_incrementally(self):
        delivered = []

        def rest():
            self.assertEqual(delivered, ["The records"])
            return " are organized."

        self.sdk.runs.append(OfflineRun("The records are organized.", ["The records", rest]))
        for chunk in self.request(stream=True):
            if chunk.choices and chunk.choices[0].delta.content:
                delivered.append(chunk.choices[0].delta.content)
        self.assertEqual("".join(delivered), "The records are organized.")

    def test_required_no_call_exhaustion_is_an_explicit_nonstream_error(self):
        self.sdk.runs.extend(OfflineRun("A plain answer.") for _ in range(3))
        with self.assertRaisesRegex(adapter.CursorSDKRunError, "required or announced") as caught:
            self.request(tool_choice="required")
        self.assertEqual(caught.exception.status_code, 502)
        self.assertEqual(caught.exception.error_code, "cursor_sdk_run_failed")
        self.assertEqual(len(self.sdk.messages), 3)
        self.assertTrue(self.sdk.agents[0].closed)

    def test_named_choice_rejects_a_different_offered_tool_during_recovery(self):
        other_tool = {"type": "function", "function": {"name": "read_file"}}
        self.sdk.runs.extend([OfflineRun(CALL.replace("skill_view", "read_file")), OfflineRun(CALL)])
        response = self.client.chat.completions.create(
            messages=[], tools=TOOLS + [other_tool],
            tool_choice={"type": "function", "function": {"name": "skill_view"}},
            _cursor_session_scope="named-choice")
        self.assertEqual(response.choices[0].message.tool_calls[0].function.name, "skill_view")
        self.assertIsNone(response.choices[0].message.content)
        self.assertEqual(len(self.sdk.messages), 2)

    def test_async_completion_and_stream_recover_without_failed_text(self):
        self.sdk.runs.extend([OfflineRun(ANNOUNCEMENT), OfflineRun(CALL),
                              OfflineRun(ANNOUNCEMENT), OfflineRun(CALL)])

        async def scenario():
            response = await self.request()
            stream = await self.request(stream=True)
            return response, [chunk async for chunk in stream]

        response, chunks = asyncio.run(scenario())
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")
        self.assertIsNone(response.choices[0].message.content)
        self.assertEqual("".join(c.choices[0].delta.content or "" for c in chunks if c.choices), "")
        self.assertEqual(len(self.sdk.messages), 4)
        self.assertFalse(self.sdk.runs)

    def test_exact_payroll_announcement_recovers_to_validated_call(self):
        self.sdk.runs.extend([OfflineRun(ANNOUNCEMENT), OfflineRun(CALL)])
        response = self.request()
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")
        self.assertEqual(response.choices[0].message.tool_calls[0].function.name, "skill_view")
        self.assertIsNone(response.choices[0].message.content)
        self.assertEqual(len(self.sdk.messages), 2)
        self.assertEqual(self.sdk.options[0]["tools"], [])
        self.assertEqual(self.sdk.options[0]["mcp_servers"], {})


if __name__ == "__main__":
    unittest.main()
