# Bounded tool-call recovery

The adapter detects a narrow class of short, unconditional first-person action
announcements that end without a validated Hermes tool request. It also recovers
unsatisfied required or named tool choices when matching tools are available.

Recovery sends at most two corrective nudges after the initial SDK run. Only a
validated, allowlisted tool request satisfies recovery; a completion claim or
malformed request does not. Exhaustion raises an explicit `CursorSDKRunError`
instead of reporting successful completion. Failed preliminary text is withheld
from the returned response and Hermes session history. Ordinary answers,
questions, conditional offers, disabled-tools calls, and auxiliary text requests
remain unaffected. Native Cursor tools and MCP remain disabled.

The detector deliberately favors false negatives. It recognizes one sentence
with a fixed first-person action-verb prefix, at most 320 characters / 60 words,
and rejects questions, conditional wording, and clause continuations after a
colon, semicolon, or spaced dash. It is not a general guarantee that every
incomplete model answer can be detected. Malformed or unoffered tool blocks are
classified separately from prose: a turn that only announces an action and then
emits an unusable block is retried or fails explicitly instead of returning raw
protocol.

## Verification

The regression suite covers the reported action-only skill-loading response,
successful corrective tool requests, bounded exhaustion, invalid/unoffered tool
requests, required and named choices, normal answers and streaming, async paths,
and retained-session continuation after a tool result.

```sh
python -m unittest discover -s tests -p test_recovery.py -v
python -m pytest -q
hermes plugins validate provider --json
hermes plugins doctor provider --ci
```

Regression tests were observed failing before implementation. The full suite
passes with 62 tests and 747 subtests before integration with current main.
The original PR author reported authenticated smoke tests that exercised
`auto` and `grok-4.7`: each requested a harmless diagnostic echo tool, the local
harness executed it, and the model returned its message after receiving the
actual tool result. These normal live round trips do not prove that every model
stall will recover; forced recovery and exhaustion are covered offline.

Related upstream work: `anpicasso/hermes-cursor-provider#1` addresses overlapping
tool-call handling more broadly. This is an independent, targeted recovery patch;
no code from that PR was imported.
