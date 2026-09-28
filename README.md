# hermes-cursor-provider

Standalone Cursor subscription provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), built for the current model-provider plugin API with **zero Hermes core changes**.

It uses Cursor's official [`cursor-sdk`](https://cursor.com/docs/sdk/python) package and its bundled local SDK bridge. A Cursor user API key charges runs to the same Cursor plan/request pool as SDK usage in Cursor.

## What it does

- Registers `cursor` through `ProviderProfile.create_client()`.
- Uses `CURSOR_API_KEY` through Hermes' normal API-key credential flow.
- Lists account-visible models through the official SDK.
- Converts Hermes chat history, images, and tool schemas into SDK agent requests.
- Disables Cursor built-in tools and MCP servers, gives each SDK agent a private working directory, and accepts only tool names offered by Hermes.
- Returns OpenAI-shaped sync/async responses so Hermes retains its normal conversation, approval, and tool loop.

Cursor's SDK is an agent surface, not a raw chat-completions API. The plugin retains one SDK agent per Hermes conversation/account/model, sends only newly appended messages, and translates explicit `<tool_call>` blocks back into Hermes tool calls. SDK custom tools are not used because they would execute outside Hermes' approval path.

## Install

```bash
hermes plugins install 'https://github.com/anpicasso/hermes-cursor-provider.git#provider' --enable
hermes auth add cursor
hermes model
```

Create a **user API key** at <https://cursor.com/dashboard/api>. User and service-account API keys are supported by Cursor's SDK; Team Admin keys are not inference credentials.

Plugins and credentials are profile-scoped:

```bash
hermes --profile work plugins install 'https://github.com/anpicasso/hermes-cursor-provider.git#provider' --enable
hermes --profile work auth add cursor
```

Test without changing the configured default model:

```bash
hermes chat --provider cursor --model auto -q 'Reply with exactly: OK'
```

A running gateway may retain the previous provider import. After install or upgrade, restart it from an external shell:

```bash
hermes gateway restart
```

## Migrating from 0.1.x

Version 0.2 removes the private `api2.cursor.sh` protocol, browser-token OAuth flow, protobuf payloads, and emulated CLI identity. Run `hermes auth add cursor` once to store a Cursor user API key. Old `CURSOR_ACCESS_TOKEN` credentials are not used.

## Security model

- Credentials are passed only to the official SDK; custom provider URLs are rejected before bridge launch.
- One SDK-owned loopback bridge is shared per Hermes profile process and closed at process exit.
- Every retained SDK agent gets a private working directory. Sessions are isolated by Hermes profile, conversation, account-key digest, and model.
- Cursor built-in tools and MCP servers are disabled. Hermes executes only allowlisted tool names through its own loop.
- Only HTTPS or bounded `data:image/*;base64` image inputs are forwarded.

See [SECURITY.md](SECURITY.md) for reporting and residual risks.

## Current limitations

- Cursor SDK `1.x` is proprietary beta software and may change; the plugin pins the supported major version.
- The plugin does not request Cursor's optional local sandbox because the SDK rejects it on unsupported hosts. Security does not depend on that sandbox: Cursor built-in tools, custom tools, and MCP servers remain disabled, and Hermes owns tool execution.
- Streaming is synthesized after the SDK run completes because the plugin preserves a simple OpenAI-compatible boundary.
- Tool calls use a prompt-level contract because the official SDK's custom-tool callbacks execute inside the SDK run rather than Hermes' normal approval loop.
- Up to 16 idle conversation agents are retained per process by default (`HERMES_CURSOR_MAX_SESSIONS` changes the LRU target); active turns can temporarily exceed it. A history rewrite/compaction starts a fresh agent; concurrent turns on one conversation use an isolated one-shot agent.
- Agent reuse ends on process restart. Cross-process SDK resume is deliberately not enabled yet.
- Cursor cache reads and writes are reported through Hermes usage metadata; Cursor's billed `Agent.get_usage()` endpoint may remain unavailable for some accounts.
- Live inference needs a real Cursor API key and is not exercised by the public CI suite.

## Development

```bash
python -m venv .venv
.venv/bin/pip install 'cursor-sdk>=1.0.32,<2' pytest
.venv/bin/pytest -q
hermes plugins validate provider --json
hermes plugins doctor provider --ci
```

## License

The plugin is MIT licensed. `cursor-sdk` is a separately distributed proprietary dependency governed by Cursor's license and terms. See [NOTICE](NOTICE).
