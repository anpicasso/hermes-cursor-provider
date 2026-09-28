# hermes-cursor-provider

Standalone Cursor subscription provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), built against the current plugin API with **zero Hermes core changes**.

> [!WARNING]
> This plugin speaks Cursor's undocumented `api2.cursor.sh` Agent protocol and identifies as Cursor's CLI. It is unofficial, may break without notice, and may conflict with [Cursor's Terms of Service](https://cursor.com/terms-of-service), including section 1.5. Using it may put your Cursor account at risk. Prefer Cursor's official SDK/CLI surfaces when that trade-off is unacceptable.

## What it does

- Registers `cursor` through `ProviderProfile.create_client()` on Hermes Agent `>=0.21.4`.
- Uses Cursor's browser login and Hermes' native credential pool/refresh lifecycle.
- Discovers the account's live Cursor models, with offline picker fallbacks.
- Streams text, reasoning, usage, and structured tool calls through Hermes' normal chat-completions loop.
- Advertises Hermes tools as MCP tools but **rejects every Cursor-native execution request**. Hermes remains responsible for approvals, policy, and execution.
- Keeps bearer tokens bound to the fixed `https://api2.cursor.sh` origin and honors standard CA/proxy environment variables.

The protocol/request logic was reworked from [NousResearch/hermes-agent#40876](https://github.com/NousResearch/hermes-agent/pull/40876) for current Hermes main. The old PR is implementation evidence only; none of its core patches are required.

## Install

```bash
hermes plugins install 'https://github.com/anpicasso/hermes-cursor-provider.git#provider' --enable
hermes auth add cursor
hermes model
```

Plugins and credentials are profile-scoped:

```bash
hermes --profile work plugins install 'https://github.com/anpicasso/hermes-cursor-provider.git#provider' --enable
hermes --profile work auth add cursor
```

To test without changing your configured default model:

```bash
hermes chat --provider cursor --model default -q 'Reply with exactly: OK'
```

A running gateway may retain the old provider import. After installing or upgrading, restart it from an external shell:

```bash
hermes gateway restart
```

## Configuration

- `CURSOR_CLIENT_VERSION`: override the emulated Cursor CLI build string when Cursor changes its protocol gate.
- `HTTPS_PROXY` / `NO_PROXY`: HTTP CONNECT proxy routing.
- `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, `SSL_CERT_DIR`: custom TLS trust.

Only `https://api2.cursor.sh:443` is accepted as a protocol endpoint. A configured custom URL is rejected before credentials are sent.

## Current limitations

- The wire protocol and embedded protobuf descriptor are private and version-fragile.
- Cursor-native tools are intentionally unavailable. A turn must surface a Hermes MCP tool call, stop, and continue after Hermes returns the result.
- Hermes current main does not route `oauth_external` model-provider clients into every auxiliary path. Configure a separate auxiliary provider if compression/title generation needs one.
- Vision is not advertised because the private request envelope has not been safely validated for active-image turns.
- Live inference requires a real Cursor account and was not exercised by the repository's offline test suite.

## Development

```bash
python -m venv .venv
.venv/bin/pip install 'h2>=4.2,<5' 'protobuf>=6.33.5,<7' 'httpx>=0.28,<1' pytest
.venv/bin/pytest -q
hermes plugins validate provider --json
hermes plugins doctor provider --ci
```

Tested against Hermes main `9b02a977bd670af61ec974ae298a11abd1a17a35`.

## Provenance and license

MIT licensed. Third-party attribution and protocol provenance are in [NOTICE](NOTICE). This project is not affiliated with or endorsed by Cursor, Anysphere, Nous Research, or the oh-my-pi maintainers.
