# Security Policy

## Supported version

Only the latest release is supported. `cursor-sdk` is a proprietary beta dependency; upgrade compatibility is tested within the supported `1.x` range.

## Reporting

Report vulnerabilities privately through GitHub Security Advisories. Do not open a public issue containing Cursor API keys, bridge tokens, account identifiers, private prompts, or tool results.

## Security model

- `CURSOR_API_KEY` is stored through Hermes' normal API-key flow and passed only to Cursor's official SDK.
- The provider rejects any runtime base URL other than `https://api.cursor.com`; it never forwards a key to caller-supplied hosts.
- The SDK-owned bridge is loopback-only, authenticated, and closed with the provider client.
- Each agent runs in a private temporary workspace with sandboxing enabled.
- Cursor built-in tools are explicitly set to an empty list; no SDK custom tools or MCP servers are registered.
- Returned tool calls are parsed only for names included in the current Hermes request, and arguments must be a JSON object. Hermes applies its normal approvals and execution policy afterward.
- Image URLs must be HTTPS. Data images must be valid base64 and are size-bounded.
- The plugin does not log prompts, tool payloads, API keys, or bridge credentials.

## Residual risk

Cursor's SDK is an agent surface, not a raw inference endpoint. Although built-in tools are disabled and the workspace is isolated, server-side model behavior and the proprietary bridge remain outside Hermes' control. SDK runs consume the user's Cursor plan and remain subject to Cursor's terms, privacy settings, rate limits, and account controls.
