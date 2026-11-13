# Security Policy

## Supported version

Only the latest release is supported. `cursor-sdk` is a proprietary beta dependency; upgrade compatibility is tested within the supported `1.x` range.

## Reporting

Report vulnerabilities privately through GitHub Security Advisories. Do not open a public issue containing Cursor API keys, bridge tokens, account identifiers, private prompts, or tool results.

## Security model

- `CURSOR_API_KEY` is stored through Hermes' normal API-key flow and passed only to Cursor's official SDK.
- The provider rejects any runtime base URL other than `https://api.cursor.com`; it never forwards a key to caller-supplied hosts.
- The SDK-owned bridge is loopback-only and authenticated. One bridge is shared per Hermes profile process and closed at process exit.
- Retained agents are keyed by Hermes profile, conversation, account-key digest, and model. Raw API keys never appear in registry keys or paths.
- Each agent gets a private working directory. The plugin does not request Cursor's optional local sandbox because unsupported hosts reject the entire run; isolation instead relies on disabled Cursor built-in tools, no MCP servers, and Hermes-owned tool execution. The registry retains 16 idle sessions by default (plus temporarily busy turns); eviction calls SDK `Agent.close()`, never destructive `Agent.delete()`.
- Cursor built-in tools are explicitly set to an empty list. SDK custom tools are off by default; the opt-in native-tools mode registers only the Hermes tools offered in the current request, with an inert executor, and Hermes still runs every call. No MCP servers are loaded.
- Returned tool calls are parsed only for names included in the current Hermes request, and arguments must be a JSON object. Hermes applies its normal approvals and execution policy afterward.
- Image URLs must be HTTPS. Data images must be valid base64 and are size-bounded.
- The plugin does not log prompts, tool payloads, API keys, or bridge credentials.

## SDK bridge process exposure

`cursor-sdk` launches its bridge with a per-process `--tool-callback-auth-token` on the command line. On a host without `hidepid`, any same-uid process can read `/proc/<pid>/cmdline`; a Hermes gateway's kanban workers share that uid. The exposed value is **not** the Cursor API key: it authorizes only the loopback custom-tool callback. With native tools off no custom tools are registered, and with them on the executor returns a fixed acknowledgement, so the plugin's exposure is limited to that inert callback. The plugin cannot fix the argv construction; `Bridge.launch` is SDK-owned. The fix belongs upstream in `cursor-sdk` (stdin or inherited-fd configuration).

The SDK also copies the gateway environment into the bridge process, so exporting `CURSOR_API_KEY` there exposes the account key through `/proc/<pid>/environ` to the same-uid processes. Use Hermes' credential pool (`hermes auth add cursor`) rather than exporting the key in the gateway environment; the plugin sends the scoped key explicitly on `CreateAgent`, `ListModels`, and `Send`.

## Residual risk

Cursor's SDK is an agent surface, not a raw inference endpoint. Although built-in tools are disabled and each workspace is private, server-side model behavior and the proprietary bridge remain outside Hermes' control. SDK runs consume the user's Cursor plan and remain subject to Cursor's terms, privacy settings, rate limits, and account controls.

SDK `Agent.close()` and `CursorClient.close()` release local handles and the bridge; they do not delete Cursor conversation state or server-side prompt-cache entries. The plugin never calls `Agent.delete()`. Cache effectiveness remains provider-controlled and must be measured from authenticated `cache_read_tokens` usage.
