# Changelog

## 0.2.0

- Replaced the reverse-engineered Cursor Agent protocol with Cursor's official Python SDK and bundled bridge.
- Replaced browser-token OAuth with `CURSOR_API_KEY`, enabling official subscription-backed SDK usage and Hermes auxiliary routing.
- Removed vendored protobuf, HTTP/2 framing, private endpoints, token refresh code, and CLI impersonation.
- Added isolated sandboxed workspaces, disabled Cursor built-in tools/MCP, allowlisted Hermes tool-call translation, vision inputs, official model discovery, and sync/async client support.
- Reworked documentation, security model, tests, and current-main verification for the SDK architecture.

## 0.1.0

- Initial standalone current-main Hermes model-provider plugin using Cursor's private Agent protocol.
