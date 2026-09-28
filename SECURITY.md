# Security Policy

## Supported version

Only the latest release is supported. The upstream protocol is private and may change without notice.

## Reporting

Report vulnerabilities privately through GitHub Security Advisories for this repository. Do not open a public issue containing access tokens, refresh tokens, captured protocol frames, account identifiers, or private prompts.

## Security model

- Credentials are stored through Hermes' native credential pool; the plugin has no private token store.
- Tokens are sent only to `https://api2.cursor.sh:443`; custom endpoints are rejected.
- TLS verification is enabled. Standard CA bundle and HTTP CONNECT proxy settings are honored.
- Cursor-native read/write/delete/shell/fetch/computer-use/MCP execution requests fail closed.
- Only tool names declared by Hermes are exposed in Cursor's request context, and returned calls execute later through Hermes' normal tool loop.
- No prompt, token, or protocol payload is logged by the plugin.

The private protocol itself is unsupported by Cursor. Treat protocol changes, account suspension, and server-side behavior outside Hermes' control as residual risk.
