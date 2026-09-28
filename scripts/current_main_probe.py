"""Offline current-main provider seam probe."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hermes-cursor-probe-") as raw_home:
        home = Path(raw_home)
        target = home / "plugins" / "cursor-provider"
        target.parent.mkdir(parents=True)
        shutil.copytree(ROOT / "provider", target)
        os.environ["HERMES_HOME"] = str(home)
        os.environ["CURSOR_API_KEY"] = "offline-probe-token"

        import providers

        profile = providers.get_provider_profile("cursor")
        assert profile is not None
        assert profile.auth_type == "api_key"
        assert profile.auth_handler is None
        assert profile.refresh_credential is None
        assert profile.env_vars == ("CURSOR_API_KEY",)
        assert profile.build_api_kwargs_extras(
            session_id="session-1", cache_scope_id="scope-1"
        ) == ({}, {"_cursor_session_scope": "scope-1"})
        assert profile.build_api_kwargs_extras(session_id="session-1") == (
            {}, {"_cursor_session_scope": "session-1"}
        )
        assert profile.build_api_kwargs_extras() == ({}, {})

        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(
            requested="cursor",
            explicit_api_key="offline-probe-token",
            explicit_base_url="https://api.cursor.com",
            target_model="auto",
        )
        assert runtime["provider"] == "cursor"
        assert runtime["api_key"] == "offline-probe-token"
        assert runtime["base_url"] == "https://api.cursor.com"
        assert runtime["api_mode"] == "chat_completions"

        from run_agent import AIAgent
        from agent.chat_completion_helpers import build_api_kwargs

        agent = AIAgent(
            api_key=runtime["api_key"],
            base_url=runtime["base_url"],
            provider="cursor",
            model="auto",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        setattr(agent, "session_id", "offline-session")
        request = build_api_kwargs(
            agent, [{"role": "user", "content": "hello"}], []
        )
        assert request.get("_cursor_session_scope")
        agent.close()

        client = profile.create_client(api_key=runtime["api_key"], base_url=runtime["base_url"])
        assert client is not None
        assert client.chat.completions.create
        assert client.api_key == "offline-probe-token"
        assert client.HERMES_SKIP_TRANSPORT_WRAP
        assert client.HERMES_SKIP_ASYNC_WRAP
        client.close()

        from agent.auxiliary_client import resolve_provider_client

        aux_client, aux_model = resolve_provider_client(
            "cursor",
            model="auto",
            async_mode=True,
            explicit_api_key="offline-probe-token",
            explicit_base_url="https://api.cursor.com",
        )
        assert aux_client is not None
        assert aux_model == "auto"
        assert aux_client.HERMES_SKIP_ASYNC_WRAP
        aux_client.close()

        print(json.dumps({
            "ok": True,
            "hermes_source": str(Path(providers.__file__).resolve()),
            "provider": runtime["provider"],
            "api_mode": runtime["api_mode"],
            "base_url": runtime["base_url"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
