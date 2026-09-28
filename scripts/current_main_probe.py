"""Offline current-main provider seam probe."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hermes-cursor-probe-") as raw_home:
        home = Path(raw_home)
        target = home / "plugins" / "cursor-provider"
        target.parent.mkdir(parents=True)
        shutil.copytree(ROOT / "provider", target)
        os.environ["HERMES_HOME"] = str(home)

        import providers
        profile = providers.get_provider_profile("cursor")
        assert profile is not None
        assert profile.auth_handler is not None
        assert profile.refresh_credential is not None
        assert profile.build_extra_body(session_id="probe") == {"_cursor_session_id": "probe"}

        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
        pool = load_pool("cursor")
        pool.add_entry(PooledCredential(
            provider="cursor", id=uuid.uuid4().hex[:6], label="probe",
            auth_type=AUTH_TYPE_OAUTH, priority=0, source="manual:probe",
            access_token="offline-probe-token", refresh_token="offline-probe-refresh",
            base_url="https://api2.cursor.sh", inference_base_url="https://api2.cursor.sh",
        ))

        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested="cursor", target_model="default")
        assert runtime["provider"] == "cursor"
        assert runtime["api_key"] == "offline-probe-token"
        assert runtime["base_url"] == "https://api2.cursor.sh"
        assert runtime["api_mode"] == "chat_completions"

        client = profile.create_client(api_key=runtime["api_key"], base_url=runtime["base_url"])
        assert client.chat.completions.create
        assert client.api_key == "offline-probe-token"
        client.close()

        print(json.dumps({
            "ok": True,
            "hermes_source": str(Path(providers.__file__).resolve()),
            "provider": runtime["provider"],
            "api_mode": runtime["api_mode"],
            "base_url": runtime["base_url"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
