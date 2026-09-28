"""Standalone Cursor model-provider plugin for Hermes Agent current main."""
from __future__ import annotations

import threading
import time
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

from .hermes_cursor_provider.credentials import auth_handler, refresh_credential
from .hermes_cursor_provider.cursor_protocol.constants import CURSOR_API_URL

_SEED_MODELS = (
    "default",
    "composer-2.5",
    "claude-4.6-opus-high",
    "claude-4.5-sonnet",
    "gpt-5.4-medium",
)
_CATALOG = _SEED_MODELS
_CATALOG_AT = 0.0
_CATALOG_LOCK = threading.Lock()


def _pool_token() -> str:
    try:
        from agent.credential_pool import load_pool
        for entry in load_pool("cursor").entries():
            if entry.access_token and entry.last_status != "dead":
                return str(entry.access_token)
    except Exception:
        pass
    return ""


def _catalog(api_key: str = "", *, timeout: float = 12.0, force: bool = False) -> tuple[str, ...]:
    global _CATALOG, _CATALOG_AT
    if not force and time.monotonic() - _CATALOG_AT < 300:
        return _CATALOG
    if not _CATALOG_LOCK.acquire(blocking=False):
        return _CATALOG
    try:
        token = (api_key or _pool_token()).strip()
        if token:
            try:
                from .hermes_cursor_provider.cursor_protocol.catalog import fetch_cursor_usable_models
                live = tuple(fetch_cursor_usable_models(api_key=token, timeout=timeout))
                if live:
                    _CATALOG = live
            except Exception:
                pass
        _CATALOG_AT = time.monotonic()
        return _CATALOG
    finally:
        _CATALOG_LOCK.release()


class CursorProfile(ProviderProfile):
    @property
    def fallback_models(self) -> tuple[str, ...]:
        # Core reads this during picker rendering; never block the UI on discovery.
        return _CATALOG

    @fallback_models.setter
    def fallback_models(self, value: tuple[str, ...]) -> None:
        global _CATALOG
        _CATALOG = tuple(value or _SEED_MODELS)

    def create_client(self, **kwargs: Any):
        from .hermes_cursor_provider.client import CursorClient
        return CursorClient(**kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 12.0,
    ) -> list[str] | None:
        # The protocol endpoint is fixed; never forward a bearer token to a custom URL.
        if base_url and str(base_url).rstrip("/") != CURSOR_API_URL:
            return None
        return list(_catalog(api_key or "", timeout=timeout, force=True))

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        del context
        return {"_cursor_session_id": session_id} if session_id else {}


profile = CursorProfile(
    name="cursor",
    aliases=("cursor-agent", "cursor-subscription"),
    display_name="Cursor (native)",
    description="Cursor subscription through the private Agent protocol; Hermes owns tool execution.",
    signup_url="https://cursor.com",
    api_mode="chat_completions",
    base_url=CURSOR_API_URL,
    hostname="api2.cursor.sh",
    auth_type="oauth_external",
    auth_handler=auth_handler,
    refresh_credential=refresh_credential,
    supports_health_check=False,
    supports_model_listing=True,
    supports_vision=False,
    fallback_models=_SEED_MODELS,
    model_aliases={"auto": "default", "composer": "composer-2.5"},
    model_capabilities={
        model: {"supports_vision": False, "supports_tools": True, "context_window": 200000}
        for model in _SEED_MODELS
    },
    default_aux_model="default",
)
register_provider(profile)
