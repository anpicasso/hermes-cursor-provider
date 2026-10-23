"""Official Cursor SDK model-provider plugin for Hermes Agent."""
from __future__ import annotations

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

from .hermes_cursor_provider.client import CURSOR_API_URL, classify_cursor_error, _context_window

_log = logging.getLogger(__name__)
_SEED_MODELS = ("auto", "composer-2.5")


class CursorProfile(ProviderProfile):
    def create_client(self, **kwargs: Any):
        from .hermes_cursor_provider.client import CursorSDKClient

        return CursorSDKClient(**kwargs)

    def get_model_context_length(self, model: str) -> int | None:
        # Explicit `context=...` parameter beats the host's generic family default.
        try:
            return _context_window(model)
        except Exception:
            return None

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        # Never forward a Cursor key through a caller-supplied endpoint.
        if base_url and str(base_url).rstrip("/") != CURSOR_API_URL:
            return None
        try:
            from .hermes_cursor_provider.client import list_cursor_models

            models = list_cursor_models(api_key=api_key or "", timeout=timeout)
        except Exception as exc:
            _log.warning(
                "Cursor model listing failed (%s: %s); showing seed models only",
                type(exc).__name__, exc,
            )
            return None
        return models or None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        scope = str(context.get("cache_scope_id") or context.get("session_id") or "").strip()
        return ({}, {"_cursor_session_scope": scope}) if scope else ({}, {})


profile = CursorProfile(
    name="cursor",
    aliases=("cursor-sdk", "cursor-subscription"),
    display_name="Cursor SDK",
    description="Cursor subscription through the official Python SDK; Hermes owns tool execution.",
    signup_url="https://cursor.com/dashboard/api",
    api_mode="chat_completions",
    base_url=CURSOR_API_URL,
    hostname="api.cursor.com",
    auth_type="api_key",
    env_vars=("CURSOR_API_KEY",),
    supports_health_check=False,
    supports_model_listing=True,
    supports_vision=True,
    fallback_models=_SEED_MODELS,
    model_aliases={"default": "auto", "composer": "composer-2.5"},
    model_capabilities={
        model: {"supports_vision": True, "supports_tools": True}
        for model in _SEED_MODELS
    },
    default_aux_model="auto",
    unsupported_response_formats=("json_schema", "json_object"),
    # Constructor kwarg, never a method: the dataclass field default would shadow a
    # same-named method in the generated __init__.
    classify_api_error=classify_cursor_error,
)
register_provider(profile)
