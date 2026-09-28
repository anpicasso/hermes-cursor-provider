"""Cursor protocol constants."""
from __future__ import annotations

import os

CURSOR_API_URL = "https://api2.cursor.sh"
CURSOR_LOGIN_URL = "https://cursor.com/loginDeepControl"
CURSOR_POLL_URL = f"{CURSOR_API_URL}/auth/poll"
CURSOR_REFRESH_URL = f"{CURSOR_API_URL}/auth/exchange_user_api_key"
CURSOR_AGENT_RUN_PATH = "/agent.v1.AgentService/Run"
CURSOR_GET_USABLE_MODELS_PATH = "/agent.v1.AgentService/GetUsableModels"
CURSOR_CLIENT_VERSION = os.getenv("CURSOR_CLIENT_VERSION", "cli-2026.07.23-e383d2b")
CONNECT_END_STREAM_FLAG = 0b00000010

CURSOR_MODEL_ID_ALIASES = {
    "cursor-composer-2.5": "composer-2.5",
    "cursor-composer": "composer-2.5",
}


def normalize_cursor_model_id(model_id: str) -> str:
    normalized = (model_id or "").strip()
    return CURSOR_MODEL_ID_ALIASES.get(normalized, normalized)
