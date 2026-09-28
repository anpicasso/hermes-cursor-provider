"""Cursor browser login and Hermes credential-pool integration."""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import uuid
import webbrowser
from typing import Any
from urllib.parse import urlencode

from .cursor_protocol.constants import CURSOR_API_URL, CURSOR_LOGIN_URL, CURSOR_POLL_URL, CURSOR_REFRESH_URL


def _httpx():
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Cursor provider requires httpx>=0.28") from exc
    return httpx


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_auth_params() -> dict[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    auth_uuid = str(uuid.uuid4())
    query = urlencode({"challenge": challenge, "uuid": auth_uuid, "mode": "login", "redirectTarget": "cli"})
    return {"verifier": verifier, "uuid": auth_uuid, "login_url": f"{CURSOR_LOGIN_URL}?{query}"}


def _tokens(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Cursor auth response was not a JSON object")
    access = str(payload.get("access_token") or payload.get("accessToken") or payload.get("token") or "").strip()
    refresh = str(payload.get("refresh_token") or payload.get("refreshToken") or access).strip()
    if not access:
        raise ValueError("Cursor auth response did not contain an access token")
    return access, refresh


def token_expiry_ms(token: str) -> int | None:
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(part.encode()).decode())
        expiry = payload.get("exp")
        return int(float(expiry) * 1000) if expiry is not None else None
    except Exception:
        return None


def poll_login(auth_uuid: str, verifier: str, *, timeout: float = 300.0) -> tuple[str, str]:
    httpx = _httpx()
    deadline = time.monotonic() + timeout
    delay = 1.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                CURSOR_POLL_URL,
                params={"uuid": auth_uuid, "verifier": verifier},
                timeout=20.0,
                follow_redirects=False,
            )
        except Exception as exc:
            last_error = exc
        else:
            if response.status_code == 404 or response.status_code >= 500:
                last_error = RuntimeError(f"Cursor login poll returned HTTP {response.status_code}")
            elif response.status_code >= 400:
                raise RuntimeError(f"Cursor login poll failed with HTTP {response.status_code}")
            else:
                return _tokens(response.json())
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)
    raise TimeoutError("Cursor login timed out waiting for browser approval") from last_error


def refresh_token(token: str) -> dict[str, Any]:
    httpx = _httpx()
    response = httpx.post(
        CURSOR_REFRESH_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        json={},
        timeout=20.0,
        follow_redirects=False,
    )
    if response.status_code in {401, 403}:
        from hermes_cli.auth_constants import AuthError
        raise AuthError(
            "Cursor refresh grant is no longer valid; run `hermes auth add cursor`.",
            provider="cursor", code="invalid_grant", relogin_required=True, retryable=False,
        )
    if response.status_code >= 400:
        from hermes_cli.auth_constants import AuthError
        raise AuthError(
            f"Cursor token refresh failed with HTTP {response.status_code}",
            provider="cursor", code=f"http_{response.status_code}", retryable=response.status_code >= 500,
        )
    access, refresh = _tokens(response.json())
    return {"access_token": access, "refresh_token": refresh, "expires_at_ms": token_expiry_ms(access)}


def refresh_credential(entry: Any) -> dict[str, Any]:
    token = str(getattr(entry, "refresh_token", None) or getattr(entry, "access_token", "") or "")
    if not token:
        from hermes_cli.auth_constants import AuthError
        raise AuthError(
            "Cursor credential has no refresh token; run `hermes auth add cursor`.",
            provider="cursor", code="missing_refresh_token", relogin_required=True, retryable=False,
        )
    return refresh_token(token)


def auth_handler(action: str, args: Any) -> bool:
    if action != "add":
        return False

    params = generate_auth_params()
    print(f"Open this URL to sign in to Cursor:\n\n{params['login_url']}\n")
    if not bool(getattr(args, "no_browser", False)):
        try:
            webbrowser.open(params["login_url"])
        except Exception:
            pass
    print("Waiting for Cursor approval...")
    try:
        access, refresh = poll_login(params["uuid"], params["verifier"])
    except Exception as exc:
        from hermes_cli.auth_constants import AuthError
        if isinstance(exc, AuthError):
            raise
        raise AuthError(str(exc), provider="cursor", code="login_failed", retryable=False) from exc

    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool

    pool = load_pool("cursor")
    label = str(getattr(args, "label", "") or "").strip() or f"cursor-account-{len(pool.entries()) + 1}"
    entry = PooledCredential(
        provider="cursor",
        id=uuid.uuid4().hex[:6],
        label=label,
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:cursor_browser",
        access_token=access,
        refresh_token=refresh,
        expires_at_ms=token_expiry_ms(access),
        base_url=CURSOR_API_URL,
        inference_base_url=CURSOR_API_URL,
    )
    pool.add_entry(entry)
    print(f'Added Cursor credential "{label}".')
    return True
