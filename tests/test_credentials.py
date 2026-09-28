from types import SimpleNamespace

from hermes_cursor_provider import credentials


def test_refresh_credential_rotates_tokens(monkeypatch):
    response = SimpleNamespace(
        status_code=200,
        json=lambda: {"accessToken": "new-access", "refreshToken": "new-refresh"},
    )
    fake_httpx = SimpleNamespace(post=lambda *args, **kwargs: response)
    monkeypatch.setattr(credentials, "_httpx", lambda: fake_httpx)
    result = credentials.refresh_credential(SimpleNamespace(refresh_token="old", access_token="access"))
    assert result["access_token"] == "new-access"
    assert result["refresh_token"] == "new-refresh"


def test_poll_login_retries_pending(monkeypatch):
    responses = iter([
        SimpleNamespace(status_code=404),
        SimpleNamespace(status_code=200, json=lambda: {"token": "access"}),
    ])
    fake_httpx = SimpleNamespace(get=lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(credentials, "_httpx", lambda: fake_httpx)
    monkeypatch.setattr(credentials.time, "sleep", lambda _: None)
    assert credentials.poll_login("uuid", "verifier", timeout=2) == ("access", "access")


def test_poll_login_retries_transient_transport_error(monkeypatch):
    responses = iter([
        OSError("temporary network failure"),
        SimpleNamespace(status_code=200, json=lambda: {"token": "access"}),
    ])

    def get(*args, **kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(credentials, "_httpx", lambda: SimpleNamespace(get=get))
    monkeypatch.setattr(credentials.time, "sleep", lambda _: None)
    assert credentials.poll_login("uuid", "verifier", timeout=2) == ("access", "access")
