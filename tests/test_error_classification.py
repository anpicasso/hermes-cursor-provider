"""A plan-only failure must not be answered silently by a different model.

The recovery loop nudges twice when a model announces an action and then produces no
tool call, and fails closed when the nudges are exhausted. That failure was
indistinguishable from a transport error, so the host could retry it and, on the next
fallback provider, answer the user with a different model without saying so. These
tests pin the structured code, the profile classification that stops it, and that every
other failure is left to the host's own retry and fallback.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

from hermes_cursor_provider import client as client_module
from hermes_cursor_provider.client import CursorSDKClient, classify_cursor_error

from test_client import _tool, fake_sdk  # noqa: F401  (autouse fixture of test_client)


def _plan_only_error(client):
    """Stream a reply that only announces an action, with a tool offered."""
    sdk = client._sdk_client()
    sdk.result.result = "I'll read the file and then tell you what it says."
    with pytest.raises(client_module.CursorSDKRunError) as excinfo:
        list(
            client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "read the file"}],
                tools=[_tool()],
                stream=True,
            )
        )
    return excinfo.value


def test_recovery_exhaustion_carries_a_structured_code(fake_sdk):  # noqa: F811
    """The fail-closed raise is identifiable by code, not by message text."""
    client = CursorSDKClient(api_key="crsr_test")
    try:
        error = _plan_only_error(client)
    finally:
        client.close()
    assert error.error_code == "cursor_plan_only"


def test_plan_only_is_classified_terminal_and_does_not_fall_back():
    """The verdict stops the retry and the cross-provider answer."""
    assert classify_cursor_error(error_code="cursor_plan_only") == {
        "reason": "format_error",
        "retryable": False,
        "should_fallback": False,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"error_code": "cursor_run_timeout"},
        {"error_code": "cursor_sdk_run_failed"},
        {"error_code": ""},
        {},
        {"status_code": 429, "message": "out of usage"},
        {"status_code": 502, "message": "Cursor SDK run ended with status error"},
    ],
)
def test_every_other_failure_is_left_to_the_host(kwargs):
    """Timeouts, quotas and unknown failures keep the host's retry and fallback."""
    assert classify_cursor_error(**kwargs) is None


def test_the_classification_is_registered_on_the_profile(monkeypatch):
    """A verdict nobody wired into the profile is dead code."""
    providers = types.ModuleType("providers")
    base = types.ModuleType("providers.base")

    class ProviderProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    base.ProviderProfile = ProviderProfile
    providers.base = base
    registered = []
    providers.register_provider = registered.append
    monkeypatch.setitem(sys.modules, "providers", providers)
    monkeypatch.setitem(sys.modules, "providers.base", base)
    sys.modules.pop("provider", None)

    provider = importlib.import_module("provider")

    assert registered and registered[0] is provider.profile
    verdict = provider.profile.classify_api_error
    assert verdict.__name__ == "classify_cursor_error"
    assert verdict(error_code="cursor_plan_only") == {
        "reason": "format_error",
        "retryable": False,
        "should_fallback": False,
    }
    assert verdict(status_code=429, message="out of usage") is None
