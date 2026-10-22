"""The carried failover codes survive the HOST's classifier, not just our own hook.

Runs under the Hermes-core interpreter (``~/.hermes/hermes-agent/venv``); skips cleanly under the
plugin venv, which cannot import ``agent.error_classifier``.

Why this file exists: every earlier test called ``classify_cursor_error(error_code=…)`` directly —
or built the exception *with* a ``body`` — so it proved the hook's logic and said nothing about the
wiring. The host derives the code from ``exc.body`` (``_body_of`` -> ``_extract_error_code``), so an
exception that carried the code only as an attribute was classified fallback-eligible while all 178
tests stayed green. These tests raise the way production raises — ``error_code=`` and nothing else —
and assert what the host decides.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

hermes_cli = pytest.importorskip(
    "hermes_cli",
    exc_type=ModuleNotFoundError,
    reason="needs the Hermes core editable install",
)

# ``providers`` imports ``hermes_cli.update_channel``, which imports the repo-local ``pm`` package.
_CORE_ROOT = Path(hermes_cli.__file__).resolve().parents[1]
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

import providers
from agent.error_classifier import FailoverReason, classify_api_error

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "provider"
_MODULE = "_cursor_provider_tree_under_test"
_NAMES = {"cursor", "cursor-sdk", "cursor-subscription"}


def _load_tree_provider():
    module = sys.modules.get(_MODULE)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        _MODULE, PROVIDER / "__init__.py", submodule_search_locations=[str(PROVIDER)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree_profile(monkeypatch):
    module = _load_tree_provider()
    original = providers.get_provider_profile

    def resolve(name):
        if str(name).strip().lower() in _NAMES:
            return module.profile
        return original(name)

    monkeypatch.setattr(providers, "get_provider_profile", resolve)
    assert callable(module.profile.classify_api_error)
    return module


def _error(message, **kwargs):
    client = sys.modules[f"{_MODULE}.hermes_cursor_provider.client"]
    return client.CursorSDKRunError(message, **kwargs)


@pytest.mark.parametrize(
    "code, reason",
    [
        ("cursor_plan_only", FailoverReason.incomplete_response),
        ("cursor_invalid_params", FailoverReason.format_error),
    ],
)
def test_a_raised_code_reaches_the_host_without_the_caller_supplying_a_body(
    tree_profile, code, reason
):
    """The class attaches the body itself: production raises pass only ``error_code=``."""
    exc = _error("forced failure", status_code=400, error_code=code)
    verdict = classify_api_error(exc, provider="cursor", model="grok-4.7")
    assert verdict.reason == reason
    assert verdict.retryable is False
    assert verdict.should_fallback is False


def test_the_exception_body_carries_the_code_in_the_shape_the_host_reads(tree_profile):
    exc = _error("forced failure", status_code=400, error_code="cursor_plan_only")
    assert exc.body["error"]["code"] == "cursor_plan_only"
