"""Offline current-main provider seam probe."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_managed_stream(profile) -> None:
    """Exercise the real Relay consumer, not only provider registration."""
    from types import SimpleNamespace
    import threading
    from agent import relay_llm, relay_runtime
    from agent.chat_completion_helpers_relay import RelayChatAccumulator

    client = profile.create_client(api_key="offline-probe-token")
    calls = []
    allow_finish = threading.Event()
    completed = threading.Event()

    def send(message):
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AssertionError("SDK work must run off the Relay event loop")
        calls.append(message)
        return Run()

    class Run:
        def stream(self):
            yield SimpleNamespace(type="assistant", message=SimpleNamespace(content=[SimpleNamespace(text="relay-")]))
            assert allow_finish.wait(5), "Relay withheld text until completion"
            yield SimpleNamespace(type="assistant", message=SimpleNamespace(content=[SimpleNamespace(text="ok")]))

        def wait(self):
            completed.set()
            return SimpleNamespace(result="relay-ok", status="FINISHED", usage=None)

        def cancel(self):
            allow_finish.set()

    client._create_agent = lambda **kw: SimpleNamespace(send=send, close=lambda: None)
    coordinator = relay_runtime.SESSION_COORDINATOR
    lease = coordinator.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="relay-probe", platform="cli")
    turn = coordinator.begin_turn(lease, turn_id="turn-1", task_id="probe")
    lease.host.retain_managed_execution("cursor-probe")
    accumulator = RelayChatAccumulator()
    stream = None
    try:
        stream = relay_llm.stream(
            {"model": "auto", "messages": [{"role": "user", "content": "hello"}], "stream": True},
            lambda request: client.chat.completions.create(**request),
            session_id="relay-probe", name="cursor", model_name="auto",
            finalizer=accumulator.finalize, on_chunk=accumulator.observe,
            completed_response_predicate=lambda value: hasattr(value, "choices"),
            metadata={"api_mode": "chat_completions"},
        )
        first = next(iter(stream))
        assert first.choices[0].delta.content == "relay-"
        assert not completed.is_set()
        allow_finish.set()
        chunks = [first, *stream]
        assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "relay-ok"
        assert chunks[-1].usage.total_tokens == 0
        assert len(calls) == 1
    finally:
        allow_finish.set()
        if stream is not None:
            stream.close()
        client.close()
        lease.host.release_managed_execution("cursor-probe")
        coordinator.end_turn(turn, outcome="success")
        coordinator.release_conversation(lease)
        relay_runtime._reset_for_tests()


def _usage_stub_agent():
    """Minimal agent surface ``agent.turn_usage.record_response_usage`` reads."""
    from types import SimpleNamespace

    class Compressor:
        threshold_tokens = 500_000

        def __init__(self):
            self.seen = []

        def update_from_response(self, usage_dict):
            self.seen.append(usage_dict)

    return SimpleNamespace(
        context_compressor=Compressor(),
        model="grok-4.7", provider="cursor", api_mode="chat_completions",
        base_url="https://api.cursor.com", api_key="offline-probe-token",
        session_id="", quiet_mode=True, verbose_logging=False, log_prefix="",
        platform="cli", _session_db=None, _session_db_created=True,
        session_api_calls=0, session_prompt_tokens=0, session_completion_tokens=0,
        session_total_tokens=0, session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_reasoning_tokens=0, session_estimated_cost_usd=0.0,
        session_cost_status=None, session_cost_source=None,
        _last_turn_usage=None, _last_prompt_size_tokens=0,
        _api_latency_history=[], _api_output_history=[],
        _safe_print=lambda *a, **k: None, _vprint=lambda *a, **k: None,
        _turn_route_task="",
    )


def verify_usage_seam(profile) -> None:
    """Billed totals and the one-turn context anchor reach Hermes on separate fields."""
    from types import SimpleNamespace
    from agent.turn_usage import record_response_usage
    from agent.relay_llm import _jsonable, _namespace

    client = profile.create_client(api_key="offline-probe-token")
    turns = [
        SimpleNamespace(type="usage", usage=SimpleNamespace(
            input_tokens=tokens, output_tokens=output,
            cache_read_tokens=cached, cache_write_tokens=0))
        for tokens, output, cached in ((240_000, 10, 3), (241_000, 10, 4), (242_000, 20, 5))
    ]
    result = SimpleNamespace(
        result="ok", status="FINISHED", model=None,
        usage=SimpleNamespace(input_tokens=723_000, output_tokens=40,
                              cache_read_tokens=12, cache_write_tokens=0),
    )

    def stream():
        yield from turns
        yield SimpleNamespace(type="assistant", message=SimpleNamespace(
            content=[SimpleNamespace(text="ok")]))

    class Run:
        def stream(self):
            return stream()

        def wait(self):
            return result

        def cancel(self):
            pass

    run = Run()
    client._create_agent = lambda **kw: SimpleNamespace(
        send=lambda message: run, close=lambda: None)
    try:
        response = client.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "hello"}])
        usage = response.usage
        assert usage.prompt_tokens == 723_000, "billing lost the summed attempts"
        assert usage.completion_tokens == 40
        assert usage.prompt_tokens_details.cached_tokens == 12
        assert usage.initial_prompt_tokens == 242_000, "anchor summed retry attempts"
        assert usage.initial_cached_tokens == 5

        chunks = list(client.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "hello"}], stream=True))
        anchor = _namespace(_jsonable(chunks[-1].usage))
        assert anchor.prompt_tokens == 723_000
        assert anchor.initial_prompt_tokens == 242_000, "Relay normalization dropped the anchor"
        assert anchor.initial_cached_tokens == 5

        # Relay's recorded response is built from JSON-normalized chunk dicts, then read back
        # as attributes; both steps must keep the separate context fields.
        from agent.chat_completion_helpers_relay import RelayChatAccumulator

        accumulator = RelayChatAccumulator()
        for chunk in chunks:
            accumulator.observe(_jsonable(chunk))
        recorded = accumulator.finalize()["usage"]
        assert recorded["prompt_tokens"] == 723_000
        assert recorded["initial_prompt_tokens"] == 242_000
        assert recorded["initial_cached_tokens"] == 5

        agent = _usage_stub_agent()
        record_response_usage(
            agent, response, messages=[{"role": "user", "content": "hello"}],
            api_call_count=1, api_duration=0.1, compression_attempts=0,
            max_compression_attempts=3,
        )
        assert agent.session_prompt_tokens == 723_000, "session billing lost attempts"
        assert agent.session_cache_read_tokens == 12
        assert agent._last_turn_usage["prompt_tokens"] == 242_000, "context used billed total"
        assert agent.context_compressor.seen[-1]["prompt_tokens"] == 242_000
    finally:
        client.close()


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
        variant_model = "grok-4.7@reasoning_effort=xhigh,fast=true"
        setattr(agent, "model", variant_model)
        variant_request = build_api_kwargs(
            agent, [{"role": "user", "content": "hello"}], []
        )
        assert variant_request["model"] == variant_model
        assert variant_request.get("_cursor_session_scope")
        assert profile.build_api_kwargs_extras(
            reasoning_config={"effort": "xhigh"}, session_id="offline-session"
        ) == ({}, {"_cursor_session_scope": "offline-session"})
        assert profile.get_model_context_length("grok-4.7@context=500k") == 500_000
        assert profile.get_model_context_length("grok-4.7") is None
        agent.close()

        original_fetch_models = profile.fetch_models
        listed_variant = "grok-4.7@fast=true,reasoning_effort=xhigh"
        profile.fetch_models = lambda **_: ["auto", listed_variant]
        try:
            from hermes_cli.models_validate import validate_requested_model

            verdict = validate_requested_model(
                listed_variant,
                "cursor",
                api_key="offline-probe-token",
                base_url=runtime["base_url"],
            )
        finally:
            profile.fetch_models = original_fetch_models
        assert verdict["accepted"] and verdict["recognized"], verdict

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
        verify_managed_stream(profile)
        verify_usage_seam(profile)

        print(json.dumps({
            "ok": True,
            "managed_relay_stream": True,
            "usage_seam": True,
            "hermes_source": str(Path(providers.__file__).resolve()),
            "provider": runtime["provider"],
            "api_mode": runtime["api_mode"],
            "base_url": runtime["base_url"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
