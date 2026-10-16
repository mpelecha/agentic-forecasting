"""LLM roles (Section 4.3) and model preflight (Section 10.2), fully offline.

These tests import ``llm_roles`` and ``response_control`` only; ``agent.py`` is
not imported here.
"""

from __future__ import annotations

from types import SimpleNamespace

import litellm
import pytest
from energy_oil_forecasting.cfm_agent_v_6_0.config import (
    DEFAULT_SETTINGS,
    LLM_ROLES,
    CfmV60Settings,
)
from energy_oil_forecasting.cfm_agent_v_6_0.llm_roles import (
    RECORDER,
    REDACTED,
    build_main_model,
    completion_kwargs,
    distinct_role_models,
    dropped_parameters,
    main_agent_config_kwargs,
    main_model_kwargs,
    preflight_models,
    require_models_available,
    role_parameters,
)
from energy_oil_forecasting.cfm_agent_v_6_0.response_control import (
    CORRECTION_ROLE,
    StructuredAssessmentController,
)
from google.adk.models.lite_llm import LiteLlm


BASE_URL = "http://proxy.test/v1"
API_KEY = "test-secret-key"
MAIN_MODEL = "gemini-3.1-flash-lite-preview"
AUX_MODEL = "gemini-3.5-flash"


@pytest.fixture(autouse=True)
def _proxy_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", BASE_URL)
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    RECORDER.reset()
    yield
    RECORDER.reset()


class FakeInner:
    predictor_id = "fake"
    _runner = None

    def prompt_builder(self, **kwargs):
        return "prompt"


def _completion_stub(captured: list[dict], content: str = "{}"):
    def completion(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )

    return completion


# ---------------------------------------------------------------- main role ----


def test_build_main_model_is_litellm_with_settings_kwargs():
    model = build_main_model(DEFAULT_SETTINGS)
    assert isinstance(model, LiteLlm)
    assert model.model == f"openai/{MAIN_MODEL}"
    assert model._additional_args == {
        "api_base": BASE_URL,
        "api_key": API_KEY,
        "timeout": 180,
        "top_p": 1.0,
        "n": 1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "parallel_tool_calls": False,
        "drop_params": True,
    }


def test_main_model_kwargs_carry_no_temperature_or_max_tokens():
    kwargs = main_model_kwargs(DEFAULT_SETTINGS)
    assert kwargs["model"] == f"openai/{MAIN_MODEL}"
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
    assert "seed" not in kwargs


def test_main_agent_config_kwargs_route_temperature_and_tokens_through_agent_config():
    kwargs = main_agent_config_kwargs(DEFAULT_SETTINGS)
    assert isinstance(kwargs["model"], LiteLlm)
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_output_tokens"] == 24576
    assert kwargs["seed"] is None


def test_main_model_follows_settings_not_a_literal():
    settings = CfmV60Settings(main_model="other-model", main_timeout_seconds=7.0, llm_candidate_count=2)
    model = build_main_model(settings)
    assert model.model == "openai/other-model"
    assert model._additional_args["timeout"] == 7.0
    assert model._additional_args["n"] == 2


def test_build_main_model_records_effective_parameters_with_redacted_key():
    build_main_model(DEFAULT_SETTINGS)
    entries = RECORDER.snapshot()["main"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["parameters"]["api_key"] == REDACTED
    assert API_KEY not in repr(entry)
    assert entry["parameters"]["model"] == f"openai/{MAIN_MODEL}"
    assert entry["parameters"]["timeout"] == 180
    assert entry["note"] == "AgentConfig.temperature=0.1, max_output_tokens=24576, seed=None"
    assert entry["parameters_dropped_by_proxy"] == []


# ----------------------------------------------------------- auxiliary roles ----


@pytest.mark.parametrize("role", LLM_ROLES)
def test_completion_kwargs_match_settings_for_each_role(role):
    settings = DEFAULT_SETTINGS
    kwargs = completion_kwargs(settings, role)
    assert kwargs == {
        "model": f"openai/{getattr(settings, f'{role}_model')}",
        "api_base": BASE_URL,
        "api_key": API_KEY,
        "temperature": getattr(settings, f"{role}_temperature"),
        "max_tokens": getattr(settings, f"{role}_max_tokens"),
        "timeout": getattr(settings, f"{role}_timeout_seconds"),
        "top_p": settings.llm_top_p,
        "n": settings.llm_candidate_count,
        "frequency_penalty": settings.llm_frequency_penalty,
        "presence_penalty": settings.llm_presence_penalty,
        "drop_params": True,
    }
    with_tools = completion_kwargs(settings, role, with_tools=True)
    assert with_tools["parallel_tool_calls"] is settings.llm_parallel_tool_calls


def test_role_parameters_reject_unknown_role():
    with pytest.raises(ValueError, match="unknown LLM role"):
        role_parameters(DEFAULT_SETTINGS, "planner")


def test_distinct_role_models_are_in_first_use_order():
    assert distinct_role_models(DEFAULT_SETTINGS) == [MAIN_MODEL, AUX_MODEL]


def test_correction_call_receives_its_role_parameters(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(litellm, "completion", _completion_stub(captured, content='{"ok": true}'))
    controller = StructuredAssessmentController(FakeInner(), DEFAULT_SETTINGS)
    content = controller._corrected_response(
        raw_response="not json",
        errors=[{"type": "ValueError", "msg": "bad"}],
        packet_id="packet",
        horizons=[5, 10, 21],
    )
    assert content == '{"ok": true}'
    assert len(captured) == 1
    call = captured[0]
    assert call["model"] == f"openai/{AUX_MODEL}"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 8192
    assert call["timeout"] == 60
    assert call["top_p"] == 1.0
    assert call["n"] == 1
    assert call["frequency_penalty"] == 0.0
    assert call["presence_penalty"] == 0.0
    assert call["drop_params"] is True
    assert call["api_base"] == BASE_URL
    assert call["api_key"] == API_KEY
    assert [message["role"] for message in call["messages"]] == ["system", "user"]
    assert "tools" not in call


def test_correction_call_is_recorded_with_redaction_and_response_model(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(litellm, "completion", _completion_stub(captured))
    controller = StructuredAssessmentController(FakeInner(), DEFAULT_SETTINGS)
    controller._corrected_response(raw_response="x", errors=[], packet_id="packet", horizons=[5])
    entries = RECORDER.snapshot()[CORRECTION_ROLE]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["parameters"]["api_key"] == REDACTED
    assert "messages" not in entry["parameters"]
    assert entry["parameters"]["model"] == f"openai/{AUX_MODEL}"
    assert entry["parameters"]["max_tokens"] == 8192
    assert entry["response_model"] == f"openai/{AUX_MODEL}"
    assert entry["parameters_dropped_by_proxy"] == []
    assert API_KEY not in repr(entry)


def test_correction_call_follows_overridden_settings(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(litellm, "completion", _completion_stub(captured))
    settings = CfmV60Settings(
        structured_output_correction_model="corrector-x",
        structured_output_correction_max_tokens=123,
        structured_output_correction_timeout_seconds=9.0,
    )
    StructuredAssessmentController(FakeInner(), settings)._corrected_response(
        raw_response="x", errors=[], packet_id="packet", horizons=[5]
    )
    assert captured[0]["model"] == "openai/corrector-x"
    assert captured[0]["max_tokens"] == 123
    assert captured[0]["timeout"] == 9.0


# --------------------------------------------------------- dropped parameters ----


def test_dropped_parameters_is_a_local_lookup():
    kwargs = completion_kwargs(DEFAULT_SETTINGS, "search")
    assert dropped_parameters(kwargs, model=kwargs["model"]) == []
    assert dropped_parameters(main_model_kwargs(DEFAULT_SETTINGS), model=f"openai/{MAIN_MODEL}") == []
    with_unsupported = {**kwargs, "top_k": 40}
    dropped = dropped_parameters(with_unsupported, model=kwargs["model"])
    assert isinstance(dropped, list)
    assert dropped == ["top_k"]


def test_dropped_parameters_never_raises(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(litellm, "get_supported_openai_params", boom)
    assert dropped_parameters({"model": "openai/x", "top_k": 1}, model="openai/x") == []


# ------------------------------------------------------------------ preflight ----


def test_preflight_uses_the_models_endpoint_when_available():
    calls: list[tuple[str, dict]] = []

    def fetch(url, headers):
        calls.append((url, headers))
        return [MAIN_MODEL, "unrelated-model"]

    def probe(model):
        raise AssertionError("probe must not run when the listing is available")

    result = preflight_models(DEFAULT_SETTINGS, fetch=fetch, probe=probe)
    assert calls == [(f"{BASE_URL}/models", {"Authorization": f"Bearer {API_KEY}"})]
    assert result["method"] == "models_endpoint"
    assert result["base_url"] == BASE_URL
    assert result["listed"] == [MAIN_MODEL, "unrelated-model"]
    assert result["models"] == {MAIN_MODEL: True, AUX_MODEL: False}
    assert result["checked_at"].endswith("+00:00")
    with pytest.raises(RuntimeError) as excinfo:
        require_models_available(DEFAULT_SETTINGS, result)
    assert AUX_MODEL in str(excinfo.value)
    assert MAIN_MODEL not in str(excinfo.value)


def test_preflight_falls_back_to_the_completion_probe():
    probed: list[str] = []

    def probe(model):
        probed.append(model)
        return True

    result = preflight_models(DEFAULT_SETTINGS, fetch=lambda url, headers: None, probe=probe)
    assert result["method"] == "completion_probe"
    assert result["listed"] is None
    assert probed == [MAIN_MODEL, AUX_MODEL]
    assert result["models"] == {MAIN_MODEL: True, AUX_MODEL: True}
    require_models_available(DEFAULT_SETTINGS, result)


def test_preflight_probe_failure_names_the_missing_model():
    result = preflight_models(
        DEFAULT_SETTINGS,
        base_url="http://explicit.test",
        api_key="k",
        fetch=lambda url, headers: None,
        probe=lambda model: model == AUX_MODEL,
    )
    assert result["base_url"] == "http://explicit.test"
    assert result["models"] == {MAIN_MODEL: False, AUX_MODEL: True}
    with pytest.raises(RuntimeError, match=MAIN_MODEL):
        require_models_available(DEFAULT_SETTINGS, result)


def test_preflight_without_base_url_probes_without_fetching(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL")

    def fetch(url, headers):
        raise AssertionError("no listing request without a base URL")

    result = preflight_models(DEFAULT_SETTINGS, fetch=fetch, probe=lambda model: True)
    assert result["method"] == "completion_probe"
    assert result["base_url"] is None


def test_require_models_available_treats_unchecked_models_as_missing():
    with pytest.raises(RuntimeError) as excinfo:
        require_models_available(DEFAULT_SETTINGS, {"models": {MAIN_MODEL: True}})
    assert AUX_MODEL in str(excinfo.value)
