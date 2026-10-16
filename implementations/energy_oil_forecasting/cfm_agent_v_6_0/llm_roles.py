"""Five independently configurable LLM roles (specification Section 4.3).

Every role reads its model, temperature, maximum tokens, and timeout from
``settings.toml`` together with the common generation parameters. The functions
here resolve those values into the exact keyword arguments handed to the
invocation layer (``litellm`` for the four auxiliary roles, the ADK ``LiteLlm``
object for the main role) and record what was handed over, so the audit shows
the effective parameters rather than the configured ones.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from energy_oil_forecasting.cfm_agent_v_6_0.config import LLM_ROLES, CfmV60Settings


OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
LITELLM_OPENAI_PREFIX = "openai/"
REDACTED = "<redacted>"
_RECORD_EXCLUDED_KEYS = frozenset({"messages", "tools", "response_format", "api_key", "llm_client"})


@dataclass(frozen=True)
class RoleParameters:
    """The resolved generation parameters of one role."""

    role: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    top_p: float
    candidate_count: int
    frequency_penalty: float
    presence_penalty: float
    parallel_tool_calls: bool

    @property
    def litellm_model(self) -> str:
        """The model name on the OpenAI-compatible proxy path (``openai/<model>``)."""
        return self.model if self.model.startswith(LITELLM_OPENAI_PREFIX) else f"{LITELLM_OPENAI_PREFIX}{self.model}"


def role_parameters(settings: CfmV60Settings, role: str) -> RoleParameters:
    """Resolve one role's parameters from settings. ``role`` is one of :data:`LLM_ROLES`."""
    if role not in LLM_ROLES:
        raise ValueError(f"unknown LLM role {role!r}; expected one of {LLM_ROLES}")
    return RoleParameters(
        role=role,
        model=getattr(settings, f"{role}_model"),
        temperature=getattr(settings, f"{role}_temperature"),
        max_tokens=getattr(settings, f"{role}_max_tokens"),
        timeout_seconds=getattr(settings, f"{role}_timeout_seconds"),
        top_p=settings.llm_top_p,
        candidate_count=settings.llm_candidate_count,
        frequency_penalty=settings.llm_frequency_penalty,
        presence_penalty=settings.llm_presence_penalty,
        parallel_tool_calls=settings.llm_parallel_tool_calls,
    )


def completion_kwargs(settings: CfmV60Settings, role: str, *, with_tools: bool = False) -> dict[str, Any]:
    """Return the exact keyword arguments for ``litellm.completion`` or ``litellm.acompletion`` for one role.

    Callers add ``messages`` (and ``tools`` or ``response_format``) and must pass
    the returned mapping through unchanged, then call
    :func:`record_effective_parameters` with the full mapping they used.
    ``parallel_tool_calls`` is included only when ``with_tools`` is true, because
    the parameter is meaningless without a tool list.
    """
    params = role_parameters(settings, role)
    kwargs: dict[str, Any] = {
        "model": params.litellm_model,
        "api_base": os.getenv(OPENAI_BASE_URL_ENV),
        "api_key": os.getenv(OPENAI_API_KEY_ENV),
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "timeout": params.timeout_seconds,
        "top_p": params.top_p,
        "n": params.candidate_count,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "drop_params": True,
    }
    if with_tools:
        kwargs["parallel_tool_calls"] = params.parallel_tool_calls
    return kwargs


def sanitized_parameters(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy of invocation kwargs safe for the audit: no messages, tools, schema, or secrets."""
    clean = {key: value for key, value in kwargs.items() if key not in _RECORD_EXCLUDED_KEYS}
    if "api_key" in kwargs:
        clean["api_key"] = REDACTED if kwargs["api_key"] else None
    if "tools" in kwargs:
        clean["tool_names"] = [_tool_name(tool) for tool in kwargs["tools"] or []]
    if "response_format" in kwargs:
        clean["response_format_type"] = _response_format_type(kwargs["response_format"])
    return clean


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            return str(function["name"])
        return ",".join(sorted(str(key) for key in tool))
    return type(tool).__name__


def _response_format_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type", "dict"))
    return type(value).__name__


class EffectiveParameterRecorder:
    """Per-workflow record of the parameters handed to the invocation layer, by role."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, list[dict[str, Any]]] = {}

    def reset(self) -> None:
        with self._lock:
            self._records = {}

    def record(
        self,
        role: str,
        kwargs: dict[str, Any],
        *,
        dropped: list[str] | None = None,
        response_model: str | None = None,
        note: str | None = None,
    ) -> None:
        entry = {
            "role": role,
            "parameters": sanitized_parameters(kwargs),
            "parameters_dropped_by_proxy": list(dropped or []),
            "response_model": response_model,
            "note": note,
        }
        with self._lock:
            self._records.setdefault(role, []).append(entry)

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {role: [dict(entry) for entry in entries] for role, entries in self._records.items()}


#: Process-wide recorder. The predictor resets it at the start of a workflow and
#: copies its snapshot into the ``llm_effective_parameters`` audit key at the end.
RECORDER = EffectiveParameterRecorder()


def record_effective_parameters(
    role: str,
    kwargs: dict[str, Any],
    *,
    dropped: list[str] | None = None,
    response_model: str | None = None,
    note: str | None = None,
) -> None:
    """Record the parameters a call site actually handed over (see :class:`EffectiveParameterRecorder`)."""
    RECORDER.record(role, kwargs, dropped=dropped, response_model=response_model, note=note)


def configured_role_parameters(settings: CfmV60Settings) -> dict[str, dict[str, Any]]:
    """Every role's resolved parameters, for the audit and the preflight."""
    return {role: asdict(role_parameters(settings, role)) for role in LLM_ROLES}


def distinct_role_models(settings: CfmV60Settings) -> list[str]:
    """Return the distinct models configured across the five roles, in first-use order."""
    return list(dict.fromkeys(settings.role_models.values()))


# ---------------------------------------------------------------- main role ----

#: Keys of a litellm call that are transport or bookkeeping, not generation
#: parameters; they are never reported as "dropped".
_TRANSPORT_KEYS = frozenset(
    {"model", "api_base", "api_key", "timeout", "drop_params", "messages", "tools", "response_format", "llm_client"}
)


def main_model_kwargs(settings: CfmV60Settings) -> dict[str, Any]:
    """Return the exact keyword arguments the main role's ``LiteLlm`` object is built from.

    Temperature and maximum output tokens are not here: they travel through
    ``AgentConfig`` (see :func:`main_agent_config_kwargs`), as the specification
    requires. ``parallel_tool_calls`` is always forwarded because the agent turn
    carries tools.
    """
    params = role_parameters(settings, "main")
    return {
        "model": params.litellm_model,
        "api_base": os.getenv(OPENAI_BASE_URL_ENV),
        "api_key": os.getenv(OPENAI_API_KEY_ENV),
        "timeout": params.timeout_seconds,
        "top_p": params.top_p,
        "n": params.candidate_count,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "parallel_tool_calls": params.parallel_tool_calls,
        "drop_params": True,
    }


def _main_note(settings: CfmV60Settings) -> str:
    return (
        f"AgentConfig.temperature={settings.main_temperature}, max_output_tokens={settings.main_max_tokens}, seed=None"
    )


def build_main_model(settings: CfmV60Settings) -> Any:
    """Construct the ADK ``LiteLlm`` model for the main role and record what it carries.

    Returned as ``Any`` so that this module stays importable without the ADK; the
    object is a :class:`google.adk.models.lite_llm.LiteLlm`, which keeps every
    keyword argument other than ``model`` in ``_additional_args`` and forwards
    them to ``litellm.acompletion`` on each generation.
    """
    from google.adk.models.lite_llm import LiteLlm  # noqa: PLC0415

    kwargs = main_model_kwargs(settings)
    model = LiteLlm(**kwargs)
    record_effective_parameters(
        "main",
        kwargs,
        dropped=dropped_parameters(kwargs, model=kwargs["model"]),
        note=_main_note(settings),
    )
    return model


def main_agent_config_kwargs(settings: CfmV60Settings) -> dict[str, Any]:
    """Keyword arguments for ``AgentConfig`` that carry the main role (Section 4.3).

    The ``model`` entry is a ``BaseLlm`` instance, which the shared factory accepts
    unchanged instead of wrapping a bare model name itself.
    """
    return {
        "model": build_main_model(settings),
        "temperature": settings.main_temperature,
        "max_output_tokens": settings.main_max_tokens,
        "seed": None,
    }


def dropped_parameters(kwargs: dict[str, Any], *, model: str) -> list[str]:
    """Best-effort list of requested generation parameters litellm drops client-side.

    Uses ``litellm.get_supported_openai_params`` for the OpenAI-compatible
    provider, which is a local table lookup (no network). Returns an empty list
    whenever the lookup is unavailable or fails, so a call site never breaks on
    audit bookkeeping.
    """
    try:
        import litellm  # noqa: PLC0415

        bare = model[len(LITELLM_OPENAI_PREFIX) :] if model.startswith(LITELLM_OPENAI_PREFIX) else model
        supported = litellm.get_supported_openai_params(model=bare, custom_llm_provider="openai")
        if not supported:
            return []
        supported_set = set(supported)
        return sorted(key for key in kwargs if key not in _TRANSPORT_KEYS and key not in supported_set)
    except Exception:  # noqa: BLE001
        return []


# ----------------------------------------------------------------- preflight ----

PREFLIGHT_METHOD_LISTING = "models_endpoint"
PREFLIGHT_METHOD_PROBE = "completion_probe"
_HTTP_OK = 200  # structural: HTTP success status code
_PROBE_MAX_TOKENS = 1  # structural: the availability probe asks for a single token
_PING = "ping"


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _default_fetch(timeout_seconds: float) -> Callable[[str, dict[str, str]], list[str] | None]:
    """GET ``{base_url}/models`` and parse the OpenAI-style listing; ``None`` when unavailable."""

    def fetch(url: str, headers: dict[str, str]) -> list[str] | None:
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                if response.status != _HTTP_OK:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None
        return [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]

    return fetch


def _default_probe(*, base_url: str | None, api_key: str | None, timeout_seconds: float) -> Callable[[str], bool]:
    """One ``max_tokens=1`` completion per model; any exception means unavailable."""

    def probe(model: str) -> bool:
        import litellm  # noqa: PLC0415

        try:
            litellm.completion(
                model=f"{LITELLM_OPENAI_PREFIX}{model}",
                api_base=base_url,
                api_key=api_key,
                messages=[{"role": "user", "content": _PING}],
                max_tokens=_PROBE_MAX_TOKENS,
                timeout=timeout_seconds,
            )
        except Exception:  # noqa: BLE001
            return False
        return True

    return probe


def preflight_models(
    settings: CfmV60Settings,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    fetch: Callable[[str, dict[str, str]], list[str] | None] | None = None,
    probe: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Check that every distinct configured role model is available (Section 10.2).

    The listing endpoint is tried first; when it is unavailable, one single-token
    completion per distinct model is sent instead. The result is the
    ``llm_preflight`` audit record. Nothing here substitutes a model.
    """
    resolved_base_url = base_url if base_url is not None else os.getenv(OPENAI_BASE_URL_ENV)
    resolved_api_key = api_key if api_key is not None else os.getenv(OPENAI_API_KEY_ENV)
    timeout_seconds = settings.cutoff_verifier_timeout_seconds
    wanted = distinct_role_models(settings)
    listed: list[str] | None = None
    if resolved_base_url:
        headers = {"Authorization": f"Bearer {resolved_api_key}"} if resolved_api_key else {}
        listed = (fetch or _default_fetch(timeout_seconds))(_models_url(resolved_base_url), headers)
    if listed is not None:
        method = PREFLIGHT_METHOD_LISTING
        available = set(listed)
        models = {model: model in available for model in wanted}
    else:
        method = PREFLIGHT_METHOD_PROBE
        check = probe or _default_probe(
            base_url=resolved_base_url, api_key=resolved_api_key, timeout_seconds=timeout_seconds
        )
        models = {model: bool(check(model)) for model in wanted}
    return {
        "method": method,
        "base_url": resolved_base_url,
        "listed": sorted(listed) if listed is not None else None,
        "models": models,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def require_models_available(settings: CfmV60Settings, result: dict[str, Any]) -> None:
    """Raise ``RuntimeError`` naming every configured model the preflight did not confirm."""
    checked = result.get("models") or {}
    missing = [model for model in distinct_role_models(settings) if not checked.get(model, False)]
    if missing:
        raise RuntimeError(
            "configured LLM models are not available on the proxy (substitution is prohibited): " + ", ".join(missing)
        )


__all__ = [
    "LITELLM_OPENAI_PREFIX",
    "OPENAI_API_KEY_ENV",
    "OPENAI_BASE_URL_ENV",
    "PREFLIGHT_METHOD_LISTING",
    "PREFLIGHT_METHOD_PROBE",
    "RECORDER",
    "REDACTED",
    "EffectiveParameterRecorder",
    "RoleParameters",
    "build_main_model",
    "completion_kwargs",
    "configured_role_parameters",
    "distinct_role_models",
    "dropped_parameters",
    "main_agent_config_kwargs",
    "main_model_kwargs",
    "preflight_models",
    "record_effective_parameters",
    "require_models_available",
    "role_parameters",
    "sanitized_parameters",
]
