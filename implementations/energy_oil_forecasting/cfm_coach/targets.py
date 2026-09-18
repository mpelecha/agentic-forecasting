"""Which agent package a coach operation is talking to.

Until v5.2 arrived the answer was always ``cfm_agent_v_5_0``, so eight modules
imported it directly and ``run_store`` hardcoded its manifest path and fingerprint
prefix. That was correct while there was one agent and actively dangerous the
moment there were two: reusing ``agent_package_fingerprint()`` for a v5.2 run
would have stamped it with **v5.0's** fingerprint -- a wrong identity field, and
one that specifically defeats ``ComparisonPolicy``'s ``single_package_fingerprint``
condition, whose entire job is to notice that the corpus spans two builds.

So this module is now the *only* place in the coach that names an agent package.
Everything else takes an :class:`AgentTarget` and asks it. Adding v5.3 later means
adding one entry here, not grepping for ``cfm_agent_v_5_0``.

**Heavy imports are lazy.** ``agent.py`` pulls in Google ADK and the whole
forecasting stack; ``config.py`` is pydantic and pathlib. Only the light half is
imported at module scope, so importing ``cfm_coach.run_store`` in a test still
costs what it did before.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from energy_oil_forecasting.cfm_agent_v_5_0.config import (
    AGENT_NAME as V50_AGENT_NAME,
)
from energy_oil_forecasting.cfm_agent_v_5_0.config import (
    PACKAGE_ROOT as V50_ROOT,
)
from energy_oil_forecasting.cfm_agent_v_5_0.config import (
    CfmV50Settings,
)
from energy_oil_forecasting.cfm_agent_v_5_2.config import (
    AGENT_NAME as V52_AGENT_NAME,
)
from energy_oil_forecasting.cfm_agent_v_5_2.config import (
    PACKAGE_ROOT as V52_ROOT,
)
from energy_oil_forecasting.cfm_agent_v_5_2.config import (
    CfmV52Settings,
)


#: Suffix marking a settings field that names an LLM. Both packages follow it
#: (``search_verifier_model``, ``claim_support_verifier_model``, and v5.2's
#: ``structured_output_retry_model``), so `AgentTarget.model_settings_fields`
#: discovers them rather than carrying a list that a new field could fall out of.
#: `tests/test_streams.py::test_model_fields_are_discovered_not_assumed` asserts
#: the discovered set against an explicit expectation, so a rename breaks a test
#: instead of silently leaving one call pointed at the wrong model.
_MODEL_FIELD_SUFFIX = "_model"


@dataclass(frozen=True)
class AgentTarget:
    """One agent package, and everything the coach needs to drive and identify it."""

    agent_id: str
    module: str
    settings_cls: type
    package_root: Path
    #: Namespaces the package fingerprint. Two builds must never share a prefix,
    #: or a corpus spanning both looks like one build to the comparison gate.
    fingerprint_prefix: str
    #: The prompt version in force while the package's own prompt builder is used
    #: unmodified. A different prompt is a different agent (R6), so this is an
    #: identity field, not a cosmetic label.
    prompt_version: str
    #: The package's config/predictor builder pair. Defaults to the names every
    #: package ships, so a target that uses them says nothing.
    #:
    #: ``v5.2 ARIMA-only`` is why these are fields. It is not a separate package:
    #: it is ``cfm_agent_v_5_2`` built through a different entry point, which swaps
    #: the three-model ensemble for ARIMA alone. Same module, same settings class,
    #: same manifest -- a different *agent*, because what it forecasts with differs.
    config_builder: str = "build_cfm_agent_config"
    predictor_builder: str = "build_cfm_agent_predictor"

    # -- identity -------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.package_root / "MANIFEST.sha256"

    @property
    def model_settings_fields(self) -> tuple[str, ...]:
        """Settings fields that name an LLM, discovered by suffix.

        v5.0 has two; v5.2 adds ``structured_output_retry_model``. Missing one
        means that call silently stays on the package default while every other
        call moved -- which would make a single-model run a mixed-model run and
        quietly invalidate the comparison it exists to support.
        """
        return tuple(sorted(name for name in self.settings_cls.model_fields if name.endswith(_MODEL_FIELD_SUFFIX)))

    # -- settings -------------------------------------------------------------

    def settings(self, **overrides: Any) -> Any:
        """Build this target's settings object. Unknown fields raise (``extra='forbid'``)."""
        return self.settings_cls(**overrides)

    def bind_models(self, settings: Any, model: str) -> Any:
        """Return `settings` with every LLM-naming field pointed at `model`.

        This covers the settings-side calls only. The two constructor-side models
        -- the main agent and the grounded-search tool -- are passed to
        ``build_config`` separately; :func:`cfm_coach.streams.RunStream.build_agent`
        is where both halves are applied together, and the only place that should
        be doing so.
        """
        return self.settings_cls(**{**settings.model_dump(), **dict.fromkeys(self.model_settings_fields, model)})

    def validate_settings(self, payload: dict[str, Any]) -> Any:
        return self.settings_cls.model_validate(payload)

    # -- lazy access to the package's heavy machinery -------------------------

    def _module(self, suffix: str) -> Any:
        return importlib.import_module(f"{self.module}.{suffix}")

    def build_config(self, **kwargs: Any) -> Any:
        return getattr(self._module("agent"), self.config_builder)(**kwargs)

    def build_predictor(self, config: Any) -> Any:
        return getattr(self._module("agent"), self.predictor_builder)(config)

    def engine(self, settings: Any) -> Any:
        """Build this package's ``PythonForecastEngine``.

        Not interchangeable across packages: v5.0's ``engine_id`` is
        ``python_forecast_engine_v50`` and v5.2's is ``..._v52``. Replaying a v5.2
        record through v5.0's engine would pass ``verify_fidelity`` today and drift
        the first time the two implementations diverge.
        """
        return self._module("forecast_engine.transformation").PythonForecastEngine(settings)

    def policy(self, settings: Any) -> Any:
        return self._module("policy.evidence_policy").EvidencePolicy(settings)

    def assessment_from(self, payload: dict[str, Any]) -> Any:
        return self._module("outputs").CfmContextAssessmentOutput.model_validate(payload)

    def packet_from(self, payload: dict[str, Any]) -> Any:
        return self._module("schemas").ResearchPacket.model_validate(payload)

    def horizon_forecast(self, **kwargs: Any) -> Any:
        return self._module("schemas").ModelHorizonForecast(**kwargs)


V50 = AgentTarget(
    agent_id=V50_AGENT_NAME,
    module="energy_oil_forecasting.cfm_agent_v_5_0",
    settings_cls=CfmV50Settings,
    package_root=V50_ROOT,
    fingerprint_prefix="cfm_v5_0_package",
    prompt_version="cfm_v5_0_builtin",
)

V52 = AgentTarget(
    agent_id=V52_AGENT_NAME,
    module="energy_oil_forecasting.cfm_agent_v_5_2",
    settings_cls=CfmV52Settings,
    package_root=V52_ROOT,
    fingerprint_prefix="cfm_v5_2_package",
    prompt_version="cfm_v5_2_builtin",
)

#: v5.2 with ARIMA alone in the ensemble -- no Kalman, no LightGBM. Shares every
#: file with :data:`V52`, so the two differ only in which builder runs.
#:
#: **The fingerprint prefix has to differ, and the manifest cannot express why.**
#: ``agent_package_fingerprint`` hashes ``MANIFEST.sha256``, which is the same file
#: for both targets -- so without a distinct prefix these two would fingerprint
#: identically, and a corpus holding both would satisfy
#: ``ComparisonPolicy.single_package_fingerprint`` while actually spanning a
#: three-model ensemble and a one-model one. That is precisely the pooling the
#: condition exists to refuse, so the prefix carries the distinction the hash
#: cannot see.
#:
#: ``agent_id`` matches the ``AgentConfig.name`` that
#: ``build_cfm_agent_config_arima_only`` stamps on the config, so a record's agent
#: id and the config that produced it agree.
V52_ARIMA_ONLY = AgentTarget(
    agent_id="cfm_agent_v_5_2_arima_only",
    module="energy_oil_forecasting.cfm_agent_v_5_2",
    settings_cls=CfmV52Settings,
    package_root=V52_ROOT,
    fingerprint_prefix="cfm_v5_2_arima_only_package",
    prompt_version="cfm_v5_2_builtin",
    config_builder="build_cfm_agent_config_arima_only",
    predictor_builder="build_cfm_agent_predictor_arima_only",
)

TARGETS: dict[str, AgentTarget] = {target.agent_id: target for target in (V50, V52, V52_ARIMA_ONLY)}

#: The coach's original target. Every entry point still defaults to it, so nothing
#: that predates v5.2 changes behaviour by being recompiled against this module.
DEFAULT_TARGET = V50


def target_for(agent_id: str) -> AgentTarget:
    if agent_id not in TARGETS:
        raise KeyError(f"no agent target {agent_id!r}; known targets are {sorted(TARGETS)}")
    return TARGETS[agent_id]


__all__ = [
    "DEFAULT_TARGET",
    "TARGETS",
    "V50",
    "V52",
    "V52_ARIMA_ONLY",
    "AgentTarget",
    "target_for",
]
