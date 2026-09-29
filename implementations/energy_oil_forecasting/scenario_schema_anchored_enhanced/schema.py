"""Scenario schema that makes memory continuation an explicit, auditable answer.

``WtiScenarioCard`` (from ``scenario_schema_core``) has no notion of memory —
it's shared with ``scenario_schema_anchored``, which never sees
``prior_frameworks`` at all. Whether a scenario here is "the same one as
before" or "genuinely new" should not be something inferred after the fact
by string-comparing names; it should be a field the LLM fills in, the same
way ``is_tail_case`` is a field rather than something guessed from a
scenario's probability.
"""

from __future__ import annotations

import json

from energy_oil_forecasting.scenario_schema_core import WtiScenarioCard, WtiScenarioForecastOutput
from pydantic import Field, model_validator


class WtiMemoryScenarioCard(WtiScenarioCard):
    """A scenario card that must declare its relationship to prior memory.

    Attributes
    ----------
    continues_prior_scenario : bool
        True if this scenario is a continuation of one shown in
        ``prior_frameworks`` (same underlying story — the name may or may
        not have changed). False for a genuinely new scenario with no prior
        counterpart, including every scenario on the first origin, when
        ``prior_frameworks`` is empty.
    continued_from : str or None
        The exact scenario name from ``prior_frameworks`` this continues.
        Required when ``continues_prior_scenario`` is true; must be omitted
        otherwise.
    """

    continues_prior_scenario: bool = Field(
        description=(
            "True if this scenario continues one shown in prior_frameworks (same "
            "underlying story, name may or may not have changed). False for a "
            "genuinely new scenario -- including every scenario when prior_frameworks is empty."
        )
    )
    continued_from: str | None = Field(
        default=None,
        description=(
            "Exact scenario name from prior_frameworks this continues. Required when "
            "continues_prior_scenario is true; omit when it is false."
        ),
    )

    @model_validator(mode="after")
    def _continuation_fields_are_consistent(self) -> "WtiMemoryScenarioCard":
        """Require continued_from exactly when continues_prior_scenario is true."""
        if self.continues_prior_scenario and not self.continued_from:
            raise ValueError(
                "continues_prior_scenario=true requires continued_from naming which prior scenario this continues."
            )
        if not self.continues_prior_scenario and self.continued_from:
            raise ValueError("continued_from must be omitted when continues_prior_scenario=false.")
        return self


class WtiMemoryScenarioForecastOutput(WtiScenarioForecastOutput):
    """``WtiScenarioForecastOutput`` whose scenarios declare memory continuity explicitly."""

    scenarios: list[WtiMemoryScenarioCard] = Field(
        min_length=2,
        description=(
            "2-3 named, competing scenarios, each declaring whether it continues "
            "a prior_frameworks scenario or is genuinely new."
        ),
    )

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Same template as the base class, with the two continuity fields added.

        Reuses :meth:`WtiScenarioForecastOutput.prompt_schema_json` rather
        than duplicating the whole template, so the forecasts/factors
        sections stay in sync with the base class automatically.
        """
        template = json.loads(WtiScenarioForecastOutput.prompt_schema_json())
        template["scenarios"][0]["continues_prior_scenario"] = (
            "<true|false -- see Memory and Framework Consistency>"
        )
        template["scenarios"][0]["continued_from"] = (
            "<exact prior scenario name if continuing one; omit if continues_prior_scenario is false>"
        )
        return json.dumps(template, indent=2)


__all__ = ["WtiMemoryScenarioCard", "WtiMemoryScenarioForecastOutput"]
