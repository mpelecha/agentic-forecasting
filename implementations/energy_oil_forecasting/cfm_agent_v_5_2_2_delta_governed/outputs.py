"""Structured LLM assessment output for CFM Agent v5.2.2 Delta-Governed.

Identical to :class:`CfmContextAssessmentOutput` except ``center_action`` is
a discrete rank in ``{-2,-1,0,1,2}`` mapped to a target percentile of the
empirical h-day price-delta distribution (see ``delta_distribution.py``),
instead of a named up/down/magnitude category mapped to a fixed fraction of
the ensemble's own quantile width.
"""

from __future__ import annotations

import json

from energy_oil_forecasting.cfm_agent_v_5_2.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.schemas import (
    RANK_TO_PERCENTILE,
    HorizonActionDeltaGoverned,
)


class CfmContextAssessmentOutputDeltaGoverned(CfmContextAssessmentOutput):
    horizon_actions: list[HorizonActionDeltaGoverned]

    @classmethod
    def prompt_schema_json(cls) -> str:
        mapping_str = ", ".join(f"{rank}->p{pct}" for rank, pct in sorted(RANK_TO_PERCENTILE.items()))
        return json.dumps(
            {
                "research_packet_id": "<exact packet id returned by tool>",
                "evidence_claims": [
                    {
                        "claim_id": "claim_001",
                        "statement": "<one atomic fact>",
                        "claim_type": "physical_supply",
                        "supporting_summary_ids": ["verified_summary:sha256:<id>"],
                        "supporting_source_ids": ["source_001"],
                        "contradicting_summary_ids": [],
                        "contradicting_source_ids": [],
                        "material_to_forecast": True,
                    }
                ],
                "physical_status": "unknown",
                "incremental_novelty": "indeterminate",
                "material_evidence_conflict": False,
                "confidence": "<0..1>",
                "center_action_guide": (
                    "center_action below must be a bare JSON INTEGER in {-2,-1,0,1,2} — never a quoted "
                    "string, never a price. A discrete rank on how bullish/bearish the evidence is. Each "
                    f"rank maps to a target percentile of the REAL historical h-day price-move distribution "
                    f"({mapping_str}). 0 = no directional view beyond the median historical move. Negative = "
                    "bearish, positive = bullish. Do not compute the dollar amount yourself — Python converts "
                    "your rank into a number using actual price history, never a number you invent."
                ),
                "horizon_actions": [
                    {
                        "horizon": "<requested>",
                        "center_action": 0,
                        "uncertainty_action": (
                            "<unchanged|small_wider|moderately_wider|substantially_wider"
                            "|small_narrower|moderately_narrower|substantially_narrower>"
                        ),
                        "persistence_profile": "<temporary|decaying|persistent|delayed|unknown>",
                        "cited_claim_ids": ["claim_001"],
                        "narrowing_uncertainty_resolved": "",
                        "rationale": "<concise>",
                    }
                ],
                "research_summary": "<summary>",
                "overall_rationale": "<assessment only>",
                "warnings": [],
            },
            indent=2,
        )


__all__ = ["CfmContextAssessmentOutputDeltaGoverned"]
