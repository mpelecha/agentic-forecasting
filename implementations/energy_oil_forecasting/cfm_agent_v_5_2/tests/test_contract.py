import inspect

from energy_oil_forecasting.cfm_agent_v_5_2.config import (
    DEFAULT_SETTINGS,
    CfmV52Settings,
)
from energy_oil_forecasting.cfm_agent_v_5_2.policy import EvidencePolicy
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ResearchPacket
from energy_oil_forecasting.cfm_agent_v_5_2.tools import (
    AuditedCodeExecutionTool,
    AuthoritativeSuiteTool,
    ResearchPipelineTool,
)


def test_locked_thresholds_and_audit_default():
    assert (
        DEFAULT_SETTINGS.tier_1_min_confidence,
        DEFAULT_SETTINGS.tier_2_min_confidence,
        DEFAULT_SETTINGS.tier_3_min_confidence,
    ) == (0.4, 0.6, 0.8)
    assert (
        DEFAULT_SETTINGS.small_action_width_fraction,
        DEFAULT_SETTINGS.moderate_action_width_fraction,
        DEFAULT_SETTINGS.large_action_width_fraction,
    ) == (0.1, 0.2, 0.3)
    assert DEFAULT_SETTINGS.audit_enabled is False


def test_active_packet_has_no_destination_page_fields():
    fields = set(ResearchPacket.model_fields)
    assert "passages" not in fields
    assert "source_validation_findings" not in fields
    assert "verified_summaries" in fields


def test_audit_outputs_do_not_appear_in_policy_source():
    text = inspect.getsource(EvidencePolicy)
    assert "SourceAuditBundle" not in text
    assert "ClaimSupportFinding" not in text


def find_key(value, target):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == target:
                found.append(key)
            found.extend(find_key(child, target))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_key(child, target))
    return found


def test_all_package_tool_schemas_are_adk_compatible():
    class Service:
        series_ids = ["wti"]

    settings = CfmV52Settings()
    tools = [
        AuthoritativeSuiteTool(Service(), settings=settings, covariate_series_ids=[]).as_function_tool(),
        ResearchPipelineTool(settings, search_model="test").as_function_tool(),
        AuditedCodeExecutionTool(settings).as_function_tool(),
    ]
    for tool in tools:
        declaration = tool._get_declaration()
        assert not find_key(declaration.parameters_json_schema, "additionalProperties")
        assert "tool_context" not in declaration.parameters_json_schema.get("properties", {})
