import inspect
from types import SimpleNamespace

from energy_oil_forecasting.cfm_agent_v_5_2 import (
    build_cfm_agent_config_arima_only,
    build_cfm_agent_predictor_arima_only,
)
from energy_oil_forecasting.cfm_agent_v_5_2_dategated import (
    DateGatedResearchPipelineTool,
    build_cfm_agent_config_arima_only_dategated,
    build_cfm_agent_predictor_arima_only_dategated,
)


SERVICE = SimpleNamespace(series_ids=["wti_crude_oil_price"])


def _tool_view(tool):
    func = tool.func
    return func.__name__, inspect.signature(func), inspect.getdoc(func)


def test_only_the_research_tool_and_predictor_id_differ_from_v52():
    v52 = build_cfm_agent_config_arima_only(data_service=SERVICE)
    gated = build_cfm_agent_config_arima_only_dategated(data_service=SERVICE)

    # ADK writes name and description into the system instruction, so all of these
    # must match for the agent to see the same prompt as v5.2.
    for field in ("name", "description", "model", "instruction", "max_output_tokens", "skills_dirs"):
        assert getattr(gated, field) == getattr(v52, field), field
    assert [_tool_view(t) for t in gated.function_tools] == [_tool_view(t) for t in v52.function_tools]
    assert isinstance(gated.function_tools[1].func.__self__, DateGatedResearchPipelineTool)

    v52_predictor = build_cfm_agent_predictor_arima_only(v52)
    gated_predictor = build_cfm_agent_predictor_arima_only_dategated(gated)
    assert (
        v52_predictor.predictor_id
        == "agent_predictor_cfm_agent_v_5_2_arima_only_gemini-3.1-flash-lite-preview_continuous"
    )
    assert gated_predictor.predictor_id == (
        "agent_predictor_cfm_agent_v_5_2_arima_only_dategated_gemini-3.1-flash-lite-preview_continuous"
    )
    assert gated_predictor.research is gated.function_tools[1].func.__self__
