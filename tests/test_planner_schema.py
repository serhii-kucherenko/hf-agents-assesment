from unittest.mock import MagicMock, patch

from agent.planner import create_plan, plan_to_prompt_section


@patch("agent.planner.build_model")
def test_create_plan_minimal_skips_llm(mock_build):
    plan = create_plan("How many?", None, "", task_id="t1")
    assert plan["steps"] == []
    mock_build.assert_not_called()


@patch("agent.planner.write_plan")
@patch("agent.planner.build_model")
def test_create_plan_standard_parses_json(mock_build, _mock_write, monkeypatch):
    monkeypatch.setenv("PIPELINE_DEPTH", "standard")
    model = MagicMock()
    model.generate.return_value = MagicMock(
        content='{"restatement":"Q","strategy":"search","steps":[{"type":"lookup","goal":"Find fact"}]}'
    )
    mock_build.return_value = model
    plan = create_plan("Who?", None, "", task_id="t2")
    assert plan["strategy"] == "search"
    assert len(plan["steps"]) == 1


def test_plan_to_prompt_section_empty():
    assert plan_to_prompt_section({"steps": []}) == ""


def test_plan_to_prompt_section_renders_steps():
    text = plan_to_prompt_section(
        {"steps": [{"type": "compute", "goal": "Count items"}]}
    )
    assert "compute" in text
    assert "Count items" in text
