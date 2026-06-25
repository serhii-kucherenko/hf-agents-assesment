from unittest.mock import MagicMock, patch

from agent.agent_runner import AgentRunner
from agent.pipeline import run_pipeline


@patch.object(AgentRunner, "__init__", lambda self: None)
@patch("agent.pipeline.run_with_self_correction")
def test_pipeline_minimal_returns_verified_answer(mock_correction):
    runner = AgentRunner()
    runner.run = MagicMock(return_value='final_answer("right")')
    mock_correction.return_value = ('final_answer("right")', "right")

    answer, trace = run_pipeline(
        runner,
        'Opposite of "left"?',
        task_id="smoke",
    )
    assert answer == "right"
    assert "final_answer" in trace
