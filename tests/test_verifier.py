from unittest.mock import MagicMock, patch

from agent.verifier import extract_answer, strip_reasoning_blocks, verify_answer


def test_strip_reasoning_blocks():
    tag = "redacted_reasoning"
    text = f"<{tag}>hidden</{tag}>The answer is 3"
    assert strip_reasoning_blocks(text) == "The answer is 3"


def test_extract_answer_from_final_answer_call():
    raw = 'Some trace\nfinal_answer("42")'
    assert extract_answer(raw) == "42"


def test_extract_answer_from_final_answer_label():
    raw = "FINAL ANSWER: hello"
    assert extract_answer(raw) == "hello"


def test_verify_answer_flags_empty():
    result = verify_answer("Q?", "")
    assert not result.approved
    assert any("empty" in issue.lower() for issue in result.issues)


@patch("agent.verifier.requests.get")
def test_verify_answer_passes_clean_answer(mock_get):
    mock_get.return_value = MagicMock(status_code=200)
    raw = 'final_answer("right")'
    result = verify_answer("Opposite of left?", raw)
    assert result.approved
    assert result.answer == "right"
