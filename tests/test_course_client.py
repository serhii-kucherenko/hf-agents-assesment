from unittest.mock import MagicMock, patch

from eval.course_client import (
    download_attachment,
    fetch_course_questions,
    fetch_random_question,
    submit_answers,
)


@patch("eval.course_client.requests.get")
def test_fetch_course_questions(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: [{"task_id": "abc", "question": "Q?"}],
    )
    mock_get.return_value.raise_for_status = MagicMock()
    data = fetch_course_questions("https://example.com")
    assert data[0]["task_id"] == "abc"


@patch("eval.course_client.requests.get")
def test_fetch_random_question(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"task_id": "xyz", "question": "Random?"},
    )
    mock_get.return_value.raise_for_status = MagicMock()
    data = fetch_random_question("https://example.com")
    assert data["task_id"] == "xyz"


@patch("eval.course_client.requests.get")
def test_download_attachment(mock_get, tmp_path):
    mock_get.return_value = MagicMock(
        status_code=200,
        content=b"file-bytes",
    )
    mock_get.return_value.raise_for_status = MagicMock()
    path = download_attachment("https://example.com", "task1", "data.csv", tmp_path)
    assert path.exists()
    assert path.read_bytes() == b"file-bytes"


@patch("eval.course_client.requests.post")
def test_submit_answers(mock_post):
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"score": 30, "correct_count": 6},
    )
    mock_post.return_value.raise_for_status = MagicMock()
    result = submit_answers("https://example.com", "user", "code", [])
    assert result["score"] == 30
