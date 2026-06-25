from scoring import (
    course_answer_matches,
    course_score_percent,
    normalize_answer,
    passes_course_threshold,
)


def test_normalize_answer_strips_prefix_and_quotes():
    assert normalize_answer('FINAL ANSWER: "hello"') == "hello"
    assert normalize_answer("Answer: 42") == "42"


def test_course_answer_matches_case_insensitive():
    assert course_answer_matches("Right", "right")
    assert not course_answer_matches("left", "right")


def test_course_score_percent():
    assert course_score_percent(6, 20) == 30.0
    assert course_score_percent(0, 0) == 0.0


def test_passes_course_threshold():
    assert passes_course_threshold(6, 20)
    assert not passes_course_threshold(5, 20)
