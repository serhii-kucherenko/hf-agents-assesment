from agent.retriever import get_strategy_hints, retriever_enabled


def test_retriever_disabled_by_default():
    assert not retriever_enabled()
    assert get_strategy_hints("How many items in the PDF?") == ""


def test_retriever_never_includes_answers(monkeypatch, tmp_path):
    monkeypatch.setenv("RETRIEVER_ENABLED", "1")
    monkeypatch.setenv("AGENT_MEMORY_DIR", str(tmp_path))
    experience = tmp_path / "experience.jsonl"
    experience.write_text(
        '{"question_tags":["numeric"],"strategy":"use execute_code","tools_used":["execute_code"],"correct":true}\n',
        encoding="utf-8",
    )
    hints = get_strategy_hints("How many total items?")
    assert "execute_code" in hints
    assert "answer" not in hints.lower() or "do not copy answers" in hints.lower()
