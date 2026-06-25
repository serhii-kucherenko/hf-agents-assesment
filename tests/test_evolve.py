import json
from pathlib import Path

from agent.evolve import record_run, self_evolve_enabled, update_strategy_deltas


def test_self_evolve_disabled_by_default():
    assert not self_evolve_enabled()


def test_record_run_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_EVOLVE", "0")
    monkeypatch.setenv("AGENT_MEMORY_DIR", str(tmp_path))
    record_run("t1", "How many?", "count", ["execute_code"], True)
    assert not (tmp_path / "experience.jsonl").exists()


def test_record_run_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_EVOLVE", "1")
    monkeypatch.setenv("AGENT_MEMORY_DIR", str(tmp_path))
    record_run("t1", "How many items?", "count", ["execute_code"], True)
    lines = (tmp_path / "experience.jsonl").read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["task_id"] == "t1"
    assert "answer" not in record


def test_update_strategy_deltas(monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_EVOLVE", "1")
    monkeypatch.setenv("AGENT_MEMORY_DIR", str(tmp_path))
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "correct": True,
                        "question_tags": ["numeric"],
                        "strategy": "compute",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    update_strategy_deltas(report_path)
    deltas = json.loads((tmp_path / "strategy_deltas.json").read_text(encoding="utf-8"))
    assert deltas["numeric"]["successes"] == 1
