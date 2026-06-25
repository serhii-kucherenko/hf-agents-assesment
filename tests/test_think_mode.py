from agent.think_mode import should_use_think_mode


def test_simple_question_disables_think(monkeypatch):
    monkeypatch.setenv("THINK_MODE", "auto")
    assert not should_use_think_mode(
        'If you understand this sentence, write the opposite of the word "left" as the answer.'
    )


def test_calculation_enables_think(monkeypatch):
    monkeypatch.setenv("THINK_MODE", "auto")
    assert should_use_think_mode(
        "According to the spreadsheet, calculate the average of all values in column B."
    )


def test_attachment_enables_think(monkeypatch, tmp_path):
    monkeypatch.setenv("THINK_MODE", "auto")
    pdf = tmp_path / "data.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert should_use_think_mode("What is the total?", file_path=str(pdf))


def test_force_on_off(monkeypatch):
    monkeypatch.setenv("THINK_MODE", "on")
    assert should_use_think_mode("Hi")
    monkeypatch.setenv("THINK_MODE", "off")
    assert not should_use_think_mode(
        "Calculate the sum of all entries in this complex multi-step puzzle."
    )
