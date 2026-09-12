import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.citation import PassageInput
from src.prompt_loader import load_prompt_template, render_closed_book_prompt, render_rag_prompt


def test_load_prompt_template_has_stable_full_hash():
    t1 = load_prompt_template("answer_de_v1")
    t2 = load_prompt_template("answer_de_v1")
    assert t1.sha256 == t2.sha256
    assert len(t1.sha256) == 64  # hex đầy đủ, KHÔNG cắt như passage_id
    assert t1.version == "answer_de_v1"


def test_render_rag_prompt_includes_numbered_passages_and_question():
    template = load_prompt_template("answer_de_v1")
    passages = [
        PassageInput("p1", "Berlin", "Berlin ist die Hauptstadt."),
        PassageInput("p2", "Paris", "Paris ist die Hauptstadt Frankreichs."),
    ]
    rendered = render_rag_prompt(template, "Was ist die Hauptstadt?", passages)
    assert "[P1] Berlin" in rendered
    assert "[P2] Paris" in rendered
    assert "Was ist die Hauptstadt?" in rendered


def test_render_closed_book_prompt_has_no_passage_section():
    template = load_prompt_template("answer_de_closed_book_v1")
    rendered = render_closed_book_prompt(template, "Was ist die Hauptstadt?")
    assert "[P1]" not in rendered
    assert "Was ist die Hauptstadt?" in rendered


def test_different_prompt_files_have_different_hashes():
    rag = load_prompt_template("answer_de_v1")
    closed = load_prompt_template("answer_de_closed_book_v1")
    assert rag.sha256 != closed.sha256


def test_changing_prompt_file_changes_hash(tmp_path, monkeypatch):
    import src.prompt_loader as pl
    monkeypatch.setattr(pl, "PROMPTS_DIR", tmp_path)
    (tmp_path / "temp_v1.txt").write_text("Frage: {question}", encoding="utf-8")
    before = load_prompt_template("temp_v1").sha256
    (tmp_path / "temp_v1.txt").write_text("Frage: {question} ", encoding="utf-8")  # +1 byte
    after = load_prompt_template("temp_v1").sha256
    assert before != after
