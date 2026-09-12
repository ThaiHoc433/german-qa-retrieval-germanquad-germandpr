import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answerer import LLMAnswerer
from src.citation import PassageInput
from src.llm_clients import ClaudeCitation, ClaudeGeneration, FakeLLMClient, RawGeneration
from src.prompt_loader import load_prompt_template

RAG_TEMPLATE = load_prompt_template("answer_de_v1")
CLOSED_BOOK_TEMPLATE = load_prompt_template("answer_de_closed_book_v1")

P1 = PassageInput("p1", "Titel A", "Albert Einstein wurde 1879 geboren. Er entwickelte die Relativitaetstheorie.")
P2 = PassageInput("p2", "Titel B", "Marie Curie erhielt zwei Nobelpreise.")

ANSWERED_TEXT = (
    "KURZ: Albert Einstein\n"
    "ANTWORT: Albert Einstein entwickelte die Relativitaetstheorie.\n"
    "QUELLE: P1\n"
    # ZITAT is the FULL P1 text (not just the second sentence) so that "Albert Einstein"
    # is a literal substring of it. is_short_answer_supported (src/citation.py) is pure
    # literal-substring matching after EM/F1-style normalization -- by design, no
    # coreference resolution -- so a quote of only "Er entwickelte ..." (which refers to
    # Einstein via the pronoun "Er") would never satisfy answer_supported=True. Same
    # fixture bug and same fix as documented in task-4-report.md for test_citation.py's
    # _parsed() base fixture.
    'ZITAT: "Albert Einstein wurde 1879 geboren. Er entwickelte die Relativitaetstheorie."\n'
)
NO_ANSWER_TEXT = "KURZ: KEINE ANTWORT\nANTWORT: None\nQUELLE: None\nZITAT: None\n"
FORMAT_VIOLATION_TEXT = "Ich weiss es nicht.\n"


def _open_answerer(scripted):
    client = FakeLLMClient(scripted)
    return LLMAnswerer("open", client, RAG_TEMPLATE, CLOSED_BOOK_TEMPLATE)


def test_rag_verified_and_supported():
    result = _open_answerer([ANSWERED_TEXT]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.generation_status == "success"
    assert result.answered is True
    assert result.citation_status == "verified"
    assert result.answer_supported is True
    assert result.citations[0].passage_id == "p1"


def test_rag_verified_but_not_supported():
    bad_kurz = ANSWERED_TEXT.replace("KURZ: Albert Einstein", "KURZ: Marie Curie")
    result = _open_answerer([bad_kurz]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.citation_status == "verified"
    assert result.answer_supported is False


def test_rag_wrong_passage():
    bad_zitat = ANSWERED_TEXT.replace(
        'ZITAT: "Albert Einstein wurde 1879 geboren. Er entwickelte die Relativitaetstheorie."',
        'ZITAT: "Marie Curie erhielt zwei Nobelpreise."',
    )
    result = _open_answerer([bad_zitat]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.citation_status == "wrong_passage"


def test_rag_no_answer():
    result = _open_answerer([NO_ANSWER_TEXT]).answer("rag", "Wer war Napoleon?", [P1, P2])
    assert result.generation_status == "success"
    assert result.answered is False
    assert result.citation_status is None
    assert result.answer_supported is None


def test_rag_format_violation():
    result = _open_answerer([FORMAT_VIOLATION_TEXT]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.generation_status == "error"
    assert result.error_type == "format_violation"
    assert result.answered is None


def test_rag_generation_error_does_not_reach_citation():
    error = RawGeneration(status="error", error_type="timeout", raw_text=None, latency_ms=5.0)
    result = _open_answerer([error]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.generation_status == "error"
    assert result.error_type == "timeout"
    assert result.citation_status is None


def test_oracle_rag_reuses_same_path_as_rag():
    # oracle_rag khác rag chỉ ở việc caller đưa gold passage thay vì kết quả retrieval —
    # LLMAnswerer không phân biệt, cùng một luồng xử lý.
    result = _open_answerer([ANSWERED_TEXT]).answer("oracle_rag", "Wer war Einstein?", [P1])
    assert result.citation_status == "verified"


def test_closed_book_has_no_citation_even_if_model_adds_one():
    # model lỡ thêm QUELLE/ZITAT dù prompt closed_book không yêu cầu -> vẫn bị bỏ qua
    # hoàn toàn (design §2.5): citation_status luôn None cho closed_book.
    leaked = ANSWERED_TEXT  # có QUELLE/ZITAT, nhưng answerer_type="closed_book"
    result = _open_answerer([leaked]).answer("closed_book", "Wer war Einstein?", [])
    assert result.answered is True
    assert result.citation_status is None
    assert result.answer_supported is None


def test_closed_book_no_answer():
    result = _open_answerer([NO_ANSWER_TEXT]).answer("closed_book", "Wer war das?", [])
    assert result.answered is False


def test_rag_with_claude_backend_verified():
    claude_gen = ClaudeGeneration(
        status="success", error_type=None, short_answer="Albert Einstein",
        answer="Albert Einstein entwickelte die Relativitaetstheorie.",
        # cited_text is the FULL P1 text for the same reason ANSWERED_TEXT's ZITAT is
        # above: is_short_answer_supported needs "Albert Einstein" to be a literal
        # substring of the cited text, which the second-sentence-only quote is not.
        citations=[ClaudeCitation(document_index=0,
                                   cited_text="Albert Einstein wurde 1879 geboren. Er entwickelte die Relativitaetstheorie.",
                                   start_char_index=0, end_char_index=77)],
        latency_ms=100.0,
    )

    class _StubClaudeClient:
        def generate_with_citations(self, prompt_instructions, question, passages):
            return claude_gen

    answerer = LLMAnswerer("claude", _StubClaudeClient(), RAG_TEMPLATE)
    result = answerer.answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.citation_status == "verified"
    assert result.answer_supported is True


def test_rag_with_claude_backend_multiple_sources():
    claude_gen = ClaudeGeneration(
        status="success", error_type=None, short_answer="X", answer="Y.",
        citations=[
            ClaudeCitation(document_index=0, cited_text="a", start_char_index=0, end_char_index=1),
            ClaudeCitation(document_index=1, cited_text="b", start_char_index=0, end_char_index=1),
        ],
        latency_ms=1.0,
    )

    class _StubClaudeClient:
        def generate_with_citations(self, prompt_instructions, question, passages):
            return claude_gen

    answerer = LLMAnswerer("claude", _StubClaudeClient(), RAG_TEMPLATE)
    result = answerer.answer("rag", "Frage?", [P1, P2])
    assert result.citation_status == "multiple_sources"
    assert result.raw_citation_refs == ("P1", "P2")
