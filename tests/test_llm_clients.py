import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answer_parser import ParsedAnswer
from src.citation import PassageInput
from src.llm_clients import (
    ClaudeApiClient,
    ClaudeCitation,
    ClaudeGeneration,
    FakeLLMClient,
    RawGeneration,
    claude_generation_to_parsed_answer,
)


def test_fake_client_returns_scripted_text_in_order():
    client = FakeLLMClient(["erste Antwort", "zweite Antwort"])
    first = client.generate("prompt 1")
    second = client.generate("prompt 2")
    assert first.status == "success"
    assert first.raw_text == "erste Antwort"
    assert second.raw_text == "zweite Antwort"


def test_fake_client_can_script_an_error():
    error = RawGeneration(status="error", error_type="timeout", raw_text=None, latency_ms=5.0)
    client = FakeLLMClient([error])
    result = client.generate("prompt")
    assert result.status == "error"
    assert result.error_type == "timeout"


def test_fake_client_raises_when_scenario_exhausted():
    client = FakeLLMClient(["only one"])
    client.generate("p")
    with pytest.raises(AssertionError, match="hết kịch bản"):
        client.generate("p again")


def _fake_block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations or [])


def _fake_citation(document_index, cited_text, start=None, end=None):
    return SimpleNamespace(document_index=document_index, cited_text=cited_text,
                            start_char_index=start, end_char_index=end)


class _StubMessages:
    def __init__(self, response):
        self._response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self._response


class _StubAnthropicClient:
    def __init__(self, response):
        self.messages = _StubMessages(response)


def test_generate_with_citations_parses_single_citation():
    response = SimpleNamespace(content=[
        _fake_block("KURZ: Albert Einstein\n"),
        _fake_block(
            "ANTWORT: Albert Einstein entwickelte die Relativitaetstheorie.\n",
            citations=[_fake_citation(0, "Einstein entwickelte sie 1915.", 10, 40)],
        ),
    ])
    stub = _StubAnthropicClient(response)
    client = ClaudeApiClient(stub)
    passages = [PassageInput("p1", "Titel", "Ein Text ueber Einstein.")]

    gen = client.generate_with_citations("Beantworte auf Deutsch.", "Wer war Einstein?", passages)

    assert gen.status == "success"
    assert gen.short_answer == "Albert Einstein"
    assert len(gen.citations) == 1
    assert gen.citations[0].document_index == 0
    assert gen.citations[0].cited_text == "Einstein entwickelte sie 1915."

    request = stub.messages.last_request
    assert request["model"] == "claude-opus-5"
    assert request["max_tokens"] >= 4096
    assert request["output_config"] == {"effort": "low"}
    documents = [c for c in request["messages"][0]["content"] if c["type"] == "document"]
    assert len(documents) == 1
    assert documents[0]["citations"] == {"enabled": True}


def test_generate_with_citations_reports_format_violation_without_labels():
    response = SimpleNamespace(content=[_fake_block("Ich bin mir nicht sicher.")])
    client = ClaudeApiClient(_StubAnthropicClient(response))
    gen = client.generate_with_citations("x", "Frage?", [PassageInput("p1", "T", "Text.")])
    assert gen.status == "error"
    assert gen.error_type == "format_violation"


def test_generate_with_citations_reports_format_violation_on_duplicate_kurz_label():
    # Finding 2: cùng định nghĩa format_violation với model_backend="open" — nhãn lặp
    # lại (KURZ xuất hiện 2 lần) phải bị coi là vi phạm định dạng, không được tolerate
    # qua setdefault như code cũ.
    response = SimpleNamespace(content=[
        _fake_block("KURZ: Albert Einstein\nANTWORT: x\nKURZ: nochmal\n"),
    ])
    client = ClaudeApiClient(_StubAnthropicClient(response))
    gen = client.generate_with_citations("x", "Frage?", [PassageInput("p1", "T", "Text.")])
    assert gen.status == "error"
    assert gen.error_type == "format_violation"


def test_generate_with_citations_reports_format_violation_on_empty_antwort():
    # Finding 2: ANTWORT rỗng phải là format_violation, không chỉ khi thiếu hoàn toàn.
    response = SimpleNamespace(content=[_fake_block("KURZ: Albert Einstein\nANTWORT: \n")])
    client = ClaudeApiClient(_StubAnthropicClient(response))
    gen = client.generate_with_citations("x", "Frage?", [PassageInput("p1", "T", "Text.")])
    assert gen.status == "error"
    assert gen.error_type == "format_violation"


def test_generate_with_citations_reports_refusal_on_stop_reason_refusal():
    response = SimpleNamespace(
        content=[_fake_block("KURZ: Albert Einstein\nANTWORT: x\n")],
        stop_reason="refusal",
    )
    client = ClaudeApiClient(_StubAnthropicClient(response))
    gen = client.generate_with_citations("x", "Frage?", [PassageInput("p1", "T", "Text.")])
    assert gen.status == "error"
    assert gen.error_type == "refusal"


def test_generate_with_citations_reports_truncated_on_stop_reason_max_tokens():
    response = SimpleNamespace(
        content=[_fake_block("KURZ: Albert Einstein\nANTWORT: x\n")],
        stop_reason="max_tokens",
    )
    client = ClaudeApiClient(_StubAnthropicClient(response))
    gen = client.generate_with_citations("x", "Frage?", [PassageInput("p1", "T", "Text.")])
    assert gen.status == "error"
    assert gen.error_type == "truncated"


def test_generate_with_citations_document_title_falls_back_to_positional_label_not_passage_id():
    # Finding 5: model không bao giờ được thấy passage_id — fallback phải là "Passage N",
    # không phải p.passage_id.
    response = SimpleNamespace(content=[_fake_block("KURZ: x\nANTWORT: y\n")])
    stub = _StubAnthropicClient(response)
    client = ClaudeApiClient(stub)
    passages = [PassageInput("secret_id_123", None, "Text.")]

    client.generate_with_citations("x", "Frage?", passages)

    documents = [c for c in stub.messages.last_request["messages"][0]["content"] if c["type"] == "document"]
    assert documents[0]["title"] == "Passage 1"
    assert "secret_id_123" not in documents[0]["title"]


def test_adapter_maps_single_citation_to_parsed_answer():
    gen = ClaudeGeneration(
        status="success", error_type=None, short_answer="Albert Einstein",
        answer="Albert Einstein entwickelte die Relativitaetstheorie.",
        citations=[ClaudeCitation(document_index=1, cited_text="Er entwickelte sie 1915.",
                                   start_char_index=5, end_char_index=30)],
        latency_ms=42.0,
    )
    parsed = claude_generation_to_parsed_answer(gen)
    assert isinstance(parsed, ParsedAnswer)
    assert parsed.quelle_refs == ("P2",)
    assert parsed.quote == "Er entwickelte sie 1915."


def test_adapter_maps_multiple_citations_without_choosing():
    gen = ClaudeGeneration(
        status="success", error_type=None, short_answer="X", answer="Y.",
        citations=[
            ClaudeCitation(document_index=0, cited_text="a", start_char_index=0, end_char_index=1),
            ClaudeCitation(document_index=2, cited_text="b", start_char_index=0, end_char_index=1),
        ],
        latency_ms=1.0,
    )
    parsed = claude_generation_to_parsed_answer(gen)
    assert parsed.quelle_refs == ("P1", "P3")
    assert parsed.quote is None


def test_adapter_maps_zero_citations_still_answered():
    gen = ClaudeGeneration(
        status="success", error_type=None, short_answer="Albert Einstein",
        answer="Albert Einstein entwickelte die Relativitaetstheorie.",
        citations=[], latency_ms=1.0,
    )
    parsed = claude_generation_to_parsed_answer(gen)
    assert parsed.answered is True
    assert parsed.quelle_refs == ()
    assert parsed.quote is None


def test_adapter_maps_keine_antwort():
    gen = ClaudeGeneration(status="success", error_type=None, short_answer="KEINE ANTWORT",
                            answer="None", citations=[], latency_ms=1.0)
    parsed = claude_generation_to_parsed_answer(gen)
    assert parsed.answered is False
