import sys
from pathlib import Path
from types import SimpleNamespace

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
    try:
        client.generate("p again")
        assert False, "phải raise khi hết kịch bản"
    except AssertionError as e:
        assert "hết kịch bản" in str(e)


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
    documents = [c for c in request["messages"][0]["content"] if c["type"] == "document"]
    assert len(documents) == 1
    assert documents[0]["citations"] == {"enabled": True}


def test_generate_with_citations_reports_format_violation_without_labels():
    response = SimpleNamespace(content=[_fake_block("Ich bin mir nicht sicher.")])
    client = ClaudeApiClient(_StubAnthropicClient(response))
    gen = client.generate_with_citations("x", "Frage?", [PassageInput("p1", "T", "Text.")])
    assert gen.status == "error"
    assert gen.error_type == "format_violation"


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
