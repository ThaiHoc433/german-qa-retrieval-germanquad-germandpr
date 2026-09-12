"""Answerer interface + LLMAnswerer — ghép llm_clients + answer_parser + citation
thành một lượt sinh câu trả lời hoàn chỉnh theo §2 của
future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md.

ExtractiveAnswerer (bọc Reader V1) KHÔNG nằm trong file này — thêm sau khi Reader V1 có
checkpoint (design §7.2 mục 5), ngoài phạm vi plan này.
"""
from dataclasses import dataclass
from typing import Literal

from src.answer_parser import ParsedAnswer, parse_llm_output
from src.citation import CitationDetail, CitationOutcome, PassageInput, verify_citation
from src.llm_clients import ClaudeGeneration, claude_generation_to_parsed_answer
from src.prompt_loader import PromptTemplate, render_closed_book_prompt, render_rag_prompt

AnswererType = Literal["closed_book", "rag", "oracle_rag"]
ModelBackend = Literal["open", "claude"]


@dataclass(frozen=True)
class AnswerResult:
    """Schema đầy đủ theo design §2.4/§2.5. Việc gắn thêm config (model_revision,
    prompt_sha256, ...) vào PredictionRecord là việc của batch runner/web sau này —
    ngoài phạm vi plan này."""
    generation_status: str  # "success" | "error"
    error_type: str | None
    answered: bool | None
    short_answer: str | None
    answer: str | None
    citation_status: str | None
    answer_supported: bool | None
    citations: list[CitationDetail]
    raw_citation_refs: tuple[str, ...] | None
    latency_ms: float


def _error_result(error_type: str, latency_ms: float) -> AnswerResult:
    return AnswerResult(
        generation_status="error", error_type=error_type, answered=None,
        short_answer=None, answer=None, citation_status=None, answer_supported=None,
        citations=[], raw_citation_refs=None, latency_ms=latency_ms,
    )


def _no_answer_result(parsed: ParsedAnswer, latency_ms: float) -> AnswerResult:
    return AnswerResult(
        generation_status="success", error_type=None, answered=False,
        short_answer=None, answer=None, citation_status=None, answer_supported=None,
        citations=[], raw_citation_refs=None, latency_ms=latency_ms,
    )


def _answered_without_citation_result(parsed: ParsedAnswer, latency_ms: float) -> AnswerResult:
    """Dùng cho closed_book (§2.5): answered=True nhưng citation_status LUÔN None, kể cả
    khi parsed có quelle_refs/quote (model lỡ tự thêm dù prompt không yêu cầu)."""
    return AnswerResult(
        generation_status="success", error_type=None, answered=True,
        short_answer=parsed.short_answer, answer=parsed.answer,
        citation_status=None, answer_supported=None, citations=[], raw_citation_refs=None,
        latency_ms=latency_ms,
    )


def _with_citation_result(parsed: ParsedAnswer, outcome: CitationOutcome, latency_ms: float) -> AnswerResult:
    return AnswerResult(
        generation_status="success", error_type=None, answered=True,
        short_answer=parsed.short_answer, answer=parsed.answer,
        citation_status=outcome.citation_status, answer_supported=outcome.answer_supported,
        citations=outcome.citations, raw_citation_refs=outcome.raw_citation_refs,
        latency_ms=latency_ms,
    )


class LLMAnswerer:
    """`answerer_type="closed_book"` chỉ hỗ trợ `model_backend="open"` trong plan này
    (khớp ma trận thực nghiệm design §3.3 — model_backend="claude" chỉ chạy "rag").
    `answerer_type="oracle_rag"` dùng lại cùng đường xử lý với "rag": khác nhau ở việc
    caller đưa gold passage hay kết quả retrieval vào `passages`, LLMAnswerer không cần
    biết khác biệt đó."""

    def __init__(self, model_backend: ModelBackend, client, prompt_template: PromptTemplate,
                 closed_book_prompt_template: PromptTemplate | None = None,
                 claude_prompt_template: PromptTemplate | None = None):
        self._model_backend = model_backend
        self._client = client
        self._prompt_template = prompt_template
        self._closed_book_prompt_template = closed_book_prompt_template
        self._claude_prompt_template = claude_prompt_template

    def answer(self, answerer_type: AnswererType, question: str,
               passages: list[PassageInput]) -> AnswerResult:
        if answerer_type == "closed_book":
            return self._answer_closed_book(question)
        return self._answer_with_passages(question, passages)

    def _answer_closed_book(self, question: str) -> AnswerResult:
        if self._closed_book_prompt_template is None:
            raise ValueError("LLMAnswerer cần closed_book_prompt_template để trả lời closed_book")
        prompt = render_closed_book_prompt(self._closed_book_prompt_template, question)
        parsed, latency_ms, error_result = self._generate_and_parse_open(prompt)
        if error_result is not None:
            return error_result
        if parsed.format_violation:
            return _error_result("format_violation", latency_ms)
        if not parsed.answered:
            return _no_answer_result(parsed, latency_ms)
        return _answered_without_citation_result(parsed, latency_ms)

    def _answer_with_passages(self, question: str, passages: list[PassageInput]) -> AnswerResult:
        if self._model_backend == "claude":
            if self._claude_prompt_template is None:
                raise ValueError("LLMAnswerer cần claude_prompt_template để trả lời model_backend=\"claude\"")
            gen: ClaudeGeneration = self._client.generate_with_citations(
                self._claude_prompt_template.text, question, passages
            )
            if gen.status == "error":
                return _error_result(gen.error_type, gen.latency_ms)
            parsed = claude_generation_to_parsed_answer(gen)
            latency_ms = gen.latency_ms
        else:
            prompt = render_rag_prompt(self._prompt_template, question, passages)
            parsed, latency_ms, error_result = self._generate_and_parse_open(prompt)
            if error_result is not None:
                return error_result

        if parsed.format_violation:
            return _error_result("format_violation", latency_ms)
        if not parsed.answered:
            return _no_answer_result(parsed, latency_ms)

        outcome = verify_citation(passages, parsed)
        return _with_citation_result(parsed, outcome, latency_ms)

    def _generate_and_parse_open(self, prompt: str):
        gen = self._client.generate(prompt)
        if gen.status == "error":
            return None, gen.latency_ms, _error_result(gen.error_type, gen.latency_ms)
        return parse_llm_output(gen.raw_text), gen.latency_ms, None
