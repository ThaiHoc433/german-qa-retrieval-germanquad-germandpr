"""Client sinh câu trả lời — future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md
§2.3. Đây là nơi DUY NHẤT import SDK model (anthropic, hoặc sau này transformers cho
model_backend="open" thật) — đổi model không dụng citation.py/answer_parser.py/answerer.py.

`import anthropic` chỉ nằm trong build_claude_api_client() (local import bên trong hàm),
KHÔNG ở top-level module — nhờ vậy toàn bộ file này và test của nó chạy được mà không
cần cài gói anthropic; chỉ cần cài khi thực sự gọi build_claude_api_client() để chạy thật.
"""
import time
from dataclasses import dataclass

from src.answer_parser import ParsedAnswer, extract_first_labels
from src.citation import PassageInput


@dataclass(frozen=True)
class RawGeneration:
    """Kết quả một lượt gọi model_backend="open" (text tự do, chưa parse)."""
    status: str  # "success" | "error"
    error_type: str | None
    raw_text: str | None  # None khi status="error"
    latency_ms: float


class FakeLLMClient:
    """Test double cho model_backend="open" — không cần model/GPU thật. Kịch bản trả về
    theo đúng thứ tự gọi, để test LLMAnswerer (Task 8) đủ mọi nhánh trạng thái mà không
    phụ thuộc mạng/model. `scripted[i]` là str (thành công) hoặc RawGeneration (kể cả
    lỗi, dùng để giả lập timeout/refusal)."""

    def __init__(self, scripted: list[str | RawGeneration]):
        self._scripted = list(scripted)
        self._calls = 0

    def generate(self, prompt: str) -> RawGeneration:
        if self._calls >= len(self._scripted):
            raise AssertionError("FakeLLMClient hết kịch bản — gọi nhiều hơn số scripted")
        item = self._scripted[self._calls]
        self._calls += 1
        if isinstance(item, RawGeneration):
            return item
        return RawGeneration(status="success", error_type=None, raw_text=item, latency_ms=1.0)


def _extract_kurz_antwort(full_text: str) -> tuple[str, str] | None:
    """None nếu nhãn lặp lại HOẶC KURZ/ANTWORT thiếu/rỗng — dùng CHUNG định nghĩa
    format_violation với model_backend="open" (answer_parser.extract_first_labels), để
    hai backend không lệch nhau khi so sánh format-violation rate (§3.5)."""
    labels = extract_first_labels(full_text, ("KURZ", "ANTWORT"))
    if labels is None:
        return None
    kurz, antwort = labels.get("KURZ", ""), labels.get("ANTWORT", "")
    if not kurz or not antwort:
        return None
    return kurz, antwort


@dataclass(frozen=True)
class ClaudeCitation:
    document_index: int
    cited_text: str
    start_char_index: int | None
    end_char_index: int | None


@dataclass(frozen=True)
class ClaudeGeneration:
    """Kết quả một lượt gọi model_backend="claude" — khác RawGeneration vì citations
    của Claude là dữ liệu có cấu trúc (document_index/cited_text), không phải text tự
    do như QUELLE/ZITAT của model_backend="open"."""
    status: str
    error_type: str | None
    short_answer: str | None
    answer: str | None
    citations: list[ClaudeCitation]
    latency_ms: float


class ClaudeApiClient:
    """model_backend="claude". `client` là object có `.messages.create(**kwargs)` —
    thường là `anthropic.Anthropic()` thật (xem build_claude_api_client), hoặc stub
    trong test. Prompt chỉ yêu cầu KURZ/ANTWORT (§2.3) — QUELLE/ZITAT lấy từ block
    `citations` mà API tự trả kèm mỗi đoạn text đã trích dẫn."""

    MODEL_ID = "claude-opus-5"

    def __init__(self, client, model_id: str = MODEL_ID, max_tokens: int = 4096):
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens

    def generate_with_citations(
        self, prompt_instructions: str, question: str, passages: list[PassageInput]
    ) -> ClaudeGeneration:
        t0 = time.perf_counter()
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": p.text},
                "title": p.title or f"Passage {i}",
                "citations": {"enabled": True},
            }
            for i, p in enumerate(passages, start=1)
        ]
        try:
            response = self._client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                output_config={"effort": "low"},
                messages=[{
                    "role": "user",
                    "content": [*documents, {"type": "text", "text": f"{prompt_instructions}\n\nQuestion: {question}"}],
                }],
            )
        except Exception:  # lỗi mạng/API — không phân loại chi tiết ở bản này
            return ClaudeGeneration(
                status="error", error_type="api_error", short_answer=None, answer=None,
                citations=[], latency_ms=(time.perf_counter() - t0) * 1000,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            return ClaudeGeneration(
                status="error", error_type="refusal", short_answer=None, answer=None,
                citations=[], latency_ms=latency_ms,
            )
        if stop_reason == "max_tokens":
            return ClaudeGeneration(
                status="error", error_type="truncated", short_answer=None, answer=None,
                citations=[], latency_ms=latency_ms,
            )

        full_text_parts, citations = [], []
        for block in response.content:
            if getattr(block, "type", None) != "text":
                continue
            full_text_parts.append(block.text)
            for c in (getattr(block, "citations", None) or []):
                citations.append(ClaudeCitation(
                    document_index=c.document_index, cited_text=c.cited_text,
                    start_char_index=getattr(c, "start_char_index", None),
                    end_char_index=getattr(c, "end_char_index", None),
                ))

        extracted = _extract_kurz_antwort("".join(full_text_parts))
        if extracted is None:
            return ClaudeGeneration(
                status="error", error_type="format_violation", short_answer=None, answer=None,
                citations=[], latency_ms=latency_ms,
            )
        kurz, antwort = extracted
        return ClaudeGeneration(
            status="success", error_type=None, short_answer=kurz, answer=antwort,
            citations=citations, latency_ms=latency_ms,
        )


def claude_generation_to_parsed_answer(gen: ClaudeGeneration) -> ParsedAnswer:
    """Chuẩn hóa ClaudeGeneration về CÙNG dạng ParsedAnswer mà citation.verify_citation
    dùng cho model_backend="open" — nhờ vậy verify_citation KHÔNG cần biết backend nào
    đã sinh ra câu trả lời."""
    if gen.short_answer is None or gen.short_answer == "KEINE ANTWORT":
        return ParsedAnswer(format_violation=False, answered=False)
    if len(gen.citations) > 1:
        refs = tuple(f"P{c.document_index + 1}" for c in gen.citations)
        return ParsedAnswer(
            format_violation=False, answered=True,
            short_answer=gen.short_answer, answer=gen.answer,
            quelle_refs=refs, quote=None,
        )
    if not gen.citations:
        return ParsedAnswer(
            format_violation=False, answered=True,
            short_answer=gen.short_answer, answer=gen.answer,
            quelle_refs=(), quote=None,
        )
    only = gen.citations[0]
    return ParsedAnswer(
        format_violation=False, answered=True,
        short_answer=gen.short_answer, answer=gen.answer,
        quelle_refs=(f"P{only.document_index + 1}",), quote=only.cited_text,
    )


def build_claude_api_client(model_id: str = ClaudeApiClient.MODEL_ID) -> ClaudeApiClient:
    """Factory dùng khi chạy THẬT — import anthropic CHỈ ở đây (local import), để phần
    còn lại của module (và mọi test) không cần cài gói anthropic. Đọc API key theo cơ
    chế resolve credential mặc định của SDK (biến môi trường ANTHROPIC_API_KEY, hoặc
    `ant auth login`) — không tự đọc file .env ở đây, đó là việc của app/service.py
    (ngoài phạm vi plan này, xem design §5.4)."""
    import anthropic
    return ClaudeApiClient(anthropic.Anthropic(), model_id=model_id)
