"""Client sinh câu trả lời — future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md
§2.3. Đây là nơi DUY NHẤT import SDK model (anthropic, hoặc sau này transformers cho
model_backend="open" thật) — đổi model không dụng citation.py/answer_parser.py/answerer.py.

`import anthropic` chỉ nằm trong build_claude_api_client() (local import bên trong hàm),
KHÔNG ở top-level module — nhờ vậy toàn bộ file này và test của nó chạy được mà không
cần cài gói anthropic; chỉ cần cài khi thực sự gọi build_claude_api_client() để chạy thật.
"""
import re
import time
from dataclasses import dataclass


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

    def __init__(self, scripted: list):
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
