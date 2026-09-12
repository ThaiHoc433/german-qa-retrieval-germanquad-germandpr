"""Parser cho định dạng output 4 dòng KURZ/ANTWORT/QUELLE/ZITAT —
future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md §2.2.

Dùng cho model_backend="open" (và FakeLLMClient trong test — xem llm_clients.py).
model_backend="claude" KHÔNG dùng parser này: Claude chỉ được yêu cầu trả KURZ/ANTWORT;
QUELLE/ZITAT lấy từ citations block của API (xem llm_clients.claude_generation_to_parsed_answer),
không phải từ text tự do.
"""
import re
from dataclasses import dataclass

_LABEL_RE = re.compile(r"^(KURZ|ANTWORT|QUELLE|ZITAT):[ \t]*(.*)$", re.MULTILINE)
_QUELLE_REF_RE = re.compile(r"P(\d+)")


@dataclass(frozen=True)
class ParsedAnswer:
    """Kết quả parse, đã chuẩn hóa về một dạng chung cho cả model_backend="open" và
    "claude" (xem llm_clients.claude_generation_to_parsed_answer). Chỉ đọc các field
    answered/short_answer/... khi format_violation=False."""
    format_violation: bool
    answered: bool | None = None
    short_answer: str | None = None       # từ KURZ
    answer: str | None = None             # từ ANTWORT
    quelle_refs: tuple[str, ...] = ()      # ví dụ ("P1",) hoặc ("P1", "P3")
    quote: str | None = None              # từ ZITAT, đã bỏ dấu ngoặc kép hai đầu


def extract_first_labels(raw_text: str, allowed_labels: tuple[str, ...]) -> dict[str, str] | None:
    """Tách các nhãn dòng (vd KURZ/ANTWORT/QUELLE/ZITAT hoặc chỉ KURZ/ANTWORT cho nhánh
    Claude) — dùng chung cho model_backend="open" (parse_llm_output) và "claude"
    (llm_clients._extract_kurz_antwort), để hai backend không lệch định nghĩa
    format_violation (§2.2/§2.3). `[ \\t]*` (không phải `\\s*`) — KHÔNG được ăn qua dấu
    xuống dòng, nếu không giá trị rỗng của một nhãn sẽ nuốt luôn dòng tiếp theo làm giá
    trị của chính nó (bug đã phát hiện ở review cuối). Trả None nếu một nhãn xuất hiện
    nhiều hơn 1 lần. Nhãn thiếu hoàn toàn thì đơn giản không có trong dict trả về — caller
    tự quyết định nhãn nào bắt buộc phải có."""
    pattern = re.compile(
        r"^(" + "|".join(re.escape(l) for l in allowed_labels) + r"):[ \t]*(.*)$",
        re.MULTILINE,
    )
    labels: dict[str, str] = {}
    for match in pattern.finditer(raw_text):
        key = match.group(1)
        if key in labels:
            return None
        labels[key] = match.group(2).strip()
    return labels


def parse_llm_output(raw_text: str) -> ParsedAnswer:
    labels = extract_first_labels(raw_text, ("KURZ", "ANTWORT", "QUELLE", "ZITAT"))
    if labels is None:
        return ParsedAnswer(format_violation=True)

    kurz = labels.get("KURZ", "")
    antwort = labels.get("ANTWORT", "")
    if not kurz or not antwort:
        return ParsedAnswer(format_violation=True)

    if kurz == "KEINE ANTWORT":
        if antwort == "None" and labels.get("QUELLE", "") == "None" and labels.get("ZITAT", "") == "None":
            return ParsedAnswer(format_violation=False, answered=False)
        return ParsedAnswer(format_violation=True)

    quelle_value = labels.get("QUELLE", "")
    zitat_value = labels.get("ZITAT", "").strip()
    if zitat_value.startswith('"') and zitat_value.endswith('"') and len(zitat_value) >= 2:
        zitat_value = zitat_value[1:-1]

    quelle_refs = tuple(f"P{n}" for n in _QUELLE_REF_RE.findall(quelle_value))
    has_quote = bool(zitat_value) and zitat_value != "None"
    if not quelle_refs or not has_quote:
        # thiếu một nửa (hoặc cả hai) của cặp QUELLE/ZITAT -> citation_status=missing ở
        # citation.verify_citation (Task 4), KHÔNG phải format_violation (design §2.2).
        quelle_refs = ()
        zitat_value = None

    return ParsedAnswer(
        format_violation=False, answered=True,
        short_answer=kurz, answer=antwort,
        quelle_refs=quelle_refs, quote=zitat_value,
    )
