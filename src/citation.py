"""Kiểm chứng trích dẫn — future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md §2.4.

Chỉ hàm thuần, không gọi model. Nhận passage đã có sẵn (đầu ra Selection của pipeline V1)
và field đã được answer_parser.py tách ra từ output model; tự tìm `quote` trong đúng
passage, tính vị trí ký tự, không tự sửa khi model trích sai passage (wrong_passage).
"""
import re
import unicodedata
from dataclasses import dataclass

from src.evaluation import normalize_answer


@dataclass(frozen=True)
class PassageInput:
    """Một trong tối đa 5 passage đưa vào prompt — dùng chung cho render prompt (module
    prompt_loader) và kiểm chứng trích dẫn (module này)."""
    passage_id: str
    title: str | None
    text: str


@dataclass(frozen=True)
class QuoteMatch:
    """Vị trí một đoạn trích trong MỘT văn bản cụ thể — dùng cho cả quote-trong-passage
    (Task 1) và short_answer-trong-quote (Task 2)."""
    quote_start: int
    quote_end: int  # exclusive
    quote_match_method: str  # "exact" | "normalized"
    quote_occurrence_count: int


def find_quote_in_passage(passage_text: str, quote: str) -> QuoteMatch | None:
    """Tìm `quote` như một chuỗi con của `passage_text`. Thử khớp chính xác trước; nếu
    không có, thử khớp theo từng từ sau khi chuẩn hóa NFC (không xóa dấu câu, không xóa
    hoa/thường — khác `normalize_answer` của EM/F1). KHÔNG fuzzy match: hai mức trên là
    toàn bộ những gì được chấp nhận (spec §2.4 bước 3)."""
    exact_positions = [m.start() for m in re.finditer(re.escape(quote), passage_text)]
    if exact_positions:
        start = exact_positions[0]
        return QuoteMatch(start, start + len(quote), "exact", len(exact_positions))
    return _find_quote_normalized(passage_text, quote)


def _find_quote_normalized(passage_text: str, quote: str) -> QuoteMatch | None:
    quote_tokens = [unicodedata.normalize("NFC", w) for w in quote.split()]
    if not quote_tokens:
        return None

    passage_tokens = [(m, unicodedata.normalize("NFC", m.group())) for m in re.finditer(r"\S+", passage_text)]
    passage_words = [w for _, w in passage_tokens]

    n = len(quote_tokens)
    occurrences = [
        i for i in range(len(passage_words) - n + 1)
        if passage_words[i : i + n] == quote_tokens
    ]
    if not occurrences:
        return None

    first = occurrences[0]
    start = passage_tokens[first][0].start()
    end = passage_tokens[first + n - 1][0].end()
    return QuoteMatch(start, end, "normalized", len(occurrences))


def is_short_answer_supported(quote: str, short_answer: str) -> bool:
    """§2.4 bước 5: dùng CÙNG cách chuẩn hóa EM/F1 của V1 (normalize_answer: NFC,
    lowercase, xóa dấu câu, gộp khoảng trắng) — KHÁC cách chuẩn hóa NFC+khoảng-trắng
    dùng khi tìm quote trong passage (find_quote_in_passage)."""
    norm_quote = normalize_answer(quote)
    norm_answer = normalize_answer(short_answer)
    return bool(norm_answer) and norm_answer in norm_quote


def find_short_answer_offsets(quote: str, short_answer: str) -> QuoteMatch | None:
    """Vị trí ký tự CHÍNH XÁC của short_answer trong quote GỐC (chưa chuẩn hóa) — chỉ có
    giá trị khi short_answer là chuỗi con chính xác. `is_short_answer_supported` vẫn có
    thể True dù hàm này trả None (khớp chỉ sau chuẩn hóa EM/F1, không có offset chính
    xác) — hai hàm độc lập, không suy hàm này từ hàm kia."""
    positions = [m.start() for m in re.finditer(re.escape(short_answer), quote)]
    if not positions:
        return None
    start = positions[0]
    return QuoteMatch(start, start + len(short_answer), "exact", len(positions))
