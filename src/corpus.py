"""Normalize, hash, chunk — module 2 (Data/Splits/Corpus) của
docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md.
"""
import hashlib
import re
import unicodedata
from dataclasses import dataclass

WINDOW_WORDS = 100
OVERLAP_WORDS = 20
STRIDE_WORDS = WINDOW_WORDS - OVERLAP_WORDS


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def passage_id(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    text: str
    char_start: int
    char_end: int  # exclusive


def chunk_text(text: str, window_words: int = WINDOW_WORDS, stride_words: int = STRIDE_WORDS) -> list[Chunk]:
    """Cửa sổ từ cố định, độc lập vị trí đáp án (spec 2.3.c) — KHÔNG mở rộng
    chunk theo đáp án. Offset ký tự lấy từ `\\S+` nên chunk.text luôn là
    substring nguyên bản của `text` (giữ đúng khoảng trắng/dấu câu gốc)."""
    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        return []
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        window = tokens[i : i + window_words]
        start, end = window[0].start(), window[-1].end()
        chunks.append(Chunk(text=text[start:end], char_start=start, char_end=end))
        if i + window_words >= n:
            break
        i += stride_words
    return chunks


def find_span_chunks(chunks: list[Chunk], answer_start: int, answer_end: int) -> list[int]:
    """Index (trong `chunks`) của mọi chunk chứa TRỌN [answer_start, answer_end)."""
    return [
        idx
        for idx, c in enumerate(chunks)
        if c.char_start <= answer_start and answer_end <= c.char_end
    ]


SOURCE_PRIORITY = {"germanquad": 0, "germandpr_positive": 1, "wikipedia_distractor": 2}


class CorpusBuilder:
    """Gom passage từ nhiều nguồn, chunk, dedup theo passage_id.
    Giữ `text` đại diện theo priority nguồn khi 2 chunk trùng hash (spec 2.3.a)."""

    def __init__(self):
        self.passages: dict[str, dict] = {}  # passage_id -> PassageRecord đại diện

    def add_text(self, full_text: str, source: str, *, title: str | None = None) -> list[Chunk]:
        if source not in SOURCE_PRIORITY:
            raise ValueError(f"source không được phép vào corpus đánh giá: {source!r}")
        chunks = chunk_text(full_text)
        if title is None and source == "germanquad":
            # Chỉ lấy dòng đầu; không đoán title từ dòng sau khi dòng đầu rỗng.
            lines = full_text.splitlines()
            title = lines[0] if lines else None
        if title is not None and not title.strip():
            title = None
        parent_id = passage_id(full_text)
        for c in chunks:
            pid = passage_id(c.text)
            entry = self.passages.get(pid)
            if entry is None or SOURCE_PRIORITY[source] < SOURCE_PRIORITY[entry["source"]]:
                # Thay cả record để text/title/length/provenance luôn cùng occurrence.
                # Cùng priority giữ bản đầu tiên theo thứ tự input.
                self.passages[pid] = {
                    "passage_id": pid,
                    "text": c.text,
                    "normalized_text": normalize_text(c.text),
                    "source": source,
                    "title": title,
                    "char_len": len(c.text),
                    "parent_document_id": parent_id,
                    "chunk_start_char": c.char_start,
                    "chunk_end_char": c.char_end,
                }
        return chunks

    def records(self) -> list[dict]:
        return [dict(p) for p in self.passages.values()]
