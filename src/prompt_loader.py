"""Nạp và render prompt template cho RAG —
future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md §2.1/§5.3.

Mỗi prompt có version + hash SHA-256 ĐẦY ĐỦ (không cắt như passage_id của V1) để tái
lập: đổi 1 byte trong file .txt thì prompt_sha256 đổi theo, cho phép audit prompt nào
thực sự tạo ra một run.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.citation import PassageInput

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    path: str
    sha256: str
    text: str


def load_prompt_template(name: str) -> PromptTemplate:
    """`name` không có phần mở rộng, ví dụ "answer_de_v1" hoặc "answer_de_closed_book_v1"."""
    path = PROMPTS_DIR / f"{name}.txt"
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    return PromptTemplate(version=name, path=str(path), sha256=sha256, text=raw_bytes.decode("utf-8"))


def render_rag_prompt(template: PromptTemplate, question: str, passages: list[PassageInput]) -> str:
    passages_block = "\n\n".join(
        f"[P{i}] {p.title or p.passage_id}\n{p.text}" for i, p in enumerate(passages, start=1)
    )
    return template.text.format(passages=passages_block, question=question)


def render_closed_book_prompt(template: PromptTemplate, question: str) -> str:
    return template.text.format(question=question)
