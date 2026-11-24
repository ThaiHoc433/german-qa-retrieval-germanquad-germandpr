"""Cấu hình API (GĐ8): đọc configs/api.yaml thành một dataclass bất biến."""
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

from germanqa.config import ROOT
from germanqa.experiment import load_config


@dataclass(frozen=True)
class ApiSettings:
    default_retriever: str = "bm25"
    retrievers: List[str] = field(default_factory=lambda: ["bm25"])
    checkpoint: Optional[str] = None
    max_question_chars: int = 500
    max_context_chars: int = 20000
    max_retrieval_top_k: int = 50
    max_reader_top_k: int = 10
    request_timeout_s: float = 30.0
    max_inflight: int = 8
    warmup_question: str = "Wer war Konrad Adenauer?"
    cors_origins: List[str] = field(default_factory=list)
    results_dir: str = "results"
    baselines_dir: str = "baselines"
    reranker: Optional[str] = None
    reranker_models: dict = field(default_factory=dict)
    max_rerank_pool: int = 20

    def results_path(self):
        p = Path(self.results_dir)
        return p if p.is_absolute() else ROOT / p

    def baselines_path(self):
        p = Path(self.baselines_dir)
        return p if p.is_absolute() else ROOT / p


def load_settings(**overrides):
    """Đọc configs/api.yaml; overrides (từ dòng lệnh hoặc test) ghi đè từng trường."""
    raw = load_config("api")["api"]
    names = {f.name for f in fields(ApiSettings)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"api.yaml có khóa lạ: {sorted(unknown)}")
    s = ApiSettings(**{**raw, **{k: v for k, v in overrides.items() if v is not None}})
    if s.default_retriever not in s.retrievers:
        raise ValueError("default_retriever phải nằm trong retrievers")
    if s.reranker is not None and s.reranker not in s.reranker_models:
        raise ValueError(f"reranker '{s.reranker}' không có trong reranker_models: {sorted(s.reranker_models)}")
    if s.max_rerank_pool < 1:
        raise ValueError("max_rerank_pool phải >= 1")
    if s.max_inflight < 1 or s.request_timeout_s <= 0:
        raise ValueError("max_inflight phải >= 1 và request_timeout_s > 0")
    return s
