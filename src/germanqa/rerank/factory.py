"""Dựng reranker từ khối 'rerank' của configs/rerank.yaml (GĐ14). Tách khỏi cross_encoder.py vì file đó đã bị khóa SHA-1 trong prereg."""
from pathlib import Path


def build_reranker(cfg):
    """cfg: khối 'rerank' (checkpoint tương đối so với gốc dự án)."""
    from germanqa.config import ROOT
    from germanqa.rerank.cross_encoder import CrossEncoderReranker
    ck = Path(cfg["checkpoint"])
    return CrossEncoderReranker(ck if ck.is_absolute() else ROOT / ck, max_length=cfg["max_length"], batch_size=cfg["batch_size"],
                                fp16=cfg["fp16"], relevant_label=cfg.get("relevant_label"), overflow=cfg["overflow"],
                                stride=cfg.get("stride", 128))
