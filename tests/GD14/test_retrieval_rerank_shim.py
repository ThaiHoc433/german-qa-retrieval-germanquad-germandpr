import ast

from germanqa.config import ROOT
from germanqa.rerank import cross_encoder, factory
from germanqa.retrieval import rerank as shim


def test_lop_boc_tra_ve_dung_doi_tuong_goc():
    assert shim.CrossEncoderReranker is cross_encoder.CrossEncoderReranker
    assert shim.stable_rank is cross_encoder.stable_rank
    assert shim.combine_logits is cross_encoder.combine_logits
    assert shim.build_reranker is factory.build_reranker
    assert shim.MAX_QUERY_TOKENS == cross_encoder.MAX_QUERY_TOKENS


def test_lop_boc_khong_chua_logic():
    """Chỉ import và __all__: không hàm, không lớp, không vòng lặp, để không có hai bản mã cùng làm một việc."""
    tree = ast.parse((ROOT / "src" / "germanqa" / "retrieval" / "rerank.py").read_text(encoding="utf-8"))
    kinds = {type(n).__name__ for n in tree.body}
    assert kinds <= {"Expr", "ImportFrom", "Assign"}, kinds
    assert set(shim.__all__) == {"CrossEncoderReranker", "build_reranker", "combine_logits", "stable_rank", "MAX_QUERY_TOKENS"}
