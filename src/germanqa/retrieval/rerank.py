"""Lớp bọc để khớp vị trí trong kế hoạch gốc (GĐ14.1: `retrieval/rerank.py`).

Mã thật nằm ở `germanqa.rerank` (gói riêng vì có nhiều module). Chuyển hẳn vào đây không được: SHA-1 trong
results/14_prereg.json khóa theo đường dẫn của các file đó. File này chỉ re-export tên công khai, KHÔNG chứa logic."""
from germanqa.rerank.cross_encoder import MAX_QUERY_TOKENS, CrossEncoderReranker, combine_logits, stable_rank
from germanqa.rerank.factory import build_reranker

__all__ = ["CrossEncoderReranker", "build_reranker", "combine_logits", "stable_rank", "MAX_QUERY_TOKENS"]
