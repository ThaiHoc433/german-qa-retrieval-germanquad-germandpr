"""Hợp đồng dữ liệu của dự án (GĐ0): mọi giai đoạn sau đọc/ghi đúng các dạng này.

Quy ước offset: ký tự, nửa mở [start, end) trên original_text (chuỗi gốc chưa sửa).
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Passage:
    passage_id: str            # "p_" + 16 ký tự băm nội dung
    original_text: str         # văn bản gốc, KHÔNG sửa (reader dùng để cắt đáp án)
    retrieval_text: str        # bản đã làm sạch + chữ thường (BM25/dense dùng)
    title: Optional[str]
    document_id: str           # "d_" + băm tiêu đề (gom các đoạn cùng bài viết)
    source: List[dict] = field(default_factory=list)   # provenance


@dataclass
class Answer:
    text: str
    start: Optional[int]       # None nếu không xác định được vị trí (GermanDPR mơ hồ)


@dataclass
class QAExample:
    question_id: str           # "q_" + 16 ký tự băm câu hỏi đã chuẩn hóa
    question: str
    passage_id: str            # đoạn chứa đáp án
    answers: List[Answer]
    origin: str                # "germanquad" | "germandpr"


@dataclass
class QAResponse:
    """Dạng phản hồi mà pipeline (GĐ5) và API (GĐ8) sẽ trả về."""
    question: str
    answer: str
    passage_id: str
    passage_text: str
    answer_start: int          # [start, end) trên passage_text gốc
    answer_end: int
    retrieval_score: float
    reader_score: float
    retrieval_time: float      # giây
    reader_time: float
    total_time: float
    model_version: str
    corpus_version: str
    
    experiment_id: str
    status: str = "ok"         # "ok" | "empty_question" | "no_passages" | "no_valid_span" | "question_truncated"
    trace: Optional[dict] = None   # vết từng bước (GĐ5); None khi không bật
    rerank_time: float = 0.0               # giây (GĐ14); 0 khi không có reranker
    rerank_score: Optional[float] = None   # logit liên quan của reranker cho đoạn được chọn; None khi không có reranker

    def to_dict(self):
        return asdict(self)


def to_utf16_offset(text, index):
    """Đổi vị trí ký tự Python (theo mã Unicode) sang vị trí UTF-16 mà JavaScript dùng.
    Khác nhau khi có ký tự ngoài BMP (ví dụ emoji): Python tính 1, JavaScript tính 2."""
    return len(text[:index].encode("utf-16-le")) // 2
