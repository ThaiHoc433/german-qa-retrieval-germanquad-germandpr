"""Schema Pydantic của API (GĐ8). Pydantic chỉ kiểm KIỂU; giới hạn (độ dài, top-k) kiểm trong check_request
vì chúng nằm trong configs/api.yaml chứ không viết cứng ở đây."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

RetrieverName = Literal["bm25", "dense", "hybrid"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")       # khóa lạ trong request là lỗi, không bị lờ đi


class QARequest(_Strict):
    question: str = Field(description="Câu hỏi tiếng Đức. Tối đa max_question_chars ký tự (mặc định 500).")
    retriever: Optional[RetrieverName] = Field(None, description="Mặc định theo configs/api.yaml.")
    retrieval_top_k: Optional[int] = Field(None, description="Số đoạn truy hồi (1..max_retrieval_top_k).")
    reader_top_k: Optional[int] = Field(None, description="Số đoạn đầu vào reader (1..max_reader_top_k, <= retrieval_top_k).")
    trace: bool = Field(False, description="Trả thêm vết: mọi đoạn truy hồi và đáp án của từng đoạn.")
    rerank: bool = Field(False, description="Xếp lại các đoạn truy hồi bằng cross-encoder trước khi đưa cho reader. Chỉ khi máy chủ có nạp reranker; retrieval_top_k khi rerank không được vượt max_rerank_pool.")


class ReadRequest(_Strict):
    question: str = Field(description="Câu hỏi tiếng Đức. Tối đa max_question_chars ký tự.")
    context: str = Field(description="Đoạn văn do người dùng đưa vào (không truy hồi). Tối đa max_context_chars ký tự.")
    expected_answer: Optional[str] = Field(None, description="Đáp án mong đợi (tùy chọn): có thì trả thêm EM/F1, chuẩn hóa giống lúc đánh giá.")


class ReadResult(BaseModel):
    request_id: str
    status: str = Field(description='"ok" | "question_truncated" | "no_valid_span"')
    question: str
    context: str
    answer: str
    answer_start: int = Field(description="Offset ký tự Python trên context; -1 khi không có đáp án.")
    answer_end: int
    answer_start_utf16: int
    answer_end_utf16: int
    reader_score: Optional[float] = Field(None, description="Điểm reader (start + end logit, trừ CLS); không phải xác suất.")
    expected_answer: Optional[str] = None
    em: Optional[float] = Field(None, description="Exact match với expected_answer (0 hoặc 1).")
    f1: Optional[float] = Field(None, description="F1 theo token với expected_answer.")
    timings: Dict[str, float]
    model: str


class RetrieveRequest(_Strict):
    question: str
    retriever: Optional[RetrieverName] = None
    top_k: Optional[int] = Field(None, description="Số đoạn trả về (1..max_retrieval_top_k).")


class Timings(BaseModel):
    queue_ms: float = Field(description="Chờ đến lượt chạy model (một request model tại một thời điểm).")
    retrieval_ms: float
    reader_ms: float
    pipeline_ms: float = Field(description="Thời gian pipeline thật sự chạy, không gồm hàng đợi.")
    total_ms: float = Field(description="Từ lúc route nhận request đến lúc có kết quả.")
    rerank_ms: float = Field(0.0, description="Thời gian xếp lại bằng cross-encoder; 0 khi không rerank.")


class Versions(BaseModel):
    model: str
    corpus: str
    experiment_id: str


class QAResult(BaseModel):
    request_id: str
    status: str = Field(description='"ok" | "question_truncated" | "no_passages" | "no_valid_span" | "empty_question"')
    question: str
    answer: str
    passage_id: str
    passage_text: str
    answer_start: int = Field(description="Offset ký tự Python, nửa mở [start, end) trên passage_text.")
    answer_end: int
    answer_start_utf16: int = Field(description="Cùng vị trí đó theo UTF-16: dùng cho passage_text.slice() trong JavaScript.")
    answer_end_utf16: int
    retrieval_score: float = Field(description="Điểm của bộ truy hồi: BM25, tích vô hướng dense, hoặc điểm RRF với hybrid. Các thang KHÁC nhau, không so với nhau và không phải xác suất.")
    reader_score: float
    reranked: bool = Field(False, description="Có xếp lại bằng cross-encoder không.")
    rerank_score: Optional[float] = Field(None, description="Logit liên quan của cross-encoder cho đoạn được chọn; None khi không rerank. Không phải xác suất.")
    retriever: RetrieverName
    timings: Timings
    versions: Versions
    trace: Optional[Dict[str, Any]] = None


class Candidate(BaseModel):
    passage_id: str
    rank: int
    score: float = Field(description="Điểm của bộ truy hồi (với hybrid là điểm RRF).")
    text: str
    sources: Optional[Dict[str, int]] = Field(None, description="Hybrid: hạng của đoạn trong từng nhánh {bm25: hạng, dense: hạng}; nhánh vắng thì không có khóa.")
    bm25_score: Optional[float] = Field(None, description="Hybrid: điểm BM25 gốc; None nếu đoạn không nằm trong top của nhánh BM25.")
    dense_score: Optional[float] = Field(None, description="Hybrid: tích vô hướng dense gốc; None nếu đoạn không nằm trong top của nhánh dense.")


class RetrieveResult(BaseModel):
    request_id: str
    question: str
    retriever: RetrieverName
    top_k: int
    candidates: List[Candidate]
    timings: Dict[str, float]
    versions: Versions


class RetrieverInfo(BaseModel):
    available: bool
    reason: Optional[str] = Field(None, description="Mã lý do khi không khả dụng: faiss_not_installed, artifacts_missing, ...")


class RerankerInfo(BaseModel):
    configured: Optional[str] = Field(None, description="Tên reranker trong cấu hình; None nếu không bật.")
    available: bool
    reason: Optional[str] = Field(None, description="Mã lý do khi cấu hình có nhưng không dùng được.")
    retrievers: List[str] = Field(default_factory=list, description="Các retriever dùng được với rerank=true.")


class Ready(BaseModel):
    status: Literal["ready", "loading", "failed"]
    default_retriever: str
    retrievers: Dict[str, RetrieverInfo]
    reranker: Optional[RerankerInfo] = None
    model: Optional[str] = None
    corpus: Optional[str] = None
    load_seconds: Optional[float] = None
    detail: Optional[str] = None


class ExperimentItem(BaseModel):
    experiment_id: str
    kind: str
    created_at: Optional[str]
    split: Optional[str]
    retriever: Optional[str]
    checkpoint: Optional[str]
    corpus_version: Optional[str]
    metrics: Dict[str, Dict[str, float]] = Field(description="Số đo theo từng tập: {tập: {tên số đo: giá trị}}.")
    intervals: Dict[str, Dict[str, List[float]]] = Field(
        default_factory=dict, description="Khoảng tin cậy bootstrap 95% của EM/F1 theo tập: {tập: {em: [lo, hi]}}. Chỉ có ở file eval_*.")
    caveats: List[str]
    code: Optional[str] = Field(None, description="Mã ma trận B0/B1/H0/H1 nếu lần chạy này là số đo tham chiếu trong baselines/*.json.")
    duplicate_of: Optional[str] = Field(None, description="experiment_id của lần chạy cùng cấu hình được giữ làm đại diện (lần chạy này là bản trùng).")


class ExperimentList(BaseModel):
    count: int
    skipped: int = Field(description="Số file bị bỏ qua vì không đọc được hoặc không đúng hình dạng.")
    items: List[ExperimentItem]


class MatrixRow(BaseModel):
    code: str
    description: str
    config_id: Optional[str] = None
    frozen_on: Optional[str] = None
    status: Optional[str] = None
    reference_file: Optional[str] = Field(None, description="Tên tệp kết quả (không có đường dẫn) mà số đo lấy từ.")
    splits: Dict[str, Dict[str, float]] = Field(description="{tập: {tên số đo: giá trị}}; tập có thể là dev, dev_unseen.")
    intervals: Dict[str, Dict[str, List[float]]] = Field(default_factory=dict)
    queries: Dict[str, int] = Field(default_factory=dict)


class MatrixResponse(BaseModel):
    count: int
    caveats: List[str]
    rows: List[MatrixRow]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None
    details: Optional[List[Dict[str, Any]]] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
