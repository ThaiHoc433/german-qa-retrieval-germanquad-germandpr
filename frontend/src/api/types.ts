// Kiểu khớp với src/germanqa/api/schemas.py (GĐ8). Đổi schema phía Python thì sửa file này.

export type RetrieverName = "bm25" | "dense" | "hybrid";

export interface Timings {
  queue_ms: number;
  retrieval_ms: number;
  reader_ms: number;
  pipeline_ms: number;
  total_ms: number;
  /** GĐ18: thời gian xếp lại bằng cross-encoder; 0 hoặc vắng khi không rerank. */
  rerank_ms?: number;
}

export interface Versions {
  model: string;
  corpus: string;
  experiment_id: string;
}

export interface TraceRetrieved {
  passage_id: string;
  score: number; // điểm của bộ truy hồi (hybrid: điểm RRF)
  rank: number;
  /** Hybrid: hạng của đoạn trong từng nhánh; nhánh vắng thì không có khóa. */
  sources?: { bm25?: number; dense?: number };
  bm25_score?: number | null;
  dense_score?: number | null;
}

export interface TraceReranked {
  passage_id: string;
  rank_before: number;
  rank: number;
  rerank_score: number; // logit của cross-encoder, KHÔNG phải xác suất
}

export interface TraceCandidate {
  passage_id: string;
  rank: number;
  retrieval_score: number;
  answer: string;
  start: number;
  end: number;
  reader_score: number | null; // -inf phía Python được API đổi thành null
  rank_before_rerank?: number;
  rerank_score?: number;
}

export interface QAResult {
  request_id: string;
  status: "ok" | "question_truncated" | "no_passages" | "no_valid_span" | "empty_question";
  question: string;
  answer: string;
  passage_id: string;
  passage_text: string;
  answer_start: number; // offset ký tự Python
  answer_end: number;
  answer_start_utf16: number; // offset UTF-16: dùng cho slice() của JavaScript
  answer_end_utf16: number;
  retrieval_score: number;
  reader_score: number;
  /** GĐ18: có xếp lại bằng cross-encoder không (vắng ở API cũ). */
  reranked?: boolean;
  rerank_score?: number | null;
  retriever: RetrieverName;
  timings: Timings;
  versions: Versions;
  trace: {
    retrieved: TraceRetrieved[];
    candidates: TraceCandidate[];
    reranked?: TraceReranked[];
    rerank_stats?: { pairs: number; windows: number; seconds: number };
  } | null;
}

export interface Candidate {
  passage_id: string;
  rank: number;
  score: number;
  text: string;
  sources?: Record<string, number> | null;
  bm25_score?: number | null;
  dense_score?: number | null;
}

export interface RetrieveResult {
  request_id: string;
  question: string;
  retriever: RetrieverName;
  top_k: number;
  candidates: Candidate[];
  timings: Record<string, number>;
  versions: Versions;
}

export interface RetrieverInfo {
  available: boolean;
  reason: string | null;
}

export interface RerankerInfo {
  configured: string | null;
  available: boolean;
  reason: string | null;
  /** Các retriever dùng được với rerank=true. */
  retrievers: string[];
}

export interface Ready {
  status: "ready" | "loading" | "failed";
  default_retriever: string;
  retrievers: Record<string, RetrieverInfo>;
  reranker?: RerankerInfo | null;
  model: string | null;
  corpus: string | null;
  load_seconds: number | null;
  detail: string | null;
}

export interface ExperimentItem {
  experiment_id: string;
  kind: string;
  created_at: string | null;
  split: string | null;
  retriever: string | null;
  checkpoint: string | null;
  corpus_version: string | null;
  metrics: Record<string, Record<string, number>>;
  intervals: Record<string, Record<string, [number, number]>>;
  caveats: string[];
  /** Mã ma trận B0/B1/H0/H1 nếu là số đo tham chiếu trong baselines/*.json. */
  code?: string | null;
  /** experiment_id của lần chạy đại diện nếu lần này trùng cấu hình. */
  duplicate_of?: string | null;
}

export interface MatrixRow {
  code: string;
  description: string;
  config_id: string | null;
  frozen_on: string | null;
  status: string | null;
  reference_file: string | null;
  splits: Record<string, Record<string, number>>;
  intervals: Record<string, Record<string, [number, number]>>;
  queries: Record<string, number>;
}

export interface MatrixResponse {
  count: number;
  caveats: string[];
  rows: MatrixRow[];
}

export interface ExperimentList {
  count: number;
  skipped: number;
  items: ExperimentItem[];
}

export interface AskParams {
  question: string;
  retriever?: RetrieverName;
  trace?: boolean;
  rerank?: boolean;
}

/** /api/v1/read: đọc hiểu với đoạn văn tự nhập (không truy hồi). */
export interface ReadParams {
  question: string;
  context: string;
  expected_answer?: string;
}

export interface ReadResult {
  request_id: string;
  status: "ok" | "question_truncated" | "no_valid_span" | string;
  question: string;
  context: string;
  answer: string;
  answer_start: number;
  answer_end: number;
  answer_start_utf16: number;
  answer_end_utf16: number;
  reader_score: number | null;
  expected_answer: string | null;
  em: number | null;
  f1: number | null;
  timings: { queue_ms: number; reader_ms: number; total_ms: number };
  model: string;
}

export interface Api {
  ask(params: AskParams, signal?: AbortSignal): Promise<QAResult>;
  read(params: ReadParams, signal?: AbortSignal): Promise<ReadResult>;
  retrieve(params: { question: string; retriever?: RetrieverName; top_k?: number }, signal?: AbortSignal): Promise<RetrieveResult>;
  getReady(signal?: AbortSignal): Promise<Ready>;
  getExperiments(signal?: AbortSignal): Promise<ExperimentList>;
  getMatrix(signal?: AbortSignal): Promise<MatrixResponse>;
}
