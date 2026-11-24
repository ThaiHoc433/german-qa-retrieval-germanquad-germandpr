import type { Api, ExperimentList, MatrixResponse, QAResult, ReadResult, Ready, RetrieveResult } from "../api/types";

export const readyBody = (over: Partial<Ready> = {}): Ready => ({
  status: "ready",
  default_retriever: "bm25",
  retrievers: { bm25: { available: true, reason: null }, dense: { available: true, reason: null } },
  model: "reader_x",
  corpus: "abc123",
  load_seconds: 18.6,
  detail: null,
  reranker: { configured: null, available: false, reason: "not_configured", retrievers: [] },
  ...over,
});

/** Máy chủ GĐ18: có hybrid và reranker; reranker chỉ dùng được sau hybrid (pipeline H1). */
export const readyHybrid = (over: Partial<Ready> = {}): Ready =>
  readyBody({
    default_retriever: "hybrid",
    retrievers: { hybrid: { available: true, reason: null }, bm25: { available: true, reason: null }, dense: { available: true, reason: null } },
    reranker: { configured: "mmarco", available: true, reason: null, retrievers: ["hybrid"] },
    ...over,
  });

/** Kết quả hybrid có xếp lại: p3 từ hạng 3 lên hạng 1; hybrid trả hạng từng nhánh. */
export const qaReranked = (over: Partial<QAResult> = {}): QAResult =>
  qa({
    retriever: "hybrid",
    reranked: true,
    rerank_score: 4.5,
    retrieval_score: 0.081,
    timings: { queue_ms: 0.2, retrieval_ms: 44, reader_ms: 70, pipeline_ms: 700, total_ms: 702, rerank_ms: 585 },
    trace: {
      retrieved: [
        { passage_id: "p1", score: 0.1, rank: 1, sources: { bm25: 1, dense: 3 } },
        { passage_id: "p_emoji", score: 0.09, rank: 2, sources: { bm25: 2, dense: 2 } },
        { passage_id: "p3", score: 0.081, rank: 3, sources: { dense: 1 } },
      ],
      candidates: [
        { passage_id: "p3", rank: 1, retrieval_score: 0.081, answer: "Vogel", start: 4, end: 9, reader_score: 2, rank_before_rerank: 3, rerank_score: 4.5 },
        { passage_id: "p_emoji", rank: 2, retrieval_score: 0.09, answer: "Katze", start: 7, end: 12, reader_score: 7.25, rank_before_rerank: 2, rerank_score: 1.5 },
      ],
      reranked: [
        { passage_id: "p3", rank_before: 3, rank: 1, rerank_score: 4.5 },
        { passage_id: "p_emoji", rank_before: 2, rank: 2, rerank_score: 1.5 },
        { passage_id: "p1", rank_before: 1, rank: 3, rerank_score: -2 },
      ],
      rerank_stats: { pairs: 3, windows: 3, seconds: 0.585 },
    },
    ...over,
  });

export const qa = (over: Partial<QAResult> = {}): QAResult => ({
  request_id: "r1",
  status: "ok",
  question: "Wer?",
  answer: "Katze",
  passage_id: "p_emoji",
  passage_text: "😀😀 Die Katze schläft leise.",
  answer_start: 7,
  answer_end: 12,
  answer_start_utf16: 9,
  answer_end_utf16: 14,
  retrieval_score: 9.5,
  reader_score: 7.25,
  retriever: "bm25",
  timings: { queue_ms: 0.2, retrieval_ms: 1.1, reader_ms: 50, pipeline_ms: 51.3, total_ms: 52 },
  versions: { model: "reader_x", corpus: "abc123", experiment_id: "pipeline_x" },
  trace: {
    retrieved: [
      { passage_id: "p1", score: 10, rank: 1 },
      { passage_id: "p_emoji", score: 9.5, rank: 2 },
      { passage_id: "p3", score: 8, rank: 3 },
    ],
    candidates: [
      { passage_id: "p1", rank: 1, retrieval_score: 10, answer: "Hund", start: 4, end: 8, reader_score: 3 },
      { passage_id: "p_emoji", rank: 2, retrieval_score: 9.5, answer: "Katze", start: 7, end: 12, reader_score: 7.25 },
    ],
  },
  ...over,
});

export const retrieved = (): RetrieveResult => ({
  request_id: "r2",
  question: "Wer?",
  retriever: "bm25",
  top_k: 3,
  candidates: [
    { passage_id: "p1", rank: 1, score: 10, text: "Der Hund bellt." },
    { passage_id: "p_emoji", rank: 2, score: 9.5, text: "😀😀 Die Katze schläft leise." },
    { passage_id: "p3", rank: 3, score: 8, text: "Ein Vogel singt." },
  ],
  timings: { queue_ms: 0, retrieval_ms: 1, total_ms: 2 },
  versions: { model: "reader_x", corpus: "abc123", experiment_id: "pipeline_x" },
});

export const experiments = (): ExperimentList => ({
  count: 2,
  skipped: 1,
  items: [
    {
      experiment_id: "eval_dev_dense_20261104T121120Z_57716bd8",
      kind: "eval_dev_dense",
      created_at: "2026-11-04T12:11:20+00:00",
      split: "dev",
      retriever: "dense",
      checkpoint: "reader_x",
      corpus_version: "abc123",
      metrics: { dev: { em: 0.3376, f1: 0.514, bm25_em: 0.2364 }, dev_unseen: { em: 0.201, f1: 0.419, bm25_em: 0.171 } },
      intervals: { dev: { em: [0.31, 0.36] }, dev_unseen: { em: [0.15, 0.25] } },
      caveats: ["Tập dev bị nhiễm với encoder dense."],
    },
    {
      experiment_id: "eval_test_20261103T170542Z_93098dcb",
      kind: "eval_test",
      created_at: "2026-11-03T17:05:42+00:00",
      split: "test",
      retriever: "bm25",
      checkpoint: "reader_x",
      corpus_version: "abc123",
      metrics: { test: { em: 0.2224, "hit@5": 0.7926 } },
      intervals: {},
      caveats: [],
    },
  ],
});

export const matrix = (): MatrixResponse => ({
  count: 3,
  caveats: ["H1 chỉ đo trên dev_unseen (234 câu)."],
  rows: [
    { code: "B0", description: "BM25 → reader", config_id: "B0_8037d4f8", frozen_on: "2026-11-08", status: null, reference_file: "eval_dev_x.json",
      splits: { dev_unseen: { "hit@5": 0.6453, "hit@20": 0.8376, em: 0.1709, f1: 0.3472 }, dev: { "hit@5": 0.6378, "hit@20": 0.8138, em: 0.2364, f1: 0.38 } },
      intervals: { dev_unseen: { em: [0.1239, 0.218] } }, queries: { dev_unseen: 234, dev: 1176 } },
    { code: "H0", description: "BM25 + dense → RRF → reader", config_id: "H0_d855091e", frozen_on: "2026-11-08", status: null, reference_file: "eval_dev_hybrid_x.json",
      splits: { dev_unseen: { "hit@5": 0.8889, "hit@20": 0.9829, em: 0.2094, f1: 0.4327 }, dev: { "hit@5": 0.8988, "hit@20": 0.9804, em: 0.3291, f1: 0.5022 } },
      intervals: { dev_unseen: { em: [0.1581, 0.2607] } }, queries: { dev_unseen: 234, dev: 1176 } },
    { code: "H1", description: "H0 + cross-encoder → reader", config_id: "H1_32d9bb5a", frozen_on: "2026-11-08", status: null, reference_file: "14f_H1_dev_unseen.json",
      splits: { dev_unseen: { "hit@5": 0.9615, "hit@20": 0.9829, em: 0.2051, f1: 0.4417 } },
      intervals: { dev_unseen: { em: [0.1538, 0.2564] } }, queries: { dev_unseen: 234 } },
  ],
});

/** Hai lần chạy hybrid cùng cấu hình: một đại diện (mã H0) và một bản trùng (lần chạy thử cũ). */
export const experimentsWithHybrid = (): ExperimentList => {
  const base = experiments();
  const h = (id: string, over = {}) => ({
    experiment_id: id, kind: "eval_dev_hybrid", created_at: "2026-11-07T13:54:23+00:00", split: "dev", retriever: "hybrid", checkpoint: "reader_x",
    corpus_version: "abc123", metrics: { dev: { em: 0.3291, "hit@5": 0.8988, dense_em: 0.3393 }, dev_unseen: { em: 0.2094, "hit@5": 0.8889 } },
    intervals: {}, caveats: ["Hybrid dùng encoder đã học GermanDPR train."], code: null, duplicate_of: null, ...over,
  });
  return { ...base, count: base.count + 2, items: [...base.items,
    h("eval_dev_hybrid_20261107T135423Z_598ddf6f", { code: "H0" }),
    h("eval_dev_hybrid_20261107T131435Z_598ddf6f", { duplicate_of: "eval_dev_hybrid_20261107T135423Z_598ddf6f" })] };
};

export const BERLIN_CTX = "Die Berliner Mauer wurde am 13. August 1961 errichtet und trennte West-Berlin von Ost-Berlin.";

/** Kết quả /read: reader trích "am 13. August 1961" (offset 25..43), so với đáp án mong đợi. */
export const readResult = (over: Partial<ReadResult> = {}): ReadResult => ({
  request_id: "rid-read",
  status: "ok",
  question: "Wann wurde die Berliner Mauer gebaut?",
  context: BERLIN_CTX,
  answer: "am 13. August 1961",
  answer_start: 25,
  answer_end: 43,
  answer_start_utf16: 25,
  answer_end_utf16: 43,
  reader_score: 5.27,
  expected_answer: "Am 13. August 1961",
  em: 1,
  f1: 1,
  timings: { queue_ms: 0.1, reader_ms: 61.5, total_ms: 62.3 },
  model: "reader_x",
  ...over,
});

/** Api giả; mỗi hàm thay được. Mặc định: ready, ask trả qa(), retrieve trả retrieved(). */
export function fakeApi(over: Partial<Api> = {}): Api {
  return {
    ask: async () => qa(),
    read: async () => readResult(),
    retrieve: async () => retrieved(),
    getReady: async () => readyBody(),
    getExperiments: async () => experiments(),
    getMatrix: async () => matrix(),
    ...over,
  };
}

export function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
