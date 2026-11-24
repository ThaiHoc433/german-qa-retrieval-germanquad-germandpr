"""Pipeline hỏi đáp end-to-end (GĐ5): câu hỏi -> truy hồi top-20 -> reader trên top-5 -> một đáp án.

Pipeline chỉ biết retriever qua retrieve(question, top_k) -> [{passage_id, score, rank}] và reader qua
read(question, passages). Thay BM25 bằng dense/hybrid (GĐ12-13) không phải sửa file này."""
import time
from pathlib import Path

from germanqa.config import PROCESSED_DIR, ROOT
from germanqa.data.io_utils import file_sha1, read_jsonl
from germanqa.data.text_utils import normalize_question
from germanqa.experiment import load_config, new_experiment_id
from germanqa.schemas import QAResponse

MAX_QUESTION_TOKENS = 64     # câu hỏi dài hơn thì cắt: cửa sổ 384 token cần chừa chỗ cho context + stride


class QAPipeline:
    def __init__(self, retriever, reader, passages, model_version, corpus_version, experiment_id,
                 retrieval_top_k=20, reader_top_k=5, reranker=None):
        self.retriever, self.reader, self.passages = retriever, reader, passages
        self.reranker = reranker      # GĐ14: None = không xếp lại (giữ đúng hành vi cũ)    # passages: {passage_id: original_text}
        self.model_version, self.corpus_version, self.experiment_id = model_version, corpus_version, experiment_id
        self.retrieval_top_k, self.reader_top_k = retrieval_top_k, reader_top_k

    def _cut_question(self, question):
        """Cắt câu hỏi còn tối đa MAX_QUESTION_TOKENS token; trả về (câu hỏi, có bị cắt không)."""
        enc = self.reader.tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        if len(offs) <= MAX_QUESTION_TOKENS:
            return question, False
        return question[:offs[MAX_QUESTION_TOKENS - 1][1]], True
    @staticmethod
    def _no_trace(trace):
        """Vết rỗng nhưng ĐÚNG HÌNH DẠNG: người gọi bật trace=True luôn đọc được ['retrieved'] và ['candidates']."""
        return {"retrieved": [], "candidates": []} if trace else None

    def _empty(self, question, status, t0, t_retrieval=0.0, trace=None):
        return QAResponse(question=question, answer="", passage_id="", passage_text="", answer_start=0,
                          answer_end=0, retrieval_score=0.0, reader_score=0.0, retrieval_time=t_retrieval,
                          reader_time=0.0, total_time=time.perf_counter() - t0, model_version=self.model_version,
                          corpus_version=self.corpus_version, experiment_id=self.experiment_id,
                          status=status, trace=trace)

    def answer(self, question, retrieval_top_k=None, reader_top_k=None, trace=False):
        t0 = time.perf_counter()
        if not isinstance(question, str):
            raise TypeError("question phải là chuỗi")
        k_ret = self.retrieval_top_k if retrieval_top_k is None else retrieval_top_k
        k_read = self.reader_top_k if reader_top_k is None else reader_top_k
        if k_ret < 1 or k_read < 1:
            raise ValueError("retrieval_top_k và reader_top_k phải >= 1")
        # NFC + bỏ ký tự vô hình như corpus (GĐ2); trước đây chỉ strip nên câu hỏi NFD làm BM25 tách "höchste" -> "ho", "chste"
        question = normalize_question(question)
        if not question:
            return self._empty(question, "empty_question", t0, trace=self._no_trace(trace))
        t = time.perf_counter()
        hits = self.retriever.retrieve(question, k_ret)
        t_retrieval = time.perf_counter() - t
        if not hits:
            return self._empty(question, "no_passages", t0, t_retrieval, self._no_trace(trace))
        t_rerank, ranked = 0.0, hits
        if self.reranker is not None:                 # GĐ14: chấm lại đúng các đoạn đã truy hồi, reader nhận k_read đoạn đầu
            t = time.perf_counter()
            ranked = [{k: v for k, v in r.items() if k != "text"} for r in self.reranker.rerank(
                question, [{**h, "text": self.passages[h["passage_id"]]} for h in hits])]
            t_rerank = time.perf_counter() - t
        used = ranked[:k_read]
        reader_question, cut = self._cut_question(question)
        t = time.perf_counter()
        spans = self.reader.read(reader_question, [{"passage_id": h["passage_id"], "text": self.passages[h["passage_id"]]}
                                                  for h in used])
        t_reader = time.perf_counter() - t

        info = None
        if trace:
            info = {"retrieved": hits,
                    "candidates": [{"passage_id": h["passage_id"], "rank": h["rank"], "retrieval_score": h["score"],
                                    "answer": s["text"], "start": s["start"], "end": s["end"],
                                    "reader_score": s["score"],
                                    **({"rank_before_rerank": h["rank_before"], "rerank_score": h["rerank_score"]}
                                       if self.reranker is not None else {})} for h, s in zip(used, spans)]}
            if self.reranker is not None:
                info["reranked"] = [{"passage_id": r["passage_id"], "rank_before": r["rank_before"], "rank": r["rank"],
                                     "rerank_score": r["rerank_score"]} for r in ranked]
                info["rerank_stats"] = dict(self.reranker.last_stats)
        valid = [(s, h) for s, h in zip(spans, used) if s["start"] >= 0]
        if not valid:
            return self._empty(question, "no_valid_span", t0, t_retrieval, info)
        span, hit = max(valid, key=lambda sh: sh[0]["score"])       # hòa điểm: max giữ phần tử đứng trước = hạng cao hơn
        passage_text = self.passages[hit["passage_id"]]
        if passage_text[span["start"]:span["end"]] != span["text"]:     # bất biến của hệ thống; sai là lỗi lập trình
            raise ValueError("đáp án không khớp văn bản gốc tại offset")
        return QAResponse(question=question, answer=span["text"], passage_id=hit["passage_id"],
                          passage_text=passage_text, answer_start=span["start"], answer_end=span["end"],
                          retrieval_score=hit["score"], reader_score=span["score"],
                          retrieval_time=t_retrieval, reader_time=t_reader, total_time=time.perf_counter() - t0,
                          model_version=self.model_version, corpus_version=self.corpus_version,
                          experiment_id=self.experiment_id, status="question_truncated" if cut else "ok", trace=info,
                          rerank_time=t_rerank, rerank_score=hit.get("rerank_score"))


def build_retriever(kind, corpus, data_dir=PROCESSED_DIR, artifact_dir=None, bm25_store_dir=None):
    """kind: 'bm25' (cấu hình trong pipeline.yaml) hoặc 'dense' (configs/dense.yaml + artifacts đã mã hóa)."""
    if kind == "bm25":
        from germanqa.retrieval import bm25_store
        b = load_config("pipeline")["bm25"]
        # GĐ11.2: nạp chỉ mục đã lưu nếu dấu vân tay khớp, không thì dựng lại và lưu. Kết quả truy hồi giống hệt.
        retriever, _ = bm25_store.load_or_build(Path(data_dir) / "corpus.jsonl", corpus, b["k1"], b["b"],
                                                b["use_stopwords"], store_dir=bm25_store_dir)
        return retriever
    if kind == "dense":
        from germanqa.retrieval.dense import DenseRetriever, DPREncoder
        d = load_config("dense")["dense"]
        art = Path(artifact_dir or d["artifact_dir"])
        art = art if art.is_absolute() else ROOT / art
        return DenseRetriever.from_dir(art, DPREncoder(d["ctx_encoder"], d["question_encoder"], fp16=d["fp16"]),
                                       Path(data_dir) / "corpus.jsonl", d["question_max_length"], expected_cfg=d)
    if kind == "hybrid":
        from germanqa.retrieval.hybrid import HybridRetriever
        # BM25 và dense dựng MỘT lần rồi bọc lại; cấu hình hybrid lấy từ configs/hybrid.yaml
        return HybridRetriever.from_config(build_retriever("bm25", corpus, data_dir, bm25_store_dir=bm25_store_dir),
                                           build_retriever("dense", corpus, data_dir, artifact_dir),
                                           load_config("hybrid")["hybrid"])
    raise ValueError("retriever phải là 'bm25', 'dense' hoặc 'hybrid'")


def build_pipeline(checkpoint=None, data_dir=PROCESSED_DIR, retriever="bm25", artifact_dir=None, bm25_store_dir=None,
                   reranker=None):
    """Dựng pipeline từ configs/pipeline.yaml và configs/reader.yaml. checkpoint=None -> lấy trong pipeline.yaml.
    retriever='dense' thay BM25 bằng dense retriever; reader và mọi thứ khác giữ nguyên."""
    from germanqa.reader.inference import Reader          # torch chỉ nạp khi thật sự dựng reader
    cfg = load_config("pipeline")
    rcfg = load_config("reader")["reader"]
    ck = Path(checkpoint or cfg["pipeline"]["checkpoint"])
    ck = ck if ck.is_absolute() else ROOT / ck
    corpus = read_jsonl(Path(data_dir) / "corpus.jsonl")
    retriever_obj = build_retriever(retriever, corpus, data_dir, artifact_dir, bm25_store_dir)
    reader = Reader(ck, rcfg["max_length"], rcfg["stride"], rcfg["n_best"], cfg["reader"]["max_answer_tokens"],
                    cfg["reader"]["score_mode"], rcfg["eval_batch_size"])
    full_cfg = {**cfg, "checkpoint_name": ck.name, **({"retriever": retriever} if retriever != "bm25" else {})}
    if retriever == "hybrid":      # tham số hybrid phải nằm trong cấu hình được băm, nếu không hai rrf_c cho cùng đuôi mã
        full_cfg["hybrid"] = load_config("hybrid")["hybrid"]
    rr = None
    if reranker:                  # GĐ14: reranker=True lấy configs/rerank.yaml; hoặc truyền sẵn một đối tượng reranker
        from germanqa.rerank.factory import build_reranker
        rcfg = load_config("rerank")["rerank"]
        rr = reranker if hasattr(reranker, "rerank") else build_reranker(rcfg)
        full_cfg["rerank"] = {k: rcfg[k] for k in rcfg}      # đưa vào cấu hình được băm để experiment_id phân biệt H0 và H1
    return QAPipeline(retriever_obj, reader, {r["passage_id"]: r["original_text"] for r in corpus},
                      model_version=ck.name, corpus_version=file_sha1(Path(data_dir) / "corpus.jsonl")[:12],
                      experiment_id=new_experiment_id("pipeline", full_cfg),
                      retrieval_top_k=cfg["pipeline"]["retrieval_top_k"], reader_top_k=cfg["pipeline"]["reader_top_k"],
                      reranker=rr)
