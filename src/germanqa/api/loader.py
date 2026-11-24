"""Nạp tài nguyên cho API (GĐ8, mở rộng GĐ18): reader MỘT lần, dùng chung cho mọi retriever.

Dựng bằng chính build_pipeline/build_retriever của CLI (GĐ5/GĐ6), nên API và CLI đi qua cùng một đường code và cùng
configs/*.yaml. Retriever nào không nạp được (thiếu faiss, thiếu artifacts, corpus đã đổi) chỉ bị đánh dấu không khả
dụng với một MÃ lý do; API vẫn chạy với các retriever còn lại.

GĐ18: hybrid KHÔNG dựng lại BM25 và dense mà dùng chung hai bộ đã nạp (không tốn thêm bộ nhớ). Reranker là tùy chọn
(settings.reranker): nạp MỘT lần rồi tạo thêm các pipeline phụ tên '<retriever>+rerank' dùng chung bộ truy hồi và reader,
nên không đụng QAPipeline. Reranker không nạp được thì chỉ rerank_unavailable, các retriever vẫn chạy."""
import logging

from germanqa.config import PROCESSED_DIR
from germanqa.data.io_utils import read_jsonl
from germanqa.experiment import load_config, new_experiment_id
from germanqa.pipeline import QAPipeline, build_pipeline, build_retriever

log = logging.getLogger("germanqa.api")
RERANK_SUFFIX = "+rerank"
COMPONENTS = ("bm25", "dense")
# Reranker chỉ đặt sau hybrid: đó đúng là H1 của GĐ14 (BM25 + dense -> RRF -> cross-encoder -> reader), cấu hình duy nhất
# đã đo. bm25/dense + rerank chưa từng được đo nên API không mở.
RERANK_RETRIEVERS = ("hybrid",)


def reason_code(exc):
    """Mã lý do ngắn, an toàn để trả cho client (không chứa đường dẫn hay nội dung ngoại lệ)."""
    if isinstance(exc, ImportError):
        return "faiss_not_installed" if "faiss" in str(exc).lower() else "dependency_missing"
    if isinstance(exc, FileNotFoundError):
        return "artifacts_missing"
    if isinstance(exc, ValueError) and "corpus đã đổi" in str(exc):
        return "corpus_changed"
    return "load_failed"


def build_reranker_from_settings(settings):
    """Dựng reranker theo settings.reranker và settings.reranker_models; lỗi được người gọi bắt và đổi thành mã lý do."""
    from germanqa.rerank.factory import build_reranker
    spec = settings.reranker_models[settings.reranker]
    if "config" in spec:                                   # dùng khối 'rerank' của configs/<tên>.yaml
        return build_reranker(load_config(spec["config"])["rerank"])
    return build_reranker({"checkpoint": spec["checkpoint"], "max_length": spec.get("max_length", 512), "batch_size": spec.get("batch_size", 16),
                           "fp16": spec.get("fp16", True), "relevant_label": spec.get("relevant_label"),
                           "overflow": spec.get("overflow", "windows"), "stride": spec.get("stride", 128)})


def load_pipelines(settings, data_dir=PROCESSED_DIR):
    """Trả về ({tên: QAPipeline}, {tên: mã lý do không khả dụng}). Tên là retriever, hoặc '<retriever>+rerank'."""
    names = list(settings.retrievers)
    want = [n for n in COMPONENTS if n in names or "hybrid" in names]      # hybrid cần cả hai thành phần dù không liệt kê
    pipelines, unavailable, retrievers, base, corpus = {}, {}, {}, None, None
    for name in want:
        try:
            if base is None:
                base = build_pipeline(settings.checkpoint, data_dir, retriever=name)
                retrievers[name] = base.retriever
                if name in names:
                    pipelines[name] = base
            else:
                corpus = corpus or read_jsonl(data_dir / "corpus.jsonl")
                retrievers[name] = build_retriever(name, corpus, data_dir)
                if name in names:
                    eid = new_experiment_id("pipeline", {"checkpoint_name": base.model_version, "corpus": base.corpus_version, "retriever": name})
                    pipelines[name] = QAPipeline(retrievers[name], base.reader, base.passages, base.model_version,
                                                 base.corpus_version, eid, base.retrieval_top_k, base.reader_top_k)
        except Exception as exc:                       # noqa: BLE001 - mỗi retriever lỗi độc lập
            log.warning("retriever %s không khả dụng: %s", name, type(exc).__name__)
            unavailable[name] = reason_code(exc)
    if "hybrid" in names:
        if all(c in retrievers for c in COMPONENTS):
            from germanqa.retrieval.hybrid import HybridRetriever
            cfg = load_config("hybrid")["hybrid"]
            hybrid = HybridRetriever.from_config(retrievers["bm25"], retrievers["dense"], cfg)
            eid = new_experiment_id("pipeline", {"checkpoint_name": base.model_version, "corpus": base.corpus_version, "retriever": "hybrid", "hybrid": cfg})
            pipelines["hybrid"] = QAPipeline(hybrid, base.reader, base.passages, base.model_version, base.corpus_version, eid,
                                             base.retrieval_top_k, base.reader_top_k)
        else:
            unavailable["hybrid"] = "dependency_unavailable"
    targets = [n for n in RERANK_RETRIEVERS if n in pipelines]
    if settings.reranker and pipelines and not targets:
        unavailable["rerank"] = "hybrid_unavailable"      # không nạp reranker vô ích (tốn VRAM) khi không có hybrid
    if settings.reranker and targets:
        try:
            rr = build_reranker_from_settings(settings)
            for name in targets:
                p = pipelines[name]
                eid = new_experiment_id("pipeline", {"checkpoint_name": p.model_version, "corpus": p.corpus_version, "retriever": name, "rerank": settings.reranker})
                pipelines[name + RERANK_SUFFIX] = QAPipeline(p.retriever, p.reader, p.passages, p.model_version, p.corpus_version, eid,
                                                             p.retrieval_top_k, p.reader_top_k, reranker=rr)
        except Exception as exc:                       # noqa: BLE001 - reranker hỏng không được làm sập các retriever
            log.warning("reranker %s không khả dụng: %s", settings.reranker, type(exc).__name__)
            unavailable["rerank"] = reason_code(exc)
    if not pipelines:
        raise RuntimeError("không nạp được retriever nào: " + ", ".join(f"{k}={v}" for k, v in unavailable.items()))
    return pipelines, unavailable
