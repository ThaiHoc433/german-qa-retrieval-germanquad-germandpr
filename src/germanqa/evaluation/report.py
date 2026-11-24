"""Tổng hợp số đo cho báo cáo (GĐ7): một nơi tính chung cho dev và test, ba tầng, kèm khoảng tin cậy."""
import platform

from germanqa.evaluation.bootstrap import bootstrap_ci
from germanqa.evaluation.retrieval_metrics import evaluate_run, latency_summary
from germanqa.reader.qa_metrics import score_answers


def first_reference_only(examples):
    """Giữ tham chiếu đầu tiên của mỗi câu: cách chấm giống dev (dev mỗi câu có 1 tham chiếu)."""
    return [{**ex, "answers": ex["answers"][:1]} for ex in examples]


def layer_summary(predictions, examples, n_boot=1000, confidence=0.95, seed=42):
    """EM/F1 kèm khoảng tin cậy bootstrap, chấm theo tất cả tham chiếu và theo tham chiếu đầu tiên."""
    out = {}
    for name, exs in (("all_refs", examples), ("first_ref", first_reference_only(examples))):
        scores = score_answers(predictions, exs)
        em, f1 = [s[0] for s in scores], [s[1] for s in scores]
        m_em, lo_em, hi_em = bootstrap_ci(em, n_boot, confidence, seed)
        m_f1, lo_f1, hi_f1 = bootstrap_ci(f1, n_boot, confidence, seed)
        out[name] = {"queries": len(exs), "em": m_em, "em_ci": [lo_em, hi_em], "f1": m_f1, "f1_ci": [lo_f1, hi_f1]}
    return out


def retrieval_summary(rows, qrels, ks):
    """Hit/Recall/MRR@k của danh sách truy hồi lưu trong file dự đoán."""
    run = {r["question_id"]: [h["passage_id"] for h in r["retrieved"]] for r in rows}
    return evaluate_run(run, {q: qrels[q] for q in run}, ks)


def timing_summary(rows):
    return {name: latency_summary([r[name] for r in rows]) for name in ("retrieval_time", "reader_time", "total_time")}


def choose_span(candidates, k):
    """Giống QAPipeline.answer: span có điểm reader cao nhất trong k đoạn đầu; hòa điểm giữ đoạn hạng cao hơn."""
    valid = [c for c in candidates[:k] if c["start"] >= 0]
    return max(valid, key=lambda c: c["reader_score"]) if valid else None


def hardware_info():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def _ci(v, lo_hi, d=3):
    f = lambda x: f"{x:.{d}f}".replace(".", ",")
    return f"{f(v)} [{f(lo_hi[0])}; {f(lo_hi[1])}]"


def markdown_tables(split, summary):
    """Bảng Markdown để dán vào báo cáo. summary là dict do 07a_run_split.py dựng."""
    lines = [f"### Kết quả tập {split}", "", "| Tầng | Chấm theo | EM [KTC 95%] | F1 [KTC 95%] |", "|---|---|---|---|"]
    for layer, label in (("reader_gold_context", "Reader, đoạn đúng cho trước"), ("end_to_end", "End-to-end")):
        for ref, ref_label in (("all_refs", "tất cả tham chiếu"), ("first_ref", "tham chiếu đầu tiên")):
            r = summary[layer][ref]
            lines.append(f"| {label} | {ref_label} | {_ci(r['em'], r['em_ci'])} | {_ci(r['f1'], r['f1_ci'])} |")
    ret = summary["retrieval"]
    lines += ["", "| Retrieval | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@10 |", "|---|---|---|---|---|---|",
              "| BM25 | " + " | ".join(f"{ret[k]:.3f}".replace(".", ",") for k in ("hit@1", "hit@5", "hit@10", "hit@20", "mrr@10")) + " |"]
    t = summary["timing"]
    lines += ["", "| Thời gian (ms) | Trung bình | p50 | p95 |", "|---|---|---|---|"]
    for name, label in (("retrieval_time", "Truy hồi"), ("reader_time", "Reader (5 đoạn)"), ("total_time", "Tổng")):
        lines.append(f"| {label} | " + " | ".join(f"{t[name][k]:.1f}".replace(".", ",") for k in ("mean_ms", "p50_ms", "p95_ms")) + " |")
    return "\n".join(lines) + "\n"