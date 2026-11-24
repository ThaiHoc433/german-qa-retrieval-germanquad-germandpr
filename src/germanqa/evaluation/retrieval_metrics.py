"""Metric truy hồi (GĐ3): Hit@k, Recall@k, MRR@k và latency."""
import numpy as np


def hit_at_k(ranked, relevant, k):
    """1 nếu có ít nhất một đoạn đúng trong top-k, ngược lại 0."""
    return 1.0 if any(p in relevant for p in ranked[:k]) else 0.0


def recall_at_k(ranked, relevant, k):
    """Số đoạn đúng tìm được trong top-k / tổng số đoạn đúng đã biết."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def mrr_at_k(ranked, relevant, k):
    """1 / thứ hạng của đoạn đúng đầu tiên trong top-k; 0 nếu vắng."""
    for rank, p in enumerate(ranked[:k], start=1):
        if p in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_run(run, qrels, ks=(1, 5, 10, 20)):
    """run = {question_id: [passage_id đã xếp hạng]}, qrels = {question_id: [passage_id đúng]}.
    Trung bình trên các câu hỏi CÓ trong qrels."""
    qids = sorted(qrels)
    out = {"queries": len(qids)}
    for k in ks:
        rel = {q: set(qrels[q]) for q in qids}
        out[f"hit@{k}"] = float(np.mean([hit_at_k(run.get(q, []), rel[q], k) for q in qids]))
        out[f"recall@{k}"] = float(np.mean([recall_at_k(run.get(q, []), rel[q], k) for q in qids]))
        out[f"mrr@{k}"] = float(np.mean([mrr_at_k(run.get(q, []), rel[q], k) for q in qids]))
    return out


def latency_summary(seconds):
    """Trung bình, p50, p95 (mili-giây) của một dãy thời gian đo bằng giây."""
    a = np.asarray(seconds, dtype=float) * 1000.0
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95))}
