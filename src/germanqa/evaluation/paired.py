"""Bổ sung cho bootstrap.py và report.py (GĐ7, thêm 2026-11-13).

Đặt ở module RIÊNG vì bootstrap.py bị khóa SHA-1 trong results/14_prereg.json và report.py trong results/15_freeze.json:
sửa hai tệp đó sẽ làm hỏng bằng chứng "mã không đổi sau khi ghi trước". Hai hàm ở đây chỉ GỌI lại hàm đã khóa."""
from germanqa.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from germanqa.evaluation.retrieval_metrics import hit_at_k, mrr_at_k, recall_at_k


def paired_bootstrap_by_id(scores_a, scores_b, n_boot=1000, confidence=0.95, seed=42):
    """Như paired_bootstrap nhưng nhận {question_id: điểm} và GHÉP THEO ID.
    paired_bootstrap ghép theo vị trí: nếu nơi gọi xếp hai dãy khác thứ tự, kết quả sai mà không báo lỗi.
    Ở đây hai bên phải có đúng cùng tập câu hỏi; câu được xếp theo ID nên kết quả tất định."""
    ka, kb = set(scores_a), set(scores_b)
    if ka != kb:
        only_a, only_b = sorted(ka - kb), sorted(kb - ka)
        raise ValueError(f"hai hệ thống không cùng tập câu hỏi: chỉ ở A {only_a[:5]} ({len(only_a)}), "
                         f"chỉ ở B {only_b[:5]} ({len(only_b)})")
    ids = sorted(ka)
    return paired_bootstrap([scores_a[q] for q in ids], [scores_b[q] for q in ids], n_boot, confidence, seed)


def retrieval_ci(rows, qrels, ks, n_boot=1000, confidence=0.95, seed=42):
    """Khoảng tin cậy bootstrap cho Hit/Recall/MRR@k. report.retrieval_summary chỉ trả trung bình.
    rows: dòng file dự đoán (question_id, retrieved). Trả về {"hit@5": [lo, hi], ...}; resample câu hỏi như bootstrap_ci."""
    run = {r["question_id"]: [h["passage_id"] for h in r["retrieved"]] for r in rows}
    qids = sorted(q for q in run if q in qrels)
    out = {}
    for k in ks:
        for name, fn in (("hit", hit_at_k), ("recall", recall_at_k), ("mrr", mrr_at_k)):
            _, lo, hi = bootstrap_ci([fn(run[q], set(qrels[q]), k) for q in qids], n_boot, confidence, seed)
            out[f"{name}@{k}"] = [lo, hi]
    return out
