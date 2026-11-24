"""Phân tích điểm reranker đã lưu (GĐ14): đọc tệp điểm, kiểm toàn vẹn, dựng thứ hạng, so với H0 và theo nhóm 'đã thấy đoạn'."""
import json
import math

import numpy as np

from germanqa.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from germanqa.evaluation.run_cache import per_question
from germanqa.rerank.select import ranked_from_scores

OVERFLOW_FIELD = {"truncate": "first", "windows": "score"}


def load_scores(path, allowed_ids):
    """Chỉ trả về các câu thuộc allowed_ids (quy tắc truy cập: 14d chỉ thấy train-select). Báo lỗi nếu dòng trùng/NaN."""
    allowed, rows = set(allowed_ids), {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["question_id"] not in allowed:
                continue
            if r["question_id"] in rows:
                raise ValueError(f"{r['question_id']}: dòng trùng")
            rows[r["question_id"]] = r
    return rows


def integrity_problems(rows, pool, qids, pool_size=20):
    """Danh sách lỗi (rỗng = hợp lệ): thiếu câu, pool khác H0, độ dài sai, NaN/Inf, số cửa sổ < 1, điểm lớn nhất < điểm đầu."""
    bad = [f"thiếu {q}" for q in qids if q not in rows]
    for q in qids:
        r = rows.get(q)
        if r is None:
            continue
        if r["passage_ids"] != pool[q]:
            bad.append(f"{q}: pool khác H0")
        n = len(r["passage_ids"])
        if n != pool_size or not (len(r["score"]) == len(r["first"]) == len(r["windows"]) == n):
            bad.append(f"{q}: độ dài sai")
            continue
        for a, b in zip(r["score"], r["first"]):
            if not (math.isfinite(a) and math.isfinite(b)):
                bad.append(f"{q}: điểm không hữu hạn")
                break
            if a < b - 1e-9:
                bad.append(f"{q}: điểm lớn nhất < điểm cửa sổ đầu")
                break
        if min(r["windows"]) < 1:
            bad.append(f"{q}: số cửa sổ < 1")
    return bad


def reranked_run(rows, qids, overflow):
    f = OVERFLOW_FIELD[overflow]
    return {q: ranked_from_scores(rows[q]["passage_ids"], rows[q][f]) for q in qids}


def h0_run(pool, qids):
    return {q: list(pool[q]) for q in qids}


def metrics(run, qrels, qids):
    pq = per_question(run, qrels, qids, ks=(1, 5, 20))
    return pq, {k: float(np.mean(v)) for k, v in pq.items()}


def gain_ci(pq_new, pq_old, metric, idx=None, **boot):
    a, b = np.asarray(pq_new[metric], float), np.asarray(pq_old[metric], float)
    if idx is not None:
        a, b = a[idx], b[idx]
    return paired_bootstrap(a, b, **boot)


def did_ci(gain_a, gain_b, n_boot=1000, confidence=0.95, seed=42):
    """Chênh của chênh: mean(gain_a) - mean(gain_b) với hai nhóm câu hỏi KHÁC nhau (resample độc lập từng nhóm)."""
    a, b = np.asarray(gain_a, float), np.asarray(gain_b, float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    lo, hi = np.percentile(d, [50 * (1 - confidence), 100 - 50 * (1 - confidence)])
    return float(a.mean() - b.mean()), float(lo), float(hi)
