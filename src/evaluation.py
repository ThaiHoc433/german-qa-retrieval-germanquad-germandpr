"""Evaluation — module 3 của docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md (V1.0).

Chỉ tính metric từ dữ liệu đã lưu (gold + predictions/retrieval results), không chạy model.
Ba chế độ:
  - end_to_end    : PredictionRecord → retrieval + QA + latency + error decomposition + manual review.
  - reader_only   : PredictionRecord của reader đọc thẳng gold passage → QA + latency. Nhãn là
                    "mốc năng lực reader trên gold passage", KHÔNG phải trần tuyệt đối.
  - retrieval_only: RetrievalResult rows (vd output BM25) → retrieval metrics, không cần reader.

CLI (chạy từ thư mục gốc repo):
  python -m src.evaluation --gold G.jsonl --predictions P.jsonl --out M.json [--mode reader_only]
  python -m src.evaluation --gold G.jsonl --retrieval-results R.jsonl --out M.json
"""
import argparse
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from src.schemas import (
    SchemaError,
    validate_prediction_record,
    validate_qa_record,
    validate_ranked_list,
    validate_retrieval_result,
)

SPEC_REF = "module-specs V1.0 (docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md, mục 3)"
K_VALUES = (1, 5, 10, 20)
MRR_CUTOFF = 10
RETRIEVAL_K = 20
READER_K = 5
MANUAL_REVIEW_SAMPLE_SIZE = 100
SEED = 42
BUCKETS = ("execution_failure", "retrieval_miss", "reranking_failure", "reader_failure", "correct")
LATENCY_STAGES = ("retrieval", "rerank", "read", "total")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------- Retrieval metrics (3.3) ----------

def _require_gold(gold: list[str]) -> None:
    if not gold:
        raise ValueError("gold rỗng — query không có gold_evidences phải bị loại khỏi eval trước khi tính metric")


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Recall@k chính thức (spec 3.3): 1{top-k ∩ gold ≠ ∅} — về công thức chính là Hit@k."""
    _require_gold(gold)
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def gold_coverage_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Metric PHỤ |top-k ∩ gold| / |gold| — không được gọi là Recall@k."""
    _require_gold(gold)
    gold_set = set(gold)
    return len(set(retrieved[:k]) & gold_set) / len(gold_set)


def first_gold_rank(retrieved: list[str], gold: list[str]) -> int | None:
    gold_set = set(gold)
    for rank, pid in enumerate(retrieved, start=1):
        if pid in gold_set:
            return rank
    return None


def reciprocal_rank(retrieved: list[str], gold: list[str], cutoff: int) -> float:
    """1/rank của gold passage đầu tiên xuất hiện trong top-`cutoff`; 0 nếu không có."""
    rank = first_gold_rank(retrieved[:cutoff], gold)
    return 1.0 / rank if rank else 0.0


def retrieval_metrics(
    eval_gold: dict[str, list[str]],
    retrieved_by_query: dict[str, list[str]],
    k_values: tuple[int, ...] = K_VALUES,
    mrr_cutoff: int = MRR_CUTOFF,
) -> dict:
    """Trung bình trên toàn bộ query của `eval_gold` (mọi gold khác rỗng); query không có
    kết quả retrieval tính như danh sách rỗng (đóng góp 0), không bị âm thầm bỏ khỏi mẫu số."""
    ids = sorted(eval_gold)
    lists = {q: retrieved_by_query.get(q, []) for q in ids}
    out = {f"recall@{k}": _mean([recall_at_k(lists[q], eval_gold[q], k) for q in ids]) for k in k_values}
    out[f"mrr@{mrr_cutoff}"] = _mean([reciprocal_rank(lists[q], eval_gold[q], mrr_cutoff) for q in ids])
    out.update({
        f"gold_coverage@{k}": _mean([gold_coverage_at_k(lists[q], eval_gold[q], k) for q in ids])
        for k in k_values
    })
    return out


def evaluate_retrieval(
    gold_by_query: dict[str, list[str]],
    retrieved_by_query: dict[str, list[str]],
    k_values: tuple[int, ...] = K_VALUES,
    mrr_cutoff: int = MRR_CUTOFF,
) -> dict:
    eval_gold = {q: g for q, g in gold_by_query.items() if g}
    return {
        "n_queries": len(eval_gold),
        "n_excluded_empty_gold": len(gold_by_query) - len(eval_gold),
        **retrieval_metrics(eval_gold, retrieved_by_query, k_values, mrr_cutoff),
    }


# ---------- QA metrics (3.3) ----------

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_answer(text: str) -> str:
    """NFC → lowercase → loại (xóa) ký tự dấu câu `[^\\w\\s]` → gộp khoảng trắng.
    `\\w` của `re` trên str là Unicode-aware nên ä/ö/ü/ß được giữ nguyên."""
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(_PUNCT_RE.sub("", text).split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(ground_truth).split()
    if not pred_toks or not gold_toks:
        return 1.0 if pred_toks == gold_toks else 0.0
    num_same = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    return max((metric_fn(prediction, gt) for gt in ground_truths), default=0.0)


# ---------- Latency (3.3) ----------

def percentile(values: list[float], q: float) -> float:
    """Nội suy tuyến tính giữa 2 rank gần nhất (như numpy method='linear'), q ∈ [0, 100]."""
    s = sorted(values)
    if not s:
        raise ValueError("percentile của danh sách rỗng")
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize_latency(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    s = sorted(values)  # cộng theo thứ tự đã sort → kết quả không phụ thuộc thứ tự file
    return {"n": len(s), "mean": sum(s) / len(s), "p50": percentile(s, 50), "p95": percentile(s, 95)}


# ---------- Load + validate ----------

def build_gold_index(gold_records: list[dict]) -> dict[str, dict]:
    errors, index = [], {}
    for i, rec in enumerate(gold_records):
        rec_errors = validate_qa_record(rec, f"gold[{i}]")
        if rec_errors:
            errors += rec_errors
            continue
        qid = rec["query_id"]
        if qid in index:
            errors.append(f"gold[{i}]: query_id trùng {qid!r}")
            continue
        index[qid] = {
            # gold_passage_ids được SUY RA từ gold_evidences (spec 1.2), không lưu riêng
            "gold_ids": sorted({ev["passage_id"] for ev in rec["gold_evidences"]}),
            "reference_answers": list(rec["reference_answers"]),
        }
    if errors:
        raise SchemaError(errors)
    return index


def _index_predictions(prediction_records: list[dict], gold_index: dict, expected_corpus_id: str | None) -> dict:
    errors, by_query = [], {}
    for i, rec in enumerate(prediction_records):
        rec_errors = validate_prediction_record(rec, f"prediction[{i}]")
        if rec_errors:
            errors += rec_errors
            continue
        qid = rec["query_id"]
        if qid in by_query:
            errors.append(f"prediction[{i}]: query_id trùng {qid!r}")
        elif qid not in gold_index:
            errors.append(f"prediction[{i}]: query_id {qid!r} không có trong gold — sai file gold?")
        else:
            by_query[qid] = rec
    if not errors:
        for field in ("run_id", "config_id", "corpus_id"):
            values = sorted({r[field] for r in by_query.values()})
            if len(values) > 1:
                errors.append(f"predictions trộn nhiều {field} {values} — mỗi file predictions phải thuộc đúng 1 run")
        if expected_corpus_id is not None:
            bad = sorted(q for q, r in by_query.items() if r["corpus_id"] != expected_corpus_id)
            if bad:
                errors.append(
                    f"{len(bad)} prediction có corpus_id khác corpus đang đánh giá {expected_corpus_id!r} "
                    f"(vd {bad[:3]}) — không tính Recall@k trên corpus sai"
                )
    if errors:
        raise SchemaError(errors)
    return by_query


def _run_identity(by_query: dict) -> dict:
    first = by_query[min(by_query)] if by_query else None
    return {f: (first[f] if first else None) for f in ("run_id", "config_id", "corpus_id")}


# ---------- End-to-end: error decomposition (3.3) ----------

def _stage_sets(rec: dict, retrieval_k: int, reader_k: int) -> tuple[list[str], list[str]]:
    """(retrieval_pool, reader_input).

    Bật reranker: pool = top-`retrieval_k` của retriever, reader_input = reranked.
    Tắt reranker: retriever là giai đoạn chọn duy nhất trước reader, nên pool = reader_input
    = top-`reader_k` (spec 8.3) — gold ở rank (reader_k, retrieval_k] là retrieval_miss,
    không phải reranking_failure vì không có reranker nào để quy lỗi.
    """
    retrieved = [r["passage_id"] for r in rec["retrieved"]]
    if rec["reranked"] is not None:
        return retrieved[:retrieval_k], [r["passage_id"] for r in rec["reranked"]]
    reader_input = retrieved[:reader_k]
    return reader_input, reader_input


def _check_pipeline_consistency(by_query: dict, retrieval_k: int, reader_k: int) -> list[str]:
    errors = []
    for qid in sorted(by_query):
        rec = by_query[qid]
        if rec["status"] != "ok":
            continue
        pool, reader_input = _stage_sets(rec, retrieval_k, reader_k)
        if rec["reranked"] is not None:
            outside = sorted(set(reader_input) - set(pool))
            if outside:
                errors.append(f"prediction[{qid}]: reranked chứa passage ngoài top-{retrieval_k} retriever: {outside[:3]}")
            expected = min(reader_k, len(pool))
            if len(reader_input) != expected:
                errors.append(
                    f"prediction[{qid}]: reranked có {len(reader_input)} passage, phải đúng "
                    f"min(reader_k={reader_k}, {len(pool)}) = {expected}"
                )
        pid = rec["predicted_passage_id"]
        if pid is not None and pid not in reader_input:
            errors.append(
                f"prediction[{qid}]: predicted_passage_id {pid!r} không nằm trong {len(reader_input)} "
                f"passage reader được đọc (reader_k={reader_k})"
            )
    return errors


def _score_answer(rec: dict | None, reference_answers: list[str]) -> tuple[float, float, bool]:
    if rec is None or rec["status"] != "ok" or rec["predicted_answer_text"] is None:
        return 0.0, 0.0, False
    text = rec["predicted_answer_text"]
    return (
        max_over_ground_truths(exact_match_score, text, reference_answers),
        max_over_ground_truths(f1_score, text, reference_answers),
        True,
    )


def classify_query(
    qid: str, rec: dict | None, gold_ids: list[str], reference_answers: list[str], retrieval_k: int, reader_k: int
) -> dict:
    """Gán đúng 1 bucket tầng 1 theo thứ tự execution_failure → retrieval_miss →
    reranking_failure → reader_failure → correct. `final_correct` lưu riêng, KHÔNG suy
    từ bucket: hệ thống có thể mất gold nhưng vẫn trả lời đúng từ passage khác."""
    em, f1, has_pred = _score_answer(rec, reference_answers)
    status = "missing" if rec is None else rec["status"]
    gold_in_topk = gold_in_topk_prime = None
    rank = None if rec is None else first_gold_rank([r["passage_id"] for r in rec["retrieved"]], gold_ids)

    if status != "ok":
        bucket = "execution_failure"
    else:
        pool, reader_input = _stage_sets(rec, retrieval_k, reader_k)
        gold = set(gold_ids)
        gold_in_topk = bool(gold & set(pool))
        gold_in_topk_prime = bool(gold & set(reader_input))
        if not gold_in_topk:
            bucket = "retrieval_miss"
        elif not gold_in_topk_prime:
            bucket = "reranking_failure"
        elif em < 1.0:
            bucket = "reader_failure"
        else:
            bucket = "correct"

    return {
        "query_id": qid,
        "bucket": bucket,
        "status": status,
        "gold_in_topk": gold_in_topk,
        "gold_in_topk_prime": gold_in_topk_prime,
        "first_gold_rank": rank,
        "has_prediction": has_pred,
        "em": em,
        "f1": f1,
        "final_correct": em == 1.0,
        "manual_review_suggested": f1 > 0.0 and em == 0.0,
    }


def _manual_review(rows: list[dict], sample_size: int, seed: int) -> dict:
    flagged = sorted(r["query_id"] for r in rows if r["manual_review_suggested"])
    sample = sorted(random.Random(seed).sample(flagged, min(sample_size, len(flagged))))
    return {
        "rule": "F1 > 0 và EM = 0 — cờ cắt ngang tầng 1, không phải bucket thứ 6, không đổi % tầng 1",
        "n_flagged": len(flagged),
        "seed": seed,
        "sample_query_ids": sample,
    }


def _latency_block(by_query: dict, eval_ids: list[str]) -> dict:
    """Chỉ record status='ok' thuộc tập eval — cùng tập query với các metric khác.
    Warm-up và điều kiện đo (thiết bị, batch=1, GPU sync) là việc của bên sinh
    PredictionRecord (Integration 8.3); ở đây chỉ tóm tắt giá trị đã đo."""
    ok = [by_query[q] for q in eval_ids if q in by_query and by_query[q]["status"] == "ok"]
    return {
        stage: summarize_latency([r["latency_ms"][stage] for r in ok if r["latency_ms"][stage] is not None])
        for stage in LATENCY_STAGES
    }


def _qa_block(rows: list[dict], label: str) -> dict:
    return {
        "label": label,
        "em": _mean([r["em"] for r in rows]),
        "f1": _mean([r["f1"] for r in rows]),
        "no_prediction": sum(not r["has_prediction"] for r in rows),
    }


def evaluate_end_to_end(
    gold_records: list[dict],
    prediction_records: list[dict],
    *,
    retrieval_k: int = RETRIEVAL_K,
    reader_k: int = READER_K,
    k_values: tuple[int, ...] = K_VALUES,
    mrr_cutoff: int = MRR_CUTOFF,
    expected_corpus_id: str | None = None,
    manual_review_sample_size: int = MANUAL_REVIEW_SAMPLE_SIZE,
    seed: int = SEED,
) -> dict:
    gold_index = build_gold_index(gold_records)
    by_query = _index_predictions(prediction_records, gold_index, expected_corpus_id)
    consistency_errors = _check_pipeline_consistency(by_query, retrieval_k, reader_k)
    if consistency_errors:
        raise SchemaError(consistency_errors)

    eval_ids = sorted(q for q, g in gold_index.items() if g["gold_ids"])
    rows = [
        classify_query(
            q, by_query.get(q), gold_index[q]["gold_ids"], gold_index[q]["reference_answers"], retrieval_k, reader_k
        )
        for q in eval_ids
    ]
    n = len(rows)
    counts = {b: sum(r["bucket"] == b for r in rows) for b in BUCKETS}
    retrieved = {q: [r["passage_id"] for r in by_query[q]["retrieved"]] for q in eval_ids if q in by_query}

    return {
        "spec": SPEC_REF,
        "mode": "end_to_end",
        **_run_identity(by_query),
        "config": {
            "k_values": list(k_values),
            "mrr_cutoff": mrr_cutoff,
            "retrieval_k": retrieval_k,
            "reader_k": reader_k,
            "manual_review_sample_size": manual_review_sample_size,
            "seed": seed,
        },
        "counts": {
            "n_gold_records": len(gold_index),
            "n_excluded_empty_gold": len(gold_index) - n,
            "n_eval_queries": n,
            "n_predictions": len(by_query),
            "n_missing_predictions": sum(r["status"] == "missing" for r in rows),
            "n_status_error": sum(r["status"] == "error" for r in rows),
        },
        "retrieval": retrieval_metrics({q: gold_index[q]["gold_ids"] for q in eval_ids}, retrieved, k_values, mrr_cutoff),
        "qa": _qa_block(rows, "end_to_end"),
        "latency_ms": _latency_block(by_query, eval_ids),
        "error_decomposition": {
            "counts": counts,
            "rates": {b: (counts[b] / n if n else 0.0) for b in BUCKETS},
            "counts_sum_equals_n": sum(counts.values()) == n,
            "final_correct_by_bucket": {b: sum(r["bucket"] == b and r["final_correct"] for r in rows) for b in BUCKETS},
        },
        "manual_review": _manual_review(rows, manual_review_sample_size, seed),
        "per_query": rows,
    }


# ---------- Reader-only (oracle) ----------

def evaluate_reader_only(
    gold_records: list[dict],
    prediction_records: list[dict],
    *,
    expected_corpus_id: str | None = None,
    manual_review_sample_size: int = MANUAL_REVIEW_SAMPLE_SIZE,
    seed: int = SEED,
) -> dict:
    gold_index = build_gold_index(gold_records)
    by_query = _index_predictions(prediction_records, gold_index, expected_corpus_id)
    errors = [
        f"prediction[{q}]: chế độ reader_only nhưng predicted_passage_id {r['predicted_passage_id']!r} không phải gold passage"
        for q, r in sorted(by_query.items())
        if r["predicted_passage_id"] is not None and r["predicted_passage_id"] not in gold_index[q]["gold_ids"]
    ]
    if errors:
        raise SchemaError(errors)

    eval_ids = sorted(q for q, g in gold_index.items() if g["gold_ids"])
    rows = []
    for q in eval_ids:
        rec = by_query.get(q)
        em, f1, has_pred = _score_answer(rec, gold_index[q]["reference_answers"])
        rows.append({
            "query_id": q,
            "status": "missing" if rec is None else rec["status"],
            "has_prediction": has_pred,
            "em": em,
            "f1": f1,
            "final_correct": em == 1.0,
            "manual_review_suggested": f1 > 0.0 and em == 0.0,
        })
    return {
        "spec": SPEC_REF,
        "mode": "reader_only",
        **_run_identity(by_query),
        "config": {"manual_review_sample_size": manual_review_sample_size, "seed": seed},
        "counts": {
            "n_gold_records": len(gold_index),
            "n_excluded_empty_gold": len(gold_index) - len(rows),
            "n_eval_queries": len(rows),
            "n_predictions": len(by_query),
            "n_missing_predictions": sum(r["status"] == "missing" for r in rows),
            "n_status_error": sum(r["status"] == "error" for r in rows),
        },
        "qa": _qa_block(rows, "reader_capability_on_gold_passage"),
        "latency_ms": _latency_block(by_query, eval_ids),
        "manual_review": _manual_review(rows, manual_review_sample_size, seed),
        "per_query": rows,
    }


# ---------- Retrieval-only (RetrievalResult rows) ----------

def evaluate_retrieval_results(
    gold_records: list[dict],
    retrieval_rows: list[dict],
    *,
    k_values: tuple[int, ...] = K_VALUES,
    mrr_cutoff: int = MRR_CUTOFF,
    declared_corpus_id: str | None = None,
) -> dict:
    gold_index = build_gold_index(gold_records)
    errors, grouped = [], defaultdict(list)
    for i, row in enumerate(retrieval_rows):
        row_errors = validate_retrieval_result(row, f"retrieval[{i}]")
        if row_errors:
            errors += row_errors
        elif row["query_id"] not in gold_index:
            errors.append(f"retrieval[{i}]: query_id {row['query_id']!r} không có trong gold — sai file gold?")
        else:
            grouped[row["query_id"]].append(row)
    ranked = {q: sorted(rows, key=lambda r: r["rank"]) for q, rows in grouped.items()}
    for q in sorted(ranked):
        errors += validate_ranked_list(ranked[q], q, f"retrieval[{q}]")
    if errors:
        raise SchemaError(errors)

    eval_gold = {q: g["gold_ids"] for q, g in gold_index.items() if g["gold_ids"]}
    retrieved = {q: [r["passage_id"] for r in rows] for q, rows in ranked.items()}
    return {
        "spec": SPEC_REF,
        "mode": "retrieval_only",
        "retriever_names": sorted({r["retriever_name"] for rows in ranked.values() for r in rows}),
        "corpus_id": declared_corpus_id,
        "config": {"k_values": list(k_values), "mrr_cutoff": mrr_cutoff},
        "counts": {
            "n_gold_records": len(gold_index),
            "n_excluded_empty_gold": len(gold_index) - len(eval_gold),
            "n_eval_queries": len(eval_gold),
            "n_queries_without_results": sum(q not in retrieved for q in eval_gold),
        },
        "retrieval": retrieval_metrics(eval_gold, retrieved, k_values, mrr_cutoff),
        "per_query": [
            {"query_id": q, "first_gold_rank": first_gold_rank(retrieved.get(q, []), eval_gold[q])}
            for q in sorted(eval_gold)
        ],
    }


# ---------- CLI ----------

def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SchemaError([f"{path}:{lineno}: JSON không hợp lệ ({e.msg})"]) from e
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.evaluation", description="Evaluation V1.0 (spec mục 3).")
    p.add_argument("--gold", required=True, help="QARecord JSONL, vd data/processed/qa_germanquad_test.jsonl")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--predictions", help="PredictionRecord JSONL (end_to_end hoặc reader_only)")
    src.add_argument("--retrieval-results", help="RetrievalResult JSONL (retrieval_only, không cần reader)")
    p.add_argument("--out", required=True, help="đường dẫn metrics JSON xuất ra")
    p.add_argument("--mode", choices=("end_to_end", "reader_only"), help="chỉ dùng với --predictions (mặc định end_to_end)")
    p.add_argument("--corpus-id", choices=("base", "mid", "large"), help="corpus đang đánh giá; predictions khác corpus → lỗi")
    p.add_argument("--retrieval-k", type=int, default=RETRIEVAL_K)
    p.add_argument("--reader-k", type=int, default=READER_K)
    p.add_argument("--manual-review-sample-size", type=int, default=MANUAL_REVIEW_SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def _summary_line(metrics: dict) -> str:
    parts = [f"mode={metrics['mode']}", f"N={metrics['counts']['n_eval_queries']}"]
    if "retrieval" in metrics:
        r = metrics["retrieval"]
        cutoff = metrics["config"]["mrr_cutoff"]
        parts += [f"R@{k}={r['recall@' + str(k)]:.4f}" for k in metrics["config"]["k_values"]]
        parts.append(f"MRR@{cutoff}={r['mrr@' + str(cutoff)]:.4f}")
    if "qa" in metrics:
        parts += [f"EM={metrics['qa']['em']:.4f}", f"F1={metrics['qa']['f1']:.4f}"]
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.retrieval_results and args.mode:
        parser.error("--mode chỉ dùng với --predictions")

    try:
        gold = read_jsonl(args.gold)
        if args.retrieval_results:
            inputs = {"gold": args.gold, "retrieval_results": args.retrieval_results}
            metrics = evaluate_retrieval_results(gold, read_jsonl(args.retrieval_results), declared_corpus_id=args.corpus_id)
        elif args.mode == "reader_only":
            inputs = {"gold": args.gold, "predictions": args.predictions}
            metrics = evaluate_reader_only(
                gold, read_jsonl(args.predictions), expected_corpus_id=args.corpus_id,
                manual_review_sample_size=args.manual_review_sample_size, seed=args.seed,
            )
        else:
            inputs = {"gold": args.gold, "predictions": args.predictions}
            metrics = evaluate_end_to_end(
                gold, read_jsonl(args.predictions), retrieval_k=args.retrieval_k, reader_k=args.reader_k,
                expected_corpus_id=args.corpus_id, manual_review_sample_size=args.manual_review_sample_size,
                seed=args.seed,
            )
    except SchemaError as e:
        print(f"[evaluation] DỪNG — dữ liệu không đạt schema V1.0:\n{e}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"inputs": inputs, **metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[evaluation] {_summary_line(metrics)} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
