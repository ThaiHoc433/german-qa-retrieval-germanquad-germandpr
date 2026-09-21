import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    evaluate_end_to_end,
    evaluate_reader_only,
    evaluate_retrieval,
    evaluate_retrieval_results,
    exact_match_score,
    f1_score,
    gold_coverage_at_k,
    main,
    max_over_ground_truths,
    percentile,
    recall_at_k,
    reciprocal_rank,
    summarize_latency,
)
from src.schemas import SchemaError

REF = "Albert Einstein"


# ---------- builders ----------

def qa(qid, gold_pids, refs=(REF,)):
    refs = list(refs)
    return {
        "query_id": qid,
        "question": f"Frage {qid}?",
        "gold_evidences": [
            {"passage_id": p, "answer_text": refs[0], "answer_start": 0, "answer_end": len(refs[0])}
            for p in gold_pids
        ],
        "reference_answers": refs,
        "source_dataset": "germanquad",
        "source_split": "test",
    }


def ranked(qid, pids, name="bm25"):
    return [
        {"query_id": qid, "passage_id": p, "rank": i + 1, "score": float(len(pids) - i), "retriever_name": name}
        for i, p in enumerate(pids)
    ]


def pred(qid, retrieved, reranked=None, answer=None, answer_pid=None, status="ok", error_stage=None,
         run_id="run-1", corpus_id="base", config_id="bm25_base_rerank-on"):
    has_answer = answer is not None
    if has_answer and answer_pid is None:
        answer_pid = (reranked or retrieved)[0]
    return {
        "run_id": run_id,
        "config_id": config_id,
        "corpus_id": corpus_id,
        "retrieval_k": 20,
        "reader_k": 5,
        "query_id": qid,
        "status": status,
        "error_stage": error_stage,
        "retrieved": ranked(qid, retrieved),
        "reranked": None if reranked is None else ranked(qid, reranked, "cross_encoder"),
        "predicted_answer_text": answer,
        "predicted_passage_id": answer_pid if has_answer else None,
        "predicted_answer_start": 0 if has_answer else None,
        "predicted_answer_end": len(answer) if has_answer else None,
        "reader_span_prob": 0.9 if has_answer else None,
        "latency_ms": {"retrieval": 1.0, "rerank": 2.0 if reranked is not None else None, "read": 3.0, "total": 6.0},
    }


def fill(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def smoke_dataset():
    """20 query đánh giá, đúng 4 query/bucket, reranker bật (retrieval_k=20, reader_k=5),
    + 1 query gold rỗng phải bị loại khỏi eval dù có prediction."""
    gold, preds = [], []

    def add(qid, gold_pids, p=None):
        gold.append(qa(qid, gold_pids))
        if p is not None:
            preds.append(p)

    for i in range(2):  # execution_failure: thiếu prediction
        add(f"q-exec-missing-{i}", [f"q-exec-missing-{i}-g"])
    for i in range(2):  # execution_failure: lỗi ở giai đoạn read, gold vẫn ở hạng 1 retriever
        q = f"q-exec-error-{i}"
        retrieved = [f"{q}-g"] + fill(f"{q}-x", 19)
        add(q, [f"{q}-g"], pred(q, retrieved, retrieved[:5], status="error", error_stage="read"))
    for i in range(4):  # retrieval_miss: gold hạng 25 (ngoài top-20); câu 0 vẫn trả lời đúng từ passage khác
        q = f"q-retrieval-{i}"
        retrieved = fill(f"{q}-x", 24) + [f"{q}-g"]
        add(q, [f"{q}-g"], pred(q, retrieved, retrieved[:5], answer=REF if i == 0 else "Marie Curie"))
    for i in range(4):  # reranking_failure: gold hạng 8 (trong top-20) nhưng reranker loại khỏi top-5
        q = f"q-rerank-{i}"
        retrieved = fill(f"{q}-a", 7) + [f"{q}-g"] + fill(f"{q}-b", 12)
        add(q, [f"{q}-g"], pred(q, retrieved, retrieved[:5], answer="Marie Curie"))
    for i, ans in enumerate(["Einstein", "Albert", "Marie Curie", None]):  # reader_failure
        q = f"q-reader-{i}"
        retrieved = [f"{q}-g"] + fill(f"{q}-x", 19)
        add(q, [f"{q}-g"], pred(q, retrieved, retrieved[:5], answer=ans))
    for i, ans in enumerate(["Albert Einstein", "albert einstein", "Albert Einstein.", "ALBERT EINSTEIN"]):  # correct
        q = f"q-correct-{i}"
        retrieved = [f"{q}-g"] + fill(f"{q}-x", 19)
        add(q, [f"{q}-g"], pred(q, retrieved, retrieved[:5], answer=ans))
    retrieved = fill("q-nogold-x", 20)
    add("q-nogold", [], pred("q-nogold", retrieved, retrieved[:5], answer="Marie Curie"))
    return gold, preds


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(path)


# ---------- Retrieval: ví dụ tính tay ----------

def test_recall_and_mrr_gold_at_rank_2_spec_example():
    retrieved = ["p1", "gold"] + fill("p", 8)
    assert recall_at_k(retrieved, ["gold"], 1) == 0.0
    assert recall_at_k(retrieved, ["gold"], 5) == 1.0
    assert reciprocal_rank(retrieved, ["gold"], 10) == 0.5


def test_recall_is_hit_indicator_not_fraction_when_multiple_gold():
    retrieved = ["g1", "x", "g2", "y"]
    assert recall_at_k(retrieved, ["g1", "g2"], 2) == 1.0  # tìm được ≥1 gold chunk là thành công
    assert gold_coverage_at_k(retrieved, ["g1", "g2"], 2) == 0.5
    assert gold_coverage_at_k(retrieved, ["g1", "g2"], 4) == 1.0


def test_mrr_uses_first_gold_inside_cutoff_only():
    retrieved = fill("x", 10) + ["gold"]  # gold ở hạng 11
    assert reciprocal_rank(retrieved, ["gold"], 10) == 0.0
    assert recall_at_k(retrieved, ["gold"], 20) == 1.0
    assert reciprocal_rank(["x", "g2", "g1"], ["g1", "g2"], 10) == 0.5


def test_aggregate_recall_uses_hit_for_multi_gold_queries():
    # query có 2 gold chunk (overlap) — bug cũ nằm ở đúng tầng aggregate này, không phải ở hàm lẻ
    m = evaluate_retrieval({"q1": ["g1", "g2"], "q2": ["h"]}, {"q1": ["g1", "x"], "q2": ["x", "h"]},
                           k_values=(1,), mrr_cutoff=10)
    assert m["recall@1"] == pytest.approx((1.0 + 0.0) / 2)
    assert m["gold_coverage@1"] == pytest.approx((0.5 + 0.0) / 2)


def test_empty_gold_is_rejected_at_metric_level():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 1)


def test_evaluate_retrieval_hand_computed():
    gold = {"q1": ["a"], "q2": ["b"], "q3": ["c"], "q4": []}
    retrieved = {"q1": ["a", "x"], "q2": ["x", "y", "b"]}  # q3 không có kết quả → đóng góp 0, không bị bỏ
    m = evaluate_retrieval(gold, retrieved, k_values=(1, 5), mrr_cutoff=10)
    assert (m["n_queries"], m["n_excluded_empty_gold"]) == (3, 1)
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@5"] == pytest.approx(2 / 3)
    assert m["mrr@10"] == pytest.approx((1 + 1 / 3 + 0) / 3)


# ---------- QA: ví dụ tính tay + normalization tiếng Đức ----------

def test_em_f1_einstein_spec_example():
    assert exact_match_score("Einstein", "Albert Einstein") == 0.0
    assert f1_score("Einstein", "Albert Einstein") == pytest.approx(2 / 3)  # P=1/1, R=1/2


def test_german_normalization():
    assert exact_match_score("Albert Einstein.", "albert einstein") == 1.0
    assert exact_match_score("München", "münchen") == 1.0
    assert exact_match_score("Straße", "strasse") == 0.0  # ß giữ nguyên, không casefold thành ss
    assert exact_match_score("München", "München") == 1.0  # NFD vs NFC
    assert exact_match_score("  über   den Wolken ", "Über den Wolken") == 1.0
    assert exact_match_score("75 %", "75") == 1.0
    assert exact_match_score("1.000", "1000") == 1.0  # dấu câu bị XÓA, không thay bằng khoảng trắng


def test_f1_edge_cases():
    assert f1_score("", "Berlin") == 0.0
    assert f1_score("", "") == 1.0
    assert f1_score("Berlin Berlin", "Berlin") == pytest.approx(2 / 3)  # P=1/2, R=1/1


def test_max_over_reference_answers():
    refs = ["Albert Einstein", "Einstein"]
    assert max_over_ground_truths(exact_match_score, "einstein", refs) == 1.0
    assert max_over_ground_truths(f1_score, "Albert", refs) == pytest.approx(2 / 3)


# ---------- Latency ----------

def test_percentile_linear_interpolation_hand_computed():
    values = [40.0, 10.0, 30.0, 20.0]
    assert percentile(values, 50) == pytest.approx(25.0)  # pos 1.5 → 20 + 0.5·10
    assert percentile(values, 95) == pytest.approx(38.5)  # pos 2.85 → 30 + 0.85·10
    assert summarize_latency(values) == pytest.approx({"n": 4, "mean": 25.0, "p50": 25.0, "p95": 38.5})


def test_latency_summary_empty():
    assert summarize_latency([]) == {"n": 0, "mean": None, "p50": None, "p95": None}


# ---------- Error decomposition: từng quy tắc ----------

def test_rerank_off_gold_beyond_reader_k_is_retrieval_miss():
    retrieved = fill("x", 7) + ["gold"] + fill("y", 12)  # gold hạng 8
    m = evaluate_end_to_end(
        [qa("q1", ["gold"])],
        [pred("q1", retrieved, answer="Marie Curie", config_id="bm25_base_rerank-off")],
    )
    row = m["per_query"][0]
    assert row["bucket"] == "retrieval_miss"
    assert (row["gold_in_topk"], row["first_gold_rank"]) == (False, 8)


def test_rerank_on_gold_dropped_by_reranker_is_reranking_failure():
    retrieved = fill("x", 7) + ["gold"] + fill("y", 12)
    m = evaluate_end_to_end([qa("q1", ["gold"])], [pred("q1", retrieved, retrieved[:5], answer="Marie Curie")])
    row = m["per_query"][0]
    assert row["bucket"] == "reranking_failure"
    assert (row["gold_in_topk"], row["gold_in_topk_prime"]) == (True, False)


def test_lost_gold_but_correct_answer_keeps_bucket_and_records_final_correct():
    retrieved = fill("x", 20)
    m = evaluate_end_to_end([qa("q1", ["gold"])], [pred("q1", retrieved, retrieved[:5], answer=REF)])
    row = m["per_query"][0]
    assert (row["bucket"], row["final_correct"]) == ("retrieval_miss", True)
    assert m["error_decomposition"]["final_correct_by_bucket"]["retrieval_miss"] == 1


def test_gold_without_reference_answers_is_excluded_not_fatal():
    gold = [qa("q1", ["g"]), {**qa("q2", []), "reference_answers": []}]
    m = evaluate_end_to_end(gold, [pred("q1", ["g"], ["g"], answer=REF)])
    assert (m["counts"]["n_eval_queries"], m["counts"]["n_excluded_empty_gold"]) == (1, 1)
    assert m["qa"]["em"] == 1.0


# ---------- Smoke: đủ 5 bucket qua CLI ----------

def test_cli_smoke_all_five_buckets(tmp_path):
    gold, preds = smoke_dataset()
    out = tmp_path / "metrics.json"
    code = main([
        "--gold", write_jsonl(tmp_path / "gold.jsonl", gold),
        "--predictions", write_jsonl(tmp_path / "pred.jsonl", preds),
        "--out", str(out),
        "--corpus-id", "base",
    ])
    assert code == 0
    m = json.loads(out.read_text(encoding="utf-8"))

    assert m["mode"] == "end_to_end" and m["corpus_id"] == "base" and m["run_id"] == "run-1"
    assert m["counts"] == {
        "n_gold_records": 21, "n_excluded_empty_gold": 1, "n_eval_queries": 20,
        "n_predictions": 19, "n_missing_predictions": 2, "n_status_error": 2,
    }

    dec = m["error_decomposition"]
    assert dec["counts"] == {b: 4 for b in
                             ("execution_failure", "retrieval_miss", "reranking_failure", "reader_failure", "correct")}
    assert dec["counts_sum_equals_n"] is True
    assert sum(dec["rates"].values()) == pytest.approx(1.0)
    assert dec["final_correct_by_bucket"]["retrieval_miss"] == 1
    assert dec["final_correct_by_bucket"]["correct"] == 4

    r = m["retrieval"]  # gold hạng 1: 2 error + 4 reader + 4 correct; hạng 8: 4 rerank; hạng 25: 4 retrieval
    assert (r["recall@1"], r["recall@5"], r["recall@10"], r["recall@20"]) == pytest.approx((0.5, 0.5, 0.7, 0.7))
    assert r["mrr@10"] == pytest.approx((10 * 1 + 4 * (1 / 8)) / 20)
    assert r["gold_coverage@20"] == pytest.approx(r["recall@20"])

    qa_m = m["qa"]  # EM: 4 correct + 1 retrieval_miss trả lời đúng; F1 cộng thêm 2 câu trùng một phần (2/3)
    assert qa_m["label"] == "end_to_end"
    assert qa_m["em"] == pytest.approx(5 / 20)
    assert qa_m["f1"] == pytest.approx((5 + 2 * (2 / 3)) / 20)
    assert qa_m["no_prediction"] == 5  # 2 thiếu + 2 lỗi + 1 reader không có đáp án

    assert m["manual_review"]["n_flagged"] == 2
    assert m["manual_review"]["sample_query_ids"] == ["q-reader-0", "q-reader-1"]

    lat = m["latency_ms"]  # 16 record ok thuộc tập eval
    assert (lat["retrieval"]["n"], lat["total"]["n"]) == (16, 16)
    assert (lat["retrieval"]["mean"], lat["rerank"]["mean"], lat["total"]["p95"]) == pytest.approx((1.0, 2.0, 6.0))

    rows = {row["query_id"]: row for row in m["per_query"]}
    assert "q-nogold" not in rows
    assert rows["q-exec-missing-0"]["status"] == "missing" and rows["q-exec-missing-0"]["gold_in_topk"] is None


def test_same_input_in_any_order_gives_identical_metrics():
    gold, preds = smoke_dataset()
    reference = json.dumps(evaluate_end_to_end(gold, preds), ensure_ascii=False)
    rng = random.Random(123)
    for _ in range(5):
        g, p = gold[:], preds[:]
        rng.shuffle(g)
        rng.shuffle(p)
        assert json.dumps(evaluate_end_to_end(g, p), ensure_ascii=False) == reference


# ---------- Kiểm tra nhất quán pipeline / run ----------

def test_predicted_passage_outside_reader_input_is_rejected():
    retrieved = ["g"] + fill("x", 19)
    p = pred("q1", retrieved, retrieved[:5], answer=REF, answer_pid="x10")  # x10 hạng 12, reader không đọc
    with pytest.raises(SchemaError, match="không nằm trong"):
        evaluate_end_to_end([qa("q1", ["g"])], [p])


def test_reranked_must_come_from_retrieval_pool_with_reader_k_items():
    retrieved = ["g"] + fill("x", 24)
    outside = pred("q1", retrieved, ["g", "x1", "x2", "x3", "x22"], answer=REF, answer_pid="g")  # x22 hạng 24
    with pytest.raises(SchemaError, match="ngoài top-20"):
        evaluate_end_to_end([qa("q1", ["g"])], [outside])
    too_many = pred("q1", retrieved, retrieved[:6], answer=REF, answer_pid="g")
    with pytest.raises(SchemaError, match="phải đúng"):
        evaluate_end_to_end([qa("q1", ["g"])], [too_many])


def test_run_level_checks():
    gold = [qa("q1", ["g1"]), qa("q2", ["g2"])]
    p1 = pred("q1", ["g1"], ["g1"], answer=REF)
    with pytest.raises(SchemaError, match="corpus_id khác"):
        evaluate_end_to_end(gold, [p1], expected_corpus_id="mid")
    with pytest.raises(SchemaError, match="trộn nhiều run_id"):
        evaluate_end_to_end(gold, [p1, pred("q2", ["g2"], ["g2"], answer=REF, run_id="run-2")])
    with pytest.raises(SchemaError, match="query_id trùng"):
        evaluate_end_to_end(gold, [p1, p1])
    with pytest.raises(SchemaError, match="không có trong gold"):
        evaluate_end_to_end(gold, [pred("q9", ["g1"], ["g1"], answer=REF)])


# ---------- Reader-only & retrieval-only ----------

def test_reader_only_mode_labels_capability_and_skips_retrieval():
    gold = [qa("q1", ["g1"]), qa("q2", ["g2"]), qa("q3", [])]
    oracle = dict(config_id="oracle_base_rerank-off")
    preds = [pred("q1", [], answer=REF, answer_pid="g1", **oracle),
             pred("q2", [], answer="Einstein", answer_pid="g2", **oracle)]
    m = evaluate_reader_only(gold, preds)
    assert m["qa"]["label"] == "reader_capability_on_gold_passage"
    assert "retrieval" not in m and "error_decomposition" not in m
    assert m["counts"]["n_eval_queries"] == 2
    assert (m["qa"]["em"], m["qa"]["f1"]) == pytest.approx((0.5, (1 + 2 / 3) / 2))
    assert m["manual_review"]["sample_query_ids"] == ["q2"]
    with pytest.raises(SchemaError, match="không phải gold passage"):
        evaluate_reader_only(gold, [pred("q1", [], answer=REF, answer_pid="other", **oracle)])


def test_retrieval_only_from_rows_is_order_independent():
    gold = [qa("q1", ["a"]), qa("q2", ["b"]), qa("q3", ["c"]), qa("q4", [])]
    rows = ranked("q1", ["a", "x"]) + ranked("q2", ["x", "y", "b"])
    random.Random(0).shuffle(rows)
    m = evaluate_retrieval_results(gold, rows, k_values=(1, 5), mrr_cutoff=10)
    assert m["mode"] == "retrieval_only" and m["retriever_names"] == ["bm25"]
    assert (m["counts"]["n_eval_queries"], m["counts"]["n_queries_without_results"]) == (3, 1)
    assert m["retrieval"]["recall@1"] == pytest.approx(1 / 3)
    assert m["retrieval"]["mrr@10"] == pytest.approx((1 + 1 / 3) / 3)
    assert [r["first_gold_rank"] for r in m["per_query"]] == [1, 3, None]


def test_retrieval_only_rejects_unknown_query_and_broken_ranks():
    gold = [qa("q1", ["a"])]
    with pytest.raises(SchemaError, match="không có trong gold"):
        evaluate_retrieval_results(gold, ranked("zz", ["a"]))
    rows = ranked("q1", ["a", "b"])
    rows[1]["rank"] = 1
    with pytest.raises(SchemaError, match="liên tục"):
        evaluate_retrieval_results(gold, rows)


def test_cli_rejects_round1_schema_gold_with_exit_code_2(tmp_path, capsys):
    old_gold = {
        "query_id": "gq-test-55538", "question": "Was ist die Hauptstadt der Marshallinseln?",
        "gold_passage_ids": ["7091e19c075e5aa6"], "answers": [{"text": "Majuro", "answer_start": 304}],
        "source_dataset": "germanquad", "source_split": "test",
    }
    out = tmp_path / "m.json"
    code = main([
        "--gold", write_jsonl(tmp_path / "g.jsonl", [old_gold]),
        "--retrieval-results", write_jsonl(tmp_path / "r.jsonl", ranked("gq-test-55538", ["7091e19c075e5aa6"])),
        "--out", str(out),
    ])
    assert code == 2
    assert "gold_evidences" in capsys.readouterr().err
    assert not out.exists()
