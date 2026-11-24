from germanqa.evaluation.retrieval_metrics import (evaluate_run, hit_at_k, latency_summary,
                                              mrr_at_k, recall_at_k)

RANKED = ["a", "b", "c", "d"]
REL = {"c", "x"}                       # "x" không có trong danh sách: qrels có thể nhiều hơn những gì tìm thấy


def test_hit():
    assert hit_at_k(RANKED, REL, 2) == 0.0
    assert hit_at_k(RANKED, REL, 3) == 1.0


def test_recall_tinh_tay():
    assert recall_at_k(RANKED, REL, 3) == 0.5          # tìm được 1 trên 2 đoạn đúng
    assert recall_at_k(RANKED, set(), 3) == 0.0


def test_mrr_tinh_tay():
    assert mrr_at_k(RANKED, REL, 2) == 0.0
    assert abs(mrr_at_k(RANKED, REL, 3) - 1 / 3) < 1e-12


def test_evaluate_run_trung_binh_tren_cau_hoi():
    run = {"q1": ["a", "b"], "q2": ["x", "a"]}
    qrels = {"q1": ["a"], "q2": ["a"]}
    m = evaluate_run(run, qrels, ks=(1, 2))
    assert m["hit@1"] == 0.5 and m["hit@2"] == 1.0
    assert m["mrr@2"] == 0.75                             # (1 + 1/2) / 2
    assert m["queries"] == 2


def test_cau_hoi_thieu_trong_run_tinh_la_0():
    m = evaluate_run({}, {"q1": ["a"]}, ks=(5,))
    assert m["hit@5"] == 0.0


def test_latency():
    s = latency_summary([0.001, 0.002, 0.003])
    assert abs(s["mean_ms"] - 2.0) < 1e-9 and abs(s["p50_ms"] - 2.0) < 1e-9
