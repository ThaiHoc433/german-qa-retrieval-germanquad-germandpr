"""GĐ3: BM25 baseline. Chọn k1, b (và biến thể stopword) trên DEV; chỉ chạy test khi có --test."""
import argparse
import itertools
import time
from pathlib import Path

from germanqa.config import PROCESSED_DIR, SEED
from germanqa.data.io_utils import read_json, read_jsonl
from germanqa.evaluation.retrieval_metrics import evaluate_run, latency_summary
from germanqa.experiment import load_config, new_experiment_id, save_run, set_seed
from germanqa.experiment_rules import guard_or_exit
from germanqa.retrieval.bm25 import BM25Retriever


def run_split(retriever, rows, top_k):
    """Chạy truy hồi cho mọi câu hỏi; trả về (run, thời gian từng câu)."""
    run, times = {}, []
    for r in rows:
        t0 = time.perf_counter()
        hits = retriever.retrieve(r["question"], top_k)
        times.append(time.perf_counter() - t0)
        run[r["question_id"]] = [h["passage_id"] for h in hits]
    return run, times


def main(data_dir, do_test):
    if do_test:
        guard_or_exit("03_bm25_eval.py", "BM25 trên test với cấu hình đã chốt trên dev", True, files=["qa_test.jsonl", "qrels_test.json"], codes=["B0"])
    cfg = load_config("baseline")
    set_seed(SEED)
    ks = cfg["evaluation"]["ks"]
    top_k = max(ks)
    corpus = read_jsonl(Path(data_dir) / "corpus.jsonl")
    dev_rows = read_jsonl(Path(data_dir) / "qa_dev.jsonl")
    dev_qrels = read_json(Path(data_dir) / "qrels_dev.json")
    select_by = cfg["evaluation"]["select_by"]

    # ---- chọn cấu hình trên DEV ----
    trials = []
    for stop, k1, b in itertools.product((False, True), cfg["bm25"]["grid"]["k1"],
                                         cfg["bm25"]["grid"]["b"]):
        retr = BM25Retriever(corpus, k1=k1, b=b, use_stopwords=stop)
        run, _ = run_split(retr, dev_rows, top_k)
        m = evaluate_run(run, dev_qrels, ks)
        trials.append({"use_stopwords": stop, "k1": k1, "b": b, "metrics": m})
        print(f"stop={stop!s:5} k1={k1:<4} b={b:<4} {select_by}={m[select_by]:.4f} "
              f"recall@20={m['recall@20']:.4f}")
    best = max(trials, key=lambda t: (t["metrics"][select_by], t["metrics"]["recall@20"]))
    print("CHỌN:", {k: best[k] for k in ("use_stopwords", "k1", "b")})

    # ---- đo latency + (tùy chọn) test với cấu hình đã đóng băng ----
    final = BM25Retriever(corpus, k1=best["k1"], b=best["b"], use_stopwords=best["use_stopwords"])
    run, times = run_split(final, dev_rows, top_k)
    result = {"dev": {"metrics": evaluate_run(run, dev_qrels, ks), "latency": latency_summary(times)}}
    if do_test:
        test_rows = read_jsonl(Path(data_dir) / "qa_test.jsonl")
        test_qrels = read_json(Path(data_dir) / "qrels_test.json")
        run_t, times_t = run_split(final, test_rows, top_k)
        result["test"] = {"metrics": evaluate_run(run_t, test_qrels, ks),
                          "latency": latency_summary(times_t)}
    config = {"bm25": {k: best[k] for k in ("use_stopwords", "k1", "b")}, "ks": ks}
    path = save_run(new_experiment_id("bm25", config), config, result,
                    extra={"dev_trials": trials})
    print("đã ghi", path)
    for split, v in result.items():
        print(split, {k: round(x, 4) for k, x in v["metrics"].items()}, v["latency"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROCESSED_DIR))
    ap.add_argument("--test", action="store_true",
                    help="chạy thêm test (chỉ sau khi đã chốt cấu hình trên dev)")
    a = ap.parse_args()
    main(a.data, a.test)