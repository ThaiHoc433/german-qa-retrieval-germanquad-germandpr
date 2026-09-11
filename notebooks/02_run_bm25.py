"""Milestone C — BM25 trên corpus_base, chạy trên dev, lưu ranks + metrics.

Điều kiện chuyển bước: có baseline tái lập được. Dùng DEV (không phải test) vì
đây là giai đoạn phát triển/kiểm tra pipeline — không đụng test theo đúng
nguyên tắc "đừng dùng kết quả test để sửa hyperparameter sau mỗi lần chạy".
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25_retriever import BM25Retriever, DEFAULT_TOP_N
from src.evaluation import evaluate_retrieval

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RUN_ID = "bm25_base_dev"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    print("Loading corpus_base...")
    corpus = load_jsonl(DATA_DIR / "corpus_base.jsonl")
    passage_ids = [r["passage_id"] for r in corpus]
    normalized_texts = [r["normalized_text"] for r in corpus]

    print(f"Indexing BM25 on {len(passage_ids)} passages...")
    t0 = time.perf_counter()
    retriever = BM25Retriever(passage_ids, normalized_texts)
    index_s = time.perf_counter() - t0
    print(f"Index build time: {index_s:.1f}s (KHÔNG tính vào latency truy vấn, spec 9.3)")

    print("Loading qa_germanquad_dev...")
    dev = load_jsonl(DATA_DIR / "qa_germanquad_dev.jsonl")

    gold_by_query = {r["query_id"]: r["gold_passage_ids"] for r in dev}
    retrieved_by_query = {}
    retrieval_records = []

    print(f"Running BM25 search on {len(dev)} dev queries (top_n={DEFAULT_TOP_N})...")
    latencies_ms = []
    for row in dev:
        t0 = time.perf_counter()
        results = retriever.search(row["question"], top_n=DEFAULT_TOP_N)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        retrieved_by_query[row["query_id"]] = [pid for pid, _ in results]
        for rank, (pid, score) in enumerate(results, start=1):
            retrieval_records.append(
                {
                    "query_id": row["query_id"],
                    "passage_id": pid,
                    "rank": rank,
                    "score": score,
                    "retriever_name": "bm25",
                }
            )

    out_path = DATA_DIR / f"retrieval_{RUN_ID}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in retrieval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(retrieval_records)} retrieval results to {out_path}")

    metrics = evaluate_retrieval(gold_by_query, retrieved_by_query, k_values=(1, 5, 10, 20), mrr_cutoff=10)
    latencies_ms.sort()
    n = len(latencies_ms)
    metrics["latency_ms"] = {
        "mean": sum(latencies_ms) / n,
        "p50": latencies_ms[int(0.5 * (n - 1))],
        "p95": latencies_ms[int(0.95 * (n - 1))],
        "n_queries_timed": n,
        "note": "đo trên máy local (không phải Colab T4), chỉ để tham khảo tương đối — số chính thức xem 3.3 (thiết bị Colab T4)",
    }
    metrics["run_id"] = RUN_ID
    metrics["corpus_id"] = "base"
    metrics["split"] = "dev"
    metrics["index_build_time_s"] = index_s

    out_metrics = RESULTS_DIR / f"metrics_{RUN_ID}.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved metrics to {out_metrics}")


if __name__ == "__main__":
    main()
