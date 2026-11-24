"""GĐ15.4: thời gian từng tầng của B0, B1, H0, H1 trong CÙNG MỘT tiến trình, cùng phần cứng, cùng trạng thái GPU, cùng 100 câu dev-unseen.
Quy trình: dựng một pipeline hybrid (có reranker) rồi lắp bốn QAPipeline dùng chung reader và văn bản đoạn, chỉ khác bộ truy hồi/reranker.
Khởi động 5 câu mỗi hệ (bỏ), rồi 3 vòng; thứ tự bốn hệ xoay giữa các vòng để nhiệt độ/xung nhịp không dồn vào một hệ. Lấy TRUNG VỊ (laptop có thể ngủ làm một mẫu hỏng).
Ghi cả số cửa sổ reader và số cặp/cửa sổ của reranker để giải thích chi phí (năm đoạn có thể tạo số cửa sổ khác nhau).
Ghi results/15_stage_timing.json (không ghi đè). Chỉ dev-unseen, không test."""
import statistics
import subprocess
import time

import numpy as np
import torch

from germanqa.config import PROCESSED_DIR, RESULTS_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.evaluation.report import hardware_info
from germanqa.experiment import load_config
from germanqa.pipeline import QAPipeline, build_pipeline

N, ROUNDS, WARM = 100, 3, 5


def smi_used():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])


def reader_windows(pipe, question, passage_ids):
    q = pipe._cut_question(question)[0]
    rc = load_config("reader")["reader"]
    enc = pipe.reader.tokenizer([q] * len(passage_ids), [pipe.passages[p] for p in passage_ids], truncation="only_second",
                                max_length=rc["max_length"], stride=rc["stride"], return_overflowing_tokens=True)
    return len(enc["input_ids"])


def main():
    out_path = RESULTS_DIR / "15_stage_timing.json"
    if out_path.exists():
        raise SystemExit(f"{out_path.name} đã tồn tại: không ghi đè")
    smi_before = smi_used()
    h1 = build_pipeline(retriever="hybrid", reranker=True)
    cfg = load_config("pipeline")["pipeline"]
    mk = lambda retriever, rr: QAPipeline(retriever, h1.reader, h1.passages, h1.model_version, h1.corpus_version, "timing", cfg["retrieval_top_k"], cfg["reader_top_k"], reranker=rr)
    systems = {"B0": mk(h1.retriever.bm25, None), "B1": mk(h1.retriever.dense, None), "H0": mk(h1.retriever, None), "H1": mk(h1.retriever, h1.reranker)}
    unseen = set(read_json(RESULTS_DIR / "select_question_ids.json")["dev"])
    qs = [r["question"] for r in read_jsonl(PROCESSED_DIR / "qa_dev.jsonl") if r["question_id"] in unseen][:N]
    assert len(qs) == N
    for c, p in systems.items():
        for q in qs[:WARM]:
            p.answer(q)
    torch.cuda.reset_peak_memory_stats()
    smi_after_build = smi_used()
    rec = {c: {"retrieval": [], "rerank": [], "reader": [], "total": [], "reader_windows": [], "rerank_windows": []} for c in systems}
    order = list(systems)
    for rnd in range(ROUNDS):
        for c in order[rnd % 4:] + order[:rnd % 4]:
            p = systems[c]
            for q in qs:
                r = p.answer(q, trace=True)
                rec[c]["retrieval"].append(r.retrieval_time)
                rec[c]["rerank"].append(r.rerank_time)
                rec[c]["reader"].append(r.reader_time)
                rec[c]["total"].append(r.total_time)
                rec[c]["reader_windows"].append(reader_windows(p, q, [x["passage_id"] for x in r.trace["candidates"]]))
                if r.trace.get("rerank_stats"):
                    rec[c]["rerank_windows"].append(r.trace["rerank_stats"]["windows"])
    med = lambda xs: round(1000 * statistics.median(xs), 1)
    res = {"purpose": "thời gian từng tầng, bốn hệ cùng tiến trình", "questions": N, "rounds": ROUNDS, "warmup_per_system": WARM, "statistic": "trung vị",
           "hardware": hardware_info(), "gpu_used_mb": {"before_loading_anything": smi_before, "after_build": smi_after_build, "max_during_run": smi_used()},
           "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "systems": {}}
    for c, d in rec.items():
        res["systems"][c] = {"median_ms": {k: med(d[k]) for k in ("retrieval", "rerank", "reader", "total")},
                             "p90_total_ms": round(1000 * float(np.percentile(d["total"], 90)), 1),
                             "reader_windows_per_question": {"mean": round(float(np.mean(d["reader_windows"])), 2), "median": float(np.median(d["reader_windows"]))},
                             "rerank_windows_per_question_mean": round(float(np.mean(d["rerank_windows"])), 2) if d["rerank_windows"] else None,
                             "samples": len(d["total"])}
    write_json(out_path, res)
    print(f"{'':4} {'truy hồi':>9} {'xếp lại':>8} {'reader':>8} {'tổng':>8} {'p90 tổng':>9} {'cửa sổ reader':>14}")
    for c, s in res["systems"].items():
        m = s["median_ms"]
        print(f"{c:4} {m['retrieval']:9.1f} {m['rerank']:8.1f} {m['reader']:8.1f} {m['total']:8.1f} {s['p90_total_ms']:9.1f} {s['reader_windows_per_question']['mean']:14.2f}")
    print("GPU (MB):", res["gpu_used_mb"], "| đỉnh cấp phát torch:", res["torch_peak_allocated_mb"])


if __name__ == "__main__":
    main()
