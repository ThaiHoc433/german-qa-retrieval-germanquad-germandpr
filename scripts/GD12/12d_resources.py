"""GĐ12.4: tài nguyên của nhánh dense LÚC PHỤC VỤ (question encoder + FAISS), từng câu một.
GPU : python scripts\\GD12\\12d_resources.py
CPU : $env:CUDA_VISIBLE_DEVICES='-1'; python scripts\\GD12\\12d_resources.py     (đặt '' sẽ XÓA biến, không phải đặt rỗng)"""
import argparse
import time

import numpy as np
import psutil
import torch

from germanqa.config import PROCESSED_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.experiment import load_config
from germanqa.retrieval.dense import DenseRetriever, DPREncoder


def main(a):
    cfg = load_config("dense")["dense"]
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    enc = DPREncoder(cfg["ctx_encoder"], cfg["question_encoder"], fp16=cfg["fp16"])
    t = time.perf_counter()
    _ = enc.question                                              # nạp question encoder (context encoder KHÔNG cần lúc phục vụ)
    t_load = time.perf_counter() - t
    t = time.perf_counter()
    r = DenseRetriever.from_dir(ROOT / "artifacts" / "dense" / a.artifact, enc, PROCESSED_DIR / "corpus.jsonl",
                                cfg["question_max_length"], expected_cfg=cfg)
    t_index = time.perf_counter() - t
    unseen = set(read_json(RESULTS_DIR / "dev_unseen_question_ids.json"))
    qs = [q["question"] for q in read_jsonl(PROCESSED_DIR / "qa_dev.jsonl") if q["question_id"] in unseen][:a.n]
    r.retrieve(qs[0], 20)                                         # khởi động, không tính
    enc_t, srch_t = [], []
    for q in qs:
        r.retrieve(q, 20)
        enc_t.append(r.last_timing["encode"] * 1000)
        srch_t.append(r.last_timing["search"] * 1000)
    out = {"device": enc.device.type, "artifact": a.artifact, "n_questions": len(qs), "fp16": enc.fp16,
           "load_question_encoder_s": t_load, "load_index_s": t_index,
           "ram_added_mb": (proc.memory_info().rss - rss0) / 2**20,
           "vram_peak_mb": torch.cuda.max_memory_allocated() / 2**20 if enc.device.type == "cuda" else None,
           "encode_ms": {"mean": float(np.mean(enc_t)), "p95": float(np.percentile(enc_t, 95))},
           "search_ms": {"mean": float(np.mean(srch_t)), "p95": float(np.percentile(srch_t, 95))}}
    write_json(RESULTS_DIR / f"12d_resources_{out['device']}.json", out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="text384")
    ap.add_argument("--n", type=int, default=100)
    main(ap.parse_args())
