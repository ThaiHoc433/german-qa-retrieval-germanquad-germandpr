"""GĐ14.4: chấm điểm pool top-20 của H0 cho 2.223 câu của tập chọn bằng từng model ứng viên, MỘT lượt (chế độ cửa sổ:
điểm cửa sổ đầu = điểm 'truncate', điểm lớn nhất = điểm 'windows'). Lưu vào results/14c_scores_<model>.jsonl kèm
.meta.json (dấu vân tay). Chạy tiếp được nếu bị ngắt; dấu vân tay lệch thì từ chối chạy tiếp.
Script này KHÔNG tính chỉ số nào: chỉ ghi điểm. Việc đọc điểm của dev-unseen do 14e đảm nhiệm (xem access_rule trong prereg)."""
import argparse
import json
import statistics
import time

import torch

from germanqa.config import PROCESSED_DIR, RESULTS_DIR, ROOT
from germanqa.data.io_utils import file_sha1, read_json, read_jsonl, write_json
from germanqa.evaluation.fusion_select import require_prereg
from germanqa.rerank.cross_encoder import CrossEncoderReranker
from germanqa.rerank.hub import local_snapshot
from germanqa.rerank.pool import hybrid_pool


def check_model_files(snap, want):
    import pathlib
    bad = [f"{n}: ghi trước {h}, hiện tại {file_sha1(pathlib.Path(snap) / n) if (pathlib.Path(snap) / n).exists() else None}"
           for n, h in want.items() if not (pathlib.Path(snap) / n).exists() or file_sha1(pathlib.Path(snap) / n) != h]
    if bad:
        raise SystemExit("tệp model đã đổi so với prereg: " + "; ".join(bad))


def main(keys, limit):
    pre = require_prereg(RESULTS_DIR / "14_prereg.json", ROOT)
    sc = pre["scoring"]
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    qids = list(sel["train"]) + list(sel["dev"])
    if limit:
        qids = qids[:limit]
    qa = {}
    for split in ("train", "dev"):
        for r in read_jsonl(PROCESSED_DIR / f"qa_{split}.jsonl"):
            qa[r["question_id"]] = r["question"]
    corpus = {r["passage_id"]: r["original_text"] for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}
    pool = hybrid_pool(qids)
    for key in keys:
        m = pre["models"][key]
        snap = local_snapshot(m["repo"])
        check_model_files(snap, m["files_sha1"])
        rr = CrossEncoderReranker(snap, max_length=sc["max_length"], batch_size=sc["batch_size"], fp16=sc["fp16"],
                                  relevant_label=m["relevant_label"], overflow="windows", stride=sc["stride_windows"])
        fp = rr.fingerprint()
        out = RESULTS_DIR / f"14c_scores_{key}{'_smoke' if limit else ''}.jsonl"
        meta_path = out.with_suffix(".meta.json")
        done = set()
        if out.exists():
            if not meta_path.exists() or read_json(meta_path)["fingerprint"]["sha1"] != fp["sha1"]:
                raise SystemExit(f"{out.name}: dấu vân tay lệch hoặc thiếu meta; xóa tay nếu muốn chấm lại")
            done = {json.loads(l)["question_id"] for l in open(out, encoding="utf-8") if l.strip()}
        else:
            write_json(meta_path, {"model": key, "repo": m["repo"], "commit": m["commit"], "fingerprint": fp,
                                   "prereg_created_at": pre["created_at"], "pool_size": 20})
        todo = [q for q in qids if q not in done]
        print(f"[{key}] đã có {len(done)}, còn {len(todo)}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        times, wins, t_all = [], 0, time.perf_counter()
        with open(out, "a", encoding="utf-8") as f:
            for i, q in enumerate(todo, 1):
                pids = pool[q]
                d = rr.score_detail(qa[q], [corpus[p] for p in pids])
                times.append(rr.last_stats["seconds"])
                wins += rr.last_stats["windows"]
                f.write(json.dumps({"question_id": q, "passage_ids": pids, "score": d["score"], "first": d["first"],
                                    "windows": d["windows"]}, ensure_ascii=False) + "\n")
                if i % 200 == 0:
                    f.flush()
                    print(f"[{key}] {i}/{len(todo)} | trung vị {1000 * statistics.median(times):.0f} ms/câu", flush=True)
        if todo:
            stats = {"questions": len(todo), "median_ms_per_question": round(1000 * statistics.median(times), 1),
                     "mean_windows_per_question": round(wins / len(todo), 2), "wall_seconds": round(time.perf_counter() - t_all, 1),
                     "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None}
            write_json(out.with_suffix(".stats.json"), stats)
            print(f"[{key}] xong", stats, flush=True)
        del rr
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mmarco", "gbert_dpr", "bge_m3"])
    ap.add_argument("--limit", type=int, default=None, help="chạy thử trên N câu đầu, ghi vào tệp *_smoke")
    a = ap.parse_args()
    main(a.models, a.limit)
