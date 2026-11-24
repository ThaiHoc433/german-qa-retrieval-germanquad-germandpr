"""GĐ14.5: đo VRAM của H1 dưới điều kiện GIỐNG máy thật, bằng MÔ PHỎNG: chiếm sẵn phần VRAM mà chương trình khác đang dùng.
Vì sao mô phỏng: reports/api_measurements.json (GĐ8) ghi trước khi nạp dịch vụ, card đã dùng 1.377 MB; mày nói 1,4-1,7 GB. Thay vì nhờ mày mở chương trình,
script chiếm đúng phần chênh giữa mức đó và mức nhàn rỗi hiện tại (đọc bằng nvidia-smi), rồi dựng pipeline trong MỘT tiến trình riêng cho từng biến thể.
Đây là mô phỏng: chương trình thật còn biến động và có thể giành thêm bộ nhớ lúc chạy; kết quả chỉ nói 'vừa/không vừa ở mức nền đã chọn'.
Tách rõ: đỉnh lúc DỰNG (nạp model) và đỉnh lúc TRẢ LỜI (sau khi reset bộ đếm). Số "own_*" đã trừ phần chiếm giả lập; "smi_*" là tổng cả card. Không mở test; dùng 30 câu dev-unseen.
Cách chạy: không tham số -> chạy mọi biến thể (mỗi biến thể một tiến trình con) và ghi results/14l_vram_real.json (không ghi đè)."""
import argparse
import json
import statistics
import subprocess
import sys
import time

from germanqa.config import PROCESSED_DIR, RESULTS_DIR, ROOT

TARGETS = (1377, 1700)                       # MB do chương trình khác chiếm: số đo GĐ8 và cận trên mày nêu
VARIANTS = [("none", 0), ("bge_m3", 16), ("bge_m3", 8), ("bge_m3", 4), ("bge_m3_offload", 16), ("mmarco", 16)]
N_QUESTIONS = 30
MARK = "@@RESULT@@"


def smi_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
    return used, total


class OffloadReranker:
    """Giữ reranker trên CPU, chỉ đưa lên GPU trong lúc xếp lại rồi trả về: tiết kiệm VRAM, đổi lại độ trễ."""

    def __init__(self, rr):
        import torch
        self.rr, self.torch = rr, torch
        self.rr.model.to("cpu")
        self.rr.device = "cpu"
        self.last_stats = rr.last_stats

    def rerank(self, query, candidates, top_n=None):
        torch = self.torch
        self.rr.model.to("cuda")
        self.rr.device = "cuda"
        try:
            return self.rr.rerank(query, candidates, top_n)
        finally:
            self.last_stats = self.rr.last_stats
            self.rr.model.to("cpu")
            self.rr.device = "cpu"
            torch.cuda.empty_cache()


def worker(target_mb, variant, batch):
    import torch
    from germanqa.data.io_utils import read_json, read_jsonl
    from germanqa.experiment import load_config
    from germanqa.pipeline import build_pipeline
    from germanqa.rerank.cross_encoder import CrossEncoderReranker
    from germanqa.rerank.hub import local_snapshot

    res = {"target_other_mb": target_mb, "variant": variant, "batch": batch}
    idle, total = smi_used()
    torch.zeros(1, device="cuda")                                   # khởi tạo ngữ cảnh CUDA của tiến trình này
    hold_mb = max(0, target_mb - idle)
    hold = torch.empty(int(hold_mb * 2 ** 20), dtype=torch.uint8, device="cuda") if hold_mb else None
    res.update(idle_used_mb_before_process=idle, total_mb=total, simulated_other_mb=hold_mb, used_mb_after_hold=smi_used()[0])
    try:
        rr = None
        if variant != "none":
            cfg = load_config("rerank")["rerank"]
            path = local_snapshot("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") if variant == "mmarco" else ROOT / cfg["checkpoint"]
            rr = CrossEncoderReranker(path, max_length=512, batch_size=batch, fp16=True, relevant_label=None, overflow="windows", stride=128)
            if variant == "bge_m3_offload":
                rr = OffloadReranker(rr)
        pipe = build_pipeline(retriever="hybrid", reranker=rr if rr is not None else None)
        res["after_build"] = {"own_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20 - hold_mb), "smi_used_mb": smi_used()[0]}
        unseen = set(read_json(RESULTS_DIR / "select_question_ids.json")["dev"])
        qs = [r["question"] for r in read_jsonl(PROCESSED_DIR / "qa_dev.jsonl") if r["question_id"] in unseen][:N_QUESTIONS]
        pipe.answer(qs[0])                                          # khởi động, không tính
        torch.cuda.reset_peak_memory_stats()
        rows, smi_max = [], 0
        for q in qs:
            r = pipe.answer(q)
            rows.append(r)
            smi_max = max(smi_max, smi_used()[0])
        med = lambda xs: round(1000 * statistics.median(xs), 1)
        res["answering"] = {"questions": len(rows), "own_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20 - hold_mb),
                            "own_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20 - hold_mb), "smi_used_max_mb": smi_max,
                            "headroom_mb": total - smi_max,
                            "median_ms": {"retrieval": med([r.retrieval_time for r in rows]), "rerank": med([r.rerank_time for r in rows]),
                                          "reader": med([r.reader_time for r in rows]), "total": med([r.total_time for r in rows])}}
        res["fits"] = True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        res["fits"] = False
        res["oom"] = str(e).splitlines()[0][:200]
    finally:
        del hold
    print(MARK + json.dumps(res, ensure_ascii=False))


def driver():
    out = RESULTS_DIR / "14l_vram_real.json"
    if out.exists():
        raise SystemExit(f"{out.name} đã tồn tại: không ghi đè")
    runs = []
    for target in TARGETS:
        for variant, batch in VARIANTS:
            t = time.perf_counter()
            p = subprocess.run([sys.executable, __file__, "--worker", "--target", str(target), "--variant", variant, "--batch", str(batch)],
                               capture_output=True, text=True, env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"})
            line = next((l for l in p.stdout.splitlines() if l.startswith(MARK)), None)
            if line is None:
                runs.append({"target_other_mb": target, "variant": variant, "batch": batch, "fits": None, "error": (p.stderr or p.stdout)[-300:]})
            else:
                runs.append(json.loads(line[len(MARK):]))
            r = runs[-1]
            a = r.get("answering", {})
            print(f"{target} MB khác | {variant:15} b{batch:<2} | vừa: {r.get('fits')} | smi đỉnh {a.get('smi_used_max_mb')} MB, còn {a.get('headroom_mb')} MB | "
                  f"tổng {a.get('median_ms', {}).get('total')} ms | {time.perf_counter() - t:.0f}s", flush=True)
    doc = {"purpose": "H1 có vừa card 4 GB khi chương trình khác chiếm sẵn một phần?", "method": "mô phỏng bằng cách chiếm sẵn VRAM (xem docstring)",
           "source_of_target": "reports/api_measurements.json: gpu_used_mb_before = 1377; mày nêu 1,4-1,7 GB",
           "questions_per_run": N_QUESTIONS, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": runs}
    from germanqa.data.io_utils import write_json
    write_json(out, doc)
    print("đã ghi", out.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--target", type=int)
    ap.add_argument("--variant")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    worker(a.target, a.variant, a.batch) if a.worker else driver()
