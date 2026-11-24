"""GĐ8.8: đo API thật. Hai giai đoạn, chạy lần lượt (GPU 4 GB không đủ cho hai bản reader cùng lúc nếu tránh được):
  --stage cli : chạy pipeline TRỰC TIẾP (đường code của CLI) trên N câu dev, ghi đáp án ra reports/api_parity_expected.json
  --stage api : khởi động máy chủ thật, đo khởi động/RAM/VRAM, so đáp án với giai đoạn cli, đo độ trễ khi 1/2/4 request
                đồng thời, health khi tải, và khởi động lại lần hai.
Kết quả: reports/api_measurements.json."""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import psutil

from germanqa.config import PROCESSED_DIR, REPORT_DIR
from germanqa.data.io_utils import read_json, write_json
from germanqa.reader.features import load_examples

EXPECTED = REPORT_DIR / "api_parity_expected.json"
OUT = REPORT_DIR / "api_measurements.json"
BASE = "http://127.0.0.1:8765"


def stage_cli(checkpoint, n):
    from germanqa.pipeline import build_pipeline
    qs = [e["question"] for e in load_examples(PROCESSED_DIR, "dev")[:n]]
    out = {}
    for name in ("bm25", "dense"):
        pipe = build_pipeline(checkpoint, retriever=name)
        pipe.answer("Wer war Konrad Adenauer?")
        out[name] = [(lambda r: {"answer": r.answer, "passage_id": r.passage_id, "start": r.answer_start,
                                 "end": r.answer_end, "status": r.status, "reader_score": r.reader_score,
                                 "retrieval_score": r.retrieval_score})(pipe.answer(q)) for q in qs]
        del pipe
    write_json(EXPECTED, {"questions": qs, "expected": out})
    print("đã ghi", EXPECTED)


def gpu_used_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:                              # noqa: BLE001
        return None


def start_server(checkpoint, log_path):
    t0 = time.perf_counter()
    p = subprocess.Popen([sys.executable, str(Path(__file__).with_name("08_serve.py")), "--port", "8765",
                          "--checkpoint", checkpoint, "--retrievers", "bm25,dense"],
                         stdout=open(log_path, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    t_live = t_ready = None
    while time.perf_counter() - t0 < 300:
        try:
            if t_live is None and httpx.get(f"{BASE}/health/live", timeout=2).status_code == 200:
                t_live = time.perf_counter() - t0
            r = httpx.get(f"{BASE}/health/ready", timeout=2)
            if r.status_code == 200:
                t_ready = time.perf_counter() - t0
                break
            if r.json().get("status") == "failed":
                raise RuntimeError("server failed: " + r.text)
        except httpx.TransportError:
            pass
        time.sleep(0.25)
    if t_ready is None:
        p.kill()
        raise RuntimeError("server không ready trong 300 s")
    return p, t_live, t_ready


def rss_mb(proc):
    ps = psutil.Process(proc.pid)
    return round(sum(c.memory_info().rss for c in [ps] + ps.children(recursive=True)) / 2**20)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def load_level(questions, conc, retriever):
    lat, codes, lock, idx = [], [], threading.Lock(), iter(range(len(questions)))
    health = []
    stop = threading.Event()

    def watch():
        with httpx.Client(timeout=5) as hc:      # kết nối bền: httpx.get() dựng client mới mỗi lần, tốn ~270 ms ở phía client kể cả khi máy chủ rảnh
            while not stop.is_set():
                t = time.perf_counter()
                try:
                    hc.get(f"{BASE}/health/live")
                    health.append((time.perf_counter() - t) * 1000)
                except httpx.TransportError:
                    health.append(float("inf"))
                time.sleep(0.1)

    def worker():
        with httpx.Client(timeout=60) as c:
            while True:
                with lock:
                    i = next(idx, None)
                if i is None:
                    return
                t = time.perf_counter()
                r = c.post(f"{BASE}/api/v1/qa", json={"question": questions[i], "retriever": retriever})
                with lock:
                    lat.append((time.perf_counter() - t) * 1000)
                    codes.append(r.status_code)

    w = threading.Thread(target=watch)
    w.start()
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(conc)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.perf_counter() - t0
    stop.set()
    w.join()
    return {"concurrency": conc, "requests": len(lat), "status_codes": {str(c): codes.count(c) for c in set(codes)},
            "latency_ms": {"mean": round(statistics.mean(lat), 1), "p50": round(pct(lat, .5), 1),
                           "p95": round(pct(lat, .95), 1), "max": round(max(lat), 1)},
            "throughput_rps": round(len(lat) / wall, 2),
            "health_live_ms": {"p50": round(pct(health, .5), 1), "max": round(max(health), 1)}}


def stage_api(checkpoint, n_load):
    exp = read_json(EXPECTED)
    qs = exp["questions"]
    out = {"parity_questions": len(qs), "gpu_used_mb_before": gpu_used_mb()}
    p, t_live, t_ready = start_server(checkpoint, REPORT_DIR / "api_server_run1.log")
    try:
        out["startup"] = {"live_s": round(t_live, 2), "ready_s": round(t_ready, 2)}
        out["rss_mb_after_ready"], out["gpu_used_mb_after_ready"] = rss_mb(p), gpu_used_mb()
        ready = httpx.get(f"{BASE}/health/ready").json()
        out["ready_body"] = ready
        with httpx.Client(timeout=60) as c:
            parity = {}
            for name in ("bm25", "dense"):
                diffs = 0
                for q, e in zip(qs, exp["expected"][name]):
                    b = c.post(f"{BASE}/api/v1/qa", json={"question": q, "retriever": name}).json()
                    got = {"answer": b["answer"], "passage_id": b["passage_id"], "start": b["answer_start"],
                           "end": b["answer_end"], "status": b["status"]}
                    want = {k: e[k] for k in got}
                    scores_ok = (abs(b["reader_score"] - e["reader_score"]) < 1e-4
                                 and abs(b["retrieval_score"] - e["retrieval_score"]) < 1e-3)
                    diffs += int(got != want or not scores_ok)
                parity[name] = {"questions": len(qs), "different": diffs}
            out["parity_api_vs_cli"] = parity
            one = c.post(f"{BASE}/api/v1/qa", json={"question": qs[0], "retriever": "dense"}).json()
            out["restart_probe_run1"] = {k: one[k] for k in ("answer", "passage_id", "versions")}
        dev_qs = (qs * 3)[:n_load]
        out["load"] = {r: [load_level(dev_qs, c, r) for c in (1, 2, 4)] for r in ("bm25", "dense")}
        out["rss_mb_after_load"], out["gpu_used_mb_after_load"] = rss_mb(p), gpu_used_mb()
    finally:
        p.kill()
        p.wait()
    time.sleep(3)
    p2, t_live2, t_ready2 = start_server(checkpoint, REPORT_DIR / "api_server_run2.log")
    try:
        with httpx.Client(timeout=60) as c:
            one2 = c.post(f"{BASE}/api/v1/qa", json={"question": qs[0], "retriever": "dense"}).json()
        def stable(r):      # experiment_id có mốc thời gian (GĐ0) nên đổi mỗi lần khởi động; so phần còn lại
            return {"answer": r["answer"], "passage_id": r["passage_id"], "model": r["versions"]["model"],
                    "corpus": r["versions"]["corpus"]}
        out["restart"] = {"ready_s": round(t_ready2, 2),
                          "same_answer_model_corpus": stable(one2) == stable(out["restart_probe_run1"]),
                          "experiment_id_run1": out["restart_probe_run1"]["versions"]["experiment_id"],
                          "experiment_id_run2": one2["versions"]["experiment_id"]}
    finally:
        p2.kill()
        p2.wait()
    write_json(OUT, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cli", "api"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-load", type=int, default=60)
    a = ap.parse_args()
    stage_cli(a.checkpoint, a.n) if a.stage == "cli" else stage_api(a.checkpoint, a.n_load)
