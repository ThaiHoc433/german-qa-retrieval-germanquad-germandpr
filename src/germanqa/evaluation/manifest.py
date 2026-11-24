"""GĐ11.4: manifest của mốc B0/B1/H0 = đủ thông tin để biết một con số đo bằng cái gì và kiểm lại sau này.

Mỗi manifest có: SHA-1 của checkpoint reader, corpus, qrels, danh sách câu hỏi từng split, chỉ mục BM25 (dấu vân tay),
artifact dense, cấu hình hiệu lực (kể cả max_answer_tokens = 80, D5), thư viện, seed, tên GPU, phiên bản torch/CUDA,
precision. `verify_manifest` tính lại các SHA-1 và trả về danh sách chỗ lệch; test chạy nó trên mọi manifest."""
import json
import time
from pathlib import Path

from germanqa.config import PROCESSED_DIR, ROOT, SEED
from germanqa.data.io_utils import file_sha1, read_json
from germanqa.experiment import load_config
from germanqa.experiment_rules import CODES, baseline_id

READER_FILES = ("model.safetensors", "config.json", "reader_config.json", "vocab.txt")
SPLIT_FILES = [f"{k}_{s}.{e}" for s in ("train", "dev", "test") for k, e in (("qa", "jsonl"), ("qrels", "json"))]
DENSE_FILES = ("embeddings.npy", "index.faiss", "passage_ids.json", "manifest.json")
REFERENCE_RUNS = {   # mốc -> tệp kết quả đã lưu dùng làm số tham chiếu trên dev (toàn bộ 1.176 câu)
    "B0": "results/eval_dev_20261103T163556Z_cb4749bc.json",
    "B1": "results/eval_dev_dense_20261107T072837Z_b97b5c1e.json",
    "H0": "results/eval_dev_hybrid_20261107T135423Z_598ddf6f.json",
    "H1": "results/14f_H1_dev_unseen.json",     # GĐ14: chỉ 234 câu dev-unseen
}
RERANK_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model")
RERANK_RESULTS = ("14_prereg.json", "14_selection.json", "14_confirmation.json", "14a_probe.json", "14b_truncation.json",
                  "14c_scores_bge_m3.jsonl")


def sha_map(base, names):
    return {n: file_sha1(Path(base) / n) for n in names}


def environment_full(reader_fp16, dense_fp16=None):
    import platform
    from importlib import metadata
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "transformers", "tokenizers", "faiss", "faiss-cpu", "PyYAML"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch
        env.update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2) if torch.cuda.is_available() else None})
    except ImportError:
        pass
    env["precision"] = {"reader_fp16": reader_fp16, "dense_fp16": dense_fp16}
    return env


def dense_artifact_info(root, name):
    d = Path(root) / "artifacts" / "dense" / name
    man = read_json(d / "manifest.json")
    return {"dir": f"artifacts/dense/{name}", "sha1": sha_map(d, DENSE_FILES), "ctx_encoder": man["ctx_encoder"],
            "question_encoder": man["question_encoder"], "passage_input": man["passage_input"],
            "truncation": man["truncation"]}


def reference_run(root, code):
    rel = REFERENCE_RUNS[code]
    rec = json.loads((Path(root) / rel).read_text(encoding="utf-8-sig"))
    if code == "H1":                                   # tệp 14f: H1 đầu-cuối trên 234 câu dev-unseen
        conf = json.loads((Path(root) / "results" / "14_confirmation.json").read_text(encoding="utf-8-sig"))
        e2e = rec["H1"]["first_ref"]
        return {"file": rel, "sha1": file_sha1(Path(root) / rel), "split": "dev-unseen",
                "note": "CHỈ 234 câu dev-unseen (không phải toàn dev), đáp án tham chiếu đầu tiên",
                "end_to_end_first_ref": {k: e2e[k] for k in ("em", "f1", "queries")},
                "retrieval": {"hit@5": conf["h1"]["hit@5"], "hit@20": conf["h1"]["hit@20"], "source": "results/14_confirmation.json"}}
    m = rec["metrics"]
    if code == "B0":                                   # tệp 07a: số đo nằm thẳng trong metrics
        e2e, retr = m["end_to_end"]["all_refs"], m["retrieval"]
    else:                                              # tệp so sánh GĐ12/GĐ13: metrics['dev'][tên nhánh]
        key = {"B1": ("dense", "retrieval_dense"), "H0": ("hybrid", "retrieval_hybrid")}[code]
        e2e, retr = m["dev"][key[0]]["all_refs"], m["dev"][key[1]]
    return {"file": rel, "sha1": file_sha1(Path(root) / rel), "split": "dev",
            "note": "toàn bộ dev, mọi đáp án tham chiếu",
            "end_to_end_all_refs": {k: e2e[k] for k in ("em", "f1", "queries")},
            "retrieval": {k: retr[k] for k in ("hit@5", "hit@20") if k in retr}}


def build_manifest(code, root=ROOT, status_note=""):
    """Dựng manifest cho B0, B1 hoặc H0 từ tệp thật trong project."""
    if code not in ("B0", "B1", "H0", "H1"):
        raise ValueError("chỉ dựng B0, B1, H0, H1")
    root = Path(root)
    pcfg, rcfg = load_config("pipeline"), load_config("reader")["reader"]
    ck = root / pcfg["pipeline"]["checkpoint"]
    gold = read_json(root / "reports" / "reader_gold_context.json")
    d5 = {"effective_max_answer_tokens": pcfg["reader"]["max_answer_tokens"], "source": "configs/pipeline.yaml",
          "reader_yaml_max_answer_tokens": rcfg["max_answer_tokens"],
          "reader_gold_context_chosen": gold["chosen"]["max_answer_tokens"],
          "note": "reader.yaml (30) chỉ là mặc định lúc huấn luyện; pipeline dùng giá trị trong pipeline.yaml. "
                  "Không sửa reader.yaml hay reader_config.json."}
    cfg = {"code": code, "pipeline": pcfg, "reader_runtime": {k: rcfg[k] for k in ("max_length", "stride", "n_best", "eval_batch_size")}}
    man = {"code": code, "frozen_on": time.strftime("%Y-%m-%d"), "status": status_note, "seed": SEED,
           "reader": {"checkpoint": pcfg["pipeline"]["checkpoint"], "sha1": sha_map(ck, READER_FILES),
                      "d5_max_answer_tokens": d5},
           "data": {"corpus.jsonl": file_sha1(root / "data" / "processed" / "corpus.jsonl"),
                    "splits": sha_map(root / "data" / "processed", SPLIT_FILES),
                    "select_question_ids.json": file_sha1(root / "results" / "select_question_ids.json")},
           "reference_dev_run": reference_run(root, code)}
    reader_fp16 = True       # mặc định của Reader (xem Reader.__init__); script gọi có thể ghi đè bằng giá trị thật
    if code in ("B0", "H0", "H1"):
        b = pcfg["bm25"]
        from germanqa.retrieval import bm25_store
        fp = bm25_store.compute_fingerprint(man["data"]["corpus.jsonl"], b["k1"], b["b"], b["use_stopwords"])
        meta_p = bm25_store.DEFAULT_DIR if root == ROOT else root / "artifacts" / "bm25"
        idx = {"params": b, "fingerprint_sha1": fp["sha1"], "fingerprint_parts": fp["parts"]}
        if (meta_p / "meta.json").exists():
            meta = json.loads((meta_p / "meta.json").read_text(encoding="utf-8"))
            idx.update({"store": "artifacts/bm25", "index_npz_sha1": meta["index_npz_sha1"], "n_terms": meta["n_terms"],
                        "store_fingerprint_matches": meta["fingerprint"]["sha1"] == fp["sha1"]})
        man["bm25"] = idx
        cfg["retriever"] = "bm25" if code == "B0" else "hybrid"
    if code in ("B1", "H0", "H1"):
        dcfg = load_config("dense")["dense"]
        man["dense"] = {"config": dcfg, "artifact": dense_artifact_info(root, Path(dcfg["artifact_dir"]).name)}
        cfg["dense"] = dcfg
        if code == "B1":
            cfg["retriever"] = "dense"
    if code in ("H0", "H1"):
        hcfg = load_config("hybrid")["hybrid"]
        man["hybrid"] = {"config": hcfg, "prereg_sha1": sha_map(root / "results", ("13_prereg.json", "13_selection.json", "13_confirmation.json", "rrf_sweep.json"))}
        cfg["hybrid"] = hcfg
    if code == "H1":
        rrc = load_config("rerank")["rerank"]
        ckd = root / rrc["checkpoint"]
        man["rerank"] = {"config": rrc, "checkpoint": rrc["checkpoint"], "sha1": sha_map(ckd, RERANK_FILES),
                         "results_sha1": sha_map(root / "results", RERANK_RESULTS),
                         "note": "H1 = H0 + cross-encoder xếp lại 20 đoạn, reader nhận 5 đầu; không fine-tune"}
        cfg["rerank"] = rrc
    man["effective_config"] = cfg
    man["config_id"] = baseline_id(code, cfg)
    man["environment"] = environment_full(reader_fp16, man.get("dense", {}).get("config", {}).get("fp16"))
    return man


def verify_manifest(man, root=ROOT):
    """Tính lại SHA-1 từ tệp thật. Trả về danh sách mô tả chỗ lệch (rỗng = khớp)."""
    return verify_manifest_detail(man, root)[0]


def verify_manifest_detail(man, root=ROOT):
    """Như verify_manifest nhưng trả về (danh sách lệch, số SHA-1 đã kiểm). Số 0 nghĩa là manifest không kiểm được gì:
    'rỗng = khớp' mà không kiểm gì thì vô nghĩa, nên test đòi số này lớn hơn 0."""
    root, bad, cnt = Path(root), [], [0]
    def chk(label, path, want):
        cnt[0] += 1
        p = root / path
        if not p.exists():
            bad.append(f"{label}: thiếu {path}")
        elif file_sha1(p) != want:
            bad.append(f"{label}: SHA-1 của {path} đã đổi")
    if "reader" in man:
        ck = man["reader"]["checkpoint"]
        for n, h in man["reader"]["sha1"].items():
            chk(f"reader/{n}", f"{ck}/{n}", h)
    if "data" in man:
        chk("corpus", "data/processed/corpus.jsonl", man["data"]["corpus.jsonl"])
        for n, h in man["data"]["splits"].items():
            chk(f"split/{n}", f"data/processed/{n}", h)
        chk("select_ids", "results/select_question_ids.json", man["data"]["select_question_ids.json"])
    for key in ("dense",):
        if key in man:
            a = man[key]["artifact"]
            for n, h in a["sha1"].items():
                chk(f"dense/{n}", f"{a['dir']}/{n}", h)
    if "bm25" in man and "index_npz_sha1" in man["bm25"]:
        chk("bm25/index.npz", "artifacts/bm25/index.npz", man["bm25"]["index_npz_sha1"])
    if "hybrid" in man:
        for n, h in man["hybrid"]["prereg_sha1"].items():
            chk(f"hybrid/{n}", f"results/{n}", h)
    if "rerank" in man:
        rk = man["rerank"]
        for n, h in rk["sha1"].items():
            chk(f"rerank/{n}", f"{rk['checkpoint']}/{n}", h)
        for n, h in rk["results_sha1"].items():
            chk(f"rerank_results/{n}", f"results/{n}", h)
    if "reference_dev_run" in man:
        r = man["reference_dev_run"]
        chk("reference_dev_run", r["file"], r["sha1"])
    if "artifacts" in man:                                    # dạng cũ của B1.json (GĐ12.5)
        for name, a in man["artifacts"].items():
            for f, h in a["sha1"].items():
                chk(f"dense/{name}/{f}", f"artifacts/dense/{name}/{f}", h)
        if "corpus_sha1" in man:
            chk("corpus", "data/processed/corpus.jsonl", man["corpus_sha1"])
        if "select_ids_sha1" in man:
            chk("select_ids", "results/select_question_ids.json", man["select_ids_sha1"])
        for f, h in man.get("top100_files", {}).items():
            chk(f"top100/{f}", f"results/{f}", h)
    return bad, cnt[0]


def load_manifest(code, root=ROOT):
    return json.loads((Path(root) / "baselines" / f"{code}.json").read_text(encoding="utf-8-sig"))
