"""GĐ6.4: kiểm tra artifacts và index. Không tin FAISS mù quáng: đối chiếu với tính thẳng bằng numpy,
và đối chiếu vector fp16 đã lưu với tính lại bằng fp32."""
import argparse

import faiss
import numpy as np

from germanqa.config import PROCESSED_DIR, REPORT_DIR, ROOT
from germanqa.data.io_utils import read_jsonl, write_json
from germanqa.experiment import load_config
from germanqa.retrieval.dense import DPREncoder, build_index, exact_topk, load_artifacts


def main(artifact, n_questions, n_passages, out_name="dense_index_check.json"):
    cfg = load_config("dense")["dense"]
    d = ROOT / artifact
    vecs, ids, man = load_artifacts(d, PROCESSED_DIR / "corpus.jsonl")
    out = {"artifact": artifact, "n_vectors": len(ids), "dim": int(vecs.shape[1]),
           "finite": bool(np.isfinite(vecs).all()), "ids_unique": len(set(ids)) == len(ids),
           "norm": {"min": float(np.linalg.norm(vecs, axis=1).min()), "mean": float(np.linalg.norm(vecs, axis=1).mean()),
                    "max": float(np.linalg.norm(vecs, axis=1).max())},
           "bytes": {"embeddings": (d / "embeddings.npy").stat().st_size, "index": (d / "index.faiss").stat().st_size}}
    corpus = read_jsonl(PROCESSED_DIR / "corpus.jsonl")
    out["ids_match_corpus_order"] = ids == [r["passage_id"] for r in corpus]

    rng = np.random.default_rng(42)
    enc = DPREncoder(cfg["ctx_encoder"], cfg["question_encoder"], fp16=cfg["fp16"])
    qs = read_jsonl(PROCESSED_DIR / "qa_dev.jsonl")
    pick = rng.choice(len(qs), n_questions, replace=False)
    q = enc.encode_questions([qs[i]["question"] for i in pick], cfg["question_max_length"])

    # 1) FAISS (build lại) = FAISS (đọc từ file) = numpy
    s_f, i_f = build_index(vecs).search(q, 20)
    s_d, i_d = faiss.read_index(str(d / "index.faiss")).search(q, 20)
    s_e, i_e = exact_topk(q, vecs, 20)
    differ = ~(i_f == i_e).all(axis=1)
    # hàng có ID khác nhau chỉ chấp nhận được nếu hai bên hòa điểm: điểm xếp theo hạng phải gần như bằng nhau
    gap = np.abs(s_f - s_e).max(axis=1)
    out["faiss_vs_numpy"] = {"questions": n_questions, "identical_top20_ids": int((~differ).sum()),
                             "rows_with_different_ids": int(differ.sum()),
                             "max_score_gap_in_those_rows": float(gap[differ].max()) if differ.any() else 0.0,
                             "same_id_set_in_those_rows": int(sum(set(a) == set(b) for a, b in zip(i_f[differ], i_e[differ]))),
                             "max_abs_score_diff": float(np.abs(s_f - s_e).max()),
                             "file_index_equals_rebuilt": bool((i_f == i_d).all() and np.allclose(s_f, s_d))}

    # 2) vector fp16 đã lưu so với tính lại fp32 trên một mẫu đoạn
    sel = rng.choice(len(corpus), n_passages, replace=False)
    enc32 = DPREncoder(cfg["ctx_encoder"], cfg["question_encoder"], fp16=False)
    v32, _ = enc32.encode_passages([corpus[i]["title"] for i in sel], [corpus[i][man["passage_field"]] for i in sel],
                                   man["passage_input"], man["truncation"]["max_length"], cfg["encode_batch_size"], 0)
    v16 = vecs[sel]
    cos = (v32 * v16).sum(1) / (np.linalg.norm(v32, axis=1) * np.linalg.norm(v16, axis=1))
    qq = q[:100]
    top16 = np.argsort(-(qq @ v16.T), axis=1)[:, :5]
    top32 = np.argsort(-(qq @ v32.T), axis=1)[:, :5]
    out["fp16_vs_fp32"] = {"passages": n_passages, "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
                           "max_abs_diff": float(np.abs(v32 - v16).max()),
                           "top1_same": float((top16[:, 0] == top32[:, 0]).mean()),
                           "top5_set_same": float(np.mean([set(a) == set(b) for a, b in zip(top16, top32)]))}
    write_json(REPORT_DIR / out_name, out)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None, help="mặc định: artifact_dir trong configs/dense.yaml")
    ap.add_argument("--questions", type=int, default=200)
    ap.add_argument("--passages", type=int, default=300)
    ap.add_argument("--out", default="dense_index_check.json", help="tên tệp báo cáo trong reports/ (đặt tên khác để không đè báo cáo cũ)")
    a = ap.parse_args()
    main(a.artifact or load_config("dense")["dense"]["artifact_dir"], a.questions, a.passages, a.out)
