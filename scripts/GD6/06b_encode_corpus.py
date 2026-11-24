"""GĐ6.3: mã hóa toàn bộ corpus bằng context encoder, lưu embeddings + passage_ids + index + manifest."""
import argparse
import time

import numpy as np

from germanqa.config import PROCESSED_DIR, ROOT
from germanqa.data.io_utils import file_sha1, read_jsonl
from germanqa.experiment import load_config
from germanqa.retrieval.dense import DPREncoder, save_artifacts


def main(a):
    cfg = load_config("dense")["dense"]
    field = a.field or cfg["passage_field"]
    pin = a.input or cfg["passage_input"]
    max_len = a.max_length or cfg["passage_max_length"]
    out = ROOT / (a.out or cfg["artifact_dir"])
    corpus_path = PROCESSED_DIR / "corpus.jsonl"
    corpus = read_jsonl(corpus_path)
    if a.limit:
        corpus = corpus[:a.limit]
    ids = [r["passage_id"] for r in corpus]

    enc = DPREncoder(cfg["ctx_encoder"], cfg["question_encoder"], fp16=cfg["fp16"])
    t0 = time.perf_counter()
    vecs, stats = enc.encode_passages([r["title"] for r in corpus], [r[field] for r in corpus], pin, max_len,
                                      cfg["encode_batch_size"])
    secs = time.perf_counter() - t0
    import faiss, torch, transformers
    manifest = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), **enc.revisions(),
                "passage_field": field, "passage_input": pin, "similarity": cfg["similarity"], "normalized": False,
                "n_passages": len(ids), "dim": int(vecs.shape[1]), "dtype": str(vecs.dtype),
                "corpus_sha1": file_sha1(corpus_path), "corpus_limit": a.limit, "truncation": stats,
                "encode_seconds": secs, "passages_per_second": len(ids) / secs, "fp16": enc.fp16,
                "device": str(enc.device), "vector_norm_mean": float(np.linalg.norm(vecs, axis=1).mean()),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "faiss": faiss.__version__}}
    save_artifacts(out, vecs, ids, manifest)
    print(f"đã ghi {out}  ({len(ids)} đoạn, {secs:.0f}s, {len(ids)/secs:.1f} đoạn/s, "
          f"cắt cụt {stats['n_truncated']} = {stats['frac_truncated']:.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="thư mục artifacts, tính từ gốc dự án")
    ap.add_argument("--input", choices=("title_text", "text"), default=None)
    ap.add_argument("--field", choices=("original_text", "retrieval_text"), default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ N đoạn đầu (chạy thử)")
    main(ap.parse_args())
