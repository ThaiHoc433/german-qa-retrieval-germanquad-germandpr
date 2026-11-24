"""GĐ12.5: ghi baselines/B1.json = mốc B1 tái lập được (SHA-1 artifact, revision encoder, phiên bản thư viện, seed, quyết định 384)."""
import platform

from germanqa.config import PROCESSED_DIR, RESULTS_DIR, ROOT, SEED
from germanqa.data.io_utils import file_sha1, read_json, write_json
from germanqa.evaluation.decision import adopt_alternative
from germanqa.experiment import load_config

ART_FILES = ("embeddings.npy", "index.faiss", "passage_ids.json", "manifest.json")


def art_info(name):
    d = ROOT / "artifacts" / "dense" / name
    man = read_json(d / "manifest.json")
    return {"sha1": {f: file_sha1(d / f) for f in ART_FILES}, "ctx_encoder": man["ctx_encoder"],
            "question_encoder": man["question_encoder"], "truncation": man["truncation"],
            "passage_input": man["passage_input"], "passage_field": man["passage_field"]}


def main():
    cfg = load_config("dense")["dense"]
    an = read_json(RESULTS_DIR / "12b_analysis.json")["C_paired_384_minus_256"]
    adopt, why = adopt_alternative((an["hit@5"]["diff"], an["hit@5"]["lo"], an["hit@5"]["hi"]),
                                   (an["hit@20"]["diff"], an["hit@20"]["lo"], an["hit@20"]["hi"]), margin=0.01)
    import faiss, tokenizers, torch, transformers
    man = {"baseline": "B1", "seed": SEED, "dense_config": cfg, "corpus_sha1": file_sha1(PROCESSED_DIR / "corpus.jsonl"),
           "artifacts": {n: art_info(n) for n in ("text256", "text384")},
           "length_decision": {"rule": "Hit@5 tăng với khoảng tin cậy ghép cặp > 0, và Hit@20 cận dưới >= -0.01",
                               "paired_384_minus_256": {k: an[k] for k in ("hit@5", "hit@20")}, "adopt_text384": adopt, "reason": why,
                               "status": "ĐỀ XUẤT: đổi artifact_dir sang text384 khi người dùng xác nhận"},
           "top100_files": {f: file_sha1(RESULTS_DIR / f) for f in ("B0_select_top100.jsonl", "B1_select_top100.jsonl", "B1_select_top100_384.jsonl")},
           "select_ids_sha1": file_sha1(RESULTS_DIR / "select_question_ids.json"),
           "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                           "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "faiss": faiss.__version__,
                           "cuda_available": torch.cuda.is_available(),
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    write_json(ROOT / "baselines" / "B1.json", man)
    print("baselines/B1.json:", man["length_decision"])


if __name__ == "__main__":
    main()
