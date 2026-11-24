"""GĐ4.1: kiểm tra luồng bằng một checkpoint QA có sẵn (không phải đánh giá độc lập)."""
import argparse
import time

import torch

from germanqa.config import PROCESSED_DIR
from germanqa.experiment import load_config
from germanqa.reader.features import load_examples, prepare_eval_features
from germanqa.reader.model_io import get_model, get_tokenizer
from germanqa.reader.trainer import evaluate


def main(model_name, n):
    cfg = load_config("reader")["reader"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = get_tokenizer(model_name), get_model(model_name).to(device)
    examples = load_examples(PROCESSED_DIR, "dev")[:n]
    feats = prepare_eval_features(examples, tok, cfg["max_length"], cfg["stride"])
    t0 = time.time()
    metrics, preds = evaluate(model, examples, feats, cfg["n_best"], cfg["max_answer_tokens"],
                              cfg["eval_batch_size"], device, fp16=True)
    print(f"{model_name}: {len(examples)} câu, {len(feats)} cửa sổ, {time.time() - t0:.1f}s trên {device}")
    print({k: round(v, 4) for k, v in metrics.items()})
    for ex in examples[:5]:
        p = preds[ex["id"]]
        ok = ex["context"][p["start"]:p["end"]] == p["text"]
        print("-" * 70)
        print("Hỏi    :", ex["question"])
        print("Gold   :", ex["answers"][0]["text"].strip())
        print("Dự đoán:", p["text"], f"[{p['start']}:{p['end']}] cắt đúng từ gốc={ok}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepset/gelectra-base-germanquad")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    main(a.model, a.n)
