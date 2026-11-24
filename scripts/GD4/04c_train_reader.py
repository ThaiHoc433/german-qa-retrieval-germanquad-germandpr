
"""GĐ4.3: fine-tune reader từ backbone tiền huấn luyện trên QA train; chọn checkpoint theo dev."""
import argparse
import json
import shutil
import time

import torch

from germanqa.config import PROCESSED_DIR, ROOT, SEED
from germanqa.experiment import load_config, new_experiment_id, save_run, set_seed
from germanqa.reader.features import load_examples, prepare_eval_features, prepare_train_features
from germanqa.reader.model_io import get_model, get_tokenizer, resolve_dir
from germanqa.reader.trainer import set_all_seeds, train


def main(limit):
    cfg = load_config("reader")["reader"]
    set_seed(SEED)
    set_all_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

    tok = get_tokenizer(cfg["model_name"])
    train_ex = load_examples(PROCESSED_DIR, "train")[:limit]
    dev_ex = load_examples(PROCESSED_DIR, "dev")[:limit]
    train_f = prepare_train_features(train_ex, tok, cfg["max_length"], cfg["stride"])
    dev_f = prepare_eval_features(dev_ex, tok, cfg["max_length"], cfg["stride"])
    log(f"train {len(train_ex)} câu -> {len(train_f)} cửa sổ; dev {len(dev_ex)} câu -> {len(dev_f)} cửa sổ; thiết bị {device}")

    model = get_model(cfg["model_name"])
    state, history = train(model, train_f, dev_ex, dev_f, cfg, SEED, device, log)

    exp_id = new_experiment_id("reader", cfg)
    out_dir = ROOT / "models" / exp_id
    model.load_state_dict(state)
    model.save_pretrained(out_dir)
    shutil.copy(resolve_dir(cfg["model_name"]) / "vocab.txt", out_dir / "vocab.txt")   # tokenizer = file từ vựng
    (out_dir / "reader_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True),
                                                encoding="utf-8")
    path = save_run(exp_id, cfg, {"history": history}, extra={"checkpoint": str(out_dir)})
    log(f"đã lưu checkpoint {out_dir} và kết quả {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="chỉ dùng N câu đầu (chạy thử)")
    main(ap.parse_args().limit)