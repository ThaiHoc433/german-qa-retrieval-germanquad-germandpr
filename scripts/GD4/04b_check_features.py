"""GĐ4.2: kiểm tra nhãn cửa sổ bằng mắt và bằng số. Không cần GPU, không cần tải trọng số."""
import argparse

from germanqa.config import PROCESSED_DIR
from germanqa.experiment import load_config
from germanqa.reader.features import answer_chars, load_examples, prepare_train_features
from germanqa.reader.model_io import get_tokenizer
from germanqa.reader.qa_metrics import normalize


def main(split):
    cfg = load_config("reader")["reader"]
    tok = get_tokenizer(cfg["model_name"])
    examples = load_examples(PROCESSED_DIR, split)
    feats = prepare_train_features(examples, tok, cfg["max_length"], cfg["stride"], keep_offsets=True)
    n_ex = len(examples)
    per_ex = {}
    for f in feats:
        per_ex.setdefault(f["example_index"], []).append(f)

    cls_windows = sum(1 for f in feats if f["start_positions"] == 0)
    no_answer = sum(1 for fs in per_ex.values() if all(f["start_positions"] == 0 for f in fs))
    multi = sum(1 for fs in per_ex.values() if len(fs) > 1)
    print(f"[{split}] {n_ex} câu -> {len(feats)} cửa sổ ({len(feats) / n_ex:.2f}/câu)")
    print(f"  câu cần >1 cửa sổ: {multi} ({100 * multi / n_ex:.1f}%)")
    print(f"  cửa sổ nhãn [CLS] (không chứa đủ đáp án): {cls_windows} ({100 * cls_windows / len(feats):.1f}%)")
    print(f"  câu KHÔNG có cửa sổ nào chứa đáp án: {no_answer}")

    bad, checked = [], 0
    for f in feats:
        if f["start_positions"] == 0:
            continue
        ex = examples[f["example_index"]]
        s, e = answer_chars(ex["answers"][0])
        a = f["offsets"][f["start_positions"]][0]
        b = f["offsets"][f["end_positions"]][1]
        checked += 1
        if normalize(ex["context"][a:b]) != normalize(ex["context"][s:e]) or f["end_positions"] < f["start_positions"]:
            bad.append((ex["question"], ex["context"][s:e], ex["context"][a:b]))
    print(f"  nhãn có span: {checked}; span cắt lại khác đáp án gốc: {len(bad)}")
    for q, gold, got in bad[:5]:
        print("   -", repr(gold), "<>", repr(got))
    ex_i = next(i for i, fs in per_ex.items() if len(fs) > 1)
    print("Ví dụ câu có nhiều cửa sổ:", examples[ex_i]["question"])
    for k, f in enumerate(per_ex[ex_i]):
        lab = "[CLS]" if f["start_positions"] == 0 else f"token {f['start_positions']}..{f['end_positions']}"
        print(f"   cửa sổ {k}: {len(f['input_ids'])} token, nhãn {lab}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    main(ap.parse_args().split)
