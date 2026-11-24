"""GĐ4.4: đánh giá reader với đoạn ĐÚNG (gold-context). Chọn max_answer_tokens và score_mode trên dev.
Test chỉ chạy khi có --test, sau khi đã chốt trên dev."""
import argparse

from germanqa.config import PROCESSED_DIR, REPORT_DIR
from germanqa.data.io_utils import write_json
from germanqa.experiment import load_config
from germanqa.experiment_rules import guard_or_exit
from germanqa.reader.features import load_examples
from germanqa.reader.inference import Reader
from germanqa.reader.qa_metrics import evaluate_answers


def by_windows(examples, feats, preds):
    """EM/F1 chia theo số cửa sổ của câu hỏi: 1, 2, 3 trở lên (báo cáo kết quả theo độ dài context)."""
    count = {}
    for f in feats:
        count[f["example_index"]] = count.get(f["example_index"], 0) + 1
    groups = {"1 cửa sổ": [], "2 cửa sổ": [], "3+ cửa sổ": []}
    for i, ex in enumerate(examples):
        groups["1 cửa sổ" if count[i] == 1 else "2 cửa sổ" if count[i] == 2 else "3+ cửa sổ"].append(ex)
    return {name: evaluate_answers(preds, exs) for name, exs in groups.items() if exs}


def main(checkpoint, do_test):
    if do_test:
        guard_or_exit("04d_eval_reader.py", "reader gold-context trên test với cấu hình đã chốt trên dev", True, files=["qa_test.jsonl"], codes=["B0"])
    cfg = load_config("reader")["reader"]
    reader = Reader(checkpoint, cfg["max_length"], cfg["stride"], cfg["n_best"],
                    batch_size=cfg["eval_batch_size"])

    def logits_for(split):
        ex = load_examples(PROCESSED_DIR, split)
        feats, s, e = reader.logits(ex)
        return ex, feats, s, e

    ex, feats, s, e = logits_for("dev")                    # chạy mô hình MỘT lần, hậu xử lý nhiều lần
    trials = []
    for mode in ("raw", "minus_cls"):
        for m in cfg["answer_grid"]:
            preds = reader.decode(ex, feats, s, e, m, mode)
            r = evaluate_answers(preds, ex)
            trials.append({"score_mode": mode, "max_answer_tokens": m, **r})
            print(f"{mode:9} max_answer_tokens={m:<3} EM {r['em']:.4f}  F1 {r['f1']:.4f}")
    best = max(trials, key=lambda t: (t["em"] + t["f1"]))
    print("CHỌN:", {k: best[k] for k in ("score_mode", "max_answer_tokens")})

    result = {"checkpoint": str(checkpoint), "chosen": {k: best[k] for k in ("score_mode", "max_answer_tokens")},
              "dev": {k: best[k] for k in ("queries", "em", "f1")}, "dev_trials": trials}
    dev_preds = reader.decode(ex, feats, s, e, best["max_answer_tokens"], best["score_mode"])
    result["dev_by_windows"] = by_windows(ex, feats, dev_preds)
    print("DEV theo số cửa sổ:", {k: {a: round(b, 4) for a, b in v.items()} for k, v in result["dev_by_windows"].items()})
    if do_test:
        ex_t, feats_t, s_t, e_t = logits_for("test")
        preds = reader.decode(ex_t, feats_t, s_t, e_t, best["max_answer_tokens"], best["score_mode"])
        result["test"] = evaluate_answers(preds, ex_t)
        result["test_by_windows"] = by_windows(ex_t, feats_t, preds)
        print("TEST theo số cửa sổ:", {k: {a: round(b, 4) for a, b in v.items()} for k, v in result["test_by_windows"].items()})
        print("TEST", {k: round(v, 4) for k, v in result["test"].items()})
    write_json(REPORT_DIR / "reader_gold_context.json", result)
    print("đã ghi", REPORT_DIR / "reader_gold_context.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    main(a.checkpoint, a.test)
