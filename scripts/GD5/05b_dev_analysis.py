"""GĐ5.7: phân tích end-to-end trên dev: số đoạn đưa vào reader, raw so với minus_cls, và đoạn đúng có trong top-5 hay không.

Mỗi chế độ chỉ chạy pipeline MỘT lần với reader_top_k = 10. Điểm span của từng đoạn không phụ thuộc đoạn đi kèm
(có test), nên kết quả top-1, 3, 5 suy ra chính xác bằng cách chỉ xét k đoạn đầu của danh sách ứng viên."""
import argparse

import numpy as np

from germanqa.config import PROCESSED_DIR, REPORT_DIR
from germanqa.data.io_utils import read_json, write_json
from germanqa.pipeline import build_pipeline
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import evaluate_answers, exact_match

KS = (1, 3, 5, 10)


def choose(candidates, k):
    """Giống QAPipeline.answer: span có điểm reader cao nhất trong k đoạn đầu; hòa điểm giữ đoạn hạng cao hơn."""
    valid = [c for c in candidates[:k] if c["start"] >= 0]
    return max(valid, key=lambda c: c["reader_score"]) if valid else None


def main(checkpoint, limit):
    pipe = build_pipeline(checkpoint)
    examples = load_examples(PROCESSED_DIR, "dev")[:limit]
    qrels = read_json(PROCESSED_DIR / "qrels_dev.json")
    pipe.answer(examples[0]["question"])

    runs, reader_ms = {}, {}
    for mode in ("raw", "minus_cls"):
        pipe.reader.score_mode = mode
        rows, times = {}, []
        for ex in examples:
            r = pipe.answer(ex["question"], reader_top_k=10, trace=True)
            rows[ex["id"]] = r.trace
            times.append(r.reader_time * 1000)
        runs[mode], reader_ms[mode] = rows, float(np.mean(times))
        print(mode, "xong; reader_time trung bình với 10 đoạn:", round(reader_ms[mode], 1), "ms", flush=True)

    table = []
    for mode in runs:
        for k in KS:
            preds = {q: {"text": (choose(t["candidates"], k) or {"answer": ""})["answer"]} for q, t in runs[mode].items()}
            table.append({"score_mode": mode, "reader_top_k": k, **evaluate_answers(preds, examples)})
            print(f"{mode:9} k={k:<2} EM {table[-1]['em']:.4f}  F1 {table[-1]['f1']:.4f}")

    # ---- cấu hình đã chọn (minus_cls, k = 5): tách theo đoạn đúng có trong top-5 hay không ----
    t_all = runs["minus_cls"]
    gold = {ex["id"]: set(qrels[ex["id"]]) for ex in examples}
    pred = {q: choose(t["candidates"], 5) for q, t in t_all.items()}
    em = {ex["id"]: max(exact_match(pred[ex["id"]]["answer"] if pred[ex["id"]] else "", a["text"]) for a in ex["answers"])
          for ex in examples}

    def where(ex):
        ids = [h["passage_id"] for h in t_all[ex["id"]]["retrieved"]]
        g = gold[ex["id"]]
        return "in_top5" if g & set(ids[:5]) else "in_top20_not_top5" if g & set(ids[:20]) else "not_in_top20"

    groups = {}
    for ex in examples:
        groups.setdefault(where(ex), []).append(ex)

    # reader nhận đúng đoạn đúng (mốc GĐ4) cho cùng các câu hỏi, để so sánh trong nhóm in_top5
    reader = pipe.reader
    feats, s, e = reader.logits(examples)
    gold_preds = reader.decode(examples, feats, s, e)
    em_gold = {ex["id"]: max(exact_match(gold_preds[ex["id"]]["text"], a["text"]) for a in ex["answers"]) for ex in examples}

    split = {}
    for name in ("in_top5", "in_top20_not_top5", "not_in_top20"):
        exs = groups.get(name, [])
        ids = [x["id"] for x in exs]
        preds = {q: {"text": pred[q]["answer"] if pred[q] else ""} for q in ids}
        split[name] = {**evaluate_answers(preds, exs), "share": len(exs) / len(examples),
                       "em_if_reader_gets_gold_passage": float(np.mean([em_gold[q] for q in ids]))}
    top5 = groups["in_top5"]
    picked_gold = [x for x in top5 if pred[x["id"]] and pred[x["id"]]["passage_id"] in gold[x["id"]]]
    split["in_top5_selected_gold_passage"] = {
        "queries": len(picked_gold), "share_of_in_top5": len(picked_gold) / len(top5),
        "em": float(np.mean([em[x["id"]] for x in picked_gold])),
        "em_selected_other_passage": float(np.mean([em[x["id"]] for x in top5 if x not in picked_gold]))}

    wrong = [x for x in examples if em[x["id"]] == 0]
    taxonomy = {"wrong_total": len(wrong),
                "retrieval_failure_not_in_top20": sum(where(x) == "not_in_top20" for x in wrong),
                "selection_failure_in_top20_not_top5": sum(where(x) == "in_top20_not_top5" for x in wrong),
                "in_top5_reader_picked_other_passage": sum(where(x) == "in_top5" and x not in picked_gold for x in wrong),
                "in_top5_reader_picked_gold_passage_wrong_span": sum(x in picked_gold for x in wrong)}
    for name, v in split.items():
        print(name, {k: round(x, 4) for k, x in v.items()})
    print("PHÂN LOẠI CÂU SAI (k=5, minus_cls):", taxonomy)

    write_json(REPORT_DIR / "pipeline_dev_analysis.json",
               {"table": table, "split_k5_minus_cls": split, "taxonomy": taxonomy, "reader_ms_top10": reader_ms})
    print("đã ghi", REPORT_DIR / "pipeline_dev_analysis.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.checkpoint, a.limit)