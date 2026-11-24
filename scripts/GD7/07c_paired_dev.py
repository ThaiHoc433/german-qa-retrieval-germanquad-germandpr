"""GĐ7.8: so cấu hình bằng bootstrap GHÉP CẶP trên dev (không chạm test).
Chạy pipeline một lần mỗi chế độ với reader_top_k = 10 rồi suy ra k nhỏ hơn từ danh sách ứng viên (xem GĐ5 mục 5.6)."""
import argparse

from germanqa.config import PROCESSED_DIR, REPORT_DIR
from germanqa.data.io_utils import write_json
from germanqa.evaluation.bootstrap import paired_bootstrap
from germanqa.evaluation.report import choose_span
from germanqa.experiment import load_config
from germanqa.pipeline import build_pipeline
from germanqa.reader.features import load_examples
from germanqa.reader.qa_metrics import score_answers

COMPARISONS = [  # (tên, cấu hình A, cấu hình B): đo A - B
    ("5 đoạn so với 3 đoạn", ("minus_cls", 5), ("minus_cls", 3)),
    ("10 đoạn so với 5 đoạn", ("minus_cls", 10), ("minus_cls", 5)),
    ("raw so với minus_cls (5 đoạn)", ("raw", 5), ("minus_cls", 5)),
    ("raw so với minus_cls (10 đoạn)", ("raw", 10), ("minus_cls", 10)),
]


def main(checkpoint, limit):
    ev = load_config("evaluation")["evaluation"]
    pipe = build_pipeline(checkpoint)
    examples = load_examples(PROCESSED_DIR, "dev")[:limit]
    pipe.answer(examples[0]["question"])
    cands = {}
    for mode in ("raw", "minus_cls"):
        pipe.reader.score_mode = mode
        cands[mode] = [pipe.answer(ex["question"], reader_top_k=10, trace=True).trace["candidates"] for ex in examples]
        print(mode, "xong", flush=True)

    scores = {}
    for mode in cands:
        for k in (3, 5, 10):
            preds = {ex["id"]: {"text": (choose_span(c, k) or {"answer": ""})["answer"]} for ex, c in zip(examples, cands[mode])}
            scores[(mode, k)] = score_answers(preds, examples)

    out = []
    for name, a, b in COMPARISONS:
        row = {"so_sanh": name}
        for i, metric in enumerate(("em", "f1")):
            d, lo, hi = paired_bootstrap([s[i] for s in scores[a]], [s[i] for s in scores[b]],
                                         ev["bootstrap_samples"], ev["confidence"])
            row[metric] = {"diff": d, "ci": [lo, hi], "distinguishable": not (lo <= 0 <= hi)}
        out.append(row)
        print(f"{name:34} EM {row['em']['diff']:+.4f} [{row['em']['ci'][0]:+.4f}; {row['em']['ci'][1]:+.4f}]  "
              f"F1 {row['f1']['diff']:+.4f} [{row['f1']['ci'][0]:+.4f}; {row['f1']['ci'][1]:+.4f}]")
    write_json(REPORT_DIR / "paired_comparisons_dev.json", {"queries": len(examples), "comparisons": out})
    print("đã ghi paired_comparisons_dev.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.checkpoint, a.limit)
