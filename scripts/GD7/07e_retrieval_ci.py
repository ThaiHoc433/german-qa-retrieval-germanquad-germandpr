"""GĐ7 (thêm 2026-11-13): khoảng tin cậy bootstrap cho Hit/Recall/MRR@k từ file dự đoán ĐÃ LƯU, không chạy lại mô hình.
07a cũ chỉ báo trung bình retrieval (retrieval_ci nằm ở evaluation/paired.py vì report.py bị khóa trong 15_freeze); 07a mới tự tính CI cho các lần chạy sau. Script này bù cho kết quả đã có.
Test chỉ đọc khi có --confirm-test (ghi sổ); không dùng kết quả test để chỉnh cấu hình."""
import argparse

from germanqa.config import PROCESSED_DIR, REPORT_DIR
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.evaluation.paired import retrieval_ci
from germanqa.evaluation.report import retrieval_summary
from germanqa.experiment import load_config
from germanqa.experiment_rules import guard_or_exit


def main(split, predictions, confirm_test):
    name = predictions or f"pipeline_{split}_predictions.jsonl"
    if split == "test":
        guard_or_exit("07e_retrieval_ci.py", "CI retrieval từ dự đoán test đã lưu; chỉ báo cáo", confirm_test,
                      files=[name, "qrels_test.json"])
    ev = load_config("evaluation")["evaluation"]
    rows = read_jsonl(REPORT_DIR / name)
    qrels = read_json(PROCESSED_DIR / f"qrels_{split}.json")
    mean = retrieval_summary(rows, qrels, ev["ks"])
    ci = retrieval_ci(rows, qrels, ev["ks"], ev["bootstrap_samples"], ev["confidence"])
    for k in ev["ks"]:
        for m in ("hit", "mrr"):
            key = f"{m}@{k}"
            print(f"{key:7} {mean[key]:.4f}  [{ci[key][0]:.4f}; {ci[key][1]:.4f}]")
    out = REPORT_DIR / f"retrieval_ci_{split}_{name.replace('.jsonl', '')}.json"
    write_json(out, {"predictions": name, "queries": mean["queries"], "mean": mean, "ci95": ci,
                     "n_boot": ev["bootstrap_samples"], "confidence": ev["confidence"], "seed": 42})
    print("đã ghi", out.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--predictions", default=None, help="tên file trong reports/, mặc định pipeline_<split>_predictions.jsonl")
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    main(a.split, a.predictions, a.confirm_test)
