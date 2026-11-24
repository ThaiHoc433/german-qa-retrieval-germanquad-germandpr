"""GĐ7 bước "nhóm -> đếm" (thêm 2026-11-13): đếm nhãn nguyên nhân gốc của các mẫu lỗi dev đã đọc tay.

Quy trình: 07b ghi reports/error_labels_dev.template.json -> người đọc chép thành reports/error_labels_dev.json,
điền "primary" (bắt buộc, một mã trong codebook), "secondary" (tùy chọn, đa nhãn), "note" -> chạy script này.
Tỉ lệ in ra là tỉ lệ TRONG MẪU (tối đa 40 câu mỗi stage), kèm khoảng Wilson; không phải tỉ lệ trên toàn dev.
Chỉ dev: mẫu đọc tay không bao giờ lấy từ test."""
import sys

from germanqa.config import REPORT_DIR
from germanqa.data.io_utils import read_json, write_json
from germanqa.evaluation.errors import count_labels, validate_labels


def main():
    path = REPORT_DIR / "error_labels_dev.json"
    if not path.exists():
        sys.exit(f"chưa có {path.name}: chép {path.stem}.template.json thành {path.name} rồi điền nhãn")
    labels = read_json(path)["labels"]
    problems = validate_labels(labels)
    done = len(labels) - len(problems["unlabeled"])
    print(f"đã gán {done}/{len(labels)} câu")
    if any(problems.values()):
        print("chưa đếm được:", {k: v[:5] for k, v in problems.items() if v})
        sys.exit(1)
    out = count_labels(labels)
    stage_totals = read_json(REPORT_DIR / "error_analysis_dev.json")["by_stage"]
    for s, cell in out.items():
        cell["errors_in_dev_for_stage"] = stage_totals.get(s)
        print(f"\n{s} (mẫu {cell['n']} / {stage_totals.get(s)} câu sai trên dev)")
        for c, v in sorted(cell["primary"].items(), key=lambda kv: -kv[1]["count"]):
            print(f"  {c:36} {v['count']:3}  {v['share']:.0%}  Wilson95 [{v['wilson95'][0]:.2f}; {v['wilson95'][1]:.2f}]")
    write_json(REPORT_DIR / "error_root_causes_dev.json", out)
    print("\nđã ghi error_root_causes_dev.json")


if __name__ == "__main__":
    main()
