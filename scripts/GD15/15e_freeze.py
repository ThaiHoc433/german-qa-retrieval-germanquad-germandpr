"""GĐ15.6a: đóng băng cấu hình TRƯỚC khi mở test. Kiểm mọi điều kiện (preflight), rồi ghi results/15_freeze.json (một lần, không ghi đè).
Cũng kiểm lại prereg của GĐ13 và GĐ14 còn khớp tệp thật. Script này KHÔNG đọc tệp test nào ngoài việc đếm dòng và băm SHA-1 tệp dự đoán B0 đã lưu;
việc băm/đếm vẫn ghi sổ phơi bày dưới dạng metadata_only."""
import time

from germanqa.config import RESULTS_DIR, ROOT
from germanqa.evaluation.fusion_select import require_prereg
from germanqa.evaluation.gd15_freeze import b0_reuse_problems, build_freeze, preflight_problems
from germanqa.experiment_rules import log_test_metadata, write_once


def main():
    out = RESULTS_DIR / "15_freeze.json"
    if out.exists():
        raise SystemExit(f"{out.name} đã tồn tại: không ghi đè")
    log_test_metadata("15e_freeze.py", "đếm dòng qa_test.jsonl và băm SHA-1 tệp dự đoán/kết quả B0 đã lưu để kiểm việc dùng lại B0; không tính chỉ số",
                      files=["qa_test.jsonl", "pipeline_test_predictions.jsonl", "eval_test_20261103T170542Z_93098dcb.json"])
    problems = preflight_problems(ROOT)
    for name in ("13_prereg.json", "14_prereg.json"):
        try:
            require_prereg(RESULTS_DIR / name, ROOT)
        except Exception as e:                                   # noqa: BLE001  (PreregError hoặc thiếu tệp)
            problems.append(f"{name}: {e}")
    if problems:
        raise SystemExit("KHÔNG đóng băng được:\n  - " + "\n  - ".join(problems))
    doc = {"phase": "GĐ15.6a", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), **build_freeze(ROOT),
           "rules": ["Cấu hình chọn chỉ bằng dev (dev-unseen + tập chọn); test không tham gia chọn.",
                     "Chạy test MỘT lần cho B1, H0, H1 (B0 dùng lại số đã chạy 2026-11-04, cấu hình khớp manifest B0 hiện tại).",
                     "Không dùng lỗi/dự đoán test để thiết kế hay chọn lại; phán định các giả thuyết theo 15_hypotheses.json + phụ lục (EM first_ref là số chính).",
                     "Báo cả khi giả thuyết bị bác bỏ."],
           "b0_reuse_check": b0_reuse_problems(ROOT) or "khớp"}
    write_once(out, doc)
    print("đã đóng băng", out.name, "|", len(doc["files_sha1"]), "tệp | B0 dùng lại:", doc["b0_reuse_check"])


if __name__ == "__main__":
    main()
