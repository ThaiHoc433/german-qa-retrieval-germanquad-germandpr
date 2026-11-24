"""GĐ11.1: dựng lại phần sổ phơi bày test cho các lần mở test xảy ra TRƯỚC khi có sổ.
Bằng chứng duy nhất là thời gian sửa tệp kết quả còn trên đĩa, nên mọi dòng ghi source='reconstructed'. Chạy một lần;
từ chối nếu sổ đã có dòng dựng lại."""
import time

from germanqa.config import ROOT
from germanqa.data.io_utils import file_sha1
from germanqa.experiment_rules import LEDGER_PATH, append_ledger, read_ledger

# (script, loại, mục đích, tệp bằng chứng: tệp đầu tiên dùng lấy thời gian, mã)
HISTORY = [
    ("03_bm25_eval.py --test", "metric_run", "BM25 trên test với cấu hình chốt trên dev (k1=0.7, b=0.2, stopwords)",
     ["results/test_20261030T201634Z_d4c8e825.json"], ["B0"]),
    ("04d_eval_reader.py --test", "metric_run", "reader gold-context trên test (EM/F1 khi đưa đúng đoạn)",
     ["reports/reader_gold_context.json"], ["B0"]),
    ("07a_run_split.py --split test", "metric_run", "pipeline end-to-end B0 trên test",
     ["reports/pipeline_test_predictions.jsonl", "results/eval_test_20261103T170542Z_93098dcb.json", "reports/summary_test.md"], ["B0"]),
    ("07b_error_analysis.py --split test", "error_analysis", "phân loại lỗi trên dự đoán test (không rút mẫu đọc thủ công)",
     ["reports/error_analysis_test.json"], ["B0"]),
    ("06a_leakage_check.py", "metadata_only", "đối chiếu câu hỏi test với GermanDPR để kiểm tra rò rỉ; không tính chỉ số",
     ["reports/dense_leakage.json"], []),
]


def main():
    if any(r.get("source") == "reconstructed" for r in read_ledger()):
        raise SystemExit("sổ đã có dòng dựng lại: không dựng lần hai")
    for script, kind, purpose, files, codes in HISTORY:
        paths = [ROOT / f for f in files]
        missing = [f for f, p in zip(files, paths) if not p.exists()]
        if missing:
            raise SystemExit(f"thiếu tệp bằng chứng {missing}; không bịa dòng sổ")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(paths[0].stat().st_mtime))
        append_ledger({"ts": ts, "source": "reconstructed", "script": script, "kind": kind, "purpose": purpose,
                       "files": files, "codes": codes, "confirmed": kind != "metadata_only",
                       "evidence": {f: file_sha1(p) for f, p in zip(files, paths)},
                       "caveat": "dựng lại từ thời gian sửa tệp; giờ chạy thực có thể sớm hơn"})
        print(ts, script)
    print("đã ghi", LEDGER_PATH)


if __name__ == "__main__":
    main()
