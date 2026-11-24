"""GĐ11.1 (bổ sung): ghi baselines/B1.v2.json = manifest B1 ĐẦY ĐỦ (SHA-1 reader, khối D5, mã cấu hình, môi trường), dựng bằng cùng
hàm build_manifest như B0/H0. KHÔNG đụng baselines/B1.json cũ (mốc không bị ghi đè): B1.v2 là mã mới, ghi lý do ở status.
Băm SHA-1 của qa_test/qrels_test là kiểm tra toàn vẹn, không đọc nội dung; vẫn ghi sổ phơi bày dưới dạng metadata_only."""
import sys

from germanqa.config import ROOT
from germanqa.evaluation.manifest import build_manifest, load_manifest, verify_manifest_detail
from germanqa.experiment_rules import log_test_metadata, write_once

NOTE = ("B1.v2: manifest đầy đủ thay cho B1.json (GĐ12.5) vốn thiếu SHA-1 reader, khối D5 và mã cấu hình. "
        "Số đo tham chiếu giống hệt B1.json (cùng tệp eval_dev_dense_20261107T072837Z_b97b5c1e). B1.json được giữ nguyên.")


def main():
    log_test_metadata("11f_b1_manifest_v2.py", "băm SHA-1 tệp test để kiểm toàn vẹn cho manifest B1.v2; không đọc nội dung",
                      files=["qa_test.jsonl", "qrels_test.json"])
    man = build_manifest("B1", status_note=NOTE)
    p = write_once(ROOT / "baselines" / "B1.v2.json", man)
    print("đã ghi", p.name, man["config_id"])
    for code in ("B1", "B1.v2"):
        bad, n = verify_manifest_detail(load_manifest(code))
        print(code, f"khớp tệp thật ({n} SHA-1)" if not bad else bad)
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
