"""GĐ14.8: ghi baselines/H1.json (không ghi đè) rồi kiểm lại B0, B1, H0, H1 với tệp thật.
Băm SHA-1 của qa_test/qrels_test là kiểm tra toàn vẹn, không đọc nội dung; vẫn ghi sổ phơi bày dưới dạng metadata_only."""
import sys

from germanqa.config import ROOT
from germanqa.evaluation.manifest import build_manifest, load_manifest, verify_manifest_detail
from germanqa.experiment_rules import log_test_metadata, write_once

NOTE = "ĐÃ ĐÓNG BĂNG: H0 + cross-encoder bge-reranker-v2-m3 (cửa sổ, max_length 512) xếp lại 20 đoạn, reader nhận 5; xác nhận một lần trên dev-unseen ở GĐ14"


def main():
    log_test_metadata("14h_manifest.py", "băm SHA-1 tệp test để kiểm toàn vẹn cho manifest H1; không đọc nội dung",
                      files=["qa_test.jsonl", "qrels_test.json"])
    man = build_manifest("H1", status_note=NOTE)
    p = write_once(ROOT / "baselines" / "H1.json", man)
    print("đã ghi", p.name, man["config_id"])
    for code in ("B0", "B1", "H0", "H1"):
        bad, n = verify_manifest_detail(load_manifest(code))
        print(code, f"khớp tệp thật ({n} SHA-1)" if not bad else bad)
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
