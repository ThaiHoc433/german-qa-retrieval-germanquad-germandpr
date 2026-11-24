"""GĐ11.4: ghi baselines/B0.json và H0.json (không ghi đè). B1.json do GĐ12 ghi từ trước; ở đây chỉ kiểm lại.
Băm SHA-1 của qa_test/qrels_test là kiểm tra toàn vẹn, không đọc nội dung; vẫn ghi vào sổ phơi bày dưới dạng metadata_only."""
import sys

from germanqa.config import ROOT
from germanqa.evaluation.manifest import build_manifest, load_manifest, verify_manifest
from germanqa.experiment_rules import log_test_metadata, write_once

NOTE = {"B0": "ĐÃ ĐÓNG BĂNG: BM25 (k1=0.7, b=0.2, stopwords) + reader minus_cls, max_answer_tokens=80",
        "H0": "ĐÃ ĐÓNG BĂNG: hybrid RRF (rrf_c=10, độ sâu 100/100, w_dense=1) trên dense text384 + BM25; xác nhận một lần trên dev-unseen ở GĐ13"}


def main():
    log_test_metadata("11d_manifests.py", "băm SHA-1 tệp test để kiểm toàn vẹn cho manifest; không đọc nội dung",
                      files=["qa_test.jsonl", "qrels_test.json"])
    for code in ("B0", "H0"):
        man = build_manifest(code, status_note=NOTE[code])
        p = write_once(ROOT / "baselines" / f"{code}.json", man)
        print("đã ghi", p.name, man["config_id"])
    for code in ("B0", "B1", "H0"):
        bad = verify_manifest(load_manifest(code))
        print(code, "khớp tệp thật" if not bad else bad)
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
