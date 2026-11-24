"""GĐ15.0b: phụ lục cho results/15_hypotheses.json, ghi TRƯỚC khi chạy test cho B1/H0/H1. Ghi một lần, từ chối nếu sổ phơi bày đã có
lần tính chỉ số test cho B1, H0 hoặc H1.

Lỗ hổng cần vá: 15_hypotheses.json phán định HA/HB bằng EM nhưng không nói dùng đáp án tham chiếu nào. Tập test có NHIỀU đáp án mỗi câu hỏi
(B0 trên test: EM first_ref 0,222 nhưng all_refs 0,330), còn dev chỉ có một đáp án (hai loại bằng nhau). Nếu không chốt trước thì sau khi thấy số test
có thể chọn loại có lợi cho giả thuyết. Phụ lục chốt: loại CHÍNH là first_ref (so được trực tiếp với các chênh đã tính trên dev); all_refs chỉ báo thêm."""
import time

from germanqa.config import RESULTS_DIR
from germanqa.data.io_utils import file_sha1
from germanqa.experiment_rules import read_ledger, write_once


def main():
    used = [r for r in read_ledger() if r["kind"] == "metric_run" and set(r.get("codes", [])) & {"B1", "H0", "H1"}]
    if used:
        raise SystemExit(f"sổ phơi bày đã có lần tính chỉ số test cho {sorted({c for r in used for c in r['codes']})}: không còn là ghi trước")
    base = RESULTS_DIR / "15_hypotheses.json"
    doc = {
        "phase": "GĐ15.0b", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "amends": "results/15_hypotheses.json",
        "amends_sha1": file_sha1(base),
        "gap": "15_hypotheses.json không chỉ rõ dùng EM theo đáp án tham chiếu nào cho test; test có nhiều đáp án tham chiếu mỗi câu, dev chỉ có một.",
        "decision": {
            "primary_reference_mode": "first_ref",
            "why": "dev chỉ có một đáp án nên first_ref và all_refs bằng nhau ở dev; chỉ first_ref so được trực tiếp với các chênh EM so với B0 đã ghi ở "
                   "15_hypotheses.json (toàn dev và dev-unseen).",
            "secondary_reference_mode": "all_refs (báo thêm, kèm khoảng tin cậy, không dùng để phán định)",
            "applies_to": ["HA", "HB", "HC (vế EM)"],
            "if_modes_disagree": "vẫn phán định theo first_ref và ghi rõ rằng all_refs cho kết luận khác",
        },
        "disclosure": "Khi viết phụ lục này, tao (trợ lý) đã đọc results/eval_test_20261103T170542Z_93098dcb.json (B0 trên test, chạy từ 2026-11-04): "
                      "EM first_ref 0,2224, all_refs 0,3300 trên 2.203 câu. Số này không dùng để chọn loại EM; lý do chọn first_ref nằm ở trên và độc lập với nó.",
        "unchanged": "ba giả thuyết HA, HB, HC và hàm phán định trong gd15_hypotheses.py không đổi.",
    }
    p = write_once(RESULTS_DIR / "15_hypotheses_addendum.json", doc)
    print("đã ghi", p.name, "|", doc["created_at"], "| sửa", doc["amends"], doc["amends_sha1"][:10])


if __name__ == "__main__":
    main()
