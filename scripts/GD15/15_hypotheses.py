"""GĐ15.0: ghi results/15_hypotheses.json TRƯỚC khi chạy test cho B1/H0/H1. Ghi một lần (không ghi đè), kèm thời điểm và SHA-1 các tệp nguồn.
Từ chối chạy nếu sổ phơi bày test đã có lần tính chỉ số (metric_run) cho B1, H0 hoặc H1: khi đó lời ghi trước không còn nghĩa.
Không đọc bất kỳ số liệu test nào (B0 đã chạy trên test từ 2026-11-04; số đó cũng KHÔNG được dùng ở đây)."""
import time

from germanqa.config import RESULTS_DIR, ROOT
from germanqa.data.io_utils import file_sha1, read_json
from germanqa.evaluation import gd15_hypotheses as H
from germanqa.experiment_rules import read_ledger, write_once

SRC = {"B0": "results/eval_dev_20261103T163556Z_cb4749bc.json", "B1": "results/eval_dev_dense_20261107T072837Z_b97b5c1e.json",
       "H0": "results/eval_dev_hybrid_20261107T135423Z_598ddf6f.json"}
CODE = ["src/germanqa/evaluation/gd15_hypotheses.py"]


def reference_numbers():
    h = read_json(ROOT / SRC["H0"])["metrics"]
    full = {"B0": h["dev"]["B0_bm25"]["first_ref"]["em"], "B1": h["dev"]["B1_dense384"]["first_ref"]["em"], "H0": h["dev"]["hybrid"]["first_ref"]["em"]}
    unseen = {"B0": h["dev_unseen"]["B0_bm25"]["first_ref"]["em"], "B1": h["dev_unseen"]["B1_dense384"]["first_ref"]["em"],
              "H0": h["dev_unseen"]["hybrid"]["first_ref"]["em"]}
    return full, unseen


def main():
    out = RESULTS_DIR / "15_hypotheses.json"
    used = [r for r in read_ledger() if r["kind"] == "metric_run" and set(r.get("codes", [])) & {"B1", "H0", "H1"}]
    if used:
        raise SystemExit(f"sổ phơi bày đã có lần tính chỉ số test cho {sorted({c for r in used for c in r['codes']})}: không còn là ghi trước")
    full, unseen = reference_numbers()
    d = lambda s, k: round(H.delta(s[k], s["B0"]), 6)
    doc = {
        "phase": "GĐ15.0", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Giả thuyết ghi TRƯỚC khi chạy test một lần cho B0/B1/H0/H1 ở GĐ15.6. Phán định bằng các hàm trong gd15_hypotheses.py.",
        "why": "Trên dev, H0 và B1 trông tốt vì encoder dense đã học đoạn của dev (nhóm 'positive': Hit@5 của H0 0,939 so với 0,857 ở nhóm 'never'); "
               "test không có đoạn đúng nào như vậy. Nếu viết dự đoán sau khi thấy số test thì mất giá trị.",
        "reference_dev_em": {"full_dev_1176": full, "dev_unseen_234": unseen},
        "reference_delta_vs_B0": {"full_dev": {"B1": d(full, "B1"), "H0": d(full, "H0")}, "dev_unseen": {"B1": d(unseen, "B1"), "H0": d(unseen, "H0")}},
        "hypotheses": {
            "HA": {"statement": H.STRICT, "judge": "gd15_hypotheses.judge_ha",
                   "supported_if": "đúng cho CẢ B1 và H0", "falsified_if": "sai cho CẢ B1 và H0", "otherwise": "hỗn hợp (báo cả hai kết quả)",
                   "note": "so chênh với BM25 để loại độ khó khác nhau giữa các tập; bằng nhau tính là sai"},
            "HB": {"statement": "Trên test, H0 không kém B1 về EM (chênh ghép cặp, điểm ước lượng >= 0).", "judge": "gd15_hypotheses.judge_hb",
                   "reference": {"full_dev": round(full["H0"] - full["B1"], 6), "dev_unseen": round(unseen["H0"] - unseen["B1"], 6)},
                   "falsified_if": "điểm ước lượng < 0"},
            "HC": {"statement": "Trên test, H1 hơn H0 về Hit@5 (cận dưới khoảng tin cậy ghép cặp > 0) NHƯNG chênh EM có khoảng tin cậy chứa 0.",
                   "judge": "gd15_hypotheses.judge_hc", "falsified_if": "cả hai vế sai; nếu chỉ một vế sai thì 'hỗn hợp'"},
        },
        "inputs_sha1": {k: file_sha1(ROOT / v) for k, v in SRC.items()},
        "code_sha1": {p: file_sha1(ROOT / p) for p in CODE},
        "rules": ["Kết quả test chỉ dùng để phán định các giả thuyết này và báo cáo; không dùng để chọn lại cấu hình.",
                  "Báo cả khi giả thuyết bị bác bỏ."],
    }
    p = write_once(out, doc)
    print("đã ghi", p.name, "|", doc["created_at"])
    print("chênh EM so với B0 — toàn dev:", doc["reference_delta_vs_B0"]["full_dev"], "| dev-unseen:", doc["reference_delta_vs_B0"]["dev_unseen"])


if __name__ == "__main__":
    main()
