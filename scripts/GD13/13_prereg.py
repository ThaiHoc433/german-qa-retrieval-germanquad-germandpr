"""GĐ13.0: ghi results/13_prereg.json TRƯỚC khi quét RRF. File khóa: chỉ số, lưới, quy tắc chọn, quy tắc nhận,
cách chia tập, SHA-1 của danh sách top-100 đầu vào và của mã quy tắc. 13a_sweep.py từ chối chạy nếu file này
thiếu hoặc nếu bất kỳ SHA-1 nào đã đổi. Không ghi đè nếu đã tồn tại (muốn ghi lại phải xóa tay và báo lý do)."""
import time

from germanqa.config import RESULTS_DIR, ROOT, SEED
from germanqa.data.io_utils import read_json, write_json
from germanqa.evaluation.fusion_select import inputs_sha1

INPUTS = ["results/B0_select_top100.jsonl", "results/B1_select_top100_384.jsonl", "results/select_question_ids.json",
          "results/dev_unseen_question_ids.json", "baselines/B1.json"]
CODE = ["src/germanqa/retrieval/rrf.py", "src/germanqa/evaluation/decision.py", "src/germanqa/evaluation/fusion_select.py"]


def main():
    out = RESULTS_DIR / "13_prereg.json"
    if out.exists():
        raise SystemExit(f"{out.name} đã tồn tại: không ghi đè (quy tắc ghi trước)")
    sel = read_json(RESULTS_DIR / "select_question_ids.json")
    unseen = read_json(RESULTS_DIR / "dev_unseen_question_ids.json")
    assert set(sel["dev"]) == set(unseen), "dev trong tập chọn phải đúng bằng dev_unseen"
    pre = {
        "phase": "GĐ13", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
        "purpose": "Hybrid BM25 + dense bằng RRF (H0). Chọn cấu hình trên train-select, xác nhận MỘT lần trên dev-unseen.",
        "inputs": inputs_sha1(ROOT, INPUTS), "code": inputs_sha1(ROOT, CODE),
        "dense_artifact": "artifacts/dense/text384",
        "splits": {"selection": f"train trong tập chọn (n={len(sel['train'])})", "confirmation": f"dev-unseen (n={len(sel['dev'])})",
                   "test": "KHÔNG mở"},
        "metrics": {"primary": "hit@5", "secondary_for_adoption": "hit@20", "also_reported": ["recall@20", "mrr@10"]},
        "grid": {"rrf_c": [10, 20, 60, 100], "depth": [20, 50, 100], "n_configs": 12,
                 "note": "depth = số đoạn lấy từ MỖI nhánh đưa vào RRF; fusion_top_k = 20 (số đoạn trả về)"},
        "default": {"rrf_c": 60, "depth": 50, "w_dense": 1.0, "fusion_top_k": 20},
        "exploratory": {"w_dense": [1.0, 1.5, 2.0], "w_bm25": 1.0,
                        "note": "THĂM DÒ, chạy tại (rrf_c, depth) của cấu hình chính cuối cùng; w=1.0 trùng cấu hình chính; "
                                "tối đa 12 + 2 = 14 cấu hình khác nhau"},
        "selection_rule": "Hit@5 cao nhất trên train-select; hòa -> Recall@20 -> MRR@10 -> gần mặc định nhất "
                          "(chính nó, rồi độ sâu gần, rồi c gần). Hàm: fusion_select.choose_config",
        "adoption_rule": {"function": "decision.adopt_alternative", "margin": 0.01,
                          "primary": "Hit@5 chênh ghép cặp > 0 với cận dưới > 0", "secondary": "Hit@20 cận dưới >= -0.01",
                          "chosen_vs": "cấu hình mặc định (lưới chính); cấu hình chính cuối cùng (nhóm thăm dò)",
                          "if_rejected": "giữ cấu hình mặc định / cấu hình chính"},
        "confirmation_rule": "Chạy cấu hình cuối cùng MỘT lần trên dev-unseen. H0 chỉ được gọi là hơn B1 nếu cận dưới khoảng "
                             "tin cậy ghép cặp của Hit@5 > 0 trên 234 câu; ngược lại ghi 'chưa đủ bằng chứng'. "
                             "Không chọn lại sau bước này.",
        "bootstrap": {"n_boot": 1000, "confidence": 0.95, "seed": 42, "unit": "câu hỏi, ghép cặp"},
        "max_configs_tried": 14,
        "definitions": {"rrf": "score = sum_nhánh w / (c + hạng), hạng từ 1, vắng = 0",
                        "tie": "điểm RRF, rồi hạng tốt nhất ở bất kỳ nhánh nào, rồi passage_id",
                        "absent_branch_score": "None (không phải 0)",
                        "overlap": "tỉ lệ từ khóa (bỏ stopword) của câu hỏi có trong đoạn đúng; nhóm 0-25/25-50/50-75/75-100%"},
    }
    write_json(out, pre)
    print("đã ghi", out.name, "| đầu vào:", len(pre["inputs"]), "| mã:", len(pre["code"]))


if __name__ == "__main__":
    main()
