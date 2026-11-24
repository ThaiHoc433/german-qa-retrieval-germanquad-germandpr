"""Chọn cấu hình reranker và quyết định nhận/không nhận (GĐ14), theo quy tắc ghi trong results/14_prereg.json.

Chỉ chứa phép tính thuần (không GPU, không đọc dữ liệu thật) để test được bằng tay."""
from germanqa.evaluation.decision import adopt_alternative
from germanqa.rerank.cross_encoder import stable_rank

MODEL_COST_ORDER = ("mmarco", "gbert_dpr", "bge_m3")      # rẻ -> đắt (số tham số / thời gian chấm đo ở 14a)
OVERFLOW_COST_ORDER = ("truncate", "windows")


def ranked_from_scores(pool_ids, scores):
    """pool_ids: passage_id theo hạng RRF cũ; scores: điểm reranker tương ứng. Trả về passage_id theo hạng mới
    (giảm dần theo điểm, hòa điểm giữ hạng RRF cũ)."""
    if len(pool_ids) != len(scores):
        raise ValueError("số điểm khác số đoạn")
    return [r["passage_id"] for r in stable_rank([{"passage_id": p} for p in pool_ids], scores)]


def cost_key(cfg):
    return (OVERFLOW_COST_ORDER.index(cfg["overflow"]), MODEL_COST_ORDER.index(cfg["model"]))


def choose_config(results, eps=1e-12):
    """results: [{'model', 'overflow', 'hit@5', 'mrr@10'}]. Hit@5 cao nhất; hòa -> MRR@10 cao hơn; hòa nữa -> rẻ hơn
    (truncate trước windows, rồi mmarco < gbert_dpr < bge_m3). Trả về phần tử được chọn."""
    if not results:
        raise ValueError("không có cấu hình nào")
    best = max(r["hit@5"] for r in results)
    tied = [r for r in results if r["hit@5"] >= best - eps]
    best_m = max(r["mrr@10"] for r in tied)
    tied = [r for r in tied if r["mrr@10"] >= best_m - eps]
    return min(tied, key=cost_key)


def exposure_guard(never_ci, margin=0.01):
    """never_ci: (chênh, cận dưới, cận trên) của Hit@5 (đã chọn - H0) trên nhóm 'never'. Đạt nếu điểm ước lượng > 0
    và cận dưới >= -margin. Trả về (đạt_không, lý_do)."""
    d, lo, _ = never_ci
    if not d > 0:
        return False, f"nhóm 'never': chênh {d:+.4f} không > 0"
    if lo < -margin:
        return False, f"nhóm 'never': cận dưới {lo:+.4f} < {-margin}"
    return True, f"nhóm 'never': chênh {d:+.4f}, cận dưới {lo:+.4f} >= {-margin}"


def decide(overall_hit5, overall_hit20, never_hit5, margin=0.01):
    """Ba bộ (chênh, thấp, cao) -> (nhận_không, lý_do). Chỉ nhận khi qua cả quy tắc nhận chung và rào 'đã thấy đoạn'."""
    ok, why = adopt_alternative(overall_hit5, overall_hit20, margin)
    if not ok:
        return False, why
    ok2, why2 = exposure_guard(never_ci=never_hit5, margin=margin)
    return (ok2, why + "; " + why2) if ok2 else (False, why + "; nhưng " + why2)
