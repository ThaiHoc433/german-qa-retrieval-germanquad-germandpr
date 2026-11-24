"""Hàm phán định cho các giả thuyết ghi trước của GĐ15 (results/15_hypotheses.json).

Ghi hàm phán định TRƯỚC khi có số test để không thể chỉnh cách đọc sau khi thấy kết quả. Chỉ phép tính thuần, test bằng tay.

Vì sao dùng chênh với BM25 thay vì EM tuyệt đối: tập test khó hay dễ hơn dev là chuyện khác với việc encoder dense đã học đoạn của dev.
B0 (BM25) không học GermanDPR nên làm đối chứng độ khó: so chênh EM (B1 - B0, H0 - B0) giữa các tập."""

STRICT = "EM của B1/H0 trừ EM của B0: chênh trên TEST gần chênh trên dev-unseen hơn chênh trên toàn dev"


def delta(em_x, em_b0):
    return em_x - em_b0


def closer_to_unseen(d_test, d_unseen, d_full):
    """Đúng khi |d_test - d_unseen| < |d_test - d_full|. Bằng nhau -> Sai (không tính là ủng hộ)."""
    return abs(d_test - d_unseen) < abs(d_test - d_full)


def judge_ha(test, unseen, full):
    """test/unseen/full: {'B0': em, 'B1': em, 'H0': em}. Trả về {'B1': bool, 'H0': bool, 'verdict': ...}.
    verdict: 'ủng hộ' nếu đúng cho cả B1 và H0, 'bác bỏ' nếu sai cho cả hai, ngược lại 'hỗn hợp'."""
    res = {k: closer_to_unseen(delta(test[k], test["B0"]), delta(unseen[k], unseen["B0"]), delta(full[k], full["B0"])) for k in ("B1", "H0")}
    res["verdict"] = "ủng hộ" if all(res.values()) else "bác bỏ" if not any(res.values()) else "hỗn hợp"
    return res


def judge_hb(em_diff_h0_minus_b1):
    """H0 không kém B1 về EM trên test: chênh ghép cặp (điểm ước lượng) >= 0."""
    return {"holds": em_diff_h0_minus_b1 >= 0, "verdict": "ủng hộ" if em_diff_h0_minus_b1 >= 0 else "bác bỏ"}


def judge_hc(hit5_ci, em_ci):
    """H1 hơn H0 về Hit@5 (cận dưới > 0) NHƯNG chênh EM có khoảng tin cậy chứa 0. Mỗi vế phán riêng, rồi gộp.
    hit5_ci, em_ci: (chênh, thấp, cao) của H1 trừ H0."""
    part_a = hit5_ci[1] > 0
    part_b = em_ci[1] <= 0 <= em_ci[2]
    return {"hit5_gain_lower_bound_positive": part_a, "em_ci_contains_zero": part_b,
            "verdict": "ủng hộ" if part_a and part_b else "bác bỏ" if not part_a and not part_b else "hỗn hợp"}
