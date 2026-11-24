"""Khoảng tin cậy bằng bootstrap (GĐ7): resample CÂU HỎI có hoàn lại, tính lại trung bình.

Điểm EM/F1 của một câu là một số cố định; sự ngẫu nhiên nằm ở việc ta chỉ có 1.176 (dev) hoặc 2.203 (test) câu.
Resample câu hỏi cho biết con số trung bình sẽ dao động bao nhiêu nếu có một bộ câu hỏi khác cùng loại."""
import numpy as np


def _rng(seed):
    return np.random.default_rng(seed)


def bootstrap_ci(scores, n_boot=1000, confidence=0.95, seed=42):
    """Trả về (trung bình, thấp, cao) của khoảng tin cậy phân vị cho trung bình của `scores`."""
    x = np.asarray(scores, dtype=float)
    idx = _rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - confidence), 100 - 50 * (1 - confidence)])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(scores_a, scores_b, n_boot=1000, confidence=0.95, seed=42):
    """So hai cấu hình trên CÙNG các câu hỏi. Trả về (chênh a - b, thấp, cao).
    Cùng một lần resample dùng cho cả hai, nên chênh lệch không bị nhiễu bởi việc câu nào được chọn.
    Khoảng chứa 0 nghĩa là dữ liệu chưa đủ để nói cấu hình nào tốt hơn."""
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("hai dãy điểm phải cùng độ dài (cùng các câu hỏi)")
    return bootstrap_ci(a - b, n_boot, confidence, seed)