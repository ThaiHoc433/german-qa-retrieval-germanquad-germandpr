"""Quy tắc nhận/bỏ một phương án thay thế, GHI TRƯỚC khi xem kết quả (GĐ12.3)."""


def adopt_alternative(primary_ci, secondary_ci, margin=0.01):
    """primary_ci, secondary_ci: (chênh lệch, cận dưới, cận trên) của phương án MỚI trừ phương án GỐC.
    Trả về (nhận_không, lý_do)."""
    d1, lo1, _ = primary_ci
    d2, lo2, _ = secondary_ci
    if not (d1 > 0 and lo1 > 0):
        return False, f"chỉ số chính chưa đủ chắc: chênh {d1:+.4f}, cận dưới {lo1:+.4f} không > 0"
    if lo2 < -margin:
        return False, f"chỉ số phụ có thể giảm quá {margin}: cận dưới {lo2:+.4f}"
    return True, f"chỉ số chính tăng {d1:+.4f} (cận dưới {lo1:+.4f} > 0); chỉ số phụ cận dưới {lo2:+.4f} ≥ {-margin}"
