from germanqa.evaluation.decision import adopt_alternative


def test_nhan_khi_chinh_tang_chac_va_phu_khong_giam_qua_margin():
    ok, why = adopt_alternative((0.0148, 0.0058, 0.0243), (-0.0004, -0.0049, 0.0040))
    assert ok and "cận dưới" in why


def test_bo_khi_khoang_tin_cay_chinh_chua_0():
    ok, _ = adopt_alternative((0.0128, -0.0171, 0.0427), (0.0, -0.001, 0.001))
    assert not ok


def test_bo_khi_chi_so_phu_co_the_giam_qua_margin():
    ok, why = adopt_alternative((0.02, 0.01, 0.03), (-0.005, -0.015, 0.004))
    assert not ok and "chỉ số phụ" in why


def test_bien_margin_la_gia_tri_dung_ngay_bien():
    assert adopt_alternative((0.02, 0.01, 0.03), (0.0, -0.01, 0.01))[0]          # cận dưới = -margin: vẫn nhận
    assert not adopt_alternative((0.02, 0.01, 0.03), (0.0, -0.0101, 0.01))[0]
