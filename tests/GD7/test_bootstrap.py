import numpy as np
import pytest

from germanqa.evaluation.bootstrap import bootstrap_ci, paired_bootstrap


def test_cung_seed_cung_khoang():
    x = np.random.default_rng(0).integers(0, 2, 300)
    assert bootstrap_ci(x, 200, seed=7) == bootstrap_ci(x, 200, seed=7)


def test_khoang_chua_trung_binh_va_co_thu_tu():
    x = np.random.default_rng(1).integers(0, 2, 400)
    m, lo, hi = bootstrap_ci(x, 500, seed=3)
    assert lo <= m <= hi and m == pytest.approx(x.mean())


def test_diem_khong_doi_thi_khoang_co_do_rong_bang_khong():
    assert bootstrap_ci([1.0] * 50, 100) == (1.0, 1.0, 1.0)


def test_nhieu_cau_hon_thi_khoang_hep_hon():
    rng = np.random.default_rng(2)
    small, big = rng.integers(0, 2, 100), rng.integers(0, 2, 4000)
    w = lambda x: (lambda r: r[2] - r[1])(bootstrap_ci(x, 500, seed=1))
    assert w(big) < w(small)


def test_ghep_cap_cung_mot_cau_hinh_cho_chenh_bang_khong():
    x = np.random.default_rng(3).random(200)
    assert paired_bootstrap(x, x, 100) == (0.0, 0.0, 0.0)


def test_ghep_cap_thay_chenh_ro_va_khong_chua_khong():
    rng = np.random.default_rng(4)
    b = rng.integers(0, 2, 500).astype(float)
    a = np.clip(b + (rng.random(500) < 0.2), 0, 1)           # a luôn >= b và thắng ở 20% số câu (khi b = 0)
    d, lo, hi = paired_bootstrap(a, b, 500)
    assert d > 0 and lo > 0


def test_ghep_cap_doi_hoi_cung_so_cau():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0, 1], [1, 0])
