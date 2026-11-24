import json

import pytest

from germanqa.evaluation.fusion_select import PreregError, choose_config, inputs_sha1, require_prereg
from germanqa.experiment import new_experiment_id

DEFAULT = {"rrf_c": 60, "depth": 50}


def row(c, d, h5, r20=0.9, m=0.6):
    return {"rrf_c": c, "depth": d, "hit@5": h5, "recall@20": r20, "mrr@10": m}


def test_chon_hit5_cao_nhat():
    best = choose_config([row(60, 50, 0.80), row(10, 20, 0.83), row(100, 100, 0.81)], DEFAULT)
    assert (best["rrf_c"], best["depth"]) == (10, 20)


def test_hoa_theo_thu_tu_tieu_chi():
    assert choose_config([row(10, 20, .8, r20=.90), row(20, 20, .8, r20=.95)], DEFAULT)["rrf_c"] == 20     # Recall@20
    assert choose_config([row(10, 20, .8, m=.60), row(20, 20, .8, m=.65)], DEFAULT)["rrf_c"] == 20          # MRR@10
    # hòa hết: ưu tiên chính mặc định, rồi độ sâu gần mặc định, rồi c gần mặc định
    assert _k(choose_config([row(10, 20, .8), row(60, 50, .8), row(100, 100, .8)], DEFAULT)) == (60, 50)
    assert _k(choose_config([row(10, 20, .8), row(10, 100, .8), row(20, 20, .8)], DEFAULT)) == (20, 20)
    assert _k(choose_config([row(10, 50, .8), row(90, 50, .8), row(20, 50, .8)], DEFAULT)) == (90, 50)


def _k(r):
    return r["rrf_c"], r["depth"]


def test_khong_co_cau_hinh():
    with pytest.raises(ValueError):
        choose_config([], DEFAULT)


def test_thieu_file_ghi_truoc_va_sha1_lech(tmp_path):
    (tmp_path / "in.txt").write_text("a", encoding="utf-8")
    (tmp_path / "code.py").write_text("x=1", encoding="utf-8")
    with pytest.raises(PreregError):
        require_prereg(tmp_path / "13_prereg.json", tmp_path)                       # thiếu file ghi trước
    pre = {"inputs": inputs_sha1(tmp_path, ["in.txt"]), "code": inputs_sha1(tmp_path, ["code.py"])}
    (tmp_path / "13_prereg.json").write_text(json.dumps(pre), encoding="utf-8")
    assert require_prereg(tmp_path / "13_prereg.json", tmp_path)["inputs"]
    (tmp_path / "in.txt").write_text("b", encoding="utf-8")                          # đầu vào đổi sau khi ghi trước
    with pytest.raises(PreregError, match="in.txt"):
        require_prereg(tmp_path / "13_prereg.json", tmp_path)
    (tmp_path / "in.txt").write_text("a", encoding="utf-8")
    (tmp_path / "code.py").write_text("x=2", encoding="utf-8")                       # mã quy tắc đổi
    with pytest.raises(PreregError, match="code.py"):
        require_prereg(tmp_path / "13_prereg.json", tmp_path)


def test_ma_thi_nghiem_khac_nhau_khi_rrf_c_khac():
    a = new_experiment_id("pipeline", {"hybrid": {"rrf_c": 60}})
    b = new_experiment_id("pipeline", {"hybrid": {"rrf_c": 20}})
    assert a.split("_")[-1] != b.split("_")[-1]
