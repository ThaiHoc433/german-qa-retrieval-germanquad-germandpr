import pytest

from germanqa.rerank import select as S
from germanqa.rerank.exposure import passage_role, question_group


def test_ranked_from_scores_dung_chieu_va_hoa_diem_giu_hang_rrf():
    assert S.ranked_from_scores(["a", "b", "c", "d"], [0.1, 3.0, 0.1, 3.0]) == ["b", "d", "a", "c"]
    assert S.ranked_from_scores([], []) == []
    with pytest.raises(ValueError):
        S.ranked_from_scores(["a"], [1.0, 2.0])


def R(model, overflow, h5, m10):
    return {"model": model, "overflow": overflow, "hit@5": h5, "mrr@10": m10}


def test_chon_hit5_cao_nhat_roi_mrr_roi_re_hon():
    assert S.choose_config([R("mmarco", "truncate", .90, .7), R("bge_m3", "truncate", .92, .6)])["model"] == "bge_m3"
    assert S.choose_config([R("mmarco", "truncate", .92, .6), R("bge_m3", "truncate", .92, .7)])["model"] == "bge_m3"
    t = S.choose_config([R("bge_m3", "truncate", .92, .7), R("mmarco", "windows", .92, .7), R("gbert_dpr", "truncate", .92, .7)])
    assert (t["model"], t["overflow"]) == ("gbert_dpr", "truncate")        # truncate rẻ hơn windows, gbert rẻ hơn bge
    t2 = S.choose_config([R("bge_m3", "truncate", .92, .7), R("mmarco", "truncate", .92, .7)])
    assert t2["model"] == "mmarco"
    with pytest.raises(ValueError):
        S.choose_config([])


def test_rao_da_thay_doan():
    assert S.exposure_guard((0.02, 0.005, 0.04))[0]
    assert not S.exposure_guard((0.0, -0.005, 0.01))[0]                  # chênh không > 0
    assert not S.exposure_guard((0.01, -0.02, 0.04))[0]                  # cận dưới quá thấp
    assert S.exposure_guard((0.002, -0.01, 0.01))[0]                     # đúng biên: cận dưới == -margin vẫn đạt


def test_nhan_can_ca_quy_tac_chung_va_rao():
    good_all, ok20 = (0.03, 0.01, 0.05), (0.0, 0.0, 0.0)
    assert S.decide(good_all, ok20, (0.02, 0.0, 0.04))[0]
    ok, why = S.decide(good_all, ok20, (-0.01, -0.03, 0.01))
    assert not ok and "never" in why                                      # chung đạt nhưng nhóm 'never' hỏng
    assert not S.decide((0.03, -0.001, 0.05), ok20, (0.02, 0.0, 0.04))[0]  # chung chưa đủ chắc
    assert not S.decide(good_all, (-0.02, -0.03, 0.0), (0.02, 0.0, 0.04))[0]


def corp(*roles_per_passage):
    return {f"p{i}": {"source": [{"dataset": "germandpr", "split": "train", "role": r} for r in roles]}
            for i, roles in enumerate(roles_per_passage)}


def test_nhom_da_thay_doan():
    c = corp(["positive"], ["negative"], [], ["negative", "positive"])
    c["p2"]["source"].append({"dataset": "germanquad", "split": "train", "role": "positive"})    # nguồn khác không tính
    c["p4"] = {"source": [{"dataset": "germandpr", "split": "test", "role": "positive"}]}         # split test của DPR không tính
    c["p5"] = {}
    assert [passage_role(c[f"p{i}"]) for i in range(6)] == ["positive", "negative", "never", "positive", "never", "never"]
    assert question_group(["p0", "p2"], c) == "positive"
    assert question_group(["p1", "p2"], c) == "negative"
    assert question_group(["p2", "p4", "p5"], c) == "never"
    assert question_group(["p1", "p3"], c) == "positive"


# ---------- phân tích ----------
def test_did_ci_va_kiem_toan_ven():
    import numpy as np
    from germanqa.rerank import analysis as A
    d, lo, hi = A.did_ci([1, 1, 0, 1] * 50, [0, 0, 0, 1] * 50)
    assert d == pytest.approx(0.5) and lo < d < hi and lo > 0.3
    assert A.did_ci([0.1] * 30, [0.1] * 40)[0] == pytest.approx(0.0)
    pool = {"q1": [f"p{i}" for i in range(20)]}
    good = {"q1": {"passage_ids": pool["q1"], "score": [float(i) for i in range(20)], "first": [float(i) - 0.5 for i in range(20)],
                   "windows": [1] * 20}}
    assert A.integrity_problems(good, pool, ["q1"]) == []
    bad = {"q1": {**good["q1"], "score": [float("nan")] + good["q1"]["score"][1:]}}
    assert any("hữu hạn" in p for p in A.integrity_problems(bad, pool, ["q1"]))
    bad2 = {"q1": {**good["q1"], "passage_ids": list(reversed(pool["q1"]))}}
    assert any("pool khác" in p for p in A.integrity_problems(bad2, pool, ["q1"]))
    assert any("thiếu" in p for p in A.integrity_problems({}, pool, ["q1"]))
    low = {"q1": {**good["q1"], "score": [x - 5 for x in good["q1"]["score"]]}}
    assert any("lớn nhất <" in p for p in A.integrity_problems(low, pool, ["q1"]))


def test_load_scores_chi_tra_ve_cau_duoc_phep(tmp_path):
    import json
    from germanqa.rerank import analysis as A
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps({"question_id": q, "passage_ids": [], "score": [], "first": [], "windows": []})
                           for q in ("a", "b", "c")) + "\n", encoding="utf-8")
    assert set(A.load_scores(p, ["a", "c"])) == {"a", "c"}                 # 'b' (dev-unseen giả) không bao giờ lọt vào
    p.write_text(p.read_text(encoding="utf-8") + json.dumps({"question_id": "a", "passage_ids": [], "score": [], "first": [], "windows": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        A.load_scores(p, ["a"])
