"""GĐ7, sửa các điểm lệch giữa mô tả và code (2026-11-13):
1. paired bootstrap ghép theo question ID, không theo vị trí;
2. ánh xạ 4 stage chi tiết về 3 loại lỗi của plan (retrieval / selection / reader);
3. nhãn nguyên nhân gốc cho mẫu đọc tay: kiểm tra và đếm;
4. khoảng tin cậy cho metric retrieval."""
import pytest

from germanqa.evaluation.bootstrap import paired_bootstrap
from germanqa.evaluation.paired import paired_bootstrap_by_id, retrieval_ci
from germanqa.evaluation.errors import (COARSE_STAGE, ROOT_CAUSES, STAGES, coarse_stage, count_labels,
                                        validate_labels, wilson_ci)


# ---- 1. paired bootstrap theo ID ----
def test_ghep_theo_id_khong_phu_thuoc_thu_tu():
    a = {"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": 1.0}
    b = {"q4": 0.0, "q3": 1.0, "q2": 0.0, "q1": 0.0}          # cùng câu, khác thứ tự chèn
    ids = sorted(a)
    expected = paired_bootstrap([a[q] for q in ids], [b[q] for q in ids], 200)
    assert paired_bootstrap_by_id(a, b, 200) == expected


def test_ghep_theo_id_bat_loi_lech_tap_cau_hoi():
    with pytest.raises(ValueError, match="q3"):
        paired_bootstrap_by_id({"q1": 1.0, "q3": 0.0}, {"q1": 1.0, "q2": 0.0}, 100)


def test_ghep_theo_vi_tri_sai_thu_tu_cho_ket_qua_sai_con_theo_id_thi_dung():
    """Minh họa vì sao cần sửa: cùng dữ liệu, đảo thứ tự một bên thì bản theo vị trí cho chênh khác hẳn."""
    a = {f"q{i}": float(i % 2) for i in range(100)}
    b = dict(a)                                                  # hai hệ thống giống hệt: chênh thật = 0
    assert paired_bootstrap_by_id(a, b, 100) == (0.0, 0.0, 0.0)
    shuffled_b = [b[q] for q in reversed(sorted(b))]
    d, lo, hi = paired_bootstrap([a[q] for q in sorted(a)], shuffled_b, 100)
    assert (lo, hi) != (0.0, 0.0)                                # theo vị trí: khoảng giả khác 0


# ---- 2. ba loại lỗi của plan ----
def test_moi_stage_co_dung_mot_loai_tho():
    assert set(COARSE_STAGE) == set(STAGES)
    assert coarse_stage("retrieval_failure") == "retrieval"
    assert coarse_stage("selection_failure") == "selection"
    assert coarse_stage("wrong_passage") == "reader"
    assert coarse_stage("wrong_span_gold_passage") == "reader"
    with pytest.raises(KeyError):
        coarse_stage("khong_co")


# ---- 3. nhãn nguyên nhân gốc ----
def _lab(qid, stage, primary, secondary=()):
    return {"question_id": qid, "stage": stage, "primary": primary, "secondary": list(secondary), "note": ""}


def test_validate_bao_cau_chua_gan_nhan_va_nhan_la():
    labels = [_lab("q1", "retrieval_failure", "lexical_mismatch"), _lab("q2", "retrieval_failure", None),
              _lab("q3", "wrong_passage", "khong_co_trong_codebook")]
    problems = validate_labels(labels)
    assert problems["unlabeled"] == ["q2"] and problems["unknown_cause"] == ["q3"]


def test_dem_nhan_theo_stage_co_ty_le_va_khoang_wilson():
    labels = [_lab(f"r{i}", "retrieval_failure", "lexical_mismatch") for i in range(10)] + \
             [_lab(f"s{i}", "retrieval_failure", "compound_or_inflection", ["lexical_mismatch"]) for i in range(30)]
    out = count_labels(labels)
    cell = out["retrieval_failure"]
    assert cell["n"] == 40
    assert cell["primary"]["lexical_mismatch"]["count"] == 10
    assert cell["primary"]["lexical_mismatch"]["share"] == pytest.approx(0.25)
    lo, hi = cell["primary"]["lexical_mismatch"]["wilson95"]
    assert lo < 0.25 < hi and hi - lo > 0.2                      # 40 mẫu: khoảng rộng, không phải ước lượng chính xác
    assert cell["any"]["lexical_mismatch"] == 40                 # đếm cả nhãn phụ (đa nhãn)


def test_dem_tu_choi_khi_con_cau_chua_gan_nhan():
    with pytest.raises(ValueError, match="chưa gán"):
        count_labels([_lab("q1", "retrieval_failure", None)])


def test_wilson_bien():
    assert wilson_ci(0, 40)[0] == 0.0 and wilson_ci(40, 40)[1] == pytest.approx(1.0)
    assert all(c in ROOT_CAUSES for c in ("lexical_mismatch", "other"))


# ---- 4. CI cho retrieval ----
def test_retrieval_ci_chua_trung_binh():
    rows = [{"question_id": f"q{i}", "retrieved": [{"passage_id": "g" if i % 2 else "x"}, {"passage_id": "g"}]}
            for i in range(50)]
    qrels = {f"q{i}": ["g"] for i in range(50)}
    ci = retrieval_ci(rows, qrels, [1, 5], n_boot=200)
    assert ci["hit@5"] == [1.0, 1.0]
    lo, hi = ci["hit@1"]
    assert lo <= 0.5 <= hi
    assert set(ci) >= {"hit@1", "recall@1", "mrr@1", "hit@5", "mrr@5"}
