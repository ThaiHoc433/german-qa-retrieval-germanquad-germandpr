import pytest

from germanqa.evaluation.report import choose_span, first_reference_only, layer_summary, markdown_tables

EX = [{"id": "q1", "answers": [{"text": "Hund", "start": 0}, {"text": "der Hund", "start": 0}]},
      {"id": "q2", "answers": [{"text": "Katze", "start": 0}]}]
PRED = {"q1": {"text": "der Hund"}, "q2": {"text": "Maus"}}


def test_tham_chieu_dau_tien_giu_nguyen_phan_con_lai():
    out = first_reference_only(EX)
    assert [len(e["answers"]) for e in out] == [1, 1] and out[0]["answers"][0]["text"] == "Hund"
    assert len(EX[0]["answers"]) == 2                                # không sửa dữ liệu gốc


def test_cham_nhieu_tham_chieu_cao_hon_cham_tham_chieu_dau_tien():
    s = layer_summary(PRED, EX, n_boot=50)
    assert s["all_refs"]["em"] == 0.5 and s["first_ref"]["em"] == 0.0
    assert s["all_refs"]["queries"] == s["first_ref"]["queries"] == 2


def test_khoang_tin_cay_chua_gia_tri_trung_binh():
    s = layer_summary(PRED, EX, n_boot=50)["all_refs"]
    assert s["em_ci"][0] <= s["em"] <= s["em_ci"][1]


CANDS = [{"start": 3, "reader_score": 2.0}, {"start": -1, "reader_score": 99.0},
         {"start": 5, "reader_score": 2.0}, {"start": 7, "reader_score": 9.0}]


def test_choose_span_chi_xet_k_doan_dau_va_bo_doan_khong_hop_le():
    assert choose_span(CANDS, 3)["start"] == 3                       # hòa điểm 2.0: giữ đoạn đứng trước
    assert choose_span(CANDS, 4)["start"] == 7
    assert choose_span(CANDS[1:2], 1) is None


def test_bang_markdown_co_du_hang():
    ci = {"em": 0.5, "em_ci": [0.4, 0.6], "f1": 0.6, "f1_ci": [0.5, 0.7], "queries": 2}
    summary = {"reader_gold_context": {"all_refs": ci, "first_ref": ci}, "end_to_end": {"all_refs": ci, "first_ref": ci},
               "retrieval": {"hit@1": .3, "hit@5": .6, "hit@10": .7, "hit@20": .8, "mrr@10": .45},
               "timing": {k: {"mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 2.0} for k in ("retrieval_time", "reader_time", "total_time")}}
    md = markdown_tables("dev", summary)
    assert "0,500 [0,400; 0,600]" in md and "Hit@20" in md and md.count("| End-to-end |") == 2