from germanqa.evaluation.errors import article_only, classify, relation, stage

PASSAGES = {"g": "Der Hund bellt laut im Garten.", "x": "Die Katze schläft leise auf dem Sofa.",
            "y": "Der Vogel singt. Radsport ist beliebt.", "z": "Ein anderer Text ohne Bezug."}


def row(retrieved, selected, answer):
    return {"retrieved": [{"passage_id": p} for p in retrieved], "passage_id": selected, "answer": answer}


def test_stage_bon_loai():
    ids = ["a", "b", "c", "d", "e", "f", "g"]
    assert stage(ids, "a", {"zzz"}) == "retrieval_failure"           # đoạn đúng không ở top-20
    assert stage(ids, "a", {"g"}) == "selection_failure"             # hạng 7: có ở top-20, vắng ở top-5
    assert stage(ids, "a", {"c"}) == "wrong_passage"                 # có ở top-5 nhưng reader chọn đoạn khác
    assert stage(ids, "c", {"c"}) == "wrong_span_gold_passage"       # chọn đúng đoạn, sai span


def test_stage_ranh_gioi_top_5_va_top_20():
    ids = [f"p{i}" for i in range(1, 21)]
    assert stage(ids, "p1", {"p5"}) == "wrong_passage"
    assert stage(ids, "p1", {"p6"}) == "selection_failure"
    assert stage(ids, "p1", {"p20"}) == "selection_failure"
    assert stage(ids[:19] + ["q"], "p1", {"p20"}) == "retrieval_failure"


def test_quan_he_dap_an():
    refs = ["der gelbe Hund"]
    assert relation("", refs) == "empty"
    assert relation("gelbe Hund", refs) == "inside_gold"             # thiếu chữ
    assert relation("Hund", refs) == "inside_gold"
    assert relation("heute der gelbe Hund bellt", refs) == "contains_gold"    # thừa chữ
    assert relation("ein gelber Vogel", refs) == "disjoint"
    assert relation("der Hund schläft", refs) == "partial_overlap"


def test_khong_khop_giua_chung_mot_tu():
    assert relation("ell", ["Pforzheimer Kellerei"]) == "disjoint"   # 'ell' không phải từ trong đáp án


def test_chon_tham_chieu_gan_nhat():
    assert relation("Radsport", ["Fußball", "Radsport und mehr"]) == "inside_gold"


def test_chi_khac_mao_tu():
    assert article_only("der Radsport", ["Radsport"])
    assert article_only("Radsport", ["der Radsport"])
    assert not article_only("Radsport", ["Radsport"])
    assert not article_only("Fußball", ["Radsport"])


def test_classify_lay_co_dap_an_trong_doan_khac():
    refs = ["Radsport"]
    r = classify(row(["x", "y", "g"], "y", "Fußball"), ["g"], refs, PASSAGES)
    assert r["stage"] == "wrong_passage"
    assert r["answer_in_other_retrieved"] is True                    # đoạn y (ngoài qrels) có 'Radsport'
    assert r["answer_in_selected_passage"] is True


def test_classify_dap_an_sai_that_su():
    r = classify(row(["x", "z", "g"], "x", "Fußball"), ["g"], ["Radsport"], PASSAGES)
    assert r["answer_in_other_retrieved"] is False and r["answer_in_selected_passage"] is False
    assert r["relation"] == "disjoint" and r["f1"] == 0.0


def test_classify_dung_doan_sai_span_khong_co_co_doan_chon():
    r = classify(row(["g", "x"], "g", "der Hund bellt"), ["g"], ["bellt"], PASSAGES)
    assert r["stage"] == "wrong_span_gold_passage" and r["answer_in_selected_passage"] is False
    assert r["relation"] == "contains_gold"
