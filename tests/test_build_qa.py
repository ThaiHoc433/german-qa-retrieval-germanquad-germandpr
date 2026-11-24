from germanqa.data.build_corpus import build_corpus
from germanqa.data.build_qa import build_qa
from germanqa.data.text_utils import make_question_id

CTX1 = "Guam\nDer Hund bellt laut im Garten."
CTX2 = "Beyoncé\nDie Katze schläft auf dem Sofa."
CTX3 = "ja ja"


def _quad(rows):
    return {"train": rows, "test": []}


def _row(i, ctx, q, texts, starts):
    return {"id": str(i), "context": ctx, "question": q,
            "answers": {"text": texts, "answer_start": starts}}


def _dpr(q, answers, pos, hard):
    c = lambda texts: {"title": ["t"] * len(texts), "text": texts, "passage_id": [""] * len(texts)}
    return {"question": q, "answers": answers, "positive_ctxs": c([pos]),
            "negative_ctxs": c([]), "hard_negative_ctxs": c(hard)}


def _raw():
    quad = _quad([
        _row(1, CTX1, "Wer bellt?", ["Hund"], [99]),               # offset sai -> FIXED
        _row(2, CTX1, "Wer  bellt? ", ["bellt"], [CTX1.index("bellt")]),  # trùng câu hỏi, cùng đoạn
        _row(3, CTX3, "Was ist das?", ["ja"], [50]),                # mơ hồ -> cách ly
        _row(4, CTX2, "Wer schläft?", ["Katze"], [CTX2.index("Katze")]),  # OK
    ])
    dpr = {"train": [_dpr("Wer schläft?", ["Katze"], CTX2, [CTX1, CTX2])], "test": []}
    return {"germanquad": quad, "germandpr": dpr}


def test_span_fixed_va_quarantine():
    pools, qrels, hardneg, quarantine, fixlog, stats = build_qa(_raw())
    assert stats["span_status"]["FIXED"] == 1 and len(fixlog) == 1
    assert stats["span_status"]["QUARANTINE_AMBIGUOUS"] == 1
    assert any(q["scope"] == "question" for q in quarantine)       # câu hỏi mất hết đáp án
    assert make_question_id("Was ist das?") not in pools["train"]


def test_cau_hoi_trung_cung_doan_duoc_gop_dap_an():
    pools, *_, stats = build_qa(_raw())
    row = pools["train"][make_question_id("Wer bellt?")]
    assert {a["text"] for a in row["answers"]} == {"Hund", "bellt"}
    assert stats["duplicate_same_passage_merged"] == 1


def test_dpr_chi_cho_hard_negative_mac_dinh():
    pools, qrels, hardneg, *_ = build_qa(_raw())
    assert len(pools["train"]) == 2                                  # chỉ 2 câu QuAD hợp lệ
    qid = make_question_id("Wer schläft?")
    assert len(hardneg[qid]) == 1                                    # CTX2 là đoạn đúng, bị loại
    assert qid in pools["train"] and pools["train"][qid]["origin"] == "germanquad"


def test_dpr_qa_khi_bat_co():
    pools, *_, stats = build_qa(_raw(), include_dpr_qa=True)
    assert stats["duplicate_same_passage_merged"] == 2               # +1 do câu DPR trùng câu QuAD


def test_corpus_loai_trung_va_giu_provenance():
    corpus, stats = build_corpus(_raw())
    assert stats["passages_after_dedup"] == 3                        # CTX1, CTX2, CTX3
    ctx2 = next(r for r in corpus if r["original_text"] == CTX2)
    assert {s["dataset"] for s in ctx2["source"]} == {"germanquad", "germandpr"}
    assert ctx2["title"] == "Beyoncé"