import pytest
from germanqa.reader.features import answer_chars, label_window, prepare_eval_features, prepare_train_features
from germanqa.tokenization import load_bert_tokenizer

WORDS = ["Wer", "bellt", "Was", "Der", "Hund", "laut", "und", "die", "Katze", "schläft", "leise", ".", "?"]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    """Tokenizer WordPiece nhỏ: mỗi từ trong WORDS là đúng một token."""
    path = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS), encoding="utf-8")
    return load_bert_tokenizer(path)


CONTEXT = "Der Hund bellt laut . Die Katze schläft leise ."


def long_example():
    context = " ".join(["Der Hund bellt laut ."] * 6 + ["Die Katze schläft leise ."] + ["Der Hund bellt laut ."] * 6)
    start = context.index("Katze")
    return {"id": "q1", "question": "Was schläft ?", "context": context,
            "answers": [{"text": "Katze", "start": start}]}


def decode(feat, tok, a, b):
    return tok.decode(feat["input_ids"][a:b + 1])


def test_answer_chars_bo_khoang_trang_thua():
    assert answer_chars({"text": "  Hund ", "start": 3}) == (5, 9)


def test_nhan_tro_dung_token_dap_an(tok):
    ex = {"id": "q", "question": "Wer bellt ?", "context": "Der Hund bellt laut .",
          "answers": [{"text": "Hund", "start": 4}]}
    [f] = prepare_train_features([ex], tok, max_length=32, stride=4)
    assert decode(f, tok, f["start_positions"], f["end_positions"]) == "Hund"


def test_dap_an_nhieu_token(tok):
    ex = {"id": "q", "question": "Wer bellt ?", "context": "Der Hund bellt laut .",
          "answers": [{"text": "Hund bellt laut", "start": 4}]}
    [f] = prepare_train_features([ex], tok, max_length=32, stride=4)
    assert decode(f, tok, f["start_positions"], f["end_positions"]) == "Hund bellt laut"


def test_dap_an_co_dau_cach_cuoi_van_dung(tok):
    ex = {"id": "q", "question": "Wer bellt ?", "context": "Der Hund bellt laut .",
          "answers": [{"text": "Hund ", "start": 4}]}
    [f] = prepare_train_features([ex], tok, max_length=32, stride=4)
    assert decode(f, tok, f["start_positions"], f["end_positions"]) == "Hund"


def test_cua_so_khong_chua_dap_an_co_nhan_cls(tok):
    ex = long_example()
    feats = prepare_train_features([ex], tok, max_length=16, stride=4)
    assert len(feats) > 1
    labelled = [f for f in feats if f["start_positions"] != 0]
    assert labelled and len(labelled) < len(feats)
    for f in feats:
        if f["start_positions"] == 0:
            assert f["end_positions"] == 0
    for f in labelled:
        assert decode(f, tok, f["start_positions"], f["end_positions"]) == "Katze"


def test_dap_an_bi_cat_doi_giua_hai_cua_so_khong_duoc_gan_nhan(tok):
    # nhãn chỉ được gán khi TOÀN BỘ đáp án nằm trong cửa sổ
    offsets = [(0, 0), (0, 3), (4, 8), (0, 0), (9, 13), (14, 18), (0, 0)]
    seq_ids = [None, 0, 0, None, 1, 1, None]
    assert label_window(offsets, seq_ids, 10, 17) == (4, 5)
    assert label_window(offsets, seq_ids, 6, 17) == (0, 0)     # đáp án bắt đầu trước cửa sổ
    assert label_window(offsets, seq_ids, 10, 25) == (0, 0)    # đáp án kết thúc sau cửa sổ


def test_cac_cua_so_phu_kin_toan_bo_context(tok):
    ex = long_example()
    feats = prepare_eval_features([ex], tok, max_length=16, stride=4)
    covered = set()
    for f in feats:
        for off in f["offsets"]:
            if off is not None:
                covered.update(range(off[0], off[1]))
    words = [i for i, ch in enumerate(ex["context"]) if ch != " "]
    assert set(words) <= covered


def test_eval_chi_giu_offset_cua_token_context(tok):
    ex = {"id": "q", "question": "Wer bellt ?", "context": "Der Hund bellt laut .", "answers": []}
    [f] = prepare_eval_features([ex], tok, max_length=32, stride=4)
    n_ctx = sum(1 for o in f["offsets"] if o is not None)
    assert n_ctx == 5                                      # Der Hund bellt laut .
    assert f["offsets"][0] is None                         # [CLS]


def test_example_index_dung_voi_nhieu_vi_du(tok):
    a = long_example()
    b = {"id": "q2", "question": "Wer bellt ?", "context": "Der Hund bellt laut .",
         "answers": [{"text": "Hund", "start": 4}]}
    feats = prepare_train_features([a, b], tok, max_length=16, stride=4)
    assert {f["example_index"] for f in feats} == {0, 1}
    assert [f["example_index"] for f in feats] == sorted(f["example_index"] for f in feats)