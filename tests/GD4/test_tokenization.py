import pytest

from germanqa.tokenization import load_bert_tokenizer

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
         "wer", "ist", "der", "könig", "kon", "##den", "##sator", "?"]


@pytest.fixture
def vocab(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("\n".join(WORDS) + "\n", encoding="utf-8")
    return p


def test_kich_thuoc_tu_vung_dung(vocab):
    assert len(load_bert_tokenizer(vocab)) == len(WORDS)


def test_tu_da_biet_khong_thanh_unk_va_giu_umlaut(vocab):
    tok = load_bert_tokenizer(vocab)
    ids = tok("wer ist der könig ?", add_special_tokens=False)["input_ids"]
    assert tok.unk_token_id not in ids
    assert len(ids) == 5


def test_tu_dai_duoc_tach_manh(vocab):
    tok = load_bert_tokenizer(vocab)
    toks = tok.tokenize("kondensator")
    assert toks == ["kon", "##den", "##sator"]


def test_offset_mapping_dung_ky_tu(vocab):
    tok = load_bert_tokenizer(vocab)
    text = "wer ist der könig"
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    spans = [text[a:b] for a, b in enc["offset_mapping"]]
    assert spans == ["wer", "ist", "der", "könig"]



def test_tra_ve_token_type_ids_cho_cap_cau(vocab):
    tok = load_bert_tokenizer(vocab)
    enc = tok("wer ist", "der könig")
    assert enc["token_type_ids"] == [0, 0, 0, 0, 1, 1, 1]
