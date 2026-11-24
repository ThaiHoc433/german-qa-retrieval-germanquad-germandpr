from germanqa.data.text_utils import (clean_title, extract_title, make_document_id,
                                 make_question_id, make_retrieval_text, normalize_text,
                                 passage_id_from_key, make_dedup_key)
from germanqa.schemas import to_utf16_offset


def test_normalize_text_gom_khoang_trang():
    assert normalize_text("a\u00a0 b\n  c ") == "a b c"


def test_retrieval_text_khong_sua_original():
    original = "Der\u200b  Hund\u00a0BELLT"
    assert make_retrieval_text(original) == "der hund bellt"
    assert original == "Der\u200b  Hund\u00a0BELLT"          # chuỗi gốc còn nguyên


def test_question_id_on_dinh():
    a = make_question_id("Wer  bellt? ")
    b = make_question_id("wer bellt?")
    assert a == b
    assert a.startswith("q_") and len(a) == 18
    assert make_question_id("Wer schläft?") != a


def test_passage_id_dang_p_16_ky_tu():
    pid = passage_id_from_key(make_dedup_key("irgendein text"))
    assert pid.startswith("p_") and len(pid) == 18


def test_extract_title():
    assert extract_title("Guam\nGuam ist eine Insel.") == "Guam"
    assert extract_title("Chỉ một dòng không xuống dòng") is None
    assert extract_title("x" * 200 + "\nvăn bản") is None        # dòng đầu quá dài, không phải tiêu đề


def test_clean_title_chi_lay_dong_dau():
    assert clean_title("Beyoncé\nBeyoncé Giselle ist eine Sängerin.") == "Beyoncé"
    assert clean_title("") is None and clean_title(None) is None


def test_document_id_cung_tieu_de_cung_id():
    k = make_dedup_key("a")
    assert make_document_id("Guam", k) == make_document_id("  guam ", make_dedup_key("b"))
    assert make_document_id(None, k) == "d_" + k[:12]
    k = make_dedup_key("a")
    assert make_document_id("Guam", k) == make_document_id("  guam ", make_dedup_key("b"))
    assert make_document_id(None, k) == "d_" + k[:12]


def test_utf16_offset():
    assert to_utf16_offset("abc", 2) == 2
    assert to_utf16_offset("a😀b", 2) == 3                   # emoji chiếm 2 đơn vị UTF-16
