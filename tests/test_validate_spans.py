from germanqa.data.validate_spans import find_all, validate

CTX = "Der Hund bellt. Die Katze schläft. Der Hund rennt."


def test_find_all():
    assert find_all(CTX, "Hund") == [4, 39]
    assert find_all(CTX, "Maus") == []


def test_ok_khi_offset_dung():
    assert validate(CTX, "Katze", 20) == ("OK", 20)


def test_fixed_khi_offset_sai_va_chi_xuat_hien_mot_lan():
    assert validate(CTX, "Katze", 3) == ("FIXED", 20)


def test_located_khi_khong_co_offset():
    assert validate(CTX, "Katze", None) == ("LOCATED", 20)


def test_ambiguous_khi_xuat_hien_nhieu_lan():
    assert validate(CTX, "Hund", 0) == ("QUARANTINE_AMBIGUOUS", None)
    assert validate(CTX, "Hund", None) == ("QUARANTINE_AMBIGUOUS", None)


def test_not_found_va_empty():
    assert validate(CTX, "Maus", 5) == ("QUARANTINE_NOT_FOUND", None)
    assert validate(CTX, "", 0) == ("QUARANTINE_EMPTY", None)


def test_offset_dung_nhung_chuoi_xuat_hien_nhieu_lan_van_ok():
    assert validate(CTX, "Hund", 39) == ("OK", 39)