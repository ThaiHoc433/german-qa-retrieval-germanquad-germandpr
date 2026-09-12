import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answer_parser import parse_llm_output

ANSWERED = (
    "KURZ: Albert Einstein\n"
    "ANTWORT: Albert Einstein entwickelte die Relativitaetstheorie.\n"
    "QUELLE: P2\n"
    'ZITAT: "Albert Einstein entwickelte die Relativitaetstheorie 1915."\n'
)

NO_ANSWER = "KURZ: KEINE ANTWORT\nANTWORT: None\nQUELLE: None\nZITAT: None\n"


def test_parses_answered_shape():
    parsed = parse_llm_output(ANSWERED)
    assert parsed.format_violation is False
    assert parsed.answered is True
    assert parsed.short_answer == "Albert Einstein"
    assert parsed.quelle_refs == ("P2",)
    assert parsed.quote == "Albert Einstein entwickelte die Relativitaetstheorie 1915."


def test_parses_no_answer_shape():
    parsed = parse_llm_output(NO_ANSWER)
    assert parsed.format_violation is False
    assert parsed.answered is False


def test_no_answer_shape_with_extra_citation_is_format_violation():
    bad = "KURZ: KEINE ANTWORT\nANTWORT: None\nQUELLE: P1\nZITAT: None\n"
    assert parse_llm_output(bad).format_violation is True


def test_missing_kurz_or_antwort_is_format_violation():
    assert parse_llm_output('ANTWORT: x\nQUELLE: P1\nZITAT: "x"\n').format_violation is True
    assert parse_llm_output('KURZ: x\nQUELLE: P1\nZITAT: "x"\n').format_violation is True


def test_duplicate_label_is_format_violation():
    dup = ANSWERED + "KURZ: nochmal\n"
    assert parse_llm_output(dup).format_violation is True


def test_answered_shape_missing_citation_is_valid_parse_not_format_violation():
    # design doc §2.2: model bỏ QUELLE/ZITAT dù được yêu cầu -> vẫn parse hợp lệ,
    # citation.verify_citation() sẽ gán citation_status=missing (Task 4), KHÔNG phải
    # format_violation.
    partial = "KURZ: Albert Einstein\nANTWORT: Albert Einstein.\n"
    parsed = parse_llm_output(partial)
    assert parsed.format_violation is False
    assert parsed.answered is True
    assert parsed.quelle_refs == ()
    assert parsed.quote is None


def test_multiple_quelle_refs_parsed():
    multi = ANSWERED.replace("QUELLE: P2", "QUELLE: P1, P3")
    parsed = parse_llm_output(multi)
    assert parsed.quelle_refs == ("P1", "P3")


def test_empty_kurz_value_is_format_violation():
    # Finding 1: trước fix, "\s*" sau dấu ":" ăn qua cả dấu xuống dòng, khiến giá trị
    # rỗng của KURZ nuốt luôn dòng ANTWORT làm giá trị của chính nó thay vì trigger
    # format_violation.
    bad = 'KURZ: \nANTWORT: Berlin\nQUELLE: P1\nZITAT: "Berlin ist die Hauptstadt."\n'
    assert parse_llm_output(bad).format_violation is True


def test_empty_antwort_value_is_format_violation():
    bad = 'KURZ: Berlin\nANTWORT: \nQUELLE: P1\nZITAT: "Berlin ist die Hauptstadt."\n'
    assert parse_llm_output(bad).format_violation is True


def test_empty_quelle_value_is_not_format_violation():
    # Khác với KURZ/ANTWORT rỗng: QUELLE rỗng (nhưng KURZ/ANTWORT/ZITAT vẫn có) là ca
    # "thiếu citation, vẫn trả lời được" — giống test_answered_shape_missing_citation_...
    # — KHÔNG phải format_violation.
    partial = 'KURZ: Berlin\nANTWORT: Berlin ist die Hauptstadt.\nQUELLE: \nZITAT: "Berlin ist die Hauptstadt."\n'
    parsed = parse_llm_output(partial)
    assert parsed.format_violation is False
    assert parsed.answered is True
    assert parsed.quelle_refs == ()
    assert parsed.quote is None


def test_empty_zitat_value_is_not_format_violation():
    partial = "KURZ: Berlin\nANTWORT: Berlin ist die Hauptstadt.\nQUELLE: P1\nZITAT: \n"
    parsed = parse_llm_output(partial)
    assert parsed.format_violation is False
    assert parsed.answered is True
    assert parsed.quelle_refs == ()
    assert parsed.quote is None
