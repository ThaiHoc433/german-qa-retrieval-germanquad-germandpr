# RAG Answer Generation (V2, phần không phụ thuộc V1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây phần lõi của bộ trả lời sinh câu (RAG) có trích dẫn kiểm chứng được —
`citation.py`, `answer_parser.py`, `prompt_loader.py`, `llm_clients.py`, `answerer.py` —
đủ để `LLMAnswerer` chạy hết trên fixture/fake client, không cần Reader V1 hay corpus
thật đã build.

**Architecture:** Pipeline nhỏ, hàm thuần trước — model output (text hoặc citations
block của Claude) → parse thành `ParsedAnswer` chuẩn hóa → `citation.verify_citation()`
kiểm chứng bằng cách tìm chuỗi con trong đúng passage → `LLMAnswerer` ghép tất cả thành
`AnswerResult`. Không file nào trong plan này gọi mạng thật trong test suite; `LLMClient`
là inject được (constructor nhận object, không tự tạo client thật bên trong lúc test).

**Tech Stack:** Python 3.13, pytest (đã cài), thư viện chuẩn (`re`, `unicodedata`,
`hashlib`, `dataclasses`, `typing`). Gói `anthropic` **không** cần cài để chạy test suite
của plan này — `import anthropic` chỉ nằm trong một factory function dùng khi chạy thật
(xem Task 7), nên toàn bộ `ClaudeApiClient`/test dùng client được inject (stub), không
đụng SDK thật.

**Spec:** [`../specs/2026-09-12-rag-answer-generation-design.md`](../specs/2026-09-12-rag-answer-generation-design.md)
§1, §2, §7 (chỉ §7.2 mục 1–4). Mục 5–8 của §7.2 (ExtractiveAnswerer, bucket 7 nhánh mở
rộng, FastAPI app, thực nghiệm RQ4 thật) **ngoài phạm vi plan này** — phụ thuộc Reader V1,
`evaluation.py` V1.1 và Integration (Chặng 9) chưa tới.

## Global Constraints

- Không sửa module spec V1.1 (`docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md`)
  hay bất kỳ file nào trong `src/`, `tests/`, `notebooks/` thuộc V1 hiện có.
- Không dùng fuzzy match khi tìm `quote`/`ZITAT` trong passage — chỉ chuỗi con chính xác,
  hoặc khớp sau chuẩn hóa NFC + gộp khoảng trắng (thiết kế §2.4 bước 3).
- Không tự sửa/tự chọn khi model trích sai passage hay trích nhiều nguồn — báo trạng
  thái, không đoán thay model (thiết kế §2.4 bước 2, 4).
- `generation_status` và `citation_status` tách biệt hoàn toàn, không trộn lẫn (thiết kế
  §2.4).
- SHA-256 của prompt dùng **hex đầy đủ 64 ký tự**, không cắt ngắn (khác `passage_id` của
  V1 vốn cắt còn 16 ký tự — mục đích khác nhau: fingerprint cần chính xác tuyệt đối).
- Không gọi mạng thật trong bất kỳ test nào của plan này (Constitution + thiết kế §7.1).
- Docstring/comment bằng tiếng Việt, tham chiếu số mục design doc, theo đúng phong cách
  `src/*.py` hiện có (`src/corpus.py`, `src/evaluation.py`, `src/schemas.py`).
- File mới đặt tại `src/` và `tests/` theo đúng vị trí chuẩn của repo (không đặt trong
  `future_RAG/`) — `future_RAG/` chỉ giữ tài liệu thiết kế/kế hoạch.
- Test theo TDD: viết test thất bại trước, cài tối thiểu để pass, không tune ngược lại
  test bằng cách nới lỏng assertion.

---

## Task 1: `citation.py` — tìm `quote` trong một passage (chính xác + chuẩn hóa)

**Files:**
- Create: `src/citation.py`
- Test: `tests/test_citation.py`

**Interfaces:**
- Consumes: không gì (chỉ dùng thư viện chuẩn)
- Produces: `PassageInput(passage_id: str, title: str | None, text: str)`,
  `QuoteMatch(quote_start: int, quote_end: int, quote_match_method: str, quote_occurrence_count: int)`,
  `find_quote_in_passage(passage_text: str, quote: str) -> QuoteMatch | None` — dùng bởi
  Task 4 (`verify_citation`)

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_citation.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.citation import find_quote_in_passage


def test_exact_substring_match():
    text = "Berlin ist die Hauptstadt Deutschlands. Es liegt an der Spree."
    match = find_quote_in_passage(text, "Es liegt an der Spree.")
    assert match is not None
    assert match.quote_match_method == "exact"
    assert text[match.quote_start:match.quote_end] == "Es liegt an der Spree."
    assert match.quote_occurrence_count == 1


def test_no_match_returns_none():
    assert find_quote_in_passage("Berlin ist die Hauptstadt.", "Paris ist die Hauptstadt.") is None


def test_multiple_occurrences_counts_and_takes_first():
    text = "Die Katze schlaeft. Die Katze spielt. Die Katze schlaeft wieder."
    match = find_quote_in_passage(text, "Die Katze schlaeft")
    assert match is not None
    assert match.quote_occurrence_count == 2
    assert match.quote_start == 0


def test_normalized_match_handles_extra_whitespace():
    text = "Der Fluss   Spree fliesst durch Berlin."
    match = find_quote_in_passage(text, "Der Fluss Spree fliesst durch Berlin.")
    assert match is not None
    assert match.quote_match_method == "normalized"
    assert match.quote_start == 0
    assert match.quote_end == len(text)


def test_normalized_match_does_not_fuzzy_match_different_words():
    text = "Der Fluss Spree fliesst durch Berlin."
    assert find_quote_in_passage(text, "Der Fluss Spree fliesst durch Hamburg.") is None
```

(Dùng chữ không dấu `fliesst`/`schlaeft` trong test để tránh phụ thuộc encoding khi chạy
trên các terminal khác nhau — không ảnh hưởng logic NFC vì đây chỉ là ASCII thuần.)

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_citation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.citation'`

- [ ] **Step 3: Viết code tối thiểu**

```python
# src/citation.py
"""Kiểm chứng trích dẫn — future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md §2.4.

Chỉ hàm thuần, không gọi model. Nhận passage đã có sẵn (đầu ra Selection của pipeline V1)
và field đã được answer_parser.py tách ra từ output model; tự tìm `quote` trong đúng
passage, tính vị trí ký tự, không tự sửa khi model trích sai passage (wrong_passage).
"""
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class PassageInput:
    """Một trong tối đa 5 passage đưa vào prompt — dùng chung cho render prompt (module
    prompt_loader) và kiểm chứng trích dẫn (module này)."""
    passage_id: str
    title: str | None
    text: str


@dataclass(frozen=True)
class QuoteMatch:
    """Vị trí một đoạn trích trong MỘT văn bản cụ thể — dùng cho cả quote-trong-passage
    (Task 1) và short_answer-trong-quote (Task 2)."""
    quote_start: int
    quote_end: int  # exclusive
    quote_match_method: str  # "exact" | "normalized"
    quote_occurrence_count: int


def find_quote_in_passage(passage_text: str, quote: str) -> QuoteMatch | None:
    """Tìm `quote` như một chuỗi con của `passage_text`. Thử khớp chính xác trước; nếu
    không có, thử khớp theo từng từ sau khi chuẩn hóa NFC (không xóa dấu câu, không xóa
    hoa/thường — khác `normalize_answer` của EM/F1). KHÔNG fuzzy match: hai mức trên là
    toàn bộ những gì được chấp nhận (spec §2.4 bước 3)."""
    exact_positions = [m.start() for m in re.finditer(re.escape(quote), passage_text)]
    if exact_positions:
        start = exact_positions[0]
        return QuoteMatch(start, start + len(quote), "exact", len(exact_positions))
    return _find_quote_normalized(passage_text, quote)


def _find_quote_normalized(passage_text: str, quote: str) -> QuoteMatch | None:
    quote_tokens = [unicodedata.normalize("NFC", w) for w in quote.split()]
    if not quote_tokens:
        return None

    passage_tokens = [(m, unicodedata.normalize("NFC", m.group())) for m in re.finditer(r"\S+", passage_text)]
    passage_words = [w for _, w in passage_tokens]

    n = len(quote_tokens)
    occurrences = [
        i for i in range(len(passage_words) - n + 1)
        if passage_words[i : i + n] == quote_tokens
    ]
    if not occurrences:
        return None

    first = occurrences[0]
    start = passage_tokens[first][0].start()
    end = passage_tokens[first + n - 1][0].end()
    return QuoteMatch(start, end, "normalized", len(occurrences))
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_citation.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/citation.py tests/test_citation.py
git commit -m "feat(rag): add exact/normalized quote matching in citation.py"
```

---

## Task 2: `citation.py` — kiểm tra `short_answer` nằm trong `quote`

**Files:**
- Modify: `src/citation.py` (thêm cuối file)
- Modify: `tests/test_citation.py` (thêm cuối file)

**Interfaces:**
- Consumes: `src.evaluation.normalize_answer` (đã có trong V1, không sửa)
- Produces: `is_short_answer_supported(quote: str, short_answer: str) -> bool`,
  `find_short_answer_offsets(quote: str, short_answer: str) -> QuoteMatch | None` — dùng
  bởi Task 4

- [ ] **Step 1: Viết test thất bại**

```python
# thêm vào cuối tests/test_citation.py
from src.citation import find_short_answer_offsets, is_short_answer_supported


def test_short_answer_supported_exact_substring():
    quote = "Albert Einstein entwickelte die Relativitaetstheorie."
    assert is_short_answer_supported(quote, "Albert Einstein") is True
    offsets = find_short_answer_offsets(quote, "Albert Einstein")
    assert offsets is not None
    assert quote[offsets.quote_start:offsets.quote_end] == "Albert Einstein"


def test_short_answer_not_in_quote():
    quote = "Marie Curie erhielt den Nobelpreis."
    assert is_short_answer_supported(quote, "Albert Einstein") is False
    assert find_short_answer_offsets(quote, "Albert Einstein") is None


def test_short_answer_supported_uses_em_f1_style_normalization():
    # normalize_answer (NFC + lowercase + xoa dau cau) -> "1300." va "1300" duoc coi la
    # ho tro, khac voi cach chuan hoa NFC+khoang-trang dung khi tim quote trong passage.
    quote = "Es gibt etwa 1300 ethnische Gruppen."
    assert is_short_answer_supported(quote, "1300.") is True
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_citation.py -v`
Expected: FAIL — `ImportError: cannot import name 'find_short_answer_offsets'`

- [ ] **Step 3: Viết code tối thiểu**

```python
# thêm vào cuối src/citation.py
from src.evaluation import normalize_answer


def is_short_answer_supported(quote: str, short_answer: str) -> bool:
    """§2.4 bước 5: dùng CÙNG cách chuẩn hóa EM/F1 của V1 (normalize_answer: NFC,
    lowercase, xóa dấu câu, gộp khoảng trắng) — KHÁC cách chuẩn hóa NFC+khoảng-trắng
    dùng khi tìm quote trong passage (find_quote_in_passage)."""
    norm_quote = normalize_answer(quote)
    norm_answer = normalize_answer(short_answer)
    return bool(norm_answer) and norm_answer in norm_quote


def find_short_answer_offsets(quote: str, short_answer: str) -> QuoteMatch | None:
    """Vị trí ký tự CHÍNH XÁC của short_answer trong quote GỐC (chưa chuẩn hóa) — chỉ có
    giá trị khi short_answer là chuỗi con chính xác. `is_short_answer_supported` vẫn có
    thể True dù hàm này trả None (khớp chỉ sau chuẩn hóa EM/F1, không có offset chính
    xác) — hai hàm độc lập, không suy hàm này từ hàm kia."""
    positions = [m.start() for m in re.finditer(re.escape(short_answer), quote)]
    if not positions:
        return None
    start = positions[0]
    return QuoteMatch(start, start + len(short_answer), "exact", len(positions))
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_citation.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/citation.py tests/test_citation.py
git commit -m "feat(rag): add short_answer-in-quote check (EM/F1-style normalization)"
```

---

## Task 3: `answer_parser.py` — parse output 4 dòng (KURZ/ANTWORT/QUELLE/ZITAT)

**Files:**
- Create: `src/answer_parser.py`
- Test: `tests/test_answer_parser.py`

**Interfaces:**
- Consumes: không gì (thư viện chuẩn)
- Produces: `ParsedAnswer(format_violation: bool, answered: bool | None, short_answer:
  str | None, answer: str | None, quelle_refs: tuple[str, ...], quote: str | None)`,
  `parse_llm_output(raw_text: str) -> ParsedAnswer` — dùng bởi Task 4 (test) và Task 8
  (`LLMAnswerer`, nhánh `model_backend="open"`)

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_answer_parser.py
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
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_answer_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.answer_parser'`

- [ ] **Step 3: Viết code tối thiểu**

```python
# src/answer_parser.py
"""Parser cho định dạng output 4 dòng KURZ/ANTWORT/QUELLE/ZITAT —
future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md §2.2.

Dùng cho model_backend="open" (và FakeLLMClient trong test — xem llm_clients.py).
model_backend="claude" KHÔNG dùng parser này: Claude chỉ được yêu cầu trả KURZ/ANTWORT;
QUELLE/ZITAT lấy từ citations block của API (xem llm_clients.claude_generation_to_parsed_answer),
không phải từ text tự do.
"""
import re
from dataclasses import dataclass

_LABEL_RE = re.compile(r"^(KURZ|ANTWORT|QUELLE|ZITAT):\s*(.*)$", re.MULTILINE)
_QUELLE_REF_RE = re.compile(r"P(\d+)")


@dataclass(frozen=True)
class ParsedAnswer:
    """Kết quả parse, đã chuẩn hóa về một dạng chung cho cả model_backend="open" và
    "claude" (xem llm_clients.claude_generation_to_parsed_answer). Chỉ đọc các field
    answered/short_answer/... khi format_violation=False."""
    format_violation: bool
    answered: bool | None = None
    short_answer: str | None = None       # từ KURZ
    answer: str | None = None             # từ ANTWORT
    quelle_refs: tuple[str, ...] = ()      # ví dụ ("P1",) hoặc ("P1", "P3")
    quote: str | None = None              # từ ZITAT, đã bỏ dấu ngoặc kép hai đầu


def parse_llm_output(raw_text: str) -> ParsedAnswer:
    labels: dict[str, str] = {}
    for match in _LABEL_RE.finditer(raw_text):
        key = match.group(1)
        if key in labels:  # nhãn lặp lại -> vi phạm định dạng, không lấy giá trị đầu
            return ParsedAnswer(format_violation=True)
        labels[key] = match.group(2).strip()

    kurz = labels.get("KURZ", "")
    antwort = labels.get("ANTWORT", "")
    if not kurz or not antwort:
        return ParsedAnswer(format_violation=True)

    if kurz == "KEINE ANTWORT":
        if antwort == "None" and labels.get("QUELLE", "") == "None" and labels.get("ZITAT", "") == "None":
            return ParsedAnswer(format_violation=False, answered=False)
        return ParsedAnswer(format_violation=True)

    quelle_value = labels.get("QUELLE", "")
    zitat_value = labels.get("ZITAT", "").strip()
    if zitat_value.startswith('"') and zitat_value.endswith('"') and len(zitat_value) >= 2:
        zitat_value = zitat_value[1:-1]

    quelle_refs = tuple(f"P{n}" for n in _QUELLE_REF_RE.findall(quelle_value))
    has_quote = bool(zitat_value) and zitat_value != "None"
    if not quelle_refs or not has_quote:
        # thiếu một nửa (hoặc cả hai) của cặp QUELLE/ZITAT -> citation_status=missing ở
        # citation.verify_citation (Task 4), KHÔNG phải format_violation (design §2.2).
        quelle_refs = ()
        zitat_value = None

    return ParsedAnswer(
        format_violation=False, answered=True,
        short_answer=kurz, answer=antwort,
        quelle_refs=quelle_refs, quote=zitat_value,
    )
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_answer_parser.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/answer_parser.py tests/test_answer_parser.py
git commit -m "feat(rag): add KURZ/ANTWORT/QUELLE/ZITAT output parser"
```

---

## Task 4: `citation.py` — `verify_citation` (orchestrator đủ 6 trạng thái)

**Files:**
- Modify: `src/citation.py` (thêm cuối file)
- Modify: `tests/test_citation.py` (thêm cuối file)

**Interfaces:**
- Consumes: `PassageInput`, `find_quote_in_passage`, `is_short_answer_supported`,
  `find_short_answer_offsets` (Task 1–2, cùng file); `ParsedAnswer` (Task 3,
  `src.answer_parser`)
- Produces: `CitationDetail(passage_id: str, quote: str, quote_start: int, quote_end:
  int, quote_match_method: str, quote_occurrence_count: int, short_answer_start: int |
  None, short_answer_end: int | None)`, `CitationOutcome(citation_status: str,
  answer_supported: bool | None, citations: list[CitationDetail], raw_citation_refs:
  tuple[str, ...] | None = None)`, `verify_citation(passages: list[PassageInput], parsed:
  ParsedAnswer) -> CitationOutcome` — dùng bởi Task 8 (`LLMAnswerer`). **Chỉ gọi khi
  `parsed.answered is True`** — `LLMAnswerer` tự xử lý nhánh `answered=False`/
  `format_violation` TRƯỚC KHI gọi hàm này (design §2.4).

- [ ] **Step 1: Viết test thất bại**

```python
# thêm vào cuối tests/test_citation.py
from src.answer_parser import ParsedAnswer
from src.citation import PassageInput, verify_citation

P1 = PassageInput("p1", "Titel A", "Albert Einstein wurde 1879 geboren. Er entwickelte die Relativitaetstheorie.")
P2 = PassageInput("p2", "Titel B", "Marie Curie erhielt zwei Nobelpreise.")


def _parsed(**overrides):
    base = dict(
        format_violation=False, answered=True, short_answer="Albert Einstein",
        answer="Albert Einstein entwickelte die Relativitaetstheorie.",
        quelle_refs=("P1",), quote="Er entwickelte die Relativitaetstheorie.",
    )
    base.update(overrides)
    return ParsedAnswer(**base)


def test_verified_and_supported():
    outcome = verify_citation([P1, P2], _parsed())
    assert outcome.citation_status == "verified"
    assert outcome.answer_supported is True
    assert len(outcome.citations) == 1
    assert outcome.citations[0].passage_id == "p1"


def test_verified_but_not_supported():
    outcome = verify_citation([P1, P2], _parsed(short_answer="Marie Curie"))
    assert outcome.citation_status == "verified"
    assert outcome.answer_supported is False


def test_wrong_passage():
    # quote thật ra nằm ở P2, nhưng model ghi QUELLE: P1
    outcome = verify_citation([P1, P2], _parsed(quote="Marie Curie erhielt zwei Nobelpreise."))
    assert outcome.citation_status == "wrong_passage"
    assert outcome.answer_supported is None


def test_quote_not_found_anywhere():
    outcome = verify_citation([P1, P2], _parsed(quote="Das steht nirgendwo im Text."))
    assert outcome.citation_status == "quote_not_found"


def test_out_of_range_quelle():
    outcome = verify_citation([P1, P2], _parsed(quelle_refs=("P9",)))
    assert outcome.citation_status == "out_of_range"


def test_missing_citation():
    outcome = verify_citation([P1, P2], _parsed(quelle_refs=(), quote=None))
    assert outcome.citation_status == "missing"
    assert outcome.answer_supported is None


def test_multiple_sources_keeps_raw_refs_without_choosing():
    outcome = verify_citation([P1, P2], _parsed(quelle_refs=("P1", "P2")))
    assert outcome.citation_status == "multiple_sources"
    assert outcome.citations == []
    assert outcome.raw_citation_refs == ("P1", "P2")
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_citation.py -v`
Expected: FAIL — `ImportError: cannot import name 'verify_citation'`

- [ ] **Step 3: Viết code tối thiểu**

```python
# thêm vào cuối src/citation.py
from src.answer_parser import ParsedAnswer


@dataclass(frozen=True)
class CitationDetail:
    passage_id: str
    quote: str
    quote_start: int
    quote_end: int
    quote_match_method: str
    quote_occurrence_count: int
    short_answer_start: int | None
    short_answer_end: int | None


@dataclass(frozen=True)
class CitationOutcome:
    citation_status: str  # verified|wrong_passage|quote_not_found|out_of_range|missing|multiple_sources
    answer_supported: bool | None
    citations: list[CitationDetail]
    raw_citation_refs: tuple[str, ...] | None = None


def verify_citation(passages: list[PassageInput], parsed: ParsedAnswer) -> CitationOutcome:
    """§2.4 bước 2–5. Chỉ gọi khi parsed.answered is True — answered=False/
    format_violation được LLMAnswerer (answerer.py, Task 8) xử lý TRƯỚC, không đi qua
    hàm này."""
    if len(parsed.quelle_refs) > 1:
        return CitationOutcome("multiple_sources", None, [], raw_citation_refs=parsed.quelle_refs)
    if not parsed.quelle_refs or parsed.quote is None:
        return CitationOutcome("missing", None, [])

    ref = parsed.quelle_refs[0]
    index = int(ref[1:]) - 1  # "P1" -> 0
    if not (0 <= index < len(passages)):
        return CitationOutcome("out_of_range", None, [])

    target = passages[index]
    match = find_quote_in_passage(target.text, parsed.quote)
    if match is None:
        for other_index, other in enumerate(passages):
            if other_index == index:
                continue
            if find_quote_in_passage(other.text, parsed.quote) is not None:
                return CitationOutcome("wrong_passage", None, [])  # KHÔNG tự sửa QUELLE
        return CitationOutcome("quote_not_found", None, [])

    supported = None
    sa_start = sa_end = None
    if parsed.short_answer is not None:
        supported = is_short_answer_supported(parsed.quote, parsed.short_answer)
        offsets = find_short_answer_offsets(parsed.quote, parsed.short_answer)
        if offsets is not None:
            sa_start, sa_end = offsets.quote_start, offsets.quote_end

    detail = CitationDetail(
        passage_id=target.passage_id, quote=parsed.quote,
        quote_start=match.quote_start, quote_end=match.quote_end,
        quote_match_method=match.quote_match_method,
        quote_occurrence_count=match.quote_occurrence_count,
        short_answer_start=sa_start, short_answer_end=sa_end,
    )
    return CitationOutcome("verified", supported, [detail])
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_citation.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/citation.py tests/test_citation.py
git commit -m "feat(rag): add verify_citation orchestrator (6 citation_status branches)"
```

---

## Task 5: Prompt files + `prompt_loader.py` (version + hash SHA-256)

**Files:**
- Create: `src/prompts/answer_de_v1.txt`
- Create: `src/prompts/answer_de_closed_book_v1.txt`
- Create: `src/prompt_loader.py`
- Test: `tests/test_prompt_loader.py`

**Interfaces:**
- Consumes: `PassageInput` (Task 1, `src.citation`)
- Produces: `PromptTemplate(version: str, path: str, sha256: str, text: str)`,
  `load_prompt_template(name: str) -> PromptTemplate`, `render_rag_prompt(template:
  PromptTemplate, question: str, passages: list[PassageInput]) -> str`,
  `render_closed_book_prompt(template: PromptTemplate, question: str) -> str` — dùng bởi
  Task 8 (`LLMAnswerer`, nhánh `model_backend="open"`)

- [ ] **Step 1: Tạo 2 file prompt (data, không phải code)**

```text
# src/prompts/answer_de_v1.txt
You are answering a question in German using ONLY the passages provided below. Do not use any outside knowledge.

Passages:
{passages}

Question: {question}

Respond in German using exactly this format, with each label at the start of its own line:

If the passages contain the answer:
KURZ: <the shortest exact answer, copied verbatim from one passage>
ANTWORT: <one full German sentence answering the question>
QUELLE: P<passage number>
ZITAT: "<the exact sentence from that passage containing the answer>"

If none of the passages contain the answer:
KURZ: KEINE ANTWORT
ANTWORT: None
QUELLE: None
ZITAT: None

Do not add any text before or after these four lines.
```

```text
# src/prompts/answer_de_closed_book_v1.txt
You are answering a question in German using only your own knowledge. No passages are provided.

Question: {question}

Respond in German using exactly this format, with each label at the start of its own line:

If you know the answer:
KURZ: <the shortest exact answer>
ANTWORT: <one full German sentence answering the question>

If you do not know the answer:
KURZ: KEINE ANTWORT
ANTWORT: None

Do not add any text before or after these two lines. Do not include QUELLE or ZITAT lines.
```

- [ ] **Step 2: Viết test thất bại cho `prompt_loader.py`**

```python
# tests/test_prompt_loader.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.citation import PassageInput
from src.prompt_loader import load_prompt_template, render_closed_book_prompt, render_rag_prompt


def test_load_prompt_template_has_stable_full_hash():
    t1 = load_prompt_template("answer_de_v1")
    t2 = load_prompt_template("answer_de_v1")
    assert t1.sha256 == t2.sha256
    assert len(t1.sha256) == 64  # hex đầy đủ, KHÔNG cắt như passage_id
    assert t1.version == "answer_de_v1"


def test_render_rag_prompt_includes_numbered_passages_and_question():
    template = load_prompt_template("answer_de_v1")
    passages = [
        PassageInput("p1", "Berlin", "Berlin ist die Hauptstadt."),
        PassageInput("p2", "Paris", "Paris ist die Hauptstadt Frankreichs."),
    ]
    rendered = render_rag_prompt(template, "Was ist die Hauptstadt?", passages)
    assert "[P1] Berlin" in rendered
    assert "[P2] Paris" in rendered
    assert "Was ist die Hauptstadt?" in rendered


def test_render_closed_book_prompt_has_no_passage_section():
    template = load_prompt_template("answer_de_closed_book_v1")
    rendered = render_closed_book_prompt(template, "Was ist die Hauptstadt?")
    assert "[P1]" not in rendered
    assert "Was ist die Hauptstadt?" in rendered


def test_different_prompt_files_have_different_hashes():
    rag = load_prompt_template("answer_de_v1")
    closed = load_prompt_template("answer_de_closed_book_v1")
    assert rag.sha256 != closed.sha256


def test_changing_prompt_file_changes_hash(tmp_path, monkeypatch):
    import src.prompt_loader as pl
    monkeypatch.setattr(pl, "PROMPTS_DIR", tmp_path)
    (tmp_path / "temp_v1.txt").write_text("Frage: {question}", encoding="utf-8")
    before = load_prompt_template("temp_v1").sha256
    (tmp_path / "temp_v1.txt").write_text("Frage: {question} ", encoding="utf-8")  # +1 byte
    after = load_prompt_template("temp_v1").sha256
    assert before != after
```

- [ ] **Step 3: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_prompt_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.prompt_loader'`

- [ ] **Step 4: Viết code tối thiểu**

```python
# src/prompt_loader.py
"""Nạp và render prompt template cho RAG —
future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md §2.1/§5.3.

Mỗi prompt có version + hash SHA-256 ĐẦY ĐỦ (không cắt như passage_id của V1) để tái
lập: đổi 1 byte trong file .txt thì prompt_sha256 đổi theo, cho phép audit prompt nào
thực sự tạo ra một run.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.citation import PassageInput

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    path: str
    sha256: str
    text: str


def load_prompt_template(name: str) -> PromptTemplate:
    """`name` không có phần mở rộng, ví dụ "answer_de_v1" hoặc "answer_de_closed_book_v1"."""
    path = PROMPTS_DIR / f"{name}.txt"
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    return PromptTemplate(version=name, path=str(path), sha256=sha256, text=raw_bytes.decode("utf-8"))


def render_rag_prompt(template: PromptTemplate, question: str, passages: list[PassageInput]) -> str:
    passages_block = "\n\n".join(
        f"[P{i}] {p.title or p.passage_id}\n{p.text}" for i, p in enumerate(passages, start=1)
    )
    return template.text.format(passages=passages_block, question=question)


def render_closed_book_prompt(template: PromptTemplate, question: str) -> str:
    return template.text.format(question=question)
```

- [ ] **Step 5: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_prompt_loader.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/prompts/answer_de_v1.txt src/prompts/answer_de_closed_book_v1.txt src/prompt_loader.py tests/test_prompt_loader.py
git commit -m "feat(rag): add versioned+hashed prompt templates for rag and closed_book"
```

---

## Task 6: `llm_clients.py` — `RawGeneration` + `FakeLLMClient` (test double)

**Files:**
- Create: `src/llm_clients.py`
- Test: `tests/test_llm_clients.py`

**Interfaces:**
- Consumes: không gì (thư viện chuẩn)
- Produces: `RawGeneration(status: str, error_type: str | None, raw_text: str | None,
  latency_ms: float)`, `FakeLLMClient(scripted: list[str | RawGeneration])` với method
  `.generate(prompt: str) -> RawGeneration` — dùng bởi Task 8 (test `LLMAnswerer` nhánh
  `model_backend="open"`, không gọi mạng thật)

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_llm_clients.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_clients import FakeLLMClient, RawGeneration


def test_fake_client_returns_scripted_text_in_order():
    client = FakeLLMClient(["erste Antwort", "zweite Antwort"])
    first = client.generate("prompt 1")
    second = client.generate("prompt 2")
    assert first.status == "success"
    assert first.raw_text == "erste Antwort"
    assert second.raw_text == "zweite Antwort"


def test_fake_client_can_script_an_error():
    error = RawGeneration(status="error", error_type="timeout", raw_text=None, latency_ms=5.0)
    client = FakeLLMClient([error])
    result = client.generate("prompt")
    assert result.status == "error"
    assert result.error_type == "timeout"


def test_fake_client_raises_when_scenario_exhausted():
    client = FakeLLMClient(["only one"])
    client.generate("p")
    try:
        client.generate("p again")
        assert False, "phải raise khi hết kịch bản"
    except AssertionError as e:
        assert "hết kịch bản" in str(e)
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_llm_clients.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm_clients'`

- [ ] **Step 3: Viết code tối thiểu**

```python
# src/llm_clients.py
"""Client sinh câu trả lời — future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md
§2.3. Đây là nơi DUY NHẤT import SDK model (anthropic, hoặc sau này transformers cho
model_backend="open" thật) — đổi model không đụng citation.py/answer_parser.py/answerer.py.

`import anthropic` chỉ nằm trong build_claude_api_client() (local import bên trong hàm),
KHÔNG ở top-level module — nhờ vậy toàn bộ file này và test của nó chạy được mà không
cần cài gói anthropic; chỉ cần cài khi thực sự gọi build_claude_api_client() để chạy thật.
"""
import re
import time
from dataclasses import dataclass

from src.citation import PassageInput


@dataclass(frozen=True)
class RawGeneration:
    """Kết quả một lượt gọi model_backend="open" (text tự do, chưa parse)."""
    status: str  # "success" | "error"
    error_type: str | None
    raw_text: str | None  # None khi status="error"
    latency_ms: float


class FakeLLMClient:
    """Test double cho model_backend="open" — không cần model/GPU thật. Kịch bản trả về
    theo đúng thứ tự gọi, để test LLMAnswerer (Task 8) đủ mọi nhánh trạng thái mà không
    phụ thuộc mạng/model. `scripted[i]` là str (thành công) hoặc RawGeneration (kể cả
    lỗi, dùng để giả lập timeout/refusal)."""

    def __init__(self, scripted: list):
        self._scripted = list(scripted)
        self._calls = 0

    def generate(self, prompt: str) -> RawGeneration:
        if self._calls >= len(self._scripted):
            raise AssertionError("FakeLLMClient hết kịch bản — gọi nhiều hơn số scripted")
        item = self._scripted[self._calls]
        self._calls += 1
        if isinstance(item, RawGeneration):
            return item
        return RawGeneration(status="success", error_type=None, raw_text=item, latency_ms=1.0)
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_llm_clients.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/llm_clients.py tests/test_llm_clients.py
git commit -m "feat(rag): add RawGeneration + FakeLLMClient test double"
```

---

## Task 7: `llm_clients.py` — `ClaudeApiClient` thật + adapter sang `ParsedAnswer`

**Files:**
- Modify: `src/llm_clients.py` (thêm cuối file)
- Modify: `tests/test_llm_clients.py` (thêm cuối file)

**Interfaces:**
- Consumes: `PassageInput` (Task 1), `ParsedAnswer` (Task 3)
- Produces: `ClaudeCitation(document_index: int, cited_text: str, start_char_index: int
  | None, end_char_index: int | None)`, `ClaudeGeneration(status: str, error_type: str |
  None, short_answer: str | None, answer: str | None, citations: list[ClaudeCitation],
  latency_ms: float)`, `ClaudeApiClient(client, model_id: str = "claude-opus-5")` với
  method `.generate_with_citations(prompt_instructions: str, question: str, passages:
  list[PassageInput]) -> ClaudeGeneration`, `claude_generation_to_parsed_answer(gen:
  ClaudeGeneration) -> ParsedAnswer`, `build_claude_api_client() -> ClaudeApiClient`
  (factory dùng khi chạy thật) — dùng bởi Task 8 (`LLMAnswerer`, nhánh
  `model_backend="claude"`)

**Ghi chú quan trọng:** thiết kế §2.3 nói Claude chỉ được yêu cầu trả `KURZ`/`ANTWORT`;
`QUELLE`/`ZITAT` không phải text tự do mà lấy từ cơ chế `citations` của chính API
(`document_index` ≈ QUELLE, `cited_text` ≈ ZITAT). `ClaudeApiClient` nhận một object
`client` qua constructor (dependency injection) để test không cần cài gói `anthropic`
hay gọi mạng thật — chỉ `build_claude_api_client()` (factory) mới `import anthropic`,
và chỉ dùng khi chạy thật.

- [ ] **Step 1: Viết test thất bại**

```python
# thêm vào cuối tests/test_llm_clients.py
from types import SimpleNamespace

from src.answer_parser import ParsedAnswer
from src.citation import PassageInput
from src.llm_clients import (
    ClaudeApiClient,
    ClaudeCitation,
    ClaudeGeneration,
    claude_generation_to_parsed_answer,
)


def _fake_block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations or [])


def _fake_citation(document_index, cited_text, start=None, end=None):
    return SimpleNamespace(document_index=document_index, cited_text=cited_text,
                            start_char_index=start, end_char_index=end)


class _StubMessages:
    def __init__(self, response):
        self._response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self._response


class _StubAnthropicClient:
    def __init__(self, response):
        self.messages = _StubMessages(response)


def test_generate_with_citations_parses_single_citation():
    response = SimpleNamespace(content=[
        _fake_block("KURZ: Albert Einstein\n"),
        _fake_block(
            "ANTWORT: Albert Einstein entwickelte die Relativitaetstheorie.\n",
            citations=[_fake_citation(0, "Einstein entwickelte sie 1915.", 10, 40)],
        ),
    ])
    stub = _StubAnthropicClient(response)
    client = ClaudeApiClient(stub)
    passages = [PassageInput("p1", "Titel", "Ein Text ueber Einstein.")]

    gen = client.generate_with_citations("Beantworte auf Deutsch.", "Wer war Einstein?", passages)

    assert gen.status == "success"
    assert gen.short_answer == "Albert Einstein"
    assert len(gen.citations) == 1
    assert gen.citations[0].document_index == 0
    assert gen.citations[0].cited_text == "Einstein entwickelte sie 1915."

    request = stub.messages.last_request
    assert request["model"] == "claude-opus-5"
    documents = [c for c in request["messages"][0]["content"] if c["type"] == "document"]
    assert len(documents) == 1
    assert documents[0]["citations"] == {"enabled": True}


def test_generate_with_citations_reports_format_violation_without_labels():
    response = SimpleNamespace(content=[_fake_block("Ich bin mir nicht sicher.")])
    client = ClaudeApiClient(_StubAnthropicClient(response))
    gen = client.generate_with_citations("x", "Frage?", [PassageInput("p1", "T", "Text.")])
    assert gen.status == "error"
    assert gen.error_type == "format_violation"


def test_adapter_maps_single_citation_to_parsed_answer():
    gen = ClaudeGeneration(
        status="success", error_type=None, short_answer="Albert Einstein",
        answer="Albert Einstein entwickelte die Relativitaetstheorie.",
        citations=[ClaudeCitation(document_index=1, cited_text="Er entwickelte sie 1915.",
                                   start_char_index=5, end_char_index=30)],
        latency_ms=42.0,
    )
    parsed = claude_generation_to_parsed_answer(gen)
    assert isinstance(parsed, ParsedAnswer)
    assert parsed.quelle_refs == ("P2",)
    assert parsed.quote == "Er entwickelte sie 1915."


def test_adapter_maps_multiple_citations_without_choosing():
    gen = ClaudeGeneration(
        status="success", error_type=None, short_answer="X", answer="Y.",
        citations=[
            ClaudeCitation(document_index=0, cited_text="a", start_char_index=0, end_char_index=1),
            ClaudeCitation(document_index=2, cited_text="b", start_char_index=0, end_char_index=1),
        ],
        latency_ms=1.0,
    )
    parsed = claude_generation_to_parsed_answer(gen)
    assert parsed.quelle_refs == ("P1", "P3")
    assert parsed.quote is None


def test_adapter_maps_keine_antwort():
    gen = ClaudeGeneration(status="success", error_type=None, short_answer="KEINE ANTWORT",
                            answer="None", citations=[], latency_ms=1.0)
    parsed = claude_generation_to_parsed_answer(gen)
    assert parsed.answered is False
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_llm_clients.py -v`
Expected: FAIL — `ImportError: cannot import name 'ClaudeApiClient'`

- [ ] **Step 3: Viết code tối thiểu**

```python
# thêm vào cuối src/llm_clients.py
from src.answer_parser import ParsedAnswer

_CLAUDE_LABEL_RE = re.compile(r"^(KURZ|ANTWORT):\s*(.*)$", re.MULTILINE)


def _extract_kurz_antwort(full_text: str) -> tuple[str | None, str | None]:
    labels: dict[str, str] = {}
    for match in _CLAUDE_LABEL_RE.finditer(full_text):
        labels.setdefault(match.group(1), match.group(2).strip())
    return labels.get("KURZ"), labels.get("ANTWORT")


@dataclass(frozen=True)
class ClaudeCitation:
    document_index: int
    cited_text: str
    start_char_index: int | None
    end_char_index: int | None


@dataclass(frozen=True)
class ClaudeGeneration:
    """Kết quả một lượt gọi model_backend="claude" — khác RawGeneration vì citations
    của Claude là dữ liệu có cấu trúc (document_index/cited_text), không phải text tự
    do như QUELLE/ZITAT của model_backend="open"."""
    status: str
    error_type: str | None
    short_answer: str | None
    answer: str | None
    citations: list[ClaudeCitation]
    latency_ms: float


class ClaudeApiClient:
    """model_backend="claude". `client` là object có `.messages.create(**kwargs)` —
    thường là `anthropic.Anthropic()` thật (xem build_claude_api_client), hoặc stub
    trong test. Prompt chỉ yêu cầu KURZ/ANTWORT (§2.3) — QUELLE/ZITAT lấy từ block
    `citations` mà API tự trả kèm mỗi đoạn text đã trích dẫn."""

    MODEL_ID = "claude-opus-5"

    def __init__(self, client, model_id: str = MODEL_ID, max_tokens: int = 512):
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens

    def generate_with_citations(
        self, prompt_instructions: str, question: str, passages: list[PassageInput]
    ) -> ClaudeGeneration:
        t0 = time.perf_counter()
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": p.text},
                "title": p.title or p.passage_id,
                "citations": {"enabled": True},
            }
            for p in passages
        ]
        try:
            response = self._client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                messages=[{
                    "role": "user",
                    "content": [*documents, {"type": "text", "text": f"{prompt_instructions}\n\nQuestion: {question}"}],
                }],
            )
        except Exception:  # lỗi mạng/API — không phân loại chi tiết ở bản này
            return ClaudeGeneration(
                status="error", error_type="api_error", short_answer=None, answer=None,
                citations=[], latency_ms=(time.perf_counter() - t0) * 1000,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        full_text_parts, citations = [], []
        for block in response.content:
            if getattr(block, "type", None) != "text":
                continue
            full_text_parts.append(block.text)
            for c in (getattr(block, "citations", None) or []):
                citations.append(ClaudeCitation(
                    document_index=c.document_index, cited_text=c.cited_text,
                    start_char_index=getattr(c, "start_char_index", None),
                    end_char_index=getattr(c, "end_char_index", None),
                ))

        kurz, antwort = _extract_kurz_antwort("".join(full_text_parts))
        if kurz is None or antwort is None:
            return ClaudeGeneration(
                status="error", error_type="format_violation", short_answer=None, answer=None,
                citations=[], latency_ms=latency_ms,
            )
        return ClaudeGeneration(
            status="success", error_type=None, short_answer=kurz, answer=antwort,
            citations=citations, latency_ms=latency_ms,
        )


def claude_generation_to_parsed_answer(gen: ClaudeGeneration) -> ParsedAnswer:
    """Chuẩn hóa ClaudeGeneration về CÙNG dạng ParsedAnswer mà citation.verify_citation
    dùng cho model_backend="open" — nhờ vậy verify_citation KHÔNG cần biết backend nào
    đã sinh ra câu trả lời."""
    if gen.short_answer is None or gen.short_answer == "KEINE ANTWORT":
        return ParsedAnswer(format_violation=False, answered=False)
    if len(gen.citations) > 1:
        refs = tuple(f"P{c.document_index + 1}" for c in gen.citations)
        return ParsedAnswer(
            format_violation=False, answered=True,
            short_answer=gen.short_answer, answer=gen.answer,
            quelle_refs=refs, quote=None,
        )
    if not gen.citations:
        return ParsedAnswer(
            format_violation=False, answered=True,
            short_answer=gen.short_answer, answer=gen.answer,
            quelle_refs=(), quote=None,
        )
    only = gen.citations[0]
    return ParsedAnswer(
        format_violation=False, answered=True,
        short_answer=gen.short_answer, answer=gen.answer,
        quelle_refs=(f"P{only.document_index + 1}",), quote=only.cited_text,
    )


def build_claude_api_client(model_id: str = ClaudeApiClient.MODEL_ID) -> ClaudeApiClient:
    """Factory dùng khi chạy THẬT — import anthropic CHỈ ở đây (local import), để phần
    còn lại của module (và mọi test) không cần cài gói anthropic. Đọc API key theo cơ
    chế resolve credential mặc định của SDK (biến môi trường ANTHROPIC_API_KEY, hoặc
    `ant auth login`) — không tự đọc file .env ở đây, đó là việc của app/service.py
    (ngoài phạm vi plan này, xem design §5.4)."""
    import anthropic
    return ClaudeApiClient(anthropic.Anthropic(), model_id=model_id)
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_llm_clients.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/llm_clients.py tests/test_llm_clients.py
git commit -m "feat(rag): add ClaudeApiClient with citations + adapter to ParsedAnswer"
```

---

## Task 8: `answerer.py` — `LLMAnswerer` (ghép tất cả, 2 backend, mọi nhánh trạng thái)

**Files:**
- Create: `src/answerer.py`
- Test: `tests/test_answerer.py`

**Interfaces:**
- Consumes: `ParsedAnswer`, `parse_llm_output` (Task 3); `PassageInput`, `CitationDetail`,
  `CitationOutcome`, `verify_citation` (Task 1/4); `PromptTemplate`,
  `render_rag_prompt`, `render_closed_book_prompt` (Task 5); `RawGeneration`,
  `FakeLLMClient`, `ClaudeGeneration`, `claude_generation_to_parsed_answer` (Task 6/7)
- Produces: `AnswererType = Literal["closed_book", "rag", "oracle_rag"]`,
  `ModelBackend = Literal["open", "claude"]`, `AnswerResult(generation_status: str,
  error_type: str | None, answered: bool | None, short_answer: str | None, answer: str |
  None, citation_status: str | None, answer_supported: bool | None, citations:
  list[CitationDetail], raw_citation_refs: tuple[str, ...] | None, latency_ms: float)`,
  `LLMAnswerer(model_backend, client, prompt_template, closed_book_prompt_template=None)`
  với method `.answer(answerer_type, question, passages) -> AnswerResult` — dùng bởi
  batch runner/web trong các bước sau của roadmap (ngoài phạm vi plan này)

**Giới hạn phạm vi:** `answerer_type="closed_book"` trong plan này chỉ hỗ trợ
`model_backend="open"` (khớp đúng ma trận thực nghiệm ở design §3.3 — `model_backend=
"claude"` chỉ chạy cấu hình `rag`, không chạy `closed_book`/`oracle_rag`). `ExtractiveAnswerer`
(bọc Reader V1) không nằm trong file này — thêm sau khi Reader V1 có checkpoint (design
§7.2 mục 5, ngoài phạm vi plan này).

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_answerer.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answerer import LLMAnswerer
from src.citation import PassageInput
from src.llm_clients import ClaudeCitation, ClaudeGeneration, FakeLLMClient, RawGeneration
from src.prompt_loader import load_prompt_template

RAG_TEMPLATE = load_prompt_template("answer_de_v1")
CLOSED_BOOK_TEMPLATE = load_prompt_template("answer_de_closed_book_v1")

P1 = PassageInput("p1", "Titel A", "Albert Einstein wurde 1879 geboren. Er entwickelte die Relativitaetstheorie.")
P2 = PassageInput("p2", "Titel B", "Marie Curie erhielt zwei Nobelpreise.")

ANSWERED_TEXT = (
    "KURZ: Albert Einstein\n"
    "ANTWORT: Albert Einstein entwickelte die Relativitaetstheorie.\n"
    "QUELLE: P1\n"
    'ZITAT: "Er entwickelte die Relativitaetstheorie."\n'
)
NO_ANSWER_TEXT = "KURZ: KEINE ANTWORT\nANTWORT: None\nQUELLE: None\nZITAT: None\n"
FORMAT_VIOLATION_TEXT = "Ich weiss es nicht.\n"


def _open_answerer(scripted):
    client = FakeLLMClient(scripted)
    return LLMAnswerer("open", client, RAG_TEMPLATE, CLOSED_BOOK_TEMPLATE)


def test_rag_verified_and_supported():
    result = _open_answerer([ANSWERED_TEXT]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.generation_status == "success"
    assert result.answered is True
    assert result.citation_status == "verified"
    assert result.answer_supported is True
    assert result.citations[0].passage_id == "p1"


def test_rag_verified_but_not_supported():
    bad_kurz = ANSWERED_TEXT.replace("KURZ: Albert Einstein", "KURZ: Marie Curie")
    result = _open_answerer([bad_kurz]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.citation_status == "verified"
    assert result.answer_supported is False


def test_rag_wrong_passage():
    bad_zitat = ANSWERED_TEXT.replace(
        'ZITAT: "Er entwickelte die Relativitaetstheorie."',
        'ZITAT: "Marie Curie erhielt zwei Nobelpreise."',
    )
    result = _open_answerer([bad_zitat]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.citation_status == "wrong_passage"


def test_rag_no_answer():
    result = _open_answerer([NO_ANSWER_TEXT]).answer("rag", "Wer war Napoleon?", [P1, P2])
    assert result.generation_status == "success"
    assert result.answered is False
    assert result.citation_status is None
    assert result.answer_supported is None


def test_rag_format_violation():
    result = _open_answerer([FORMAT_VIOLATION_TEXT]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.generation_status == "error"
    assert result.error_type == "format_violation"
    assert result.answered is None


def test_rag_generation_error_does_not_reach_citation():
    error = RawGeneration(status="error", error_type="timeout", raw_text=None, latency_ms=5.0)
    result = _open_answerer([error]).answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.generation_status == "error"
    assert result.error_type == "timeout"
    assert result.citation_status is None


def test_oracle_rag_reuses_same_path_as_rag():
    # oracle_rag khác rag chỉ ở việc caller đưa gold passage thay vì kết quả retrieval —
    # LLMAnswerer không phân biệt, cùng một luồng xử lý.
    result = _open_answerer([ANSWERED_TEXT]).answer("oracle_rag", "Wer war Einstein?", [P1])
    assert result.citation_status == "verified"


def test_closed_book_has_no_citation_even_if_model_adds_one():
    # model lỡ thêm QUELLE/ZITAT dù prompt closed_book không yêu cầu -> vẫn bị bỏ qua
    # hoàn toàn (design §2.5): citation_status luôn None cho closed_book.
    leaked = ANSWERED_TEXT  # có QUELLE/ZITAT, nhưng answerer_type="closed_book"
    result = _open_answerer([leaked]).answer("closed_book", "Wer war Einstein?", [])
    assert result.answered is True
    assert result.citation_status is None
    assert result.answer_supported is None


def test_closed_book_no_answer():
    result = _open_answerer([NO_ANSWER_TEXT]).answer("closed_book", "Wer war das?", [])
    assert result.answered is False


def test_rag_with_claude_backend_verified():
    claude_gen = ClaudeGeneration(
        status="success", error_type=None, short_answer="Albert Einstein",
        answer="Albert Einstein entwickelte die Relativitaetstheorie.",
        citations=[ClaudeCitation(document_index=0, cited_text="Er entwickelte die Relativitaetstheorie.",
                                   start_char_index=10, end_char_index=50)],
        latency_ms=100.0,
    )

    class _StubClaudeClient:
        def generate_with_citations(self, prompt_instructions, question, passages):
            return claude_gen

    answerer = LLMAnswerer("claude", _StubClaudeClient(), RAG_TEMPLATE)
    result = answerer.answer("rag", "Wer war Einstein?", [P1, P2])
    assert result.citation_status == "verified"
    assert result.answer_supported is True


def test_rag_with_claude_backend_multiple_sources():
    claude_gen = ClaudeGeneration(
        status="success", error_type=None, short_answer="X", answer="Y.",
        citations=[
            ClaudeCitation(document_index=0, cited_text="a", start_char_index=0, end_char_index=1),
            ClaudeCitation(document_index=1, cited_text="b", start_char_index=0, end_char_index=1),
        ],
        latency_ms=1.0,
    )

    class _StubClaudeClient:
        def generate_with_citations(self, prompt_instructions, question, passages):
            return claude_gen

    answerer = LLMAnswerer("claude", _StubClaudeClient(), RAG_TEMPLATE)
    result = answerer.answer("rag", "Frage?", [P1, P2])
    assert result.citation_status == "multiple_sources"
    assert result.raw_citation_refs == ("P1", "P2")
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_answerer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.answerer'`

- [ ] **Step 3: Viết code tối thiểu**

```python
# src/answerer.py
"""Answerer interface + LLMAnswerer — ghép llm_clients + answer_parser + citation
thành một lượt sinh câu trả lời hoàn chỉnh theo §2 của
future_RAG/superpowers/specs/2026-09-12-rag-answer-generation-design.md.

ExtractiveAnswerer (bọc Reader V1) KHÔNG nằm trong file này — thêm sau khi Reader V1 có
checkpoint (design §7.2 mục 5), ngoài phạm vi plan này.
"""
from dataclasses import dataclass
from typing import Literal

from src.answer_parser import ParsedAnswer, parse_llm_output
from src.citation import CitationDetail, CitationOutcome, PassageInput, verify_citation
from src.llm_clients import ClaudeGeneration, claude_generation_to_parsed_answer
from src.prompt_loader import PromptTemplate, render_closed_book_prompt, render_rag_prompt

AnswererType = Literal["closed_book", "rag", "oracle_rag"]
ModelBackend = Literal["open", "claude"]


@dataclass(frozen=True)
class AnswerResult:
    """Schema đầy đủ theo design §2.4/§2.5. Việc gắn thêm config (model_revision,
    prompt_sha256, ...) vào PredictionRecord là việc của batch runner/web sau này —
    ngoài phạm vi plan này."""
    generation_status: str  # "success" | "error"
    error_type: str | None
    answered: bool | None
    short_answer: str | None
    answer: str | None
    citation_status: str | None
    answer_supported: bool | None
    citations: list[CitationDetail]
    raw_citation_refs: tuple[str, ...] | None
    latency_ms: float


def _error_result(error_type: str, latency_ms: float) -> AnswerResult:
    return AnswerResult(
        generation_status="error", error_type=error_type, answered=None,
        short_answer=None, answer=None, citation_status=None, answer_supported=None,
        citations=[], raw_citation_refs=None, latency_ms=latency_ms,
    )


def _no_answer_result(parsed: ParsedAnswer, latency_ms: float) -> AnswerResult:
    return AnswerResult(
        generation_status="success", error_type=None, answered=False,
        short_answer=None, answer=None, citation_status=None, answer_supported=None,
        citations=[], raw_citation_refs=None, latency_ms=latency_ms,
    )


def _answered_without_citation_result(parsed: ParsedAnswer, latency_ms: float) -> AnswerResult:
    """Dùng cho closed_book (§2.5): answered=True nhưng citation_status LUÔN None, kể cả
    khi parsed có quelle_refs/quote (model lỡ tự thêm dù prompt không yêu cầu)."""
    return AnswerResult(
        generation_status="success", error_type=None, answered=True,
        short_answer=parsed.short_answer, answer=parsed.answer,
        citation_status=None, answer_supported=None, citations=[], raw_citation_refs=None,
        latency_ms=latency_ms,
    )


def _with_citation_result(parsed: ParsedAnswer, outcome: CitationOutcome, latency_ms: float) -> AnswerResult:
    return AnswerResult(
        generation_status="success", error_type=None, answered=True,
        short_answer=parsed.short_answer, answer=parsed.answer,
        citation_status=outcome.citation_status, answer_supported=outcome.answer_supported,
        citations=outcome.citations, raw_citation_refs=outcome.raw_citation_refs,
        latency_ms=latency_ms,
    )


class LLMAnswerer:
    """`answerer_type="closed_book"` chỉ hỗ trợ `model_backend="open"` trong plan này
    (khớp ma trận thực nghiệm design §3.3 — model_backend="claude" chỉ chạy "rag").
    `answerer_type="oracle_rag"` dùng lại cùng đường xử lý với "rag": khác nhau ở việc
    caller đưa gold passage hay kết quả retrieval vào `passages`, LLMAnswerer không cần
    biết khác biệt đó."""

    def __init__(self, model_backend: ModelBackend, client, prompt_template: PromptTemplate,
                 closed_book_prompt_template: PromptTemplate | None = None):
        self._model_backend = model_backend
        self._client = client
        self._prompt_template = prompt_template
        self._closed_book_prompt_template = closed_book_prompt_template

    def answer(self, answerer_type: AnswererType, question: str,
               passages: list[PassageInput]) -> AnswerResult:
        if answerer_type == "closed_book":
            return self._answer_closed_book(question)
        return self._answer_with_passages(question, passages)

    def _answer_closed_book(self, question: str) -> AnswerResult:
        if self._closed_book_prompt_template is None:
            raise ValueError("LLMAnswerer cần closed_book_prompt_template để trả lời closed_book")
        prompt = render_closed_book_prompt(self._closed_book_prompt_template, question)
        parsed, latency_ms, error_result = self._generate_and_parse_open(prompt)
        if error_result is not None:
            return error_result
        if not parsed.answered:
            return _no_answer_result(parsed, latency_ms)
        return _answered_without_citation_result(parsed, latency_ms)

    def _answer_with_passages(self, question: str, passages: list[PassageInput]) -> AnswerResult:
        if self._model_backend == "claude":
            gen: ClaudeGeneration = self._client.generate_with_citations(
                self._prompt_template.text, question, passages
            )
            if gen.status == "error":
                return _error_result(gen.error_type, gen.latency_ms)
            parsed = claude_generation_to_parsed_answer(gen)
            latency_ms = gen.latency_ms
        else:
            prompt = render_rag_prompt(self._prompt_template, question, passages)
            parsed, latency_ms, error_result = self._generate_and_parse_open(prompt)
            if error_result is not None:
                return error_result

        if parsed.format_violation:
            return _error_result("format_violation", latency_ms)
        if not parsed.answered:
            return _no_answer_result(parsed, latency_ms)

        outcome = verify_citation(passages, parsed)
        return _with_citation_result(parsed, outcome, latency_ms)

    def _generate_and_parse_open(self, prompt: str):
        gen = self._client.generate(prompt)
        if gen.status == "error":
            return None, gen.latency_ms, _error_result(gen.error_type, gen.latency_ms)
        return parse_llm_output(gen.raw_text), gen.latency_ms, None
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_answerer.py -v`
Expected: 11 passed

- [ ] **Step 5: Chạy toàn bộ test suite của plan này, xác nhận không phá vỡ gì**

Run: `python -m pytest tests/test_citation.py tests/test_answer_parser.py tests/test_prompt_loader.py tests/test_llm_clients.py tests/test_answerer.py -v`
Expected: 46 passed (test_citation.py 15 + test_answer_parser.py 7 + test_prompt_loader.py 5
+ test_llm_clients.py 8 + test_answerer.py 11 = 46)

- [ ] **Step 6: Chạy test suite V1 hiện có, xác nhận không đụng gì (không import lẫn nhau)**

Run: `python -m pytest tests/test_schemas.py tests/test_evaluation.py tests/test_bm25_retriever.py -v`
Expected: 51 passed (không đổi so với trước khi có plan này — file mới không sửa
`src/schemas.py`, `src/evaluation.py`, `src/corpus.py`, `src/bm25_retriever.py`,
`src/splits.py`, `src/hf_data.py`)

- [ ] **Step 7: Commit**

```bash
git add src/answerer.py tests/test_answerer.py
git commit -m "feat(rag): add LLMAnswerer combining prompt/client/parser/citation (open+claude)"
```

---

## Ngoài phạm vi plan này

Theo đúng ranh giới đã chốt ở design doc §7.2 mục 5–8 — không lập kế hoạch, không ước
lượng thời gian cho các mục này ở đây, vì còn phụ thuộc mốc V1 chưa tới:

- **`ExtractiveAnswerer`** (bọc Reader V1) — chờ Reader V1 có checkpoint (Chặng 8).
- **Bucket 7 nhánh mở rộng** trong `evaluation.py` — chờ `evaluation.py` V1.1 ổn định
  (Chặng 6). `src/schemas_rag.py` (PredictionRecord mở rộng cho RAG) cũng thuộc đây.
- **`app/`** (FastAPI + static, route public/dev, read-only) — khung dựng sớm được với
  pipeline giả lập, nhưng nối thật vào Retriever/Selection chờ Integration (Chặng 9).
- **Thực nghiệm RQ4 thật** (ma trận 6+1 run, chọn prompt/model trên dev, chạy test) —
  chỉ sau khi V1 qua Chặng 9 và candidate top-20 đã khóa.
- **`OpenModelClient`** (model mở thật chạy qua Transformers/Colab T4) — plan này chỉ có
  `FakeLLMClient` (test double) và `ClaudeApiClient` (thật). Client cho model mở thật
  cần môi trường GPU để phát triển/test, sẽ lập plan riêng khi tới Chặng 7/8 của V1 và có
  máy T4 khả dụng.

## Self-review

**1. Spec coverage (design doc §1–§2, §7.2 mục 1–4):**
- §1 (kiến trúc, ranh giới component) → Task 1–8 hiện thực đúng từng ranh giới (citation.py
  thuần, llm_clients.py là nơi duy nhất import SDK, answerer.py ghép).
- §2.1 (prompt version+hash) → Task 5.
- §2.2 (định dạng output, 2 dạng hợp lệ, format_violation) → Task 3, đã vá thêm quy tắc
  "thiếu QUELLE/ZITAT vẫn là parse hợp lệ" trong design doc trước khi viết plan.
- §2.3 (model_backend, prompt riêng closed_book, Claude chỉ trả KURZ/ANTWORT) → Task 5
  (2 file prompt), Task 7 (ClaudeApiClient + adapter), Task 8 (nhánh theo backend).
- §2.4 (generation_status/citation_status tách biệt, 6 trạng thái citation, không tự
  sửa/tự chọn) → Task 4 (verify_citation), Task 8 (answer()).
- §2.5 (ánh xạ extractive vs rag/closed_book/oracle_rag; closed_book không có citation) →
  Task 8 (`_answer_closed_book`, test `test_closed_book_has_no_citation_even_if_model_adds_one`).
- §7.1 (bảng test) → từng dòng có task tương ứng (citation.py, parser, LLMAnswerer với
  fake client, không gọi mạng thật). Riêng dòng "ExtractiveAnswerer" và "Route FastAPI"
  của bảng đó thuộc mục 5/7 — ngoài phạm vi plan này, đã nêu rõ ở trên.
- §7.2 mục 1–4 → đúng thứ tự Task 1–8 (parser/citation trước, prompt+client sau, answerer
  ghép cuối).

**2. Placeholder scan:** không còn "TBD/TODO/tương tự Task N" — mọi step có code đầy đủ.

**3. Type consistency:** đã rà lại tên hàm/field xuyên suốt 8 Task —
`find_quote_in_passage`, `is_short_answer_supported`, `find_short_answer_offsets`,
`ParsedAnswer`/`parse_llm_output`, `PassageInput`/`CitationDetail`/`CitationOutcome`/
`verify_citation`, `PromptTemplate`/`load_prompt_template`/`render_rag_prompt`/
`render_closed_book_prompt`, `RawGeneration`/`FakeLLMClient`, `ClaudeCitation`/
`ClaudeGeneration`/`ClaudeApiClient`/`claude_generation_to_parsed_answer`/
`build_claude_api_client`, `AnswerResult`/`LLMAnswerer` — dùng nhất quán ở nơi gọi
(Task 4 dùng đúng tên từ Task 1–3; Task 8 dùng đúng tên từ Task 1, 3, 5, 6, 7).
