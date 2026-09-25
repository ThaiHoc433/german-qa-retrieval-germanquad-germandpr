---
tasks: [T5.6]
buoi: 15
trang_thai: chưa làm
---

# Thẻ T5.6 — Corpus base D6 và title exclusions P4

Soạn ngày 2026-09-24, hoàn thiện đối chiếu ngày 2026-09-25. Thẻ chờ duyệt trước code. T5.6 đóng gói việc xây base từ nguồn được phép và thu title trước dedup. Không tải Wikipedia hoặc tự mở rộng sang T5.7. Spec V1.1 frozen giữ nguyên; tên API/report dưới đây là đề nghị triển khai. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi]

## 1. Phạm vi và phụ thuộc

- T5.4/T5.5 đã có producer QA/DPR, lần lượt commit `1102836`/`945f317`. Chúng nhận map passage đã chọn xong; T5.6 tạo map này trước khi gọi producer. `CorpusBuilder` đã có 9 field provenance, dedup và priority; không viết lại chunker hoặc nói provenance chưa tồn tại. [src/corpus.py §CorpusBuilder] [src/qa_records.py §build_qa_records] [src/retrieval_records.py §build_retrieval_training_records]
- **Base D6** dùng mọi context GermanQuAD train+test và mọi positive context GermanDPR train+test, trước loại leakage/unmappable. Lấy mọi chunk, kể cả chunk không chứa answer. Không nhận danh sách QA/DPR processed đã lọc làm nguồn base. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]
- Hard negative không phải nguồn base. Chunk trùng một context nguồn hợp lệ vẫn ở base; không dùng danh sách hard negative làm blacklist. Wikipedia chỉ thêm distractor khi dựng tier sau này, không vào hàm build_base_corpus.
- **P4** lấy title từ mọi occurrence context nguồn base trước dedup, kể cả context của QA loại D1 hoặc DPR unmappable. Title không quyết định có giữ chunk base hay không. Tập title khác rỗng sau chuẩn hóa, duy nhất, đã sort, được dùng chung cho mid/large. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §Nhật ký quyết định V1.1]

## 2. Files và interface đề nghị

| File | Phần việc |
|---|---|
| `src/corpus_builder.py` — chưa có | Thêm orchestration thu nguồn D6/P4, gọi CorpusBuilder hiện có |
| `tests/test_corpus_builder.py` — chưa có | Test fixture trước code; nguồn/title/report/tính thuần |
| `src/corpus.py` | Reuse API công khai và SOURCE_PRIORITY; không thay chính sách của T5.1 |
| `notebooks/01_build_corpus_base.py` | Chuyển caller ở T5.8; không import notebook để chạy task này |

Tách orchestration trong file catalog đã cho phép, tránh để primitive normalize/chunk phụ thuộc schema/dataset. Hàm thuần, không gọi Hub/ghi file và không sửa input.

```python
# ĐỀ NGHỊ — chưa tồn tại trong repo
from collections.abc import Mapping, Sequence
from typing import TypedDict

class BaseCorpusResult(TypedDict):
    passages: dict[str, dict]             # passage_id -> PassageRecord đại diện
    base_title_exclusions: list[str]      # P4: sort + unique, không title rỗng
    source_report: dict

def normalize_title(title: str | None) -> str | None: ...

def build_base_corpus(
    gq_train_rows: Sequence[Mapping[str, object]],
    gq_test_rows: Sequence[Mapping[str, object]],
    gdpr_train_rows: Sequence[Mapping[str, object]],
    gdpr_test_rows: Sequence[Mapping[str, object]],
) -> BaseCorpusResult: ...
```

Không nhận SplitDecision, answers, positive ID đã map hoặc hard-negative blacklist. Producer base không cần question/id/answers để quyết định nguồn; chỉ đọc các field sau:

- GQ: `context` str; lấy title từ dòng đầu bằng cùng quy tắc `splitlines` của CorpusBuilder, không tìm dòng đầu khác rỗng.
- DPR: `positive_ctxs` là struct các list `title`, `text`, `passage_id` song song như raw Parquet. Kiểm đủ ba list, cùng độ dài; text str, title str hoặc None, passage_id raw str (được rỗng). ID raw không dùng để hash. Không đọc `answers`, `negative_ctxs`, `hard_negative_ctxs` để chọn context, chunk hoặc title.
- Rows phải là sequence mapping, không nhận DataFrame/list câu hỏi. Field cần dùng thiếu/sai kiểu → lỗi rõ; chuỗi context rỗng/whitespace hợp lệ, tạo 0 chunk và report. Không từ chối context chỉ vì title thiếu. [src/corpus.py §CorpusBuilder] [doi-chieu-T5.6.md §1. Nguồn và phạm vi]

## 3. Thứ tự xử lý và provenance

1. Kiểm input cần dùng. Duyệt GQ train, GQ test, DPR train, DPR test; mỗi split theo index raw, mỗi positive context DPR theo index list.
2. Với **mỗi occurrence context**, xác định source/title gốc; tính title key P4 và thu vào tập trước khi chunk hoặc dedup. Title thiếu/rỗng theo P4 ghi metadata; không tự gán title.
3. Gọi `CorpusBuilder.add_text(text, source, title=title_gốc)`. GQ dùng source germanquad; DPR dùng germandpr_positive. Không normalize text gốc trước chunk, không đưa title P4 thay vào PassageRecord.title.
4. Đếm mọi chunk occurrence trước dedup. Context không tạo chunk phải được báo, nhưng title không rỗng của context đó vẫn thuộc tập loại trừ.
5. Đại diện giữ đúng priority GQ > DPR positive > Wikipedia của primitive hiện có; cùng source giữ bản đầu theo thứ tự input. Không đổi riêng text/title/offset một phần. Trong base output chỉ có hai source đầu. [src/corpus.py §SOURCE_PRIORITY]
6. Trả map passage theo ID đã sort, với bản sao record từ builder; trả title list sort. Report phải bảo toàn số occurrence, không chỉ đếm document hash duy nhất. Cùng input/thứ tự phải ra cùng kết quả. Đảo hai occurrence cùng source nhưng khác text có thể đổi đại diện theo quy tắc giữ bản đầu; không hứa invariant sai với spec.
7. Mọi PassageRecord trả ra phải qua validator; provenance phải đúng parent đã chọn. Tests đối chiếu cả slice/hash parent chứ không chỉ ID. Không dùng index/offset của parent bị dedup thay provenance đại diện. [src/schemas.py §validate_passage_record]

## 4. Chuẩn hóa title P4

`normalize_title` theo đúng thứ tự **NFC → thay `_` bằng space → lowercase → gộp whitespace → strip**. Đề nghị trả None nếu input None hoặc kết quả rỗng; kiểu khác str/None báo lỗi. Không dùng casefold/NFKC, bỏ dấu, fuzzy match hoặc tách dòng đầu của title DPR. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §Nhật ký quyết định V1.1]

Title GQ lấy từ dòng đầu context; title DPR lấy nguyên field raw title. Không suy title DPR từ positive text khi title thiếu. Không dùng title của passage đại diện làm nguồn duy nhất cho P4.

**Quan sát cache cần giữ đúng:** 802 context DPR train + 79 test có field title bằng toàn bộ context và có newline. Theo V1.1 vẫn lấy field đó rồi chuẩn hóa P4; không tự cắt dòng đầu, vì sẽ đổi tập loại trừ. Đây là dữ liệu nguồn khác thường cần ghi nhận, không tự gọi là title thiếu. Có thể đề nghị audit metadata/amendment riêng; task này không đổi frozen spec. [doi-chieu-T5.6.md §2. Mốc thật]

## 5. Report và output

`source_report` đề nghị (tên field là quyết định triển khai):

```text
input_sources:
  germanquad / germandpr_positive:
    train / test:
      n_contexts
      n_chunks_before_dedup
      n_empty_contexts
      n_missing_titles
      n_unique_title_keys
n_contexts_total
n_chunks_before_dedup_total
n_passages
representative_sources: {germanquad: ..., germandpr_positive: ...}
n_title_exclusions
empty_contexts: [metadata occurrence, ...]
missing_title_contexts: [metadata occurrence, ...]
```

Metadata occurrence có `source`, `original_split`, `source_row_index`, `source_context_index`, `parent_document_id`. GQ context index luôn 0; DPR dùng index positive context raw. ID parent là hash toàn context như primitive, không thay index gốc khi trùng hash. Giữ thứ tự duyệt §3, không dedup report. Title thiếu đếm sau P4; title chỉ gồm underscore có thể thành rỗng dù raw không whitespace.

Bốn nhóm input_sources luôn có, kể cả rỗng; counts = 0, danh sách = [] nếu không có dữ liệu. `representative_sources` có đủ hai source với 0 khi vắng; n_passages bằng tổng chúng. `n_contexts_total`/`n_chunks_before_dedup_total` là tổng bốn nhóm; n_missing/n_empty khớp số metadata của nhóm tương ứng. n_unique_title_keys theo từng nhóm không cộng thẳng thành n_title_exclusions vì title chồng giữa nguồn/split.

Trong T5.6 chỉ trả object. T5.8 mới ghi `corpus_base.jsonl`, `base_title_exclusions.json`, đưa counts/missing-title vào `corpus_manifest.json`, tính hash byte file và nối 5 report dữ liệu. source_report này là dữ liệu trung gian đề nghị, không tự thêm một artifact bắt buộc vào spec. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.7 Artifact]

## 6. Fixture có expected

Tất cả là **ví dụ minh họa**, không phải trích dataset. Với DPR tạo raw struct title/text/passage_id cùng độ dài, ID raw có thể `""`. Các field question/id/answers có thể có nhưng không quyết định corpus.

| Ca | Input | Expected |
|---|---|---|
| P4 cơ bản | `Ä_B`, ` ä b `, `A__B`, None, `___` | `ä b`, `ä b`, `a b`, None, None; title list `[a b, ä b]` |
| NFC/lower đúng mức | `A\u0308_B`, `Straße`, `STRASSE` | `ä b`, `straße`, `strasse`; hai title cuối khác nhau, không casefold |
| Title trước dedup | GQ context `Ä_B\nBerlin`; DPR positive text giống hệt nhưng title `Zweite_Quelle` | 1 passage ID `65aabd0ccb0e2bbf`, source GQ/title `Ä_B`; exclusions `[zweite quelle, ä b]` |
| GQ thiếu title | GQ context `\nBerlin` | Passage text Berlin/start=1, title None; report thiếu title=1; không lấy Berlin làm title |
| DPR title thiếu | Positive text Berlin, title None | Passage Berlin vẫn ở base; thiếu title=1; không suy title Berlin |
| Context rỗng nhưng title có | DPR positive text whitespace, title `Known_Title` | 0 passage; empty_contexts=1; exclusions `[known title]` |
| Mọi chunk đều là base | DPR positive context w000…w200, answers chỉ w010 | Giữ cả 3 ID `c3cbe252c56e270c`, `8451b74d508323dc`, `78eaa148763a161a`, kể cả chunk không chứa answer |
| Không lọc D1/unmappable | Raw GQ train question trùng DPR test; raw DPR có answers không tìm được | Context nguồn vẫn vào base/title; không nhận quyết định loại QA/DPR làm filter corpus |
| Hard negative không là nguồn | GQ context Berlin; DPR hard contexts Berlin, Paris, positive context Ulm | Base Berlin/Ulm, không Paris; Berlin vẫn source GQ, không blacklist |
| Representative khác casing | GQ Berlin; DPR positive berlin với title `Andere_Quelle` | 1 ID `fb38a93fd89e1a5f`, text/source/provenance GQ giữ nguyên; thu cả title GQ và title DPR |
| Title DPR dài | Positive text `Gott\nBerlin`, title cũng `Gott\nBerlin` | P4 key `gott berlin`, không tự cắt thành gott |
| Bảo toàn occurrence | Hai raw positive context cùng text/title, trong cùng hoặc khác dòng | n_contexts đếm 2; passage/title dedup; metadata thiếu/rỗng nếu có vẫn giữ từng index |
| Hai occurrence cùng source, khác text/title (cả hai thứ tự) | Cùng source `germandpr_positive`, hai occurrence hash trùng `fb38a93fd89e1a5f`: (1) text `"  Berlin  "`/title `Erste_Quelle`, (2) text `"BERLIN"`/title `Zweite_Quelle`. Chạy cả thứ tự (1)→(2) và (2)→(1) | Đại diện giữ **nguyên vẹn** text/title/toàn bộ provenance của occurrence **đứng trước** trong thứ tự input đang chạy, không trộn field giữa hai occurrence — thứ tự (1)→(2) ra đại diện text `"Berlin"` (chunk `char_start=2, char_end=8`), title `Erste_Quelle`; thứ tự (2)→(1) ra đại diện text `"BERLIN"` (chunk `char_start=0, char_end=6`), title `Zweite_Quelle`. `parent_document_id` bằng nhau ở cả hai occurrence (`fb38a93fd89e1a5f`, vì cả hai text đều chuẩn hóa cùng "berlin") nên **không** dùng để phân biệt ca này — chỉ `chunk_start_char`/`chunk_end_char` mới đổi theo occurrence thắng. Cả hai thứ tự đều cho `base_title_exclusions` chứa **cả hai** key `[erste quelle, zweite quelle]`, không phụ thuộc occurrence nào thành đại diện |
| Title trùng key P4 giữa hai nhóm nguồn/split khác nhau | Title `Ä_B` xuất hiện ở một context GQ train; title `ä b` xuất hiện ở một context DPR positive test — hai raw khác nhau, cùng key P4 `ä b` | Mỗi nhóm báo `n_unique_title_keys=1` riêng (đúng cho nhóm đó); tổng `n_title_exclusions` toàn cục là **1**, không phải 2 — không được cộng thẳng `n_unique_title_keys` của các nhóm để ra tổng, vì title chồng lẫn giữa nguồn/split |
| `positive_ctxs.passage_id` raw sai kiểu | Struct DPR positive có `passage_id=[123]` (int) thay vì `[""]`/chuỗi, dù giá trị này không dùng để tính hash chunk | Lỗi input rõ ràng ngay ở bước kiểm schema struct (§3: "passage_id raw là str và được phép rỗng"); không được âm thầm bỏ qua chỉ vì field không ảnh hưởng kết quả hash |
| Sai schema nguồn | context không str, title số/bool, positive_ctxs list lệch, rows là chuỗi | Lỗi rõ; không tự string-cast hoặc bỏ dòng |
| Rỗng/tính thuần | Bốn list rỗng; gọi lại; sửa object trả về | Map/title rỗng và counts 0; input không đổi, kết quả không alias input |

Test integration fixture cần dựng base trước rồi gọi QA/DPR producer để kiểm membership/offset; không sửa hai producer để hợp fixture. Giữ kiểm tra source priority đã có ở T5.1, bổ sung bằng chứng orchestration không làm mất invariant này.

## 7. Mốc cache trước implementation

Đã tính chỉ đọc bằng CorpusBuilder hiện có và công thức P4 trong script học liệu, **chưa chạy API build_base_corpus mới**. [doi-chieu-T5.6.md §3. Lệnh tái kiểm]

| Nguồn/split | Context occurrence | Chunk trước dedup | Title key duy nhất |
|---|---:|---:|---:|
| GQ train | 11.518 | 29.589 | 378 |
| GQ test | 2.204 | 10.249 | 233 |
| DPR positive train | 9.275 | 23.733 | 523 |
| DPR positive test | 1.025 | 2.683 | 137 |

Tổng **24.022 context**, **66.254 chunk occurrence**, **7.520 passage duy nhất**, source đại diện đều GQ. Context rỗng và title thiếu theo P4 đều 0 trên cache này.

Title trước dedup **565 key**; lấy sau dedup chỉ còn **388**, mất **177**. Cả 1.026 context GQ bị D1 loại mang 131 title key, đều phải thuộc exclusions. Không dùng 565 làm hằng số production hoặc suy ngưỡng coverage từ số passage/title.

Hash đối chiếu list title trong bộ nhớ (sort, JSON ensure_ascii=False/separators=(",",":"), UTF-8, không newline): `c82921ab6a524979391398884b6144929445116bd4e1eb2b6422b6c7cb906113`. Fingerprint file artifact sẽ được tính từ byte thật ở T5.8, không lấy hash object này thay thế.

## 8. Ranh giới với T5.7/T5.8 và Wikipedia

| Task | Trách nhiệm |
|---|---|
| T5.6 | Base từ hai nguồn D6, title exclusion P4 và source report trong bộ nhớ |
| T5.7 | Consumer title set, lọc Wikipedia trên fixture, stable_seed/R schedule, thứ tự/dedup chuỗi, prefix mid/large; có thẻ riêng trước code |
| T5.8 | Entrypoint ghi namespace V1.1, QA/DPR/report/manifest, acceptance base |
| T9.3 theo catalog | Streaming snapshot thật/đóng băng distractor, tier lớn và benchmark sau các điều kiện trước đó |

P4 exact match title đã normalize; title distractor thiếu thì giữ để xử lý tiếp và báo. Lịch R cố định 100→200→400→800→1600→3200→6400→10000; chunk base không bị preprocess lại khi mở rộng tier. Các quy tắc này phải đi vào thẻ T5.7, không làm thêm downloader trong T5.6. Revision Wikipedia phải chốt trước streaming thật; config spec là `20231101.de`, không tự bịa revision. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý] [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án]

## 9. Test-first, gate và điểm duyệt

Viết test fixture §6 trước implementation. Khi chưa có module, ghi rõ thiếu module; sau khung interface phải có assertion hành vi RED trước GREEN. Kiểm hồi quy repo và chạy API mới trên cache để so §7. Chưa gọi notebook build hoặc tạo artifact trong data/results.

Gate code: đúng nguồn/mọi chunk, title thu trước dedup, priority/provenance đúng, report bảo toàn, schema pass và input bất biến. Test pass không có nghĩa sinh viên đã qua gate hoặc artifact base đã nghiệm thu.

**LỆNH MẪU – chưa xác minh** (hai file mới chưa có):

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_corpus_builder.py -q -p no:cacheprovider --tb=short
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t56-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider --basetemp $taskTemp
```

- [ ] Tự giải thích nguồn D6, title trước dedup và mọi trường hợp fixture.
- [ ] Tự đối chiếu source_report với output và metadata occurrence.
- **Chờ duyệt:** file wrapper/API §2, field source_report §5, thứ tự và ranh giới T5.7 trước code.
- CẦN XÁC MINH: title DPR bằng toàn context có cần audit/amendment riêng không — chủ project kiểm nguồn metadata; trong V1.1 vẫn dùng raw title đúng §4.
- CẦN XÁC MINH: revision thực tế, file byte fingerprint và kết quả acceptance — xác minh khi T5.8/T9.3 triển khai, không thay bằng hash mô phỏng.

**Dừng ở thẻ T5.6 để duyệt; chưa tạo code/test T5.6.**
