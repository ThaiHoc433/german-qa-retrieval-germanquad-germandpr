---
tasks: [T5.0, T5.1]
buoi: 11
trang_thai: chưa làm
---

# Thẻ T5.0/T5.1 — Validators và chunk/provenance

Thẻ được dùng để bắt đầu nhánh thực thi T5.0/T5.1 sau khi người dùng giao quyết định chuyển sang code cho người triển khai. Các signature **mới/đổi** là lựa chọn triển khai của thẻ, không sửa spec. CPU local, dùng `.venv/Scripts/python.exe`; không cần download, GPU hoặc build dataset. Học liệu và gate sinh viên vẫn **chưa làm**; kết quả kiểm thử code được ghi riêng.

Nguồn: [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Khuôn một thẻ task sẵn sàng thực thi] [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi] [docs/learning/lessons/chang-05/buoi-11.md §7. Thao tác từng bước].

## 1. Phạm vi và thứ tự

- T5.0 phụ thuộc T4.2: validators cho Passage/QA/Training/Prediction theo §1.1–1.5; giữ cách trả `list[str]`, `[]` là không có lỗi. T5.1 phụ thuộc T5.0: giữ `chunk_text`, bổ sung title/provenance cho record CorpusBuilder. [src/schemas.py §validate_qa_record] [src/corpus.py §chunk_text] [src/corpus.py §CorpusBuilder]
- Sau T5.0/T5.1 vẫn còn T5.2–T5.8; chưa có base V1.1. T6.1–T6.3 được làm fixture độc lập theo lịch học liệu; BM25 thật T6.4 chờ cả base T5.8 và T6.1–T6.3 đạt.
- Không xử lý split, QA producer/collision, DPR producer, title exclusions hay build ở hai task này. CorpusBuilder lưu title đại diện ở T5.1; tập **mọi** title trước dedup là T5.6. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi]

## 2. Files và interface

| File | Tình trạng lúc lập thẻ | Thao tác dự kiến |
|---|---|---|
| `src/schemas.py` | Có QA/Prediction/RetrievalResult validators; chưa có Passage/Training validators, Prediction chưa yêu cầu hai field k | Mở rộng, không tạo module validator thứ hai |
| `tests/test_schemas.py` | Có fixture `qa_record`, `prediction` và test hiện hữu | Thêm ca mới; thêm k vào fixture Prediction, không nới validator để giữ schema cũ |
| `tests/test_evaluation.py` | Fixture `pred` chưa có hai field k | Chỉ thêm `retrieval_k=20`, `reader_k=5` vào fixture để tương thích schema; bucket logic để T6.3 |
| `src/corpus.py` | Có `Chunk`, `chunk_text`, `find_span_chunks`, `CorpusBuilder` | Reuse; bổ sung metadata đại diện và ưu tiên nguồn đúng §2.3.a |
| `tests/test_corpus.py` | Chưa có | Tạo test chunk, provenance/title và thay đại diện khi dedup |

Nguồn: [src/schemas.py §PREDICTION_RECORD_FIELDS] [tests/test_schemas.py §prediction] [tests/test_evaluation.py §pred] [src/corpus.py §CorpusBuilder.records]. `notebooks/01_build_corpus_base.py` hiện gọi `add_text` không có title và còn thêm hard-negative; cập nhật caller ở T5.8, không chạy entrypoint cũ để kiểm hai task này. [notebooks/01_build_corpus_base.py §build_qa_records] [notebooks/01_build_corpus_base.py §build_retrieval_training_records]

**Hệ quả tương thích đã kiểm kê:**

- `src/evaluation.py` gọi `validate_prediction_record` trong `_index_predictions`. Khi bắt buộc hai field k, Evaluation sẽ từ chối file predictions cũ thiếu `retrieval_k`/`reader_k`; đây là hệ quả mong muốn của P1, không chỉ là thay đổi fixture. Trong `src/`, `tests/`, `notebooks/`, caller được tìm thấy ở Evaluation và hai file test đã liệt kê; chưa tuyên bố bao phủ caller ngoài ba thư mục này. [src/evaluation.py §_index_predictions] [src/schemas.py §validate_prediction_record]
- `build_retrieval_training_records` của entrypoint cũ gọi `builder.add_text(t, "germandpr_hard_negative")`. Sau T5.1, lời gọi này sẽ bị từ chối nên `notebooks/01_build_corpus_base.py` chưa chạy được cho đến khi caller được chuyển ở T5.8. Giữ lỗi tường minh để không đưa nguồn hard-negative vào corpus; không chạy entrypoint cũ trong hai task này. [notebooks/01_build_corpus_base.py §build_retrieval_training_records] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]

**Signature dùng cho triển khai:** `Mapping` lấy từ `collections.abc`; `rec: object` cho phép validator trả lỗi cả khi input không phải dict.

```python
# src/schemas.py — mới
def validate_passage_record(rec: object, where: str = "PassageRecord", *,
                            parent_text: str | None = None) -> list[str]: ...
def validate_retrieval_training_record(rec: object,
                                      where: str = "RetrievalTrainingRecord") -> list[str]: ...
# src/schemas.py — thêm tham số keyword tùy chọn, giữ cách gọi cũ
def validate_qa_record(rec: object, where: str = "QARecord", *,
                       passage_texts: Mapping[str, str] | None = None) -> list[str]: ...
# src/schemas.py — giữ signature, bổ sung field k bắt buộc
def validate_prediction_record(rec, where: str = "PredictionRecord") -> list[str]: ...
# src/corpus.py — giữ nguyên
def chunk_text(text: str, window_words: int = 100, stride_words: int = 80) -> list[Chunk]: ...
def find_span_chunks(chunks: list[Chunk], answer_start: int, answer_end: int) -> list[int]: ...
# src/corpus.py — đổi; giữ hai đối số vị trí cho caller hiện hữu
def add_text(self, full_text: str, source: str, *, title: str | None = None) -> list[Chunk]: ...
def records(self) -> list[dict]: ...
```

- Passage validator kiểm đủ field §1.1, nguồn được phép, normalization/hash/char_len, offset nguyên không âm và `end-start=len(text)`. Khi có `parent_text`, kiểm thêm parent hash và `parent_text[start:end] == text`; không có parent thì không tuyên bố đã kiểm substring/provenance thực tế. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.1 Passage record (Corpus)]
- QA giữ gold rỗng hợp lệ, evidence text thuộc references, offset/end-exclusive. Chỉ khi có `passage_texts` mới kiểm ID tồn tại và slice thật; test lệch cả start/end một ký tự nhưng giữ độ dài phải dùng map này. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.2 QA / gold record — chỉ dùng cho GermanQuAD]
- Training kiểm positive lists không rỗng, song song, ID khác None và đúng hash text; mỗi group có `source_context_id`, lists text/ID song song. Không buộc hard-negative IDs có trong corpus; mask `gold_set` thuộc sampling sau này. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.3 Retrieval training record — dùng cho GermanDPR]
- Prediction bắt buộc hai k kiểu int (không nhận bool), kiểm `1 ≤ reader_k ≤ retrieval_k`; fixture V1 dùng 20/5. Kiểm đúng cấu hình V1 và khớp manifest ở T6.3, không suy mode từ chuỗi `run_id`. Giữ quy tắc năm field answer cùng None hoặc cùng có giá trị. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.5 Prediction record (đầu ra pipeline — lưu để tính lại metric không cần chạy lại model)]
- `add_text`: parent ID tính từ toàn văn gốc; offset từ `Chunk`. GQ lấy dòng đầu nếu không truyền title; DPR/Wikipedia dùng title nguồn truyền vào, thiếu thì None. Không tự đoán title. Representative ưu tiên GQ > DPR positive > Wikipedia; hòa nguồn giữ occurrence đầu. Khi đổi đại diện phải cập nhật đồng thời text, source, title, char_len và ba field provenance. Hard-negative-only không phải nguồn CorpusBuilder; caller training dùng `chunk_text`. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]

## 3. Fixture và expected viết trước code

Các fixture dưới là **ví dụ minh họa**, không phải record dataset. Nguồn quy tắc: spec §1.1–1.5 và §2.3; số chunk đã đối chiếu với [src/corpus.py §chunk_text].

| Ca | Input cụ thể | Expected/assertion |
|---|---|---|
| Passage hợp lệ | Parent `"  Berlin  "`, chunk `"Berlin"`, start=2/end=8, normalized=`"berlin"`, char_len=6, source=`germanquad`, title=`"Berlin"`; passage_id và parent_document_id đều `fb38a93fd89e1a5f` | `[]` khi đủ chín field và truyền parent_text |
| Passage hỏng | Ca trên nhưng mất parent_document_id / end=9 / char_len=7 / source=`germandpr_hard_negative` | Lỗi field tương ứng, không tự sửa |
| QA | `gq-train-1`, question=`"Welche Stadt?"`, source_dataset=`germanquad`, source_split=`dev`, references=`["Berlin"]`, evidence `{passage_id: "fb38a93fd89e1a5f", answer_text: "Berlin", answer_start: 0, answer_end: 6}` | Map passage ID → `"Berlin"`: hợp lệ; start=1/end=7 vẫn dài 6 nhưng slice sai → lỗi; gold và references cùng rỗng → hợp lệ |
| Training | `gdpr-train-0`, question=`"Welche Stadt?"`, source_split=`train`, positive_texts=`["Berlin"]`, positive_passage_ids=`["fb38a93fd89e1a5f"]`, một group `{source_context_id: "37bfc8ba5ec56136", texts: ["Ulm"], passage_ids: ["37bfc8ba5ec56136"]}` | Hợp lệ; bỏ 1 ID khỏi lists / positive None / group thiếu key → lỗi; không đòi `Ulm` trong corpus |
| Prediction | Reuse `prediction()`; thêm `retrieval_k=20`, `reader_k=5`; no-answer đặt cả năm field thành None | Hợp lệ; thiếu k / k=True / reader_k=21 / None một phần / answer="" → lỗi |
| Chunk rỗng/biên | `""`, whitespace-only; 1, 100, 101, 201 từ | Lần lượt 0, 0, 1, 1, 2, 3 chunk; không tạo chunk thừa ở cuối |
| Chunk 201 từ | `text = " ".join(f"w{i:03d}" for i in range(201))`, dài 1004 ký tự | Ba khoảng char `[0,499)`, `[400,899)`, `[800,1004)`; độ dài 499/499/204, slice giữ nguyên |
| Overlap offset | `w085` ở parent `[425,429)` của text trên | `find_span_chunks` → `[0,1]`; local spans `[425,429)`, `[25,29)`; không kéo dài chunk |
| Thay representative | Thêm `"  BERLIN  "` từ DPR positive/title=`"DPR title"`, rồi `"Berlin"` từ GQ/title=`"GQ title"` | Cùng ID, giữ GQ; char_len=6, start=0/end=6 và title GQ; không giữ offset 2/8 của đại diện cũ |
| Hòa nguồn | Hai text trên đều GQ, input lần đầu được giữ | Toàn bộ metadata vẫn thuộc occurrence đầu, không trộn provenance |

## 4. Test-first và lệnh

1. Viết expected theo bảng, sau đó thêm test vào `tests/test_schemas.py`; quan sát RED do thiếu validator/field k, không chấp nhận RED do sai import hoặc fixture typo. Test hành vi đã có (no-answer, gold rỗng) có thể xanh từ đầu: đó là regression, không giả tạo lỗi.
2. Sửa tối thiểu `src/schemas.py` để GREEN; cập nhật hai fixture Prediction đã nêu, không đổi kỳ vọng bucket cũ trong T5.0.
3. Tạo `tests/test_corpus.py`: test hành vi chunk hiện có làm baseline, rồi test metadata/title/representative mới để quan sát RED. Sửa `src/corpus.py`, chạy lại GREEN.
4. Chạy hồi quy thích hợp và lưu số pass/fail thực tế, không dùng lại “110 passed” để chứng nhận V1.1. Không build base trong T5.0/T5.1.

Lệnh chỉ-đọc kiểm tên API/test hiện hữu (đã đối chiếu repo):

```powershell
Select-String -LiteralPath 'src/schemas.py' -Pattern '^def validate_|PREDICTION_RECORD_FIELDS'
Select-String -LiteralPath 'src/corpus.py' -Pattern '^def chunk_text|^def find_span_chunks|def add_text|def records'
Test-Path -LiteralPath 'tests/test_corpus.py'
```

Kỳ vọng trước triển khai: thấy các hàm cũ và `tests/test_corpus.py` là `False`. Sau nhánh thực thi ghi trong [log kiểm chứng](kiem-chung-T5.0-T5.1.md), có thêm hai validator Passage/Training và file test mới trả `True`.

**Lệnh đã xác minh trong nhánh thực thi:** test schema và corpus đã được chạy; hồi quy toàn bộ cần quyền truy cập thư mục tạm ngoài repo. Kết quả và tùy chọn `--tb=line` dùng để rút gọn lỗi RED được ghi trong [log kiểm chứng §2](kiem-chung-T5.0-T5.1.md#2-lệnh-và-kết-quả-thực-tế).

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_schemas.py -q -p no:cacheprovider --tb=line
& '.venv/Scripts/python.exe' -B -m pytest tests/test_schemas.py tests/test_corpus.py -q -p no:cacheprovider --tb=line
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t50-t51-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
```

Acceptance: test mới bắt được hành vi thiếu/sai trước sửa, sau sửa pass; slice/metadata representative đúng và regression được kiểm. Lưu command, lý do RED, kết quả GREEN, các test cũ phải cập nhật vì schema và lý do. Không có tiêu chí score cao; không tuyên bố T5.8/base gate đạt.

## 5. Đối chiếu nhánh thực thi và gate học tập

Người dùng đã giao quyết định chuyển sang code; người triển khai đã thực hiện T5.0/T5.1 theo phạm vi §1–2. Kết quả: schema GREEN 83 test; schema + corpus GREEN 116 test; hồi quy **205 passed**. Đây là bằng chứng code của nhánh thực thi, không phải xác nhận sinh viên qua gate. [kiem-chung-T5.0-T5.1.md §2. Lệnh và kết quả thực tế]

Sau review thêm hai test thiếu ID cuối: schema **85 passed**, hồi quy **207 passed**; bỏ riêng kiểm tra độ dài trong bộ nhớ làm cả hai test mới thất bại. Không sửa thêm code production trong lượt vá này. [kiem-chung-T5.0-T5.1.md §5. Bổ sung sau review: thiếu ID cuối]

Các ô dưới dành cho sinh viên, giữ nguyên chưa làm:

- [ ] Giải thích được signature/file/test và phạm vi T5.0/T5.1.
- [ ] Đối chiếu log RED/GREEN và tự giải thích vì sao test thất bại trước sửa.
- [ ] Nêu được vì sao T5.2–T5.8 và base gate vẫn còn; T6.4 thật vẫn chờ base.

Đã kiểm kê caller trong `src/`, `tests/`, `notebooks/` như ghi ở §2. CẦN XÁC MINH: caller ngoài ba thư mục này — chủ project cung cấp hoặc người triển khai kiểm tiếp trước khi tích hợp; không còn để ngỏ signature của thẻ.
