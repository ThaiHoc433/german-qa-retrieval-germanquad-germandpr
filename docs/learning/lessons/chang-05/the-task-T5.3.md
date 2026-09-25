---
tasks: [T5.3, T5.4]
buoi: 13
trang_thai: chưa làm
---

# Thẻ T5.3/T5.4 — QA producer, offset local và remap collision

Chốt trước code ngày 2026-09-22: gộp T5.3/T5.4 trong một lượt triển khai theo hai phương án người dùng cho phép, giữ hai trách nhiệm trong cùng producer. Không tạo exception chuyển tiếp rồi bỏ. Tie-break và field report dưới đây là quyết định triển khai, không sửa spec V1.1 frozen. Trạng thái sinh viên vẫn chưa làm. Hai file code/test chưa tồn tại khi chốt interface này.

Nguồn: [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.2 QA / gold record — chỉ dùng cho GermanQuAD] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý].

## 1. Phạm vi và phụ thuộc

- T5.3 phụ thuộc T5.1/T5.2: dùng `chunk_text`, `find_span_chunks`, `passage_id`, `SplitDecision` và validator QA hiện có. Không viết lại chunker hoặc chia split lần nữa. [src/corpus.py §chunk_text] [src/corpus.py §find_span_chunks] [src/splits.py §SplitDecision] [src/schemas.py §validate_qa_record]
- T5.3 làm đường ánh xạ char-offset, giữ mọi reference hợp lệ, nhiều evidence khi overlap và record gold rỗng. T5.4 làm fallback collision và report 0/1/>1 occurrence trong cùng lượt code/test; phạm vi gộp được ghi tường minh, không đổi phân bổ Buổi 13. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi]
- CPU/fixture local. Producer thuần, không gọi Hub, không ghi file, không thay đổi corpus/input. Entrypoint và ghi artifact thật thuộc T5.8. Test T5.3 dùng corpus fixture tạo bằng `CorpusBuilder` hiện có; chưa đòi base thật hoặc chạy toàn bộ T5.6 để viết/test module này.

## 2. Files, caller và interface

| File | Tình trạng lúc lập thẻ | Phần triển khai |
|---|---|---|
| `src/qa_records.py` | Chưa có | Tạo producer QA có thể import/test trực tiếp |
| `tests/test_qa_records.py` | Chưa có | Tạo test trước producer |
| `src/corpus.py`, `src/splits.py`, `src/schemas.py` | Đã có các API cần dùng | Reuse, không đổi API để hợp fixture |
| `notebooks/01_build_corpus_base.py` | Có `build_qa_records` cũ, schema và caller split cũ | Chuyển caller ở T5.8; không sửa hoặc import notebook để test producer mới |

Notebook hiện có `OUT_DIR.mkdir` ở top-level và build entrypoint cũ; không dùng import notebook làm cách lấy producer. [notebooks/01_build_corpus_base.py §build_qa_records] [notebooks/01_build_corpus_base.py §main]

```python
# Interface chốt trước triển khai cho src/qa_records.py
from collections.abc import Mapping, Sequence
from typing import TypedDict
from src.splits import SplitDecision

class QAProductionResult(TypedDict):
    records: dict[str, list[dict]]  # train/dev/test; mỗi phần tử đúng QARecord §1.2
    coverage_report: dict
    ambiguous_offset_remap_report: list[dict]

def build_qa_records(
    gq_train_rows: Sequence[Mapping[str, object]],
    gq_test_rows: Sequence[Mapping[str, object]],
    split_decisions: Sequence[SplitDecision],
    passages: Mapping[str, Mapping[str, object]],
) -> QAProductionResult: ...
```

**Input raw:** hai split GermanQuAD đầy đủ và đúng thứ tự gốc; mỗi dòng có `id`, `question`, `context`, `answers={text: [...], answer_start: [...]}` như Parquet. Dùng `to_pylist()` để nhận list Python. `split_decisions` nhận nguyên output T5.2 (kể cả DPR); producer chỉ sử dụng quyết định GermanQuAD. Không truyền QARecord processed cũ với `answers` là list object. [docs/learning/00-project-analysis.md §5.1 Cache đã đọc bằng đường dẫn local, không gọi Hub]

**Input corpus:** map `passage_id -> PassageRecord` đại diện đã chọn xong trong bộ nhớ. Không gọi `builder.add_text` trong producer vì thêm nguồn sau đó có thể thay đại diện và làm evidence đã tạo bị sai. T5.6/T5.8 chuẩn bị map thật; test dùng map fixture. Corpus vẫn có chunk của context QA bị loại do D1 theo D6; quyết định loại QA không phải lệnh xóa passage. [src/corpus.py §CorpusBuilder] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]

**Kiểm ràng buộc:** mỗi raw GQ có đúng một quyết định khớp `query_id`, `source_dataset`, `original_split`, `source_row_index` và `question_key`. Quyết định GQ thừa/thiếu/trùng, split sai nguồn, raw schema sai hoặc passage cần dùng thiếu/sai hash/schema → lỗi rõ ràng, không silently skip. Dùng `question_key` chỉ để đối chiếu danh tính, không tự chia split. Raw `answers` phải là hai list song song; answer text không trắng, offset int không bool, không âm và `context[start:start+len(text)] == text`. Hai list cùng rỗng hợp lệ. Không dùng fallback để chữa raw offset sai.

## 3. Quy tắc tạo QARecord

1. Kiểm nguồn raw/decision và quyết định D1. Dòng có `source_split=None` là loại leakage, không tạo QARecord; vẫn được đếm riêng để bảo toàn input.
2. Với dòng được giữ, tạo `reference_answers` từ **toàn bộ** answer text gốc hợp lệ, giữ nguyên Unicode, casing, whitespace và thứ tự; giữ cả reference trùng text. Không strip answer hoặc chỉ giữ answer thuộc primary chunk. Question cũng giữ nguyên raw. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.2 QA / gold record — chỉ dùng cho GermanQuAD]
3. Chunk context bằng `chunk_text` hiện có. Với **từng cặp** text/start gốc, tính end exclusive và lấy **mọi** chunk chứa trọn span qua `find_span_chunks`; không search thêm occurrence khác trong context, không mở rộng chunk.
4. Với chunk nguồn `c`, tính `local_start = raw_start - c.char_start`, `local_end = raw_end - c.char_start`; dùng `passage_id(c.text)` để lấy đại diện. Offset trừ theo chunk của **chính context QA**, không lấy `chunk_start_char` của đại diện thuộc document khác. Nếu text hai occurrence giống hệt nhau thì local offset vẫn dùng được dù parent/offset đại diện khác. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.1 Passage record (Corpus)]
5. Nếu `representative_text[local_start:local_end] == answer_text`, dùng evidence này. Nếu text đại diện khác chunk nguồn nhưng phép slice vẫn đúng thì vẫn dùng đường char-offset; không search không cần thiết.
6. Nếu slice không đúng, chỉ chuyển sang nhánh collision khi `c.text != representative_text` và normalize/hash của hai text trùng nhau; remap theo §4 ngay trong lượt T5.3/T5.4 này. Nếu không thỏa điều kiện collision thì lỗi input/provenance cần dừng, không fallback. Raw schema của cả dòng bị loại D1 cũng được kiểm; chỉ bỏ qua lookup corpus cho dòng bị loại.
7. Nếu không chunk nào chứa trọn answer thì answer đó không tạo evidence, **vẫn giữ reference**. Nếu mọi answer đều không map, QARecord vẫn tồn tại với `gold_evidences=[]`. Context rỗng và answers rỗng cũng giữ QARecord.
8. Dedup evidence theo đúng bộ `(passage_id, answer_text, answer_start, answer_end)`, giữ thứ tự gặp đầu (reference theo raw, chunk theo thứ tự nguồn). Không dedup chỉ theo passage_id vì một passage có thể chứa nhiều answer span. Lựa chọn thứ tự này là quyết định triển khai để output lặp lại được.
9. QARecord có đúng sáu field §1.2; `query_id` và `source_split` lấy từ quyết định đã đối chiếu. Chạy `validate_qa_record` với map text đại diện trước khi trả. Chỉ `None` ở quyết định nội bộ biểu thị loại; không đưa `source_split=None` vào QARecord. Giữ thứ tự dòng gốc khi nhóm train/dev/test.

## 4. Ranh giới T5.4 và report

T5.3 giữ trách nhiệm dựng evidence/reference/coverage; helper T5.4 chỉ nhận collision đã xác định, trả offset được chọn hoặc None và ghi report. Producer trả kết quả hoàn chỉnh sau cả hai bước. Report rỗng có thể nghĩa là không remap, hoặc chỉ có fallback đúng một occurrence; không dùng report rỗng để suy ra số collision bằng 0.

T5.4 xử lý theo V1.1 §2.3.a, không mặc định chọn occurrence đầu:

- 0 occurrence: bỏ evidence đó, giữ QARecord và các evidence khác, **ghi report**.
- 1 occurrence: lấy offset đó, không cần ghi report.
- >1 occurrence: chọn offset trong text đại diện có `offset / len(representative_text)` gần nhất với `raw_answer_start / len(raw_context)`; **luôn ghi report**, gồm query_id, passage_id, answer_text, toàn bộ offset ứng viên, offset đã chọn. Không dùng local offset chia độ dài chunk nguồn thay cho vị trí trong document QA gốc.

**Quyết định triển khai đã chốt:** khi hai khoảng cách bằng nhau, chọn offset nhỏ hơn. So sánh bằng số nguyên `abs(candidate_offset * len(raw_context) - raw_answer_start * len(representative_text))`, rồi offset; cùng mẫu số dương nên tương đương so khoảng cách tương đối, không tạo hòa giả vì float. Tìm mọi occurrence, kể cả các occurrence chồng nhau. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]

Entry report có đúng các field sau (tên field là lựa chọn triển khai; ý nghĩa theo §2.3.a):

| Field | Giá trị |
|---|---|
| query_id, passage_id, answer_text | Danh tính QA, đại diện và answer gốc |
| answer_index | Chỉ số annotation trong hai list raw, bắt đầu 0 |
| source_answer_start, source_context_char_len | Offset gốc trong document và độ dài document để tái tính vị trí tương đối |
| source_chunk_start_char, source_chunk_end_char | Khoảng chunk của chính document QA, không phải parent đại diện |
| representative_char_len | Độ dài text đại diện |
| candidate_offsets | Đủ offset ứng viên, tăng dần; [] nếu 0 occurrence |
| selected_offset | Offset được chọn, hoặc None (JSON null) nếu 0 occurrence |

Mỗi cặp annotation/chunk cần fallback 0 hoặc >1 occurrence tạo một entry, theo thứ tự raw train rồi test, annotation rồi chunk. Không dedup report: reference lặp vẫn có answer_index riêng. Dedup exact evidence không làm mất dấu vết remap.

`coverage_report` đề nghị cho mỗi lần producer hoàn tất:

```text
input_rows: số dòng GQ raw trước D1
excluded_leakage_rows: số quyết định GQ bị loại theo D1
files:
  qa_germanquad_train.jsonl: {n_records, n_empty_gold, coverage_rate, empty_gold_query_ids}
  qa_germanquad_dev.jsonl:   {n_records, n_empty_gold, coverage_rate, empty_gold_query_ids}
  qa_germanquad_test.jsonl:  {n_records, n_empty_gold, coverage_rate, empty_gold_query_ids}
```

`coverage_rate=(n_records-n_empty_gold)/n_records`; khi file không có record, ghi None, không tự cho 100%. Danh sách ID gold rỗng đầy đủ và sort. `sum(n_records)+excluded_leakage_rows == input_rows`. Tên field là lựa chọn của thẻ; ý nghĩa report theo [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.7 Artifact]. Coverage dưới 98% cần báo để review, không xóa QARecord, đổi overlap hay tạo evidence để ép đạt gate. Trong T5.3 chỉ trả object, chưa ghi file.

## 5. Fixture có expected cụ thể

Tất cả là **ví dụ minh họa**, không phải record dataset. Raw fixture tối thiểu: GQ train `id=1`, question=`"Frage A"`, context và answers theo hàng bên dưới; decision `gq-train-1`, original_split=train, source_row_index=0, question_key=`"frage a"`, source_split=dev. Có thể tạo decision qua T5.2 với fixture ratio=1; đây không phải cấu hình chia dev thật.

Đặt `text = " ".join(f"w{i:03d}" for i in range(201))`. Các chunk đã kiểm bằng code hiện có:

| Chunk | Khoảng char | passage_id |
|---|---|---|
| 0 | [0,499) | c3cbe252c56e270c |
| 1 | [400,899) | 8451b74d508323dc |
| 2 | [800,1004) | 78eaa148763a161a |

| Ca | Input cụ thể | Expected |
|---|---|---|
| Hai chunk overlap | Context `text`, answers text `["w085"]`, start `[425]` | References `["w085"]`; hai evidence: chunk 0 local [425,429), chunk 1 local [25,29) |
| Nhiều references | Context `text`, texts `["w085","w185"]`, starts `[425,925]` | Ba evidence: hai của w085 ở trên, một của w185 trong chunk 2 local [125,129); giữ đủ hai reference |
| Một reference không map | Context `text`, texts `["w085", text[395:505]]`, starts `[425,395]` | Span thứ hai dài 110 char, không chunk nào chứa trọn; references vẫn có hai phần tử, evidence chỉ của w085 |
| Toàn bộ không map | Context `text`, texts `[text[395:505]]`, starts `[395]` | Một QARecord gold rỗng; n_records=1, n_empty_gold=1, coverage=0.0 ở dev |
| Không có answers | Context `""`, texts/starts đều `[]`, passages `{}` | Một QARecord, references/evidences rỗng; không lỗi chia cho 0 ở các split không có record |
| Không lấy occurrence đầu | Context `"alpha Berlin omega Berlin end"`, text `["Berlin"]`, start `[19]`, đại diện nguyên bản | Evidence local [19,25), không phải [6,12); không remap, report remap rỗng dù có hai occurrence |
| Reference lặp | Context `text`, texts `["w085","w085"]`, starts `[425,425]` | References giữ hai phần tử; evidence exact trùng chỉ còn hai entry overlap |
| Cùng passage, hai answer span | Context `"alpha Berlin omega Berlin end"`, texts `["Berlin","Berlin"]`, starts `[6,19]` | Cùng passage_id nhưng giữ cả hai evidence [6,12) và [19,25) |
| Đại diện ở parent khác | QA context `"Berlin"`, start `[0]`; corpus đại diện tạo từ parent `"  Berlin  "` có chunk_start=2 | Evidence local [0,6), không trừ offset 2 của parent đại diện |
| Text khác nhưng slice đúng | QA `"alpha Berlin omega  end"`, start `[6]`; đại diện `"alpha Berlin omega end"` | Cùng hash, slice [6,12) vẫn đúng; dùng trực tiếp, không remap |
| D1 loại QA | Raw có decision source_split=None | Không có QARecord; excluded_leakage_rows tăng 1; map corpus input không bị xóa/sửa |

Fixture collision kiểm trực tiếp kết quả T5.3/T5.4:

| Context QA / đại diện | Raw answer/start | Điều đã kiểm | Expected cuối |
|---|---|---|---|
| `alpha Berlin omega` / `alpha berlin omega` | Berlin / 6 | Cùng ID `1aeb9dce1a833b3a`, đại diện có 0 occurrence chính xác | Bỏ evidence, giữ reference/QARecord và log []/None |
| `alpha  Berlin omega` / `alpha Berlin omega` | Berlin / 7 | Cùng ID trên; source dài 19, đại diện dài 18, ứng viên [6] | Chọn 6, không log |
| `alpha  Berlin omega Berlin end` / `alpha Berlin omega Berlin end` | Berlin / 20 | Cùng ID `09f06d8cf4a66bda`; dài 30/29, ứng viên [6,19], vị trí nguồn 20/30 | Chọn 19 và luôn log |
| `"A  a     "` / `"a a"` | a / 3 | Document dài 9, đại diện dài 3; ứng viên [0,2], khoảng cách cùng 1/3 | Chọn 0 theo tie-break, luôn log |
| `"a\u0308ää"` / `"äää"` | ää / 2 | Cùng NFC/hash, document dài 4, đại diện dài 3; ứng viên chồng nhau [0,1] | Chọn 1 và log cả hai ứng viên |

Không dùng cặp `"alpha omega"` và `"alpha Berlin omega"` làm ví dụ dedup collision: chúng không có cùng nội dung sau normalize. Kết quả 0/1/>1 occurrence phải gắn với một cặp source/representative hợp lệ, không chỉ một chuỗi đứng riêng.

## 6. Mốc dữ liệu thật và giới hạn bằng chứng

| Mốc | Ý nghĩa và nguồn |
|---|---|
| 12.696 dòng GQ sau D1 | 9.445 train + 1.047 dev + 2.204 test; cộng 1.026 dòng loại = 13.722. Đây là số QARecord cuối phải giữ, kể cả gold rỗng. [kiem-chung-T5.2.md §3. Đối chiếu từng dòng trên dữ liệu thật] |
| 1.285 answer nhiều occurrence; 575 không dùng occurrence đầu | 575/1.285 = 44,7%, tính trên 18.054 annotation answer của **toàn bộ** raw GQ train+test trước D1. Không phải số collision cần remap trên dữ liệu sau D1. [results/data_audit_report.json §answers_with_multiple_occurrences_in_context] [results/data_audit_report.json §answer_start_not_first_occurrence_count] [notebooks/00_data_audit.py §main] |
| 184 uncovered cũ | Artifact cũ có total=13.722, uncovered=184, coverage_rate=0.9866. Counter tăng tại `if not gold_indices`, **trước** khi chọn primary chunk. Không quy 184 dòng này cho luật primary-chunk. [data/processed/chunk_coverage_report.json §uncovered_rows] [notebooks/01_build_corpus_base.py §build_qa_records] |

Producer cũ vừa xóa dòng không có gold_indices, vừa chỉ lưu answer thuộc primary chunk; đây là **hai lỗi khác nhau**. Không lấy 184 làm expected mới hoặc dùng coverage gộp để cho qua gate.

Mốc đối chiếu người dùng đo bằng mô phỏng char-offset trên cache, **chưa qua producer mới khi chốt thẻ**:

| Split | Số dòng | Gold rỗng | Coverage |
|---|---:|---:|---:|
| train | 9.445 | 158 | 98,33% |
| dev | 1.047 | 16 | 98,47% |
| test | 2.204 | 8 | 99,64% |
| Bị loại D1 | 1.026 | 2 không phủ khi tái kiểm độc lập | Không áp dụng |

Tổng gold rỗng ở ba split là 182. Producer mới đã đối chiếu đúng ba hàng này. Riêng hàng loại D1, tái kiểm cho **2** dòng không có chunk chứa trọn answer: `gq-train-67129` và `gq-train-70207`, khác mốc 0 người dùng cung cấp. Hai dòng này không tạo QARecord sau D1; không cộng vào 182 hoặc mẫu số coverage của ba file. Lệnh tái kiểm và kết quả ở [kiem-chung-T5.3-T5.4.md §3. Cache thật và hiệu chỉnh mốc D1].

Corpus chỉ từ toàn bộ context GQ: 7.520 ID, 0 ID có nhiều text thô khác nhau, 0 cặp answer/chunk cần remap. Với cache này, đại diện GQ không bị nguồn ưu tiên thấp hơn thay thế theo SOURCE_PRIORITY; đây chưa phải phép build base đầy đủ D6. [src/corpus.py §CorpusBuilder]

Train chỉ hơn ngưỡng 98% khoảng 0,33 điểm phần trăm. Chỉ báo số quan sát; không thay tham số để ép coverage. Base gate artifact vẫn chờ T5.8 và report trên file đã ghi.

## 7. Test-first và lệnh

Nhóm test đã triển khai trong `tests/test_qa_records.py` (80 ca gồm tham số hóa):

- `test_overlap_emits_all_local_evidences`, `test_all_references_survive_unmapped_answers`: bắt luật primary và mất reference.
- `test_unmapped_and_empty_answers_keep_qa_record`: bắt xóa gold rỗng hoặc chia coverage sai mẫu số.
- `test_annotated_occurrence_is_used_without_search`: hai occurrence nhưng giữ raw_start=19.
- `test_duplicate_references_dedup_only_exact_evidence`: dedup exact evidence, không dedup mất span cùng passage.
- `test_source_chunk_offset_is_used_with_another_representative_parent`: bắt trừ provenance của parent khác.
- `test_collision_with_valid_direct_slice_does_not_require_remap`: không gọi fallback khi slice trực tiếp đúng.
- Các ca collision ở §5 kiểm 0/1/>1, tie-break, occurrence chồng nhau, khoảng cách theo document nguồn và đủ field report. Không viết test exception chuyển tiếp.
- `test_d1_decisions_preserve_order_ids_exclusions_and_original_test`, `test_split_identity_must_match_raw`: đúng ID gốc, split/source/index; không xuất dòng loại.
- `test_invalid_raw_records_fail_before_mapping`, `test_missing_or_corrupt_representative_fails`: không tự search sửa raw offset, không coi passage thiếu là gold rỗng; kiểm list answers lệch độ dài và type/bool.
- `test_inputs_are_not_mutated_results_validate_and_repeat`: validator QA kiểm slice trên đại diện, kết quả JSON-serializable, cùng input cho cùng output; `test_coverage_per_file_uses_all_retained_rows_and_sorted_empty_ids` kiểm report từng file.

1. Viết fixture/test trước `src/qa_records.py`; quan sát RED do producer chưa có hoặc hành vi chưa đáp ứng. Khi module chưa tồn tại, ghi rõ trạng thái thiếu module; sau khi tạo khung interface, xác nhận assertion hành vi thất bại trước triển khai, không coi typo/import sai là bằng chứng RED.
2. Viết đường char-offset, helper fallback T5.4 và report đúng thẻ. Không search ngoài nhánh collision hợp lệ.
3. Chạy GREEN và hồi quy; lưu log dưới `docs/learning/lessons/chang-05/`. Chưa chạy notebook build, chưa tạo artifact trong `data/processed/v1.1/`.
4. Qua code gate T5.3/T5.4 khi fixture và validator pass, coverage/report đếm đúng và kiểm lại cache. Chuyển thẻ T5.5 sau review; không đánh dấu sinh viên đã qua gate.

Lệnh chỉ đọc đã đối chiếu repo:

```powershell
Test-Path 'src/qa_records.py'
Test-Path 'tests/test_qa_records.py'
Select-String -LiteralPath 'notebooks/01_build_corpus_base.py' -Pattern 'if not gold_indices|uncovered|primary =|answers_out'
Get-Content 'data/processed/chunk_coverage_report.json' -Encoding UTF8
```

Kỳ vọng lúc chốt thẻ: hai False; sau triển khai hai True. Counter uncovered xuất hiện trước primary; report cũ total=13722/uncovered=184. Lệnh test dưới đã chạy sau triển khai; kết quả cụ thể lưu trong [kiem-chung-T5.3-T5.4.md §2. RED, GREEN và hồi quy].

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_qa_records.py -q -p no:cacheprovider --tb=short
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t53-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
```

## 8. Điểm duyệt và cần xác minh

- Interface và phạm vi gộp đã chốt trước code; implementation đã được người dùng kiểm độc lập và duyệt chuyển bước, commit `1102836`. Bằng chứng ở [kiem-chung-T5.3-T5.4.md §5. Commit sau review]. Checklist dưới dành cho sinh viên, vẫn chưa làm.
- [ ] Giải thích producer trong `src/qa_records.py`, input raw + SplitDecision + map passage đại diện, dạng kết quả §2.
- [ ] Tự kiểm ranh giới T5.3/T5.4: char-offset mặc định, chỉ fallback khi cần, report 0/>1 đúng §4.
- [ ] Tự giải thích fixture và khác biệt giữa giữ QARecord với đưa vào N_eval.
- Tie-break nhỏ hơn và tên field entry đã chốt ở §4 trước code; đây là quyết định triển khai trong phạm vi được người dùng cho chọn, không gán thêm yêu cầu cho V1.1.
- CẦN XÁC MINH: coverage/report trên các file artifact sau build T5.8 — đối chiếu mốc §6 bằng bản ghi thật, không suy từ 184 cũ hoặc 1.285 annotation nhiều occurrence.
- CẦN XÁC MINH: caller ngoài `src/`, `tests/`, `notebooks/`, snapshot và thứ tự xây corpus thật — người triển khai/chủ project kiểm khi tích hợp; không cản fixture T5.3.
