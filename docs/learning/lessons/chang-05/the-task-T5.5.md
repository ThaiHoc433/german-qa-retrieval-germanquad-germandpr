---
tasks: [T5.5]
buoi: 14
trang_thai: chưa làm
---

# Thẻ T5.5 — DPR positive union và hard-negative groups

Thẻ lập ngày 2026-09-22; cập nhật sau review ngày 2026-09-24. `src/retrieval_records.py` và `tests/test_retrieval_records.py` đã có, đúng vị trí dự kiến trong catalog, commit `945f317`. Interface, thứ tự output và field report đã được người dùng kiểm và cho chuyển bước; đây là quyết định triển khai, không phải amendment spec V1.1. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi] [kiem-chung-T5.5.md §5. Vá test sau review và commit]

## 1. Phạm vi, phụ thuộc và caller

- T5.5 phụ thuộc T5.1/T5.2: reuse chunk/hash, quyết định D1 theo từng dòng và validator training. T5.3/T5.4 đã commit `1102836`, không cần sửa producer QA để làm DPR. [src/corpus.py §chunk_text] [src/splits.py §assign_splits] [src/schemas.py §validate_retrieval_training_record]
- Producer thuần CPU: nhận raw + decisions + corpus đại diện đã chọn xong, trả records/report trong bộ nhớ. Không gọi Hub, không ghi artifact, không thêm hard negative vào corpus, không train.
- T5.5 lưu mọi positive hợp lệ và mọi chunk còn lại của từng hard-negative group. Chọn một positive/epoch và tối đa một chunk/group/epoch thuộc sampler training ở module 5, không viết sampler trong T5.5. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.3 Retrieval training record — dùng cho GermanDPR] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.3 Quy tắc xử lý]
- Caller cũ chỉ ở `notebooks/01_build_corpus_base.py`: `main` gọi `build_retrieval_training_records` hai lần, hiện gán mọi chunk positive, flatten hard negative và truyền source bị D6 cấm vào builder. Chuyển caller/ghi file thuộc T5.8. Không import notebook để lấy hàm vì có `OUT_DIR.mkdir` ở top-level. [notebooks/01_build_corpus_base.py §build_retrieval_training_records] [notebooks/01_build_corpus_base.py §main]

## 2. Interface đề nghị

```python
# Interface đã triển khai trong src/retrieval_records.py
from collections.abc import Mapping, Sequence
from typing import TypedDict
from src.splits import SplitDecision

class RetrievalProductionResult(TypedDict):
    records: dict[str, list[dict]]  # train/dev/test, đúng RetrievalTrainingRecord §1.3
    germandpr_chunk_mapping_report: dict

def build_retrieval_training_records(
    gdpr_train_rows: Sequence[Mapping[str, object]],
    gdpr_test_rows: Sequence[Mapping[str, object]],
    split_decisions: Sequence[SplitDecision],
    passages: Mapping[str, Mapping[str, object]],
) -> RetrievalProductionResult: ...
```

Input là hai split DPR raw đầy đủ, còn nguyên thứ tự trước lọc; `to_pylist()` của Parquet trả Python list/dict. `split_decisions` có thể là toàn bộ output T5.2 gồm GQ; chỉ lấy phần DPR, không tự chia split lại. Map passage dùng để xác nhận positive ID tồn tại và kiểm PassageRecord/normalize/hash; không đòi hard-negative ID có trong map. [src/splits.py §SplitDecision] [doi-chieu-T5.5.md §1. Nguồn và cách đo]

Output record có đúng sáu field: `query_id`, `question`, `positive_texts`, `positive_passage_ids`, `hard_negative_groups`, `source_split`. Mỗi group có đúng `source_context_id`, `texts`, `passage_ids`. Không thêm answers/offset/source_dataset vào record; metadata báo cáo nằm ngoài record. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.3 Retrieval training record — dùng cho GermanDPR]

**Lựa chọn text đã triển khai và review:** `positive_texts` lấy text chunk **gốc được tìm thấy answer**, theo thứ tự context/chunk, giữ bản hợp lệ đầu khi trùng ID. Không thay bằng text đại diện khác casing/whitespace, vì §2.3.f tìm occurrence trên positive context gốc và field training lưu chunk đó. Map chỉ ràng buộc positive ID/normalize/schema. Ví dụ raw `Berlin`, đại diện `berlin`: cùng ID, positive_texts vẫn `Berlin`; DPR không có span output nên không dùng remap QA. V1.1 không ghi rõ phải chọn raw hay representative text khi cùng ID có text khác nhau; đây là quyết định triển khai, không phải câu chữ bổ sung vào spec. Trên cache đã đo, 0 positive chunk khác text đại diện nên lựa chọn này không đổi các mốc thật. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý] [doi-chieu-T5.5.md §2. Mốc đối chiếu] [kiem-chung-T5.5.md §5. Vá test sau review và commit]

## 3. Kiểm input và quyết định D1

1. `question` phải là chuỗi không trắng. `answers` là list chuỗi không trắng; list rỗng hợp lệ nhưng không có positive để xuất. Không nhận dict answers của GermanQuAD hoặc processed cũ thiếu raw context.
2. Raw `positive_ctxs`/`hard_negative_ctxs` là **struct có list song song** `title`, `text`, `passage_id`, không phải list dict. Kiểm mapping, ba list bằng độ dài; text là str (cho phép rỗng/whitespace để đếm unmappable/empty group), title str hoặc None, passage_id raw là str và được phép rỗng. Không dùng ID raw làm ID chunk/parent: tự hash bằng công thức V1.1. `negative_ctxs` nếu có kiểm cùng dạng, nhưng không chuyển thành hard-negative groups; spec T5.5 chỉ chỉ định positive/hard negative. [doi-chieu-T5.5.md §1. Nguồn và cách đo]
3. Không hard-code một positive hoặc ba hard-negative context. Duyệt mọi positive context và mọi answer. Raw hiện có positive=1 ở mọi dòng; hard-negative train có 8 dòng chỉ 1–2 context. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]
4. Mỗi raw DPR phải có đúng một decision có `source_dataset=germandpr`, `query_id=gdpr-{split_gốc}-{index_gốc}`, `original_split`, `source_row_index` int không bool và `question_key` khớp `question_key(raw_question)`. Decision DPR thừa/thiếu/trùng hoặc sai field phải báo lỗi. Decision GQ không tham gia producer DPR; dataset không nhận diện được là lỗi. Không lấy index sau khi loại record.
5. Test gốc chỉ nhận `source_split=test`; train gốc chỉ train/dev/None. `None` loại vì D1, không xuất record, không lookup positive vào corpus. Vẫn kiểm schema raw trước khi skip dòng loại. Giữ thứ tự raw khi gom records vào từng split. Question giữ nguyên raw.
6. Với positive đã được chọn, thiếu corpus ID, key map khác record ID, normalize/hash/schema sai → dừng lỗi, không tự thêm corpus và không đổi thành unmappable. Passage không dùng không phải kiểm lại toàn bộ. Validator hiện có kiểm hash/field training nhưng producer phải tự kiểm membership positive, mask và bảo toàn dòng. [src/schemas.py §validate_retrieval_training_record] [src/schemas.py §validate_passage_record]

## 4. Positive occurrence union

Occurrence là một lần xuất hiện chính xác của answer trong chuỗi. Theo V1.1 §2.3.f:

1. Chunk mỗi positive context bằng `chunk_text` 100 từ, stride 80, độc lập answer.
2. Với **mọi** answer text, tìm **mọi** occurrence exact/case-sensitive trên context gốc; không strip/lower answer, không chỉ `.find` lần đầu. Bước tìm kế tiếp tăng 1 ký tự để không bỏ occurrence chồng nhau.
3. Với mỗi span, dùng `find_span_chunks` lấy mọi chunk chứa trọn span. Hợp các index trong context; sau đó dedup toàn query theo `passage_id`.
4. Thứ tự đề nghị: context raw rồi index chunk tăng dần; mỗi ID giữ text của chunk hợp lệ đầu tiên. Không dùng thứ tự set hoặc thứ tự answers làm thứ tự positive. Danh sách song song, khác rỗng ở mọi record được xuất.
5. Không tìm được bất kỳ chunk hợp lệ nào → loại **cả DPR record** và liệt kê query_id trong mapping report theo split. Khác QA: không xuất record positive rỗng hoặc `[None]`, không fallback về context dài chưa chunk.

[docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]

## 5. Hard-negative groups và mask

- `gold_set` là toàn bộ positive_passage_ids của chính query, không chỉ positive chọn để train một epoch.
- Với mỗi hard-negative context raw: `source_context_id=passage_id(toàn context chưa chunk)`, lấy mọi chunk bằng cùng chunker. Không tìm answer để quyết định chunk negative.
- Loại mọi chunk có ID trong gold_set, kể cả chunk có text khác casing nhưng cùng hash. Nếu group hết chunk, bỏ group và ghi filtering report. Context rỗng không tạo group rỗng; ghi lý do `empty_context`.
- Đề nghị giữ một entry theo **mỗi occurrence context raw**, đúng §2.3.g; không dedup/ghép hai context khác vị trí chỉ vì cùng source_context_id. Trong group giữ thứ tự mọi chunk còn lại, kể cả hai chunk trùng ID; spec chỉ bắt dedup positive, không bắt dedup hard negative. Fixture phải khóa lựa chọn này; nếu muốn đổi cần duyệt lại.
- Text hard negative lấy chunk gốc của context đó, kể cả khi corpus có ID với đại diện khác text. ID vắng corpus vẫn hợp lệ; không thêm nó vào corpus, không blacklist ID này khỏi corpus nếu nguồn hợp lệ khác đã có.
- Lúc train mới cap ≤1 chunk/group/epoch và mask in-batch theo toàn gold_set. Bài này chỉ lưu group đầy đủ và mask overlap gold trong record. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.3 Quy tắc xử lý]

## 6. Report đề nghị và mẫu số

`germandpr_chunk_mapping_report` trả object sau, T5.8 mới ghi `germandpr_chunk_mapping_report.json`. Tên field là lựa chọn đề nghị, ý nghĩa theo §2.7:

```text
input_rows: mọi DPR raw train+test
excluded_leakage_rows: số quyết định DPR source_split=None
excluded_leakage_query_ids: toàn bộ ID loại D1, sort
files:
  retrieval_germandpr_{train,dev,test}.jsonl:
    n_input_after_d1: số quyết định gán vào split, trước loại unmappable
    n_records: số record xuất
    n_unmappable: số record không map được positive
    unmappable_query_ids: toàn bộ ID này, sort
    n_single_positive: số record xuất có đúng 1 positive ID
    n_multiple_positive: số record xuất có >1 positive ID
    positive_chunks_per_record: histogram số positive ID trên record xuất
    n_positive_chunks: tổng độ dài positive_passage_ids của record xuất
    hard_negative_groups_per_record: histogram số group trên record xuất, kể cả 0
    hard_negative_chunks_per_group: histogram số chunk trên group xuất
    n_hard_negative_groups, n_hard_negative_chunks: tổng output
    n_masked_hard_negative_chunks: số occurrence chunk bị mask trên record xuất
    n_dropped_hard_negative_groups: số context group rỗng/bị mask hết trên record xuất
    n_positive_text_not_raw_chunk: 0 khi mọi positive có chứng cứ raw chunk
    n_null_positive_ids: 0 khi không có None
    all_positive_texts_are_raw_chunks: true nếu bộ đếm tương ứng = 0
    no_null_positive_ids: true nếu bộ đếm tương ứng = 0
hard_negative_filtering: danh sách entry context bị mask hoặc rỗng
```

Histogram dùng key chuỗi số khi trả object để JSON round-trip không đổi; cả ba file luôn có, kể cả split rỗng. Hai boolean kiểm bất biến rỗng là true khi n_records=0, phải đọc kèm n_records. Không so text với context để kết luận “dùng nguyên context”: context ≤100 từ vốn tạo đúng một chunk có text bằng context; phải truy ngược về output `chunk_text` chứa occurrence.

Entry filtering gồm `query_id`, `source_context_index` (0-based raw), `source_context_id`, `removed_passage_ids` (đủ occurrence ID bị mask theo thứ tự), `group_removed` bool, `reason` (`gold_overlap` hoặc `empty_context`). Nếu hết chunk vì mask, reason vẫn gold_overlap. Chỉ ghi cho record được xuất; record unmappable có danh sách riêng, không cộng group/chunk của nó vào output histogram. Trường `source_context_index` phân biệt hai occurrence context cùng ID. Các bộ đếm phải được tính/kiểm từ output và chứng cứ mapping, không hard-code true/0 để qua gate.

Thứ tự filtering là raw train rồi raw test, index dòng rồi index context tăng dần. Không sort bằng passage_id hoặc dedup entry; mỗi context bị lọc có đúng một entry.

Bảo toàn: `input_rows = excluded_leakage_rows + sum(n_records+n_unmappable)`; mỗi file `n_input_after_d1=n_records+n_unmappable`, `n_records=n_single_positive+n_multiple_positive`. Không tính record bị D1 loại vào n_unmappable. Không áp ngưỡng QA coverage ≥98% cho DPR; yêu cầu là positive hợp lệ và công bố số bị loại. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.7 Artifact]

## 7. Fixture và expected trước code

Tất cả là **ví dụ minh họa**. Raw fixture: `question='Frage A'`, `answers` theo bảng, nhóm struct `{'title': [...], 'text': [...], 'passage_id': ['']*n}`; mỗi context có title `None`. Decision có thể lấy từ assign_splits với dev_ratio=1 cho fixture, không phải config thật. Corpus fixture dùng CorpusBuilder với positive contexts, không thêm hard negatives.

Đặt `text=' '.join(f'w{i:03d}' for i in range(201))`. Parent ID `2708af420817cfc6`; ba chunk:

| Chunk | Khoảng | ID |
|---|---|---|
| 0 | [0,499) | c3cbe252c56e270c |
| 1 | [400,899) | 8451b74d508323dc |
| 2 | [800,1004) | 78eaa148763a161a |

| Ca | Input | Expected |
|---|---|---|
| Mọi answer | Context text, answers `[w010,w185]` | Span [50,54), [925,929) → chỉ chunk 0,2; không gán chunk 1 positive |
| Overlap | Context text, answer w085 | Span [425,429) → chunk 0,1; dedup theo ID |
| Mọi occurrence | Thay token 10 và 185 trong text bằng Berlin, answers `[Berlin]` | Occurrence start 50,927; positive chunk 0 ID `5a94dae7398cce00`, chunk 2 ID `d5f147a2841f1e2f`; chunk 1 ID `8451b74d508323dc` không positive |
| Không map | Context text, answers `[text[395:505]]`, hoặc answers `[]`, hoặc positive_ctxs ba list rỗng | Record bị loại khỏi output, n_unmappable tăng 1, query_id trong report |
| Nhiều positive context | Contexts Berlin, Ulm; answers `[missing,Ulm,Berlin]` | Hai ID theo thứ tự context: `fb38a93fd89e1a5f`, `37bfc8ba5ec56136`; không dừng ở answer đầu |
| Dedup positive | Lặp context Berlin, answers Berlin lặp | Chỉ 1 positive ID/text; đảo answers không đổi thứ tự positives |
| Mask cả gold_set | Positive context text, answers w010/w185; hard contexts text và Paris | Gold chunk 0,2; group parent `2708af420817cfc6` chỉ còn chunk 1. Group Paris có ID `1670f2e42fefa504`, hợp lệ dù vắng corpus |
| Group bị mask hết | Positive Berlin; hard contexts Berlin và Paris | Bỏ group Berlin, log ID bị mask và group_removed=true; giữ group Paris |
| Context rỗng | Positive Berlin; hard context whitespace | Không xuất group rỗng; filtering reason=empty_context, removed_passage_ids=[] |
| Cùng source_context_id | Hai hard context Paris lặp, positive Berlin | Hai group occurrence cùng ID, mỗi group 1 chunk; không gộp ngầm |
| Chunk trùng trong một group | Positive Berlin; hard context `' '.join(['x']*201)` | Một group giữ ba chunk 100/100/41 từ; hai ID đầu đều `efa6349ac9ff5f40`, ID cuối `66bc7fc85822c579`. Không dedup mất chunk hoặc lấy cap ở data prep |
| Raw/representative khác text | Positive raw Berlin, đại diện berlin; answer Berlin | Theo đề xuất §2: positive_texts=[Berlin], cùng hash/membership; không remap QA |
| Context ngắn | Positive context Berlin, answer Berlin | Một chunk có text bằng context vẫn hợp lệ; không gọi đây là fallback context chưa chunk |
| Không mask theo query khác | Query A positive Berlin, hard Ulm; query B positive Ulm | Query A giữ Ulm làm negative, chỉ mask positive của chính A |
| D1 và lỗi input | Dòng loại None, decision thiếu/thừa/trùng/sai index/key/split, raw struct lệch list, positive thiếu corpus | None bỏ đúng dòng; các input vi phạm báo lỗi rõ, không biến lỗi thành unmappable |

Expected text là slice chunk nguyên bản, không chỉ test số lượng. Trong test code cần kiểm schema, thứ tự, bất biến input, JSON round-trip và đầy đủ danh sách ID bị loại. [src/corpus.py §chunk_text] [src/corpus.py §find_span_chunks] [src/schemas.py §validate_retrieval_training_record]

## 8. Mốc thật và lệnh đối chiếu

Các số sau là **phép tính chỉ đọc trước producer**, dùng cache và API chunk/splits hiện có. Lệnh đầy đủ và histogram ở [doi-chieu-T5.5.md §3. Lệnh tái kiểm chỉ đọc]. Không gọi đây là kết quả nghiệm thu producer hoặc artifact.

| Split sau D1 | Trước mapping | Không map được | Dự kiến output | Positive chunk | HN group | HN chunk sau mask |
|---|---:|---:|---:|---:|---:|---:|
| train | 8.346 | 17 | 8.329 | 9.440 | 24.972 | 49.442 |
| dev | 928 | 2 | 926 | 1.042 | 2.776 | 5.566 |
| test | 1.025 | 2 | 1.023 | 1.212 | 3.069 | 6.286 |

`10.278 output + 21 unmappable + 1 loại D1 = 10.300`. Mask lần lượt 7/0/1 chunk trên record xuất; drop group 3/0/0. Không hard-code 3 group: raw đã có context count khác 3, và mask có thể giảm group nữa.

Record `gdpr-train-0` (gán dev): 230 từ sinh 3 chunk, nhưng answer chỉ được chunk **index 2** chứa trọn → **1 positive ID `d20519f819d35708`**, text dài 394 ký tự. Hard-negative groups có 3/3/2 chunk như cũ. “3 positive” trong artifact cũ là gán nhãn tràn lan, không phải expected T5.5.

Phép dựng corpus trong bộ nhớ từ **mọi GQ context + mọi DPR positive context** cho 7.520 ID, tất cả đại diện GQ; 0 positive raw chunk khác text đại diện. Đây chưa có title exclusions/manifest/acceptance T5.6/T5.8, không đóng dấu base đã xong. Thứ tự ưu tiên nguồn GQ > DPR positive > Wikipedia phải tiếp tục được giữ. [src/corpus.py §SOURCE_PRIORITY] [doi-chieu-T5.5.md §2. Mốc đối chiếu]

Lệnh đọc nguồn đã đối chiếu:

```powershell
Test-Path 'src/retrieval_records.py'
Test-Path 'tests/test_retrieval_records.py'
Select-String -LiteralPath 'notebooks/01_build_corpus_base.py' -Pattern 'def build_retrieval_training_records|germandpr_hard_negative|pos_ids.extend|hn_ids.extend'
```

Kỳ vọng hiện tại: True/True; notebook vẫn có hàm cũ và hai dòng extend gán/flatten mọi chunk. T5.8 mới chuyển caller sang producer trong src.

## 9. Test-first và gate thực thi

File test đã có: `tests/test_retrieval_records.py`. Quy trình dưới là đường thực thi để người học tái hiện; mỗi hàng fixture §7 cần test hành vi, cộng lỗi input/schema, negative IDs vắng corpus, evidence positive membership và report bảo toàn từng split.

1. Viết test với raw struct thật; xác nhận RED có nguyên nhân hành vi, tách rõ thiếu module với assertion thất bại sau khung interface.
2. Implement producer thuần theo thẻ đã duyệt, không sửa chunker/QA để hợp fixture. Không import helper private của qa_records vì DPR có schema và loại record khác; reuse API công khai corpus/splits/schemas.
3. GREEN, hồi quy, kiểm producer thật trên cache và so expected §8. Không sửa expected để hợp code; giải thích mọi chênh lệch từ source/thuật toán trước khi nghiệm thu.
4. Gate code: fixture/schema/report pass, mọi positive có chunk raw chứa occurrence và ID trong corpus, negative group/mask đúng, đủ ID loại, input bất biến. Base artifact và đổi notebook caller vẫn thuộc T5.8. Trạng thái sinh viên chưa làm.

Lệnh đã xác minh với file test hiện có; targeted sau review có 68 ca. Log §5 ghi lệnh hồi quy toàn repo:

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_retrieval_records.py -q -p no:cacheprovider --tb=short
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t55-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
```

## 10. Điểm cần duyệt

- [ ] Interface/đường dẫn §2 và lựa chọn positive_texts từ raw chunk, corpus map kiểm ID/normalize/schema.
- [ ] Thứ tự/dedup positive, giữ context occurrence riêng ở hard negatives và report §5–6.
- [ ] Tự giải thích fixture và số đo §7–8 trước triển khai.
- CẦN XÁC MINH: caller ngoài src/tests/notebooks và revision khi build artifact — người triển khai kiểm tại T5.8.
- Các số §8 đã được tái đo bằng producer thật trong log kiểm chứng §3; khi đổi revision hoặc build artifact phải kiểm lại, không dùng baseline mô phỏng thay nghiệm thu.

**Code T5.5 đã review và commit; chuyển sang thẻ T5.6.** Những checkbox trên dành cho người học, vẫn chưa làm. Mốc mô phỏng §8 đã được so với producer thật trong [kiem-chung-T5.5.md §3. Cache thật và đối chiếu mốc §8]; nghiệm thu artifact vẫn chờ T5.8.
