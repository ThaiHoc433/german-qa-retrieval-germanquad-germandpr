---
tasks: [T5.5]
trang_thai_hoc_vien: chưa làm
---

# Kiểm chứng T5.5 — DPR positive union và hard-negative groups

## 1. Quyết định và phạm vi

Ngày 2026-09-23, triển khai đúng interface/quy tắc đã đề nghị (chưa duyệt riêng từng điểm) trong
`the-task-T5.5.md §2–§6`: signature `build_retrieval_training_records(gdpr_train_rows, gdpr_test_rows,
split_decisions, passages)`, `positive_texts` lấy nguyên text chunk raw tìm thấy answer (không thay bằng
text đại diện corpus), mỗi hard-negative context raw giữ một entry riêng (không gộp theo
`source_context_id`), report `germandpr_chunk_mapping_report` đúng field liệt kê ở §6. Đây là lựa chọn
triển khai theo đề xuất trong thẻ tại thời điểm 2026-09-23. Ngày 2026-09-24, người dùng đã kiểm độc lập,
xác nhận interface/text/report đúng thẻ và yêu cầu vá test rồi commit; xem §5. [the-task-T5.5.md §2. Interface đề nghị]
[the-task-T5.5.md §10. Điểm cần duyệt]

Hai file mới `src/retrieval_records.py` và `tests/test_retrieval_records.py`. Producer chỉ tạo object
trong bộ nhớ; không ghi dataset, không gọi Hub, không train, không đổi caller `notebooks/01_build_corpus_base.py`
(thuộc T5.8). Trạng thái học viên vẫn chưa làm; review code không thay trạng thái học viên.

## 2. RED, GREEN và hồi quy

1. Trước khi viết test, tính tay 12 ca trong bảng fixture §7 bằng trực tiếp `chunk_text`/`find_span_chunks`/
   `passage_id` đã có (không qua module mới) để xác nhận số trong thẻ đúng trước khi đưa vào assertion.
   Toàn bộ khớp thẻ, gồm overlap giữa chunk và ID trùng trong `' '.join(['x']*201)` (`efa6349ac9ff5f40` × 2,
   `66bc7fc85822c579`).
2. Viết `tests/test_retrieval_records.py` (65 test) trước khi tạo `src/retrieval_records.py`. Collection báo
   `ModuleNotFoundError: No module named 'src.retrieval_records'` — bằng chứng thiếu module, chưa phải
   assertion hành vi.
3. Viết implementation theo §2–§6 của thẻ. Chạy lại: **65 passed in 0.15s**, không phải sửa test sau khi
   thấy fail. Hồi quy toàn repo: **395 passed in 1.12s** = 330 nền (từ T5.3/T5.4) + 65 mới.

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_retrieval_records.py -q -p no:cacheprovider --tb=short
& '.venv/Scripts/python.exe' -B -m pytest tests -q -p no:cacheprovider --tb=short
```

65 test ban đầu phủ: union nhiều occurrence, dedup positive theo ID giữ thứ tự
context/chunk, record không map bị loại toàn bộ, mask hard-negative theo gold_set của chính query (không
rò query khác), context rỗng vs hết chunk vì mask (hai `reason` khác nhau), hai occurrence cùng
`source_context_id` giữ hai group riêng, ID trùng trong một group không bị dedup, raw/đại diện khác casing
cùng ID, D1 (test gốc không bị gỡ/gán lại, dòng loại vẫn kiểm raw trước khi bỏ), decision thiếu/thừa/trùng/
sai field, raw struct lệch list, positive thiếu corpus/sai schema, bảo toàn số theo từng split và thứ tự
`hard_negative_filtering` (train rồi test, index dòng rồi index context tăng dần), input bất biến +
determinism + JSON round-trip + qua `validate_retrieval_training_record`.
[tests/test_retrieval_records.py §test_hard_negative_group_masks_gold_overlap_chunks]
[tests/test_retrieval_records.py §test_duplicate_chunk_ids_within_one_hard_negative_group_are_not_deduped]
[tests/test_retrieval_records.py §test_report_conserves_counts_and_orders_filtering_train_then_test]

## 3. Cache thật và đối chiếu mốc §8

Đọc Parquet local, Python 3.13.3; GQ revision `a2f3a59f0be843fc305d0417d7292ef0b1a66884`, DPR revision
`d0b867a49c45a44d172041351b5df73a87714372`. Corpus trong bộ nhớ = mọi GQ context + mọi DPR positive context,
7.520 ID (khớp `doi-chieu-T5.5.md §2`). D1 dùng `assign_splits` mặc định seed=42/ratio=0.10.

Chạy **producer thật** (không phải thuật toán chỉ đọc của `doi-chieu-T5.5.md §3`) trên toàn bộ raw và so
với bảng mốc ở `the-task-T5.5.md §8`:

| Split | n_input_after_d1 | n_unmappable | n_records | n_positive_chunks | HN group | HN chunk | masked | dropped group |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 8.346 | 17 | 8.329 | 9.440 | 24.972 | 49.442 | 7 | 3 |
| dev | 928 | 2 | 926 | 1.042 | 2.776 | 5.566 | 0 | 0 |
| test | 1.025 | 2 | 1.023 | 1.212 | 3.069 | 6.286 | 1 | 0 |

Cả ba split khớp **chính xác từng số** với mốc §8, kể cả danh sách đầy đủ 21 `unmappable_query_ids` (17
train + 2 dev + 2 test). `report['input_rows']=10300`, `excluded_leakage_rows=1`, tổng record xuất
`8.329+926+1.023=10.278` — đúng bảo toàn `10.278+21+1=10.300`. Record `gdpr-train-0` (gán dev): 1 positive
ID `d20519f819d35708`, text dài 394 ký tự, 3 hard-negative group độ dài `[3,3,2]` — khớp `the-task-T5.5.md §8`
dòng "Record gdpr-train-0".

Kiểm bất biến/determinism trên dữ liệu thật: digest input (raw DPR + decisions + passages) trước và sau
lời gọi bằng nhau (`bc23bdd6...b06d742`), gọi lại `build_retrieval_training_records` với cùng input trả
kết quả JSON giống hệt (`result_sha256=9237317...06cfddd`), và mọi record xuất qua
`validate_retrieval_training_record` không lỗi. Tổng record xuất trên toàn raw = **10.278**, khớp bảng
trên. Đây là kiểm module hiện có bằng dữ liệu thật trong bộ nhớ, **chưa phải acceptance base/artifact** —
T5.6/T5.8 mới ghi corpus D6 + title exclusions + manifest lên đĩa. [src/retrieval_records.py
§build_retrieval_training_records] [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T5.8 — Phạm vi báo cáo và nghiệm thu]

Lệnh đã chạy (rút gọn; script đầy đủ tải 4 file Parquet, build corpus, gọi producer, so từng field với bảng
trên và in digest):

```powershell
& '.venv/Scripts/python.exe' -B -c "
from pathlib import Path
import pyarrow.parquet as pq
from src.corpus import CorpusBuilder
from src.splits import assign_splits
from src.retrieval_records import build_retrieval_training_records
hub = Path('C:/Users/PC/.cache/huggingface/hub')
sources = [('germanquad','a2f3a59f0be843fc305d0417d7292ef0b1a66884'),
           ('germandpr','d0b867a49c45a44d172041351b5df73a87714372')]
inputs = []
for ds, rev in sources:
    for sp in ('train','test'):
        inputs.append(pq.read_table(hub/('datasets--deepset--'+ds)/'snapshots'/rev/'plain_text'/sp/'0000.parquet').to_pylist())
gq_train, gq_test, gdpr_train, gdpr_test = inputs
decisions, _ = assign_splits(gq_train, gq_test, gdpr_train, gdpr_test)
builder = CorpusBuilder()
for r in gq_train+gq_test: builder.add_text(r['context'],'germanquad')
for r in gdpr_train+gdpr_test:
    for t,title in zip(r['positive_ctxs']['text'], r['positive_ctxs']['title']):
        builder.add_text(t,'germandpr_positive',title=title)
passages = {p['passage_id']:p for p in builder.records()}
result = build_retrieval_training_records(gdpr_train, gdpr_test, decisions, passages)
print(sum(len(r) for r in result['records'].values()))
"
```

## 4. Giới hạn và bước tiếp theo

- T5.5 chỉ tạo `RetrievalTrainingRecord`/report trong bộ nhớ. Không sửa spec/roadmap/catalog, không đổi
  notebook caller, không ghi `retrieval_germandpr_{train,dev,test}.jsonl` lên đĩa — thuộc T5.8.
- Chưa chọn positive/hard-negative chunk theo epoch (sampler thuộc module 5 huấn luyện, ngoài phạm vi T5.5).
- Interface, text raw, thứ tự/group/report đã được người dùng kiểm và cho chuyển bước ngày 2026-09-24.
  Checklist tự giải thích của học viên vẫn chưa làm; kết quả review ghi ở §5.
- T5.6 (Base D6 + title exclusions P4) phụ thuộc T5.4 và T5.5; bước kế tiếp là soạn thẻ, chưa viết code.
- CẦN XÁC MINH: caller ngoài src/tests/notebooks, revision và artifact/report trên đĩa khi tích hợp T5.8.

## 5. Vá test sau review và commit

Ngày 2026-09-24, chạy baseline targeted **65 passed in 0.16s**. Ba ca mới đều qua public producer và
CorpusBuilder thật, không mock chunker hoặc kiểm chuỗi source:

| Bất biến | Fixture thêm | Mutation bị bắt |
|---|---|---|
| Answer tự chồng vẫn phủ chunk cuối | 110 token `a`; answer 30 token `a`. Hai chunk [0,199), [160,219); occurrence tại 160 chỉ tìm được khi giữ chồng nhau | Đổi `start+1` thành `start+len(answer)` |
| Giữ text hợp lệ đầu tiên khi trùng ID | Một context 100 `Berlin` + 80 `berlin`; hai chunk 100 từ cùng ID `08269b9f6f3a1a29` nhưng khác casing | Bỏ điều kiện `pid not in positives` làm text sau ghi đè |
| Thứ tự chunk theo nguồn | Context w000…w739; answers w110/w670 chỉ thuộc index 1/8 | Bỏ `sorted(indices)` |

Ca đơn giản `aaa`/`aa` không đủ bắt lỗi ở output producer: tìm 1 hay 2 occurrence vẫn tạo cùng một
positive ID. Fixture dài làm lỗi bỏ occurrence thay đổi chính tập positive. Ca index {1,8} bắt mutation
trên CPython 3.13.3 (thứ tự set 8,1); expected output 1,8 vẫn đúng ở runtime khác, nhưng khả năng bắt
mutation không được bảo đảm trên mọi triển khai set. [tests/test_retrieval_records.py §test_self_overlapping_answer_adds_positive_at_last_chunk_boundary]
[tests/test_retrieval_records.py §test_positive_dedup_keeps_first_raw_text_for_same_id_with_different_case]
[tests/test_retrieval_records.py §test_positive_chunks_follow_source_order_for_sparse_indices]

Mỗi mutation được compile trong bộ nhớ của một process Python riêng, thay callable rồi chạy chính ba
test mới: mỗi lượt **1 failed, 2 passed**, exit 1 vì assertion kết quả. Tổng **3/3 KILLED**. Hash file
`src/retrieval_records.py` trước/sau không đổi. Đây là số root tự kiểm cho ba mutation còn sống; không
gộp thành tuyên bố root đã chạy lại toàn bộ 13 mutation của người dùng.

Code đúng: **68 passed in 0.16s**. Hồi quy tại repo root, không giới hạn vào một file hoặc thư mục test:

```powershell
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t55-review-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider --basetemp $taskTemp
```

Kết quả **398 passed in 1.14s** = 330 nền + 65 test T5.5 ban đầu + 3 ca bổ sung. Reviewer độc lập kiểm
ba fixture/ID và chạy targeted 68 passed, không phát hiện vấn đề chặn. Không sửa production trong lượt vá.
Commit **`945f317`** trên `feature/v1.1-data-pipeline` chỉ có `src/retrieval_records.py` và
`tests/test_retrieval_records.py`, 755 dòng thêm. Staged diff check sạch; không kèm docs, không push.
