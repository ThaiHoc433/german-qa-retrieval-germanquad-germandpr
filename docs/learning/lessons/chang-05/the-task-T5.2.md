---
tasks: [T5.2]
buoi: 12
trang_thai: chưa làm
---

# Thẻ T5.2 — Gán split D1 theo từng dòng

Người dùng đã cho bắt đầu T5.2 sau khi sửa nhận định về whitespace và thêm mốc train/dev dưới đây. Nhánh thực thi đã có [log RED/GREEN và kiểm cache](kiem-chung-T5.2.md). Interface của thẻ là lựa chọn triển khai theo spec V1.1 frozen, không sửa spec hoặc roadmap. CPU local, không download, không train và không build base trong task này. Trạng thái học của sinh viên vẫn chưa làm.

Nguồn: [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T5.2 — Thẻ mẫu: Gán split theo D1] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.7 Artifact].

## 1. Phạm vi và tiền đề

- Phụ thuộc T5.0; nhánh T5.0/T5.1 đã có test và [log kiểm chứng](kiem-chung-T5.0-T5.1.md). Không suy ra sinh viên đã đạt gate từ kết quả code.
- T5.2 sở hữu normalization câu hỏi, quyết định theo nguồn từng dòng, chia train/dev chung theo key và report leakage. Không đọc answer để quyết định split; chưa loại GermanDPR không map được chunk — đó là T5.5. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]
- T5.3/T5.5 sẽ dùng quyết định theo dòng này để tạo record; T5.8 chuyển caller và ghi `data/processed/v1.1/split_report.json`. T5.2 chỉ trả dữ liệu trong bộ nhớ và kiểm trên fixture/cache, không ghi đè artifact cũ. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi]

## 2. Files và hệ quả tương thích

| File | Hiện trạng lúc lập thẻ | Việc dự kiến |
|---|---|---|
| `src/splits.py` | `question_key` có thêm collapse whitespace; `assign_splits` nhận bốn list câu hỏi, trả assignment theo key | Bỏ collapse; nhận record, trả quyết định theo từng dòng và report |
| `tests/test_splits.py` | Chưa tồn tại lúc lập thẻ | Tạo test-first cho D1, ID, normalization, số key/dòng và ổn định |
| `notebooks/01_build_corpus_base.py` | `main` truyền list câu hỏi; hai producer tra assignment bằng question_key | Chuyển caller ở T5.8, không sửa/chạy entrypoint này trong T5.2 |

Trong `src/`, `tests/`, `notebooks/`, caller import `assign_splits` từ `src.splits` được tìm thấy ở notebook build trên. `notebooks/00_data_audit.py` có hàm `question_key` riêng để audit; không sửa lại báo cáo cũ. [src/splits.py §assign_splits] [notebooks/01_build_corpus_base.py §main] [notebooks/01_build_corpus_base.py §build_qa_records] [notebooks/01_build_corpus_base.py §build_retrieval_training_records] [notebooks/00_data_audit.py §question_key]

Sau T5.2, API cũ truyền list chuỗi sẽ bị từ chối bằng lỗi input rõ ràng. Không giữ adapter trả assignment theo key vì cùng một key có thể là dòng test được giữ và dòng train bị loại. Notebook cũ vốn đã chờ T5.8 do D6; đây là thêm một điểm cần chuyển trong cùng caller.

## 3. Interface đã chốt

```python
# src/splits.py — interface đã triển khai theo thẻ
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

class SplitDecision(TypedDict):
    query_id: str
    source_dataset: Literal["germanquad", "germandpr"]
    original_split: Literal["train", "test"]
    source_row_index: int
    question_key: str
    source_split: Literal["train", "dev", "test"] | None

def question_key(q: str) -> str: ...

def assign_splits(
    gq_train_rows: Sequence[Mapping[str, object]],
    gq_test_rows: Sequence[Mapping[str, object]],
    gdpr_train_rows: Sequence[Mapping[str, object]],
    gdpr_test_rows: Sequence[Mapping[str, object]],
    *, dev_ratio: float = DEV_RATIO, seed: int = SEED,
) -> tuple[list[SplitDecision], dict]: ...
```

**Input:** bốn danh sách record theo đúng thứ tự dòng trong split gốc. GermanQuAD cần `id` là số nguyên không âm, không nhận bool, và `question` là chuỗi không trắng; GermanDPR cần `question`. Các field raw khác được phép có nhưng không dùng. Không yêu cầu Passage/QARecord đã xử lý. Cache hiện có `id:int32` và câu hỏi chuỗi. [docs/learning/00-project-analysis.md §5.1 Cache đã đọc bằng đường dẫn local, không gọi Hub]

**Định danh:** tạo `gq-{original_split}-{id}`; DPR dùng `gdpr-{original_split}-{source_row_index}`, index từ 0 **trước mọi lọc**. Không dùng split mới trong ID. Nếu sinh trùng `query_id` thì dừng bằng `ValueError`, không ghi đè quyết định. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.2 QA / gold record — chỉ dùng cho GermanQuAD] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.3 Retrieval training record — dùng cho GermanDPR]

**Output:** một `SplitDecision` cho **mọi dòng input**, kể cả dòng bị loại. `source_split=None` chỉ có nghĩa loại leakage trong quyết định nội bộ này; không phải cho phép null trong QARecord/TrainingRecord. Dòng bị loại không được ghi vào bất kỳ file train/dev/test nào. Thứ tự output: GQ train → GQ test → DPR train → DPR test, giữ thứ tự trong từng input; không sửa input tại chỗ.

**Normalization:** NFC → lowercase → strip; không gọi `re.sub` và không gộp khoảng trắng nội bộ. Xác nhận giữ lựa chọn này theo V1.1. Trên cache hiện tại, hai cách cho cùng số key duy nhất và số dòng bị loại, **nhưng khác tập dev** vì key là đầu vào của danh sách đã sort để sample. DPR dev có 928 dòng khi không gộp, so với 929 khi gộp; không được coi hai cách chia split là tương đương. [docs/learning/lessons/chang-05/buoi-12.md §16. Điểm cần xác minh] [the-task-T5.2.md §6. Mốc dữ liệu thật và lệnh đã kiểm]

**Chia split:** TestKeys là hợp key test gốc của cả hai dataset. Giữ nguyên mọi dòng test; dòng train có key thuộc TestKeys nhận None; pool còn lại sample dev từ **danh sách key duy nhất đã sort**. Giữ `int(len(pool) * dev_ratio)` và `random.Random(seed).sample` của code hiện có; làm tròn xuống là lựa chọn triển khai được ghi rõ, không phải câu trích spec. Default V1 là 0,10 và seed 42. Ratio khác chỉ dùng fixture; kiểm `0 <= dev_ratio <= 1`, hữu hạn, không bool; seed là int không bool. [src/splits.py §assign_splits] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.4 Cấu hình]

“Ổn định” là cùng bốn input có cùng thứ tự và cùng cấu hình cho cùng toàn bộ output. Đổi thứ tự DPR sẽ đổi ID theo định nghĩa index nguồn; không hứa ID giữ nguyên khi tự xáo dataset. Việc chọn dev theo key vẫn không phụ thuộc thứ tự duyệt key nhờ sort trước sample.

## 4. Hợp đồng report

Report là dict JSON-serializable, tên khóa sau là lựa chọn của thẻ để kiểm rõ key/dòng, chưa phải schema có sẵn:

- `seed`, `dev_ratio`: cấu hình đã dùng.
- `key_counts`: `test_union`, `train_dev_pool`, `train`, `dev`; mỗi số là số **key duy nhất hợp nhất hai dataset**.
- `datasets`: hai khóa `germanquad`, `germandpr`. Mỗi dataset có:
  - `input`: `train`, `test`, mỗi split `{rows, keys}` trước lọc.
  - `assigned`: `train`, `dev`, `test`, mỗi split `{rows, keys}` sau D1.
  - `removed`: `{rows, keys, query_ids}`; danh sách query_id đầy đủ và sort, không cắt còn mười ví dụ.

`rows` luôn đếm số dòng; `keys` đếm tập question_key duy nhất trong nhóm đó. Không cộng key của hai dataset để suy `key_counts`. Dòng bị loại vẫn xuất hiện trong list quyết định để truy vết, nhưng không thuộc `assigned`.

Theo từng dataset: `assigned.train.rows + assigned.dev.rows + assigned.test.rows + removed.rows == input.train.rows + input.test.rows`. Mọi dòng test gốc giữ test; mọi key trong quyết định **không bị loại** chỉ có một source_split. Report T5.2 không bịa số bị loại vì chunk mapping chưa chạy; T5.8 đối chiếu thêm report T5.5 khi ghi file DPR cuối cùng. Metadata revision/path/fingerprint lấy từ input thực tế ở caller build T5.8, không đoán trong hàm thuần này. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.7 Artifact]

## 5. Fixture và expected trước code

**Ví dụ minh họa sáu dòng**, không phải nội dung câu hỏi thật của các ID:

```python
gq_train = [
    {"id": 51870, "question": "Frage A"},
    {"id": 101, "question": "Frage B"},
    {"id": 102, "question": "Frage C"},
]
gq_test = [{"id": 40369, "question": "Frage D"}]
gdpr_train = [{"question": "Frage B"}]
gdpr_test = [{"question": "Frage A"}]
```

| query_id | source_dataset / original_split / source_row_index | Key | Expected default 0,10 / seed 42 |
|---|---|---|---|
| gq-train-51870 | germanquad / train / 0 | frage a | None, loại leakage |
| gq-train-101 | germanquad / train / 1 | frage b | train |
| gq-train-102 | germanquad / train / 2 | frage c | train |
| gq-test-40369 | germanquad / test / 0 | frage d | test |
| gdpr-train-0 | germandpr / train / 0 | frage b | train |
| gdpr-test-0 | germandpr / test / 0 | frage a | test |

Pool có hai key, `int(2 * 0.10)=0` nên fixture default không có dev. Thêm một ca **dev_ratio=1 chỉ cho fixture** để buộc B/C vào dev: `gq-train-101` và `gdpr-train-0` vẫn giữ ID gốc. Không dùng ratio này cho dữ liệu thật.

Report fixture default: `key_counts={test_union: 2, train_dev_pool: 2, train: 2, dev: 0}`. GQ input train 3 dòng/3 key, test 1/1; assigned train 2/2, dev 0/0, test 1/1; removed 1/1, query_ids `["gq-train-51870"]`. DPR input và assigned train/test đều 1/1, dev 0/0; removed 0/0, query_ids `[]`.

Các test chính đã triển khai trong `tests/test_splits.py` (43 ca tính cả parametrization):

| Test | Lỗi cần bắt |
|---|---|
| `test_cross_dataset_test_key_excludes_train_row_only` | Trả cùng quyết định theo key cho GQ train A và DPR test A; thiếu provenance hoặc mất dòng khỏi list quyết định |
| `test_dev_assignment_preserves_original_query_ids` | Đổi gq-train thành gq-dev hoặc đánh lại index DPR sau lọc |
| `test_same_dataset_test_overlap_excludes_dpr_train` | Thêm DPR train A ở index 1: phải loại `gdpr-train-1`, giữ `gdpr-test-0` |
| `test_duplicate_questions_keep_rows_and_share_split` | Thêm GQ id=103/question B: giữ cả hai dòng B; assigned train GQ 3 dòng/2 key, không dedup dòng |
| `test_question_key_preserves_internal_whitespace` | `"Frage  A"` khác `"Frage A"`; không gộp khoảng trắng |
| `test_question_key_normalizes_unicode_case_and_edges` | `" A\u0308  B \t"` → `"ä  b"`, giữ umlaut và hai khoảng trắng trong câu |
| `test_sorted_pool_sampling_is_repeatable_without_changing_global_rng` | Fixture 10 key `Frage 00`…`Frage 09`, seed 42, default ratio: đúng một key dev `frage 01`; lặp lại cùng input cho cùng output; đảo thứ tự GQ vẫn giữ quyết định theo key |
| `test_report_conserves_rows_and_lists_all_excluded_ids` | Báo key thay rows, cắt mất query_id bị loại, bỏ dòng có answers rỗng; kiểm cả JSON round-trip |
| `test_empty_inputs_and_no_train_pool` | Rỗng cả bốn input: list quyết định rỗng, report số 0; chỉ có test vẫn giữ toàn bộ test |
| `test_invalid_input_fails_without_mutating_rows` | List chuỗi cũ, question thiếu/rỗng, GQ id bool/âm/trùng, ratio/seed không hợp lệ; không silently drop dòng |

Các fixture không thêm logic QA gold hay DPR positive mapping vào split.

## 6. Mốc dữ liệu thật và lệnh đã kiểm

Đọc lại cache local ngày 2026-09-21, chỉ cột question (và id GQ khi kiểm định danh), không gọi loader Hub. Nguồn đường dẫn: [docs/learning/00-project-analysis.md §5.1 Cache đã đọc bằng đường dẫn local, không gọi Hub]. Snapshot Parquet GQ `a2f3a59f0be843fc305d0417d7292ef0b1a66884`; DPR `d0b867a49c45a44d172041351b5df73a87714372`.

| Dataset | Train gốc: rows / keys | Test gốc: rows / keys | Train+dev sau D1 | Rows loại | Keys loại |
|---|---:|---:|---:|---:|---:|
| GermanQuAD | 11.518 / 11.477 | 2.204 / 2.203 | 10.492 | 1.026 | 1.025 |
| GermanDPR | 9.275 / 9.254 | 1.025 / 1.025 | 9.274 | 1 | 1 |

GQ: **10.492 + 2.204 + 1.026 = 13.722**. DPR trước chunk mapping: **9.274 + 1.025 + 1 = 10.300**. Con số 1.026 của GQ là số dòng đếm trực tiếp, không phải phép cộng 1.025 key GQ với một key DPR. Đây là audit, chưa phải kết quả chạy API T5.2 mới hay nghiệm thu base.

Kiểm thêm ngày 2026-09-22 bằng **Python 3.13.3**, không collapse whitespace, `seed=42`, `dev_ratio=0.10`:

| Chỉ số key hợp nhất | Expected |
|---|---:|
| TestKeys | 3.228 |
| Pool train/dev | 10.452 |
| Dev | 1.045 |
| Train | 9.407 |

| Dataset | Rows loại | Dev | Train | Test gốc | Tổng gồm rows loại |
|---|---:|---:|---:|---:|---:|
| GermanQuAD | 1.026 | 1.047 | 9.445 | 2.204 | 13.722 |
| GermanDPR trước chunk mapping | 1 | 928 | 8.346 | 1.025 | 10.300 |

Đây là mốc đối chiếu API mới sau khi viết code, không chỉ đối chiếu tổng train+dev. Thành viên tập dev phụ thuộc key đã sort và thuật toán `random.sample` của runtime. Các số trên đã tính trên Python 3.13.3; khi đổi phiên bản Python, kể cả Colab, phải đối chiếu lại cả quyết định/ID và số đếm. Không coi cùng số dòng là đủ để chứng minh cùng tập dev.

Lệnh chỉ đọc dưới đã được chạy với `.venv`; dùng code ASCII để tránh PowerShell 5.1 đổi ký tự Unicode khi pipe. Khi copy từ Markdown, giữ `@'` và `'@` ở cột 0.

```powershell
@'
from pathlib import Path
import platform, random, unicodedata
import pyarrow.parquet as pq
hub = Path('C:/Users/PC/.cache/huggingface/hub')
revisions = {
    'germanquad': 'a2f3a59f0be843fc305d0417d7292ef0b1a66884',
    'germandpr': 'd0b867a49c45a44d172041351b5df73a87714372',
}
rows, keys = {}, {}
for dataset, revision in revisions.items():
    for split in ('train', 'test'):
        p = hub / ('datasets--deepset--' + dataset) / 'snapshots' / revision / 'plain_text' / split / '0000.parquet'
        cols = ['id', 'question'] if dataset == 'germanquad' else ['question']
        rows[dataset, split] = pq.read_table(p, columns=cols).to_pylist()
        keys[dataset, split] = [unicodedata.normalize('NFC', r['question']).lower().strip() for r in rows[dataset, split]]
        if dataset == 'germanquad':
            ids = [r['id'] for r in rows[dataset, split]]
            assert len(ids) == len(set(ids))
        print(dataset, split, len(rows[dataset, split]), len(set(keys[dataset, split])))
test_keys = set(keys['germanquad', 'test']) | set(keys['germandpr', 'test'])
for dataset in revisions:
    train_keys = keys[dataset, 'train']
    removed = sum(k in test_keys for k in train_keys)
    print(dataset, 'train_dev', len(train_keys) - removed, 'test', len(keys[dataset, 'test']),
          'removed_rows', removed, 'removed_keys', len(set(train_keys) & test_keys))
pool = (set(keys['germanquad', 'train']) | set(keys['germandpr', 'train'])) - test_keys
dev = set(random.Random(42).sample(sorted(pool), int(len(pool) * 0.10)))
train = pool - dev
print('Python', platform.python_version(), 'TestKeys', len(test_keys), 'pool', len(pool), 'dev_keys', len(dev), 'train_keys', len(train))
for dataset in revisions:
    ks = keys[dataset, 'train']
    print(dataset, 'dev', sum(k in dev for k in ks), 'train', sum(k in train for k in ks))
'@ | & '.venv/Scripts/python.exe' -B -
```

Kỳ vọng: bốn dòng counts khớp bảng; hai dòng D1 là `germanquad train_dev 10492 test 2204 removed_rows 1026 removed_keys 1025` và `germandpr train_dev 9274 test 1025 removed_rows 1 removed_keys 1`. Phần sample in `Python 3.13.3 TestKeys 3228 pool 10452 dev_keys 1045 train_keys 9407`, `germanquad dev 1047 train 9445`, `germandpr dev 928 train 8346`. Cache thiếu/khác revision thì dừng và ghi CẦN XÁC MINH, không tải bổ sung.

## 7. Thực thi sau khi duyệt thẻ

1. Viết fixture và test trong `tests/test_splits.py` trước khi sửa `src/splits.py`; chạy để quan sát RED do hợp đồng theo dòng/normalization hiện tại chưa đúng.
2. Sửa `src/splits.py` theo §3–4; giữ các test có hành vi đúng, không tạo RED giả. Không sửa notebook build hay spec.
3. Chạy GREEN và hồi quy. Đối chiếu API mới trên **bốn split raw đầy đủ**, trước mọi lọc/chunk; so counts §6 và bảo toàn từng dòng/ID. Chưa ghi file dataset.
4. Lưu log test và đối chiếu cache dưới `docs/learning/lessons/chang-05/`; kiểm phạm vi diff. Sau T5.2 mới chuyển T5.3; base tiếp tục chờ T5.8.

**Lệnh đã chạy trong nhánh thực thi:** test mới RED 43 failed trước sửa, GREEN 43 passed sau sửa; hồi quy 250 passed. Xem [log kiểm chứng](kiem-chung-T5.2.md) để phân biệt kết quả code và gate học tập.

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_splits.py -q -p no:cacheprovider --tb=line
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t52-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
```

Gate code: fixture bắt sai trước sửa; test pass; report có đủ key/dòng/ID bị loại; giữ test gốc và ID gốc; cùng key hợp lệ có cùng split; dữ liệu cache bảo toàn đúng số §6. Không cần score cao, cần đúng giao thức. Không xác nhận gate học của sinh viên hoặc base gate từ việc có thẻ.

## 8. Điểm cần duyệt hoặc xác minh

- Người dùng đã cho bắt đầu nhánh thực thi với interface theo dòng và mốc §6. Hai ô dưới dành cho sinh viên, không phải yêu cầu duyệt lại việc triển khai.
- [ ] Giải thích interface bốn list record → list `SplitDecision` và report, thay API theo key.
- [ ] Tự đối chiếu fixture/report và các hệ quả caller.
- Chuẩn hóa giữ đúng V1.1: không collapse whitespace; không mở lại quyết định này để hợp code cũ.
- CẦN XÁC MINH: caller ngoài `src/`, `tests/`, `notebooks/` — chủ project cung cấp hoặc người triển khai kiểm trước tích hợp.
- CẦN XÁC MINH: revision chuẩn được chọn khi build T5.8 — chủ project chốt; cache §6 là nguồn đã audit, không tự coi là manifest chính thức.
