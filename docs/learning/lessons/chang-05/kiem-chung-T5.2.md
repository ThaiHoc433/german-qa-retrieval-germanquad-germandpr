---
tasks: [T5.2]
buoi: 12
trang_thai: chưa làm
---

# Kiểm chứng nhánh thực thi T5.2

Ngày 2026-09-22. Log của người triển khai; không xác nhận sinh viên đã làm bài hay qua gate. Phạm vi theo [the-task-T5.2.md §3. Interface đã chốt] và [the-task-T5.2.md §4. Hợp đồng report].

## 1. Mốc trước triển khai

- Đã xác nhận giữ normalization NFC → lowercase → strip, **không gộp khoảng trắng nội bộ**. Kiểm hai cách trên cache cho cùng 3.228 TestKeys, 10.452 pool keys và số dòng loại, nhưng tập dev khác; DPR dev là 928 khi không gộp, 929 khi gộp. [the-task-T5.2.md §6. Mốc dữ liệu thật và lệnh đã kiểm]
- Baseline trước khi tách commit: **207 passed in 0.65s**, dùng `.venv` và basetemp ngoài repo.
- Tạo nhánh `feature/v1.1-data-pipeline`; commit T5.0/T5.1 **`1a0285e`**, đúng bảy file: `.gitignore`, `requirements.txt`, `src/schemas.py`, `src/corpus.py`, `tests/test_schemas.py`, `tests/test_corpus.py`, `tests/test_evaluation.py`. Không đưa học liệu, roadmap-preview hoặc file untracked khác vào commit. Git yêu cầu quyền ghi `.git` và đã được cấp; không push.

## 2. RED rồi GREEN

Viết `tests/test_splits.py` trước khi sửa `src/splits.py`. Chạy:

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_splits.py -q -p no:cacheprovider --tb=line
```

RED: **43 failed in 0.11s**, exit 1. Test được collect đầy đủ, không lỗi import. API cũ đưa dict raw vào normalization dành cho chuỗi; các ca normalization trực tiếp cho thấy mất whitespace; ca input rỗng nhận dict assignment thay vì list quyết định. Các test lỗi input cũng không nhận thông báo field tương ứng vì code cũ chưa có validation theo interface mới.

Sau sửa `src/splits.py`, cùng lệnh: **43 passed in 0.08s**, exit 0. Không đổi expected để hợp code cũ. File test có fixture cụ thể cho test chéo dataset, trùng trong cùng dataset, ID trước lọc, duplicates, mẫu dev, bảo toàn dòng và đủ danh sách query_id loại. [tests/test_splits.py §test_cross_dataset_test_key_excludes_train_row_only] [tests/test_splits.py §test_dev_assignment_preserves_original_query_ids] [tests/test_splits.py §test_report_conserves_rows_and_lists_all_excluded_ids]

Hồi quy:

```powershell
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t52-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
```

Kết quả: **250 passed in 0.66s**, exit 0 = 207 nền trước T5.2 + 43 ca mới. Lệnh chạy với quyền truy cập thư mục tạm ngoài repo; không tạo `.pytest-tmp/` trong repo. Đây là test của các module hiện có, chưa chứng nhận toàn bộ pipeline V1.1.

## 3. Đối chiếu từng dòng trên dữ liệu thật

Đọc bốn Parquet raw ở đúng snapshot ghi trong thẻ, chỉ cột question và id GQ. Python **3.13.3**, ratio **0.10**, seed **42**. Không gọi loader Hub, không lọc/chunk trước khi gán ID, không ghi dataset/report ra `data/`.

| Dataset | Train | Dev | Test gốc | Loại leakage | Tổng decisions |
|---|---:|---:|---:|---:|---:|
| GermanQuAD | 9.445 | 1.047 | 2.204 | 1.026 | 13.722 |
| GermanDPR trước chunk mapping | 8.346 | 928 | 1.025 | 1 | 10.300 |

`key_counts`: `test_union=3228`, `train_dev_pool=10452`, `train=9407`, `dev=1045`. Tổng **24.022 quyết định**, mỗi dòng raw có đúng một quyết định và query_id duy nhất. Mọi quyết định được so với nguồn dòng/ID/index/key và D1; các danh sách loại đầy đủ, report đếm đúng key/dòng từng assigned split. Không key nào trong các dòng được giữ nhận hai split khác nhau. Chạy lại cùng input cho cùng toàn bộ decisions/report; input không bị thay đổi.

SHA-256 của **list decisions trong bộ nhớ**: `02743a997bea35c53e3632093459f2ae9ccda798975a3f3f440e48dab9d2ef3f`. Byte hash là `json.dumps(decisions, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")`, không thêm newline. Đây là dấu kiểm đối chiếu tập và thứ tự quyết định, **không phải fingerprint của artifact đã build**.

Lệnh tái kiểm chỉ đọc, giữ delimiters here-string ở cột 0; mã ASCII để dùng với PowerShell 5.1:

```powershell
@'
from pathlib import Path
from collections import Counter
import hashlib, json, platform
import pyarrow.parquet as pq
from src.splits import assign_splits
hub = Path('C:/Users/PC/.cache/huggingface/hub')
sources = [('germanquad', 'a2f3a59f0be843fc305d0417d7292ef0b1a66884'),
           ('germandpr', 'd0b867a49c45a44d172041351b5df73a87714372')]
inputs = []
for dataset, revision in sources:
    for split in ('train', 'test'):
        path = hub / ('datasets--deepset--' + dataset) / 'snapshots' / revision / 'plain_text' / split / '0000.parquet'
        cols = ['id', 'question'] if dataset == 'germanquad' else ['question']
        inputs.append(pq.read_table(path, columns=cols).to_pylist())
before = json.dumps(inputs, ensure_ascii=False)
decisions, report = assign_splits(*inputs)
assert before == json.dumps(inputs, ensure_ascii=False)
assert len(decisions) == len({d['query_id'] for d in decisions}) == 24022
assert assign_splits(*inputs) == (decisions, report)
assert report['key_counts'] == {'test_union':3228, 'train_dev_pool':10452, 'train':9407, 'dev':1045}
expected = {'germanquad': {'train':9445, 'dev':1047, 'test':2204, None:1026},
            'germandpr': {'train':8346, 'dev':928, 'test':1025, None:1}}
for dataset, _ in sources:
    counts = dict(Counter(d['source_split'] for d in decisions if d['source_dataset'] == dataset))
    assert counts == expected[dataset]
    print(dataset, counts)
blob = json.dumps(decisions, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
digest = hashlib.sha256(blob).hexdigest()
assert digest == '02743a997bea35c53e3632093459f2ae9ccda798975a3f3f440e48dab9d2ef3f'
print('Python', platform.python_version(), 'decisions', len(decisions), 'sha256', digest)
'@ | & '.venv/Scripts/python.exe' -B -
```

Kỳ vọng: hai bảng counts khớp ở trên và digest khớp. Khi khác, kiểm snapshot, thứ tự raw, normalization và phiên bản Python; không tự đổi expected để thông qua. Các số DPR này chưa trừ record không map được positive chunk ở T5.5.

## 4. Review, ranh giới và bước tiếp theo

- Reviewer độc lập đã đọc implementation/tests so với thẻ và spec, chạy lại test split **43 passed**, không phát hiện lỗi chặn hoặc thiếu test liên quan. Kết luận này bổ sung cho kiểm cache/hồi quy, không thay nghiệm thu artifact.
- `src/splits.py` và `tests/test_splits.py` là phần code T5.2; thẻ và Buổi 12 được cập nhật để không còn yêu cầu chạy API cũ. [src/splits.py §assign_splits] [tests/test_splits.py §test_report_counts_keys_and_rows_per_dataset]
- Notebook build cũ vẫn truyền list câu hỏi và tra assignment theo key, nên chưa chạy được với API mới; chuyển caller ở T5.8 theo phạm vi đã duyệt. Không sửa notebook, spec, roadmap hoặc dữ liệu cũ trong T5.2. [notebooks/01_build_corpus_base.py §main] [the-task-T5.2.md §2. Files và hệ quả tương thích]
- Chưa build base V1.1, chưa train/download. T5.3 là bước kế tiếp để tạo QA evidence theo provenance; T6.4 thật vẫn đợi T5.8. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi]
- CẦN XÁC MINH: caller ngoài `src/`, `tests/`, `notebooks/`, revision chuẩn và runtime được dùng khi build — chủ project/người triển khai chốt trước tích hợp. Không tự đánh dấu bài học hoặc gate sinh viên đã hoàn thành.

## 5. Commit sau review

Sau khi người dùng kiểm chứng và cho chuyển sang thẻ T5.3, chạy lại baseline: **250 passed in 0.76s**. Commit **`0926212`** trên `feature/v1.1-data-pipeline` chỉ gồm `src/splits.py` và `tests/test_splits.py`; không stage `docs/learning/` hoặc file untracked khác, không push. `git diff --cached --check` sạch trước commit.

Lưu ý phạm vi ratio: thẻ giữ phép `int(len(pool) * dev_ratio)` của code cũ. Với ratio fixture khác 0,10, phép nhân float có thể xuống dưới số nguyên mong đợi (ví dụ `int(100 * 0.29)` là 28). Không sửa cách chia ở lượt này; cấu hình V1 đã kiểm vẫn là 0,10/seed 42 trên Python 3.13.3.
