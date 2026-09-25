---
tasks: [T5.0, T5.1]
buoi: 11
trang_thai: chưa làm
---

# Kiểm chứng nhánh thực thi T5.0/T5.1

Ngày lập báo cáo: 2026-09-21. Đây là log kiểm thử code do người triển khai thực hiện sau khi người dùng giao quyết định chuyển sang code; không xác nhận tiến độ hoặc gate học tập của sinh viên.

## 1. Phạm vi đã triển khai

| File | Thay đổi |
|---|---|
| `src/schemas.py` | Thêm `validate_passage_record`, `validate_retrieval_training_record`; QA nhận map tùy chọn `passage_texts` để kiểm slice thật; Prediction bắt buộc hai k nguyên dương, reader_k không vượt retrieval_k. |
| `tests/test_schemas.py` | Ban đầu thêm 62 ca kiểm thử; fixture Prediction có 20/5. Bao phủ malformed input, hash/normalization, provenance/slice, lists text/ID song song, no-answer và giới hạn k. Sau review thêm hai ca ở §5. |
| `src/corpus.py` | `add_text` nhận title, lưu đủ chín field PassageRecord; thay đồng thời metadata của đại diện theo ưu tiên nguồn. Từ chối nguồn ngoài ba nguồn cho phép trước khi thêm passage. |
| `tests/test_corpus.py` | Tạo mới 33 ca về normalize, hash, chunk biên/overlap, title, provenance, thay đại diện và nguồn corpus. |
| `tests/test_evaluation.py` | Chỉ thêm `retrieval_k=20`, `reader_k=5` vào fixture `pred`; chưa đổi bucket hoặc metric cũ. |

Phạm vi và signature theo [the-task-T5.0-T5.1.md §1. Phạm vi và thứ tự] [the-task-T5.0-T5.1.md §2. Files và interface]. Hợp đồng dữ liệu theo [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1. Schema chung] và [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý].

## 2. Lệnh và kết quả thực tế

Chạy tại gốc repo bằng PowerShell và interpreter `.venv/Scripts/python.exe`. `-B` tránh ghi bytecode; `-p no:cacheprovider` tránh cache pytest trong repo. Không train, download, build corpus hay sửa dữ liệu.

**Schema: viết test trước, rồi sửa production.**

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_schemas.py -q -p no:cacheprovider --tb=line
```

- RED trước sửa `src/schemas.py`: **64 failed, 19 passed in 0.23s**, exit 1. Các lỗi nằm ở validator Passage/Training chưa tồn tại, QA chưa nhận `passage_texts`, Prediction chưa nhận/bắt buộc hai k. Hai test valid cũ cũng thất bại sau khi fixture được thêm k; không hạ yêu cầu mới để giữ schema cũ.
- GREEN sau sửa: **83 passed in 0.17s**, exit 0.
- Test collect thành công; RED không phải lỗi import hay lỗi gõ fixture. Các test hợp đồng cũ vốn đúng được giữ xanh, không tạo lỗi giả.

**Corpus: viết file test mới trước, rồi sửa CorpusBuilder.**

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_corpus.py -q -p no:cacheprovider --tb=line
```

RED: **23 failed, 10 passed in 0.09s**, exit 1. Thiếu ba field provenance; `add_text` chưa nhận title; nguồn hard-negative/unknown chưa bị từ chối. Mười test normalize/chunk/hành vi hiện hữu đã qua.

Sau sửa, kiểm chung hai module:

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_schemas.py tests/test_corpus.py -q -p no:cacheprovider --tb=line
```

GREEN: **116 passed in 0.24s**, exit 0 (83 schema + 33 corpus).

**Hồi quy toàn bộ:**

```powershell
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t50-t51-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
```

Lượt sandbox đầu gặp `PermissionError: [WinError 5]` ở thư mục tạm ngoài repo; không coi lượt này là GREEN. Sau khi được cấp quyền chạy lại cùng lệnh với một thư mục tạm mới: **205 passed in 0.66s**, exit 0. Tổng số ca = 110 test nền + 62 schema mới + 33 corpus mới. Kết quả này chứng nhận phạm vi hồi quy hiện có, không chứng nhận toàn bộ V1.1.

`git diff --check` trả exit 0, không có lỗi whitespace. Thư mục tạm của test nằm ngoài repo; không tạo `.pytest-tmp/` mới trong repo.

## 3. Những hành vi đã được kiểm tra

- Hash fixture `Berlin` = `fb38a93fd89e1a5f`, `Ulm` = `37bfc8ba5ec56136`; normalization vẫn giữ umlaut/ß. [tests/test_corpus.py §test_normalization_preserves_german_letters_and_raw_input]
- Fixture 201 từ có ba khoảng `[0,499)`, `[400,899)`, `[800,1004)`; `w085` nằm trong hai chunk, giữ offset và slice gốc. [tests/test_corpus.py §test_201_word_offsets_and_answer_in_overlap]
- Thay đại diện cập nhật cả title, char_len, parent hash và offset; cùng nguồn giữ occurrence đầu. Kiểm cả trường hợp text khác độ dài do whitespace và chunk trùng nhau nhưng parent khác nhau. [tests/test_corpus.py §test_dedup_replacement_updates_char_length_when_raw_whitespace_differs] [tests/test_corpus.py §test_replacement_changes_parent_identity_not_just_chunk_identity] [tests/test_corpus.py §test_equal_priority_preserves_first_occurrence_and_all_its_metadata]
- Passage/QA kiểm slice thật khi được cung cấp parent/map; không tự sửa offset. Khi không có text đối chiếu, chỉ xác nhận hợp đồng nội bộ của record. [src/schemas.py §validate_passage_record] [src/schemas.py §validate_qa_record]
- Training kiểm text/ID song song, hash đúng; không đòi ID hard-negative nằm trong corpus. Kiểm membership positive, chọn answer-bearing chunk và mask gold_set thuộc các task sau. [src/schemas.py §validate_retrieval_training_record] [the-task-T5.0-T5.1.md §2. Files và interface]

## 4. Hệ quả và việc còn lại

1. Evaluation gọi validator Prediction nên từ chối predictions cũ thiếu `retrieval_k`/`reader_k`. Đây là hệ quả P1 đã ghi trong thẻ. Kiểm 20/5 theo cấu hình V1, khớp manifest và tính nhất quán cả run vẫn thuộc T6.3; validator record không suy mode từ `run_id`. [src/evaluation.py §_index_predictions] [src/schemas.py §validate_prediction_record]
2. `notebooks/01_build_corpus_base.py` còn gọi `add_text(t, "germandpr_hard_negative")`; lời gọi này sẽ nhận `ValueError` sau T5.1. Entrypoint đó chờ chuyển ở T5.8, chưa chạy lại. [notebooks/01_build_corpus_base.py §build_retrieval_training_records]
3. CorpusBuilder giữ đại diện theo thứ tự input trong cùng nguồn. Chuỗi distractor Wikipedia theo khóa P4, mọi title trước dedup, QA collision/remap, DPR grouping, split và acceptance vẫn thuộc T5.2–T5.8. Không dùng builder hiện tại để tuyên bố đã hoàn tất thứ tự distractor P4. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]
4. T5.0/T5.1 có bằng chứng test; base V1.1 chưa build. T6.4 thật vẫn chờ T5.8 và T6.1–T6.3. Các bài học và checkbox sinh viên vẫn chưa làm. [the-task-T5.0-T5.1.md §1. Phạm vi và thứ tự]
5. CẦN XÁC MINH: caller ngoài `src/`, `tests/`, `notebooks/` — chủ project cung cấp hoặc người triển khai kiểm trước khi tích hợp.

## 5. Bổ sung sau review: thiếu ID cuối

Ngày 2026-09-21, thêm hai test cho trường hợp còn nguyên một cặp text/ID đúng hash, nhưng thiếu ID của text cuối:

- `test_training_rejects_missing_last_positive_id_with_valid_prefix`: positives `["Berlin", "Ulm"]`, IDs ban đầu `[BERLIN_ID, ULM_ID]`; kiểm record hợp lệ rồi bỏ ID cuối.
- `test_training_rejects_missing_last_group_id_with_valid_prefix`: một group texts `["Ulm", "Berlin"]`, IDs ban đầu `[ULM_ID, BERLIN_ID]`; kiểm hợp lệ rồi bỏ ID cuối.

Cả hai đi qua API `validate_retrieval_training_record`, không mock kết quả trả về. Code production đã xử lý đúng, nên lần này chỉ bổ sung test, không sửa validator để tạo RED giả. [tests/test_schemas.py §test_training_rejects_missing_last_positive_id_with_valid_prefix] [tests/test_schemas.py §test_training_rejects_missing_last_group_id_with_valid_prefix]

Chạy schema với lệnh `& '.venv/Scripts/python.exe' -B -m pytest tests/test_schemas.py -q -p no:cacheprovider`: **85 passed in 0.20s**.

Để kiểm hai test có bắt đúng lỗi sống sót, chạy mutation chỉ trong bộ nhớ của một process riêng. AST bỏ đúng một nhánh kiểm độ dài; không ghi file production. Lệnh dưới đã chạy, code chỉ chứa ASCII để pipe qua PowerShell 5.1:

```powershell
@'
import ast
from pathlib import Path
import pytest
from src import schemas

class DropLengthGuard(ast.NodeTransformer):
    removed = 0
    def visit_If(self, node):
        if ast.unparse(node.test) == 'len(texts) != len(ids)':
            self.removed += 1
            return None
        return self.generic_visit(node)

source = Path('src/schemas.py').read_text(encoding='utf-8')
mutation = DropLengthGuard()
tree = mutation.visit(ast.parse(source))
assert mutation.removed == 1
namespace = {'__name__': 'length_guard_mutant'}
exec(compile(ast.fix_missing_locations(tree), '<length_guard_mutant>', 'exec'), namespace)
schemas._text_id_errors = namespace['_text_id_errors']
raise SystemExit(pytest.main([
    'tests/test_schemas.py::test_training_rejects_missing_last_positive_id_with_valid_prefix',
    'tests/test_schemas.py::test_training_rejects_missing_last_group_id_with_valid_prefix',
    '-q', '-p', 'no:cacheprovider', '--tb=short',
]))
'@ | & '.venv/Scripts/python.exe' -B -
```

Kỳ vọng và kết quả thực tế: **2 failed in 0.13s**, exit 1; cả hai assertion nhận `errors=[]` trên mutant. Đây là bằng chứng mutation cho hai test mới, không phải tái tạo RED lịch sử ở §2 và không phải tái chạy toàn bộ 26 mutation người dùng đã kiểm.

Sau đó chạy lại bộ test ở process sạch với đúng lệnh hồi quy §2, thư mục tạm mới ngoài repo: **207 passed in 0.74s**, exit 0. Tổng mới = 110 nền + 64 schema bổ sung + 33 corpus. Các số 83/116/205 ở §2 giữ nguyên vì là log của lượt triển khai ban đầu.
