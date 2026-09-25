---
tasks: [T5.3, T5.4]
trang_thai_hoc_vien: chưa làm
---

# Kiểm chứng T5.3/T5.4 — producer QA và remap

## 1. Quyết định và phạm vi

Ngày 2026-09-22, chọn phương án người dùng cho phép: gộp T5.3/T5.4 trong một lượt, không tạo `OffsetRemapRequired` chuyển tiếp. Interface, tie-break chọn offset nhỏ hơn khi khoảng cách tương đối bằng nhau và 11 field remap report được chốt trong thẻ trước triển khai. Đây là lựa chọn triển khai, không sửa spec V1.1 frozen. [the-task-T5.3.md §2. Files, caller và interface] [the-task-T5.3.md §4. Ranh giới T5.4 và report]

Hai file code/test mới là `src/qa_records.py` và `tests/test_qa_records.py`. Producer chỉ tạo object trong bộ nhớ; không ghi dataset, không gọi Hub, không train, không chạy notebook build. Caller cũ còn chờ T5.8; chưa tạo `data/processed/v1.1/`. Trạng thái 19 bài học vẫn chưa làm. [src/qa_records.py §build_qa_records]

## 2. RED, GREEN và hồi quy

1. Viết test trước module: lần đầu collection báo `ModuleNotFoundError: No module named 'src.qa_records'`; đây là bằng chứng thiếu module, chưa phải assertion hành vi.
2. Tạo khung interface trả `records={train:[],dev:[],test:[]}`, coverage `{}`, remap `[]`. Không triển khai mapping/validation. Thu thập đủ **79 test**, kết quả **79 failed in 0.24s**: thiếu record/report và input sai không bị từ chối. Trước triển khai, kiểm tay bổ sung đã sửa hai offset fixture tự tạo (Äpfel bắt đầu 17, candidate Berlin thứ hai là 128); không đổi expected dựa trên output producer.
3. Viết implementation: cùng 79 test **passed in 0.22s**. Hồi quy đầu: **329 passed in 2.92s** = 250 nền + 79 mới.
4. Review độc lập không thấy lỗi production; chỉ ra fixture collision có slice đúng cần thêm nhiều occurrence để phát hiện fallback thừa. Bổ sung một ca: source `alpha Berlin omega  Berlin end`, đại diện `alpha Berlin omega Berlin end`, Berlin/start=6 → vẫn [6,12), remap report rỗng. Targeted **80 passed in 0.23s**. Mutation trong bộ nhớ ép mọi collision khác text chạy fallback bị ca mới bắt: **KILLED**. Không sửa file production để tạo lỗi.
5. Hồi quy cuối sau bổ sung: **330 passed in 1.09s** = 250 nền + 80 mới, exit 0. `.pytest-tmp/` không tồn tại; basetemp ở thư mục tạm ngoài repo. Đây là kiểm module hiện có, chưa chứng nhận toàn bộ pipeline V1.1.

Lệnh đã chạy:

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_qa_records.py -q -p no:cacheprovider --tb=short
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t53-t54-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
```

Test phủ đủ references, span trong mọi chunk overlap, gold rỗng, quyết định D1 theo từng dòng, provenance của chunk QA khác parent đại diện, kiểm input sai, 0/1/>1 occurrence, occurrence chồng nhau, tie-break, vị trí tương đối trong document, report và input bất biến. Nhiều occurrence trên dữ liệu thật không tự kích hoạt fallback. [tests/test_qa_records.py §test_collision_remap_zero_one_many_tie_and_overlapping] [tests/test_qa_records.py §test_remap_uses_relative_position_in_source_document_not_chunk] [tests/test_qa_records.py §test_collision_with_valid_direct_slice_does_not_require_remap]

## 3. Cache thật và hiệu chỉnh mốc D1

Đọc Parquet local, Python 3.13.3; GQ revision `a2f3a59f0be843fc305d0417d7292ef0b1a66884`, DPR revision `d0b867a49c45a44d172041351b5df73a87714372`. D1 dùng `assign_splits` mặc định seed=42/ratio=0.10, không gộp whitespace nội bộ question_key. Corpus kiểm ở đây chỉ gồm toàn bộ context GermanQuAD train+test trong bộ nhớ, **chưa phải base D6 đầy đủ**. [src/splits.py §assign_splits] [src/corpus.py §CorpusBuilder]

| Split | QARecord | Gold rỗng | Coverage | Evidence sau dedup exact | References giữ nguyên |
|---|---:|---:|---:|---:|---:|
| train | 9.445 | 158 | 98,327157% | 10.057 | 9.445 |
| dev | 1.047 | 16 | 98,471824% | 1.109 | 1.047 |
| test | 2.204 | 8 | 99,637024% | 5.078 | 6.536 |

Tổng **12.696 QARecord**, **182 gold rỗng**, cộng **1.026 dòng loại D1 = 13.722 raw**. Tất cả references so nguyên danh sách với raw; mọi QARecord qua validator với text đại diện, không chỉ kiểm counts. Các số coverage vượt 98% từng split, nhưng không thay nghiệm thu file/manifest T5.8. [src/qa_records.py §build_qa_records] [src/schemas.py §validate_qa_record]

**Hiệu chỉnh mốc người dùng:** trong 1.026 dòng loại D1 có **2** dòng không có chunk chứa trọn answer, không phải 0: `gq-train-67129`, `gq-train-70207`. Đây là kết quả tính lại trực tiếp bằng char-offset trên raw. Chúng không thuộc output producer hoặc mẫu số coverage sau D1. Trên đúng cache này, tổng trước D1 là 182+2=184; không quy số 184 cho luật primary chunk. [notebooks/01_build_corpus_base.py §build_qa_records] [data/processed/chunk_coverage_report.json §uncovered_rows]

Toàn bộ context GQ tạo **7.520 passage_id**, **0 ID có nhiều raw text khác nhau**, **0 cặp answer/chunk cần remap** (đã đếm trực tiếp, không suy từ report rỗng). Producer trả remap report `[]`. Do GQ có priority cao nhất, nguồn DPR/Wikipedia thêm sau không thay đại diện GQ; kết quả trên snapshot này không chứng minh nhánh fallback hoạt động — bằng chứng fallback là fixture. [src/corpus.py §SOURCE_PRIORITY] [src/qa_records.py §_remap_offset]

Lệnh tái kiểm chỉ đọc dưới đây dùng cache, không ghi file và không mạng. Here-string mở/đóng phải ở cột 0; code dùng ASCII để pipe qua Windows PowerShell 5.1. Digest chỉ cho object kết quả trong bộ nhớ, không phải fingerprint artifact đã build.

Đã trích khối lệnh dưới từ chính file này và chạy: exit 0, mọi assertion pass, digest **`963114ddfab3408d4b5010d7e950c98afeef7d7b8a45ec13163d36083541e885`**. Kết quả lặp lại bằng nhau; hash input/decisions/corpus trước và sau producer không đổi. Không tự đổi expected khi lệch; kiểm revision, Python, D1, đại diện và offset.

```powershell
@'
from pathlib import Path
from collections import defaultdict
import hashlib, json, platform
import pyarrow.parquet as pq
from src.corpus import CorpusBuilder, chunk_text, passage_id, find_span_chunks
from src.splits import assign_splits
from src.qa_records import build_qa_records
from src.schemas import validate_qa_record
hub = Path('C:/Users/PC/.cache/huggingface/hub')
sources = [('germanquad', 'a2f3a59f0be843fc305d0417d7292ef0b1a66884'),
           ('germandpr', 'd0b867a49c45a44d172041351b5df73a87714372')]
inputs = []
for dataset, revision in sources:
    for split in ('train', 'test'):
        path = hub / ('datasets--deepset--' + dataset) / 'snapshots' / revision / 'plain_text' / split / '0000.parquet'
        columns = ['id','question','context','answers'] if dataset == 'germanquad' else ['question']
        inputs.append(pq.read_table(path, columns=columns).to_pylist())
decisions, _ = assign_splits(*inputs)
builder, variants = CorpusBuilder(), defaultdict(set)
for row in inputs[0] + inputs[1]:
    for c in builder.add_text(row['context'], 'germanquad'):
        variants[passage_id(c.text)].add(c.text)
passages = {p['passage_id']:p for p in builder.records()}
assert len(passages) == 7520 and not any(len(v)>1 for v in variants.values())
def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()
before = digest([inputs, decisions, passages])
result = build_qa_records(inputs[0], inputs[1], decisions, passages)
assert digest([inputs, decisions, passages]) == before
texts = {pid:p['text'] for pid,p in passages.items()}
raw_by_id = {f'gq-{split}-{r["id"]}':r for split,rows in [('train',inputs[0]),('test',inputs[1])] for r in rows}
expected = {'train':(9445,158,10057,9445), 'dev':(1047,16,1109,1047), 'test':(2204,8,5078,6536)}
for split, records in result['records'].items():
    report = result['coverage_report']['files'][f'qa_germanquad_{split}.jsonl']
    counts = (len(records), report['n_empty_gold'], sum(len(r['gold_evidences']) for r in records), sum(len(r['reference_answers']) for r in records))
    assert counts == expected[split] and report['n_records'] == len(records)
    assert report['empty_gold_query_ids'] == sorted(r['query_id'] for r in records if not r['gold_evidences'])
    assert report['coverage_rate'] == (len(records)-report['n_empty_gold'])/len(records)
    assert [r['query_id'] for r in records] == [d['query_id'] for d in decisions if d['source_dataset']=='germanquad' and d['source_split']==split]
    for rec in records:
        assert not validate_qa_record(rec, passage_texts=texts)
        assert rec['reference_answers'] == raw_by_id[rec['query_id']]['answers']['text']
    print(split, counts, format(report['coverage_rate'], '.8f'))
assert result['coverage_report']['input_rows'] == 13722
assert result['coverage_report']['excluded_leakage_rows'] == 1026
assert sum(len(r) for r in result['records'].values()) == 12696
assert result['ambiguous_offset_remap_report'] == []
assert build_qa_records(inputs[0], inputs[1], decisions, passages) == result
by_decision = {d['query_id']:d for d in decisions}
needed_remap, removed_uncovered = 0, []
for qid,row in raw_by_id.items():
    chunks, covered = chunk_text(row['context']), False
    for answer,start in zip(row['answers']['text'], row['answers']['answer_start']):
        for i in find_span_chunks(chunks, start, start+len(answer)):
            covered = True
            c = chunks[i]
            local = start-c.char_start
            needed_remap += texts[passage_id(c.text)][local:local+len(answer)] != answer
    if by_decision[qid]['source_split'] is None and not covered:
        removed_uncovered.append(qid)
assert needed_remap == 0
assert sorted(removed_uncovered) == ['gq-train-67129','gq-train-70207']
print('Python', platform.python_version(), 'passages', len(passages), 'raw_text_collisions', 0, 'needs_remap', needed_remap)
print('excluded_uncovered', sorted(removed_uncovered))
print('result_sha256', digest(result))
'@ | & '.venv/Scripts/python.exe' -B -
```

## 4. Giới hạn và bước tiếp theo

- T5.3/T5.4 chỉ tạo QARecord/report trong bộ nhớ. Không thay spec, roadmap, catalog, notebook hoặc artifact cũ. Build base, đổi caller và ghi các report lên đĩa thuộc T5.8.
- Thứ tự đại diện GQ cùng priority phụ thuộc thứ tự input theo CorpusBuilder; caller phải chuẩn bị corpus cuối trước producer. Không thêm nguồn/đổi đại diện sau khi đã tạo evidence.
- T5.5 cần thẻ riêng cho DPR positive union, hard-negative groups và mapping report; chưa triển khai trong lượt này.
- CẦN XÁC MINH: caller ngoài src/tests/notebooks, revision và coverage/report file artifact — người triển khai kiểm khi tích hợp. Chưa đánh dấu gate học viên đã qua.

Kiểm phạm vi cuối: git không có diff ở file tracked; chỉ thêm hai file code/test chưa commit kể trên cùng cập nhật thẻ, Buổi 13 và log học liệu. Không stage file nào. Hai khối PowerShell của Buổi 13 được trích từ Markdown, bỏ thụt lề danh sách rồi chạy, đều exit 0; fixture in đúng `gq-train-1 ['Berlin'] 19` và `[6, 19] 19`. Đủ 19 bài, mỗi bài 16 mục, trạng thái chưa làm; không có lệnh `rg`. `git diff --check` sạch.

## 5. Commit sau review

Người dùng đã kiểm độc lập producer trên toàn raw và các mutation, đề nghị duyệt/commit riêng. Ngày 2026-09-22, chạy lại baseline trước commit: **330 passed in 2.49s**. Commit **`1102836`** trên nhánh `feature/v1.1-data-pipeline`, đúng hai file `src/qa_records.py` và `tests/test_qa_records.py` (563 dòng thêm); không kèm học liệu hoặc file untracked khác, không push. Staged diff đã kiểm sạch trước commit.

Đồng ý giữ nhánh raise phòng thủ khi text nguồn/đại diện giống hệt nhưng slice sai: với raw validation và find_span_chunks đúng, điều kiện này không tới được. Không thêm test giả chỉ để ép kích hoạt nhánh bất khả đạt. Kết quả mutation do người dùng báo là bằng chứng review bổ sung, không được ghi thành số test root đã tự chạy.
