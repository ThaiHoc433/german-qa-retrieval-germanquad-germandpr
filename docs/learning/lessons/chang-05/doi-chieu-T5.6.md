---
tasks: [T5.6]
trang_thai: chưa làm
---

# Đối chiếu chỉ đọc trước producer T5.6

## 1. Nguồn và phạm vi

Mốc chuẩn bị thẻ ngày 2026-09-24, tái kiểm ngày 2026-09-25 bằng Python 3.13.3. Chỉ đọc bốn Parquet và dùng `CorpusBuilder` hiện có trong bộ nhớ; **chưa chạy API build_base_corpus đề nghị**, chưa tạo artifact base/manifest. [src/corpus.py §CorpusBuilder]

| Dataset | Revision cache | File |
|---|---|---|
| GermanQuAD | `a2f3a59f0be843fc305d0417d7292ef0b1a66884` | `plain_text/train/0000.parquet`, `plain_text/test/0000.parquet` |
| GermanDPR | `d0b867a49c45a44d172041351b5df73a87714372` | `plain_text/train/0000.parquet`, `plain_text/test/0000.parquet` |

Gốc mỗi snapshot là `C:/Users/PC/.cache/huggingface/hub/datasets--deepset--{dataset}/snapshots/{revision}/`. GQ đọc `id`, `question`, `context`; DPR đọc `question`, `positive_ctxs`. DPR raw có struct positive_ctxs với ba list song song `title`, `text`, `passage_id`; script kiểm độ dài, không dùng ID raw thay hash chunk. Nguồn trực tiếp và phép kiểm schema nằm trong §3.

Nguồn D6: mọi context GQ và mọi positive context DPR của train/test, không lọc theo D1/answer mapping. Title GQ là dòng đầu context; title DPR là field raw. Thu title trước dedup theo P4: NFC → `_` thành space → lowercase → gộp whitespace → strip. Công thức P4 trong script chỉ phục vụ đối chiếu, không phải module production mới. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.1 Passage record (Corpus)] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §Nhật ký quyết định V1.1]

## 2. Mốc thật

| Nguồn/split | Context occurrence | Chunk trước dedup | Title key duy nhất | Context rỗng | Title thiếu theo P4 |
|---|---:|---:|---:|---:|---:|
| GQ train | 11.518 | 29.589 | 378 | 0 | 0 |
| GQ test | 2.204 | 10.249 | 233 | 0 | 0 |
| DPR positive train | 9.275 | 23.733 | 523 | 0 | 0 |
| DPR positive test | 1.025 | 2.683 | 137 | 0 | 0 |

Tổng **24.022 context occurrence**, **66.254 chunk occurrence**, **7.520 passage_id duy nhất**; cả 7.520 đại diện source=germanquad. Title union trước dedup **565**; title từ passage đại diện chỉ **388**, bỏ sót **177**. Không cộng bốn số title riêng để tính union. [doi-chieu-T5.6.md §3. Lệnh tái kiểm]

T5.2 loại 1.026 dòng GQ khỏi QA; context của chúng có **131 title key**, tất cả vẫn nằm trong exclusions. D1 lọc record QA/training, không lọc nguồn corpus/title. [src/splits.py §assign_splits] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]

**Metadata title cần ghi nhận:** 802 positive context DPR train và 79 test có title bằng toàn context, đều chứa newline: tổng **881**. Ví dụ train index 2637 có title mở đầu `Buddhismus\nDer Buddhismus…`; test index 82 mở đầu `Eritrea\nEritrea…`. Đây là field title thật, không phải title thiếu. T5.6 giữ raw title và chuẩn hóa P4; không tự cắt dòng đầu theo quy tắc dành cho GQ. CẦN XÁC MINH: có cần audit/amendment metadata riêng không — chủ project quyết định; frozen V1.1 không đổi. [doi-chieu-T5.6.md §3. Lệnh tái kiểm]

Hai hash dưới chỉ kiểm object trong bộ nhớ, **không phải SHA-256 file artifact**:

- Title: list sort, JSON `ensure_ascii=False`, `separators=(',', ':')`, UTF-8, không newline → `c82921ab6a524979391398884b6144929445116bd4e1eb2b6422b6c7cb906113`.
- Corpus: records sort theo passage_id, JSON như trên và `sort_keys=True` → `2995167bcab5159adcfea76de4c70f13247dba9a65bfe3e227df1a7f839c3967`.

Các số là expected cho bản triển khai cuối trên đúng snapshot/thứ tự này; không hard-code vào producer. Test/acceptance T5.6/T5.8 vẫn phải chạy API thật và tính fingerprint byte artifact riêng.

## 3. Lệnh tái kiểm

Chạy từ repo root. Hai dấu `@'` và `'@` của PowerShell phải ở cột 0, kể cả khi copy từ Markdown nguồn. Lệnh chỉ đọc cache, không import notebook, không gọi Hub, không ghi data/results. Mỗi dòng số in theo thứ tự: context, chunk, title key, context rỗng, title thiếu.

```powershell
@'
from pathlib import Path
from collections import Counter
import hashlib, json, platform, re, unicodedata
import pyarrow.parquet as pq
from src.corpus import CorpusBuilder
from src.splits import assign_splits

hub = Path('C:/Users/PC/.cache/huggingface/hub')
sources = [('germanquad', 'a2f3a59f0be843fc305d0417d7292ef0b1a66884'),
           ('germandpr', 'd0b867a49c45a44d172041351b5df73a87714372')]
inputs = []
for ds, rev in sources:
    cols = ['id', 'question', 'context'] if ds == 'germanquad' else ['question', 'positive_ctxs']
    for split in ('train', 'test'):
        path = hub/('datasets--deepset--'+ds)/'snapshots'/rev/'plain_text'/split/'0000.parquet'
        inputs.append(pq.read_table(path, columns=cols).to_pylist())

def p4(title):
    if title is None:
        return None
    assert isinstance(title, str)
    key = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', title).replace('_', ' ').lower()).strip()
    return key or None

def gq_title(text):
    lines = text.splitlines()
    return lines[0] if lines else None

builder, all_titles, measured = CorpusBuilder(), set(), []
labels = ['gq-train', 'gq-test', 'dpr-train', 'dpr-test']
title_equals_text = Counter()
for pos, rows in enumerate(inputs):
    source = 'germanquad' if pos < 2 else 'germandpr_positive'
    counts, title_keys = Counter(), set()
    for row in rows:
        if pos < 2:
            contexts = [(row['context'], gq_title(row['context']))]
        else:
            ctx = row['positive_ctxs']
            assert len(ctx['text']) == len(ctx['title']) == len(ctx['passage_id'])
            contexts = zip(ctx['text'], ctx['title'])
        for text, title in contexts:
            key = p4(title)
            counts['contexts'] += 1
            if key is None:
                counts['missing'] += 1
            else:
                title_keys.add(key)
                all_titles.add(key)
            chunks = builder.add_text(text, source, title=title)
            counts['chunks'] += len(chunks)
            counts['empty'] += int(not chunks)
            if pos >= 2 and title == text:
                assert '\n' in title
                title_equals_text[labels[pos]] += 1
    values = (counts['contexts'], counts['chunks'], len(title_keys), counts['empty'], counts['missing'])
    measured.append(values)
    print(labels[pos], *values)
assert measured == [(11518,29589,378,0,0), (2204,10249,233,0,0),
                    (9275,23733,523,0,0), (1025,2683,137,0,0)]
records = sorted(builder.records(), key=lambda r: r['passage_id'])
representative_titles = {p4(r['title']) for r in records} - {None}
source_counts = Counter(r['source'] for r in records)
assert source_counts == {'germanquad': 7520}
assert representative_titles <= all_titles
print('totals', sum(v[0] for v in measured), sum(v[1] for v in measured), len(records))
print('title_union representative lost', len(all_titles), len(representative_titles), len(all_titles-representative_titles))
assert (len(all_titles), len(representative_titles), len(all_titles-representative_titles)) == (565,388,177)
decisions, _ = assign_splits(*inputs)
removed_gq = [d for d in decisions if d['source_dataset']=='germanquad' and d['source_split'] is None]
removed_titles = {p4(gq_title(inputs[0][d['source_row_index']]['context'])) for d in removed_gq} - {None}
assert len(removed_gq) == 1026 and len(removed_titles) == 131 and removed_titles <= all_titles
print('removed_gq_rows title_keys_in_exclusions', len(removed_gq), len(removed_titles))
assert title_equals_text == {'dpr-train':802, 'dpr-test':79}
print('dpr_title_equals_context', dict(title_equals_text))
for pos, idx, prefix in [(2,2637,'Buddhismus\n'), (3,82,'Eritrea\n')]:
    assert inputs[pos][idx]['positive_ctxs']['title'][0].startswith(prefix)
    print('title_example', labels[pos], idx, repr(prefix))
title_hash = hashlib.sha256(json.dumps(sorted(all_titles), ensure_ascii=False, separators=(',',':')).encode('utf-8')).hexdigest()
corpus_hash = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()
assert title_hash == 'c82921ab6a524979391398884b6144929445116bd4e1eb2b6422b6c7cb906113'
assert corpus_hash == '2995167bcab5159adcfea76de4c70f13247dba9a65bfe3e227df1a7f839c3967'
print('title_object_sha256', title_hash)
print('corpus_object_sha256', corpus_hash)
print('python', platform.python_version(), 'all_checks_passed')
'@ | & '.venv/Scripts/python.exe' -B -
```

Kỳ vọng: bốn dòng đầu khớp bảng §2; `totals 24022 66254 7520`; `title_union representative lost 565 388 177`; hai hash khớp và dòng cuối `python 3.13.3 all_checks_passed`. Nếu khác: kiểm revision/cache, thứ tự nguồn, công thức P4 và phiên bản code trước khi kết luận lỗi producer chưa được viết. Không đổi expected để che sai khác.

Đã trích chính here-string trên từ file Markdown và chạy lại ngày 2026-09-25: exit 0, tất cả assert pass, stdout khớp bảng và hai hash. Kết quả chỉ xác nhận phép đối chiếu trước implementation; trạng thái học viên vẫn chưa làm.
