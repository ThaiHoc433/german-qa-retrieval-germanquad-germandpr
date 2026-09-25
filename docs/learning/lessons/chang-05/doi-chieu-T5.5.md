---
tasks: [T5.5]
trang_thai: chưa làm
---

# Đối chiếu chỉ đọc trước producer T5.5

## 1. Nguồn và cách đo

Ngày 2026-09-22, Python 3.13.3. Đây là phép tính đối chiếu bằng API đã có, **không phải chạy producer T5.5**, không tạo training record/artifact trên đĩa. Snapshot:

- GermanQuAD: `C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/{train,test}/0000.parquet`.
- GermanDPR: `C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/{train,test}/0000.parquet`.

Schema đọc trực tiếp: question:string; answers:list[string]; positive_ctxs/negative_ctxs/hard_negative_ctxs là struct gồm ba list song song title/text/passage_id. Không dùng list dict hoặc IDs của artifact processed cũ. Record raw đầu có answers một chuỗi, positive title Gott, raw passage_id rỗng; không dùng raw passage_id này làm hash chunk V1.1.

D1 dùng assign_splits mặc định seed 42, ratio 0.10, question_key không gộp whitespace nội bộ. Chunk dùng 100 từ/stride 80; tìm mọi occurrence chính xác của mọi answer rồi hợp index chunk, dedup positive theo ID. HN chunk theo từng context, mask toàn bộ positive của chính query; bỏ group hết chunk. Chưa chọn positive/chunk theo epoch. [src/splits.py §assign_splits] [src/corpus.py §chunk_text] [src/corpus.py §find_span_chunks] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]

## 2. Mốc đối chiếu

Toàn raw trước D1:

| Split gốc | Dòng/positive context | HN context | HN chunk trước mask | Không map positive |
|---|---:|---:|---:|---:|
| train | 9.275 | 27.811 | 55.140 | 19 |
| test | 1.025 | 3.075 | 6.300 | 2 |
| Tổng | 10.300 | 30.886 | 61.440 | 21 |

Mọi dòng có một positive context; negative_ctxs có 0 context ở cả hai split. Phân phối số positive ID trên toàn raw: `{0:21,1:8996,2:1172,3:90,4:20,5:1}`. Các số 8.996 một positive / 1.283 nhiều positive / 21 không map khớp audit tham chiếu trong spec; không dùng chúng thay số sau D1.

| Split gán D1 | Dòng trước mapping | Unmappable | Dự kiến records | Positive ID occurrence | HN group xuất | HN chunk xuất |
|---|---:|---:|---:|---:|---:|---:|
| train | 8.346 | 17 | 8.329 | 9.440 | 24.972 | 49.442 |
| dev | 928 | 2 | 926 | 1.042 | 2.776 | 5.566 |
| test | 1.025 | 2 | 1.023 | 1.212 | 3.069 | 6.286 |
| Tổng xuất | 10.299 | 21 | 10.278 | 11.694 | 30.817 | 61.294 |

Một dòng bị loại D1 map được một positive; có 3 HN context/7 chunk, không được cộng vào output trên. Bảo toàn **10.278 + 21 + 1 = 10.300**. “Positive ID occurrence” là tổng độ dài list positive sau dedup trong từng query, không phải số ID duy nhất toàn dataset.

Danh sách unmappable đầy đủ sau D1 (sort theo chuỗi query_id):

```json
{
  "dev": ["gdpr-train-1640", "gdpr-train-224"],
  "test": ["gdpr-test-221", "gdpr-test-978"],
  "train": ["gdpr-train-121", "gdpr-train-1492", "gdpr-train-1571", "gdpr-train-2222", "gdpr-train-2427", "gdpr-train-3828", "gdpr-train-3938", "gdpr-train-4752", "gdpr-train-5216", "gdpr-train-5363", "gdpr-train-5441", "gdpr-train-5743", "gdpr-train-7619", "gdpr-train-7662", "gdpr-train-7780", "gdpr-train-8059", "gdpr-train-8657"]
}
```

Trên các record được giữ, mask HN lần lượt **7/0/1 chunk** ở train/dev/test, loại **3/0/0 group** vì hết chunk. Sau mask có **47.021/5.288/5.988 HN chunk occurrence vắng corpus**; đây là hợp lệ theo D6, không phải positive missing. Histogram dưới cũng chỉ lấy record/group được xuất:

| Split | Số positive/record → số record | Số group/record → số record |
|---|---|---|
| train | 1:7314, 2:935, 3:64, 4:16 | 1:5, 2:5, 3:8319 |
| dev | 1:825, 2:90, 3:8, 4:2, 5:1 | 1:1, 3:925 |
| test | 1:856, 2:147, 3:18, 4:2 | 3:1023 |

| Split | Số chunk/group → số group |
|---|---|
| train | 1:8057, 2:11174, 3:4046, 4:1581, 5:113, 10:1 |
| dev | 1:897, 2:1213, 3:440, 4:207, 5:19 |
| test | 1:937, 2:1313, 3:561, 4:250, 5:8 |

Record đầu `gdpr-train-0` gán dev: positive context 230 từ sinh 3 chunk, nhưng union chứa **index [2]**, positive ID **d20519f819d35708**, text dài 394 ký tự. HN có 3 group, độ dài 3/3/2. Bởi vậy con số 3 positive trong processed cũ không được mang sang expected mới.

Đọc mọi GQ context và thêm mọi DPR positive vào CorpusBuilder trong bộ nhớ: **7.520 ID, cả 7.520 đại diện source=germanquad**. Không có positive raw chunk được chọn khác text đại diện; mọi positive ID có trong map. Kiểm riêng mọi positive đại diện cũng chứa ít nhất một raw answer. Đây là đối chiếu nguồn hợp lệ trong bộ nhớ, chưa có title exclusions/manifest/acceptance của T5.6/T5.8; không tuyên bố artifact base hoàn tất. [src/corpus.py §CorpusBuilder]

## 3. Lệnh tái kiểm chỉ đọc

Chạy tại repo root; here-string phải ở cột 0. Chỉ đọc cache, không import notebook, không gọi Hub, không train, không ghi file. Các scope raw_* dùng để audit trước D1; scope train/dev/test là dự kiến output sau D1 và mapping. Histogram positive_distribution có bucket 0 để đối chiếu unmappable; report producer được đề nghị trong thẻ chỉ histogram positive trên output và có n_unmappable riêng.

```powershell
@'
from pathlib import Path
from collections import Counter
import json, platform
import pyarrow.parquet as pq
from src.corpus import CorpusBuilder, chunk_text, find_span_chunks, passage_id
from src.splits import assign_splits
hub = Path('C:/Users/PC/.cache/huggingface/hub')
inputs = []
sources = [('germanquad','a2f3a59f0be843fc305d0417d7292ef0b1a66884'),
           ('germandpr','d0b867a49c45a44d172041351b5df73a87714372')]
for ds,rev in sources:
    for sp in ('train','test'):
        inputs.append(pq.read_table(hub/('datasets--deepset--'+ds)/'snapshots'/rev/'plain_text'/sp/'0000.parquet').to_pylist())
decisions, _ = assign_splits(*inputs)
dec = {d['query_id']:d['source_split'] for d in decisions if d['source_dataset']=='germandpr'}
builder = CorpusBuilder()
for r in inputs[0]+inputs[1]:
    builder.add_text(r['context'],'germanquad')
for r in inputs[2]+inputs[3]:
    for t,title in zip(r['positive_ctxs']['text'],r['positive_ctxs']['title']):
        builder.add_text(t,'germandpr_positive',title=title)
assert len(builder.passages)==7520
assert Counter(p['source'] for p in builder.records())=={'germanquad':7520}
scopes = ['raw_train','raw_test','train','dev','test','excluded']
counts = {s:Counter() for s in scopes}
posdist = {s:Counter() for s in scopes}
groupdist = {s:Counter() for s in scopes}
chunkdist = {s:Counter() for s in scopes}
removed = {s:[] for s in ('train','dev','test')}
filtering = []
for original,rows in [('train',inputs[2]),('test',inputs[3])]:
    for i,r in enumerate(rows):
        qid = f'gdpr-{original}-{i}'
        split = dec[qid]
        positives, selected_indices = {}, []
        for t in r['positive_ctxs']['text']:
            chunks, indices = chunk_text(t), set()
            for a in r['answers']:
                assert isinstance(a,str) and a.strip()
                start = t.find(a)
                while start!=-1:
                    indices.update(find_span_chunks(chunks,start,start+len(a)))
                    start = t.find(a,start+1)
            selected_indices.append(sorted(indices))
            for j in sorted(indices):
                c = chunks[j]
                pid = passage_id(c.text)
                assert pid in builder.passages
                assert c.text == builder.passages[pid]['text']
                assert any(a in builder.passages[pid]['text'] for a in r['answers'])
                positives.setdefault(pid,c.text)
        groups, masked, empty_groups, raw_chunks = [], 0, 0, 0
        for context_index,t in enumerate(r['hard_negative_ctxs']['text']):
            chunks = chunk_text(t)
            raw_chunks += len(chunks)
            kept = [c for c in chunks if passage_id(c.text) not in positives]
            masked += len(chunks)-len(kept)
            if kept:
                groups.append(kept)
            else:
                empty_groups += 1
            if positives and split is not None and (len(kept)!=len(chunks) or not chunks):
                filtering.append((qid,context_index,len(chunks)-len(kept),not kept))
        if not positives and split is not None:
            removed[split].append(qid)
        for scope in ['raw_'+original,'excluded' if split is None else split]:
            c = counts[scope]
            c['rows'] += 1
            c['positive_contexts'] += len(r['positive_ctxs']['text'])
            c['negative_contexts'] += len(r['negative_ctxs']['text'])
            c['hard_negative_contexts_raw'] += len(r['hard_negative_ctxs']['text'])
            c['hard_negative_chunks_raw'] += raw_chunks
            posdist[scope][len(positives)] += 1
            if not positives:
                c['unmappable'] += 1
                continue
            c['mapped'] += 1
            c['positive_chunks'] += len(positives)
            c['hard_negative_groups'] += len(groups)
            c['hard_negative_chunks'] += sum(len(g) for g in groups)
            c['masked_chunks'] += masked
            c['empty_groups'] += empty_groups
            c['hard_negative_absent_from_corpus'] += sum(passage_id(x.text) not in builder.passages for g in groups for x in g)
            groupdist[scope][len(groups)] += 1
            chunkdist[scope].update(len(g) for g in groups)
        if qid=='gdpr-train-0':
            assert split=='dev' and selected_indices==[[2]]
            assert list(positives)==['d20519f819d35708']
            assert [len(t) for t in positives.values()]==[394]
            assert [len(g) for g in groups]==[3,3,2]
expected = {'train':(8346,17,8329,9440,24972,49442,7,3),
            'dev':(928,2,926,1042,2776,5566,0,0),
            'test':(1025,2,1023,1212,3069,6286,1,0)}
fields = ['rows','unmappable','mapped','positive_chunks','hard_negative_groups','hard_negative_chunks','masked_chunks','empty_groups']
for scope in expected:
    assert tuple(counts[scope][f] for f in fields)==expected[scope]
assert counts['excluded']['rows']==counts['excluded']['mapped']==1
assert posdist['raw_train']+posdist['raw_test']=={0:21,1:8996,2:1172,3:90,4:20,5:1}
assert sum(counts[s]['mapped']+counts[s]['unmappable'] for s in expected)+counts['excluded']['rows']==10300
for s in scopes:
    print(s,json.dumps({'counts':dict(counts[s]),'positive_distribution':dict(sorted(posdist[s].items())),
          'groups_per_record':dict(sorted(groupdist[s].items())),
          'chunks_per_group':dict(sorted(chunkdist[s].items()))},sort_keys=True))
print('unmappable_after_D1',json.dumps({s:sorted(ids) for s,ids in removed.items()},sort_keys=True))
print('filtering',filtering)
print('Python',platform.python_version(),'corpus_memory',len(builder.passages))
'@ | & '.venv/Scripts/python.exe' -B -
```

## 4. Giới hạn và chờ duyệt

Các số đã tính bằng thuật toán chỉ đọc, cần đối chiếu lại khi producer thật được viết. Thẻ T5.5 chốt riêng chữ ký, text nguồn/đại diện và report; không dùng baseline này để tự đánh dấu bài học/gate đã hoàn thành. Không dùng dataset test để tune tham số, chỉ kiểm đúng quy tắc cố định và báo số dòng.

Đã trích khối lệnh §3 từ chính file này và chạy: exit 0, mọi assertion pass, Python 3.13.3, tất cả counts/histogram/list ID khớp các bảng. Các khối PowerShell Buổi 14 được trích, bỏ thụt lề danh sách và chạy: exit 0; fixture cho positive indices [0,2], group đầu còn chunk 1 và group Paris giữ nguyên. Hai file src/retrieval_records.py và tests/test_retrieval_records.py chưa tồn tại tại điểm dừng duyệt.
