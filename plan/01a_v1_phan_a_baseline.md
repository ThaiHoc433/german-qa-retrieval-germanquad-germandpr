# V1 — Phần A: Baseline (giai đoạn 0–10)

Cập nhật: 2026-10-14. Trạng thái: kế hoạch đề xuất; chưa đối chiếu tiến độ thực tế.

[Mục lục](README.md) · [V1 phần B — Cải tiến](01b_v1_phan_b_cai_tien.md) · [V2 — Nghiên cứu](02_v2_nghien_cuu.md)

## Phạm vi

Xây hệ thống QA trích xuất tiếng Đức: tìm passage trong corpus xác định, dự đoán span và trả đáp án kèm bằng chứng.

```text
Câu hỏi → BM25 → Top-20 → Lấy top-5 → Reader
                                       ↓
                  Đáp án + passage + offsets + thời gian
```

Phần A gồm dữ liệu, corpus, BM25, fine-tune reader, đánh giá, API, React và deploy, chạy được độc lập như một baseline. Giai đoạn 6 về dense là nhánh tùy chọn từ pic_plan, làm ở đây hoặc ở giai đoạn 12 (cùng V1). Hybrid, reranker và mining bổ sung thuộc V1 phần B (giai đoạn 11–18); router thuộc V2.

## Giai đoạn 0 — Xác định bài toán và dựng khung

### Mục tiêu và đầu vào

Chốt hợp đồng dữ liệu và tổ chức lõi Python dùng chung giữa notebook, benchmark, API. Đầu vào là yêu cầu QA tiếng Đức và pic_plan.

### Công việc

- Chốt corpus, ngôn ngữ, nhiệm vụ extractive QA và phạm vi câu hỏi.
- Request: question, retrieval_top_k, reader_top_k.
- Response: answer, passage_id, text nguồn, offsets, scores, timings, versions.
- Thống nhất offsets ký tự dạng `[start, end)` trên văn bản gốc; ghi rõ đơn vị và chuyển đúng khi frontend dùng UTF-16.
- Colab dành cho huấn luyện; ứng dụng phục vụ suy luận bằng artifacts đã xuất.
- Tạo experiment_id và cấu hình thí nghiệm; lưu seed, model/data revision.

### Cấu trúc đề xuất

```text
data/{raw,processed}/
src/germanqa/{data,retrieval,reader,evaluation}/
src/germanqa/pipeline.py
configs/
notebooks/
artifacts/
results/
tests/
backend/
frontend/
deployment/
```

### Kiến thức và kỹ thuật

Python package, interfaces, YAML/JSON, Git, môi trường phụ thuộc, seed và logging. Notebook điều phối/trình bày; logic dùng lại nằm trong package.

### Đầu ra và nghiệm thu

Khung dự án, schema và cấu hình baseline. Notebook/CLI/API dùng được cùng lõi; mỗi kết quả có configuration và version đi kèm.

## Giai đoạn 1 — Thu thập và phân tích hai dataset

### Mục tiêu

Hiểu cấu trúc, chất lượng, quan hệ GermanQuAD/GermanDPR và vai trò reader/retriever. Lưu revision/checksum bản tải thực tế.

| Thuộc tính | GermanQuAD | GermanDPR |
|---|---|---|
| Nhiệm vụ chính | Extractive QA | Passage retrieval |
| Train theo dataset card | 11.518 câu hỏi | 9.275 mẫu |
| Test theo dataset card | 2.204 câu hỏi | 1.025 mẫu |
| Nhãn | Answer text và answer_start | Positive context và hard negatives |
| Đặc điểm | Test có nhiều đáp án tham chiếu | Một positive và ba hard negatives/mẫu theo mô tả nguồn |
| Vai trò trong V1 | Reader và đánh giá QA | Phân tích retrieval, nguồn passage theo protocol |

GermanDPR được phát triển từ GermanQuAD; không coi là hai tập kiểm tra độc lập. Số câu hỏi không bằng số passage duy nhất. Nguồn: [GermanQuAD](https://huggingface.co/datasets/deepset/germanquad), [GermanDPR](https://huggingface.co/datasets/deepset/germandpr).

### Công việc EDA

- Kiểm tra schema, kiểu dữ liệu, ID thiếu, question/context rỗng.
- Đếm queries, unique passages, bài viết, số queries/context.
- Phân phối độ dài question/context/answer theo ký tự và tokenizer dự kiến.
- Đếm số answer references; giữ toàn bộ khi đánh giá.
- Đo overlap câu hỏi, context và nguồn giữa dataset/split.
- Kiểm tra `context[start:start+len(answer)] == answer`.
- Đọc mẫu ngắn/dài, answer dài và mẫu bất thường.
- Đếm context vượt giới hạn token reader.

### Phương pháp

Thống kê mô tả, histogram, phân vị, hash trùng chính xác, giao tập ID/hash. Dò gần trùng hỗ trợ kiểm tra; không tự gộp vì giống ngữ nghĩa.

### Đầu ra và nghiệm thu

EDA report, thống kê theo split, danh sách lỗi và quyết định xử lý. Đối chiếu số thực tế với dataset card; ghi giải thích sai khác phiên bản/đọc dữ liệu.

## Giai đoạn 2 — Tiền xử lý, chia tập và corpus

### Mục tiêu và schema

Tạo dữ liệu nhất quán, giữ answer offsets và provenance.

| Loại | Trường chính |
|---|---|
| Corpus | passage_id, original_text, title, document_id, source |
| QA | question_id, question, passage_id, answers[{text,start}] |
| Qrels | question_id → relevant_passage_ids đã biết |
| Manifest | revisions, split IDs, corpus version, preprocessing config |

### Xử lý văn bản

1. Lưu raw bất biến.
2. Giữ original_text cho reader; retrieval_text riêng cho viết thường/tokenization.
3. Kiểm tra encoding, trường thiếu và offsets.
4. Loại passage trùng chính xác; giữ các câu hỏi khác nhau và mọi provenance.
5. Gán passage_id ổn định, tạo ID nội bộ nếu ID nguồn rỗng.
6. Sửa văn bản ảnh hưởng vị trí phải có mapping và kiểm tra lại. Cách ly mẫu không sửa đáng tin cậy, không sửa im lặng.
7. Bản chuẩn hóa để dò gần trùng không thay thế raw text. Không gộp các bản raw/offset khác nhau mà thiếu mapping.

### Chia tập

- Giữ test gốc cho đánh giá cuối.
- Tách train thành train/dev theo context hoặc article group. 90/10 là ứng viên, chốt sau khi xem phân phối.
- Nếu thiếu article ID, dùng provenance/context grouping và công bố giới hạn.
- Kiểm tra overlap giữa cả hai dataset nếu sử dụng chung để học.
- Bảo vệ tập test; xử lý mẫu huấn luyện xung đột theo quy tắc ghi trước, không tùy ý xóa câu khó khỏi test.
- Lưu seed và split IDs; không chia lại mỗi lần chạy.

### Corpus

Đề xuất corpus giới hạn: contexts GermanQuAD, cộng positive/hard-negative contexts GermanDPR nếu chọn; hợp nhất và deduplicate. Công bố nguồn, số passage, split nguồn. Đây chưa phải toàn Wikipedia.

Query tìm trên toàn corpus, không chỉ bốn context của từng mẫu DPR. Negative gắn với query; passage negative cho A có thể đúng cho B.

Văn bản test có thể nằm trong corpus theo protocol tìm kiếm, nhưng câu hỏi/nhãn test không dùng học hoặc tuning. Qrels không được đưa vào input suy luận.

### Passage dài

V1 dùng passage gốc cho retrieval và sliding windows ở reader. Nếu đổi chunk retrieval, tạo lại chunk IDs, offsets về nguồn, qrels chứa đủ answer, corpus version và index. Không lẫn reader window với corpus passage.

### Đầu ra

```text
corpus.jsonl
qa_train.jsonl / qa_dev.jsonl / qa_test.jsonl
qrels_dev.json / qrels_test.json
split_manifest.json
data_quality_report.json
```

### Nghiệm thu

Spans hợp lệ hoặc có lý do xử lý rõ; mọi qrels ID có trong corpus; không có overlap trái protocol; script tạo lại được dữ liệu.

## Giai đoạn 3 — BM25 và metrics retrieval

### Luồng

```text
Corpus → Tokenization → BM25 index
Question → Cùng tokenization → BM25 → Top-k IDs/ranks/scores
```

### Thuật toán và công việc

- BM25 Okapi: TF, IDF, bão hòa tần suất và chuẩn hóa độ dài.
- Tokenizer Unicode tiếng Đức; viết thường trên retrieval_text, giữ umlaut/ß. Stemming/stopword là biến thể đo riêng trên dev.
- Tuning tập nhỏ `k1`, `b`; lưu cấu hình.
- Interface `retrieve(question, top_k)`; xử lý đồng điểm ổn định.
- Corpus nhỏ có thể dùng in-memory; tăng quy mô phải đo lại RAM/latency và cân nhắc inverted index.

| Metric | Ý nghĩa |
|---|---|
| Hit@k | Có ít nhất một relevant đã biết trong top-k |
| Recall@k | Số relevant tìm được / tổng relevant đã biết |
| MRR@k | Trung bình 1/rank relevant đầu tiên; 0 nếu vắng khỏi top-k |
| Latency | Thời gian truy hồi theo điều kiện đo đã ghi |

Đo k = 1, 5, 10, 20. Một gold passage/query thì Hit và Recall trùng nhau. Qrels có thể chưa đầy đủ; không đồng nhất unjudged với chắc chắn sai.

### Đầu ra và kiểm tra

Index, config, retrieval predictions và dev benchmark. Tính tay metrics trên tập nhỏ; kiểm tra query rỗng, đồng điểm, top-k vượt corpus. Không dùng test chọn tham số.

## Giai đoạn 4 — Reader và fine-tuning

### Luồng

```text
Question + context → Tokenizer/sliding windows → GBERT hoặc GELECTRA
                  → Start/end logits → Span hợp lệ → Original text
```

### 4.1. Kiểm tra luồng

Dùng checkpoint QA sẵn có để kiểm tra loading, inference và offsets. Checkpoint đã học toàn GermanQuAD train có thể đã thấy dev tự tách; ghi rõ lịch sử, không coi đó là đánh giá độc lập.

### 4.2. Tạo features và labels

- Tokenize question/context bằng tokenizer tương thích.
- Dùng offset_mapping đổi ký tự ↔ token.
- Sliding windows có overlap; lưu window → query/passage mapping.
- Chỉ gán span khi toàn answer nằm trong window.
- Window không chứa đủ answer dùng nhãn theo kiến trúc, ví dụ CLS; ghi chính sách.
- Mask question, padding, special tokens khỏi span thông thường.
- Chọn max length, stride, max answer length theo dữ liệu và model.

Tham khảo quy trình nhãn span: [Transformers QA](https://huggingface.co/docs/transformers/tasks/question_answering).

### 4.3. Huấn luyện

| Thành phần | Lựa chọn khởi đầu |
|---|---|
| Backbone | Một GBERT/GELECTRA phù hợp tài nguyên |
| Head | Start/end token prediction |
| Loss | Trung bình cross-entropy start và end |
| Optimizer | AdamW |
| Chọn checkpoint | Dev EM/F1 |
| Bộ nhớ | Batch phù hợp; gradient accumulation/mixed precision nếu đã kiểm tra |
| Logging | Loss, metrics, learning rate, seed, thời gian và versions |

Đánh giá tự fine-tune bắt đầu từ backbone tiền huấn luyện, chỉ học QA train đã chia. Chính sách nhiều annotations phải cố định. Theo dõi ổn định số khi dùng mixed precision.

### 4.4. Hậu xử lý

- Start/end cùng window, đúng context, end không trước start.
- Giới hạn answer length theo cấu hình dev.
- Gộp windows, loại span trùng khi cần.
- Cắt original_text bằng offsets, không ghép token đã chuẩn hóa để hiển thị.
- Score giữa windows/passages là quy tắc cần kiểm chứng, không mặc nhiên là xác suất.

### Đầu ra và nghiệm thu

Checkpoint/tokenizer, training config, predictions và gold-context EM/F1. Nạp lại được; context dài trả span hợp lệ. Chấm nhiều gold answers theo quy tắc đã công bố, thường lấy điểm tốt nhất trên references mỗi query.

## Giai đoạn 5 — Ghép pipeline BM25 + reader

### Công việc

1. BM25 lấy top-20.
2. Chọn top-5 đầu tiên.
3. Reader xử lý passage/windows.
4. Gộp và chọn span cuối bằng score reader thống nhất.
5. Trả bằng chứng, offsets, scores, timings và versions.

Top-20 hỗ trợ benchmark/phân tích; chỉ top-5 tham gia QA ở cấu hình này.

### Response đề xuất

```text
question, answer, passage_id, passage_text
answer_start, answer_end
retrieval_score, reader_score
retrieval_time, reader_time, total_time
model_version, corpus_version, experiment_id
```

### Kỹ thuật và lưu ý

- Có thể chọn span hợp lệ với tổng start/end logits cao nhất làm baseline; đánh giá heuristic này trên dev.
- Không cộng tùy ý BM25 score và reader score.
- Kiểm tra answer là substring đúng tại offsets; query/corpus nhỏ và input lỗi được xử lý.
- Không đưa gold context hoặc qrels vào inference end-to-end.
- Reader answerable có thể trả span cho câu ngoài corpus. No-answer cần dữ liệu/ngưỡng/đánh giá riêng nếu thêm.

### Đầu ra và nghiệm thu

CLI/pipeline Python và dev end-to-end predictions; timings/trace từng bước; kết quả tái tạo được từ cùng artifacts.

## Giai đoạn 6 — Dense đối chiếu, tùy chọn

### Vai trò và luồng

Giữ từ pic_plan; có thể làm ở đây hoặc để giai đoạn 12 (phần B). Nếu hoàn thành, tái sử dụng ở giai đoạn 12.

```text
Corpus → Context encoder → Embeddings → FAISS
Query → Query encoder → Vector search → Top-k → Cùng reader
```

### Phương pháp và kiểm tra

- Encoder/tokenizer/pooling đúng checkpoint; similarity đúng model.
- Batch encode, lưu vector-to-passage mapping.
- Flat index chính xác làm mốc trước khi thử approximate search.
- Đối chiếu FAISS với tính similarity trực tiếp trên mẫu nhỏ.
- Đánh giá retrieval/end-to-end cùng protocol BM25.

### Đầu ra

Dense retriever, index/embeddings, manifest và bảng so sánh. GermanDPR phục vụ retriever; không mặc định đủ offset labels để thay GermanQuAD huấn luyện reader.

## Giai đoạn 7 — Đánh giá và phân tích lỗi

| Tầng | Input | Metrics |
|---|---|---|
| Retrieval | Query + corpus | Hit/Recall@k, MRR@k |
| Reader | Query + gold context | EM, token F1 |
| End-to-end | Query + corpus | EM, token F1, latency |

Chọn cấu hình bằng dev, đóng băng rồi chạy test. Công bố normalization EM/F1, nhiều references, truncation/windows. Không tự áp dụng quy tắc bỏ article tiếng Anh cho tiếng Đức mà thiếu giải thích.

### Phân loại các đáp án cuối sai

```text
Gold evidence vắng ở top-20 → Retrieval failure
Có ở top-20, vắng ở top-5  → Selection failure
Có ở top-5, đáp án sai     → Reader/extraction hoặc answer selection failure
```

Chỉ phân loại khi đáp án cuối sai và đủ nhãn bằng chứng. Chuỗi answer xuất hiện không tự chứng minh passage trả lời query. Kiểm tra thủ công bằng chứng đúng ngoài qrels.

### Đầu ra và nghiệm thu

Metrics, query-level predictions, bảng lỗi, config và phần cứng. Kết quả tái chạy được, gắn corpus/model versions và điều kiện đo.

## Giai đoạn 8 — Backend FastAPI

| Endpoint | Chức năng |
|---|---|
| POST /api/v1/qa | Answer và bằng chứng |
| POST /api/v1/retrieve | Candidates |
| GET /api/v1/experiments | Benchmark đã lưu |
| GET /health/ready | Tài nguyên đã nạp và sẵn sàng |

### Công việc/kỹ thuật

- Pydantic schema; validation độ dài query, top-k, lựa chọn model.
- Nạp model/index qua lifespan và dùng lại: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/).
- Training chạy riêng; endpoint experiments chỉ đọc kết quả.
- Error/timing logs, concurrency và timeout.
- Một worker trước, đo RAM rồi mới tăng. Async không tự tăng tốc model.

### Nghiệm thu

API và CLI nhất quán cùng cấu hình; readiness đúng; restart nạp artifacts thành công.

## Giai đoạn 9 — React

### Công việc/kỹ thuật

- React + TypeScript: nhập query, submit, loading, kết quả/lỗi.
- Hiển thị answer, source passage, span highlight, candidates và latency.
- Gọi API bất đồng bộ; hủy hoặc bỏ qua response cũ.
- Unicode offsets đúng; render corpus như text an toàn.
- Trang benchmark đọc số đo đã lưu, kèm versions.
- Câu hỏi mới thiếu gold không tự chấm EM/F1.
- Chỉ hiển thị lựa chọn dense khi backend/artifacts hỗ trợ.

### Nghiệm thu

UI nối API thật, highlight đúng, lỗi rõ; response query cũ không ghi đè query mới.

## Giai đoạn 10 — Docker và triển khai

```text
Người dùng → HTTPS/reverse proxy
                   ├── React build
                   └── FastAPI → Model + tokenizer + corpus + index
Colab → Artifacts đúng phiên bản → Máy chủ
```

### Công việc/kỹ thuật

- Dockerfile, Compose, environment config, artifact paths/persistent storage.
- Reverse proxy/HTTPS, health checks và logs.
- Kiểm tra artifact versions/checksum/schema khi nạp.
- Đo RAM/VRAM, disk, startup, latency, nhiều requests.
- Tách phép đo Colab và máy chủ.
- Hướng dẫn tải/tạo artifacts, khởi động và kiểm tra.

### Nghiệm thu

Website chạy trên môi trường chọn; restart nạp đúng tài nguyên; có hướng dẫn tái triển khai. Cấu hình máy được chọn dựa trên đo thực tế.

## Phụ thuộc và mốc

```text
0 → 1 → 2 → [3 BM25 và 4 Reader] → 5 Pipeline
                                     ↓
                         6 Dense (tùy chọn) → 7 Đánh giá
                                               ↓
                                          8 API → 9 UI → 10 Deploy
```

Giai đoạn 3/4 có thể làm song song khi schema đã ổn định. Trước giai đoạn 11 phải lưu pipeline BM25 + reader và kết quả baseline.

## Checklist V1 phần A (giai đoạn 0–10)

- [ ] Raw data, revisions, EDA.
- [ ] Corpus/split/qrels/offsets đã kiểm tra.
- [ ] BM25 và metrics có kiểm tra mẫu nhỏ.
- [ ] Reader xử lý windows, trả span trong original_text.
- [ ] Pipeline không dùng gold lúc suy luận.
- [ ] Dev/test đúng protocol, có phân tích lỗi.
- [ ] API/UI hiển thị bằng chứng đúng.
- [ ] Deploy restart được, có hướng dẫn.
- [ ] Kết quả gắn config/version/điều kiện đo.

## Nguồn nền

- [GermanQuAD/GermanDPR — MRQA 2021](https://aclanthology.org/2021.mrqa-1.4/).
- [GermanQuAD](https://huggingface.co/datasets/deepset/germanquad).
- [GermanDPR](https://huggingface.co/datasets/deepset/germandpr).
- [Transformers QA](https://huggingface.co/docs/transformers/tasks/question_answering).
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/).

Checkpoint và phiên bản thư viện cụ thể sẽ được kiểm tra/khóa khi cài đặt; tài liệu không giả định một phiên bản mới nhất.
