# German Open-Domain Extractive QA — Thiết kế đề tài

- **Ngày:** 2026-09-10
- **Bài báo nền tảng:** Möller, Risch & Pietsch, *GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval* (MRQA/ACL 2021)
- **Dataset:** GermanQuAD (13.722 cặp hỏi-đáp extractive; train 11.518 / test 2.204), GermanDPR (train 9.275 / test 1.025, mỗi mẫu 1 positive context + 3 hard negatives)

> **Lưu ý:** file này giữ vai trò định hướng (core contribution, lộ trình V1/V2). Mọi chi tiết kỹ thuật — quy mô và thành phần corpus, chunking, split, metric, error decomposition, latency — đã được thay thế bởi `2026-09-10-german-qa-retrieval-module-specs.md` (V1.1). Khi hai file khác nhau, module spec là chuẩn; không lập implementation plan từ các con số trong file này.

## Core contribution

Xây dựng và đánh giá có hệ thống một German open-domain extractive QA pipeline, trong đó:
1. Phân tích ảnh hưởng của corpus scale (5K / 50K / 100K passage) lên chất lượng retrieval và QA end-to-end.
2. Xác định bottleneck dịch chuyển thế nào giữa retriever, reranker và reader khi corpus scale tăng.

Hai trục này giao nhau là insight chính: không chỉ "corpus lớn thì tệ hơn" hay "component X là bottleneck", mà là bottleneck dịch chuyển ra sao theo quy mô corpus.

## 1. Lộ trình 2 phiên bản

- **V1 (core + 2 cải tiến):** BM25 baseline, dual-encoder dense retriever (fine-tune theo DPR), extractive reader, cross-encoder reranking, full evaluation (gồm ablation theo corpus scale), robustness/error analysis theo độ dài & loại câu hỏi.
- **V2 (mở rộng):** V1 + hybrid BM25+dense retrieval (score fusion) + hard-negative mining (dùng corpus 50K/100K để mine negative thật khó, retrain retriever) + bản dựng lại bằng Haystack `Pipeline` (so sánh tooling, không tính điểm cải tiến rubric).

## 2. Kiến trúc pipeline tổng thể

```
Query → [Retriever: BM25 | Dual-Encoder Dense | Hybrid(V2)] → top-k passages
      → [Cross-Encoder Reranker] → top-k' passages
      → [Extractive Reader] → answer span
```

## 3. Cấu trúc thư mục project

```
nlp/
  data/{raw,processed}/          # dataset gốc + corpus 3 tầng đã dedup
  notebooks/                     # 1 notebook Colab / giai đoạn
  src/                           # corpus.py, bm25.py, dense_retriever.py,
                                  # reranker.py, reader.py, eval_metrics.py
  checkpoints/                   # model weights (gitignore, sync Google Drive)
  results/                       # bảng số liệu, biểu đồ theo từng corpus scale
  docs/superpowers/specs/        # spec này
```

## 4. Data pipeline & corpus 3 tầng quy mô

- Load `deepset/germanquad` + `deepset/germandpr` qua HF `datasets`.
- Gán `passage_id` ổn định (hash text đã normalize) cho mọi passage, dùng chung cho toàn bộ corpus và để so khớp gold khi tính Recall@k/MRR (so theo ID, không so string).
- Xây 3 tầng corpus eval, dùng chung quy trình dedup:
  - **5K:** passage gốc dedup từ GermanQuAD + GermanDPR (train+test).
  - **50K:** 5K + distractor thật từ German Wikipedia (`wikimedia/wikipedia`, config `20231101.de`, streaming — không tải toàn bộ dump), chunk ~100 từ/passage.
  - **100K:** mở rộng thêm distractor tương tự.
- Model **chỉ train một lần** trên train set gốc (không phụ thuộc corpus eval); corpus 3 tầng chỉ ảnh hưởng bước **index + eval retrieval** (encode corpus, build FAISS/BM25 index, chạy eval — lặp lại 3 lần, không train lại).

## 5. BM25 baseline

`rank_bm25.BM25Okapi` cho corpus 5K/50K; chuyển sang `bm25s` (numpy/scipy, nhanh hơn) cho corpus 100K để latency đo được có ý nghĩa. Tokenizer tiếng Đức đơn giản (lowercase + regex, hoặc spaCy `de_core_news_sm`).

## 6. Dense retriever — dual-encoder fine-tune theo DPR

Không gọi đây là "train DPR từ đầu" — đây là **fine-tune một dual-encoder retriever theo phương pháp DPR**, khởi tạo từ checkpoint pretrained.

Kiến trúc hai tháp độc lập, không share weight:

```python
question_encoder = AutoModel.from_pretrained("deepset/gbert-base")  # tower 1
passage_encoder  = AutoModel.from_pretrained("deepset/gbert-base")  # tower 2
```

Cả hai khởi tạo cùng checkpoint pretrained nhưng **cập nhật độc lập** khi fine-tune trên GermanDPR train (question, positive, 3 hard-negatives), loss contrastive/NLL kiểu DPR gốc (in-batch negatives + hard negatives). Similarity = dot product giữa output `question_encoder` và `passage_encoder`. Train trên Colab T4, batch 16-32, 2-3 epoch. Sau train: encode corpus qua `passage_encoder` → FAISS `IndexFlatIP` (exact search, đủ nhanh ở scale 100K).

## 7. Extractive reader

`deepset/gbert-base` hoặc `deepset/gelectra-base` + `AutoModelForQuestionAnswering`, fine-tune kiểu SQuAD (start/end logits) trên GermanQuAD train, xử lý context dài bằng `doc_stride`/`return_overflowing_tokens`.

## 8. Cross-encoder reranker (V1)

`sentence_transformers.CrossEncoder`, fine-tune trên cặp (question, passage) từ GermanDPR (positive=1, hard-negative=0). Rerank top-k retriever (vd top-20) → top-k' (vd top-5) trước khi đưa vào reader.

## 9. Evaluation framework

### 9.1 Retrieval

- **Recall@k:** với mỗi câu hỏi, 1 nếu gold passage nằm trong top-k, ngược lại 0. Trung bình trên toàn bộ câu hỏi.
- **MRR@10:** nghịch đảo hạng của gold passage đầu tiên tìm được trong top-10; 0 nếu không có trong top-10. Trung bình trên toàn bộ câu hỏi.
- Giả định: mỗi câu hỏi có một gold passage duy nhất (khớp cấu trúc GermanQuAD/GermanDPR).
- Ví dụ minh họa (gold ở hạng 2): Recall@1 = 0, Recall@5 = 1, Reciprocal Rank@10 = 1/2.
- Chạy trên cả 3 tầng corpus (5K/50K/100K).

### 9.2 QA

- **Exact Match (EM):** so câu trả lời dự đoán với đáp án chuẩn sau chuẩn hóa thống nhất (lowercase, bỏ dấu câu, chuẩn hóa khoảng trắng); khớp hoàn toàn = 1, ngược lại = 0.
- **Token-level F1:** Precision = số token trùng / số token dự đoán; Recall = số token trùng / số token gold; F1 = 2PR/(P+R). Ví dụ: dự đoán "Einstein" vs gold "Albert Einstein" → EM = 0, P = 1/1, R = 1/2, F1 ≈ 0,667.
- Hai chế độ đo:
  - **Reader-only (oracle):** đưa thẳng gold passage vào reader — đo trần năng lực reader, độc lập corpus scale.
  - **End-to-end:** reader nhận đoạn do retrieval + reranking chọn — đo cả hệ thống, ở cả 3 tầng corpus.

### 9.3 Latency

- **Retrieval-only:** từ câu hỏi đầu vào đến danh sách top-k, có tính thời gian query encoding với dense retriever.
- **End-to-end:** từ câu hỏi đầu vào đến đáp án cuối cùng.
- Không tính thời gian train hay build index vào latency trả lời (đây là bước offline, tách riêng).
- Báo cáo mean, p50, p95; đo trên cùng một thiết bị, có warm-up trước khi đo để tránh lệch do lazy-loading.
- Đo ở cả 3 tầng corpus (kỳ vọng latency BM25/dense tăng theo corpus size).

### 9.4 Error decomposition — quy trình 2 giai đoạn

**Giai đoạn 1 — phân loại tự động theo thứ tự pipeline**, với mỗi câu trả lời sai:
1. **Retrieval miss:** gold không có trong top-k ban đầu (theo gold đã gán nhãn).
2. **Reranking failure:** gold có trong top-k retriever nhưng bị loại khỏi top-k' sau rerank.
3. **Reader failure:** gold vẫn đến reader (có trong top-k') nhưng đáp án sai — lỗi reader hoặc bước chọn đáp án.

**Giai đoạn 2 — kiểm tra thủ công**: sample một tập con (vd 50–100 case) trong số còn lại sau 3 nhánh tự động trên — đây là các trường hợp nghi ngờ **nhãn thiếu hoặc đáp án tương đương** (annotation issue), cần review tay để không gán nhầm là "lỗi hệ thống" khi thực chất là lỗi đánh giá.

Insight kỳ vọng: tỉ lệ *retrieval miss* tăng rõ khi corpus tăng 5K→100K; *reader failure* gần như ổn định — cho thấy retrieval là nút thắt chính ở quy mô lớn.

## 10. Robustness analysis (V1)

Chia câu hỏi test theo (a) độ dài (ngắn/vừa/dài theo số token) và (b) loại từ hỏi (Wer/Was/Wann/Wo/Warum/Wie...), so sánh Recall@k và EM/F1 theo từng nhóm, ở từng tầng corpus.

## 11. Mở rộng V2

- **Hybrid retrieval:** fusion điểm BM25 + dense (Reciprocal Rank Fusion hoặc weighted sum sau normalize), so với baseline riêng lẻ, ở cả 3 tầng corpus.
- **Hard-negative mining:** dùng dense retriever V1 chạy trên corpus 50K/100K để mine negative thật khó (không chỉ 3 negative có sẵn trong dataset) từ train set, retrain dual-encoder, so Recall@k trước/sau.
- **Haystack version:** dựng lại pipeline V1 bằng Haystack `Pipeline` (load checkpoint đã train), so sánh code complexity/latency với bản thủ công — giá trị portfolio/tooling, không tính là "cải tiến" điểm rubric.

## 12. Môi trường tính toán

Hybrid: code/debug/smoke-test trên local (RTX 3050 Laptop 4GB, CPU cho BM25/FAISS) trên subset nhỏ → train thật (dual-encoder, reader, cross-encoder) trên Colab free (T4) với batch size hợp lý, checkpoint định kỳ lên Google Drive → tải checkpoint về eval/inference local khi cần.

## 13. Sanity checks trước khi train full

BM25 trả list không rỗng; similarity(positive) > similarity(negative) ở dual-encoder sau vài step train; reader trả span index hợp lệ trong context — chạy trên vài chục mẫu, local, trước khi lên Colab train full.

## 14. Hình thức triển khai

Bắt đầu bằng roadmap + hướng dẫn khái niệm cho từng giai đoạn (tự viết code để hiểu sâu cơ chế), chuyển sang code khung sườn (skeleton + TODO) khi cần đẩy nhanh tiến độ ở các phần lặp lại/ít giá trị học thuật.

## 15. Công cụ & thư viện

Python, Colab, PyTorch, HuggingFace Transformers/Datasets, Sentence-Transformers (`CrossEncoder`), `rank_bm25`/`bm25s`, FAISS, `wikimedia/wikipedia` (distractor sourcing), Haystack (V2 only).
