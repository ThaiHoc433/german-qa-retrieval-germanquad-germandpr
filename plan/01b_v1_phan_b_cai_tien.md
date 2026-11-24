# V1 — Phần B: Cải tiến (giai đoạn 11–18)

Cập nhật: 2026-10-14. Trạng thái: kế hoạch đề xuất; không phải báo cáo kết quả đã đo.

[Mục lục](README.md) · [V1 phần A](01a_v1_phan_a_baseline.md) · [V2 — Nghiên cứu](02_v2_nghien_cuu.md)

## Mục tiêu, phạm vi và điều kiện bắt đầu

Sau phần A (giai đoạn 0–10), bổ sung truy hồi ngữ nghĩa và đánh giá lại candidates để cải thiện đáp án. Mức tăng chất lượng phải được benchmark xác nhận; thêm module không bảo đảm tốt hơn.

Điều kiện bắt đầu: corpus/splits/qrels ổn định, BM25 + reader chạy được, reader và evaluator đã kiểm tra, baseline artifacts/predictions được lưu.

```text
Câu hỏi
   ├── BM25 ── candidates ──┐
   └── Dense ─ candidates ──┤
                           ↓
                     Hợp nhất RRF
                           ↓
                        Top-20
                           ↓
                  Cross-encoder reranker
                           ↓
                        Top-5
                           ↓
                        Reader
                           ↓
                 Đáp án + bằng chứng
```

- Phần chính: dense → hybrid RRF → cross-encoder → đánh giá.
- Bổ sung: hard-negative mining và robustness.
- Router thích nghi không thuộc V1; xem V2 (nghiên cứu).
- Top-20/top-5 là cấu hình khởi điểm, chọn lại trên dev nếu thí nghiệm cho thấy cần.

## Giai đoạn 11 — Đóng băng baseline và tổ chức thí nghiệm

### Mục tiêu và đầu vào

Tạo mốc đối chiếu có thể chạy lại. Input là artifacts và kết quả phần A.

### Công việc

- Lưu reader/tokenizer, corpus, BM25 index, qrels, split IDs.
- Lưu predictions, metrics, timing, phần cứng, seed và dependency versions.
- Đặt mã thí nghiệm và quy tắc lưu kết quả, không ghi đè baseline.
- Dùng chung evaluator và quy tắc normalization.
- Cố định reader/corpus trong so sánh chính.
- Tách kết quả từ dev, test và dữ liệu đã dùng trong phát triển.

| Mã | Cấu hình |
|---|---|
| B0 | BM25 → reader |
| B1 | Dense → reader |
| H0 | BM25 + dense → RRF → reader |
| H1 | BM25 + dense → RRF → cross-encoder → reader |

### Kiến thức và lưu ý

Experiment tracking, versioning, reproducibility và controlled comparison. Đổi đồng thời reader/corpus/retriever sẽ khó xác định nguồn cải thiện.

Nếu đã xem test ở phần A, vẫn có thể báo cáo đối chiếu trên benchmark đó, nhưng lựa chọn ở phần B chỉ dựa dev. Nếu thiết kế đã dựa trên lỗi test, cần công bố và bổ sung tập độc lập cho kiểm chứng xác nhận.

### Đầu ra và nghiệm thu

Baseline manifest và bảng thí nghiệm. B0 chạy lại nhất quán trong sai khác giải thích được; mọi model/data path có version đi kèm.

## Giai đoạn 12 — Dense retriever

### Mục tiêu và dữ liệu

Tìm passages theo biểu diễn ngữ nghĩa trên cùng corpus. Tái sử dụng giai đoạn 6 nếu đã hoàn thành. GermanDPR train dùng khi fine-tune; qrels dev dùng đánh giá/chọn cấu hình.

```text
Offline: corpus → context encoder → embeddings → FAISS
Online:  query → query encoder → query vector → FAISS → top-k
```

### Phương pháp và công việc

- Chọn cặp encoder hoặc model có quy tắc query/passage encoding rõ.
- Tuân thủ tokenizer, pooling, prompt/prefix và title–text format của model.
- Batch encode corpus; lưu embeddings và vector-to-passage mapping cùng một manifest.
- Chọn dot product/cosine theo model. Cosine qua inner product yêu cầu chuẩn hóa cả query và corpus.
- Dùng tìm kiếm chính xác làm mốc trước khi tối ưu ANN.
- Interface dense giống BM25: query vào, IDs/ranks/scores ra.
- Inference ở eval mode, không tính gradient; đo query encoding riêng với search.

### Kiến thức cần nắm

Bi-encoder, vector dimension, pooling, similarity, normalization, batching, exact/approximate nearest-neighbor search. FAISS tìm vector; nội dung passage và provenance phải được quản lý riêng.

### Lỗi cần phòng tránh

- Encoder không tương thích hoặc sai pooling.
- Index vector đúng nhưng mapping passage sai thứ tự.
- Normalize một bên nhưng không normalize bên còn lại.
- Truncation làm mất nội dung quan trọng mà không ghi nhận.
- Dùng index cũ sau khi đổi context encoder hoặc preprocessing.

### Kiểm tra và đầu ra

- Số vectors bằng số passage IDs, đúng dimension, không NaN/Inf.
- So FAISS với similarity tính trực tiếp trên corpus nhỏ.
- Lưu top-k thực tế để đọc mẫu đúng/sai.
- Đo Recall/Hit/MRR và latency, RAM, index size.

Đầu ra: encoder manifest, embeddings, index, ID mapping và B1 results. Đổi context encoder/tokenizer/pooling/corpus thì tạo lại embeddings/index; đổi query encoder phải kiểm tra compatibility và đánh giá lại.

## Giai đoạn 13 — Hybrid retrieval bằng RRF

### Mục tiêu

Hợp nhất kết quả lexical và semantic, tận dụng candidates bổ sung của hai retriever.

### Thuật toán

Với passage d:

```text
RRF(d) = tổng trên các retriever r của 1 / (c + rank_r(d))
```

- Rank bắt đầu từ 1.
- Vắng khỏi danh sách r: đóng góp bằng 0.
- c là hằng số làm trơn, không phải top-k.
- Gộp theo passage_id, có quy tắc đồng điểm ổn định.

RRF kết hợp thứ hạng, tránh cộng trực tiếp hai thang raw score không tương đương. Nền tảng: [RRF, SIGIR 2009](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/).

### Các tham số phải tách biệt

| Tham số | Ví dụ khởi điểm | Vai trò |
|---|---:|---|
| bm25_top_k | 50 | Candidates BM25 |
| dense_top_k | 50 | Candidates dense |
| fusion_top_k | 20 | Giữ sau fusion |
| reader_top_k | 5 | Reader nhận |
| rrf_c | Chọn trên dev | Hằng số làm trơn |

### Công việc

- Truy hồi cả hai danh sách trên cùng corpus/query.
- Hợp nhất, deduplicate, tính RRF, sắp xếp và lấy top-k.
- Lưu source retriever, ranks/scores từng nhánh và rank sau fusion.
- Chạy H0 trước khi thêm reranker để đo lợi ích hybrid riêng.
- Tuning candidate depths/RRF trên dev với ngân sách thử nghiệm được ghi lại.

### Kiểm tra và nghiệm thu

Tính tay RRF trên hai danh sách nhỏ, gồm ID chỉ xuất hiện một bên, xuất hiện hai bên và đồng điểm. Không còn ID trùng trong output. Có bảng B0/B1/H0, Recall@20, evidence-in-top5, QA EM/F1 và thời gian fusion.

## Giai đoạn 14 — Cross-encoder reranking

### Mục tiêu và luồng

```text
Hybrid top-20 → 20 cặp (query, passage) → Cross-encoder
             → Điểm liên quan → Xếp lại → Top-5 → Reader
```

Reranker chấm liên quan; reader trích answer span. Nền tảng: [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085).

### Công việc và kỹ thuật

- Chọn model hỗ trợ tiếng Đức; đọc model card và lịch sử training.
- Tokenize query–passage chung; chấm batch.
- Xác nhận output: một score hay nhiều label logits, đúng label liên quan.
- Chọn thứ hạng từ score theo quy ước model; không mặc định score là xác suất answer đúng.
- Xử lý max tokens: truncation hoặc windows, ghi rõ cách tổng hợp nếu dùng windows.
- Lưu mapping input pair → passage_id, rank trước/sau.
- Khởi đầu checkpoint sẵn có. Fine-tune reranker là thí nghiệm riêng nếu cần, với positive/negative data và loss phù hợp, không tự gộp vào kết quả H1 đầu tiên.

### Đối chiếu công bằng

```text
Cùng candidate pool hybrid top-20
   ├── Lấy top-5 trực tiếp → Cùng reader
   └── Rerank → Top-5     → Cùng reader
```

### Lỗi cần chú ý

- Bằng chứng cuối passage bị truncation cắt mất.
- Sort ngược chiều score hoặc đọc sai cột logits.
- Trộn score reranker với raw BM25/dense tùy ý.
- Checkpoint đã học các query được coi là dev độc lập.
- Cho rằng reranker cứu được evidence vắng khỏi candidate pool.

### Đầu ra và nghiệm thu

Reranker module, preprocessing config, H1 predictions. Đo evidence-in-top5, EM/F1, số pairs/windows và latency thêm. Kiểm tra thủ công passages tụt hạng mạnh và các câu H0 đúng/H1 sai.

## Giai đoạn 15 — So sánh đóng góp của cải tiến

### Mục tiêu

Xác định component nào tạo lợi ích và ở chi phí nào. Không yêu cầu mọi cải tiến đều phải thắng; kết quả có thể dẫn đến bỏ component khỏi cấu hình triển khai.

### Ma trận thí nghiệm

| Mã | Retrieval | Reranker | Reader |
|---|---|---|---|
| B0 | BM25 | Không | Cố định |
| B1 | Dense | Không | Cố định |
| H0 | BM25 + dense + RRF | Không | Cố định |
| H1 | BM25 + dense + RRF | Có | Cố định |
| B0-R, tùy tài nguyên | BM25 | Có | Cố định |
| B1-R, tùy tài nguyên | Dense | Có | Cố định |

### Đánh giá ba tầng

1. Candidates: Hit/Recall@N, MRR@N.
2. Selection: giữ ít nhất một evidence hợp lệ trong top-5.
3. QA: EM/F1, source/span validity, latency.

### Phân tích

- BM25 đúng/dense sai và chiều ngược lại.
- Evidence được RRF nâng/hạ khỏi vùng reader.
- Reranker cứu bao nhiêu câu và làm hỏng bao nhiêu câu.
- Reader tận dụng passage tốt hơn được bao nhiêu.
- Nếu score QA tăng, xác định nguồn tăng theo trace; không chỉ nhìn retrieval metric.

### Độ tin cậy và timing

- Paired comparison cùng query; báo cáo chênh lệch và khoảng tin cậy khi thực hiện.
- Cùng hardware, precision, batch size, warm-up, cache và tải.
- Tính query encoder + search + RRF + reranker + reader; tách startup/offline indexing.
- Năm passages có thể tạo số reader windows khác nhau; ghi số tokens/windows để giải thích chi phí.
- Đóng băng cấu hình trên dev rồi mới đánh giá cuối.

### Đầu ra và nghiệm thu

Bảng kết quả, quality–latency plots, error cases, configuration được chọn. Có query traces đủ để giải thích quyết định giữ/bỏ từng component.

## Giai đoạn 16 — Hard-negative mining bổ sung

### Mục tiêu và vị trí

Huấn luyện dense phân biệt passages đúng với passages dễ nhầm. Thực hiện sau khi H0/H1 đã có kết quả để tách tác động huấn luyện.

```text
Train queries → Retriever lấy candidates đứng cao
             → Sàng lọc false negatives
             → Query + positive + hard negatives
             → Fine-tune dense
             → Encode lại corpus + tạo index mới
             → Chạy lại đánh giá giai đoạn 15
```

### Dữ liệu và phương pháp

- Train queries, positives đã biết, mining corpus được định nghĩa rõ.
- GermanDPR đã có negatives; dùng chúng làm mốc, không coi việc có negatives là cải tiến mới.
- Mining bằng BM25/dense hiện tại.
- Contrastive learning, negative log-likelihood trên positive và negatives; in-batch negatives nếu phù hợp.
- Lọc known positives/trùng; reranker có thể hỗ trợ sàng lọc, kèm kiểm tra mẫu bằng người.
- Lưu provenance, mining model, ranks/scores và lý do chọn/loại.

Lọc false negatives có tiền lệ trong [RocketQA](https://aclanthology.org/2021.naacl-main.466/); không nhận kỹ thuật cơ bản này là novelty.

### Lưu ý dữ liệu

- Không dùng query/gold answers/qrels test để mining hoặc filtering cho training.
- Corpus mining và chính sách document holdout phải nhất quán protocol.
- Unjudged passage không chắc negative; positive của query A có thể đúng cho B.
- Khi dùng in-batch negatives, kiểm tra quan hệ nhiều query cùng passage để tránh negative giả.
- Tách kiểm tra substring answer khỏi kiểm tra relevance thực; chúng không tương đương.

### Đối chiếu

```text
Dense ban đầu
vs fine-tune với negatives sẵn có
vs fine-tune với negatives được mining
```

Giữ budget training, backbone và evaluator tương đương. Nếu thay nhiều yếu tố, thêm ablation để tách ảnh hưởng.

### Đầu ra và nghiệm thu

Negative dataset, training config/logs, checkpoint mới, index mới và trước/sau metrics. Reader giữ nguyên trong so sánh mining. Tính cả chi phí encode lại; không dùng index từ encoder cũ.

## Giai đoạn 17 — Phân tích nhóm và robustness

### Mục tiêu

Biết hệ thống yếu ở nhóm và điều kiện nào. Subgroup analysis khác robustness.

| Hướng | Thực hiện |
|---|---|
| Nhóm tự nhiên | Query/context/answer ngắn-dài, loại câu hỏi, lexical overlap |
| Controlled robustness | Lỗi gõ, viết hoa/dấu câu, diễn đạt lại giữ nghĩa |

### Công việc

- Chọn bins độ dài từ train/dev, giữ chung cho tất cả models.
- Xây schema loại query; từ hỏi đầu câu chỉ là heuristic, có nhóm mơ hồ.
- Báo số mẫu/nhóm và độ bất định nếu đủ dữ liệu.
- Tạo cùng bộ perturbations cho mọi cấu hình, seed cố định.
- Giữ corpus nguyên vẹn khi bắt đầu để cô lập tác động query.
- Kiểm tra thay đổi giữ semantics/answer; lưu original–variant mapping và lý do loại.
- Nếu biến đổi context ở thí nghiệm khác, cập nhật offsets/qrels theo thiết kế.

### Metrics và phân tích

EM/F1 clean vs perturbed, mức giảm Recall, tỷ lệ clean-correct → perturbed-wrong và lỗi theo stage. Không dùng consistency một mình: hệ thống trả cùng đáp án sai vẫn nhất quán.

### Đầu ra và nghiệm thu

Subgroup tables, perturbation dataset/protocol và error examples. Kết luận giới hạn theo loại nhiễu/miền đã đo; không suy rộng vài lỗi gõ thành mọi tình huống thực tế.

## Giai đoạn 18 — Tích hợp ứng dụng và triển khai lại

### Backend

- Chọn `bm25`, `dense`, `hybrid`; bật/tắt reranking.
- Nạp thêm encoder/index/reranker đúng versions.
- Giữ schema response tương thích; scores/ranks/timings theo từng stage.
- Giới hạn top-k/query length và concurrency.
- Kiểm tra checkpoint–tokenizer–embedding–index–corpus compatibility.

### Frontend

- Chọn cấu hình, xem rank trước/sau rerank.
- Xem benchmark đã lưu và thời gian từng bước.
- Bỏ qua response cũ; offsets/highlight giữ đúng nguyên tắc phần A.
- Không hiển thị score thô như phần trăm độ chính xác.

### Deployment

- Đo RAM/VRAM khi nạp đồng thời models; nhiều workers có thể nhân số bản sao.
- Đo startup, latency nhiều requests; async không tự tăng tốc GPU/CPU inference.
- Health/readiness, artifacts manifest và khả năng quay lại baseline.
- So API với script offline; không so latency ở các hardware khác nhau mà bỏ qua khác biệt.

### Nghiệm thu

Ứng dụng hỗ trợ cấu hình được benchmark chọn; restart nạp đúng artifacts; request giới hạn không làm vượt tài nguyên đã đo; API/offline nhất quán.

## Ghi chú cài đặt dùng xuyên suốt

| Rủi ro | Biểu hiện | Cách phát hiện |
|---|---|---|
| Sai ID mapping | Passage không khớp vector | Kiểm tra mapping trên corpus nhỏ |
| Sai similarity | Rank bất thường | So matrix similarity và FAISS |
| Truncation | Passage đúng bị rerank thấp | Log token/window và vị trí evidence |
| Sort/output logits sai | Candidate không liên quan lên đầu | Mẫu đơn giản có expected behavior |
| Leakage | Dev tốt bất thường | Audit splits và lịch sử checkpoint |
| False negatives | Fine-tune giảm chất lượng | Đọc mẫu mining, kiểm tra positives nhiều query |
| Đo thiếu chi phí | Router/reranker có vẻ rẻ giả tạo | Timing end-to-end và từng stage |
| Score bị hiểu sai | UI báo độ chính xác không có căn cứ | Giữ tên score, không gán probability |

## Thứ tự học và làm

1. Embedding/bi-encoder/similarity → dense độc lập.
2. Rank fusion/RRF → hybrid.
3. Cross-encoder/token limit/batching → reranking.
4. Metrics và error traces → quyết định cấu hình.
5. Contrastive learning/false negatives → mining bổ sung.
6. Subgroup/perturbation validation → robustness.
7. Resource lifecycle/concurrency → deploy.

```text
11 → 12 → 13 → 14 → 15
                     ├── 16 Mining → đánh giá lại 15
                     └── 17 Nhóm/robustness
                               ↓
                         18 Tích hợp/deploy
```

## Checklist V1 phần B (giai đoạn 11–18)

- [ ] Baseline phần A đóng băng và tái chạy được.
- [ ] Dense similarity/index/IDs đã kiểm tra.
- [ ] RRF qua ví dụ tính tay, tham số từng stage tách biệt.
- [ ] Reranker output/truncation/ranking được kiểm tra.
- [ ] B0/B1/H0/H1 cùng protocol và reader.
- [ ] Đo đủ end-to-end latency và tài nguyên.
- [ ] Nếu mining: negatives được kiểm tra, không dùng test labels, index được tạo lại.
- [ ] Nếu robustness: variants giữ nghĩa, cùng protocol cho mọi model.
- [ ] Có trace và phiên bản cho mọi kết quả.
- [ ] API/UI/deploy được nghiệm thu lại.

Mốc hoàn thành phần cải tiến chính là giai đoạn 15. Mining/robustness bổ sung bằng chứng; chỉ đưa cấu hình vào ứng dụng sau khi đo được chất lượng và chi phí.
