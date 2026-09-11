# German Open-Domain Extractive QA — Thiết kế đề tài

- Ngày khởi tạo: 2026-09-10. Cập nhật định hướng/RQ: 2026-09-11.
- Tên đề tài: Phân tích điểm nghẽn theo quy mô corpus trong hệ thống hỏi đáp mở trích xuất tiếng Đức.
- English: Corpus Scale and Pipeline Bottlenecks in German Open-Domain Extractive QA:
  A Leakage-Controlled Evaluation on GermanQuAD and GermanDPR.
- Bài nền: Möller, Risch & Pietsch, GermanQuAD and GermanDPR (MRQA 2021).
- [Module spec V1.1 đang hiệu lực](2026-09-10-german-qa-retrieval-module-specs.md)
  là chuẩn kỹ thuật. [V1.2 amendment draft](2026-09-11-german-qa-retrieval-v1.2-amendments-draft.md)
  chưa thay thế V1.1.
- [Constitution](../../../.specify/memory/constitution.md) quy định quản trị.
  Design này chưa frozen; không dùng nó để ghi đè hợp đồng module.

## Core contribution

Đóng góp dự kiến ở mức giao thức và phân tích thực nghiệm, không đề xuất model/loss mới:

1. Quy trình dữ liệu có kiểm soát leakage khi kết hợp GermanQuAD/GermanDPR, có audit,
   provenance và quy tắc positive chunk/hard-negative grouping.
2. Bộ thực nghiệm end-to-end trên ba corpus lồng nhau, giữ gold và checkpoint,
   so sánh BM25/Dense và reranker bật/tắt với cùng ngân sách reader.
3. Phân tích điểm nghẽn dịch chuyển theo quy mô corpus, kết hợp error decomposition,
   final_correct và mốc năng lực reader-only.

Chỉ chuyển thành đóng góp đạt được sau nghiệm thu artifact và kết quả thực nghiệm.
Không tuyên bố mới tuyệt đối trước khi đối chiếu related work.

## Research questions

Mọi RQ trả lời trên GermanQuAD test gốc, mẫu số chất lượng N_eval theo V1.1 §3.3.
Checkpoint và cấu hình khóa trên dev trước test; cùng query/gold ở mọi tier.
Base = số thật sau build (dự kiến ~7,5K); mid tổng 50.000; large tổng 150.000 passage.
GermanDPR test chỉ là retrieval-only sanity-check riêng.

### RQ1 — Ảnh hưởng của quy mô corpus

Khi corpus tăng từ base lên mid và large, chất lượng retrieval và QA end-to-end của
BM25 và Dense thay đổi thế nào; mức thay đổi có khác nhau giữa hai retriever không?

- Đo Recall@1/5/10/20 và MRR@10 trên retrieved trước selection; EM/F1 end-to-end.
- 2 retriever × 3 tier, tắt reranker để giữ cố định cách selection top-5.
  Biến động EM/F1 vẫn phản ánh cả retriever, selection và reader.
- Giữ N_eval và checkpoint từng thành phần giữa các tier; không retrain theo tier.
- Bảng 1: retriever × tier × metric. Hình 1: Recall@20 và EM theo quy mô.
- Giả thuyết: chất lượng retrieval có thể giảm khi thêm distractor; không giả định
  trước retriever nào bền hơn. Đây không phải điều kiện nghiệm thu.

### RQ2 — Bottleneck dịch chuyển

Lỗi pipeline nằm chủ yếu ở retrieval, selection hay reader; phân bố đó thay đổi
thế nào khi corpus lớn dần?

- Đo 5 bucket trên N_eval cho 2 retriever × bật/tắt reranker × 3 tier.
- Oracle đọc toàn bộ gold passage; báo EM/F1 reader-only và khoảng cách tới end-to-end.
  Đây là mốc năng lực reader, không phải trần tuyệt đối.
- Báo final_correct theo bucket; không suy kết quả trả lời chỉ từ việc giữ/mất gold.
- Bảng 2: tỉ lệ bucket theo cấu hình/tier. Hình 2: cột chồng, mẫu số thống nhất N_eval.
- Không giả định reader_failure ổn định: tỉ lệ trên N_eval có thể giảm khi nhiều query
  mất gold ở bước trước dù reader không đổi. Retrieval là bottleneck chỉ được kết luận
  khi bảng kết quả hỗ trợ trong phạm vi nghiên cứu.

### RQ3 — Giá trị của khâu selection

Với reader_k=5, reranker giữ gold tốt hơn cắt thẳng top-5 bao nhiêu, chuyển thành
bao nhiêu EM/F1, tốn thêm bao nhiêu latency; lợi ích có đổi theo retriever/tier không?

- So từng cặp bật/tắt cùng retriever/checkpoint, tier, query và retrieval pool.
- Đo chênh selection_failure, EM/F1; latency rerank và total (mean/p50/p95, n_latency).
- Bảng 3: so sánh theo cặp với dấu chênh lệch được ghi rõ.
- Chênh selection_failure chỉ đo khả năng giữ gold (D3).
  Nếu retrieval_miss khác giữa hai run do execution_failure, báo rõ số query liên quan.
  Latency phải ghi số mẫu của từng run, không giả định hai tập run thành công giống nhau.

### Giới hạn và phạm vi kết luận

Ba mốc tier, một seed, distractor Wikipedia lọc title P4, gold theo nhãn, tài nguyên
Colab T4/RTX 3050 4GB. P4 không bảo đảm loại hết near-duplicate hoặc bằng chứng chưa gán gold.
Kết luận hiện tại là mô tả: “Trong cấu hình/tập đánh giá này, A đạt F1 cao hơn B … điểm”.
Chưa có giao thức khoảng tin cậy/kiểm định; không khẳng định ưu thế ổn định hay có ý nghĩa
thống kê. Nếu bổ sung bootstrap, phải chốt giao thức trong amendment riêng trước khi
xem kết quả để chọn cách phân tích.

## Kiến trúc và ranh giới module

Question → Retriever → retrieval pool (20) → Selection (tối đa 5) → Reader → answer span.

- Retriever: BM25 hoặc dual-encoder DPR; Hybrid thuộc V2.
- Selection: reranker chọn lại hoặc lấy trực tiếp top-5 theo rank retriever.
- Reader: trích text[start:end], lưu passage và offset.
- Data sở hữu split/corpus/provenance. Evaluation chỉ chấm artifact.
- Hạ tầng run dùng chung được xây cùng Evaluation; orchestration đầy đủ ở Integration.
- Định dạng và edge case lấy từ V1.1 §1–8, không định nghĩa lại tại đây.

## Dữ liệu và định hướng mô hình

GermanQuAD phục vụ reader/QA; GermanDPR phục vụ Dense/reranker. Wikipedia snapshot
20231101.de cung cấp distractor theo D6/P4. Corpus base chỉ lấy nguồn được phép;
hard negatives nhóm theo context và lưu riêng cho training.

BM25 dùng bm25s ở mọi tier. Dense là hai gbert-base độc lập, CLS pooling,
raw dot product và FAISS IndexFlatIP. Reader khởi tạo gbert-base với QA head.
Reranker dùng cross-encoder và BCE; cấu hình/alternative theo §7.
Mọi hyperparameter, sampling, mask, span scoring, checkpoint selection theo module spec.

## Lộ trình V1 và V2

V1: dữ liệu → Evaluation + hạ tầng run → BM25 → Dense → Reader + oracle runner →
Reranker → Integration → mở rộng corpus và phân tích RQ.

V2 mở rộng: hybrid retrieval, hard-negative mining từ train, dựng lại bằng Haystack.
Chưa thuộc cổng hoàn tất V1; nếu triển khai cần spec riêng. Không dùng test để mine/tune.

Phân tích phụ V1: robustness theo độ dài/loại từ hỏi, từ per_query artifacts;
định nghĩa nhóm phải được ghi rõ trước phân tích, không chọn nhóm chỉ vì kết quả đẹp.

## Môi trường và cách học

Code, fixture, data/BM25/Evaluation chạy CPU local; kiểm tra GPU bằng smoke rồi
train trên Colab khi có T4. Checkpoint gồm optimizer/RNG/step để resume.
Không hứa thời gian train hoặc dung lượng tải trước khi đo thử.
Logic tái sử dụng nằm trong src; notebooks/scripts làm entrypoint và hướng dẫn.

[Lộ trình học 10 chặng](../plans/2026-09-11-german-qa-learning-roadmap.md)
và [danh mục task](../plans/2026-09-11-german-qa-task-catalog.md) tổ chức việc triển khai.
Tài liệu này không xác nhận code/artifact hiện tại đã đạt V1.1.

## Nhật ký cập nhật

2026-09-11: áp dụng F1/F2/C1/D1 đã duyệt; bổ sung RQ1–RQ3, chọn kết luận mô tả;
thay nội dung định hướng kỹ thuật cũ bằng tham chiếu V1.1 để tránh quy mô, thư viện,
bucket và oracle cũ tiếp tục dẫn dắt plan. Không sửa module spec V1.1 frozen.
