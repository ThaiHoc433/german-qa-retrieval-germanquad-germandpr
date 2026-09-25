# Roadmap kiến thức, bài báo và kiến trúc German QA V1.1

Ngày: 2026-09-12. Ảnh được tạo bằng ImageGen built-in, đối chiếu module spec V1.1, design và learning roadmap hiện tại. Đây là lộ trình lý thuyết đề xuất từ nhu cầu triển khai; không phải danh sách paper bắt buộc đã được spec quy định. Chưa có số liệu benchmark trong hình.

## Bài đọc theo số trong ảnh

1. Devlin et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/). Học encoder, contextual representation, fine-tuning và QA head.
2. Chan, Schweter & Möller (2020). [German’s Next Language Model](https://aclanthology.org/2020.coling-main.598/). Nền tảng GBERT/GELECTRA; project dùng gbert-base làm mặc định.
3. Rajpurkar et al. (2016). [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/). Học span extraction, references, EM và token-F1. Chuẩn hóa QA tiếng Đức của project lấy theo spec, không sao chép nguyên xi script SQuAD.
4. Möller, Risch & Pietsch (2021). [GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval](https://aclanthology.org/2021.mrqa-1.4/). Bài nền của đề tài; hiểu dữ liệu QA và retrieval.
5. Robertson & Zaragoza (2009). [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019). Học lexical ranking, IDF, TF saturation, length normalization, k1 và b.
6. Karpukhin et al. (2020). [Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2020.emnlp-main.550/). Học dual-encoder, dot product, contrastive learning và negative sampling.
7. Nogueira & Cho (2019). [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085). Học cross-encoder và relevance classification. Sampling, BCE và checkpoint selection thực tế theo module spec.
8. Reimers & Gurevych (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/). Đọc bổ trợ để hiểu sentence embedding và bi-encoder. Không phải mô hình Dense mặc định V1.

Thứ tự học là 8 khối trong ảnh; số paper chỉ dùng để tra cứu. Có thể đọc lướt [4] đầu tiên để hiểu đề tài rồi quay lại từng bài theo khối.

## Diễn giải kiến trúc

- BM25 và Dense là hai lựa chọn thay thế trong một run, không hợp nhất kết quả trong V1.
- Corpus dùng chung cho cả hai retriever; base ⊂ mid ⊂ large. Base là số thực sau build, ~7,5K chỉ dự kiến; mid tổng 50.000 và large tổng 150.000 passage.
- Base lấy context GermanQuAD và positive context GermanDPR. Wikipedia bổ sung distractor, lọc title theo P4. Hard-negative contexts không được thêm vào corpus chỉ vì là hard negative; chunk trùng nguồn hợp lệ vẫn giữ theo D6.
- Retriever có thể lưu top-50 artifact, nhưng retrieval pool của pipeline là top-20; Selection đưa tối đa 5 passage vào Reader, thiếu thì không pad.
- OFF lấy top-5 theo rank retriever; ON dùng cross-encoder chọn 5 từ cùng pool. Reader gbert-base với QA head trả span và offset.
- GermanQuAD train phục vụ reader; GermanDPR train phục vụ Dense và reranker. Chống leakage trước training.
- Oracle runner đưa toàn bộ gold passages vào cùng reader, rồi Evaluation chấm artifact riêng. Không có bước oracle bắt buộc nằm giữa pipeline và Evaluation.
- Reranker thuộc V1; Hybrid/RRF, hard-negative mining và Haystack thuộc V2.

## Metrics và phân tích lỗi

Chất lượng dùng N_eval = số QARecord có gold_evidences khác rỗng trên GermanQuAD test gốc. Gold rỗng vẫn lưu và báo riêng. Query thiếu prediction hoặc lỗi vẫn thuộc N_eval.

- Recall@k = Hit@k = số câu có ít nhất một gold trong top-k / N_eval; k=1,5,10,20. Lấy danh sách trước selection.
- MRR@10: trung bình 1/rank của gold đầu tiên trong top-10; không có thì 0.
- GoldCoverage@k: trung bình tỷ lệ gold tìm được của từng query; metric phụ, khác Hit@k.
- EM: khớp hoàn toàn sau chuẩn hóa theo spec. Token-F1 = 2PR/(P+R), với P và R là precision/recall của token đáp án dựa trên số lần xuất hiện; không có token chung thì F1=0. Lấy max qua reference_answers.
- Chuẩn hóa QA: NFC, lowercase, xóa punctuation theo regex spec, gộp whitespace. Giữ ä/ö/ü/ß, không casefold và không bỏ mạo từ tiếng Đức.
- Latency: mean/p50/p95 cho retrieval, rerank, read, total; báo n_latency riêng cho mỗi stage, chỉ record status=ok thuộc N_eval có giá trị stage đó. Integration đo, Evaluation chỉ tổng hợp.

Gán bucket theo thứ tự, mỗi câu đúng một bucket:

1. execution_failure: thiếu record hoặc status=error.
2. retrieval_miss: không gold trong retrieval pool top-20.
3. selection_failure: có gold trong pool nhưng không trong reader input, áp dụng cả bật/tắt reranker.
4. reader_failure: có gold trong reader input nhưng EM=0.
5. correct: có gold trong reader input và EM=1.

Tỷ lệ bucket = count(bucket)/N_eval × 100%; tổng năm nhóm = 100%.
Tỷ lệ đáp án sai theo EM = count(final_correct=false)/N_eval × 100% = (1 − EM trung bình) × 100%.
Không đồng nhất tỷ lệ đáp án sai với tổng bốn failure buckets: retrieval_miss hoặc selection_failure vẫn có thể trả đúng từ passage không được gán gold. Vì vậy final_correct lưu độc lập.

Review thủ công các case F1>0, EM=0: tối đa 100 case, seed 42, lấy trên danh sách query_id đã sort. Ghi chú định tính, không sửa tỷ lệ bucket tự động.
Oracle gap là chênh EM/F1 giữa oracle reader và end-to-end; oracle là mốc năng lực reader trên gold, không phải trần tuyệt đối.

## Ma trận nghiên cứu

2 retriever × 2 trạng thái reranker × 3 corpus = 12 cấu hình end-to-end, kèm oracle.
RQ1 dùng 6 cấu hình tắt reranker; RQ2 phân tích bucket và final_correct; RQ3 so cặp bật/tắt cùng pool và reader_k.
Chọn checkpoint trên dev rồi khóa trước test; không retrain theo tier.
GermanDPR test chỉ là retrieval-only sanity check riêng.
V1.2 A3 còn draft, cần chốt trước production checkpoint selection reranker phụ thuộc metadata mới.

## Nguồn nội bộ

- [Module spec V1.1](../superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md)
- [Design và RQ](../superpowers/specs/2026-09-10-german-qa-retrieval-design.md)
- [Learning roadmap](../superpowers/plans/2026-09-11-german-qa-learning-roadmap.md)

