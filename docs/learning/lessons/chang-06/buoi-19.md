---
buoi: 19
chang: 6
ten_chang: Run infra, Evaluation và BM25
tuan_goi_y: 7
thu_goi_y: Thứ 2 (lịch gợi ý, có thể đổi)
tasks: [T6.4]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 150 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 6 — Hạ tầng run, Evaluation và BM25
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §4. Module: BM25 và §4.7 Artifact
  - docs/learning/00-project-analysis.md §4.1 Tokenizer BM25: một lệch cụ thể, chưa sửa và §5.2 Bộ processed hiện tại không phải dữ liệu V1.1 đã nghiệm thu
  - docs/learning/lessons/chang-06/buoi-18.md §7. Thao tác từng bước
  - results/metrics_bm25_base_dev.json §recall@1, recall@5, recall@10, recall@20
trang_thai: chưa làm
---

# Buổi 19 — BM25 retrieval-only trên base/dev

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- giải thích BM25 index/search và giới hạn `top-50` lưu so với `top-20` đánh giá;
- phát hiện khác biệt tokenizer giữa spec và `bm25s` mặc định bằng fixture CPU;
- lập entry retrieval-only cho GermanQuAD dev với manifest/fingerprint đúng;
- biết khi nào không được chạy BM25 thật vì base V1.1 hoặc T6.1–T6.3 chưa đạt.

## 2. Vị trí trong bức tranh lớn

T6.4 là baseline retrieval đầu tiên, chỉ dùng GermanQuAD dev và corpus base V1.1; không dùng test để chỉnh tokenizer hay kết luận RQ. Kết quả phải đi qua run infrastructure và Evaluation retrieval-only, lưu trong namespace riêng theo `run_id`. Score BM25 chỉ là mốc so sánh; không mặc định Dense sẽ thắng hay thua.

## 3. Tiền đề

- Buổi 16: base V1.1 đã build/acceptance theo từng file.
- Buổi 17: run identity, fingerprint và manifest.
- Buổi 18: retrieval metrics, validator, `N_eval`, `retrieval_k=20`.
- Biết `src/bm25_retriever.py` hiện dùng `bm25s.tokenize` và `notebooks/02_run_bm25.py` là runner cũ ghi vào artifact cũ.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ fixture | Sai lầm hay gặp |
|---|---|---|
| BM25 | Inverted index chấm query theo term frequency/document frequency và độ dài tài liệu. | Gọi mọi score là xác suất hoặc so score giữa corpus khác nhau. |
| Tokenizer | Fixture `A B C.` giúp thấy token một ký tự/punctuation; spec yêu cầu whitespace split rồi bỏ token không chữ/số, code hiện tại cần audit. | Tin test OOV hiện có đã chứng minh tokenizer khớp spec. |
| Retrieval-only | Output mỗi dòng có `query_id`, `passage_id`, rank, score, retriever; không có reader prediction. | Đưa RetrievalResult vào funnel QA hoặc đọc test ở baseline. |
| Top-50/top-20 | Lưu top-50 để phân tích; Recall@20 chỉ nhìn 20 đầu. | Tính Recall@50 rồi ghi là Recall@20. |
| Latency null | Không đo latency theo giao thức → field đo/warm-up `null`, không bịa số. | Dùng thời gian build index làm query latency. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–25 | Giải thích bằng ví dụ | BM25 rank và retrieval-only. |
| 25–55 | Xem input | Đọc §4, code retriever/runner và artifact cũ. |
| 55–95 | Fixture CPU | Tokenizer, tie OOV, top-50/top-20. |
| 95–125 | Kiểm tra | Lập manifest entry và acceptance checklist. |
| 125–150 | Tự giải thích + nhật ký | Quyết định stop/continue trước run thật. |

## 6. Input

- `src/bm25_retriever.py`, `tests/test_bm25_retriever.py` và `notebooks/02_run_bm25.py` chỉ đọc.
- Spec §4.3/§4.7 và §8.7 về tokenizer, artifact, run manifest.
- Fixture CPU: corpus nhỏ `p1="A Berlin"`, `p2="Berlin Berlin"`, `p3="C"`; query `Berlin`, query rỗng và query OOV.
- `results/metrics_bm25_base_dev.json`, `data/processed/retrieval_bm25_base_dev.jsonl` và gold `data/processed/qa_germanquad_dev.jsonl` là artifact cũ để truy vết, không phải kết quả V1.1.

## 7. Thao tác từng bước

1. Đọc retriever và test hiện tại:

   ```powershell
   Select-String -LiteralPath 'src/bm25_retriever.py' -Pattern 'class BM25Retriever|bm25s\.tokenize|top_n|OOV|return \[\]'
   Select-String -LiteralPath 'tests/test_bm25_retriever.py' -Pattern 'empty_query|fully_oov|no_duplicate|ranks_start'
   ```

   Kỳ vọng: thấy query rỗng trả rỗng, OOV có tie-break ổn định theo passage_id; test chưa đủ để kết luận tokenizer §4.3.
2. Kiểm tra tokenizer fixture bằng package đang ghim (chỉ in output, không ghi artifact):

   ```powershell
   & '.venv/Scripts/python.exe' -B -c "import bm25s; x=bm25s.tokenize(['A B C. Berlin'], stopwords=None, show_progress=False); print(x.ids, sorted(x.vocab.keys()))"
   ```

   Kỳ vọng: `[[0]] ['berlin']`, cho thấy token một ký tự bị loại theo mặc định `bm25s`; đối chiếu với spec, ghi lệch cần sửa trong T6.4 nếu đúng trên runtime.
3. Tính tay fixture rank: query `Berlin` phải có `p2` và `p1` là ứng viên; query rỗng trả list rỗng; query OOV trả top passage IDs theo tie-break ổn định, mọi score 0. Không suy score số thật cho BM25 từ ví dụ này.
4. Phân biệt artifact: `top_n=50` có thể lưu top-50 cho mỗi query, nhưng Evaluation tính Recall@1/5/10/20; không dùng top-50 để đổi tên metric.
5. Đọc artifact cũ mà không chạy runner:

   ```powershell
   Get-Content -LiteralPath 'results/metrics_bm25_base_dev.json' -TotalCount 35
   Get-FileHash -LiteralPath 'data/processed/retrieval_bm25_base_dev.jsonl' -Algorithm SHA256
   Test-Path -LiteralPath 'data/processed/v1.1/corpus_base.jsonl'
   ```

   Kỳ vọng: thấy `run_id`/metrics cũ và hash file cũ; `data/processed/v1.1/corpus_base.jsonl` chỉ được `True` sau nhánh T5 đã build, không coi `False` hiện tại là lỗi của BM25.

   **Bẫy tên metric trong file cũ:** đối chiếu bằng lệnh chỉ-đọc của Buổi 18 bước 3, không chạy lại BM25. Trên 1.031 query có gold và 51.550 dòng ranking cũ:

   | k | Field `recall@k` cũ (khớp GoldCoverage) | Hit@k tính lại |
   |---:|---:|---:|
   | 1 | 0,1896 | 0,1979 |
   | 5 | 0,4481 | 0,4617 |
   | 10 | 0,5708 | 0,5839 |
   | 20 | 0,6639 | 0,6780 |

   Các số làm tròn bốn chữ số. Hit dùng công thức Recall V1.1 nhưng vẫn trên **bộ artifact cũ**, không được đưa vào bảng benchmark V1.1; tên field không đủ chứng minh công thức. Giữ nguyên các file cũ. Nếu không khớp, xem lại input/path/hash và mẫu số theo Buổi 18. [docs/learning/lessons/chang-06/buoi-18.md §7. Thao tác từng bước] [results/metrics_bm25_base_dev.json §recall@1, recall@5, recall@10, recall@20]
6. Lập manifest retrieval-only **đủ sáu nhóm của Buổi 17 bước 3**: `mode="retrieval_only"`, `dataset="germanquad"`, `split="dev"`, retriever `bm25`, reranker tắt, `retrieval_k=20`, `reader_k=5`, đường dẫn/hash gold, corpus và corpus manifest, `retrieval_results_path`; đủ run identity/seed, bốn checkpoint `null` và bundle hash. Không có reader prediction.
7. Ghi điều kiện latency: nếu run không đo, metadata thiết bị/warm-up/latency là `null`; index build time không phải query latency. Entry phải được ghi trước query đầu và artifact nằm trong namespace `run_id` riêng.
8. **Nhánh thực thi sau gate (không tính vào 150 phút):** sửa tokenizer/runner nếu test T6.4 xác nhận lệch spec; chỉ khi T5.0–T5.8 và T6.1–T6.3 pass mới chạy BM25 trên `data/processed/v1.1/` GermanQuAD dev. Lệnh runner hiện tại ghi artifact cũ nên chỉ là **LỆNH MẪU – chưa xác minh cho namespace V1.1**; không chạy trong giờ học liệu.

## 8. Dự đoán trước, kiểm tra sau

1. Query rỗng có tạo BM25 result không? **Không; trả list rỗng theo test/code hiện có.**
2. Query OOV có score khác nhau không? **Không; các score 0 và tie-break ổn định theo passage_id.**
3. Tại sao lưu top-50 nhưng metric Recall@20? **Top-50 phục vụ phân tích; metric chính chỉ dùng 20 đầu theo P1.**
4. Base cũ có thể dùng để gọi là BM25 V1.1 không? **Không; phải có base V1.1, manifest và fingerprint mới.**

## 9. Output cần có

- Bảng tokenizer fixture và rank/tie-break dự đoán.
- Manifest retrieval-only mẫu, quy tắc top-50/top-20 và latency null.
- Bảng phân biệt artifact BM25 cũ với run V1.1, gồm nhãn `recall@k` cũ khớp GoldCoverage và Hit tính lại.
- Sau nhánh thực thi: retrieval results top-50, metrics dev và manifest entry trong namespace `run_id`; chưa chạy thì ghi chưa làm.

## 10. Kiểm tra hiểu bài

1. BM25 index build time có nằm trong query latency không? **Không; spec tách build/load index khỏi retrieval latency.** Chấm: nêu đúng mốc đo.
2. OOV tie-break cần ổn định vì sao? **Để kết quả tái lập, không phụ thuộc tie-break nội bộ không đảm bảo.** Chấm: nêu passage_id.
3. Retrieval-only có cần reader prediction không? **Không; chỉ RetrievalResult và gold retrieval.** Chấm: đúng mode.
4. Nếu base V1.1 chưa qua gate thì làm gì? **Dừng run thật, chỉ làm fixture/kiểm code.** Chấm: không chạy trên artifact cũ.
5. Không đo latency thì ghi gì? **Metadata đo/warm-up và latency tương ứng là null, không bịa số.** Chấm: phân biệt với missing error.
6. Có thể dùng 0,6780 vừa tính lại làm Recall@20 của run V1.1 không? **Không; nó chỉ sửa cách tính trên gold/corpus/ranking cũ chưa được nghiệm thu.** Chấm: phải nêu nguồn artifact, không chỉ đọc tên metric.

## 11. Bài tự giải thích

Vẽ đường đi `corpus_base V1.1 → tokenizer → BM25 index → top-50 retrieval result → top-20 Evaluation` và đánh dấu nơi manifest/fingerprint được kiểm tra.

Rubric 4 điểm: đúng thứ tự; tách top-50/top-20; có run identity; có điều kiện stop khi base chưa đạt.

## 12. Gate của buổi

Qua giờ hướng dẫn khi sinh viên tính đúng fixture tokenizer/rank, lập manifest retrieval-only và phân biệt artifact cũ. Chỉ sau khi nhánh thực thi T6.4 pass trên base V1.1/dev, có top-50, metrics và manifest riêng, mới kết thúc Chặng 6; không yêu cầu score cao.

## 13. Lỗi thường gặp & cách gỡ

- **Tokenizer:** dùng mặc định bm25s mà không đối chiếu §4.3 → giữ fixture một ký tự/punctuation.
- **Artifact:** đọc score cũ như V1.1 → kiểm namespace, hash và manifest.
- **Metric:** dùng top-50 cho Recall@20 → cắt đúng k.
- **Latency:** tính index build vào query → tách mốc đo.
- **Gate:** chạy trước base acceptance → dừng và ghi CẦN XÁC MINH.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T6.4
Lệnh chỉ-đọc đã chạy:
Output path:
Tokenizer/rank fixture:
Manifest retrieval-only:
Artifact cũ đã đối chiếu:
Nhánh thực thi đã chạy/chưa chạy:
Hiểu/chưa hiểu:
Lỗi:
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Chỉ khi T6.4 qua gate mới đọc T7.1 về sampling/mask/dual encoder; nếu chưa, hoàn tất base/run infrastructure trước.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: tokenizer implementation sau test T6.4 có khớp §4.3 hay cần sửa.
- CẦN XÁC MINH: runner mới, namespace run và manifest schema sau T6.1–T6.3.
- CẦN XÁC MINH: base V1.1 corpus revision và số query dev trước BM25 thật.
