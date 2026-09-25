---
buoi: 2
chang: 1
ten_chang: Hiểu đề tài/project
tuan_goi_y: 1
thu_goi_y: Thứ 4 (lịch gợi ý, có thể đổi)
tasks: [T1.2]
moi_truong: [Đọc tài liệu, Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 1 — Hiểu đề tài và khảo sát project
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §Research questions
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §Giới hạn và phạm vi kết luận
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §3.3 Quy tắc xử lý
  - docs/learning/00-project-analysis.md §2.2 RQ → bảng/hình → bằng chứng
  - docs/learning/00-project-analysis.md §4. Hiện trạng module: giữ / sửa / viết mới / build lại
  - docs/learning/00-project-analysis.md §1. Kiểm kê nguồn trước phân tích
trang_thai: chưa làm
---

# Buổi 2 — Nối RQ với bảng, hình và bằng chứng

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- giải thích câu hỏi RQ1, RQ2, RQ3 bằng ngôn ngữ của pipeline;
- lập bản đồ mỗi RQ tới bảng/hình, cấu hình so sánh và artifact cần có;
- lập bảng hiện trạng module theo các cột giữ / sửa / viết mới / build lại, có đường dẫn bằng chứng;
- phân biệt giả thuyết, kết luận mô tả và bằng chứng benchmark;
- phát hiện một con số không đủ provenance vì thiếu run, config, checkpoint, corpus hoặc mẫu số.

## 2. Vị trí trong bức tranh lớn

Buổi 2 hoàn tất T1.2: nối câu hỏi nghiên cứu với bảng kết quả cần có và lập bản đồ module hiện trạng. RQ1–RQ3 sẽ được dùng lại ở Chặng 10; hôm nay chỉ thiết kế đường truy vết, chưa chạy benchmark và chưa kết luận BM25 hay Dense tốt hơn. Bảng module là output bắt buộc của cùng buổi, không phải phần tùy chọn. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §Research questions] [docs/learning/00-project-analysis.md §4. Hiện trạng module: giữ / sửa / viết mới / build lại]

## 3. Tiền đề

- Buổi 1: biết pipeline và phân biệt Retriever, Selection, Reader, Evaluation.
- Có thể đọc bảng Markdown và xác định một file JSON/JSONL là artifact.
- Ôn nhanh: **checkpoint** là trạng thái mô hình được lưu; **manifest** là file ghi cấu hình và nguồn artifact; **N_eval** là số query có gold evidence. Các phép tính chi tiết để buổi 18.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| RQ1 — scale | So BM25/Dense trên base, mid, large; xem Recall@1/5/10/20, MRR@10, EM/F1. | Đổi checkpoint theo từng tier rồi gọi đó là ảnh hưởng quy mô corpus. |
| RQ2 — bottleneck | Phân bố `execution_failure`, `retrieval_miss`, `selection_failure`, `reader_failure`, `correct`; thêm oracle. | Gọi retrieval là bottleneck trước khi có bảng bucket. |
| RQ3 — selection | So cặp bật/tắt reranker với cùng pool, `reader_k=5`, query và checkpoint. | Cho baseline tắt reranker đọc 20 passage, làm thay đổi ngân sách Reader. |
| Recall@k và Hit@k | V1.1 định nghĩa Recall@k là có ít nhất một gold trong top-k; GoldCoverage là metric phụ. | Dùng hai công thức khác nhau cùng tên Recall@k. |
| Oracle | Reader đọc toàn bộ gold passage để đo năng lực Reader trên gold. | Gọi oracle là trần tuyệt đối hoặc là end-to-end. |
| Kết luận mô tả | “Trong cấu hình/tập đánh giá này, A đạt F1 cao hơn B … điểm.” | Viết “Dense luôn tốt hơn” khi chưa có số liệu hoặc kiểm định. |

Ví dụ số chỉ để minh họa giao thức: nếu gold ở hạng 2 thì Recall@1 = 0, Recall@5 = 1 và MRR@10 = 0,5; đây không phải benchmark của project. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §3.6 Kiểm thử]

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | Lấy một câu hỏi và đi qua ba câu hỏi RQ. |
| 20–45 | Xem input | Đọc design RQ1–RQ3, bảng `00-project-analysis.md §2.2` và bảng hiện trạng module ở `§4`. |
| 45–75 | Thao tác từng lệnh | Tìm RQ, metric, bảng/hình, artifact trong tài liệu; đọc bảng hiện trạng module chỉ ở mức truy vết. |
| 75–100 | Kiểm tra output | Hoàn thành hai ma trận: RQ → bằng chứng và module → giữ/sửa/viết mới/build lại. |
| 100–120 | Tự giải thích + nhật ký | Giải thích vì sao một con số không có run/manifest chưa đủ bằng chứng. |

## 6. Input

- `docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md` — Research questions và Giới hạn.
- `docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md` — mục 3.3.
- `docs/learning/00-project-analysis.md` — bảng RQ → bảng/hình → bằng chứng.
- `docs/learning/00-project-analysis.md` — §4 Hiện trạng module: giữ / sửa / viết mới / build lại.
- `results/metrics_bm25_base_dev.json` — artifact cũ dùng để tập truy vết, không dùng làm kết luận.

## 7. Thao tác từng bước

1. Tìm phần RQ đã được kiểm chứng bằng PowerShell:

   ```powershell
   Select-String -LiteralPath 'docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md' -Pattern 'RQ1|RQ2|RQ3|Bảng 1|Bảng 2|Bảng 3|Hình 1|Hình 2'
   ```

   Kỳ vọng: thấy RQ và yêu cầu bảng/hình; nếu không thấy, kiểm tra encoding/đường dẫn, không tự viết lại nguồn.
2. Đọc đúng đoạn RQ1–RQ3 bằng `Get-Content` và số dòng tương ứng từ kết quả bước 1. Ghi cho mỗi RQ: câu hỏi, cấu hình, metric, artifact. Kỳ vọng: RQ1 có Bảng/Hình 1; RQ2 có Bảng/Hình 2; RQ3 có Bảng 3.
3. Tìm các quy tắc khóa công bằng trong spec:

   ```powershell
   Select-String -LiteralPath 'docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md' -Pattern 'retrieval_k|reader_k|Recall@k|selection_failure|final_correct|N_eval|n_latency|oracle'
   ```

   Kỳ vọng: thấy `retrieval_k=20`, `reader_k=5`, mẫu số và bucket; không tự đổi giá trị nếu gặp code cũ khác spec.
4. Đọc artifact cũ chỉ để luyện provenance:

   ```powershell
   Get-Content -LiteralPath 'results/metrics_bm25_base_dev.json' -Encoding UTF8
   ```

   Điền bốn câu: run_id là gì, split là gì, corpus_id là gì, latency đo trên môi trường nào. Kỳ vọng: nhận ra file ghi `bm25_base_dev`, `dev`, `base` và latency local theo nội dung file; không gọi đây là kết quả test hay nghiệm thu V1.1.
5. Tạo ma trận RQ cá nhân với các cột: `RQ`, `so sánh`, `metric`, `Bảng/Hình`, `artifact cần truy ngược`, `buổi`. Dùng phân tích A mục 2.2 để đối chiếu, nhưng tự mở lại design khi có khác biệt.
6. Lập thêm bảng hiện trạng module từ `docs/learning/00-project-analysis.md §4. Hiện trạng module: giữ / sửa / viết mới / build lại`, với các cột: `thành phần`, `bằng chứng file/test`, `phân loại`, `điều cần xác minh`. Kỳ vọng: có các hàng Data/Corpus, Splits, Schemas, Evaluation, BM25, Run infra, Dense/Reader/Reranker, Nhánh LLM/RAG và README đúng theo nguồn §4. Thêm một hàng `website/` để nối với quan sát Buổi 1, nhưng ghi rõ `CẦN XÁC MINH: §4 không có hàng website riêng`; không tự gán website vào một phân loại kỹ thuật.
7. Viết dòng output ngay dưới hai bảng: “Output buổi 2: ma trận RQ và bảng giữ/sửa/viết mới/build lại, mỗi hàng có ít nhất một đường dẫn nguồn; hàng `website/` phải ghi `CẦN XÁC MINH` vì không có hàng tương ứng trong `00-project-analysis §4`; mục chưa có file cũng ghi CẦN XÁC MINH.”
8. Câu kiểm tra sau khi lập bảng: “Nếu một module có file trong `src/` nhưng spec và artifact chưa khớp, ta xếp nó vào cột nào và bằng chứng nào cần thêm?” Đáp án gợi ý: phân loại theo hiện trạng trong bảng, thường là sửa/review hoặc build lại tùy bằng chứng; không gọi là đã đạt chỉ vì file tồn tại.
9. Vẽ một cặp RQ3: cùng retriever/checkpoint/tier/query/pool; nhánh reranker off lấy top-5, nhánh on rerank top-20 rồi giữ tối đa 5. Kỳ vọng: chỉ thay selection/rerank và latency tương ứng.
10. Ghi một câu kết luận mô tả có chỗ trống số liệu: “Trong cấu hình ___, trên tập ___, ___ đạt ___ cao hơn ___ ___ điểm.” Không điền số từ cảm giác.

## 8. Dự đoán trước, kiểm tra sau

1. RQ nào cần so sánh theo ba kích thước corpus? Vì sao không retrain theo tier?
2. Nếu oracle có F1 cao hơn end-to-end, điều đó cho biết gì và chưa cho biết gì?
3. Nếu file metrics có `run_id` nhưng thiếu corpus fingerprint, có đủ truy vết không?

Gợi ý sau kiểm tra: RQ1 là scale và giữ checkpoint giữa các tier; oracle cho biết năng lực Reader trên gold nhưng không chứng minh bottleneck nào; fingerprint thiếu thì chưa đủ truy vết nội dung artifact. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §Giới hạn và phạm vi kết luận]

## 9. Output cần có

- `docs/learning/lessons/chang-01/ghi-chu-buoi-02.md` — **chưa tạo sẵn**, sinh viên tự lưu ghi chú ở vị trí được giảng viên thống nhất.
- Ma trận RQ → cấu hình → metric → Bảng/Hình → artifact → buổi 30/31.
- Bảng module → giữ/sửa/viết mới/build lại, gồm hàng `website/` với nhãn `CẦN XÁC MINH` và nguồn kiểm kê §1.
- Một sơ đồ cặp RQ3 và một câu kết luận mô tả chưa điền số.

Không tạo metrics mới, không sửa `results/metrics_bm25_base_dev.json`, không chạy benchmark. Gate chỉ được xem là đủ khi cả ma trận RQ và bảng module đã có, kể cả hàng `website/` được đánh dấu chưa có nguồn phân loại trong §4.

## 10. Kiểm tra hiểu bài

1. RQ1 trả lời điều gì? **Đáp án gợi ý:** chất lượng retrieval và QA thay đổi thế nào khi corpus tăng, so BM25/Dense trên các tier. **Chấm:** nêu được scale, retriever, tier và metric.
2. Vì sao RQ1 tắt reranker? **Đáp án gợi ý:** giữ selection top-5 cố định để biến động phản ánh thay đổi retriever/corpus/Reader trong điều kiện đã định. **Chấm:** không nói “vì reranker không quan trọng”.
3. `selection_failure` khác `retrieval_miss` thế nào? **Đáp án gợi ý:** gold có trong pool 20 nhưng không vào reader input tối đa 5 so với gold không có trong pool. **Chấm:** nêu đúng hai tập passage.
4. Oracle có phải trần tuyệt đối không? **Đáp án gợi ý:** không; là mốc năng lực Reader trên gold passage. **Chấm:** dùng đúng nhãn `reader_only`.
5. Vì sao không được kết luận Dense thắng trước bảng? **Đáp án gợi ý:** design yêu cầu kết luận mô tả theo cấu hình/tập đã chạy, không mặc định ưu thế. **Chấm:** chỉ ra cần artifact/metric.
6. Một metric có `N_eval` nhưng không có `run_id` thì thiếu điều gì? **Đáp án gợi ý:** không truy ra cấu hình, checkpoint, corpus/run tạo ra số đó. **Chấm:** nêu ít nhất hai thành phần provenance.

## 11. Bài tự giải thích

Viết 180–250 từ hoặc vẽ sơ đồ trả lời: “Nếu RQ2 cho thấy nhiều `retrieval_miss` hơn khi corpus lớn, ta được phép nói gì?”

Rubric 4 điểm:

- 1 điểm: nhắc đúng cấu hình/tier/tập đánh giá;
- 1 điểm: phân biệt quan sát `retrieval_miss` với giả thuyết bottleneck;
- 1 điểm: giữ `N_eval` và truy được run/manifest;
- 1 điểm: dùng kết luận mô tả, nêu giới hạn không có kiểm định/seed bổ sung.

## 12. Gate của buổi

Qua buổi học khi sinh viên lập được ma trận RQ → bảng/hình/artifact và bảng module → giữ/sửa/viết mới/build lại; hàng `website/` phải có nhãn `CẦN XÁC MINH` thay vì đoán. Sinh viên cũng giải thích được vì sao không dùng test để chọn checkpoint hoặc kết luận Dense thắng. Gate Chặng 1 yêu cầu đủ cả hai bảng; task liên quan là T1.2. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 1 — Hiểu đề tài và khảo sát project]

## 13. Lỗi thường gặp & cách gỡ

- **Spec:** dùng tên bucket từ test cũ làm chuẩn → đối chiếu V1.1 D3 và mục 3.3; ghi lệch, không sửa spec.
- **Evidence:** lấy số từ file metrics cũ rồi gọi là benchmark chính thức → ghi rõ split/dev/local và thiếu điều kiện nào.
- **Concept:** gọi oracle là end-to-end → kiểm tra mode `reader_only` và toàn bộ gold passage.
- **Scope:** muốn chạy BM25 để “hiểu RQ” → buổi này chỉ đọc artifact; chạy máy không cần thiết và ngoài nguyên tắc học liệu.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T1.2
Hiểu được:
Chưa hiểu:
Lệnh chỉ-đọc đã chạy:
Output path / ma trận RQ:
Bảng module / trạng thái hàng website:
Lỗi gặp (code / data / spec / environment):
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc phần Chặng 2 của roadmap và chuẩn bị kiểm tra interpreter thật; không cài package, không chạy notebook, không tải dataset.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: sinh viên đã biết EM/F1 ở mức nào; buổi 18 mới tính tay theo fixture.
- CẦN XÁC MINH: nguồn paper/card/license GermanDPR và Wikipedia; chưa suy từ card GermanQuAD.
- CẦN XÁC MINH: có artifact/checkpoint ngoài repo hay không; nếu có, chủ project cung cấp đường dẫn và fingerprint.
- CẦN XÁC MINH: lịch Thứ 4 chỉ là gợi ý theo yêu cầu giao diện, không có trong roadmap gốc.
