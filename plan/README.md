# Kế hoạch phát triển hệ thống QA trích xuất tiếng Đức

Cập nhật: 2026-10-14; điều chỉnh cấu trúc 2026-10-26 (V2 cũ gộp vào V1, V3 cũ thành V2).

Bộ tài liệu này chuyển nội dung đã thống nhất thành hai phiên bản kế tiếp nhau. Số giai đoạn 0–28 giữ nguyên để các tham chiếu chéo không bị vỡ. Đây là kế hoạch và tiêu chí nghiệm thu, không phải xác nhận các hạng mục đã được cài đặt hoặc đạt benchmark.

## Tài liệu

| Phiên bản | Giai đoạn | Mục tiêu | File |
|---|---|---|---|
| V1 — Phần A: Baseline | 0–10 | Baseline BM25 + reader và ứng dụng | [Chi tiết V1 phần A](01a_v1_phan_a_baseline.md) |
| V1 — Phần B: Cải tiến | 11–18 | Dense, hybrid RRF, cross-encoder; mining và robustness; tích hợp lại ứng dụng | [Chi tiết V1 phần B](01b_v1_phan_b_cai_tien.md) |
| V2 — Nghiên cứu | 19–28 | Xác nhận gap, kiểm chứng phương pháp, phân tích và công bố | [Chi tiết V2](02_v2_nghien_cuu.md) |

V1 là hệ thống hoàn chỉnh theo đề bài (BM25, dense, hybrid, rerank, đánh giá, web). V2 là hướng nghiên cứu ứng viên, nằm ngoài phạm vi đề bài.

Nguồn kế hoạch gốc: `D:/Project/2026-2027-HK1/nlp_v1/pic_plan/`. Các file này có thể được chuyển cùng nhau vào thư mục `plans/` của dự án; liên kết giữa chúng là liên kết tương đối.

## Pipeline

### V1 phần A — Baseline (giai đoạn 0–10)

```text
Câu hỏi → BM25 → Top-20 → Lấy top-5 → Reader → Đáp án + bằng chứng
```

Giai đoạn 6 về dense là nhánh đối chiếu tùy chọn từ kế hoạch gốc. V1 tối thiểu không phụ thuộc nhánh này. Nếu đã làm, tái sử dụng ở giai đoạn 12.

### V1 phần B — Hybrid và reranking (giai đoạn 11–18)

```text
Câu hỏi
   ├── BM25 ── ứng viên ──┐
   └── Dense ─ ứng viên ──┤
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

Hard-negative mining là nhánh huấn luyện. Phân tích nhóm và robustness là nhánh đánh giá. Hai phần này không chạy trong từng request.

### V2 — Hướng nghiên cứu ứng viên (giai đoạn 19–28)

```text
Câu hỏi → BM25 + Dense → RRF → Đặc trưng → Router chọn ngân sách
                                                  ↓
                                Bỏ rerank / rerank 10 / 20 / 50
                                                  ↓
                                        Top-5 → Reader
                                                  ↓
                                        Đáp án + bằng chứng
```

Router là phương án cần kiểm chứng, chưa phải novelty đã xác nhận. Bắt đầu V2 bằng pilot và literature review; chỉ triển khai phương pháp sau khi xác định vấn đề và khác biệt.

## Quy tắc chung

- Mỗi giai đoạn có mục tiêu, đầu vào, công việc, phương pháp, đầu ra và điều kiện chuyển tiếp.
- Tên file/module trong kế hoạch là đề xuất, không phải xác nhận chúng đã tồn tại.
- Top-k, tỷ lệ split, seed và siêu tham số là lựa chọn thí nghiệm; chọn bằng dev và ghi lại.
- Train dùng học; dev dùng chọn cấu hình; test dùng sau khi đóng băng quyết định.
- Nếu test đã ảnh hưởng đến thiết kế, ghi nhận và bổ sung kiểm tra độc lập cho tuyên bố xác nhận khi cần.
- Các so sánh chính giữ cùng corpus, passage IDs, reader, query set và evaluator, trừ thành phần được thay đổi có chủ đích.
- Giữ văn bản gốc cho reader/answer offsets; quản lý bản xử lý retrieval riêng.
- Corpus chứa văn bản phục vụ tìm kiếm khác với dùng câu hỏi/nhãn test để học. Công bố rõ protocol.
- Score model không tự là xác suất đáp án đúng. Câu hỏi mới không có gold không được chấm EM/F1.
- Những nguồn đã tra cứu trong cuộc trao đổi là điểm xuất phát; V2 vẫn cần rà soát literature đầy đủ trước khi nhận novelty.

## Mốc nghiệm thu

V1-M1…M4 thuộc phần A, V1-M5…M8 thuộc phần B.

| Mốc | Điều kiện |
|---|---|
| V1-M1 | Dữ liệu, split, corpus, qrels và offsets hoàn tất |
| V1-M2 | BM25 và reader chạy độc lập, metrics được kiểm tra |
| V1-M3 | Pipeline BM25 + reader có kết quả end-to-end và phân tích lỗi |
| V1-M4 | API, React và deploy hoạt động với artifacts đồng bộ |
| V1-M5 | Dense/hybrid chạy đúng và so sánh được với baseline |
| V1-M6 | Reranking được kiểm tra trên cùng candidate pool |
| V1-M7 | Chốt cấu hình; mining/robustness có kết quả nếu làm |
| V1-M8 | Ứng dụng cải tiến đã đo lại tài nguyên |
| V2-M1 | Có vấn đề thực nghiệm và gap ứng viên cụ thể |
| V2-M2 | Protocol, phương pháp và baseline được chốt |
| V2-M3 | Có thí nghiệm, độ bất định, ablation và phân tích |
| V2-M4 | Gói tái lập và bản thảo phản ánh đúng bằng chứng |

## Chuyển thành lịch công việc

Với từng checklist, điền người phụ trách, phụ thuộc, tài nguyên, ước lượng và bằng chứng nghiệm thu. Chỉ ước lượng lịch sau khi biết số người, GPU/RAM, thời gian Colab và độ lớn corpus.

| Công việc | Phụ trách | Phụ thuộc | Ước lượng | Trạng thái | Bằng chứng |
|---|---|---|---|---|---|
| Điền từ từng giai đoạn | Chưa phân công | Giai đoạn liên quan | Sau pilot | Chưa đối chiếu hiện trạng | Log/artifact/kết quả kiểm tra |

Theo dõi trạng thái thực hiện riêng với nội dung yêu cầu để tránh biến kế hoạch thành báo cáo kết quả.
