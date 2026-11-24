# Phạm vi dự án (GĐ0)

Nguồn: `plan/01a_v1_phan_a_baseline.md`, `plan/README.md`, `docs/preprocessing_rules.md`.

## Bài toán
Hệ thống QA trích xuất tiếng Đức: tìm passage trong corpus xác định, dự đoán span trong passage và trả đáp án kèm bằng chứng (passage, offsets, điểm, thời gian, phiên bản).

```text
Câu hỏi → BM25 → Top-20 → Lấy top-5 → Reader → Đáp án + passage + offsets + thời gian
```

## Trong phạm vi
- V1 phần A (GĐ0–10): dữ liệu, corpus, BM25, reader, pipeline, đánh giá, FastAPI, React, Docker.
- V1 phần B (GĐ11–18): dense, hybrid RRF, cross-encoder, mining, robustness, tích hợp lại ứng dụng.

## Ngoài phạm vi
- V2 (GĐ19–28): router chọn ngân sách rerank, hướng nghiên cứu ứng viên, nằm ngoài đề bài.
- Không phải toàn bộ Wikipedia: corpus giới hạn (contexts GermanQuAD, cộng positive/hard-negative của GermanDPR).
- No-answer: không hỗ trợ ở V1; reader luôn trả một span.
- Câu hỏi mới không có gold không được chấm EM/F1.

## Hai quyết định đã chốt
1. **Tập test = GermanQuAD test.** GermanDPR chỉ góp đoạn văn (positive, hard negative) cho corpus; cờ `preprocess.include_dpr_qa: false` trong `configs/baseline.yaml`. Lý do: GermanDPR phát triển từ GermanQuAD nên không phải tập kiểm tra độc lập, và GermanDPR không có answer offset.
2. **Layout `src/germanqa/`.** Mã lõi là package `germanqa` (`data`, `retrieval`, `reader`, `evaluation`), cài bằng `pip install -e .`, dùng chung cho script, notebook, benchmark và API. Import dạng `from germanqa.… import …`, không dùng `src.`.

## Quy ước dữ liệu
- Offset: ký tự Python, nửa mở `[start, end)` trên `original_text`; chỉ đổi sang UTF-16 (`to_utf16_offset`) ở bước hiển thị (GĐ9).
- `original_text` không bao giờ sửa; `retrieval_text` là bản riêng cho tìm kiếm.
- Seed = 42, dev = 10% train chia theo nhóm (`src/germanqa/config.py`).
- Train dùng học; dev dùng chọn cấu hình; test chỉ dùng sau khi đóng băng quyết định.
- Quy tắc xử lý dữ liệu R1–R9: `docs/preprocessing_rules.md`.
