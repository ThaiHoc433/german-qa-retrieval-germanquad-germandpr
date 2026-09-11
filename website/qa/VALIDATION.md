# Website validation — 2026-09-11

## Nguồn và nội dung

Đã đọc toàn bộ Module Implementation Specs V1.1 trong workspace (635 dòng xấp xỉ,
bao gồm decision log đến bảng nghiệm thu), file không có hậu tố `(3)`.
SHA-256 của file nguồn đã dùng:

`84CD545D30A97660122310B3D7E657B170BD90EA089109E89886027988A7BE35`

- Base ≈7,5K ghi rõ dự kiến; mid tổng 50.000, large tổng 150.000 là mốc thiết kế.
- Không có các quy mô corpus cũ trong HTML.
- Hard negatives chỉ là nguồn training; corpus được xác định theo D6, P4.
- Test gốc giữ riêng; N_eval lấy theo gold_evidences hợp lệ, n_latency riêng.
- Recall@k = Hit@k; ví dụ rank 2 và EM/F1 được ghi là minh họa.
- Pipeline đúng Retriever → Selection/Reranker → Reader; reader_k=5 cố định,
  thiếu candidate không pad; oracle đọc toàn bộ gold.
- selection_failure dùng được cho cả bật/tắt reranker; correct yêu cầu có gold
  trong reader input và EM=1; final_correct độc lập; manual review không là bucket 6.
- Reader gbert-base, dedup span lấy max, softmax trong passage; không tuyên bố calibrated.
- Trạng thái roadmap là kế hoạch; audit cũ không được dùng để xác nhận triển khai V1.1.
- Không có kết quả benchmark giả, không gọi hệ thống là chatbot tạo sinh.
- Source spec không bị sửa (git diff rỗng cho file nguồn).

## Kiểm tra trình duyệt

Chạy Chromium headless với Playwright đã cài trong môi trường, mở trực tiếp file://.
Xem `report.json` cho kết quả kiểm tra máy đọc được và ảnh PNG cùng thư mục.

- 375 × 1000, 768 × 1000, 1440 × 1000.
- Không tràn ngang ở trạng thái mặc định và khi mở toàn bộ details.
- Không có anchor thiếu đích hoặc ID trùng; không có lỗi JavaScript.
- Walkthrough: cả 5 nút chọn bước, Next, Previous, vô hiệu hóa đúng ở hai đầu.
- 22 details/summary mở bằng Enter và đóng bằng Space.
- 6 mục menu có active state theo cuộn; nút quay đầu trang hoạt động.
- JavaScript tắt: vẫn có cả 5 bước và native accordion.
- Phóng chữ gốc lên 200% tại 768px: đã sửa menu wrap và overflow trường kỹ thuật.
- Reduced motion bật trong QA; CSS loại smooth scroll/transition khi được yêu cầu.
- Đã xem screenshot desktop, tablet và mobile để đối chiếu bố cục.

Đây là kiểm tra chức năng và bố cục trình duyệt, không phải chứng nhận WCAG hay
đánh giá chất lượng model QA. Chưa deploy hoặc kiểm thử trên URL GitHub Pages thật.

## Chạy lại QA (tùy chọn cho người phát triển)

Website không cần Playwright để xem. Nếu muốn chạy lại kiểm tra, cài Playwright
trong môi trường kiểm thử riêng và chạy:

```sh
node website/qa/check.cjs /absolute/path/to/playwright
```

Script tạo lại screenshots và report.json, chỉ thao tác website local.
