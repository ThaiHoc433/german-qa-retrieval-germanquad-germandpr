---
buoi: 16
chang: 5
ten_chang: Schema và dữ liệu V1.1
tuan_goi_y: 6
thu_goi_y: Thứ 2 (lịch gợi ý, có thể đổi)
tasks: [T5.8]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 150 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T5.8 — Phạm vi báo cáo và nghiệm thu
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §9 Bảng tiêu chí nghiệm thu tổng hợp
  - docs/learning/00-project-analysis.md §5.2 và §6. Hợp đồng dữ liệu và giao thức cần giữ
trang_thai: chưa làm
---

# Buổi 16 — Entrypoint, reports và gate base V1.1

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- lập checklist acceptance base theo schema, offset, source, split và coverage;
- phân biệt namespace mới `data/processed/v1.1/` với artifact cũ cần giữ;
- nối report, manifest và checker với từng gate;
- biết khi nào phải dừng báo vấn đề thay vì sửa frozen spec hoặc hứa mid/large.

## 2. Vị trí trong bức tranh lớn

Buổi 16 chốt T5.8 trong giờ hướng dẫn và mở **nhánh thực thi build/acceptance** sau đó. Giờ học không chạy build thật; nhánh thực thi phải chạy producer/checker theo thứ tự T5.0 → T5.1/T5.2 → T5.3/T5.4 → T5.5 → T5.6/T5.7 → T5.8, ghi artifact vào `data/processed/v1.1/` và chỉ cho phép T6 khi base gate đủ bằng chứng.

## 3. Tiền đề

- Buổi 11–15: validators, split, gold, DPR groups, D6/P4.
- Đọc namespace/report trong catalog T5.8.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ | Sai lầm |
|---|---|---|
| Base acceptance | Schema/offset/source/split/coverage theo từng file. | Chỉ xem tổng coverage gộp. |
| Namespace | `data/processed/v1.1/` tách artifact mới. | Ghi đè processed cũ. |
| Coverage | ≥98% trên từng file theo gate. | Dùng 98% gộp. |
| Report | split/chunk/gold/mapping/manifest ghi trạng thái. | Tạo report không có revision. |
| Stop condition | Không đạt coverage/offset → dừng báo vấn đề. | Đổi overlap để làm score đẹp. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–25 | Giải thích | Base gate và namespace. |
| 25–50 | Xem input | Catalog T5.8, spec §9, artifact cũ. |
| 50–95 | Checklist fixture | Ghép từng gate với report/test. |
| 95–130 | Kiểm tra | Đọc entrypoint/checker hiện có, đánh dấu thiếu. |
| 130–150 | Tự giải thích | Quyết định stop/continue và nhật ký. |

## 6. Input

- `notebooks/01_build_corpus_base.py`, `data/processed/` hiện có chỉ đọc.
- Catalog T5.8 và spec §9.
- Reports cũ: `split_report.json`, `chunk_coverage_report.json`, `corpus_manifest.json`.

## 7. Thao tác từng bước

1. Đọc entrypoint/report names:

   ```powershell
   Select-String -LiteralPath 'notebooks/01_build_corpus_base.py' -Pattern 'OUT_DIR|build_qa_records|build_retrieval_training_records|main'
   Get-ChildItem -LiteralPath 'data/processed' -File | Select-Object Name,Length
   ```

   Kỳ vọng: thấy producer/namespace cũ và artifact hiện hữu; không chạy notebook.
2. Lập acceptance matrix: PassageRecord fields, QA evidence, TrainingRecord groups, split D1, source D6, title P4, coverage từng file, gold rỗng.
3. Gắn report bắt buộc: `corpus_manifest`, `split_report`, `chunk_coverage_report`, `ambiguous_offset_remap_report`, `germandpr_chunk_mapping_report`.
4. Ghi namespace đề xuất `data/processed/v1.1/`; artifact cũ giữ nguyên để audit.
5. Viết stop/continue table: thiếu offset → stop; coverage <98% từng file → stop; fixture pass → được vào nhánh thực thi, chưa được chuyển T6 và chưa chứng nhận mid/large.
6. **Nhánh thực thi sau gate (không tính vào 150 phút):** dùng entrypoint hiện tại `notebooks/01_build_corpus_base.py` sau khi các task trước đã được triển khai, chạy producer/checker theo thứ tự phụ thuộc, ghi `data/processed/v1.1/`, các report bắt buộc và manifest có revision; nếu checker fail thì dừng và ghi nguyên nhân, không sửa frozen spec để qua.
7. Kiểm tra các số audit không tự trở thành acceptance; ghi revision và lệnh cần chạy sau khi producer được duyệt. Lệnh chưa có entrypoint thật phải ghi **LỆNH MẪU – chưa xác minh**.

## 8. Dự đoán trước, kiểm tra sau

1. Coverage gộp 98% nhưng một file 95% có qua gate không? **Không; gate xét từng file nên phải dừng và báo file dưới 98%.**
2. Base fixture pass có chứng minh mid/large thật đã build không? **Không; chỉ nhánh thực thi với producer/checker và report hợp lệ mới cho biết base thật.**
3. Producer cũ ghi vào `data/processed/` thì có nên ghi đè không? **Không; giữ artifact cũ và ghi build mới vào `data/processed/v1.1/`.**

## 9. Output cần có

- Base acceptance matrix và stop/continue table.
- Namespace/report checklist.
- Danh sách task/interface cần chốt trước implementation.
- Sau nhánh thực thi: `data/processed/v1.1/`, checker log, manifest và các report theo từng file; nếu chưa chạy thì ghi rõ chưa làm.

## 10. Kiểm tra hiểu bài

1. Coverage tính thế nào? **Theo từng file.** Chấm: không dùng tổng gộp.
2. Artifact cũ xử lý sao? **Giữ nguyên để audit.** Chấm: dùng namespace mới.
3. Khi offset sai? **Dừng báo vấn đề.** Chấm: không đổi frozen spec.
4. Fixture pass chứng minh gì? **Logic nhỏ đúng.** Chấm: chưa phải mid/large.
5. Report cần revision vì sao? **Truy nguồn dữ liệu.** Chấm: không dùng số mồ côi.

## 11. Bài tự giải thích

Trình bày quyết định stop khi coverage một file dưới 98% và nêu report cần cập nhật.

Rubric 4 điểm: đúng gate; đúng report; giữ namespace; không sửa spec để qua.

## 12. Gate của buổi

Qua giờ hướng dẫn khi acceptance matrix đủ, fixture D6/P4/split/gold đã có expected, namespace/report/interface được ghi rõ. Qua Chặng 5 chỉ khi nhánh thực thi đã chạy producer/checker hợp lệ, có `data/processed/v1.1/`, report theo từng file và revision; nếu chưa có thì gate vẫn chưa làm.

## 13. Lỗi thường gặp & cách gỡ

- **Acceptance:** dùng coverage gộp → tách từng file.
- **Artifact:** ghi đè cũ → dùng namespace v1.1.
- **Spec:** đổi overlap để qua → dừng và báo.
- **Scope:** chạy notebook ghi file → chỉ đọc source.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T5.8
Lệnh chỉ-đọc đã chạy:
Output path:
Acceptance matrix:
Namespace:
Reports:
Stop/continue decision:
Nhánh thực thi producer/checker đã chạy/chưa chạy:
Hiểu/chưa hiểu:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc T6.1 về run_id, fingerprint và manifest; chuẩn bị fixture bản sao gold sửa một byte.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: interface entrypoint/checker và namespace cuối.
- CẦN XÁC MINH: revision dữ liệu dùng build base.
- CẦN XÁC MINH: coverage report theo từng file trước gate.
