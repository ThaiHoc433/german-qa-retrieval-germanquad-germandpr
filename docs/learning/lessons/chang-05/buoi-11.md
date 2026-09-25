---
buoi: 11
chang: 5
ten_chang: Schema và dữ liệu V1.1
tuan_goi_y: 4
thu_goi_y: Thứ 4 (lịch gợi ý, có thể đổi)
tasks: [T5.0, T5.1]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 150 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.1, §1.2, §1.3 và §1.5
  - docs/learning/00-project-analysis.md §4. Hiện trạng module: giữ / sửa / viết mới / build lại
  - docs/learning/lessons/chang-05/the-task-T5.0-T5.1.md §2. Files và interface
trang_thai: chưa làm
---

# Buổi 11 — Validator V1.1, chunk và provenance

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- đọc và kiểm tra field bắt buộc của PassageRecord, QARecord, RetrievalTrainingRecord và PredictionRecord;
- phân biệt validator (kiểm hợp đồng) với producer (tạo dữ liệu);
- tính tay chunk 100 từ, overlap 20 và offset provenance;
- dự đoán lỗi fixture trước khi viết/sửa code.

## 2. Vị trí trong bức tranh lớn

Buổi 11 bắt đầu Chặng 5 bằng hai khối nội bộ: T5.0 validator trước, T5.1 chunk/provenance sau. Các schema này là đầu vào cho split, gold evidence, DPR groups và acceptance base. Giờ hướng dẫn chỉ chốt contract bằng fixture; sau gate có **nhánh thực thi T5.0/T5.1** để viết producer/validator và test trong repo, trước khi sang T5.2. Nhánh này không đổi spec V1.1 và không tính vào 150 phút hướng dẫn.

## 3. Tiền đề

- Chặng 3–4: biết QA offset, DPR group, D1–D6/P1–P4.
- Biết JSON dict/list và phép slice nửa mở.
- Ôn `passage_id` là hash text chuẩn hóa; provenance là thông tin truy về context/chunk nguồn.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ fixture | Sai lầm |
|---|---|---|
| PassageRecord | `text`, `source`, `title`, `parent_document_id`, `chunk_start_char/end_char`. | Dùng normalized_text thay text gốc. |
| QARecord | `gold_evidences`, `reference_answers`, `source_dataset/split`. | Xóa record khi gold evidence rỗng. |
| RetrievalTrainingRecord | `positive_texts/ids`, `hard_negative_groups`. | Dùng `answer_start` GermanDPR khi field không có. |
| PredictionRecord | `retrieval_k=20`, `reader_k=5`, answer fields all-or-none. | Bỏ `None` hoặc dùng chuỗi rỗng cho no-answer. |
| Chunk/provenance | Cửa sổ 100 từ, bước 80; `n=201` tạo `1 + ceil((201−100)/80) = 3` chunk; local offset = offset gốc − chunk start. | Search lại chuỗi rồi quên occurrence/provenance. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–25 | Giải thích | Đọc schema và phân biệt validator/producer. |
| 25–55 | Xem input | Đối chiếu §1 với `src/schemas.py`, `src/corpus.py`. |
| 55–95 | Fixture CPU | Viết expected trước: field thiếu, None, chunk/offset. |
| 95–125 | Kiểm tra | Lập bảng lỗi validator và tính chunk tay. |
| 125–150 | Tự giải thích | Nhật ký và gate T5.0/T5.1. |

## 6. Input

- Spec §1.1/1.2/1.3/1.5.
- `src/schemas.py`, `src/corpus.py` và test hiện có chỉ đọc.
- Fixture nhỏ do bài cung cấp, không phải dữ liệu thật: văn bản 201 từ (các cửa sổ bắt đầu ở 0, 80, 160), một QARecord có evidence lệch một ký tự, và một PredictionRecord no-answer.
- [Thẻ T5.0/T5.1](the-task-T5.0-T5.1.md): signature, tên test và fixture char-offset cụ thể. Nhánh thực thi đã có [log kiểm chứng](kiem-chung-T5.0-T5.1.md); trạng thái học của sinh viên vẫn chưa làm.

## 7. Thao tác từng bước

1. Dùng PowerShell đọc tên field hiện có:

   ```powershell
   Select-String -LiteralPath 'src/schemas.py' -Pattern 'QA_RECORD_FIELDS|PREDICTION_RECORD_FIELDS|def validate|def _field_errors'
   Select-String -LiteralPath 'src/corpus.py' -Pattern 'def chunk_text|parent_document_id|chunk_start_char|chunk_end_char'
   ```

   Kỳ vọng: thấy validator/code hiện hữu; không gọi đó là đủ V1.1.
2. Lập bảng validator trước producer: field bắt buộc, kiểu, null/no-answer, range `retrieval_k/reader_k`, evidence answer text/reference.
3. Fixture text rỗng/whitespace-only: 0 chunk; 1–100 từ: một chunk; 101 từ: hai chunk; 201 từ: ba chunk với cửa sổ từ bắt đầu 0, 80, 160. Công thức `1 + ceil((n−100)/80)` áp dụng khi `n>100`. Với text `" ".join(f"w{i:03d}" for i in range(201))`, các khoảng char là `[0,499)`, `[400,899)`, `[800,1004)`; kiểm chi tiết ở thẻ §3. [src/corpus.py §chunk_text]
4. Fixture QARecord có evidence offset sai một ký tự: truyền map `passage_texts` vào `validate_qa_record` để kiểm slice thật. Nếu không truyền map, validator chỉ kiểm hợp đồng nội bộ (kiểu/độ dài/reference), nên chưa thể phát hiện start/end cùng dịch một ký tự nhưng giữ độ dài. Evidence rỗng với reference rỗng vẫn hợp lệ theo D2. [src/schemas.py §validate_qa_record] [the-task-T5.0-T5.1.md §2. Files và interface]
5. Fixture PredictionRecord no-answer: năm field answer đều `None`; dự đoán chuỗi rỗng bị reject.
6. Ghi test-first table: input, expected valid/error, nguồn §1, task T5.0/T5.1. Chưa viết file test trong giờ hướng dẫn.
7. **Nhánh thực thi sau gate (không tính vào 150 phút):** theo [thẻ T5.0/T5.1](the-task-T5.0-T5.1.md), sửa `tests/test_schemas.py` trước `src/schemas.py`, rồi tạo `tests/test_corpus.py` trước phần provenance/title trong `src/corpus.py`. Test chunk đã có có thể xanh từ đầu; test hành vi mới phải quan sát RED → sửa tối thiểu → GREEN. Người triển khai đã ghi [log RED/GREEN](kiem-chung-T5.0-T5.1.md) cho nhánh này. Sinh viên đối chiếu log và fixture; không xóa code đang có để tạo RED giả, không tự đánh dấu đã học xong dựa vào kết quả người khác.

## 8. Dự đoán trước, kiểm tra sau

1. 101 từ tạo mấy chunk với window 100/step 80? **Hai chunk: cửa sổ bắt đầu ở 0 và 80.**
2. Evidence `answer_text` không nằm trong `reference_answers` xử lý thế nào? **Validator báo lỗi evidence; không tự thêm answer vào references.**
3. Một prediction có `answer_text=None` nhưng `answer_start=0` có hợp lệ không? **Không; các field answer phải cùng None hoặc cùng có giá trị.**

## 9. Output cần có

- Bảng contract/validator test-first cho T5.0.
- Bảng tính chunk/provenance cho T5.1, gồm trường hợp 201 từ → 3 chunk.
- Thẻ T5.0/T5.1 đã được xem và các quyết định interface được ghi; không đánh dấu đã triển khai chỉ vì có thẻ.
- Sau nhánh thực thi: validator/chunk producer, test fixture và log kết quả; trước nhánh này chỉ ghi bảng, không tuyên bố base đã build.

## 10. Kiểm tra hiểu bài

1. Validator khác producer? **Validator kiểm hợp đồng; producer tạo record.** Chấm: không trộn trách nhiệm.
2. Vì sao giữ gold rỗng? **Để bảo toàn dòng và loại khỏi N_eval.** Chấm: viện dẫn D2.
3. Vì sao offset local cần provenance? **Để map answer_start gốc vào chunk.** Chấm: nêu phép trừ.
4. No-answer dùng gì? **Cả năm field None.** Chấm: không dùng chuỗi rỗng.

## 11. Bài tự giải thích

Vẽ đường đi `raw context → chunk → PassageRecord → QARecord evidence`.

Rubric 4 điểm: đúng schema; đúng chunk; đúng offset; phân biệt code T5.0/T5.1 đã kiểm với base T5.8 chưa build.

## 12. Gate của buổi

Qua giờ hướng dẫn khi sinh viên viết được expected validator và tính đúng fixture chunk/offset. Chỉ sau khi nhánh thực thi T5.0/T5.1 có test fixture pass mới được bắt đầu T5.2; chưa có cơ sở tuyên bố base acceptance.

## 13. Lỗi thường gặp & cách gỡ

- **Schema:** lấy schema cũ làm frozen → đối chiếu §1.
- **Offset:** dùng offset context cho chunk → trừ chunk start.
- **Scope:** tạo code ngay → giữ test-first table.
- **Data:** bỏ gold rỗng → giữ record, ghi D2.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T5.0, T5.1
Lệnh chỉ-đọc đã chạy:
Output path:
Fixture/expected:
Nhánh thực thi đã chạy/chưa chạy:
Schema/provenance đã hiểu:
Điểm lệch code hiện tại:
Hiểu/chưa hiểu:
Lỗi:
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc D1 và quyết định `question_key`: V1.1 dùng NFC + lowercase + strip, chưa thêm collapse whitespace.

## 16. Điểm cần xác minh

- Signature và tên file test đã triển khai theo [thẻ T5.0/T5.1 §2](the-task-T5.0-T5.1.md#2-files-và-interface); bằng chứng ở [log kiểm chứng](kiem-chung-T5.0-T5.1.md). Trạng thái học của sinh viên vẫn chưa làm.
- CẦN XÁC MINH: caller ngoài `src/`, `tests/`, `notebooks/` — chủ project cung cấp hoặc người triển khai kiểm trước tích hợp. Entrypoint build cũ trong notebook còn dùng nguồn hard-negative, phải chuyển ở T5.8 trước khi chạy lại.
