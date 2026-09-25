---
tasks: [T5.6]
buoi: 15
trang_thai: chưa làm
---

# Kiểm chứng nhánh thực thi T5.6

Ngày lập báo cáo: 2026-09-25. Log kiểm thử code sau khi thẻ T5.6 (đã bổ sung 3 assertion theo yêu cầu duyệt) được chấp thuận; không xác nhận tiến độ học tập của sinh viên.

## 1. Phạm vi đã triển khai

| File | Thay đổi |
|---|---|
| `src/corpus_builder.py` | Mới. `normalize_title` (P4), `build_base_corpus` orchestration D6: duyệt GQ train/test (đọc `context`) và DPR positive train/test (đọc `positive_ctxs`), gọi `CorpusBuilder.add_text` hiện có cho mọi chunk, thu title trước dedup vào `base_title_exclusions`, trả `source_report` theo occurrence. Không đọc `answers`/`negative_ctxs`/`hard_negative_ctxs`, không nhận `SplitDecision`. |
| `tests/test_corpus_builder.py` | Mới, 43 ca: P4 normalize, nguồn D6 (bỏ hard-negative), title trước dedup, GQ/DPR thiếu title, context rỗng, priority GQ>DPR, title dài không cắt dòng, hai occurrence cùng hash khác provenance (cả hai thứ tự), title trùng key giữa hai nhóm không cộng thẳng, occurrence trùng lặp, lỗi schema/kiểu, rows rỗng dạng chuỗi, tính thuần, thứ tự sort passages. |

Phạm vi theo [the-task-T5.6.md §2. Files và interface đề nghị] [the-task-T5.6.md §6. Fixture có expected]. Hợp đồng theo [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý] và [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §Nhật ký quyết định V1.1] (P4).

## 2. RED rồi GREEN

```powershell
& '.venv/Scripts/python.exe' -B -m pytest tests/test_corpus_builder.py -q -p no:cacheprovider --tb=line
```

- RED trước khi tạo `src/corpus_builder.py`: `ModuleNotFoundError: No module named 'src.corpus_builder'` khi collect — đúng nguyên nhân thiếu module, không phải lỗi import khác trong file test.
- Sau khi viết `src/corpus_builder.py` lần đầu: **38 passed**. Vì 38/38 xanh ngay ở bản tối thiểu đầu tiên, đã chạy thêm 16 mutation (xóa/hoán từng nhánh kiểm tra một) trên bản sao ngoài repo để xác nhận test thật sự ràng buộc hành vi, không xanh giả.
  - 14/16 mutation bị bắt (fail rõ ràng khi lỗi bị cấy).
  - 1 mutation sống sót ("bỏ kiểm tra kiểu title DPR ở `_dpr_positive_ctxs`") nhưng xác nhận là dự phòng vô hại: `normalize_title` gọi ngay sau đó vẫn tự kiểm kiểu và raise, nên hành vi công khai không đổi — cùng dạng "nhánh không thể chạy tới" đã gặp ở T5.3.
  - 2 mutation lộ lỗ hổng thật trong **test**, không phải trong code: (a) đếm `n_unique_title_keys` theo nhóm dùng nhầm tập title toàn cục vẫn qua được test cũ vì fixture cũ có số cục bộ trùng số toàn cục; (b) không có test khẳng định `passages` trả về đã sort theo `passage_id`. Đã sửa test (thêm nhóm thứ ba có key riêng biệt; thêm test sort tường minh) rồi chạy lại mutation: **14/14 mutation liên quan bị bắt sau khi sửa test** (gộp 2 mutation mới xác nhận + 12 mutation cũ vẫn bắt được, xem chi tiết đếm bên dưới).
- Sau khi sửa test: **43 passed**, mutation lại **14/16** với đúng 2 lỗ hổng cũ đã đóng, chỉ còn 1 dự phòng vô hại và 1 do script mutation tự tham chiếu nhầm pattern không tồn tại (không phải phát hiện thật).

## 3. Hồi quy toàn bộ

```powershell
$taskTemp = Join-Path ([IO.Path]::GetTempPath()) ('german-qa-t56-' + [guid]::NewGuid().ToString('N'))
& '.venv/Scripts/python.exe' -B -m pytest tests -q -p no:cacheprovider --basetemp $taskTemp
```

**441 passed in 1.03s**, exit 0 = 398 nền (sau T5.0–T5.5) + 43 ca mới. Không tạo `.pytest-tmp/` trong repo.

## 4. Chạy API thật trên cache và đối chiếu bốn mốc

Chạy `build_base_corpus` thật (không mô phỏng) trên toàn bộ bốn split raw GermanQuAD/GermanDPR, snapshot như [doi-chieu-T5.6.md §1]. Kết quả:

| Mốc | Kỳ vọng (doi-chieu-T5.6.md) | Thực đo |
|---|---:|---:|
| `n_contexts_total` | 24.022 | 24.022 |
| `n_chunks_before_dedup_total` | 66.254 | 66.254 |
| `n_passages` | 7.520 | 7.520 |
| `n_title_exclusions` | 565 | 565 |
| `representative_sources` | `{germanquad: 7520, germandpr_positive: 0}` | khớp |

Đối chiếu bốn nhóm input_sources (rows/chunks/title keys theo GQ train, GQ test, DPR train, DPR test) khớp đúng bảng §2 của `doi-chieu-T5.6.md`, không lệch số nào.

**Hai hash byte-level cũng khớp tuyệt đối** với bản đối chiếu chỉ-đọc trước đó:
- `title_object_sha256 = c82921ab6a524979391398884b6144929445116bd4e1eb2b6422b6c7cb906113`
- `corpus_object_sha256 = 2995167bcab5159adcfea76de4c70f13247dba9a65bfe3e227df1a7f839c3967`

Đây là bằng chứng mạnh nhất có thể có ở mức không build artifact: implementation thật cho đúng cùng kết quả byte-for-byte với script đối chiếu độc lập trên toàn bộ 24.022 context thật, không chỉ trên fixture nhỏ.

Kiểm mẫu 500 passage đầu bằng `validate_passage_record` (chỉ kiểm hợp đồng nội bộ, không truyền `parent_text` vì producer T5.6 không giữ context gốc để so lại hàng loạt): **0 lỗi**.

## 5. Chưa ghi artifact

`data/processed/v1.1/` chưa tồn tại sau khi chạy API thật — đúng thẻ, vì entrypoint ghi file thuộc T5.8. `git status` chỉ có hai file mới `src/corpus_builder.py`, `tests/test_corpus_builder.py`; không có thay đổi nào khác trong `src/`, `tests/`, `notebooks/`, `docs/superpowers/`, `.specify/`, `data/`.

## 6. Hệ quả và việc còn lại

1. `build_base_corpus` không đọc `SplitDecision`/answers nên hoàn toàn không có khái niệm D1/unmappable — bất kỳ context GQ/DPR positive nào cũng vào base, kể cả dòng sẽ bị loại D1 hoặc DPR không map được positive ở producer khác. Đây là thiết kế đúng theo D6, không phải thiếu sót.
2. Chuỗi distractor Wikipedia, lịch R, prefix mid/large vẫn thuộc T5.7 — chưa làm, có thẻ riêng.
3. `notebooks/01_build_corpus_base.py` vẫn dùng entrypoint/API cũ; chuyển caller sang `build_base_corpus`+`build_qa_records`+`build_retrieval_training_records` thuộc T5.8, chưa sửa/chạy notebook này.
4. CẦN XÁC MINH: caller ngoài `src/`, `tests/`, `notebooks/` — người triển khai kiểm khi tích hợp T5.8.
5. CẦN XÁC MINH: title DPR bằng toàn context (881 trường hợp đã ghi ở `doi-chieu-T5.6.md`) — vẫn dùng nguyên theo V1.1, chưa có amendment.

Code chưa commit tại thời điểm ghi log này.
