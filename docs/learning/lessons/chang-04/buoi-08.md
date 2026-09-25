---
buoi: 8
chang: 4
ten_chang: Data audit và quyết định spec
tuan_goi_y: 3
thu_goi_y: Thứ 4 (lịch gợi ý, có thể đổi)
tasks: [T4.1]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 4 — Audit và đối chiếu V1.1
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T4.1
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2. Module: Data / Splits / Corpus và §2.3 Quy tắc xử lý
  - docs/learning/00-project-analysis.md §5.1 Cache đã đọc bằng đường dẫn local, không gọi Hub
  - docs/learning/00-project-analysis.md §5.2 Bộ processed hiện tại không phải dữ liệu V1.1 đã nghiệm thu
trang_thai: chưa làm
---

# Buổi 8 — Audit rows, keys, độ dài và phân phối

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- lập bảng audit riêng cho bốn split Parquet, không trộn số dòng với số key;
- đo độ dài context/answer theo field thật và ghi đơn vị đo;
- mô tả phân phối positive/negative của GermanDPR mà không gọi đó là corpus V1.1;
- truy mỗi con số về file, revision và lệnh đã dùng.

## 2. Vị trí trong bức tranh lớn

Buổi 8 mở T4.1. Audit cung cấp bằng chứng để Buổi 9 kiểm overlap/split và Buổi 10 lập requirement traceability matrix. Đây là mô tả dữ liệu đang quan sát; không build lại, không sửa spec, không coi số cũ là nghiệm thu.

## 3. Tiền đề

- Buổi 5: biết rows/key và bốn Parquet cache.
- Buổi 6: biết answer offset và reference.
- Buổi 7: biết DPR group và hard negative training-only.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| Row count | Parquet metadata GermanQuAD test là 2.204 dòng. | Dùng 2.203 key duy nhất thay cho 2.204 dòng. |
| Key count | GermanQuAD test có 2.203 key theo audit. | Cộng key giữa dataset rồi gọi là số record cần loại. |
| Độ dài context | Đo số ký tự hoặc số từ, phải ghi rõ đơn vị. | So số ký tự của một file với số từ của file khác. |
| Answer length | Đo từng reference answer, không chỉ answer đầu. | Dùng `answer_start` như độ dài. |
| Phân phối positive/negative | GermanDPR raw nhóm text theo positive/negative/hard-negative context. | Gọi hard negative là nguồn corpus đánh giá. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | Tách row/key/length bằng một bảng nhỏ. |
| 20–45 | Xem input | Đọc schema bốn Parquet và §5.2. |
| 45–80 | Thao tác từng lệnh | Đọc metadata, tính length và đếm nhóm. |
| 80–105 | Kiểm tra output | Hoàn thiện audit table kèm nguồn/revision/đơn vị. |
| 105–120 | Tự giải thích + nhật ký | Nói vì sao audit không phải nghiệm thu. |

## 6. Input

- Bốn Parquet cache ở `docs/learning/00-project-analysis.md §5.1`.
- `results/data_audit_report.json` và `data/processed/split_report.json` chỉ để đối chiếu, không ghi đè.
- Spec §2 và D1–D2.

## 7. Thao tác từng bước

1. Kiểm tra tồn tại bốn cache bằng `Test-Path`; ghi revision từ đường dẫn.
2. Đọc `metadata.num_rows` và schema từng file bằng `pyarrow.parquet.ParquetFile`; kết quả kỳ vọng là `11518`, `2204`, `9275`, `1025`.
3. Chạy lệnh chỉ-đọc dưới đây để tính min/median/max **số từ** của context và **số ký tự** của mọi reference answer. **Lưu ý khi copy:** khối here-string PowerShell phải có `@'` và `'@` ở cột 0; bản Markdown render có thể thụt mã trong danh sách, nên khi copy từ file nguồn hãy bỏ ba khoảng trắng đầu dòng của khối lệnh:

   ```powershell
   @'
   import statistics, pyarrow.parquet as pq
   files = {
       "gq-train": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/train/0000.parquet",
       "gq-test": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/test/0000.parquet",
   }
   for name, path in files.items():
       context_words, answer_chars = [], []
       for batch in pq.ParquetFile(path).iter_batches(columns=["context", "answers"], batch_size=4096):
           for row in batch.to_pylist():
               context_words.append(len(row["context"].split()))
               answer_chars.extend(len(x) for x in row["answers"]["text"])
       print(name, "context_words", min(context_words), statistics.median(context_words), max(context_words), "answer_chars", min(answer_chars), statistics.median(answer_chars), max(answer_chars), "n_answer", len(answer_chars))
   '@ | & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B
   ```

   Kỳ vọng: `gq-train` context `54/175/369`, answer `1/33/972`, `n_answer=11518`; `gq-test` context `66/268/1513`, answer `1/23/863`, `n_answer=6536`. Không gộp train/test vì phân phối khác nhau.
4. Chạy lệnh GermanDPR chỉ đọc để đếm số context trong `positive_ctxs`, `negative_ctxs`, `hard_negative_ctxs`. Cùng lưu ý cột 0 cho `@'`/`'@` như bước 3:

   ```powershell
   @'
   import pyarrow.parquet as pq
   from collections import Counter
   files = {"gdpr-train": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/train/0000.parquet", "gdpr-test": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/test/0000.parquet"}
   for name, path in files.items():
       pos, neg, hard = [], [], []
       for batch in pq.ParquetFile(path).iter_batches(columns=["positive_ctxs", "negative_ctxs", "hard_negative_ctxs"], batch_size=4096):
           for row in batch.to_pylist():
               pos.append(len(row["positive_ctxs"]["text"]) if row["positive_ctxs"] else 0); neg.append(len(row["negative_ctxs"]["text"]) if row["negative_ctxs"] else 0); hard.append(len(row["hard_negative_ctxs"]["text"]) if row["hard_negative_ctxs"] else 0)
       print(name, "positive", Counter(pos), "negative", Counter(neg), "hard_negative", Counter(hard))
   '@ | & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B
   ```

   Kỳ vọng: positive là `1` ở mọi dòng; negative là `0` ở mọi dòng; train hard-negative `{3: 9267, 1: 6, 2: 2}`; test hard-negative `{3: 1025}`. Đây là context count, không phải chunk count.
5. Đặt bảng audit: dataset, split gốc, revision, rows, unique keys, context length unit, answer length unit, positive/negative observation, source path.
6. Đối chiếu processed GermanQuAD dev (`1031` dòng, `1029` key) và ghi “trùng key nội bộ, xử lý ở Buổi 9”, không gọi ngay là leakage hay bất thường.

## 8. Dự đoán trước, kiểm tra sau

1. Split nào có row count khác unique key? **Đáp án gợi ý:** GQ train `11518/11477`, GQ test `2204/2203`, DPR train `9275/9254`; DPR test `1025/1025`. **Chấm:** đọc đúng từng cột, không cộng.
2. Vì sao hai người có thể báo độ dài khác nhưng đều đúng? **Đáp án gợi ý:** một người dùng ký tự, người kia dùng từ; phải ghi đơn vị. **Chấm:** chỉ ra output kỳ vọng của bước 3.
3. Có được cộng hard-negative context vào corpus size không? **Đáp án gợi ý:** không; D6 tách training-only. **Chấm:** phân biệt context count với corpus/chunk count.

## 9. Output cần có

- Audit table bốn split với nguồn/revision và đơn vị đo.
- Ghi chú phân phối GermanDPR raw và giới hạn hard negative.
- Danh sách bất thường chuyển sang Buổi 9.

## 10. Kiểm tra hiểu bài

1. Row/key khác nhau thế nào? **Đáp án gợi ý:** row là dòng; key là định danh sau chuẩn hóa. **Chấm:** nêu được các cặp `11518/11477` hoặc `2204/2203`.
2. Vì sao phải ghi đơn vị length? **Đáp án gợi ý:** ký tự và từ không thay thế nhau. **Chấm:** nêu đúng context dùng từ, answer dùng ký tự trong bài.
3. Hard negative có vào corpus đánh giá không? **Đáp án gợi ý:** không chỉ vì là hard negative. **Chấm:** viện dẫn D6 và phân phối bước 4.
4. Audit số cũ có phải acceptance không? **Đáp án gợi ý:** không, cần spec/revision/gate. **Chấm:** phân biệt quan sát, bất thường và nghiệm thu.

## 11. Bài tự giải thích

Giải thích một bảng audit cho người chưa biết vì sao `2.204` và `2.203` đều đúng trong hai cột khác nhau.

Rubric 4 điểm: đúng row/key; đúng đơn vị; có nguồn/revision; nêu giới hạn.

## 12. Gate của buổi

Qua buổi khi audit có đủ bốn split, nguồn/revision, đơn vị đo và danh sách bất thường; không cộng máy móc các key.

## 13. Lỗi thường gặp & cách gỡ

- **Data:** lấy số processed thay raw → ghi hai bảng riêng.
- **Metric:** không ghi đơn vị length → chạy lại phép đo nhỏ.
- **Spec:** coi hard negative là corpus → đọc D6.
- **Scope:** sửa dữ liệu để số đẹp → giữ input, ghi bất thường.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T4.1
Dataset/revision:
Rows/unique keys:
Đơn vị length:
Phân phối positive/negative:
Bất thường chuyển Buổi 9:
Hiểu/chưa hiểu:
Lệnh và output path:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc fixture 6 dòng T5.2 và chuẩn bị dự đoán giữ/loại/split.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: revision chính thức cho audit mới.
- CẦN XÁC MINH: quy ước length dùng trong báo cáo cuối.
