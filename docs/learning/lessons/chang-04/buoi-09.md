---
buoi: 9
chang: 4
ten_chang: Data audit và quyết định spec
tuan_goi_y: 3
thu_goi_y: Thứ 6 (lịch gợi ý, có thể đổi)
tasks: [T4.1]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 4 — Audit và đối chiếu V1.1
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T5.2 — Thẻ mẫu: Gán split theo D1
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §Nhật ký quyết định V1.1 và §2.3.e
  - docs/learning/00-project-analysis.md §5.2 Bộ processed hiện tại không phải dữ liệu V1.1 đã nghiệm thu
trang_thai: chưa làm
---

# Buổi 9 — Overlap question_key và dự đoán leakage theo dòng

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- tính giao `question_key` giữa bốn split mà không cộng các giao nhau thành một số tổng;
- phân biệt overlap key với số dòng bị loại;
- dự đoán fixture 6 dòng sẽ giữ/loại/split thế nào theo D1;
- giải thích vì sao `query_id` giữ split gốc khi `source_split` thay đổi.

## 2. Vị trí trong bức tranh lớn

Buổi 9 hoàn thiện phần overlap của T4.1 và làm quen fixture T5.2 trước khi Chặng 5 viết code. Hai dữ kiện từ Chặng 3 — processed dev có 1.031 dòng/1.029 key và các row/key raw — được đưa vào danh sách audit, không bị biến thành kết luận leakage.

## 3. Tiền đề

- Buổi 8: audit rows/keys/length đã có nguồn.
- Buổi 5: biết chuẩn hóa `question_key`.
- Đọc D1 và fixture 6 dòng trong catalog.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| Overlap key | Một key xuất hiện ở hai split/dataset. | Gọi mọi overlap là cùng số dòng. |
| D1 | Test giữ đúng dòng test gốc; train trùng test bị loại khỏi train/dev. | Chuyển dòng train sang test của dataset khác. |
| Split gốc/gán | `gq-train-51870` có thể được gán `source_split=dev`. | Đổi `query_id` theo split mới. |
| Fixture 6 dòng | A/B/C gq train, D gq test, B gdpr train, A gdpr test. | Tự thay fixture bằng dữ liệu thật rồi kết luận. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | Vẽ hai tập key và giao của chúng. |
| 20–45 | Xem input | Đọc audit Buổi 8, D1 và fixture catalog. |
| 45–80 | Thao tác từng lệnh | Tính giao key, viết bảng quyết định fixture. |
| 80–105 | Kiểm tra output | Đối chiếu dự đoán với quy tắc, không sửa code. |
| 105–120 | Tự giải thích + nhật ký | Phân biệt overlap key, removed row và leakage. |

## 6. Input

- Bảng audit Buổi 8 và bốn split raw.
- Fixture 6 dòng trong catalog T5.2.
- `src/splits.py` chỉ đọc để xem `question_key` và assignment hiện tại.

## 7. Thao tác từng bước

1. Tạo tập key theo từng split từ batch/fixture; không gộp ngay.
2. Chạy lệnh chỉ-đọc tính giao key và số dòng ở hai phía. **Lưu ý khi copy:** khối here-string PowerShell phải có `@'` và `'@` ở cột 0; nếu copy từ file nguồn Markdown, bỏ ba khoảng trắng đầu dòng của khối lệnh:

   ```powershell
   @'
   import pyarrow.parquet as pq, unicodedata
   paths = {"gq-train": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/train/0000.parquet", "gq-test": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/test/0000.parquet", "gdpr-train": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/train/0000.parquet", "gdpr-test": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/test/0000.parquet"}
   def load(path):
       values=[]
       for batch in pq.ParquetFile(path).iter_batches(columns=["question"], batch_size=4096): values += [unicodedata.normalize("NFC", x).lower().strip() for x in batch.column(0).to_pylist()]
       return values
   data={name:load(path) for name,path in paths.items()}
   for a,b in [("gq-train","gdpr-test"),("gq-train","gdpr-train"),("gdpr-train","gdpr-test"),("gq-train","gq-test"),("gq-test","gdpr-train"),("gq-test","gdpr-test")]:
       sa,sb=set(data[a]),set(data[b]); print(a,b,"key",len(sa&sb),"rows_a",sum(x in sb for x in data[a]),"rows_b",sum(x in sa for x in data[b]))
   '@ | & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B
   ```

   Kỳ vọng: GQ train×DPR test `1025 / 1026 / 1025`; GQ train×DPR train `9254 / 9289 / 9275`; DPR train×DPR test `1 / 1 / 1`; các cặp còn lại `0 / 0 / 0`. Thứ tự là `key / rows_a / rows_b`.
3. Ghi mốc catalog: không cộng `1025` và `1` thành tổng bắt buộc; mỗi cặp có dataset, revision và dòng riêng.
4. Với processed GermanQuAD dev, chạy lệnh tìm key trùng:

   ```powershell
   & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B -c "import json,unicodedata,collections; p='data/processed/qa_germanquad_dev.jsonl'; d=collections.defaultdict(list); [d[unicodedata.normalize('NFC',r['question']).lower().strip()].append(r['query_id']) for r in map(json.loads,open(p,encoding='utf-8'))]; print({k:v for k,v in d.items() if len(v)>1})"
   ```

   Kỳ vọng: hai nhóm `['gq-dev-56945','gq-dev-56948']` và `['gq-dev-47465','gq-dev-67922']`. Kiểm tra raw GQ train cho thấy mỗi key có đúng 2 dòng, không xuất hiện ở test; chúng cùng đi dev theo D1, nên ghi **trùng key nội bộ, không phải leakage**.
5. Viết bảng fixture 6 dòng: gq train A/B/C, gq test D, gdpr train B, gdpr test A. Dự đoán: gq A loại, gdpr A giữ test, gq/gdpr B cùng split, gq D giữ test.
6. Ghi `query_id`/`source_split` cho trường hợp gq train A bị loại hoặc train được gán dev; không đổi mã nguồn gốc.
7. Kiểm tra tính ổn định: cùng seed/config phải ra cùng quyết định; đây là tiêu chí học, không chạy test triển khai trong buổi.

## 8. Dự đoán trước, kiểm tra sau

1. Vì sao `1025` key GQ train×DPR test tương ứng `1026` dòng phía GQ train?
2. Hai key trùng trong processed dev có phải leakage không? **Đáp án gợi ý:** không theo kiểm tra raw; mỗi key có 2 dòng train và 0 test, cùng gán dev. **Chấm:** nêu đúng query_id hoặc điều kiện D1.
3. Dòng gq train A trùng gdpr test A được giữ ở đâu?
4. Nếu query_id bắt đầu `gq-train` nhưng source_split là dev, có mâu thuẫn không?

Gợi ý bổ sung: câu 1 có `1026` dòng nhưng `1025` key vì đúng một key xuất hiện hai dòng trong GQ train; câu 3 giữ dòng test gốc ở phía DPR và loại dòng train trùng theo D1; câu 4 không mâu thuẫn vì `query_id` giữ split gốc còn `source_split` là split được gán.

## 9. Output cần có

- Bảng overlap key theo cặp dataset/split, tách key và row.
- Ghi chú hai key trùng processed dev, bốn `query_id`, và kết luận tạm thời “trùng nội bộ, không phải leakage theo kiểm tra raw”.
- Bảng dự đoán fixture 6 dòng và lý do từng dòng.
- Danh sách câu hỏi cần kiểm ở Buổi 10.

## 10. Kiểm tra hiểu bài

1. Overlap key có bằng removed rows không? **Đáp án gợi ý:** không; phải xem dataset, dòng và quyết định D1. **Chấm:** không cộng key máy móc.
2. D1 giữ test nào? **Đáp án gợi ý:** đúng dòng test gốc của từng dataset. **Chấm:** không chuyển train sang test dataset kia.
3. Vì sao query_id giữ split gốc? **Đáp án gợi ý:** để truy nguồn ổn định. **Chấm:** phân biệt query_id với source_split.
4. Hai key processed dev được xử lý thế nào? **Đáp án gợi ý:** ghi trùng nội bộ, chuyển audit; không tự gọi leakage. **Chấm:** nêu đúng bốn query_id hoặc điều kiện raw.
5. Fixture có phải dữ liệu thật không? **Đáp án gợi ý:** không, là ví dụ do task tạo. **Chấm:** không dùng fixture để báo cáo metric.

## 11. Bài tự giải thích

Vẽ decision tree cho một key xuất hiện ở gq train và gdpr test.

Rubric 4 điểm: tách dataset; giữ split test; ghi removed row; giữ query_id nguồn.

## 12. Gate của buổi

Qua buổi khi sinh viên báo overlap theo cặp, xử lý được hai key trùng processed dev bằng bằng chứng raw, dự đoán đúng fixture và không suy số dòng từ số key.

## 13. Lỗi thường gặp & cách gỡ

- **Data:** chuẩn hóa key khác Buổi 5 → dùng cùng công thức.
- **Spec:** chuyển train sang test dataset khác → đọc D1.
- **Reasoning:** cộng giao nhau → báo từng cặp riêng.
- **Scope:** sửa assignment trong code → chỉ lập bảng dự đoán.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T4.1
Overlap pair:
Key count / row count:
Fixture decision:
Hiểu/chưa hiểu:
Lệnh và output path:
Lỗi:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc bảng D1–D6/P1–P4 trong phân tích và chuẩn bị chọn bằng chứng code/test/artifact.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: revision dùng khi tính overlap chính thức.
- CẦN XÁC MINH: signature/CLI split mới trước Chặng 5.
