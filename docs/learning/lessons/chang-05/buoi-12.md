---
buoi: 12
chang: 5
ten_chang: Schema và dữ liệu V1.1
tuan_goi_y: 4
thu_goi_y: Thứ 6 (lịch gợi ý, có thể đổi)
tasks: [T5.2]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T5.2 — Thẻ mẫu: Gán split theo D1
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3.e và §2.6
  - docs/learning/00-project-analysis.md §4. Hiện trạng module: giữ / sửa / viết mới / build lại
  - docs/learning/lessons/chang-05/the-task-T5.2.md §3. Interface đã chốt
trang_thai: chưa làm
---

# Buổi 12 — Split D1 và bảo toàn rows

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- áp dụng D1 trên fixture 6 dòng theo từng dòng và dataset;
- phân biệt `query_id` gốc với `source_split` được gán;
- ghi quyết định chuẩn hóa key đúng spec: NFC + lowercase + strip, không gộp khoảng trắng nội bộ;
- dự đoán phương trình bảo toàn rows trước khi code.

## 2. Vị trí trong bức tranh lớn

Buổi 12 thực hiện T5.2 trên giấy/fixture, sau validator. V1.1 giữ đúng NFC + lowercase + strip, không gộp whitespace nội bộ. Nhánh thực thi T5.2 đã bỏ bước gộp của code cũ và có [log kiểm chứng](kiem-chung-T5.2.md). Cách chuẩn hóa này giữ số key/dòng loại trên cache nhưng đổi tập dev; DPR dev có 928 dòng, không phải 929 của cách gộp. Sinh viên vẫn cần tự làm fixture và gate của buổi; kết quả code của người triển khai không xác nhận đã học xong.

## 3. Tiền đề

- Buổi 9: fixture 6 dòng, overlap key và D1.
- Buổi 11: validator/test-first.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ | Sai lầm |
|---|---|---|
| TestKeys | Tập hợp hợp nhất key test gốc của cả hai dataset; quyết định D1 theo từng dòng và dataset, không chuyển dòng sang test dataset khác. | Dùng unique key thay row hoặc chuyển dòng train sang test của dataset kia. |
| D1 | Train trùng TestKeys bị loại, không đẩy vào test. | Loại toàn bộ key trùng khỏi mọi split. |
| Stable query_id | `gq-train-51870` giữ nguồn gốc khi assigned dev. | Đổi thành `gq-dev-51870`. |
| Key normalization | NFC/lower/strip theo §2.3.e. | Thêm collapse whitespace chưa có trong spec. |
| Row conservation | `10.492 (GQ train+dev) + 2.204 (GQ test) + 1.026 (loại do trùng key DPR test) = 13.722` dòng GermanQuAD. | Dùng unique key thay rows. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích | D1 bằng fixture. |
| 20–45 | Xem input | Đọc catalog T5.2 và `src/splits.py`. |
| 45–80 | Fixture test-first | Dự đoán quyết định từng dòng. |
| 80–105 | Kiểm tra | Tính conservation và ghi lệch whitespace. |
| 105–120 | Tự giải thích | Nhật ký/gate. |

## 6. Input

- Fixture 6 dòng catalog T5.2.
- Kiểm chứng dữ liệu đã audit: GermanQuAD có `10.492 + 2.204 + 1.026 = 13.722` dòng; 1.026 là số dòng GQ train trùng key với DPR test, không phải 1.026 key duy nhất.
- `src/splits.py` chỉ đọc.
- Spec §2.3.e/§2.6.
- [Thẻ T5.2](the-task-T5.2.md): signature theo dòng, schema report và fixture đã dùng; [log kiểm chứng](kiem-chung-T5.2.md) ghi kết quả nhánh thực thi riêng với tiến độ sinh viên.

## 7. Thao tác từng bước

1. Đọc hàm `question_key`/assignment bằng PowerShell:

   ```powershell
   Select-String -LiteralPath 'src/splits.py' -Pattern '^class SplitDecision|^def question_key|^def assign_splits|normalize|lower|strip|source_split'
   ```

   Kỳ vọng: thấy `SplitDecision`, hai hàm và normalization `NFC`/`lower`/`strip`; không còn bước collapse trong `question_key`. Ghi rõ: sửa này có thể đổi thành viên dev dù số key duy nhất và số dòng loại giữ nguyên.
2. Viết fixture 6 dòng đúng catalog: gq train A/B/C, gq test D, gdpr train B, gdpr test A.
3. Viết bảng expected trước code: gq A loại; gdpr A giữ test; B cùng split; gq D giữ test; `query_id` gốc không đổi.
4. Tính conservation theo từng dataset; không cộng overlap key thành row count. Đối chiếu GermanQuAD: `9.445 train + 1.047 dev + 2.204 test + 1.026 loại = 13.722`. GermanDPR trước chunk mapping: `8.346 + 928 + 1.025 + 1 = 10.300`. Với Python 3.13.3, seed 42, ratio 0,10: pool 10.452 key, dev 1.045 key, train 9.407 key; xem lệnh và cảnh báo đổi runtime ở [thẻ §6](the-task-T5.2.md#6-mốc-dữ-liệu-thật-và-lệnh-đã-kiểm).
5. Ghi quyết định học liệu: V1.1 §2.3.e chỉ NFC + lowercase + strip; không gộp whitespace nội bộ. Nếu chủ project muốn thêm bước, đó là amendment/quyết định riêng.
6. Chạy lệnh đã xác minh trên fixture, sau khi tự dự đoán kết quả:

   ```powershell
   & '.venv/Scripts/python.exe' -B -m pytest tests/test_splits.py -q -p no:cacheprovider --tb=line
   ```

   Kỳ vọng ở phiên đã kiểm: **43 passed**. Đối chiếu cụ thể ca duplicate/key/row; test xanh không thay thế kiểm dữ liệu thật hoặc base gate.
7. **Nhánh thực thi sau gate (không tính vào 120 phút):** theo [thẻ T5.2](the-task-T5.2.md), người triển khai đã viết `tests/test_splits.py` trước `src/splits.py` và lưu [log RED/GREEN](kiem-chung-T5.2.md). Sinh viên đọc log và tự giải thích fixture, không xóa code để tạo RED giả. Report/decisions trả trong bộ nhớ; ghi artifact và chuyển caller notebook ở T5.8. Code T5.2 đã có bằng chứng pass để phục vụ T5.3; trạng thái học vẫn chưa làm.

## 8. Dự đoán trước, kiểm tra sau

1. Gq train A trùng gdpr test A đi đâu? **GQ train A bị loại theo D1; GermanDPR test A vẫn là test gốc của DPR.**
2. Nếu hai câu khác khoảng trắng nội bộ, bài này coi chúng cùng key không? **Không theo V1.1; chỉ NFC + lowercase + strip.**
3. Conservation tính theo key hay row? **Theo row; 1.026 dòng giao key không đồng nghĩa 1.026 key duy nhất.**

## 9. Output cần có

- Fixture 6 dòng và bảng expected.
- Quyết định chuẩn hóa key tách khỏi code hiện tại.
- Phương trình conservation theo dataset, gồm số đã audit `10.492 + 2.204 + 1.026 = 13.722`.
- Thẻ T5.2 đã được đối chiếu về input, quyết định từng dòng và report.
- Sau nhánh thực thi: decisions/report trong bộ nhớ và log test T5.2; ghi `data/processed/v1.1/split_report.json` ở T5.8. Trước thực thi chỉ có fixture expected và mốc audit, chưa có artifact được nghiệm thu.

## 10. Kiểm tra hiểu bài

1. D1 bảo vệ test nào? **Test gốc từng dataset.** Chấm: không chuyển chéo.
2. Vì sao giữ query_id? **Truy nguồn ổn định.** Chấm: phân biệt source_split.
3. Collapse whitespace có trong spec không? **Không.** Chấm: ghi lệch code.
4. Conservation dùng gì? **Rows.** Chấm: không dùng unique keys.

## 11. Bài tự giải thích

Giải thích một dòng bị loại và một dòng test được giữ bằng decision tree D1.

Rubric 4 điểm: đúng key; đúng dataset; đúng query_id; đúng conservation.

## 12. Gate của buổi

Qua giờ hướng dẫn khi expected fixture đúng, quyết định normalization rõ, và phương trình rows được viết trước implementation. Chặng chỉ được chuyển sang T5.3 sau khi nhánh thực thi T5.2 chạy pass.

## 13. Lỗi thường gặp & cách gỡ

- **Spec:** sửa spec để hợp collapse → giữ spec frozen.
- **Data:** loại cả hai phía overlap → xét D1 từng dataset.
- **ID:** đổi query_id → giữ split gốc.
- **Runtime:** số dev lệch mốc → kiểm normalization, snapshot, thứ tự dòng gốc và phiên bản Python trước khi sửa kỳ vọng.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T5.2
Lệnh chỉ-đọc đã chạy:
Output path:
Fixture decision:
Normalization decision:
Conservation equation:
Lệch code cần sửa:
Nhánh thực thi đã chạy/chưa chạy:
Hiểu/chưa hiểu:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Ôn evidence/reference và collision 0/1/>1 occurrence.

## 16. Điểm cần xác minh

- Signature đã triển khai và caller đã kiểm kê trong [thẻ T5.2 §2–3](the-task-T5.2.md#3-interface-đã-chốt); bằng chứng ở [log kiểm chứng](kiem-chung-T5.2.md).
- CẦN XÁC MINH: caller ngoài `src/`, `tests/`, `notebooks/` và revision chuẩn khi build — chủ project/người triển khai kiểm trước T5.8; mốc cache của thẻ chưa phải manifest chính thức.
- Quyết định đã xác nhận và triển khai: giữ đúng V1.1 §2.3.e (NFC + lowercase + strip), không gộp whitespace nội bộ. Đổi Python, kể cả Colab, phải đối chiếu lại tập dev; không coi số key/dòng loại giống nhau là hai cách chia split tương đương.
