---
buoi: 5
chang: 3
ten_chang: Làm quen dataset
tuan_goi_y: 2
thu_goi_y: Thứ 4 (lịch gợi ý, có thể đổi)
tasks: [T3.1]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 3 — Tìm hiểu dataset từ từng record
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T3.1
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1. Schema chung
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.2 Input/Output
  - docs/learning/00-project-analysis.md §5.1 Cache đã đọc bằng đường dẫn local, không gọi Hub
  - docs/learning/00-project-analysis.md §5.2 Bộ processed hiện tại không phải dữ liệu V1.1 đã nghiệm thu
trang_thai: chưa làm
---

# Buổi 5 — Nhận diện JSON, JSONL, Parquet và khóa dòng dữ liệu

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- phân biệt object, list, một dòng JSONL và file Parquet theo cách quan sát được;
- đọc metadata Parquet và một record đầu bằng thao tác chỉ-đọc trên cache đã có;
- phân biệt số dòng với số `question_key` duy nhất;
- chỉ ra vì sao `question_key`/`query_id` không được tự đổi khi một split được gán lại.

## 2. Vị trí trong bức tranh lớn

Buổi 5 mở Chặng 3 và làm T3.1. Sinh viên học cách nhìn hình dạng dữ liệu trước khi đọc quy tắc split, offset hay gold. Record processed hiện tại là artifact cũ; nó dùng để luyện quan sát, không được gọi là dữ liệu V1.1 đã nghiệm thu. Kết quả buổi này là nền cho Buổi 6 đọc QA và Buổi 7 đọc DPR.

## 3. Tiền đề

- Buổi 3: biết interpreter Python thật và phân biệt lệnh chỉ-đọc.
- Buổi 4: biết cache local khác tài nguyên mục tiêu Colab.
- Ôn nhanh: JSON object có cặp khóa–giá trị; list có thứ tự; JSONL thường có một JSON object mỗi dòng; Parquet lưu schema và cột theo dạng nhị phân.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| JSON object | Một record QA có các khóa `query_id`, `question`, `answers`. | Gọi tên khóa là schema V1.1 mà chưa đối chiếu spec. |
| JSONL | `data/processed/qa_germanquad_dev.jsonl` có mỗi dòng là một record độc lập. | Đọc cả file lớn vào bộ nhớ chỉ để xem một record. |
| Parquet | Cache GermanQuAD/GermanDPR có schema và metadata số dòng đọc qua `pyarrow.parquet.ParquetFile`. | Nghĩ Parquet và JSONL có cùng chi phí đọc và cùng schema. |
| Dòng và khóa duy nhất | Audit cũ có `question_counts_raw` đếm key duy nhất; đó không phải luôn là số dòng. | Cộng số key để tuyên bố bảo toàn số dòng. |
| `query_id` và `source_split` | Spec tách split gốc trong `query_id` khỏi split được gán trong `source_split`. | Parse hậu tố `query_id` rồi tự suy split hiện tại. |

Ví dụ thật: record processed QA đầu hiện có `query_id: gq-dev-46411`, `question`, `gold_passage_ids` và `answers`; đó là schema cũ cần đối chiếu, không phải mẫu đủ cho `QARecord` V1.1. [data/processed/qa_germanquad_dev.jsonl §record đầu] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.2 QA / gold record]

## 5. Kế hoạch buổi (timeline)

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | So sánh một dòng JSONL với metadata Parquet. |
| 20–45 | Xem input | Đọc record QA/DPR processed và đường dẫn cache trong phân tích. |
| 45–75 | Thao tác từng lệnh | Đọc schema, số dòng metadata và record đầu; không build lại. |
| 75–100 | Kiểm tra output | Điền bảng loại file, khóa định danh, số dòng/key và giới hạn bằng chứng. |
| 100–120 | Tự giải thích + nhật ký | Giải thích vì sao số dòng và số key không thể thay thế nhau. |

## 6. Input

- `data/processed/qa_germanquad_dev.jsonl` và `data/processed/retrieval_germandpr_dev.jsonl` — artifact cũ để quan sát một dòng.
- GermanQuAD train/test và GermanDPR train/test Parquet cache được liệt kê tại `docs/learning/00-project-analysis.md §5.1`.
- `docs/learning/00-project-analysis.md §5.2` — bảng số dòng processed và giới hạn.

## 7. Thao tác từng bước

1. Kiểm tra file JSONL mà không đọc toàn bộ:

   ```powershell
   Get-Content -LiteralPath 'data/processed/qa_germanquad_dev.jsonl' -Encoding UTF8 -TotalCount 1
   Get-Content -LiteralPath 'data/processed/retrieval_germandpr_dev.jsonl' -Encoding UTF8 -TotalCount 1
   ```

   Kỳ vọng: mỗi lệnh trả một object một dòng; ghi khóa nhìn thấy và không sửa record.
2. Đọc manifest cũ để nhận diện các trường mô tả tier:

   ```powershell
   Get-Content -LiteralPath 'data/processed/corpus_manifest.json' -Encoding UTF8
   ```

   Kỳ vọng: thấy `corpus_tier`, `actual_size`, `chunk_window_words`, `chunk_overlap_words`; ghi “artifact cũ, cần đối chiếu D6”, không gọi base V1.1 đã đạt.
3. Đọc metadata Parquet cache bằng interpreter đã xác định. Với mỗi đường dẫn trong `00-project-analysis §5.1`, ghi `num_rows` và schema; chỉ đọc batch đầu khi cần.

   ```powershell
   & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B -c "import pyarrow.parquet as pq; p=r'C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/train/0000.parquet'; f=pq.ParquetFile(p); print(f.metadata.num_rows); print(f.schema_arrow)"
   ```

   Kỳ vọng: metadata và schema đọc được nếu cache còn tồn tại. Nếu đường dẫn/interpreter không có, ghi `CẦN XÁC MINH`; không tải lại từ Hub.
4. Lập bảng cho bốn loại: JSON object, JSONL, Parquet, manifest JSON. Cột bắt buộc: đường dẫn, đơn vị record, khóa định danh thấy được, số dòng/metadata, schema cũ hay mục tiêu.
5. Đối chiếu `question_counts_raw` trong `results/data_audit_report.json` với giải thích tại `00-project-analysis §5.2`: đánh dấu đây là số key duy nhất, không phải lời hứa số dòng.
6. Viết hai câu phân biệt: `query_id` là định danh nguồn ổn định; `source_split` là split được gán theo spec. Không tự đổi giá trị artifact cũ.

7. Đếm dòng và `question_key` duy nhất theo bảng audit đã đối chiếu. Với question key chuẩn hóa bằng NFC + lowercase + strip (gộp khoảng trắng nội bộ cho dữ liệu này không đổi kết quả), bảng mẫu là:

   | Dataset/split | Số dòng | Số key duy nhất |
   |---|---:|---:|
   | GermanQuAD train | 11.518 | 11.477 |
   | GermanQuAD test | 2.204 | 2.203 |
   | GermanDPR train | 9.275 | 9.254 |
   | GermanDPR test | 1.025 | 1.025 |

   Lệnh mẫu chỉ đọc một file JSONL để kiểm tra cách đếm:

   ```powershell
   & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B -c "import json, unicodedata; p='data/processed/qa_germanquad_dev.jsonl'; rows=[json.loads(x) for x in open(p,encoding='utf-8')]; key=lambda r: unicodedata.normalize('NFC', r['question']).lower().strip(); print(len(rows), len({key(r) for r in rows}))"
   ```

   Kỳ vọng: `1031 1029` cho file processed dev. Hai key trùng này là một quan sát cần mang sang audit Buổi 9; không dùng nó để sửa file hay gọi bảng Parquet là nghiệm thu V1.1.

8. Tái tạo đúng bốn hàng Parquet bằng một lệnh đọc metadata và batch qua từng file. **Lưu ý khi copy:** khối here-string PowerShell phải có `@'` và `'@` ở cột 0; bản Markdown render có thể thụt mã trong danh sách, nên khi copy từ file nguồn hãy bỏ ba khoảng trắng đầu dòng của khối lệnh.

   ```powershell
   @'
   import pyarrow.parquet as pq, unicodedata
   files = {
       "GermanQuAD train": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/train/0000.parquet",
       "GermanQuAD test": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/test/0000.parquet",
       "GermanDPR train": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/train/0000.parquet",
       "GermanDPR test": r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/test/0000.parquet",
   }
   for name, path in files.items():
       pf = pq.ParquetFile(path)
       field = "question"
       keys = set()
       for batch in pf.iter_batches(columns=[field], batch_size=4096):
           for value in batch.column(0).to_pylist():
               keys.add(unicodedata.normalize("NFC", value).lower().strip())
       print(name, pf.metadata.num_rows, len(keys))
   '@ | & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B
   ```

   Kỳ vọng: lần lượt `11518 11477`, `2204 2203`, `9275 9254`, `1025 1025`. Nếu cache/field không khớp, ghi `CẦN XÁC MINH`; không đổi bảng theo output bất ngờ.

## 8. Dự đoán trước, kiểm tra sau

1. Một dòng JSONL có thể được đọc độc lập không? Bằng chứng nào cho điều đó?
2. Nếu metadata Parquet có số dòng khác file processed, có được sửa một bên để khớp không?
3. Vì sao `query_id` không đủ để suy `source_split`?

Gợi ý: JSONL có đơn vị dòng; khác biệt là tín hiệu cần audit; spec tách split gốc và split gán.

## 9. Output cần có

- Notebook khám phá theo output của roadmap; bảng Markdown trong bài là bản phác thảo có thể đặt vào notebook. Nếu giảng viên chưa cấp notebook, ghi rõ dùng ghi chú Markdown tạm thời — đây là lệch hình thức có chủ ý, không đổi nội dung học.
- Một dòng schema và metadata cho từng Parquet cache đã kiểm tra hoặc nhãn `CẦN XÁC MINH`.
- Bảng phân biệt số dòng, số key duy nhất, `query_id` và `source_split`.

Không tạo dataset mới, không tải Hub, không sửa `data/processed/`.

## 10. Kiểm tra hiểu bài

1. JSONL khác JSON một object ở đơn vị đọc nào? **Đáp án gợi ý:** JSONL thường tách record theo dòng; **Chấm:** không nhầm với một object bao ngoài toàn file.
2. Parquet metadata giúp biết gì trước khi đọc record? **Đáp án gợi ý:** số dòng/schema; **Chấm:** nêu được đây là quan sát cấu trúc.
3. Vì sao `question_counts_raw` không thay thế số dòng? **Đáp án gợi ý:** audit đếm key duy nhất; **Chấm:** nêu được duplicate/key normalization có thể làm lệch.
4. `source_split` có phải split gốc không? **Đáp án gợi ý:** không nhất thiết; **Chấm:** phân biệt với phần mã hóa trong `query_id`.
5. Khi cache thiếu, hành động đúng là gì? **Đáp án gợi ý:** ghi `CẦN XÁC MINH`, không tải lại; **Chấm:** giữ phạm vi chỉ đọc.

## 11. Bài tự giải thích

Giải thích bằng sơ đồ “file → record → khóa → schema → bằng chứng giới hạn” cho một file JSONL và một file Parquet.

Rubric 4 điểm:

- 1 điểm: phân biệt đúng JSONL/Parquet;
- 1 điểm: chỉ ra khóa và schema quan sát được;
- 1 điểm: phân biệt dòng với key duy nhất;
- 1 điểm: ghi đúng giới hạn artifact cũ và nhãn `CẦN XÁC MINH`.

## 12. Gate của buổi

Qua buổi khi sinh viên đọc được một record JSONL, mô tả được schema/metadata Parquet hoặc ghi rõ cache chưa xác minh, và không gọi artifact cũ là V1.1. Gate T3.1 là hiểu đơn vị dữ liệu để chuyển sang kiểm offset QA.

## 13. Lỗi thường gặp & cách gỡ

- **Data:** đọc nhầm `answers` cũ với `reference_answers` V1.1 → đối chiếu §1.2.
- **Data:** dùng số key làm số dòng → quay lại §5.2 và ghi hai cột riêng.
- **Environment:** thiếu `pyarrow`/cache → ghi `CẦN XÁC MINH`, không cài trong buổi.
- **Scope:** đọc toàn bộ file lớn → dùng `-TotalCount 1` hoặc metadata Parquet.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T3.1
File/Parquet đã đọc:
Schema và khóa quan sát được:
Số dòng / số key (nếu có):
Hiểu được:
Chưa hiểu:
Lệnh chỉ-đọc và output path:
Lỗi (code / data / spec / environment):
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Chuẩn bị đọc `context`, `answers.text`, `answers.answer_start` của GermanQuAD; không sửa record.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: cache Parquet còn tồn tại và revision nào được chủ project chốt.
- CẦN XÁC MINH: nơi lưu ghi chú khám phá cá nhân.
- CẦN XÁC MINH: schema processed cũ có được dùng lại ở bất kỳ gate nào hay chỉ làm ví dụ.
