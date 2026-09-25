---
buoi: 6
chang: 3
ten_chang: Làm quen dataset
tuan_goi_y: 2
thu_goi_y: Thứ 6 (lịch gợi ý, có thể đổi)
tasks: [T3.1]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 3 — Tìm hiểu dataset từ từng record
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T3.1
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.2 QA / gold record — chỉ dùng cho GermanQuAD
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.2 Input/Output
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý
  - docs/learning/00-project-analysis.md §5.1 Cache đã đọc bằng đường dẫn local, không gọi Hub
  - docs/learning/00-project-analysis.md §5.2 Bộ processed hiện tại không phải dữ liệu V1.1 đã nghiệm thu
trang_thai: chưa làm
---

# Buổi 6 — Đọc GermanQuAD và kiểm tra offset đáp án

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- chỉ đúng vai trò của `question`, `context`, `answers.text` và `answers.answer_start` trong GermanQuAD;
- tính bằng tay và kiểm tra bằng code điều kiện `context[start:start+len(answer)] == answer`;
- phân biệt offset trên context gốc với offset trong chunk V1.1;
- nhận diện nhiều reference answer và trường hợp answer không map được vào chunk mà không tự xóa record.

## 2. Vị trí trong bức tranh lớn

Buổi 6 tiếp T3.1 và chuẩn bị T3.3/T4.1. Offset đúng là tiền đề cho `gold_evidences`, sau đó mới có thể tính Recall và phân biệt `retrieval_miss` với lỗi Reader. Record processed hiện tại có `answers` và `answer_start` nhưng schema cũ; schema V1.1 yêu cầu chuyển sang evidence/reference rõ ràng.

## 3. Tiền đề

- Buổi 5: biết cách đọc một record và phân biệt số dòng với key.
- Biết Python slice nửa mở `text[start:end]`.
- Chưa cần biết chunking triển khai; chỉ cần hiểu offset gốc và quy tắc 100 từ/overlap 20 ở mức đọc spec.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| QA record | GermanQuAD raw có `context`, `question`, `answers`. | Dùng field `gold_passage_ids` cũ làm thay cho evidence offset. |
| `answer_start` | Nếu `start=146`, lấy từ context tại vị trí 146; phải kiểm bằng slice. | Đếm từ thay vì ký tự hoặc dùng offset của chunk khác. |
| Answer end | `end = start + len(answer)` là exclusive. | Trừ/ cộng sai một ký tự vì coi end là inclusive. |
| Nhiều references | `reference_answers` giữ toàn bộ đáp án hợp lệ; evidence map được là tập con. | Chỉ giữ answer đầu tiên rồi làm mất reference hợp lệ. |
| Gold evidence | Evidence trỏ tới passage/chunk và offset trong chính chunk. | Gán offset context gốc vào chunk mà không trừ `chunk_start_char`. |
| Gold rỗng | Không map được answer vào chunk vẫn ghi record; chỉ loại khỏi mẫu số metric chất lượng. | Xóa record để làm coverage đẹp hơn. |

Record GermanQuAD cache đã đọc có `id=51870`, câu hỏi “Von welchem Gesetzt stammt das Amerikanische ab? ”, đáp án “britischen Common Laws” và `answer_start=146`; slice trên record này được báo là đúng. Đây là ví dụ dữ liệu cache đã quan sát, không đại diện mọi record. [docs/learning/00-project-analysis.md §5.1 Cache đã đọc bằng đường dẫn local, không gọi Hub]

## 5. Kế hoạch buổi (timeline)

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | Tính tay start/end trên một chuỗi ngắn. |
| 20–45 | Xem input | Đọc schema GermanQuAD và record cache. |
| 45–75 | Thao tác từng lệnh | In record, kiểm slice answer và ghi trường hợp nhiều/không map. |
| 75–100 | Kiểm tra output | Đối chiếu offset gốc với quy tắc chunk/evidence V1.1. |
| 100–120 | Tự giải thích + nhật ký | Giải thích vì sao offset sai làm hỏng gold và metric sau này. |

## 6. Input

- GermanQuAD Parquet cache train/test trong `docs/learning/00-project-analysis.md §5.1`.
- `data/processed/qa_germanquad_dev.jsonl` — record processed cũ để so schema.
- `docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.2` và §2.2.

## 7. Thao tác từng bước

1. Đọc record processed đầu:

   ```powershell
   Get-Content -LiteralPath 'data/processed/qa_germanquad_dev.jsonl' -Encoding UTF8 -TotalCount 1
   ```

   Kỳ vọng: thấy `answers.text` và `answers.answer_start` trong schema cũ; ghi rõ chưa có `gold_evidences` V1.1 trong dòng này.
2. Đọc một record raw GermanQuAD từ **test** Parquet cache để có cơ hội quan sát nhiều reference answer. Lệnh dưới là **LỆNH MẪU – đã đối chiếu đường dẫn trong `00-project-analysis §5.1`, cần xác minh lại cache trên máy học**:

   ```powershell
   & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B -c "import pyarrow.parquet as pq; p=r'C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/test/0000.parquet'; r=next(iter(pq.ParquetFile(p).iter_batches(batch_size=1))).to_pylist()[0]; print(r['id']); print(r['question']); print(r['answers'])"
   ```

   Kỳ vọng: thấy các field raw; nếu thiếu cache, ghi `CẦN XÁC MINH`, không tải dataset.
3. Chọn một cặp `answer`, `answer_start` từ record vừa đọc và dự đoán trước `answer_end = answer_start + len(answer)`. Ghi công thức trước khi chạy. Split test có các record chứa 2–3 reference; nếu record đầu chỉ có một, chọn tiếp một record test có nhiều phần tử trong `answers.text`.
4. Kiểm tra điều kiện offset bằng đoạn Python chỉ đọc (đoạn này tự đọc lại record raw vào biến `record`, nên không phụ thuộc biến chưa khai báo):

   ```python
   import pyarrow.parquet as pq
   path = r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germanquad/snapshots/a2f3a59f0be843fc305d0417d7292ef0b1a66884/plain_text/test/0000.parquet"
   record = next(iter(pq.ParquetFile(path).iter_batches(batch_size=1))).to_pylist()[0]
   # raw Parquet: answers là dict/list song song; processed JSONL là list object.
   answer = record["answers"]["text"][0]
   start = record["answers"]["answer_start"][0]
   end = start + len(answer)
   print(record["context"][start:end] == answer, start, end)
   ```

   Kỳ vọng: với record/answer hợp lệ, biểu thức là `True`. Nếu `False`, kiểm tra kiểu dữ liệu, answer đang chọn, offset ký tự và record gốc; không sửa offset tại chỗ. Với processed JSONL, `answers` lại là list các object `{text, answer_start}`; phải đổi cách lấy phần tử trước khi kiểm tra, không dùng nguyên đoạn raw như thể hai schema giống nhau.
5. Lập bảng cho tất cả answer/reference nhìn thấy: `answer_text`, `start`, `end`, `slice_equal`, `map được/không map được`; bắt buộc ghi một record có nhiều reference từ test nếu tìm được.
6. Đối chiếu với spec: evidence V1.1 lưu offset trong chính chunk; nếu chunk bắt đầu tại `chunk_start_char`, offset local là offset gốc trừ start chunk. Không tự tạo chunk hay build corpus trong buổi này.
7. Ghi một ví dụ gold rỗng: record vẫn phải lưu, nhưng không vào `N_eval` của metric chất lượng. Đây là quy tắc đọc spec, chưa phải thao tác sửa file.

## 8. Dự đoán trước, kiểm tra sau

1. Nếu `answer_start` tăng 1, slice có còn bằng answer không?
2. Nếu answer xuất hiện hai lần trong context, giữ một hay nhiều occurrence?
3. Nếu answer không nằm trọn trong chunk, có xóa QA record không?

Gợi ý: offset sai làm điều kiện slice thất bại; spec giữ đủ references/occurrence hợp lệ; record gold rỗng vẫn ghi và chỉ loại khỏi mẫu số quality.

## 9. Output cần có

- Bảng kiểm offset với ít nhất một record thật/cache và trạng thái `slice_equal`.
- Một sơ đồ context gốc → chunk → evidence offset local.
- Ghi chú riêng cho nhiều references và gold rỗng.

Không sửa `data/processed/`, không build chunk, không chạy audit notebook.

## 10. Kiểm tra hiểu bài

1. `answer_end` được tính thế nào? **Đáp án gợi ý:** `start + len(answer)`; **Chấm:** end exclusive.
2. Vì sao phải kiểm `context[start:start+len(answer)] == answer`? **Đáp án gợi ý:** xác nhận offset trỏ đúng text; **Chấm:** không chỉ kiểm số nguyên.
3. Evidence offset trong chunk khác offset context gốc thế nào? **Đáp án gợi ý:** trừ `chunk_start_char`; **Chấm:** nêu đúng hướng biến đổi.
4. Gold rỗng có bị xóa không? **Đáp án gợi ý:** không; giữ record, loại khỏi `N_eval`; **Chấm:** phân biệt lưu dữ liệu và tính metric.
5. Hai occurrence hợp lệ xử lý thế nào? **Đáp án gợi ý:** giữ đủ evidence/reference theo spec; **Chấm:** không chỉ lấy occurrence đầu.

## 11. Bài tự giải thích

Viết 150–200 từ giải thích đường đi của một `answer_start` từ context gốc sang `gold_evidences` trong chunk.

Rubric 4 điểm:

- 1 điểm: nêu đúng slice và end exclusive;
- 1 điểm: phân biệt context gốc/chunk;
- 1 điểm: giữ multiple references/occurrences;
- 1 điểm: xử lý đúng gold rỗng và giới hạn bằng chứng.

## 12. Gate của buổi

Qua buổi khi sinh viên kiểm được ít nhất một offset bằng slice, giải thích được offset local và không xóa record gold rỗng. Gate T3.1 cần đúng nền QA trước khi chuyển sang GermanDPR; chưa phải gate schema V1.1 hay corpus build.

## 13. Lỗi thường gặp & cách gỡ

- **Data:** chọn `answers` của schema cũ rồi gọi là `gold_evidences` → đối chiếu §1.2.
- **Code:** dùng end inclusive → viết lại công thức end exclusive.
- **Offset:** trừ sai `chunk_start_char` → vẽ hai trục context/chunk.
- **Scope:** sửa trực tiếp record để slice thành True → giữ nguyên input, ghi lỗi và nguồn.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T3.1
Dataset/split/record:
Answer, start, end:
Kết quả context[start:end] == answer:
Offset context gốc / offset chunk:
Hiểu được:
Chưa hiểu:
Lỗi (code / data / spec / environment):
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc field `positive_ctxs`, `negative_ctxs`, `hard_negative_ctxs` của GermanDPR và ghi câu hỏi về group.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: revision Parquet GermanQuAD dùng cho các buổi audit sau.
- CẦN XÁC MINH: cách chủ project muốn lưu ghi chú offset cá nhân.
- CẦN XÁC MINH: record nào được chọn làm fixture chính thức; không suy từ record đầu.
