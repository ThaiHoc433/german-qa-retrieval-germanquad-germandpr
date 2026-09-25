---
buoi: 7
chang: 3
ten_chang: Làm quen dataset
tuan_goi_y: 3
thu_goi_y: Thứ 2 (lịch gợi ý, có thể đổi)
tasks: [T3.2, T3.3]
moi_truong: [CPU, Đọc tài liệu, Local, Streaming]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 3 — Tìm hiểu dataset từ từng record
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T3.2 và §T3.3
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.3 Retrieval training record — dùng cho GermanDPR
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.2 và §2.3.f–g
  - docs/learning/00-project-analysis.md §5.1 Cache đã đọc bằng đường dẫn local, không gọi Hub
  - docs/learning/00-project-analysis.md §8. Điểm cần xác minh trước học liệu chi tiết
trang_thai: chưa làm
---

# Buổi 7 — GermanDPR, positive/negative và nguồn dữ liệu

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- chỉ ra `positive_ctxs`, `negative_ctxs`, `hard_negative_ctxs` trong GermanDPR và giải thích vai trò từng nhóm;
- giải thích vì sao positive của V1.1 là chunk có occurrence hợp lệ, còn hard negative được nhóm theo context gốc;
- phân biệt dữ liệu training với corpus đánh giá và không đưa hard negative vào corpus chỉ vì nó tồn tại;
- lập inventory dataset card/paper/license/revision/cache, ghi `CẦN XÁC MINH` khi nguồn chưa được chốt;
- mô tả streaming/Wikipedia ở mức giao thức và điều kiện trước khi tải, không tải dữ liệu trong buổi này.

## 2. Vị trí trong bức tranh lớn

Buổi 7 hoàn tất Chặng 3, phủ T3.2 và T3.3. GermanDPR cung cấp tín hiệu positive/negative cho training retriever; GermanQuAD cung cấp QA/evidence. D6 quy định hard negative chỉ dùng training, còn corpus đánh giá lấy các nguồn được phép. Inventory revision/license hôm nay là tiền đề cho audit Buổi 8–10. V1.2/A3 chưa có hiệu lực và không cần dùng để kết luận buổi này.

## 3. Tiền đề

- Buổi 5: biết đọc JSONL/Parquet và ghi giới hạn artifact cũ.
- Buổi 6: hiểu occurrence, offset và evidence GermanQuAD.
- Biết `positive` là passage có liên quan, `negative` là passage không dùng làm positive; chưa cần biết loss hay FAISS.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| GermanDPR record | Record processed đầu có `query_id=gdpr-train-0`, `question`, ba `positive_passage_ids` và tám `hard_negative_passage_ids`. | Gọi mọi ID trong record là corpus gold QA. |
| Positive chunk | V1.1 yêu cầu positive text/ID là chunk chứa occurrence hợp lệ; có thể có nhiều positive. | Dùng context gốc chưa chunk làm `positive_passage_ids`. |
| Hard-negative group | Các chunk từ cùng context gốc là một group; mỗi group đóng góp tối đa một chunk/epoch. | Cho mọi chunk của một context trọng số như nhiều ví dụ độc lập. |
| Corpus đánh giá | Base gồm nguồn được phép; hard negative GermanDPR không tự động vào corpus. | Lấy processed corpus cũ 46.551 hard negative làm base V1.1. |
| Revision/license | Revision là snapshot/cache cụ thể; license phải có nguồn card/paper, không suy từ dataset khác. | Ghi license GermanDPR theo card GermanQuAD. |
| Streaming | Đọc tuần tự snapshot theo lịch khi được phê duyệt; không đồng nghĩa đã tải hoặc đã freeze dữ liệu. | Chạy download/streaming để “xem thử” trong buổi học. |

Record processed hiện tại cho thấy `source_split: dev` nhưng `query_id: gdpr-train-0`; đây là tín hiệu schema/processing cũ cần audit, không được tự sửa hoặc suy split chuẩn. Raw record tương ứng có **1 positive context**, **3 hard-negative context** và không có negative context riêng; với cửa sổ 100 từ/overlap 20, chúng tạo **3 positive chunk** và **3+3+2 = 8 hard-negative chunk**. Vì vậy con số `3 positive + 8 hard-negative` trong processed là dấu vết quy tắc cũ làm phẳng/chunk mọi context, không phải bằng chứng có nhiều occurrence gold; V1.1 yêu cầu positive chunk có occurrence hợp lệ và hard negative được nhóm theo context gốc. [data/processed/retrieval_germandpr_dev.jsonl §record đầu] [docs/learning/00-project-analysis.md §5.2]

## 5. Kế hoạch buổi (timeline)

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | Phân biệt positive, hard negative group và corpus đánh giá. |
| 20–45 | Xem input | Đọc record GermanDPR, schema §1.3 và bảng nguồn cache. |
| 45–75 | Thao tác từng lệnh | Đọc field, đếm group/ID trên một record, lập inventory nguồn. |
| 75–100 | Kiểm tra output | Vẽ luồng GermanDPR → training record; đánh dấu dữ liệu không được đưa vào corpus. |
| 100–120 | Tự giải thích + nhật ký | Giải thích vì sao phải chốt revision/license trước audit. |

## 6. Input

- `data/processed/retrieval_germandpr_dev.jsonl` và raw GermanDPR Parquet cache trong `00-project-analysis §5.1`.
- `docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.3`, §2.3.f–g và D6.
- Dataset card/paper/license/revision nếu chủ project cung cấp; nếu không có, ghi `CẦN XÁC MINH`.
- Không dùng internet/download trong buổi này; chỉ đọc nguồn local đã có.

## 7. Thao tác từng bước

1. Đọc một record GermanDPR processed:

   ```powershell
   Get-Content -LiteralPath 'data/processed/retrieval_germandpr_dev.jsonl' -Encoding UTF8 -TotalCount 1
   ```

   Kỳ vọng: ghi các field `query_id`, `question`, `positive_passage_ids`, `hard_negative_passage_ids`, `source_split`; đánh dấu schema cũ.
2. Đọc schema raw GermanDPR Parquet bằng cache đã liệt kê trong `00-project-analysis §5.1`. Ghi rằng nhóm context là struct chứa các list song song `title`, `text`, `passage_id`; không giả định list dict.
3. Trên record processed, lập bảng `ID`, nhóm positive/hard-negative, có trùng hay không, có bằng chứng passage text/occurrence hay chưa. Không gọi ID là gold evidence vì GermanDPR không bảo đảm `answer_start`.
4. In số từ và số chunk của raw record đầu bằng lệnh chỉ-đọc. **Lưu ý khi copy:** khối here-string PowerShell phải có `@'` và `'@` ở cột 0; nếu copy từ file nguồn Markdown, bỏ ba khoảng trắng đầu dòng của khối lệnh.

   ```powershell
   @'
   import math, pyarrow.parquet as pq
   path = r"C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/train/0000.parquet"
   record = next(iter(pq.ParquetFile(path).iter_batches(batch_size=1))).to_pylist()[0]
   for group_name in ("positive_ctxs", "negative_ctxs", "hard_negative_ctxs"):
       group = record[group_name]
       texts = group["text"] if group else []
       for index, text in enumerate(texts):
           words = len(text.split())
           chunks = 1 if words <= 100 else 1 + math.ceil((words - 100) / 80)
           print(group_name, index, words, chunks)
   '@ | & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B
   ```

   Kỳ vọng: positive có `230 → 3`; hard negative lần lượt `212 → 3`, `216 → 3`, `101 → 2`; negative có 0 context. Công thức là `1` chunk nếu `n ≤ 100`, ngược lại `1 + ceil((n−100)/80)` với overlap 20.

   Sau đó so sánh raw với processed trong bảng mẫu: raw `1 positive context / 3 hard-negative context / 0 negative context`; processed `3 positive IDs / 8 hard-negative IDs`. Ghi rõ đây là quy tắc cũ cần audit, không phải cách tạo gold V1.1.
5. Vẽ sơ đồ chuyển đổi mục tiêu V1.1: raw context → chunk có occurrence → `positive_texts`/`positive_passage_ids`; hard-negative context → `source_context_id` group → tối đa một chunk/group/epoch. Chỉ mô tả, không viết producer.
6. Đối chiếu D6: đánh dấu hard negative là training-only; corpus base V1.1 lấy mọi chunk context GermanQuAD và positive context GermanDPR, không lấy passage chỉ vì là hard negative.
7. Lập inventory bốn cột: dataset/split, đường dẫn hoặc revision cache, paper/license/source, trạng thái xác minh. Ghi riêng GermanQuAD Parquet revision `a2f3a59…` và card README revision `fff05ce…` vì đây là hai snapshot khác nhau; GermanDPR card/license, Wikipedia và paper còn thiếu thì ghi `CẦN XÁC MINH` theo §8 của phân tích.
8. Đọc mô tả streaming trong roadmap/spec và viết checklist trước khi có quyền tải: revision được chốt, lịch R, filter P4, nơi lưu, fingerprint và người phê duyệt. Không chạy download hoặc streaming thật.
9. Ghi điều kiện chuyển sang Chặng 4: audit rows/keys/length/overlap được phép bắt đầu trên cache hiện có; revision/license chưa chốt phải treo ở inventory, không tự chọn.

## 8. Dự đoán trước, kiểm tra sau

1. Vì sao record processed có 3 positive và 8 hard-negative, trong khi raw chỉ có 1 positive context và 3 hard-negative context?
2. Vì sao hard negative không tự động vào corpus đánh giá?
3. Nếu `query_id` và `source_split` mâu thuẫn trong processed file, trường nào được tự sửa?

Gợi ý: đó là kết quả chunk hóa/làm phẳng artifact cũ (`3` và `3+3+2`), không tự chứng minh nhiều occurrence gold; D6 tách training hard negative khỏi corpus đánh giá; không sửa tại chỗ, ghi lệch và chờ audit/spec.

## 9. Output cần có

- Bảng field GermanDPR, so sánh raw–processed `1/3/0` với `3/8`, và sơ đồ positive/group.
- Inventory dataset/revision/license/cache với nhãn `đã đọc` hoặc `CẦN XÁC MINH`.
- Ghi chú streaming/Wikipedia dạng checklist điều kiện, không có artifact download.

Không tải Wikipedia, không gọi Hub, không build positive/negative mới, không sửa processed data.

## 10. Kiểm tra hiểu bài

1. GermanDPR có `answer_start` bắt buộc như GermanQuAD không? **Đáp án gợi ý:** không; **Chấm:** phân biệt retrieval training record.
2. Một hard-negative context bị chunk thành năm mảnh thì có năm group không? **Đáp án gợi ý:** không, cùng context là một group; **Chấm:** nêu tối đa một chunk/group/epoch.
3. D6 cho phép hard negative vào corpus base không? **Đáp án gợi ý:** không chỉ vì nó là hard negative; **Chấm:** nêu nguồn corpus được phép.
4. License chưa có card local thì ghi gì? **Đáp án gợi ý:** `CẦN XÁC MINH`; **Chấm:** không suy từ GermanQuAD.
5. Streaming có phải đã tải dữ liệu không? **Đáp án gợi ý:** không; **Chấm:** phân biệt giao thức dự kiến với hành động thực tế.

## 11. Bài tự giải thích

Trình bày trong 3 phút vì sao GermanDPR positive/negative phục vụ training nhưng không được dùng để tự định nghĩa corpus đánh giá.

Rubric 4 điểm:

- 1 điểm: mô tả đúng positive và hard-negative group;
- 1 điểm: nêu D6/corpus nguồn được phép;
- 1 điểm: phân biệt không có answer offset;
- 1 điểm: nêu giới hạn revision/license và không tải dữ liệu.

## 12. Gate của buổi

Qua buổi khi sinh viên chỉ đúng positive/hard-negative/group, giải thích được D6 và có inventory revision/license/cache với các mục chưa xác minh được đánh dấu rõ. Gate Chặng 3 yêu cầu hiểu split, offset và role của negative; chưa phải gate producer hay corpus V1.1.

## 13. Lỗi thường gặp & cách gỡ

- **Data:** gọi `hard_negative_passage_ids` là gold → đọc lại D6 và §1.3.
- **Data:** gộp mọi chunk thành nhiều group → dùng `source_context_id`.
- **Spec:** dùng card GermanQuAD cho GermanDPR → ghi `CẦN XÁC MINH` nguồn riêng.
- **Environment:** tải dataset để kiểm tra license → dừng; chỉ đọc nguồn local được cấp.
- **Scope:** tự xây RetrievalTrainingRecord → chỉ lập sơ đồ, producer thuộc Chặng 5.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T3.2, T3.3
Record/dataset/split:
Positive và hard-negative group:
Revision/license/cache:
Streaming checklist:
Hiểu được:
Chưa hiểu:
Lỗi (code / data / spec / environment):
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc phần Chặng 4; chuẩn bị bảng đếm rows/keys/length/overlap trên từng dataset, không cộng các giao nhau máy móc.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: GermanDPR card/license và paper được chủ project phê duyệt.
- CẦN XÁC MINH: revision cache chuẩn cho run mới.
- CẦN XÁC MINH: revision/snapshot Wikipedia, lịch streaming R, quyền tải và nơi lưu.
- CẦN XÁC MINH: cách xử lý lệch `source_split` trong processed artifact cũ trước audit.
