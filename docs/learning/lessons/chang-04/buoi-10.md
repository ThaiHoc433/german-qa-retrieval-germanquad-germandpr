---
buoi: 10
chang: 4
ten_chang: Data audit và quyết định spec
tuan_goi_y: 4
thu_goi_y: Thứ 2 (lịch gợi ý, có thể đổi)
tasks: [T4.2]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 4 — Audit và đối chiếu V1.1
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T4.2
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §Nhật ký quyết định V1.1 và §9 Bảng tiêu chí nghiệm thu tổng hợp
  - docs/learning/00-project-analysis.md §4. Hiện trạng module: giữ / sửa / viết mới / build lại
  - docs/learning/00-project-analysis.md §6. Hợp đồng dữ liệu và giao thức cần giữ
trang_thai: chưa làm
---

# Buổi 10 — Requirement traceability matrix D1–D6/P1–P4

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- bắt đầu từ bảng hiện trạng module §4 và nối mỗi quyết định với code/test/artifact;
- lập traceability matrix cho D1–D6 và P1–P4;
- phân biệt “đã quan sát”, “lệch cần sửa”, “chưa có bằng chứng” và “chờ A3”;
- viết danh sách lỗi cần sửa trước Chặng 5 mà không sửa spec/code trong buổi.

## 2. Vị trí trong bức tranh lớn

Buổi 10 đóng Chặng 4 và T4.2. Bảng §4 là điểm bắt đầu để không audit rời rạc: mỗi module có phân loại, bằng chứng và phụ thuộc. Matrix hôm nay là đầu vào cho T5.0–T5.8; V1.2 draft/A3 vẫn chưa có hiệu lực.

## 3. Tiền đề

- Buổi 8: số liệu audit có nguồn.
- Buổi 9: overlap và D1 theo fixture.
- Đọc D1–D6/P1–P4 ở `00-project-analysis §6`.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| Traceability | D1 → `src/splits.py`/test/artifact/report. | Có tên file là xem như requirement đạt. |
| Evidence status | observed / lệch / thiếu / chờ chốt. | Dùng “pass” cho code chưa chạy. |
| Frozen spec | V1.1 là chuẩn; code/artifact cũ có thể lệch. | Sửa spec để hợp code cũ. |
| A3 | Metadata draft cho stage/candidate chưa hiệu lực. | Dùng schema draft cho production selection. |
| Gate | Điều kiện chuyển chặng, không phải điểm số. | Cho qua gate vì có bảng nhưng thiếu nguồn. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | Đọc một hàng §4 và đi tới D1. |
| 20–45 | Xem input | Đọc §4, §6, §9 và code/test/artifact tham chiếu. |
| 45–80 | Thao tác từng lệnh | Điền matrix 10 quyết định, không chạy producer. |
| 80–105 | Kiểm tra output | Đánh dấu lỗi cần sửa và phụ thuộc Chặng 5. |
| 105–120 | Tự giải thích + nhật ký | Nói vì sao traceability không phải nghiệm thu. |

## 6. Input

- `docs/learning/00-project-analysis.md §4` — bảng module bắt đầu.
- `docs/learning/00-project-analysis.md §6` — D1–D6/P1–P4.
- Spec §9 và các file code/test/artifact được trích trong từng hàng.

## 7. Thao tác từng bước

1. Dùng PowerShell để đọc đúng bảng §4 và §6:

   ```powershell
   $t = Get-Content -LiteralPath 'docs/learning/00-project-analysis.md' -Encoding UTF8
   $s = ($t | Select-String '^## 4\.').LineNumber
   $e = ($t | Select-String '^### 4\.1').LineNumber
   $t[$s..($e-2)] | Where-Object { $_ -match '^\| ' } | ForEach-Object { ($_ -split '\|')[1].Trim() }
   Select-String -LiteralPath 'docs/learning/00-project-analysis.md' -Pattern '^\| (D1|D2|D3|D4|D5|D6|P1|P2|P3|P4) \|'
   ```

   Kỳ vọng: phần cắt từ `## 4` tới trước `### 4.1` trả 18 dòng gồm tiêu đề `Thành phần` và đúng 17 hàng dữ liệu: Normalize/chunk, CorpusBuilder, Splits, QA producer, DPR producer, HF loader, Schemas, Metric đơn lẻ, Evaluation tổng hợp, Test Evaluation cũ, BM25, Run infra / oracle / Integration, Dense / Reader / Reranker, Distractor / acceptance, Artifact cũ, Nhánh LLM/RAG và README. Lệnh §6 trả đủ D1–D6/P1–P4.
2. Chép đúng 17 hàng dữ liệu vừa kiểm tra từ §4 vào matrix: Normalize/chunk, CorpusBuilder, Splits, QA producer, DPR producer, HF loader, Schemas, Metric đơn lẻ, Evaluation tổng hợp, Test Evaluation cũ, BM25, Run infra / oracle / Integration, Dense / Reader / Reranker, Distractor / acceptance, Artifact cũ, Nhánh LLM/RAG và README. Không gắn `Data/Splits/Corpus` của §3 như một hàng §4 riêng.
3. Thêm cột `requirement`, `code evidence`, `test evidence`, `artifact evidence`, `status`, `next task`, `source/revision`.
4. Điền mẫu D1: code `src/splits.py`, fixture catalog T5.2, test hiện chưa có/`CẦN XÁC MINH`, status `lệch/chưa nghiệm thu`, next T5.2.
5. Điền D3: test evaluation cũ ở dòng 213/224 lệch bucket, spec D3 là chuẩn, next T6.2–T6.3.
6. Điền P1/P2: retrieval_k=20, reader_k=5, checkpoint selection dev/base; nếu artifact chưa có manifest/fingerprint ghi thiếu.
7. Điền D6/P4: corpus source/title exclusion; phân biệt code hiện hữu với acceptance mới; thêm CorpusBuilder/Distractor/acceptance và Artifact cũ vào hàng traceability.
8. Thêm cột phụ thuộc: T5 validators trước producer; A3 `chờ chốt` cho production selection, không dùng draft.
9. Viết danh sách “sửa ở Chặng 5”, “chờ revision/nguồn”, “không được làm trong học liệu”; không sửa file nguồn.

## 8. Dự đoán trước, kiểm tra sau

1. Một file `src/` tồn tại nhưng thiếu test và manifest nên status gì? **Đáp án gợi ý:** thiếu bằng chứng hoặc `CẦN XÁC MINH`, không ghi đạt. **Chấm:** yêu cầu được nối với test/artifact còn thiếu.
2. Test xanh nhưng bucket lệch D3 có được ghi đạt không? **Đáp án gợi ý:** không; ghi lệch V1.1 và dẫn D3. **Chấm:** phân biệt trạng thái test với trạng thái requirement.
3. A3 chưa chốt thì T8.4 có thể dùng schema draft production không? **Đáp án gợi ý:** không; chỉ học fixture/logic độc lập và ghi `chờ chốt A3`. **Chấm:** không biến draft thành chuẩn frozen.

## 9. Output cần có

- Requirement traceability matrix D1–D6/P1–P4.
- Danh sách lỗi/độ lệch theo module §4 và task Chặng 5.
- Danh sách `CẦN XÁC MINH` gồm revision, interface, manifest và A3.

## 10. Kiểm tra hiểu bài

1. Traceability khác file inventory thế nào? **Đáp án gợi ý:** có requirement và bằng chứng liên kết. **Chấm:** chỉ ra được một hàng §4 nối tới code/test/artifact.
2. Vì sao test xanh chưa đủ? **Đáp án gợi ý:** test có thể lệch V1.1, như hai test bucket D3. **Chấm:** nêu được nguồn spec và status.
3. P1 khóa gì? **Đáp án gợi ý:** `retrieval_k/reader_k` và tính nhất quán run. **Chấm:** ghi đúng 20/5 và manifest.
4. A3 hiện trạng? **Đáp án gợi ý:** chờ chốt, chưa dùng production. **Chấm:** không đưa draft vào acceptance.
5. D6 kiểm tra gì? **Đáp án gợi ý:** nguồn corpus và hard-negative boundary. **Chấm:** phân biệt training với evaluation corpus.
6. Nếu không có revision dữ liệu, matrix ghi gì? **Đáp án gợi ý:** `CẦN XÁC MINH`, không dùng số audit làm chuẩn chính thức. **Chấm:** nối được revision với gate Chặng 4.

## 11. Bài tự giải thích

Chọn một hàng §4 và trình bày đường đi từ quan sát → requirement → bằng chứng → task sửa.

Rubric 4 điểm: đúng nguồn; đúng status; nối đúng task; nêu giới hạn/chờ xác minh.

## 12. Gate của buổi

Qua Chặng 4 khi matrix đủ D1–D6/P1–P4, mỗi hàng có status, nguồn và revision dữ liệu làm căn cứ hoặc nhãn `CẦN XÁC MINH`; sinh viên biết lỗi nào chuyển Chặng 5 và không sửa frozen spec/code trong buổi.

## 13. Lỗi thường gặp & cách gỡ

- **Evidence:** ghi file tồn tại là đạt → thêm test/artifact hoặc `CẦN XÁC MINH`.
- **Spec:** dùng code cũ làm chuẩn → ưu tiên V1.1 frozen.
- **Planning:** dùng A3 draft → đánh dấu chờ chốt.
- **Scope:** sửa ngay để matrix đẹp → chỉ lập kế hoạch task.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T4.2
Hàng §4 đã truy vết:
D/P đã lập matrix:
Status và bằng chứng:
Task chuyển Chặng 5:
CẦN XÁC MINH:
Hiểu/chưa hiểu:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc T5.0/T5.1; chuẩn bị fixture nhỏ và phân biệt validator với producer.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: interface mới trước khi viết schema/producer.
- CẦN XÁC MINH: test split và manifest nào sẽ được tạo ở Chặng 5–6.
- CẦN XÁC MINH: A3 trước production selection.
