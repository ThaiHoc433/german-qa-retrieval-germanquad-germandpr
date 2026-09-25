---
buoi: 13
chang: 5
ten_chang: Schema và dữ liệu V1.1
tuan_goi_y: 5
thu_goi_y: Thứ 2 (lịch gợi ý, có thể đổi)
tasks: [T5.3, T5.4]
moi_truong: [CPU, Fixture, Local]
thoi_luong: 150 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án và §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.2 và §2.3.a–c
  - docs/learning/00-project-analysis.md §6. Hợp đồng dữ liệu và giao thức cần giữ
  - docs/learning/lessons/chang-05/the-task-T5.3.md §2. Files, caller và interface
trang_thai: chưa làm
---

# Buổi 13 — Gold evidences, references và collision

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- tạo bảng expected cho evidence offset và toàn bộ reference answers;
- phân biệt occurrence 0/1/>1 và chọn cách ghi collision;
- giữ record gold rỗng thay vì xóa;
- đối chiếu interface producer trong thẻ và nối T5.3 với T5.4 mà không làm mất provenance.

## 2. Vị trí trong bức tranh lớn

Buổi 13 chia nội bộ thành T5.3 (evidence/reference) và T5.4 (collision). Gold evidence là cầu nối từ offset Buổi 6 sang Recall/Evaluation. [Thẻ T5.3/T5.4](the-task-T5.3.md) chốt triển khai gộp trong `src/qa_records.py`: char-offset là mặc định, helper T5.4 xử lý fallback đúng điều kiện và ghi report. Sau gate có **nhánh thực thi T5.3/T5.4**; giờ hướng dẫn không tạo artifact production. Code đã được kiểm riêng không có nghĩa sinh viên đã hoàn thành bài.

## 3. Tiền đề

- Buổi 6: slice offset và multi-reference.
- Buổi 11–12: validator và row conservation.

## 4. Khái niệm cốt lõi

| Khái niệm | Fixture minh họa | Sai lầm |
|---|---|---|
| Evidence | passage_id, answer_text, start/end trong chunk. | Lưu offset context gốc vào chunk. |
| Reference answers | Toàn bộ answer hợp lệ cho EM/F1 max-over-references. | Chỉ giữ primary answer. |
| Collision 0 | Không tìm occurrence trong đại diện khác text nhưng cùng hash; T5.4 bỏ evidence đó, giữ QARecord và ghi report. | Xóa record hoặc không log. |
| Collision 1 | Map thẳng occurrence. | Ghi thiếu source position. |
| Collision >1 | Khi cần fallback: luôn ghi report; chọn occurrence có vị trí tương đối gần offset gốc, kể cả khi lựa chọn rõ. | Search lại mọi answer nhiều occurrence dù char-offset vẫn đúng. |
| Dedup collision | Hai text khác nhau sau chuẩn hóa cho cùng `passage_id`; đây là điều kiện để dùng fallback, còn mặc định luôn map bằng char-offset. | Gọi mọi duplicate passage là lỗi offset. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích | Evidence/reference/collision. |
| 20–45 | Xem input | Spec §1.2/§2.3 và record test Buổi 6. |
| 45–85 | T5.3 fixture | Tính evidence/local offset và reference union. |
| 85–125 | T5.4 fixture | Xử lý 0/1/>1 occurrence, viết report expected. |
| 125–150 | Tự giải thích | Gate và nhật ký. |

## 6. Input

- GermanQuAD test record nhiều reference từ Buổi 6.
- [Thẻ T5.3](the-task-T5.3.md): signature producer, fixture overlap và các **cặp** context QA/đại diện có cùng hash nhưng khác text, với 0/1/2 occurrence. Không dùng hai text khác nội dung chuẩn hóa như `"alpha omega"` và `"alpha Berlin omega"` làm collision.
- Audit toàn raw GQ trước D1: 1.285 annotation answer nhiều occurrence; 575 (44,7%) có `answer_start` gốc không phải occurrence đầu. Đây không phải số case remap sau D1. Sau D1 cần giữ 12.696 QARecord, kể cả gold rỗng. [the-task-T5.3.md §6. Mốc dữ liệu thật và giới hạn bằng chứng]
- Spec §1.2/§2.3.a–c.
- `src/qa_records.py`, `tests/test_qa_records.py` và [log kiểm chứng](kiem-chung-T5.3-T5.4.md): producer thuần, fixture có expected, mốc 158/16/8 gold rỗng đã đo trên cache; artifact trên đĩa vẫn chờ T5.8.

## 7. Thao tác từng bước

1. Đọc các hàm/field cũ bằng PowerShell:

   ```powershell
   Select-String -LiteralPath 'notebooks/01_build_corpus_base.py' -Pattern 'gold_indices|gold_passage_ids|answers|build_qa_records'
   ```

   Kỳ vọng: thấy hai vấn đề riêng: producer cũ bỏ dòng không có gold_indices và chỉ giữ answers của primary chunk. `uncovered_rows=184` được đếm trước primary, không quy số này cho luật primary. Coverage mới phải đo lại từng file sau D1 và remap.
2. Fixture-first: answer xuất hiện đúng một lần trong chunk; viết expected evidence và `reference_answers`.
3. Fixture nhiều reference: giữ toàn bộ text, evidence map được là tập con; không xóa reference không map.
4. Collision 0: context QA `"alpha Berlin omega"`, answer Berlin/start=6; đại diện `"alpha berlin omega"`. Cùng hash nhưng 0 occurrence chính xác. T5.4 bỏ evidence này và **ghi report**, giữ QARecord kể cả gold rỗng. `candidate_offsets=[]`, `selected_offset=None`; không raise exception chuyển tiếp khi đây là collision hợp lệ.
5. Collision 1: context QA `"alpha  Berlin omega"`, answer/start=7; đại diện `"alpha Berlin omega"` có ứng viên 6. T5.4 lấy span [6,12), không cần log; đối chiếu hash chung trong thẻ.
6. Collision >1: context QA `"alpha  Berlin omega Berlin end"` dài 30, raw_start=20; đại diện `"alpha Berlin omega Berlin end"` dài 29 có ứng viên [6,19]. So với 20/30, 19/29 gần hơn 6/29 nên chọn 19. **Luôn ghi report cho fallback >1**, không đợi đến khi chưa quyết định được. Ngược lại, context/đại diện giống hệt với raw_start=19 thì dùng char-offset 19, không search hoặc log chỉ vì có nhiều occurrence.
7. Ghi điều kiện fallback: text nguồn/đại diện khác nhưng cùng normalize/hash và slice trực tiếp không khớp. Khi trừ offset, dùng chunk của chính context QA; không trừ chunk_start của parent đại diện khác. Thiếu passage hoặc raw offset sai là lỗi input, không phải lý do tự search.
8. Lập test table theo [thẻ T5.3/T5.4](the-task-T5.3.md), đối chiếu `build_qa_records` và `_remap_offset`. Tie-break đã chốt: hai khoảng cách tương đối bằng nhau thì chọn offset nhỏ hơn. Fixture `"A  a     "` / `"a a"`, answer a/start=3: document dài 9, đại diện dài 3, ứng viên [0,2] cách đều 3/9 → chọn 0. Report luôn có cả hai ứng viên và provenance. Notebook chỉ chuyển caller ở T5.8.
9. Dự đoán rồi chạy fixture chỉ trong bộ nhớ; here-string mở/đóng phải ở cột 0 khi copy từ file Markdown nguồn:

   ```powershell
   @'
   from src.corpus import CorpusBuilder
   from src.splits import assign_splits
   from src.qa_records import build_qa_records
   row = {'id':1, 'question':'Frage A', 'context':'alpha  Berlin omega Berlin end',
          'answers':{'text':['Berlin'], 'answer_start':[20]}}
   decisions, _ = assign_splits([row], [], [], [], dev_ratio=1)
   builder = CorpusBuilder()
   builder.add_text('alpha Berlin omega Berlin end', 'germanquad')
   passages = {p['passage_id']:p for p in builder.records()}
   result = build_qa_records([row], [], decisions, passages)
   rec = result['records']['dev'][0]
   remap = result['ambiguous_offset_remap_report'][0]
   print(rec['query_id'], rec['reference_answers'], rec['gold_evidences'][0]['answer_start'])
   print(remap['candidate_offsets'], remap['selected_offset'])
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: `gq-train-1 ['Berlin'] 19` và `[6, 19] 19`. Nếu khác, đối chiếu raw_start=20, độ dài 30/29 và cặp text; không đổi expected để hợp code. Toàn bộ test/code thuộc nhánh thực thi; giờ hướng dẫn tập trung tính tay và giải thích, không lặp triển khai hai module trong 150 phút.

## 8. Dự đoán trước, kiểm tra sau

1. Gold rỗng có bị xóa không? **Không; giữ QARecord, chỉ loại khỏi N_eval.**
2. Hai occurrence giống nhau xử lý thế nào? **Nếu char-offset trực tiếp đúng thì giữ span gốc, không search. Chỉ trong fallback collision mới chọn theo vị trí tương đối và luôn ghi report cho >1 ứng viên.**
3. Reference không map được có giữ không? **Có; giữ toàn bộ reference cho EM/F1, chỉ evidence không map bị bỏ.**
4. Khi nào fallback được phép dùng? **Text đại diện khác nguồn nhưng cùng normalize/hash và slice trực tiếp không còn khớp. Mặc định vẫn map char-offset; helper T5.4 xử lý ngay trong producer. Nhiều occurrence nhưng slice đúng không được tạo remap report.**

## 9. Output cần có

- Bảng expected QARecord cho 0/1/>1 occurrence.
- Bảng references/evidence và mẫu `ambiguous_offset_remap_report.json`.
- Bảng đối chiếu interface và ranh giới T5.3/T5.4 theo thẻ; 12.696 dòng giữ lại sau D1, với mốc gold rỗng 158/16/8 chỉ cho snapshot đã kiểm, không gán trước cho snapshot khác.
- Sau nhánh thực thi: QARecord/evidence producer, remap report và test T5.3/T5.4; số audit 1.285/575 chỉ là bằng chứng đầu vào, chưa phải gate pass.

## 10. Kiểm tra hiểu bài

1. Evidence offset nằm ở đâu? **Trong chính chunk.** Chấm: nêu local offset.
2. Reference không map được có mất không? **Không.** Chấm: giữ cho EM/F1.
3. Collision >1 cần gì? **Trong fallback, chọn theo relative offset và luôn ghi report.** Chấm: phân biệt với nhiều occurrence nhưng char-offset đã đúng.
4. Gold rỗng vào N_eval? **Không.** Chấm: record vẫn lưu.

## 11. Bài tự giải thích

Giải thích vì sao “record giữ lại” và “record vào mẫu số metric” là hai quyết định khác nhau.

Rubric 4 điểm: phân biệt lưu/chấm; đúng offset; giữ references; report collision.

## 12. Gate của buổi

Qua giờ hướng dẫn khi expected T5.3/T5.4 bao phủ collision 0/1/>1, tie-break, đầy đủ references và gold rỗng; sinh viên giải thích được khi nào dùng char-offset, khi nào fallback và field report. Chỉ sau khi nhánh thực thi có test remap/report pass mới được chuyển T5.5. Kiểm cache của producer không thay nghiệm thu base/file report ở T5.8; trạng thái người học vẫn chưa làm.

## 13. Lỗi thường gặp & cách gỡ

- **Offset:** nhầm context/chunk → kiểm provenance.
- **Data:** bỏ reference → giữ toàn bộ.
- **Spec:** gọi mọi collision là lỗi fatal hoặc lấy occurrence đầu → áp dụng quy tắc relative offset và dedup collision.
- **Scope:** chạy notebook cũ ngay → dùng producer thuần/fixture; chuyển caller và ghi artifact thuộc T5.8.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T5.3, T5.4
Lệnh chỉ-đọc đã chạy:
Output path:
Fixture collision:
Evidence/reference expected:
Interface và ranh giới T5.3/T5.4 đã hiểu/chưa hiểu:
Report cần có:
Nhánh thực thi đã chạy/chưa chạy:
Hiểu/chưa hiểu:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc positive occurrence union và hard-negative groups.

## 16. Điểm cần xác minh

- Signature producer, tie-break nhỏ hơn và field report đã chốt ở thẻ §2/§4, code nằm trong `src/qa_records.py`.
- CẦN XÁC MINH: coverage/report trên file artifact thật, revision và thứ tự chọn đại diện khi tích hợp — người triển khai đối chiếu mốc cache trong log khi làm T5.8.
- CẦN XÁC MINH: caller ngoài src/tests/notebooks — kiểm trước tích hợp; không suy số collision từ 1.285 answer nhiều occurrence trong audit.
