---
buoi: 14
chang: 5
ten_chang: Schema và dữ liệu V1.1
tuan_goi_y: 5
thu_goi_y: Thứ 4 (lịch gợi ý, có thể đổi)
tasks: [T5.5]
moi_truong: [CPU, Fixture, Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §1.3 và §5.3
  - docs/learning/00-project-analysis.md §4. Hiện trạng module: giữ / sửa / viết mới / build lại
  - docs/learning/lessons/chang-05/the-task-T5.5.md §2. Interface đề nghị
  - docs/learning/lessons/chang-05/doi-chieu-T5.5.md §2. Mốc đối chiếu
trang_thai: chưa làm
---

# Buổi 14 — Positive occurrence union và hard-negative groups

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- hợp mọi occurrence answer-bearing thành positive chunk set;
- nhóm hard negative theo context gốc;
- giải thích vì sao một context dài chỉ đóng góp tối đa một chunk/group mỗi epoch;
- lập mask `gold_set` không làm positive của query hiện tại thành negative.

## 2. Vị trí trong bức tranh lớn

Buổi 14 là T5.5, nối positive occurrence với RetrievalTrainingRecord và Dense sampling sau này. Chỉ dùng fixture CPU; không train và không tạo record production trong giờ hướng dẫn. Sau gate có **nhánh thực thi T5.5** theo [thẻ T5.5](the-task-T5.5.md). Code đã review/commit `945f317`; người học vẫn chưa làm, dùng log để đối chiếu khi tự thực hiện. Sampler chọn một positive và tối đa một chunk/group/epoch thuộc module Dense sau này; T5.5 lưu toàn bộ candidate hợp lệ. [kiem-chung-T5.5.md §5. Vá test sau review và commit]

## 3. Tiền đề

- Buổi 7: raw/processed DPR và chunk count.
- Buổi 13: occurrence/evidence.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ | Sai lầm |
|---|---|---|
| Positive union | Hợp ID của mọi chunk chứa trọn occurrence. Record DPR đầu có 3 chunk nhưng chỉ chunk index 2 chứa trọn answer → 1 positive. | Gán mọi chunk positive context là positive. |
| Context group | Mọi chunk cùng `source_context_id`; ba chunk hard-negative của một context vẫn là một group. | Mỗi chunk thành một group. |
| Sampling cap | Tối đa một chunk/group/epoch. | Context dài được nhân trọng số. |
| `gold_set` mask | Loại toàn bộ gold IDs của query khỏi hard negatives. | Chỉ mask positive đang chọn. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích | Positive union/group/mask. |
| 20–45 | Xem input | Spec §1.3/§5.3 và record Buổi 7. |
| 45–85 | Fixture | Tính union, group, mask bằng tay trước code. |
| 85–105 | Kiểm tra | Đối chiếu expected và lỗi false-negative. |
| 105–120 | Tự giải thích | Nhật ký/gate. |

## 6. Input

- GermanDPR raw/processed record đã đọc: record đầu có 1 positive context 230 từ → 3 chunk, **1 positive hợp lệ** ID `d20519f819d35708`, text dài 394 ký tự. Ba hard-negative contexts (212/216/101 từ) → `3+3+2=8` chunk trong 3 group. [doi-chieu-T5.5.md §2. Mốc đối chiếu]
- [Thẻ T5.5 §7](the-task-T5.5.md#7-fixture-và-expected-trước-code): chuỗi 201 token cụ thể, fixture nhiều occurrence/mọi answers và mask toàn gold_set.
- Cache và lệnh chỉ đọc cho toàn bộ DPR train/test trong [đối chiếu §3](doi-chieu-T5.5.md#3-lệnh-tái-kiểm-chỉ-đọc); đó là baseline trước producer. Kết quả producer thật ở [log kiểm chứng](kiem-chung-T5.5.md).

## 7. Thao tác từng bước

1. Đọc producer cũ:

   ```powershell
   Select-String -LiteralPath 'notebooks/01_build_corpus_base.py' -Pattern 'build_retrieval_training_records|positive|hard_negative'
   ```

   Kỳ vọng: nhận ra flatten/gán mọi chunk là lệch T5.5.
2. Fixture nhiều answers trên context 201 token: `w010` nằm trong chunk 0, `w185` nằm trong chunk 2; chunk 1 không được gán positive. Dự đoán rồi chạy phép tính chỉ đọc (here-string phải ở cột 0 khi copy từ nguồn Markdown):

   ```powershell
   @'
   from src.corpus import chunk_text, find_span_chunks, passage_id
   text = ' '.join(f'w{i:03d}' for i in range(201))
   chunks = chunk_text(text)
   indices = set()
   for answer in ['w010','w185']:
       start = text.find(answer)
       while start != -1:
           indices.update(find_span_chunks(chunks,start,start+len(answer)))
           start = text.find(answer,start+1)
   gold = {passage_id(chunks[i].text) for i in indices}
   print('positive_indices',sorted(indices))
   print('positive_ids',[passage_id(chunks[i].text) for i in sorted(indices)])
   for context in [text,'Paris']:
       kept = [c for c in chunk_text(context) if passage_id(c.text) not in gold]
       print('group',passage_id(context),[passage_id(c.text) for c in kept])
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: indices `[0,2]`, IDs `c3cbe252c56e270c` và `78eaa148763a161a`. Group context 201 token chỉ còn ID chunk 1 `8451b74d508323dc`; group Paris còn `1670f2e42fefa504`. Nếu khác, kiểm answers, cửa sổ/stride và mask toàn gold_set; không đổi expected.
3. Từ fixture nhiều occurrence Berlin trong thẻ §7, tính lại hợp: hai occurrence ở start 50/927 thuộc chunk 0/2. Đối chiếu record thật đầu: 3 chunk sinh ra nhưng **1 positive**; hard negatives vẫn 3 context → 8 chunks → 3 groups. Không suy số positive từ số chunk context.
4. Fixture query `gold_set={p1,p2}` với candidate hard negatives chứa p1/p2: mask loại cả hai; ghi expected còn lại.
5. Viết sampling table minh họa cho module Dense sau này: group, mọi candidate lưu ở T5.5, tối đa một chunk sẽ được chọn/epoch, lý do. Không triển khai sampler trong task này.
6. Ghi test-first cases: two occurrences, unmappable, one context=one group, false-negative mask.
7. **Nhánh thực thi sau gate (không tính vào 120 phút):** producer/test đã có trong `src/retrieval_records.py` và `tests/test_retrieval_records.py`; input hai split DPR raw + SplitDecision + corpus map. Người học tái hiện test-first theo thẻ, không sửa notebook trước T5.8. Đối chiếu producer với mốc 8.329/926/1.023 record và 24.972/2.776/3.069 group; không dùng phép tính baseline thay test producer.

## 8. Dự đoán trước, kiểm tra sau

1. Hai occurrence hợp lệ tạo mấy positive? **Không cố định: hợp mọi chunk chứa trọn rồi dedup ID. Fixture Berlin trong thẻ cho hai ID; hai occurrence cùng chunk có thể chỉ một ID, overlap có thể thêm chunk.**
2. Ba chunk cùng context tạo mấy group? **Một group.**
3. `gold_set` có chứa positive chưa chọn không? **Có; mask toàn bộ gold_set.**

## 9. Output cần có

- Bảng positive union/group/mask.
- Test-first table T5.5, kèm đối chiếu record đầu `3 chunk → 1 positive`, `8 hard-negative chunks`, `3 groups`.
- Bảng đối chiếu input/interface và report trong thẻ: phân biệt loại D1 với loại unmappable; corpus map chỉ bắt buộc positive ID.
- Sau nhánh thực thi: RetrievalTrainingRecord producer, grouping/mask/report và test fixture; sampler theo epoch thuộc module Dense.

## 10. Kiểm tra hiểu bài

1. Positive union làm gì? **Giữ mọi chunk có occurrence.** Chấm: không lấy một chunk.
2. Group theo gì? **Context gốc.** Chấm: dùng source_context_id.
3. Vì sao cap một chunk/group? **Cân trọng số context.** Chấm: không nhân mẫu.
4. Mask gì? **Toàn bộ gold_set hiện tại.** Chấm: không false negative.

## 11. Bài tự giải thích

Vẽ từ raw context tới batch sampling với group và mask.

Rubric 4 điểm: union; group; cap; mask.

## 12. Gate của buổi

Qua giờ hướng dẫn khi tính đúng fixture union/group/mask, phân biệt 3 chunk với 1 positive trên record đầu và ghi test-first expected. Chỉ sau khi nhánh thực thi T5.5 pass mới chuyển T5.6; số mô phỏng trong thẻ chưa phải nghiệm thu producer. Trạng thái sinh viên vẫn chưa làm.

## 13. Lỗi thường gặp & cách gỡ

- **Data:** flatten hard negative → nhóm source context.
- **Sampling:** chọn nhiều chunk/group → áp cap.
- **Mask:** chỉ mask positive chọn → dùng toàn gold_set.
- **Scope:** train Dense → chỉ fixture CPU.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T5.5
Lệnh chỉ-đọc đã chạy:
Output path:
Positive union:
Groups:
Gold mask:
Nhánh thực thi đã chạy/chưa chạy:
Hiểu/chưa hiểu:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc D6/P4 và title normalization trước dedup.

## 16. Điểm cần xác minh

- Interface, lựa chọn positive_texts từ raw chunk và format report đã được review/commit theo thẻ §2/§6; đối chiếu [log kiểm chứng §5](kiem-chung-T5.5.md#5-vá-test-sau-review-và-commit), không đánh dấu gate sinh viên thay người học.
- CẦN XÁC MINH: mốc row/group/chunk bằng producer thật sau triển khai — đối chiếu lệnh và bảng trong doi-chieu-T5.5.md.
