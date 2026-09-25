---
buoi: 15
chang: 5
ten_chang: Schema và dữ liệu V1.1
tuan_goi_y: 5
thu_goi_y: Thứ 6 (lịch gợi ý, có thể đổi)
tasks: [T5.6, T5.7]
moi_truong: [CPU, Fixture, Local]
thoi_luong: 150 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3.d, §2.6 và §Nhật ký quyết định V1.1
  - docs/learning/00-project-analysis.md §4. Hiện trạng module: giữ / sửa / viết mới / build lại
  - docs/learning/lessons/chang-05/the-task-T5.6.md §2. Files và interface đề nghị
  - docs/learning/lessons/chang-05/doi-chieu-T5.6.md §2. Mốc thật
trang_thai: chưa làm
---

# Buổi 15 — Corpus base D6, title exclusions P4 và distractor fixture

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- phân loại nguồn được phép vào base theo D6;
- chuẩn hóa title theo P4 trước dedup và lập tập loại trừ;
- tính tay fixture distractor theo lịch/order/collision mà không tải Wikipedia;
- phân biệt acceptance logic trên fixture với build mid/large thật.

## 2. Vị trí trong bức tranh lớn

Buổi 15 chia nội bộ T5.6 (nguồn base/title) và T5.7 (distractor fixture). Đây là điểm nặng nên chỉ dùng fixture CPU; logic phải đúng trước T5.8 build/acceptance. Sau gate có **nhánh thực thi T5.6/T5.7**: ghép nguồn/title bằng CorpusBuilder đã có, rồi viết distractor producer và test. T5.6 có [thẻ chờ duyệt](the-task-T5.6.md); T5.7 cần thẻ riêng trước code. Không tải Wikipedia trong giờ hướng dẫn.

## 3. Tiền đề

- Buổi 14: positive chunk/group.
- Biết normalize NFC/lower/strip và dedup.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ | Sai lầm |
|---|---|---|
| D6 base | Chunk mọi context GermanQuAD + positive context GermanDPR. | Đưa hard negative vào corpus vì có ID. |
| P4 title exclusion | NFC → `_` thành space → lowercase → gộp space → strip. Fixture `Ä_B`, ` ä b `, `A__B`, title thiếu cho 2 key chuẩn hóa (`ä b`, `a b`) và 1 trường hợp cần báo. | Lọc title sau dedup hoặc lọc theo hard-negative list. |
| Distractor | Wikipedia passage theo chuỗi cố định, không phải nguồn gold. | Tải snapshot thật trong fixture. |
| Acceptance fixture | Kiểm order/collision/dedup/prefix. | Gọi fixture pass là base thật đạt. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–25 | Giải thích | D6/P4 bằng title fixture. |
| 25–50 | Xem input | Spec và hiện trạng CorpusBuilder. |
| 50–95 | T5.6 fixture | Tính source/title exclusion trước dedup. |
| 95–130 | T5.7 fixture | Chạy logic bằng tay, không download. |
| 130–150 | Tự giải thích | Gate và nhật ký. |

## 6. Input

- Spec §2.3.d/§2.6/P4.
- `src/corpus.py`/CorpusBuilder chỉ đọc.
- [Thẻ T5.6 §6–7](the-task-T5.6.md) có fixture/source_report đề nghị; [đối chiếu T5.6](doi-chieu-T5.6.md) có mốc cache và lệnh tái kiểm, đọc khi cần đối chiếu ngoài fixture.
- Fixture titles và distractor IDs do bài cung cấp: 4 title input (`Ä_B`, ` ä b `, `A__B`, rỗng) và 3 ID minh họa theo thứ tự cố định `[fixture-distractor-01, fixture-distractor-02, fixture-distractor-03]`; đây không phải ID Wikipedia thật. Kỳ vọng title: hai dòng đầu cùng key `ä b`, `A__B` thành `a b`, title rỗng không được tự đặt.

## 7. Thao tác từng bước

1. Đọc CorpusBuilder/source priority:

   ```powershell
   Select-String -LiteralPath 'src/corpus.py' -Pattern 'SOURCE_PRIORITY|class CorpusBuilder|def records|title|parent_document_id'
   ```

   Kỳ vọng: SOURCE_PRIORITY có GQ/DPR positive/Wikipedia; CorpusBuilder đã lưu 9 field gồm title và provenance. Phần còn thiếu là ghép mọi nguồn base và thu title P4 trước dedup, không phải viết lại primitive. [src/corpus.py §CorpusBuilder]
2. Lập bảng source giữ/loại: GermanQuAD context, GermanDPR positive, GermanDPR hard negative, Wikipedia distractor.
3. Chuẩn hóa fixture titles `Ä_B`, ` ä b `, `A__B`, title thiếu; kỳ vọng có 4 input, 2 key không trùng (`ä b`, `a b`) và 1 trường hợp title thiếu cần báo. Sau đó dự đoán fixture hai nguồn cùng text dưới đây: title DPR có còn thấy nếu chỉ đọc passage đại diện không?

   Công thức P4 inline chỉ để minh họa; chưa gọi API T5.6 mới. Khi copy từ file nguồn, đưa hai dấu here-string về cột 0.

   ```powershell
   @'
   import json, re, unicodedata
   from src.corpus import CorpusBuilder
   def p4(t):
       return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', t).replace('_', ' ').lower()).strip()
   text = '\u00c4_B\nBerlin'
   titles = ['\u00c4_B', 'Zweite_Quelle']
   before = sorted({p4(t) for t in titles})
   b = CorpusBuilder()
   b.add_text(text, 'germanquad')
   b.add_text(text, 'germandpr_positive', title=titles[1])
   after = sorted({p4(r['title']) for r in b.records()})
   print('before', json.dumps(before))
   print('after', json.dumps(after))
   print('passages', len(b.records()), b.records()[0]['passage_id'])
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: `before ["zweite quelle", "\u00e4 b"]`; `after ["\u00e4 b"]`; `passages 1 65aabd0ccb0e2bbf`. JSON hiển thị `\u00e4` là ký tự `ä`. Title DPR bị mất khi chỉ đọc đại diện. Trên cache, tình huống này cho **565 title trước dedup → 388 sau dedup, mất 177**; không dùng số cache làm hằng số trong code. [doi-chieu-T5.6.md §2. Mốc thật]
4. Dự đoán chuỗi distractor nhỏ theo R/order/prefix/collision với đúng thứ tự `[fixture-distractor-01, fixture-distractor-02, fixture-distractor-03]`; đánh dấu exact title bị loại.
5. Viết test-first expected cho T5.6/T5.7: source, title, dedup, order, collision, prefix.
6. Không chạy download/streaming; fixture không chứng nhận base 7,5K/mid 50K/large 150K.
7. **Nhánh thực thi sau gate (không tính vào 150 phút):** thẻ T5.6 đề nghị `src/corpus_builder.py`/`tests/test_corpus_builder.py` để reuse CorpusBuilder và thêm source_report; chờ duyệt interface/report trước code. T5.7 theo catalog dự kiến `src/distractors.py`/`tests/test_distractors.py`, cần chốt thẻ riêng. Chỉ sau test pass mới chuyển T5.8. Mọi build artifact vẫn cần manifest/revision và không được suy ra từ fixture. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi]

## 8. Dự đoán trước, kiểm tra sau

1. Hard negative có vào base không? **Không nếu chỉ là hard-negative-only; D6 giữ GQ context và DPR positive.**
2. `Ä_B` và ` ä b ` sau normalize bị giữ hay loại? **Cùng key `ä b`, nên dedup/exclusion theo P4.**
3. Fixture distractor pass có chứng minh Wikipedia thật đã build không? **Không; chỉ chứng minh logic fixture.**

## 9. Output cần có

- Bảng D6 source và P4 title exclusion trước dedup, kèm fixture giữ cả `zweite quelle`/`ä b` và giải thích vì sao title đại diện thiếu một key.
- Distractor fixture expected và test-first table.
- Danh sách điều chưa được phép kết luận về mid/large.
- Sau nhánh thực thi: wrapper base/source_report, distractor producer và test log; không ghi artifact corpus thật trong giờ hướng dẫn. Bốn nhóm source_report và metadata thiếu title theo thẻ T5.6 §5.

## 10. Kiểm tra hiểu bài

1. D6 cho base nguồn nào? **GQ context + DPR positive.** Chấm: loại hard-negative-only.
2. P4 normalize theo thứ tự nào? **NFC, `_`, lower, gộp space, strip.** Chấm: đúng thứ tự.
3. Title thiếu xử lý sao? **Base context vẫn tạo chunk, thiếu title được báo và không thêm key rỗng vào exclusions; distractor thiếu title giữ để xử lý tiếp và báo.** Chấm: không tự đặt title, phân biệt title với text rỗng.
4. Fixture pass nghĩa gì? **Logic fixture đúng, chưa nghiệm thu corpus thật.** Chấm: phân biệt acceptance.
5. Vì sao title exclusions có thể nhiều hơn title từ passage đại diện? **Dedup giữ một đại diện, nhưng hai nguồn cùng ID có thể mang title khác.** Chấm: nêu fixture `Zweite_Quelle`, thu title trước dedup; không suy số title bằng số passage.

## 11. Bài tự giải thích

Giải thích vì sao lọc title phải làm trước dedup và không liên quan danh sách hard negative.

Rubric 4 điểm: D6; P4; thứ tự; giới hạn fixture.

## 12. Gate của buổi

Qua giờ hướng dẫn khi fixture D6/P4/order/collision đúng, giải thích được title trước dedup và không tải Wikipedia. Chỉ sau khi nhánh thực thi T5.6/T5.7 pass mới chuyển T5.8; thẻ chờ duyệt không thay test code hoặc gate học viên.

## 13. Lỗi thường gặp & cách gỡ

- **Spec:** đưa hard negative vào corpus → đọc D6.
- **Title:** normalize sai thứ tự → viết từng bước.
- **Scope:** tải dữ liệu thật → dừng ở fixture.
- **Acceptance:** gọi fixture là base → ghi giới hạn.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T5.6, T5.7
Lệnh chỉ-đọc đã chạy:
Output path:
Source base:
Title exclusions:
Title trước/sau dedup và source_report:
Distractor fixture:
Nhánh thực thi đã chạy/chưa chạy:
Hiểu/chưa hiểu:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc T5.8 về reports, namespace và acceptance base.

## 16. Điểm cần xác minh

- Chờ duyệt wrapper/API và source_report trong thẻ T5.6 §2/§5; CorpusBuilder hiện hữu đã có provenance/priority.
- CẦN XÁC MINH: có cần audit/amendment riêng cho 881 title DPR bằng toàn context không — chủ project quyết định; T5.6 theo V1.1 giữ field raw rồi P4, không cắt dòng đầu. [doi-chieu-T5.6.md §2. Mốc thật]
- Lịch R/prefix đã có ở spec §2.3.d; CẦN XÁC MINH revision Wikipedia và kết quả thực tế trước streaming, không coi fixture là nghiệm thu T9.3.
