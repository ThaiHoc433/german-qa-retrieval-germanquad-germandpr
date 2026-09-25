---
buoi: 20
chang: 7
ten_chang: Dense Retriever
tuan_goi_y: 7
thu_goi_y: Thứ 4 (lịch gợi ý, có thể đổi)
tasks: [T7.1]
moi_truong: [Đọc tài liệu, Local, CPU]
thoi_luong: 150 phút hướng dẫn và fixture; triển khai module là nhánh riêng
chay_may_rieng: không; buổi này không train
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 7 — Dense Retriever
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.3 Quy tắc xử lý
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.6 Kiểm thử
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §RQ1 — Ảnh hưởng của quy mô corpus
  - src/corpus.py §chunk_text
  - src/corpus.py §find_span_chunks
  - docs/learning/lessons/chang-05/kiem-chung-T5.5.md §3. Cache thật và đối chiếu mốc §8
trang_thai: chưa làm
---

# Buổi 20 — Từ vector đến batch, target và mask toàn gold_set

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- tính dot product của vector nhỏ và xác định kích thước ma trận score;
- chuyển một record DPR thành một positive được chọn và tối đa một chunk/group;
- đánh dấu target và loại đúng false negative bằng toàn bộ gold_set, vẫn giữ negative hợp lệ của query khác;
- tính NLL trên fixture và giải thích vì sao fixture pass chưa cho phép train Dense thật.

## 2. Vị trí trong bức tranh lớn

Dense Retriever tìm passage bằng vector thay cho điểm từ vựng BM25, phục vụ so sánh RQ1 và phân tích retrieval bottleneck RQ2. Dual encoder là hai bộ mã hóa độc lập cho question và passage, cùng khởi tạo từ `deepset/gbert-base`; spec dùng CLS pooling và dot product thô, không normalize embedding. Bài này đặt nền cho sampling/mask T7.1, chưa tạo encoder hoặc checkpoint. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.1 Mục tiêu & phạm vi] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.3 Quy tắc xử lý] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §RQ2 — Bottleneck dịch chuyển]

## 3. Tiền đề

- Buổi 14: positive union, hard-negative groups và gold_set; buổi 18: metric có mẫu số, loss không phải Recall.
- Ôn list/set và phép nhân/cộng Python; chưa cần biết PyTorch. Đọc định nghĩa mới ở §4 trước khi chạy code.
- Theo catalog, **thực thi T7.1 phụ thuộc T6.4**. Tại lúc soạn, chưa có base V1.1 và run BM25 V1.1 nghiệm thu; chỉ học khái niệm/fixture độc lập, gate project vẫn chờ. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án]
- Snapshot môi trường/nguồn: [kiem-chung-chang-07.md §1. Nguồn và hiện trạng]. Soạn bài không tự phê chuẩn T5.6 hoặc mở nhánh triển khai Dense.

## 4. Khái niệm cốt lõi

| Khái niệm | Giải thích bằng ví dụ minh họa | Sai lầm hay gặp |
|---|---|---|
| Vector / dot product | Vector là dãy số; `[1,0] · [2,0] = 2`, biểu diễn một score. | Gọi dot product là cosine dù chưa chia độ dài. |
| CLS pooling / hai tower | CLS pooling lấy vector token CLS sau lớp cuối; hai tower có tham số độc lập. | Lấy trung bình token hoặc buộc hai tower dùng chung tham số. |
| Tensor / batch | Tensor là mảng nhiều chiều; 2 query × 5 passage cho ma trận score 2×5. | Nghĩ mỗi query chỉ nhìn passage do chính nó đóng góp. |
| Target / softmax / NLL | Target là một vị trí positive được chọn; softmax biến score thành phân phối, NLL = `-log(p_target)`. | Gán đồng thời mọi positive làm target của loss single-target. |
| Gold_set / false negative | `gold_set(qA)={A,B}`: B vẫn là gold dù epoch này chọn A. False negative là gold bị dùng nhầm làm negative. | Mask chỉ positive được chọn; hoặc mask mọi positive của query khác. |
| Group / sampling | Group 5 chunk chỉ chọn 1 chunk/epoch sau lọc gold; group hết chunk bỏ qua. | Context dài đóng góp cả 5 chunk, hoặc luôn giả định 3 group. |

Các quy tắc model/loss/mask lấy từ [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.3 Quy tắc xử lý]. Vector/ID chữ cái trong bài là minh họa, không phải embedding/PassageRecord thật.

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–25 | Giải thích bằng ví dụ | Vector, hai tower, batch và NLL. |
| 25–50 | Xem input | Kiểm môi trường, đọc lại raw DPR đầu tiên. |
| 50–100 | Thao tác từng lệnh | Dự đoán group, target/mask rồi chạy fixture. |
| 100–125 | Kiểm tra output | So ma trận/mẫu số, thử sai một mask trên giấy. |
| 125–150 | Tự giải thích và nhật ký | Rubric, gate, việc còn chờ. |

150 phút chỉ bao gồm nội dung trên. Báo cáo gap ước lượng buổi đầy đủ 180–240 phút; việc viết encoder/collator/test không được ngầm coi là đã làm trong 150 phút. [docs/learning/02-roadmap-gap-report.md §3. Đối chiếu từng buổi 1–32]

## 6. Input

- `src/corpus.py`, `src/retrieval_records.py`, `tests/test_retrieval_records.py` hiện có; chỉ đọc/reuse hàm công khai.
- Raw DPR train: `C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/train/0000.parquet`, các field `answers`, `positive_ctxs`, `hard_negative_ctxs`. [../chang-05/doi-chieu-T5.5.md §1. Nguồn và cách đo]
- Fixture tự tạo: qA gold `{A,B}`, target A; qB gold `{B}`, target B. Candidate theo thứ tự `[A,N,Z,B,A]`. ID chữ cái chỉ dùng tính tay, không đưa vào validator đòi hash.
- Input train tương lai là `data/processed/v1.1/retrieval_germandpr_train.jsonl` sau T5.8; **file chưa có**, không thay bằng processed cũ. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T5.8 — Phạm vi báo cáo và nghiệm thu]

## 7. Thao tác từng bước

Chạy PowerShell tại repo root; mỗi khối Python tự đủ biến. Khi copy Markdown nguồn, bỏ thụt lề để dấu mở `@'` và đóng `'@` đều ở cột 0. Các khối này không ghi file hoặc tải model.

1. **Kiểm tiền đề hiện có**, ghi kết quả vào nhật ký:

   ```powershell
   Test-Path '.venv/Scripts/python.exe'
   Test-Path 'data/processed/v1.1'
   Get-ChildItem 'src' -File -Filter '*.py' | Select-Object -ExpandProperty Name
   @'
   from importlib.metadata import version, PackageNotFoundError
   for name in ('torch','transformers','faiss-cpu'):
       try: print(name, version(name))
       except PackageNotFoundError: print(name, 'ABSENT')
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng lúc soạn: True, False; src có corpus/QA/DPR producer nhưng chưa có module Dense; ba package in ABSENT. Nếu hiện trạng thay đổi, ghi phiên bản/path thực, không cài đặt để ép khớp bài.

2. **Nối record thật với candidate**, dự đoán raw có bao nhiêu positive context và bao nhiêu chunk thực chứa answer:

   ```powershell
   @'
   import pyarrow.parquet as pq
   from src.corpus import chunk_text, find_span_chunks, passage_id
   path = 'C:/Users/PC/.cache/huggingface/hub/datasets--deepset--germandpr/snapshots/d0b867a49c45a44d172041351b5df73a87714372/plain_text/train/0000.parquet'
   r = pq.read_table(path, columns=['answers','positive_ctxs','hard_negative_ctxs']).to_pylist()[0]
   print('contexts', len(r['positive_ctxs']['text']), len(r['hard_negative_ctxs']['text']))
   text = r['positive_ctxs']['text'][0]
   chunks, indices = chunk_text(text), set()
   for answer in r['answers']:
       start = text.find(answer)
       while start != -1:
           indices.update(find_span_chunks(chunks, start, start+len(answer)))
           start = text.find(answer, start+1)
   print('positive_chunk_indices', sorted(indices))
   print('positive_ids', [passage_id(chunks[i].text) for i in sorted(indices)])
   print('hn_chunk_counts', [len(chunk_text(t)) for t in r['hard_negative_ctxs']['text']])
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: `contexts 1 3`; indices `[2]`; ID `d20519f819d35708`; HN counts `[3,3,2]`. T5.5 lưu đủ các chunk HN còn lại sau mask gold; sampler sau này chọn một/group. Nếu khác, kiểm snapshot, record index, schema struct các list và chunk 100/stride 80. Không lấy 3 positive của artifact cũ làm expected. [../chang-05/kiem-chung-T5.5.md §3. Cache thật và đối chiếu mốc §8]

3. **Tính tay lọc group trước sampling.** qA có ba group `[[A,N],[B],[Z,Z2,Z3,Z4,Z5]]`, gold `{A,B}`. Sau mask: `[[N],[],[Z,Z2,Z3,Z4,Z5]]`; group giữa không đóng góp. Kỳ vọng qA góp 1 target + 2 HN = 3 vị trí, dù group cuối dài 5. Để minh họa ma trận, chọn Z ở group cuối; đây là lựa chọn fixture cố định, **không thay chính sách RNG theo epoch** của buổi 21.

4. **Vẽ target/mask trước khi chạy.** qB góp target B và HN A: A không thuộc gold của qB nên hợp lệ. Ghép theo thứ tự qA rồi qB thành `[A,N,Z,B,A]`, target index `[0,3]` (zero-based). Với qA, giữ target index 0, mask B ở index 3 và A lặp ở index 4; với qB giữ cả 5 vị trí. Không xóa cột khỏi batch của mọi query chỉ vì nó bị mask ở một query.

   ```powershell
   @'
   import math
   ids = ['A','N','Z','B','A']
   Q = [[1,0],[0,1]]
   P = [[2,0],[0,2],[-1,0],[1,1],[2,0]]
   gold, targets = [{'A','B'},{'B'}], [0,3]
   groups = [['A','N'],['B'],['Z','Z2','Z3','Z4','Z5']]
   filtered = [[pid for pid in g if pid not in gold[0]] for g in groups]
   print('filtered_groups', filtered)
   print('qA_contributions', 1 + sum(bool(g) for g in filtered))
   for name,q,g,target in zip(['qA','qB'],Q,gold,targets):
       scores = [sum(a*b for a,b in zip(q,p)) for p in P]
       keep = [j for j,pid in enumerate(ids) if j == target or pid not in g]
       peak = max(scores[j] for j in keep)
       loss = math.log(sum(math.exp(scores[j]-peak) for j in keep)) + peak - scores[target]
       print(name, 'scores', scores, 'keep', keep, 'NLL', f'{loss:.6f}')
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: filtered như bước 3; `qA_contributions 3`. qA scores `[2,0,-1,1,2]`, keep `[0,1,2]`, NLL `0.169846`; qB scores `[0,2,0,1,0]`, keep `[0,1,2,3,4]`, NLL `1.573172`. qA dùng mẫu số softmax `exp(2)+exp(0)+exp(-1)`; qB dùng đủ 5 score. Nếu sai, kiểm target index, gold toàn bộ và giữ target trước điều kiện mask. Trừ peak chỉ ổn định phép tính softmax, không normalize embedding.

5. **Lập bảng test-first T7.1**: hai tower độc lập/CLS/dot thô; group ngắn/dài cùng góp một; group hết sau mask góp zero; positive query khác hợp lệ vẫn giữ; positive query khác thuộc gold hiện tại phải mask; target không bị mask. Expected lấy từ §3–4 và spec, không dựa vào implementation chưa có. Đây là bảng thiết kế test, **chưa phải test production đã chạy**.

6. **Nhánh thực thi T7.1:** sau T6.4 đạt gate và thẻ T7.1 được duyệt, chốt file/API encoder–sampler–collator cùng tên test, rồi RED/GREEN và kiểm §5.6. Không tự thêm module/train trong phiên soạn bài này. Nếu chưa đủ tiền đề, dừng phần thực thi; bài CPU vẫn cho phép nhận biết điều kiện còn thiếu. Thẻ phải chuyển bảng bước 5 thành test cụ thể trước sửa code.

## 8. Dự đoán trước, kiểm tra sau

1. Với qA, B đang là target của qB có được làm negative không? **Không: B thuộc toàn gold_set(qA).** Chấm: nêu quan hệ với query đang xét, không chỉ vai trò của B trong batch.
2. Với qB, A có bị loại vì là target của qA không? **Không: A không thuộc gold_set(qB).** Chấm: giữ cả hai occurrence A trong fixture này, không mask toàn bộ positive của query khác.
3. Group dài 5 chunk có làm qA góp thêm 5 HN không? **Không: góp một sau lọc; group [B] góp zero.** Chấm: tính tổng 3 vị trí do qA đóng góp.

## 9. Output cần có

Người học lưu tại **đường dẫn đề xuất, chưa tạo** `docs/learning/lessons/chang-07/thuc-hanh/buoi-20-nhat-ky.md`:

- inventory môi trường và tiền đề còn thiếu;
- record thật: positive index/ID và HN group sizes;
- bảng group trước/sau mask, ma trận 2×5, target, keep và hai NLL;
- bảng test-first T7.1 cùng điều kiện mở nhánh thực thi.

Sau thực thi riêng mới có module/test/log T7.1 theo thẻ được duyệt; không coi ghi chú là encoder hoặc dữ liệu train.

## 10. Kiểm tra hiểu bài

1. Dot product `[1,0]·[2,0]` bằng bao nhiêu? **2.** Chấm: nhân từng tọa độ rồi cộng.
2. Hai tower có phải một encoder dùng hai lần không? **Không theo V1.1: hai bộ tham số độc lập.** Chấm: phân biệt cùng checkpoint khởi tạo với chia sẻ tham số.
3. Vì sao positive chưa chọn vẫn cần mask? **Nó vẫn thuộc gold_set, nếu làm negative sẽ tạo tín hiệu học sai.** Chấm: dùng A/B trong fixture.
4. Tại sao không mask mọi passage của query khác? **DPR dùng chúng làm in-batch negative khi không thuộc gold hiện tại.** Chấm: chỉ được điều kiện theo từng query.
5. NLL qA có dùng score của B không? **Không; B bị mask.** Chấm: viết đúng ba số hạng mẫu số và giữ target A.
6. Fixture pass đã đủ chạy train thật chưa? **Chưa: T6.4/base/run, thẻ/API/test và môi trường còn gate riêng.** Chấm: nêu ít nhất hai bằng chứng còn thiếu.

## 11. Bài tự giải thích

Vẽ một batch hai query, kể đường đi của B và hai occurrence A khi tính loss cho từng query. Sau đó giải thích cách từ record DPR thật đi đến một target và ba HN đã chọn.

Rubric 4 điểm, mỗi tiêu chí 1: đúng input/group; target rõ index; mask theo toàn gold_set và theo từng query; tách fixture khỏi nghiệm thu encoder/train. Người học tự chấm, chưa ghi đạt thay người học.

## 12. Gate của buổi

Gate hiểu bài: dự đoán rồi giải thích đúng group/ma trận/target/mask/NLL; lưu output §9 và nêu tiền đề còn thiếu. Gate thực thi T7.1: T6.4 đã nghiệm thu, thẻ task chốt, test encoder/sampling/mask theo spec pass. **Không cần score cao; cần đúng giao thức.** Chưa tuyên bố Chặng 7 qua gate hay sinh viên hoàn thành.

## 13. Lỗi thường gặp & cách gỡ

- **Data:** lấy mọi chunk positive context → so lại raw record đầu và positive index 2.
- **Spec:** mask chỉ target hoặc mask mọi positive query khác → vẽ membership gold_set từng hàng.
- **Code:** target bị mask vì trùng ID → giữ đúng vị trí target; occurrence gold khác không làm negative.
- **Environment:** import torch lỗi → kiểm metadata bước 1; fixture bài này dùng stdlib, không tự cài.
- **Scope:** dùng artifact cũ thay V1.1 → kiểm namespace và gate T5.8/T6.4.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T7.1
Hiểu/chưa hiểu:
Lệnh đã chạy và môi trường:
Raw path/revision, positive ID, HN group sizes:
Group/matrix/target/mask/NLL dự đoán và thực đo:
Output path:
Lỗi và phân loại code/data/spec/environment:
Gate hiểu bài: chưa làm
Gate thực thi T7.1: chưa làm
Bước tiếp theo:
```

## 15. Việc chuẩn bị cho buổi sau

Ôn target/mask và phép softmax. Đọc spec §5.3 về SHA-256 seed, §5.8 về resume; chưa cài GPU hoặc tải checkpoint.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: base T5.8 và BM25 T6.4 đã nghiệm thu chưa — người triển khai cung cấp artifact, manifest và log.
- CẦN XÁC MINH: file/API/tên test T7.1 — chốt trong thẻ thực thi trước code; hiện repo chưa có module Dense.
- CẦN XÁC MINH: model/tokenizer revision và khả năng vector/softmax của sinh viên — chủ project chốt revision, giảng viên kiểm bằng câu hỏi §8; nếu thiếu cần đề xuất giờ bổ sung.
