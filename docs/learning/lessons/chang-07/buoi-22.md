---
buoi: 22
chang: 7
ten_chang: Dense Retriever
tuan_goi_y: 8
thu_goi_y: Thứ 2 (lịch gợi ý, có thể đổi)
tasks: [T7.2, T7.3]
moi_truong: [Đọc tài liệu, Local, CPU, Colab T4]
thoi_luong: 150 phút hướng dẫn và fixture; train/encode/evaluation thật là nhánh riêng
chay_may_rieng: có; train Dense, encode/search base cho checkpoint selection và đóng gói FAISS, chưa đo thời gian
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 7 — Dense Retriever
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.2 Input/Output
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.4 Cấu hình
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.7 Artifact
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.8 Tài nguyên
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §8.7 Artifact
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §RQ1 — Ảnh hưởng của quy mô corpus
  - src/evaluation.py §recall_at_k
trang_thai: chưa làm
---

# Buổi 22 — Chọn checkpoint đúng P2 và nối FAISS với passage_id

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- chọn checkpoint bằng Recall@20 trước selection trên GermanQuAD dev/base, xử lý tie và gold rỗng;
- tính tay truy hồi dot product, chuyển index số sang passage_id đúng ID map và phát hiện map bị sort sai;
- lập chuỗi train → checkpoint dev runs → lựa chọn P2 → index cuối, với fingerprints của đúng checkpoint;
- phân biệt phép tính fixture với train/FAISS/run/artifact thật còn chờ nghiệm thu.

## 2. Vị trí trong bức tranh lớn

Dense và BM25 chỉ so sánh có ý nghĩa khi cùng corpus/gold và giao thức. T7.2 cần encode/search base của từng checkpoint để đo P2 trước khi chọn; T7.3 đóng gói index/ID map và retrieval run của checkpoint được chọn. Đây là phân ranh thao tác trong bài, **không đổi phụ thuộc catalog** T7.2 → T7.3. Chưa có kết quả để kết luận Dense thắng BM25. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.4 Cấu hình] [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §RQ1 — Ảnh hưởng của quy mô corpus]

## 3. Tiền đề

- Buổi 17–18: run_id, manifest, hash byte, N_eval, Recall=Hit; buổi 20–21: sampler/mask, hai tower, seed, smoke và resume.
- **Thực thi thật** cần base T5.8, T6.1–T6.4 và T7.1/smoke T7.2 đạt gate; runtime/checkpoint path và revision được chốt. Fixture dưới không mở các gate này.
- Tại lúc soạn chưa có `data/processed/v1.1/`, module Dense, run infrastructure mới hoặc package faiss-cpu/torch/transformers trong .venv. Ghi thiếu bằng chứng, không dùng artifacts cũ thay input V1.1. [kiem-chung-chang-07.md §1. Nguồn và hiện trạng]

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ minh họa | Sai lầm hay gặp |
|---|---|---|
| P2 / checkpoint selection | Chọn weights ở epoch có Recall@20 dev/base cao nhất; bằng nhau lấy epoch sớm hơn. | Chọn theo DPR dev loss hoặc GermanQuAD test. |
| N_eval / Recall@20 | 4 query, 1 gold rỗng → N_eval=3; gold đầu hạng 21 cho Hit@20=0. | Chia cho 4 hoặc tính số gold thu được thay Hit. |
| Encode / IndexFlatIP | Encode biến passage thành vector; IndexFlatIP tìm inner product chính xác trên các vector đã thêm. | Normalize vector để biến phép đo thành cosine. |
| FAISS ID map | Index số 0 ánh xạ ID đúng vị trí thêm, ví dụ pZ; map phải lưu kèm index. | Sort ID map riêng rồi cho rằng index vẫn trỏ đúng passage. |
| Fingerprint / bundle | Hash nhận diện byte checkpoint; bundle nối fingerprints các thành phần với run. | Đổi weights nhưng dùng lại index/run manifest của checkpoint cũ. |
| Reuse embeddings | Mở tier cùng checkpoint thì encode passage mới; giữ vector passage đã có. | Reuse vector base của checkpoint khác trong lúc chọn checkpoint. |

P2 và IndexFlatIP: [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.4 Cấu hình]. ID map bắt buộc: [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.2 Input/Output]. Mở tier: [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.8 Tài nguyên].

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–25 | Giải thích bằng ví dụ | P2, encode/search, ID map. |
| 25–50 | Xem input | Phụ thuộc và artifact dự kiến, checkpoint table. |
| 50–100 | Thao tác từng lệnh | Fixture selection và dot/ID map. |
| 100–125 | Kiểm tra output | Kiểm fingerprint/input/run; lịch thực thi riêng. |
| 125–150 | Tự giải thích + nhật ký | Chọn checkpoint, giới hạn kết luận/gate. |

Gap ước lượng 150–210 phút khi tính tải đầy đủ; bài giữ 150 phút hướng dẫn/fixture. Train/encode/chấm checkpoint và xử lý lỗi có phiên riêng, không ấn định thời gian máy hoặc coi phần đó đã xong. [docs/learning/02-roadmap-gap-report.md §3. Đối chiếu từng buổi 1–32]

## 6. Input

- Spec §3.2/3.3, §5 và §8.4/8.7; `src/evaluation.py` hiện có hàm lẻ `recall_at_k`. Chỉ dùng hàm lẻ đã đối chiếu cho fixture, không suy toàn bộ Evaluation cũ đã khớp V1.1.
- Fixture selection §7: 4 query tự tạo và 3 checkpoint giả lập; không phải run/model/score thật.
- Fixture vector: q=`[1,0]`; passage vectors `[[2,2],[1,0],[-1,2]]`; ID map theo thứ tự thêm `['pZ','pA','pM']`.
- Input **dự kiến, chưa có**: `data/processed/v1.1/corpus_base.jsonl`, `qa_germanquad_dev.jsonl`, `retrieval_germandpr_train.jsonl`, corpus manifest và reports trong cùng namespace. GermanDPR dev theo dõi loss; GermanQuAD dev chọn checkpoint. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T5.8 — Phạm vi báo cáo và nghiệm thu] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.4 Cấu hình]

## 7. Thao tác từng bước

Chạy PowerShell tại repo root. Here-string mở/đóng phải ở cột 0 khi copy file nguồn. Fixture chỉ trong RAM, không ghi metrics/checkpoint/index giả vào results.

1. **Kiểm gate và nguồn chuẩn**:

   ```powershell
   Test-Path 'data/processed/v1.1/corpus_base.jsonl'
   Test-Path 'src/run_manifest.py'
   Select-String -LiteralPath 'docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md' -Pattern 'Chọn checkpoint retriever|retrieval_germandpr_dev|IndexFlatIP' | ForEach-Object { $_.Line }
   ```

   Kỳ vọng tại lúc soạn: False/False; source ghi P2 Recall@20 trước selection, dev/base, tie sớm hơn; DPR dev chỉ theo dõi loss. Nếu file đã xuất hiện sau này, cần log nghiệm thu/fingerprint, không mở gate chỉ bằng Test-Path=True.

2. **Dự đoán checkpoint trước khi chạy.** Gold q1=`{a,b}`, q2=`{c}`, q3=`{d}`, q-empty=`{}`. Cả 3 checkpoint trả a cho q1 và c hạng 20 cho q2; epoch 1 trả d hạng 21, epoch 2/3 trả d hạng 1. DPR dev loss minh họa lần lượt `0.8,0.9,0.2`.

   ```powershell
   @'
   from src.evaluation import recall_at_k
   gold = {'q1':{'a','b'}, 'q2':{'c'}, 'q3':{'d'}, 'q-empty':set()}
   prefix = [f'x{i:02d}' for i in range(1,21)]
   rankings = {
       1:{'q1':['a'], 'q2':prefix[:19]+['c'], 'q3':prefix+['d']},
       2:{'q1':['a'], 'q2':prefix[:19]+['c'], 'q3':['d']},
       3:{'q1':['a'], 'q2':prefix[:19]+['c'], 'q3':['d']},
   }
   valid = {q:g for q,g in gold.items() if g}
   dev_loss = {1:0.8, 2:0.9, 3:0.2}
   scores = {}
   print('total',len(gold),'N_eval',len(valid),'excluded_empty',len(gold)-len(valid))
   for epoch, rows in rankings.items():
       hits = sum(recall_at_k(rows.get(q,[]),list(g),20) for q,g in valid.items())
       scores[epoch] = hits/len(valid)
       coverage = sum(len(set(rows.get(q,[])[:20]) & g)/len(g) for q,g in valid.items())/len(valid)
       print(epoch,'Recall@20',f'{scores[epoch]:.6f}','Coverage@20',f'{coverage:.6f}','DPR_dev_loss',dev_loss[epoch])
   chosen = min(scores,key=lambda epoch:(-scores[epoch],epoch))
   print('chosen_epoch',chosen)
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: total 4/N_eval 3/excluded_empty 1; Recall `0.666667/1.000000/1.000000`; Coverage `0.500000/0.833333/0.833333`; chọn epoch **2**, dù epoch 3 có DPR loss thấp nhất. Đây là danh sách retrieval minh họa trước selection, không score benchmark. Nếu khác, kiểm cutoff 20, mẫu số và tie; không dùng Coverage/loss thay Recall. Trên producer QA thật trước đây, dev có 1.047 dòng và 16 gold rỗng, nên mốc dự kiến N_eval là **1.031**; phải lấy lại từ artifact/report T5.8 khi có, không đồng nhất với 1.031 dòng của dev cũ. [../chang-05/kiem-chung-T5.3-T5.4.md §3. Cache thật và hiệu chỉnh mốc D1]

3. **Tính tay truy hồi và ID map**, dự đoán top-2 và hậu quả sort map riêng:

   ```powershell
   @'
   import math
   q, vectors = [1,0], [[2,2],[1,0],[-1,2]]
   id_map = ['pZ','pA','pM']
   scores = [sum(a*b for a,b in zip(q,p)) for p in vectors]
   order = sorted(range(len(scores)),key=lambda i:-scores[i])[:2]
   print('dot',scores,'indices',order,'ids',[id_map[i] for i in order])
   wrong_map = sorted(id_map)
   print('wrong_sorted_map_ids',[wrong_map[i] for i in order])
   cosine = [score/math.sqrt(sum(v*v for v in p)) for score,p in zip(scores,vectors)]
   cosine_order = sorted(range(len(cosine)),key=lambda i:-cosine[i])[:2]
   print('cosine_ids',[id_map[i] for i in cosine_order])
   old_ids, larger_ids = set(id_map), ['pZ','pA','pM','pN']
   print('new_ids_to_encode',[pid for pid in larger_ids if pid not in old_ids])
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: dot `[2,1,-1]`, indices `[0,1]`, IDs `['pZ','pA']`; map sort sai cho `['pA','pM']`; cosine cho `['pA','pZ']`; chỉ `['pN']` cần encode thêm nếu cùng checkpoint khi mở tier. q dài 1 trong fixture nên công thức cosine ở đây chỉ cần chia norm passage. Đây là phép tính Python minh họa, **chưa chạy FAISS**, không có trường hợp tie score để tự đặt tie-break production. Nếu khác, kiểm thứ tự thêm vectors/map và có vô tình normalize hay không.

   **LỆNH MẪU – chưa xác minh trên runtime thật:** cell Python sau khi môi trường có FAISS được duyệt; không tải model và không ghi index:

   ```python
   import numpy as np
   import faiss
   index = faiss.IndexFlatIP(2)
   index.add(np.array([[2,2],[1,0],[-1,2]], dtype='float32'))
   scores, indices = index.search(np.array([[1,0]], dtype='float32'), 2)
   id_map = ['pZ','pA','pM']
   print(index.ntotal, scores.tolist(), indices.tolist())
   print([id_map[i] for i in indices[0]])
   ```

   Expected toán học là `3 [[2.0,1.0]] [[0,1]]` và `['pZ','pA']`; chưa phải output đo trên FAISS tại phiên này. Nếu khác, kiểm dtype/shape/index type/map; không sửa map để ép top-k hợp ý.

4. **Lập bảng traceability cho một checkpoint dev run**: dataset/split GermanQuAD dev; corpus_id base; gold/corpus fingerprint; fingerprint cả hai tower và bundle hash; seed/revisions/config; retrieval_k=20/reader_k=5; run_id/UUID; mode retrieval_only. Entry và fingerprints phải có **trước run**; corpus_manifest_path/sha256, checkpoints và điều kiện đo theo đủ §8.7. Nếu retrieval-only không đo latency, ghi null cho metadata đo/warm-up theo giao thức, không bịa thời gian từ đồng hồ encode. Với ba checkpoint cần ba nhận diện run riêng. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §8.4 Cấu hình] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §8.7 Artifact]

5. **Ghi lịch nhánh thực thi cụ thể**, chỉ mở khi đủ gate/thẻ được duyệt:

   | Thứ tự | Task, thời điểm | Việc và bằng chứng cần có |
   |---|---|---|
   | 1 | T7.2 sau smoke buổi 21 | Chốt config/revisions; train GermanDPR train sau D1; checkpoint mỗi epoch cùng resume state. Spec dự kiến 3 epoch, không cam kết thời gian GPU. |
   | 2 | T7.2 cho mỗi checkpoint ứng viên | Dùng đúng hai tower, encode/search base để lấy retrieval cho GermanQuAD dev; manifest trước run; Evaluation Recall@20 trước selection. |
   | 3 | T7.2 kết thúc selection | Chọn Recall cao nhất, tie sớm nhất; ghi lý do và fingerprints, khóa checkpoint. Không chọn bằng DPR loss/test. |
   | 4 | T7.3 sau lựa chọn | Đóng gói/kiểm index base và ID map của đúng checkpoint; có thể giữ index candidate đã chứng minh cùng checkpoint/corpus, không ghép với vector từ checkpoint khác. |
   | 5 | T7.3 nghiệm thu | Đọc lại index/map, truy hồi fixture/known vectors, kiểm membership/counts; run retrieval-only dev có nguồn; lưu training_config và các artifact §5.7. |

   Phần triển khai phải chốt tên file/API/test/CLI trong thẻ T7.2/T7.3; chưa có trainer nên không cung cấp lệnh `python -m ...` giả. Phụ thuộc encode/search tạm ở bước 2 phải được ghi ngay trong thẻ, tránh vòng lặp “chọn checkpoint trước khi có metric”. Test/index save-load chạy trên implementation thật trong nhánh này, không thay bằng bước 3 tính tay.

6. **Khoanh giới hạn**: chưa mở tier hoặc tune theo test tại buổi 22. Khóa checkpoint để dùng lại các tier; encode passage mới ở T9.3 theo spec. Dense selection trước rerank dùng V1.1; metadata production post-rerank của Chặng 8 vẫn **chờ chốt A3**, không để việc duyệt bài Dense tự phê chuẩn draft. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.4 Cấu hình] [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 8 — Reader và Reranker]

## 8. Dự đoán trước, kiểm tra sau

1. Epoch 3 loss thấp hơn epoch 2, Recall bằng nhau: chọn gì? **Epoch 2, đúng tie sớm hơn; DPR loss không phải metric selection.** Chấm: nêu cả metric và tie.
2. Gold q3 hạng 21 của epoch 1 có Hit@20 không? **Không; góp zero, vẫn nằm trong N_eval.** Chấm: không bỏ query hoặc tăng cutoff.
3. Sort riêng ID map có đổi score FAISS không? **Không, nhưng gán score/index sang passage sai.** Chấm: dự đoán sai thành pA/pM thay pZ/pA.

## 9. Output cần có

Người học ghi **đường dẫn đề xuất, chưa tạo** `docs/learning/lessons/chang-07/thuc-hanh/buoi-22-nhat-ky.md`: bảng checkpoint/Recall/Coverage/loss và lựa chọn; dot/map/cosine fixture; bảng traceability run; lịch thực thi T7.2/T7.3 với gate còn thiếu.

Sau nhánh thực thi riêng mới có artifact thật: `question_encoder/`, `passage_encoder/`, `training_config.json`, `faiss_index_base.bin`, `faiss_id_map_base.json`, retrieval logic name `retrieval_dense_base_dev.jsonl` và metrics/manifest theo run. **Tên artifact theo spec; đường dẫn thư mục checkpoint/run cụ thể còn chốt trong thẻ**, không ghi đè run cũ. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.2 Input/Output] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.7 Artifact] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §3.2 Input/Output]

## 10. Kiểm tra hiểu bài

1. Dataset/split/tier nào chọn Dense checkpoint? **GermanQuAD dev/base.** Chấm: đủ ba thành phần và trước selection.
2. Vì sao Coverage epoch 2 bằng 5/6 nhưng Recall=1? **Cả ba query hit; q1 mới lấy 1/2 gold.** Chấm: tính đúng hai công thức, không gọi Coverage là Recall.
3. Encoder weights đổi có reuse embeddings checkpoint cũ để chấm checkpoint mới được không? **Không; vector không còn từ đúng encoder/cấu hình đang đánh giá.** Chấm: nối fingerprint với index/corpus.
4. Ba hàng vectors có ba ID là đủ nghiệm thu map chưa? **Chưa: còn thứ tự mapping, uniqueness/membership và đúng checkpoint; cùng độ dài vẫn có thể sort sai.** Chấm: dùng fixture pZ/pA/pM.
5. Score test thuận lợi hơn có được chọn lại epoch không? **Không; config/checkpoint khóa trước test.** Chấm: phân biệt sửa lỗi có log với tune theo test.
6. Chưa có base artifact thì số 7.520 trong audit dùng làm gì? **Mốc đối chiếu, không thay corpus/manifest/gate thật.** Chấm: yêu cầu hash byte và nghiệm thu artifact.

## 11. Bài tự giải thích

Vẽ đường từ 3 checkpoint ứng viên tới checkpoint được chọn và index cuối. Chỉ nơi encode/search xảy ra trước selection, nơi manifest được ghi, và cách một FAISS index số trở lại passage_id đúng.

Rubric 4 điểm: đúng P2/tie/mẫu số; phá được vòng phụ thuộc selection/index; truy vết checkpoint/corpus/run; tách fixture/dev selection/test chính thức. Không dùng điểm cao làm tiêu chí thay giao thức.

## 12. Gate của buổi

Gate hiểu bài: output §9 đủ, chọn epoch 2 đúng lý do, chỉ ra ID map sai và tiền đề chưa đạt. Gate **Chặng 7 thực thi**: mọi kiểm thử §5.6 có bằng chứng, train/selection P2 trên dev/base, checkpoint/config ghi đủ, FAISS+map và retrieval run đúng nguồn. Không đánh dấu đạt khi chỉ có bảng trên giấy hoặc test module lẻ. Chỉ sang thực thi Reader khi T7.3 đạt theo catalog; việc soạn học liệu tiếp theo vẫn cần lượt duyệt riêng.

## 13. Lỗi thường gặp & cách gỡ

- **Spec:** chọn theo DPR dev loss/Recall sau rerank → trở lại P2 trước selection trên GQ dev/base.
- **Data:** gold rỗng hoặc query thiếu prediction bị bỏ sai mẫu số → dùng quy tắc Evaluation và reports, không tự sửa dữ liệu để chấm tiếp.
- **Code:** vector và map lệch thứ tự → kiểm fixture index số, uniqueness/membership và save/load.
- **Environment:** thiếu FAISS/T4 → giữ cell mẫu chưa chạy; dùng phép tính CPU, chưa công bố index thật.
- **Provenance:** dùng embeddings/checkpoint khác hoặc ghi đè run → đối chiếu fingerprint, run_id và namespace trước đánh giá.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Tasks: T7.2, T7.3
Hiểu/chưa hiểu:
Lệnh CPU đã chạy / cell mẫu chưa chạy:
N_eval, n_excluded_empty_gold, Recall/Coverage và tie:
Checkpoint chọn và lý do (fixture hay run thật):
FAISS index số -> passage_id; checkpoint/corpus fingerprint:
Output path; run_id/manifest path (chưa có thì ghi chưa có):
Lỗi và cách gỡ:
Gate hiểu bài / gate thực thi Chặng 7: chưa làm
Bước tiếp theo:
```

## 15. Việc chuẩn bị cho buổi sau

Chờ duyệt Chặng 7 trước soạn Chặng 8. Khi học Reader, ôn gold evidence/char-offset và mang log T7.3; thiếu gate thì không chạy pipeline phụ thuộc.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: thẻ T7.2/T7.3 và file/API/test/CLI, đường dẫn run/index/checkpoint — chủ project duyệt trước nhánh triển khai; có bước encode/search candidate trước P2 như §7.
- CẦN XÁC MINH: base reports/fingerprint, N_eval thật, model/tokenizer revision và checkpoint được chọn — người triển khai đo bằng artifact/run, không lấy số fixture.
- CẦN XÁC MINH: runtime FAISS/T4 và save/load index/model thật — kiểm trong nhánh được duyệt, chưa có bằng chứng từ cell mẫu.
- Chờ chốt A3 cho provenance post-rerank ở Chặng 8; Dense P2 hiện tại không tự phát hành V1.2.
