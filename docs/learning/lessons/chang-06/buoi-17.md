---
buoi: 17
chang: 6
ten_chang: Run infra, Evaluation và BM25
tuan_goi_y: 6
thu_goi_y: Thứ 4 (lịch gợi ý, có thể đổi)
tasks: [T6.1]
moi_truong: [CPU, Đọc tài liệu, Local]
thoi_luong: 150 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 6 — Hạ tầng run, Evaluation và BM25
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §8.4 Cấu hình và §8.7 Artifact
  - docs/learning/00-project-analysis.md §6. Hợp đồng dữ liệu và giao thức cần giữ
trang_thai: chưa làm
---

# Buổi 17 — Run identity, fingerprint và manifest

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- giải thích vì sao mỗi run mới cần `run_id` và `run_uuid` riêng;
- tính và đối chiếu SHA-256 của gold, corpus, checkpoint bundle;
- đọc một entry `run_manifest.json` và kiểm tra các field bắt buộc;
- dự đoán vì sao bản sao gold sửa một byte phải bị từ chối.

## 2. Vị trí trong bức tranh lớn

T6.1 tạo định danh và bằng chứng nguồn trước khi Evaluation hoặc BM25 ghi kết quả. Nó nối base V1.1 với RQ1–RQ3 bằng cách bảo đảm mọi metric sau này truy ngược được tới đúng gold, corpus, config, seed và checkpoint. Nếu gate Chặng 5 chưa có `data/processed/v1.1/` và report theo file, buổi này chỉ làm fixture, chưa tạo run thật.

## 3. Tiền đề

- Buổi 16: namespace `data/processed/v1.1/`, report và điều kiện qua base gate.
- Biết byte file khác nội dung hiển thị; biết JSON canonical và SHA-256.
- Chưa cần biết GPU; checkpoint trong buổi này chỉ là fixture hoặc đường dẫn được fingerprint.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| `run_id`/`run_uuid` | `run_id` ghép config, dataset/split, bundle hash, seed và UUID4; mỗi lần chạy mới có UUID4 mới. | Dùng lại tên run cũ rồi ghi đè artifact. |
| File fingerprint | SHA-256 của đúng byte `gold_path` hoặc `corpus_path`. | Hash tên file hoặc chỉ hash tier `base`. |
| Checkpoint fingerprint | Hash từng file trong checkpoint; bundle hash là JSON canonical ánh xạ thành phần → fingerprint. | Hash thư mục theo thứ tự liệt kê ngẫu nhiên. |
| Manifest | Entry được ghi trước query đầu tiên, kể cả predictions rỗng; chứa mode, dataset/split, k, path và hash. | Chỉ tạo manifest sau khi chạy xong. |
| Resume | Run bị ngắt giữ nguyên `run_id`, kiểm tra fingerprint rồi chỉ ghi query còn thiếu. | Sinh UUID mới khi tiếp tục và tạo query trùng. |

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–25 | Giải thích bằng ví dụ | Theo dấu một gold file từ path tới manifest. |
| 25–50 | Xem input | Đọc §8.4/§8.7 và artifact hiện có. |
| 50–95 | Fixture CPU | Tính hash, canonical bundle và kiểm tra sửa một byte. |
| 95–125 | Kiểm tra output | Lập bảng field bắt buộc, resume và file predictions rỗng. |
| 125–150 | Tự giải thích + nhật ký | Nêu điều kiện stop trước run thật. |

## 6. Input

- `docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md` §8.4/§8.7.
- `data/processed/corpus_manifest.json`, `data/processed/qa_germanquad_dev.jsonl`, `results/metrics_bm25_base_dev.json` để đọc hash và metadata hiện có.
- Fixture minh họa trong bài: gold bytes `b'{"q":1}\n'`, bản sao sửa thành `b'{"q":2}\n'`. Trong literal Python này, `\n` là một byte xuống dòng `0x0A`; không phải hai byte ký tự `\` và `n`. Mỗi payload dài 8 byte. Đây là fixture hash, không phải QARecord đủ schema.
- Bundle minh họa có bốn khóa `question_encoder`, `passage_encoder`, `reader`, `reranker`; run BM25 không dùng checkpoint ghi từng giá trị là `null`.
- Có thể triển khai và kiểm tra T6.1–T6.3 bằng fixture khi T5 chưa xong; run BM25 thật T6.4 phải chờ base V1.1 đã nghiệm thu. Đây là phạm vi làm fixture được tách ra trong học liệu; phụ thuộc T6.1 → T5.8 trong catalog vẫn áp dụng cho run thật. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án]

## 7. Thao tác từng bước

1. Đọc quy tắc run/config/manifest:

   ```powershell
   Select-String -LiteralPath 'docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md' -Pattern 'run_id|run_uuid|checkpoint_bundle_hash|ghi entry TRƯỚC|gold_sha256|corpus_sha256'
   ```

   Kỳ vọng: thấy UUID4 mới cho run mới, manifest ghi trước query đầu và hash đầy đủ của file.
2. Kiểm tra artifact hiện có mà không chạy pipeline:

   ```powershell
   Get-FileHash -LiteralPath 'data/processed/corpus_manifest.json' -Algorithm SHA256
   Get-FileHash -LiteralPath 'data/processed/qa_germanquad_dev.jsonl' -Algorithm SHA256
   Test-Path -LiteralPath 'run_manifest.json'
   ```

   Kỳ vọng trên repo hiện tại: hai lệnh đầu in SHA-256; `run_manifest.json` ở root có thể là `False`. Đây là quan sát artifact cũ, chưa phải manifest V1.1.
3. Viết entry **đủ nhóm field ngay từ đầu**, đối chiếu từng dòng dưới với [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §8.7 Artifact] và [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §8.4 Cấu hình]:

   | Nhóm | Nội dung bắt buộc |
   |---|---|
   | Định danh/cấu hình | `run_id`, `run_uuid`, `config_id`, `mode`, `dataset`, `split`, `retriever`, `reranker_enabled`, `corpus_id`, `retrieval_k=20`, `reader_k=5`, `seed` |
   | File input | `gold_path`, `gold_sha256`, `corpus_path`, `corpus_sha256`, `corpus_manifest_path`, `corpus_manifest_sha256`; hash đầy đủ, đường dẫn tương đối resolve theo thư mục chứa manifest |
   | File output | `predictions_path` hoặc `retrieval_results_path`, tùy mode |
   | Checkpoint | `checkpoints` có bốn khóa `question_encoder`, `passage_encoder`, `reader`, `reranker`; mỗi thành phần dùng có path, model/tokenizer revision và fingerprint; thành phần không dùng ghi `null`, không bỏ khóa |
   | Bundle | `checkpoint_bundle_hash` đầy đủ, tính từ bản đồ fingerprint của bốn thành phần; bundle toàn null vẫn có hash của JSON canonical |
   | Điều kiện đo | Thiết bị/GPU/CPU, phiên bản thư viện, batch query=1, danh sách 10 warm-up query_id, fingerprint file dev dùng warm-up, số lượt đo chính thức/query=1; run `retrieval_only` không đo thì các field điều kiện đo/warm-up ghi `null` |

   Tên khóa chi tiết của nhóm điều kiện đo chưa được §8.7 quy định đầy đủ; thẻ T6.1 cần đặt tên nhất quán trước khi viết runner, không được bỏ thông tin của nhóm này. Trạng thái entry minh họa: **chưa làm**, chưa có run thật.
4. Dự đoán hash sẽ khác rồi chạy fixture sau, chỉ tính trong bộ nhớ. Khi copy từ Markdown nguồn, đưa hai dấu `@'` và `'@` về cột 0; copy bản render thì giữ nguyên khối lệnh.

   ```powershell
   @'
   import hashlib
   for payload in (b'{"q":1}\n', b'{"q":2}\n'):
       print(payload.hex(), hashlib.sha256(payload).hexdigest())
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng (hex kết thúc bằng `0a`, không phải `5c6e`):

   ```text
   7b2271223a317d0a f1232c45800aae8f8ab483528aa58e69b0a2428349e68e93fcf22721904b3000
   7b2271223a327d0a 4dc331dd304154a350c2b044590baa17aa1cdbef9909565b206da611a9743547
   ```

   Chỉ byte `1` → `2` đổi; hash khác nghĩa là không thể reuse fingerprint gold cũ. Đoạn này chứng minh hash đổi; việc validator thật từ chối mismatch phải có test trong nhánh thực thi T6.1.
5. Tạo JSON canonical cho bundle theo thứ tự khóa sắp xếp, UTF-8, `ensure_ascii=False`, separators `(',', ':')`; hash chuỗi canonical, không hash biểu diễn tùy ý của Python.
6. Lập bảng resume: fingerprint không đổi → giữ `run_id` và chỉ ghi query thiếu; fingerprint đổi → từ chối trước inference; query đã có → không ghi bản thứ hai.
7. Đối chiếu trường hợp predictions rỗng: manifest vẫn tồn tại và giữ cấu hình; Evaluation sau đó coi các query đủ gold là `execution_failure`, không coi file rỗng là thiếu giao thức.
8. **Nhánh thực thi sau gate (không tính vào 150 phút):** viết module run infrastructure và test T6.1 trong `src/`/`tests/` sau khi chốt signature; tạo manifest entry trước query đầu, chạy fixture sửa một byte và ghi log. Nếu base V1.1 chưa qua gate, chỉ chạy test fixture, không tạo run thật.

## 8. Dự đoán trước, kiểm tra sau

1. Hai run mới cùng config/checkpoint/seed có cùng `run_id` không? **Không; UUID4 mới làm định danh khác.**
2. Sửa một byte gold nhưng giữ tên file có vượt kiểm tra fingerprint không? **Không; SHA-256 đổi nên manifest không khớp.**
3. File predictions rỗng có cần manifest không? **Có; manifest phải ghi trước query đầu, kể cả không có prediction.**

## 9. Output cần có

- Bảng đầy đủ sáu nhóm field của `run_manifest.json` ở bước 3, gồm corpus manifest, checkpoints, bundle và điều kiện đo; nguồn §8.7.
- Kết quả hai hash fixture (hex có byte cuối `0a`) và quy tắc resume.
- Sau nhánh thực thi: module run identity/manifest, test từ chối bản sao gold sửa một byte và log kết quả; chưa có thì ghi chưa làm.

## 10. Kiểm tra hiểu bài

1. `run_uuid` khác `config_id` ở điểm nào? **Config mô tả thiết lập; UUID phân biệt từng lần chạy mới.** Chấm: nêu được cả hai vai trò.
2. Vì sao hash file phải là hash byte? **Để phát hiện nội dung thực tế đổi dù tên/path giữ nguyên.** Chấm: không nói chỉ hash tên.
3. Bundle hash dùng input nào? **JSON canonical của bản đồ thành phần tới fingerprint checkpoint.** Chấm: nêu canonical hóa.
4. Resume có sinh run mới không? **Không nếu fingerprint hợp lệ; giữ run cũ.** Chấm: không tạo query trùng.
5. Manifest ghi khi nào? **Trước query đầu tiên, kể cả predictions cuối cùng rỗng.** Chấm: đúng thời điểm.

## 11. Bài tự giải thích

Vẽ đường đi `gold/corpus/checkpoint → fingerprint → manifest → run_id → artifact` và giải thích điểm dừng khi hash không khớp.

Rubric 4 điểm: đủ năm nút; phân biệt file hash/bundle hash; nêu UUID4; nêu stop khi mismatch.

## 12. Gate của buổi

Qua giờ hướng dẫn khi sinh viên viết đủ sáu nhóm field ở bước 3, đối chiếu đúng hai hash và giải thích resume. Chỉ sau khi nhánh thực thi T6.1 có test pass mới chuyển T6.2 trên fixture; T6.4 chạy thật vẫn chờ base V1.1 và T6.1–T6.3 pass. Không yêu cầu score metric.

## 13. Lỗi thường gặp & cách gỡ

- **Artifact:** hash tên file → dùng SHA-256 byte.
- **Config:** suy mode/dataset từ dòng prediction đầu → đọc manifest.
- **Resume:** sinh UUID mới khi tiếp tục → giữ run cũ nếu fingerprint không đổi.
- **Scope:** chạy pipeline khi base chưa đạt → dừng ở fixture và ghi gate.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T6.1
Lệnh chỉ-đọc đã chạy:
Output path:
Fixture hash/bundle:
Manifest fields:
Corpus manifest path/hash:
Checkpoints/bundle hash:
Điều kiện đo/warm-up (hoặc null có lý do):
Resume decision:
Nhánh thực thi đã chạy/chưa chạy:
Hiểu/chưa hiểu:
Lỗi:
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Ôn N_eval, n_latency, Recall@k và GoldCoverage@k; giữ base gate và manifest fixture làm đầu vào.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: module/signature run infrastructure và vị trí test mới trong repo.
- CẦN XÁC MINH: revision cụ thể của base V1.1 trước run đầu tiên.
- CẦN XÁC MINH: checkpoint bundle thực tế; chưa suy từ tên thư mục.
