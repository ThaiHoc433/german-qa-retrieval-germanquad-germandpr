---
buoi: 1
chang: 1
ten_chang: Hiểu đề tài/project
tuan_goi_y: 1
thu_goi_y: Thứ 2 (lịch gợi ý, có thể đổi)
tasks: [T1.1]
moi_truong: [Đọc tài liệu, Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 1 — Hiểu đề tài và khảo sát project
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §Kiến trúc và ranh giới module
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §6.3 Quy tắc xử lý
  - docs/learning/00-project-analysis.md §3. Pipeline và ranh giới trách nhiệm
trang_thai: chưa làm
---

# Buổi 1 — Nhìn đúng bài toán và đọc cây project

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- giải thích sự khác nhau giữa closed-context, open-domain và extractive QA bằng một ví dụ;
- chỉ ra vai trò của corpus, passage, Retriever, Selection, Reader và Evaluation trên sơ đồ pipeline;
- dùng lệnh chỉ-đọc PowerShell đã kiểm chứng để tìm các thư mục `src/`, `tests/`, `notebooks/`, `data/`, `results/`, `docs/` và `website/`;
- phát hiện một file là artifact cũ hay nguồn chuẩn bằng cách đọc metadata, schema và đường dẫn, chưa gọi nó là nghiệm thu V1.1.

## 2. Vị trí trong bức tranh lớn

Buổi 1 mở Chặng 1 và phục vụ T1.1: khảo sát cây repo và tự vẽ pipeline. Nó chưa trả lời RQ1–RQ3; nhiệm vụ là hiểu vì sao câu hỏi phải đi qua nhiều khối trước khi có đáp án. Kết quả ghi chú hôm nay sẽ được dùng ở buổi 2 để nối RQ với bảng/hình, rồi làm tiền đề cho việc đọc record ở Chặng 3. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 1 — Hiểu đề tài và khảo sát project]

## 3. Tiền đề

- Biết Python cơ bản, thư mục, file và chuỗi ký tự.
- Chưa cần biết PyTorch, GPU, BM25 hay FAISS; các thuật ngữ này chỉ nhận biết ở mức vai trò.
- Nếu chưa rõ slice chuỗi, ôn `text[start:end]` bằng một ví dụ nhỏ trong Python; không cần chạy dataset.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| Closed-context QA | Hệ thống đã nhận một context và Reader tìm câu trả lời trong context đó. | Nghĩ rằng Reader tự đi tìm trong toàn bộ Wikipedia. |
| Open-domain QA | Câu hỏi đi vào corpus; Retriever tìm passage trước, rồi Reader mới đọc passage được chọn. | Gọi việc tìm đúng passage là đã trả lời đúng. |
| Extractive QA | Đáp án là một đoạn có thật trong `text`, kèm offset bắt đầu/kết thúc. | Tưởng hệ thống được tự viết câu trả lời mới như LLM. |
| Corpus và passage | Corpus là tập passage có thể tìm; mỗi passage là một đoạn của corpus. | Dùng một passage training hard negative như thể nó đương nhiên thuộc corpus đánh giá. |
| Pipeline | `Data → Retriever → Selection/Reranker → Reader → Evaluation`. | Cho Evaluation chạy model hoặc cho Reader tự retrieval. |
| Artifact và gate | Artifact là file kết quả có thể truy vết; gate là điều kiện cần kiểm chứng trước khi chuyển chặng. | Thấy file tồn tại hoặc test được collect rồi kết luận đã qua gate. |

Ví dụ minh họa về Reader: nếu passage có chuỗi `Albert Einstein`, Reader phải lưu text và vị trí của chuỗi đó; buổi này chưa yêu cầu tính offset trên dataset thật. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §0. Nguyên tắc chung & phân chia trách nhiệm module]

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | So sánh closed-context/open-domain và extractive QA bằng câu hỏi Einstein trong roadmap. |
| 20–40 | Xem input | Đọc design, spec và danh sách thư mục; đánh dấu đâu là nguồn chuẩn, đâu là code, đâu là artifact. |
| 40–75 | Thao tác từng lệnh | Chạy các lệnh chỉ-đọc ở mục 7, đọc một record GermanDPR và ghi output ngắn vào nhật ký. |
| 75–100 | Kiểm tra output | Vẽ pipeline sáu khối và gắn trách nhiệm “sở hữu/không được làm”. |
| 100–120 | Tự giải thích + nhật ký | Nói lại vì sao “tìm được passage” và “trả lời đúng” là hai việc khác nhau. |

## 6. Input

- `docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md` — mục Chặng 1.
- `docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md` — mục Kiến trúc và ranh giới module.
- `docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md` — mục 0 và 6.3.
- Cây repo hiện có: `src/`, `tests/`, `notebooks/`, `data/`, `results/`, `docs/`, `website/`.

## 7. Thao tác từng bước

1. Đọc phần Chặng 1 trong roadmap. Ghi ba từ khóa: `closed-context`, `open-domain`, `extractive QA`. Kỳ vọng: mỗi từ có một cách giải thích riêng; nếu chỉ ghi “hỏi đáp”, hãy quay lại mục 4.
2. Đọc phần kiến trúc trong design. Vẽ bằng giấy: `Question → Retriever → retrieval pool → Selection → Reader → answer span`; thêm `Evaluation` nhận artifact. Kỳ vọng: Reader không có mũi tên trực tiếp tới corpus.
3. Tại thư mục gốc repo, dùng PowerShell thuần để liệt kê file:

   ```powershell
   Get-ChildItem -LiteralPath src,tests,notebooks,data,results,docs,website -Recurse -File | Select-Object -ExpandProperty FullName
   ```

   Kỳ vọng: thấy các đường dẫn thuộc bảy thư mục trên. Nếu không thấy, kiểm tra thư mục hiện tại bằng `Get-Location`; không tự tạo thư mục. Ghi `website/` vào bảng phân loại là giao diện/tài liệu trình bày, không phải nguồn chuẩn kỹ thuật.
4. Đọc đầu file có trách nhiệm của một số module bằng lệnh chỉ-đọc:

   ```powershell
   Get-Content -LiteralPath 'src/corpus.py' -Encoding UTF8 -TotalCount 35
   Get-Content -LiteralPath 'src/bm25_retriever.py' -Encoding UTF8 -TotalCount 35
   Get-Content -LiteralPath 'src/evaluation.py' -Encoding UTF8 -TotalCount 45
   ```

   Kỳ vọng: nhìn thấy tên hàm/lớp và mô tả module; không kết luận code đã khớp V1.1 chỉ từ docstring.
5. Đọc một record processed mà không sửa file:

   ```powershell
   Get-Content -LiteralPath 'data/processed/qa_germanquad_dev.jsonl' -Encoding UTF8 -TotalCount 1
   Get-Content -LiteralPath 'data/processed/corpus_manifest.json' -Encoding UTF8
   ```

   Kỳ vọng: thấy artifact có dữ liệu; ghi chú rằng analysis đã xác định schema cũ khác QARecord V1.1. Nếu file không có, ghi “CẦN XÁC MINH” thay vì đoán đường dẫn mới.
6. Đọc thêm đúng một record GermanDPR từ artifact hiện có:

   ```powershell
   Get-Content -LiteralPath 'data/processed/retrieval_germandpr_dev.jsonl' -Encoding UTF8 -TotalCount 1
   ```

   Kỳ vọng: thấy `query_id`, `question`, `positive_passage_ids` và `hard_negative_passage_ids` của schema cũ. Ghi rõ đây là record processed để khảo sát; chưa gọi nó là `RetrievalTrainingRecord` V1.1 hoàn chỉnh.
7. Đối chiếu một test với nguồn chuẩn:

   ```powershell
   Get-ChildItem -LiteralPath tests -Recurse -File -Filter '*.py' | Select-String -Pattern 'test_old_round1_schema_is_rejected|test_near_verbatim_query_ranks_first'
   ```

   Kỳ vọng: nhận ra test là bằng chứng về hành vi được kiểm tra, không phải chứng nhận toàn project.
8. Lưu một sơ đồ ngắn và bảng “nguồn chuẩn / code / artifact / website / chưa biết” vào ghi chú buổi học. Không đánh dấu gate project.

## 8. Dự đoán trước, kiểm tra sau

Trả lời trước khi chạy lệnh:

1. `src/` có phải là toàn bộ project không? Vì sao?
2. Nếu corpus tìm đúng passage nhưng Reader chọn sai span, RQ1 nên quy lỗi ngay cho Retriever không?
3. `corpus_manifest.json` có đủ để kết luận corpus hiện tại là base V1.1 không?

Sau khi xem output, sửa câu trả lời bằng một đường dẫn nguồn cho mỗi kết luận. Gợi ý: `src/` chỉ là một phần cây repo; lỗi Reader không tự biến thành lỗi Retriever; manifest cũ phải đối chiếu D1/D6 và schema trước khi gọi là nghiệm thu.

## 9. Output cần có

- `docs/learning/lessons/chang-01/ghi-chu-buoi-01.md` — **chưa tạo sẵn**, sinh viên tự lưu ghi chú ở vị trí được giảng viên thống nhất.
- Một pipeline tự vẽ trong ghi chú, có mũi tên và trách nhiệm từng khối.
- Một bảng khảo sát repo ghi đường dẫn đã đọc, loại thông tin và điều chưa biết.

Tên file ghi chú là đề xuất học tập; nếu giảng viên chưa chốt nơi lưu, ghi “CẦN XÁC MINH” thay vì tự đặt artifact project.

## 10. Kiểm tra hiểu bài

1. Closed-context khác open-domain ở bước nào? **Đáp án gợi ý:** open-domain có bước tìm passage trong corpus trước Reader; closed-context nhận context sẵn. **Chấm:** nêu đúng đầu vào và vị trí Retriever.
2. Extractive QA lưu điều gì ngoài answer text? **Đáp án gợi ý:** passage và offset trong text. **Chấm:** không gọi đó là câu sinh tự do.
3. Vì sao Evaluation không tự chạy model? **Đáp án gợi ý:** Evaluation chỉ tính metric từ artifact đã lưu, để có thể tính lại và truy vết. **Chấm:** nêu được ranh giới module.
4. Một file `results/metrics_bm25_base_dev.json` chứng minh điều gì? **Đáp án gợi ý:** có một artifact metrics cũ; cần đọc run/config/corpus và tình trạng schema trước khi dùng. **Chấm:** phân biệt “có file” với “đạt gate”.
5. Nếu `Get-ChildItem` không thấy `tests/`, kiểm tra gì trước? **Đáp án gợi ý:** thư mục hiện tại và đường dẫn repo; không sửa code hay tự tạo thư mục. **Chấm:** chọn kiểm tra môi trường/chỉ-đọc.

## 11. Bài tự giải thích

Vẽ pipeline trên một trang và nói trong 2 phút: câu hỏi đi qua khối nào, mỗi khối sở hữu gì, và vì sao `Reader` không được tự truy vấn corpus.

Rubric 4 điểm:

- 1 điểm: đúng thứ tự Retriever → Selection → Reader;
- 1 điểm: phân biệt corpus/passage và answer span;
- 1 điểm: nói đúng Evaluation nhận artifact;
- 1 điểm: nêu một giới hạn hoặc điều chưa xác minh, có nguồn hoặc nhãn “CẦN XÁC MINH”.

## 12. Gate của buổi

Qua buổi học khi sinh viên tự giải thích được pipeline, chỉ đúng các thư mục nguồn và không nhầm file hiện hữu với nghiệm thu V1.1. Đây là gate học tập của buổi, chưa phải gate nghiệm thu code. Gate Chặng 1 yêu cầu biết nguồn chuẩn và chưa nhầm website/spec/code/metrics; task liên quan là T1.1. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 1 — Hiểu đề tài và khảo sát project]

## 13. Lỗi thường gặp & cách gỡ

- **Environment:** chạy lệnh ngoài root → dùng `Get-Location`, rồi quay lại root; không đổi đường dẫn theo phỏng đoán.
- **Data:** thấy số dòng trong artifact cũ → ghi “đã quan sát”, đối chiếu spec; không gọi đó là số V1.1.
- **Spec:** dùng design để ghi đè module spec → V1.1 là chuẩn kỹ thuật khi có mâu thuẫn.
- **Concept:** gọi Retriever là Reader → quay lại sơ đồ và viết một câu trách nhiệm cho mỗi khối.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T1.1
Hiểu được:
Chưa hiểu:
Lệnh chỉ-đọc đã chạy:
Output path / ghi chú:
Lỗi gặp (code / data / spec / environment):
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc phần RQ1–RQ3 trong design; mang theo pipeline đã vẽ và bảng thư mục. Không cần đọc chi tiết GPU hay chạy dataset.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: nơi lưu ghi chú cá nhân của sinh viên nếu giảng viên muốn thu artifact.
- CẦN XÁC MINH: ngày bắt đầu và ngày thực của Thứ 2; lịch trong file chỉ có 3 buổi/tuần, không chỉ định thứ.
- CẦN XÁC MINH: đường dẫn repo nếu học trên máy khác; các lệnh trên đã đối chiếu tại root hiện tại.
