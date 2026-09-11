# German Open-Domain Extractive QA — Module Implementation Specs

- **Phiên bản:** V1.1 — frozen. **Ngày đóng băng: 2026-09-11.** Gồm D1–D6 và P1–P4 đã duyệt. V1.0 = bản 2026-09-10. Mọi thay đổi sau ngày đóng băng phải thành phiên bản mới (V1.2+) có nhật ký quyết định riêng, không sửa trực tiếp vào V1.1.
- **Phụ thuộc:** `2026-09-10-german-qa-retrieval-design.md` (thiết kế tổng thể, core contribution, V1/V2). Khi 2 file mâu thuẫn về chi tiết kỹ thuật, **file này là chuẩn**.
- **Mục đích file này:** spec triển khai chi tiết từng module, đủ cụ thể để code không cần quay lại hỏi "quy ước là gì".
- **Thứ tự viết/triển khai:** schema chung → data/splits/corpus → evaluation → BM25 → dense → reader → reranker → integration.

## Nhật ký quyết định V1.1

Bằng chứng từ kiểm chứng code/data Milestone B (2026-09-10) chỉ dùng để chỉnh spec, không phải để chuyển sang triển khai.

| # | Mâu thuẫn ở V1.0 | Quyết định V1.1 | Mục sửa |
|---|---|---|---|
| D1 | 2.3.e đưa 1.024 dòng GermanQuAD **train** (câu hỏi trùng GermanDPR test) vào `qa_germanquad_test`, trái với 3.3 "tập chính thức = GermanQuAD test"; `query_id` mã hoá split đã gán nên không phân biệt được | Test của mỗi dataset = đúng test gốc của dataset đó. Trùng test chéo dataset chỉ dẫn tới **loại khỏi train/dev**, không chuyển sang test của dataset kia. `query_id` mã hoá split **gốc** | 1.2, 1.3, 2.3.e, 2.5–2.7, 9 |
| D2 | 3.3 ghi "phủ đúng 100% N=2.204" nhưng 1.2/2.3.c loại câu gold rỗng khỏi eval | Mọi metric **chất lượng** (Recall@k, MRR, GoldCoverage, EM/F1, error decomposition) dùng chung mẫu số `N_eval` = số QARecord có `gold_evidences` khác rỗng; latency dùng `n_latency` = số record hợp lệ để đo (`status="ok"`, thuộc `N_eval`); câu gold rỗng báo cáo riêng | 3.3, 3.5, 3.7 |
| D3 | Tắt reranker thì bucket `reranking_failure` không có nghĩa rõ | Đổi thành `selection_failure`: gold có trong top-`retrieval_k` của retriever nhưng không nằm trong `reader_k` passage được chọn — do reranker loại HOẶC do cắt thẳng top-`reader_k`. Chênh lệch `selection_failure` giữa bật/tắt reranker chỉ phản ánh khả năng **giữ gold** ở khâu chọn, không phải toàn bộ lợi ích reranker (vẫn phải so EM/F1 và latency) | 1.5, 3.3, 3.6, 7.3, 8.3, 9 |
| D4 | 6.5 trả `predicted_answer_text=""`, schema 1.5 trả `None` | `None` cho cả 5 field đáp án; `""` là dữ liệu sai, validator từ chối | 1.5, 6.5 |
| D5 | Warm-up và "sample 200 query" nằm trong config Evaluation, nhưng Evaluation không bỏ "N query đầu" được mà không phụ thuộc thứ tự file | Integration chịu trách nhiệm đo (thiết bị, batch=1, mỗi query đo 1 lần); Evaluation chỉ tóm tắt. Warm-up là 1 **lượt chạy thử bị bỏ kết quả**, không phải loại vĩnh viễn các query đó: query dùng để warm-up nếu thuộc tập đang chạy vẫn được chạy lại và đo 1 lượt chính thức | 3.3, 3.4, 8.3, 8.4, 8.6, 8.7 |
| D6 | 86% `corpus_base` V1.0 (46.551/54.071) là hard negative GermanDPR được tuyển theo từng query train/dev | Corpus đánh giá xây theo danh sách **nguồn được phép**: base = mọi chunk của context GermanQuAD + positive context GermanDPR; mid/large = base + một chuỗi distractor Wikipedia lấy mẫu cố định. Hard negative GermanDPR chỉ dùng cho training. Tầng: base = số thật sau build (dự kiến ~7,5K), mid = tổng 50.000, large = tổng 150.000 | 0, 1.1, 1.3, 2.3.a, 2.3.d, 2.3.f, 2.3.g, 2.4, 2.6, 2.7, 9 |

Bổ sung đã duyệt cùng V1.1:
- **P1** — `PredictionRecord` và `run_manifest.json` ghi `retrieval_k`, `reader_k`; ở end-to-end bắt buộc `1 ≤ reader_k ≤ retrieval_k` và nhất quán trong 1 run; manifest giữ cấu hình kể cả khi file predictions rỗng. Oracle là ngoại lệ (1.5, 8.3).
- **P2** — Chọn checkpoint trên `qa_germanquad_dev`, tier base, bằng metric cụ thể cho từng thành phần (5.4, 6.4, 7.4); checkpoint đóng băng khi đánh giá các tier còn lại; metric bằng nhau → chọn checkpoint sớm hơn. Không dùng Hit@`reader_k` để chọn chính `reader_k` (tăng số passage giữ lại trên cùng danh sách xếp hạng không bao giờ làm Hit giảm) — V1 cố định `reader_k`=5.
- **P3** — Sanity-check GermanDPR test chỉ ở `retrieval_only`, gold = `positive_passage_ids`; mọi positive ID phải tồn tại trong corpus đang đánh giá (thiếu → dừng, không âm thầm bỏ khỏi gold); báo riêng số record bị loại lúc chuẩn bị dữ liệu.

**Lý do D6 và giới hạn của nó**: bằng chứng hiện có cho thấy bất đối xứng về hard negative được tuyển theo query — trên BM25 dev, 60,3% câu có top-1 là chính hard negative của nó (89,6% có ≥1 trong top-5), trong khi 0 câu test gốc có hard negative của mình hoặc của câu cùng context trong corpus. Số liệu này chưa đủ để kết luận mọi hyperparameter chọn trên dev đều lệch, hay dense hưởng lợi hơn BM25. Phương án (b) loại bỏ nguồn bất đối xứng này nhưng không bảo đảm dev và test khó bằng nhau. Corpus vẫn chứa context lấy từ benchmark nên không gọi là "hoàn toàn độc lập nhãn"; quy tắc chính xác là: **không đưa passage được tuyển theo từng query vào corpus đánh giá**.

**P4 — đã duyệt**: chuỗi distractor Wikipedia bỏ qua bài có title chuẩn hóa trùng chính xác với title của bất kỳ context nguồn base nào, thu thập TRƯỚC dedup. Chuẩn hóa title: NFC → thay `_` bằng khoảng trắng → lowercase → gộp khoảng trắng → strip. Title thiếu/rỗng không được suy đoán; bỏ qua title đó khi lập tập loại trừ, giữ bài distractor nếu không có title để đối chiếu, và báo số lượng. Một tập title loại trừ cố định áp dụng cho mọi tier (2.3.d).

P4 giảm passage cùng nguồn chưa được gán gold, không bảo đảm loại hết passage gần trùng (bài đổi tên hoặc diễn đạt lại vẫn có thể lọt qua). Báo cáo phải mô tả điều kiện **distractor đã lọc theo title nguồn**, không khẳng định corpus đã sạch near-duplicate. Đây là lọc theo nguồn bài viết, không lọc theo danh sách hard negative.

## 0. Nguyên tắc chung & phân chia trách nhiệm module

- **Data/Splits/Corpus** sở hữu: load dataset gốc, normalize, dedup, gán `passage_id`, định nghĩa train/dev/test, kiểm tra leakage, xây corpus đánh giá theo danh sách nguồn được phép (2.3.d). Không module nào khác được dedup hay chia split lại.
- **BM25 / Dense / Hybrid retriever** chỉ nhận corpus đã chuẩn bị sẵn (đã có `passage_id`), trả `RetrievalResult`, không tự ý sửa corpus.
- **Reranker** chỉ sắp xếp lại/lọc trong retrieval pool (top-`retrieval_k` retriever trả về), không tự truy vấn corpus, không thêm passage mới.
- **Reader** chỉ nhận passage đã được chọn (từ retriever hoặc reranker), không tự retrieval.
- **Evaluation** chỉ tính metric từ dữ liệu đã lưu sẵn (`PredictionRecord`, hoặc `RetrievalResult` ở chế độ retrieval_only), không tự chạy training hay inference model, không tự đo latency.
- **Integration** sở hữu việc đo latency (thiết bị, batch, warm-up, GPU sync) — xem 8.3.

Nhờ ranh giới rõ, đổi 1 module (vd đổi BM25 sang implementation khác) không kéo theo phải sửa Evaluation/Reranker.

## 1. Schema chung

### 1.1 Passage record (Corpus)
```python
PassageRecord = {
    "passage_id": str,          # sha256(normalized_text).hexdigest()[:16]
    "text": str,                 # nguyên bản của CHUNK này — reader dùng để khớp answer_start
    "normalized_text": str,      # dùng để hash + BM25 tokenize
    "source": str,                # "germanquad" | "germandpr_positive" | "wikipedia_distractor" — chỉ các nguồn được phép vào corpus đánh giá (2.3.d); hard negative GermanDPR không phải nguồn corpus
    "title": str | None,          # title bài Wikipedia nguồn (dòng đầu context GermanQuAD, field `title` của GermanDPR, field `title` của dump)
    "char_len": int,
    "parent_document_id": str,    # sha256(normalize_text(toàn văn context/ctx gốc CHƯA chunk))[:16]
    "chunk_start_char": int,      # offset bắt đầu của chunk này trong parent_document gốc
    "chunk_end_char": int,        # offset kết thúc (exclusive) trong parent_document gốc
}
```
`parent_document_id`/`chunk_start_char`/`chunk_end_char` là thông tin **provenance** — dùng để ánh xạ answer_start (tính trên context gốc) sang offset trong chunk bằng phép trừ, KHÔNG cần string-search (xem 2.3.c). Khi 2 chunk từ nguồn khác nhau trùng `passage_id` (cùng nội dung sau normalize), 3 field provenance này lấy theo `text`/`source` đại diện (cùng quy tắc ưu tiên nguồn ở 2.3.a) — chỉ phản ánh 1 occurrence, không phải toàn bộ occurrence.

### 1.2 QA / gold record — chỉ dùng cho GermanQuAD
```python
QARecord = {
    "query_id": str,                     # "gq-{split_gốc}-{id_gốc}", vd "gq-test-40369", "gq-train-51870" — xem ghi chú dưới
    "question": str,
    "gold_evidences": [
        {
            "passage_id": str,             # chunk chứa TRỌN đáp án này
            "answer_text": str,
            "answer_start": int,            # offset trong `text` của CHÍNH passage_id này (không phải context gốc)
            "answer_end": int,               # exclusive
        }
    ],
    "reference_answers": list[str],       # TOÀN BỘ đáp án tham chiếu hợp lệ (kể cả đáp án không map được vào chunk nào) — dùng cho EM/F1 max-over-references, xem 3.3
    "source_dataset": str,                 # luôn "germanquad"
    "source_split": str,                   # "train" | "dev" | "test" — gán theo split hợp nhất, xem 2.3.e
}
```
`gold_passage_ids` (dùng ở module Evaluation/BM25/Dense) được **suy ra**, không lưu riêng: `gold_passage_ids = sorted({e["passage_id"] for e in gold_evidences})` — tránh 2 nguồn dữ liệu có thể lệch nhau. `gold_evidences` rỗng nếu không map được đáp án nào vào chunk nào (xem 2.3.c) — record vẫn được GHI RA (không xóa khỏi file), chỉ bị loại khỏi mọi metric vì suy ra `gold_passage_ids` rỗng (mẫu số `N_eval`, xem 3.3). Mỗi `gold_evidences[].answer_text` phải thuộc `reference_answers`; vì vậy `reference_answers` rỗng chỉ hợp lệ khi `gold_evidences` cũng rỗng.

**`query_id` mã hoá split GỐC của dataset, không phải split đã gán** (`source_split` mới là split đã gán): 1 câu gốc train được đưa vào dev vẫn có `query_id` `gq-train-…`, `source_split="dev"`. Nhờ vậy `query_id` ổn định khi đổi seed/tỉ lệ dev và luôn truy vết được nguồn gốc. Evaluation không được parse `query_id` để suy ra split.

### 1.3 Retrieval training record — dùng cho GermanDPR
```python
RetrievalTrainingRecord = {
    "query_id": str,                            # "gdpr-{split_gốc}-{số_thứ_tự_dòng_trong_split_gốc}" (GermanDPR không có id), vd "gdpr-test-17"
    "question": str,
    "positive_texts": list[str],                # LUÔN là text của chunk (KHÔNG bao giờ là context gốc chưa chunk, xem 2.3.f) — có thể >1 nếu nhiều chunk chứa occurrence hợp lệ
    "positive_passage_ids": list[str],           # song song index với positive_texts — LUÔN khác None (record không map được thì bị loại, không giữ lại với None); luôn tồn tại trong mọi tier vì positive context là nguồn base (D6)
    "hard_negative_groups": [
        {
            "source_context_id": str,             # sha256(normalize_text(hard-negative context gốc chưa chunk))[:16] — cùng công thức parent_document_id ở 1.1; nhóm mọi chunk cùng nguồn
            "texts": list[str],                    # mọi chunk sinh ra từ context này — text được lưu ngay trong record, không cần corpus
            "passage_ids": list[str],              # cùng công thức passage_id ở 1.1; KHÔNG bắt buộc tồn tại trong corpus đánh giá (D6)
        }
    ],
    "source_split": str,   # "train" | "dev" | "test" — gán theo split hợp nhất, xem 2.3.e
}
```
Tách khỏi `QARecord` vì GermanDPR **không đảm bảo có `answer_start`** như GermanQuAD. `positive_texts`/`positive_passage_ids` là nguồn chính cho training (module 5/7 đọc trực tiếp từ đây, KHÔNG đọc từ `qa_germanquad_*.jsonl`). `hard_negative_groups` nhóm chunk theo **context gốc** — mỗi group đóng góp **tối đa 1 chunk/epoch** khi train (xem 5.3), tránh 1 context dài (nhiều chunk) chiếm trọng số lớn hơn context ngắn chỉ vì bị chunk nhiều mảnh hơn. Xem thuật toán xác định positive/hard-negative ở 2.3.f/2.3.g.

### 1.4 Retrieval result (dùng chung BM25/Dense/Hybrid/Reranker)
```python
RetrievalResult = {
    "query_id": str,
    "passage_id": str,
    "rank": int,          # bắt đầu từ 1
    "score": float,        # lớn hơn = tốt hơn; KHÔNG cùng thang đo giữa các retriever
    "retriever_name": str, # "bm25" | "dense" | "hybrid" | "cross_encoder"
}
```
Quy ước bắt buộc: score giữa BM25 và Dense **không so sánh trực tiếp được** (thang đo khác nhau). Mọi bước fusion (hybrid V2) phải normalize (min-max hoặc z-score theo từng query) trước khi cộng điểm.

### 1.5 Prediction record (đầu ra pipeline — lưu để tính lại metric không cần chạy lại model)
```python
PredictionRecord = {
    "run_id": str,        # gồm config_id + dataset/split + checkpoint bundle hash + seed + UUID của lần chạy (8.4); không trùng giữa dev/test hoặc hai lần chạy mới
    "config_id": str,      # "{retriever}_{tier}_rerank-{on|off}", retriever ∈ bm25|dense|hybrid|oracle; 1 thành phần con của run_id
    "corpus_id": str,       # "base" | "mid" | "large" — corpus/index thực tế đã dùng cho query này
    "retrieval_k": int,      # số candidate đầu của retriever tạo thành retrieval pool (V1: 20, xem 7.4)
    "reader_k": int,         # số passage reader được đọc (V1: 5), như nhau dù bật/tắt reranker; end-to-end bắt buộc 1 ≤ reader_k ≤ retrieval_k
                             # 2 giá trị này bằng nhau ở mọi record của 1 run và khớp run_manifest.json (8.7); oracle: vẫn ghi nhưng không dùng
    "query_id": str,
    "status": str,           # "ok" | "error"
    "error_stage": str | None,  # "retrieval" | "rerank" | "read" | None — chỉ set khi status="error"
    "retrieved": list[RetrievalResult],          # tối đa top-N từ retriever (N cấu hình ≥ retrieval_k), trước rerank; được ít hơn nếu không đủ kết quả; [] khi oracle hoặc chưa retrieval xong
    "reranked": list[RetrievalResult] | None,     # reader_k passage do reranker chọn; None nếu không dùng reranker
    "predicted_answer_text": str | None,           # None nếu không có đáp án hợp lệ — KHÔNG dùng ""
    "predicted_passage_id": str | None,
    "predicted_answer_start": int | None,   # offset trong `text` gốc của passage, KHÔNG phải normalized_text; None nếu không có đáp án
    "predicted_answer_end": int | None,      # exclusive — theo quy ước Python slice text[start:end], None nếu không có đáp án
    "reader_span_prob": float | None,        # xác suất span đã softmax-normalize TRONG passage đó (heuristic, xem 6.3); None nếu không có đáp án
    "latency_ms": {"retrieval": float | None, "rerank": float | None, "read": float | None, "total": float | None},
}
```
Quy ước suy ra (không lưu riêng, tránh 2 nguồn lệch nhau):
- **retrieval pool** = `retrieved[:retrieval_k]` — như nhau dù bật hay tắt reranker.
- **reader input** = `reranked` nếu có, ngược lại `retrieved[:reader_k]`; với `retriever="oracle"` (chế độ reader-only, 8.3) = toàn bộ `gold_passage_ids` của query.
- **Không có đáp án** ⇔ cả 5 field đáp án đều `None` (cùng None hoặc cùng có giá trị). `predicted_answer_text=""` là dữ liệu sai. Reader chạy xong mà không tìm được span hợp lệ vẫn là `status="ok"` với 5 field `None`, không phải `status="error"`.
- Khi `status="ok"`: `latency_ms.retrieval`, `read`, `total` là số hữu hạn không âm; `rerank` là số hữu hạn không âm nếu bật reranker, ngược lại là `None`. Oracle: `retrieval=0`, `rerank=None`.
- Khi `status="error"`: 5 field đáp án bắt buộc `None`; giữ các kết quả và latency đã hoàn thành, giai đoạn chưa hoàn thành có latency `None`; `total` được phép `None`. Các field định danh/cấu hình vẫn đầy đủ; `retrieved=[]` nếu chưa có kết quả retrieval, `reranked=None` nếu chưa có kết quả rerank. Evaluation xác định bật/tắt reranker từ manifest, không suy từ record lỗi.

## 2. Module: Data / Splits / Corpus

### 2.1 Mục tiêu và phạm vi
V1. Load dataset gốc, normalize text, dedup + gán `passage_id`, định nghĩa train/dev/test, kiểm tra leakage chéo GermanQuAD↔GermanDPR, xây 3 tầng corpus lồng nhau. KHÔNG train model, KHÔNG tính metric.

### 2.2 Input/Output
- Input: `deepset/germanquad` (train/test), `deepset/germandpr` (train/test), `wikimedia/wikipedia` config `20231101.de` (streaming, chỉ dùng cho distractor).
- Output: `corpus_{base,mid,large}.jsonl` (mỗi dòng 1 `PassageRecord`), `qa_germanquad_{train,dev,test}.jsonl` (mỗi dòng 1 `QARecord`), `retrieval_germandpr_{train,dev,test}.jsonl` (mỗi dòng 1 `RetrievalTrainingRecord`), `wikipedia_distractors.jsonl`, `base_title_exclusions.json`, cùng 5 báo cáo liệt kê ở 2.7.

### 2.3 Quy tắc xử lý

**a. Normalization 2 tầng** — tách `text` (nguyên bản, dùng khớp `answer_start`) khỏi `normalized_text` (lowercase, strip khoảng trắng thừa, unicode NFC, **KHÔNG bỏ dấu umlaut** ä/ö/ü/ß — đây là chữ cái Đức hợp lệ, không phải dấu câu). `normalized_text` dùng để hash → `passage_id` và làm input BM25 tokenizer.

Nếu hai `text` khác nhau normalize ra cùng `passage_id` (dedup collision — hiếm, thường do khác biệt whitespace/casing giữa 2 nguồn): giữ bản xuất hiện đầu tiên theo thứ tự ưu tiên source (germanquad > germandpr_positive > wikipedia_distractor) làm `text` đại diện. Việc ánh xạ offset đáp án sang chunk **mặc định LUÔN dùng char-offset** (xem 2.3.c) — string-search **chỉ** là fallback cho đúng trường hợp collision này, khi answer_start gốc (tính trên document nguồn của QARecord) không còn khớp trực tiếp với `text` đại diện (đến từ document khác). Khi fallback:
1. Tìm mọi occurrence của `answer_text` trong `text` đại diện.
2. Đúng 1 occurrence → dùng offset đó, không cần log.
3. 0 occurrence → không map được, loại evidence này (không phải cả QARecord — record vẫn giữ các evidence khác nếu có), log vào `ambiguous_offset_remap_report.json`.
4. **>1 occurrence (mơ hồ)** → **không được mặc định lấy occurrence đầu tiên** (audit thực tế: 44,7% trường hợp multi-occurrence có answer_start gốc KHÔNG phải occurrence đầu). Chọn occurrence có vị trí tương đối (`offset / len(text)`) gần nhất với vị trí tương đối của answer gốc trong document nguồn; luôn log case này vào `ambiguous_offset_remap_report.json` (query_id, passage_id, answer_text, toàn bộ candidate offset, offset đã chọn) để có thể review tay.

**b. passage_id** = `sha256(normalized_text).hexdigest()[:16]`.

**c. Chunking — quy tắc độc lập với đáp án, gán gold sau khi chunk** — MỌI context/ctx (GermanQuAD, GermanDPR positive, GermanDPR hard negative, bài Wikipedia distractor) đều chunk bằng cùng một rule cố định: cửa sổ 100 từ, overlap 20 từ (bước 80 từ; chunk cuối lấy phần còn lại). Chunk của hard negative chỉ đi vào `hard_negative_groups` (training), không vào corpus đánh giá (2.3.d). Rule này **không phụ thuộc vị trí đáp án nằm ở đâu** — không "mở rộng chunk theo đáp án", vì cách đó làm việc xây corpus phụ thuộc ngược vào nhãn đánh giá.

Sau khi chunk xong (độc lập đáp án), mới **ánh xạ gold vào `gold_evidences`** — LUÔN bằng char-offset, KHÔNG string-search (string-search chỉ là fallback của 2.3.a cho case dedup-collision riêng): với mỗi `answer_text`/`answer_start` gốc (tính trên context CHƯA chunk) trong dữ liệu GermanQuAD, so trực tiếp `answer_start`/`answer_end` với `chunk_start_char`/`chunk_end_char` (đã lưu trong `PassageRecord`, xem 1.1) của từng chunk sinh từ CÙNG context đó — chunk nào thỏa `chunk_start_char ≤ answer_start` và `answer_end ≤ chunk_end_char` thì tạo 1 entry `gold_evidences` (xem 1.2) với `answer_start = answer_start_gốc − chunk_start_char`, `answer_end = answer_end_gốc − chunk_start_char` (offset LOCAL trong chunk, ghi đè tên field `answer_start`/`answer_end` của entry — không phải offset trong context gốc nữa). Có thể >1 chunk thỏa (2 chunk overlap cùng chứa trọn đáp án) → tạo nhiều entry, tất cả coi là tương đương cho mục đích retrieval (xem định nghĩa Recall@k ở 3.3). Nếu KHÔNG chunk nào chứa trọn đáp án (answer rơi vào khoảng overlap không đủ che phủ) → answer đó không tạo evidence, chỉ còn nằm trong `reference_answers`. Nếu TOÀN BỘ answer của 1 câu hỏi đều không map được → `gold_evidences = []` (suy ra `gold_passage_ids` rỗng), loại câu hỏi đó khỏi retrieval/end-to-end eval, **nhưng QARecord vẫn được ghi ra file** (không âm thầm xóa) và log vào `chunk_coverage_report.json`.

Báo cáo bắt buộc: `chunk_coverage_report.json` ghi % câu hỏi có ≥1 gold chunk hợp lệ (coverage rate) trên từng dataset/split — nếu coverage <98%, cần xem lại overlap size trước khi build corpus chính thức.

**Bằng chứng cho quyết định chunk mọi context**: 47,3% context vượt 300 gbert-token (đo trên sample 2.000, `results/data_audit_report.json`) — không chunk thì `passage_encoder` (max_length=300) và reader sẽ cắt mất nội dung ở gần một nửa context.

**d. Corpus đánh giá — xây theo danh sách nguồn được phép, 3 tầng lồng nhau (D6)**. Corpus đánh giá không chứa passage được tuyển theo từng query. Quy tắc viết theo **nguồn được đưa vào**, không phải theo việc xóa những passage từng mang nhãn hard negative:
- **Nguồn base**: TOÀN BỘ chunk (không chỉ chunk chứa đáp án) của mọi context GermanQuAD (train+test) và mọi positive context GermanDPR (train+test), dedup theo quy tắc chung 2.3.a–b.
- **Nguồn distractor**: một chuỗi chunk Wikipedia lấy mẫu cố định (bên dưới).
- **Hard negative GermanDPR không phải nguồn corpus**: chỉ lưu trong `hard_negative_groups` để train (2.3.g); `passage_id` của chúng không bắt buộc tồn tại trong corpus. Không lọc gì theo danh sách hard negative: chunk trùng với context thuộc nguồn base hợp lệ vẫn nằm trong base (mang `source` của nguồn base); chunk Wikipedia trùng hard negative vẫn được giữ.

**Tầng** — mốc thiết kế, không phải con số tối ưu đã được thực nghiệm xác nhận:

| Tầng | Thành phần | Quy mô |
|---|---|---|
| `corpus_base` | nguồn base | số thật sau build (dự kiến ~7,5K) |
| `corpus_mid` | base + (50.000 − số passage của base) chunk đầu của chuỗi distractor | tổng 50.000 |
| `corpus_large` | base + (150.000 − số passage của base) chunk đầu của CÙNG chuỗi distractor | tổng 150.000 |

**Chuỗi distractor Wikipedia — một chuỗi duy nhất, cố định cho mọi tầng**:
1. Thu thập title từ TOÀN BỘ context nguồn base trước dedup (kể cả context của QA bị loại vì leakage). Chuẩn hóa theo P4; lưu tập title khác rỗng đã sort vào `base_title_exclusions.json`. Không chỉ lấy title đại diện còn lại trong corpus. Tập này dùng chung cho mọi tier.
2. Duyệt streaming toàn bộ snapshot đã ghim của `wikimedia/wikipedia`, config `20231101.de`. Chuẩn hóa `article_id` thành chuỗi số thập phân không có số 0 thừa ở đầu. Chọn bài khi `stable_seed("wiki-sample", article_id) % 10000 < R` (`stable_seed` ở 5.3). Bắt đầu `R=100`; nếu chưa đủ chunk hợp lệ thì thử lần lượt `200, 400, 800, 1600, 3200, 6400, 10000`, mỗi lần duyệt hết snapshot. Chọn R đầu tiên đủ số lượng; R không được chọn theo kết quả retrieval/QA. R=10000 vẫn thiếu thì dừng với báo cáo thiếu hụt.
3. Bỏ bài có title chuẩn hóa trùng chính xác tập loại trừ. Không fuzzy-match, không suy đoán title thiếu. Báo số context nguồn base thiếu title, số bài được chọn thiếu title và số bài bị lọc. Title của distractor thiếu/rỗng thì giữ bài để xử lý tiếp.
4. Chunk bài còn lại theo 2.3.c, bỏ chunk rỗng và chunk có `passage_id` trong base. Khóa thứ tự mỗi occurrence là `(hash_order, article_id, chunk_start_char)`, trong đó `hash_order = sha256(("wiki-order|" + article_id).encode("utf-8")).hexdigest()` dùng ĐỦ 64 ký tự hex; `article_id` so theo chuỗi, `chunk_start_char` so theo số nguyên. Với các occurrence trùng `passage_id`, giữ occurrence có khóa nhỏ nhất làm đại diện, KHÔNG giữ theo thứ tự đọc dump. Sắp các đại diện theo khóa này để tạo chuỗi. Vì thế cả thứ tự lẫn text/title/provenance đại diện đều không phụ thuộc thứ tự dump, kể cả khi hash_order trùng nhau.
5. Chỉ sau khi lọc/dedup hoàn tất và chuỗi đủ dài mới đóng băng R, tập title và chuỗi. mid lấy tiền tố `50000 − |base|`, large lấy tiền tố `150000 − |base|` của CÙNG chuỗi → base ⊆ mid ⊆ large và tổng đúng mốc. Mỗi lần tăng R trước khi đóng băng phải dựng lại toàn bộ chuỗi theo quy tắc trên, không nối bù vào chuỗi cũ. Sau khi đóng băng không đổi R khi mở rộng tier.
6. Lưu toàn bộ chuỗi đã chọn dưới dạng `PassageRecord` trong `wikipedia_distractors.jsonl`, đúng thứ tự (gồm text và provenance, không chỉ ID). Lưu ID theo thứ tự, R, snapshot và SHA-256 của các artifact vào `corpus_manifest.json`. Khi dựng mid/large chỉ đọc tiền tố artifact đã lưu; không đọc lại dump hay preprocess lại passage đã có.

Cùng một `qa_*_test` (gold không đổi) dùng cho cả 3 tầng.

**Tái sử dụng khi mở rộng tier**: vì base ⊆ mid ⊆ large, khi build `corpus_mid` chỉ cần preprocess (normalize/chunk/hash) và encode dense embedding cho các passage MỚI (distractor thêm vào), KHÔNG preprocess/encode lại các passage đã có từ `corpus_base`; tương tự khi build `corpus_large` từ `corpus_mid`. FAISS index của tier lớn hơn = merge index tier nhỏ hơn + embedding của passage mới (xem 5.8).

**e. Split — một quyết định train/dev hợp nhất theo question identity; test của mỗi dataset giữ đúng test gốc**: chia dev độc lập cho GermanQuAD và GermanDPR có thể khiến 1 câu hỏi rơi vào dev của reader nhưng vẫn nằm trong train của retriever. Nhưng hợp nhất cũng không được đẩy dòng train của dataset này vào test của dataset kia (lỗi V1.0: 1.024 dòng GermanQuAD train rơi vào `qa_germanquad_test`). Quy trình:
1. Chuẩn hóa câu hỏi (lowercase+strip+NFC) → `question_key`. `TestKeys` = hợp `question_key` của test gốc GermanQuAD và test gốc GermanDPR.
2. Với mỗi dòng r thuộc dataset X:
   - r thuộc test gốc của X → `source_split="test"` (ghi vào file test của X).
   - r thuộc train gốc của X và `question_key(r) ∈ TestKeys` → **loại vì leakage**: không ghi vào file split nào, liệt kê `query_id` vào `split_report.json`. Không bao giờ chuyển r sang test của X hay của dataset kia.
   - còn lại → vào pool train/dev ở bước 3.
3. Pool = các `question_key` không thuộc `TestKeys`. Sample 10% số key (seed=42, trên danh sách key đã sort) làm `dev`, phần còn lại làm `train`. Mọi dòng của CẢ HAI dataset có cùng key nhận cùng split — một quyết định duy nhất áp dụng cho cả GermanQuAD lẫn GermanDPR.

Hệ quả: `qa_germanquad_test` = đúng các dòng GermanQuAD test gốc (kể cả dòng có `gold_evidences=[]`); `retrieval_germandpr_test` = đúng các dòng GermanDPR test gốc map được chunk (2.3.f). Nguyên tắc chống leakage: câu hỏi của BẤT KỲ tập test nào không bao giờ xuất hiện trong train/dev của BẤT KỲ thành phần nào.

Passage test nằm trong corpus index **không** bị coi là leakage (đây là điều kiện cần để retrieval hoạt động) — chỉ câu hỏi/QA-pair xuất hiện ở training mới tính leakage.

**Số liệu thật đã xác nhận** (data audit 2026-09-10): `germandpr_train` ⊆ `germanquad_train` (9.254/9.254 key); **`germandpr_test` (1.025/1.025 key) nằm trọn trong `germanquad_train`** → các dòng GermanQuAD train mang 1.025 key này bị loại khỏi train/dev (~8,9% GermanQuAD train) và KHÔNG vào `qa_germanquad_test`; `germandpr_train` ∩ `germandpr_test` = 1 key → dòng train đó bị loại. `germanquad_test` không trùng key nào với 2 split GermanDPR.

**f. GermanDPR sau chunking — xác định answer-bearing chunk bằng HỢP tất cả occurrence, không dùng nguyên context dài khi mơ hồ**: coi TẤT CẢ chunk sinh từ 1 positive context là positive (cách làm cũ) SAI. Cách làm "fallback về context gốc chưa chunk khi mơ hồ" (bản trước) CŨNG sai — `passage_encoder` truncate ở `max_length=300` token (xem 5.3), context gốc dài có thể bị cắt mất đúng đoạn chứa đáp án. Quy trình đúng, dùng HỢP của mọi occurrence thay vì chỉ thử tìm 1 occurrence duy nhất:
1. Với **mọi** `answer_text` trong `answers` của câu hỏi (không dừng ở answer đầu tiên): tìm **toàn bộ** occurrence (không chỉ occurrence đầu) trong `positive_ctx.text` GỐC (chưa chunk).
2. Với mỗi occurrence tìm được: xác định mọi chunk chứa TRỌN occurrence đó bằng `chunk_start_char`/`chunk_end_char` (như 2.3.c).
3. **Hợp nhất** toàn bộ chunk tìm được qua bước 2 (từ mọi occurrence của mọi answer_text), dedup theo `passage_id` → đây là `positive_texts`/`positive_passage_ids` của record (luôn khác None, có thể có 1 hoặc nhiều phần tử). Chunk KHÁC của cùng positive context KHÔNG chứa occurrence nào thì KHÔNG được gán positive — nhưng vẫn nằm trong corpus base như mọi chunk khác của positive context ("positive" là nhãn training, không phải điều kiện vào corpus, 2.3.d).
4. Khi `len(positive_texts) > 1`: dùng CHUNG chính sách chọn 1 positive/epoch với dense retriever (SHA-256 seed, xem 5.3) — không có xử lý riêng ở bước data prep.
5. **Không chunk nào chứa trọn BẤT KỲ occurrence nào** (hiếm — occurrence rơi vào khoảng overlap không đủ che phủ, tương tự 2.3.c) → loại HẲN record này khỏi dense/reranker training (không giữ lại dưới dạng text thô), log vào `germandpr_chunk_mapping_report.json`.

**Số liệu audit tham chiếu** (2026-09-10, toàn bộ 10.300 positive_ctxs train+test): **87,3%** (8.996/10.300) có đúng 1 positive chunk; **12,5%** (1.283/10.300) có nhiều positive chunk (2-5, phân phối: 2→1.172, 3→90, 4→20, 5→1); **0,20%** (21/10.300) không map được và phải loại theo thuật toán trên. Đây không phải kết quả nghiệm thu artifact V1.1. Sau khi đổi split/loại leakage, phải đo lại theo từng file; không yêu cầu tái hiện đúng các tỷ lệ này. Yêu cầu nghiệm thu là positive luôn là chunk hợp lệ, không dùng `positive_passage_ids=[None]` hay context gốc chưa chunk.

**g. Hard-negative grouping — nhóm theo context gốc, không trộn chunk từ nhiều context vào 1 danh sách phẳng**: hard negative KHÔNG áp dụng logic tìm answer-bearing chunk (không có "đáp án đúng" cần định vị — vai trò của nó là "không liên quan" ở bất kỳ phần nào), nhưng KHÔNG được đưa toàn bộ chunk của TẤT CẢ hard-negative context vào 1 danh sách phẳng (`hard_negative_texts` bản cũ) — cách đó làm batch không kiểm soát được kích thước và khiến context dài (nhiều chunk) có trọng số lớn hơn context ngắn. Thay vào đó: với mỗi hard-negative context của 1 câu hỏi, tạo đúng 1 `hard_negative_groups` entry — `source_context_id` = `parent_document_id` của context đó, `texts`/`passage_ids` = TOÀN BỘ chunk sinh ra từ context này (song song). Số group/query = số hard-negative context gốc của câu hỏi đó (thường 3, nhưng KHÔNG hard-code — data audit xác nhận 8/9.275 mẫu train có 1-2, xem 5.5). Loại khỏi `hard_negative_groups` bất kỳ group/chunk nào trùng `passage_id` với gold positive của CHÍNH câu hỏi đó (xem 5.3/5.5). Việc chọn ≤1 chunk/group để đưa vào batch là quyết định **lúc train** (per-epoch, xem 5.3), không phải lúc build data — data chỉ lưu đủ TOÀN BỘ chunk để module 5 tự chọn. Text của hard negative được lưu ngay trong record (`texts`); chúng không phải nguồn corpus đánh giá (D6, 2.3.d) nên `passage_ids` của group không bắt buộc tồn tại trong corpus.

### 2.4 Cấu hình
| Tham số | Giá trị mặc định | Tune trên |
|---|---|---|
| dev split ratio | 0.10 | cố định |
| seed | 42 | cố định |
| chunk size / overlap | 100 từ / 20 từ (áp dụng cho mọi context, xem 2.3.c) | cố định |
| `corpus_base` | nguồn base (2.3.d) — số thật sau build, dự kiến ~7,5K | không tune |
| `corpus_mid` | tổng **50.000** passage | mốc thiết kế, không tune |
| `corpus_large` | tổng **150.000** passage | mốc thiết kế, không tune |
| chuỗi distractor: seed / `R` | "wiki-sample"/"wiki-order"; R thử 100→200→400→800→1600→3200→6400→10000, lấy mức đầu đủ chunk sau lọc/dedup | chỉ theo số lượng, đóng băng trước eval |

### 2.5 Trường hợp biên
- Context rỗng/toàn khoảng trắng → loại khỏi corpus, log.
- Câu hỏi unanswerable / không có reference answer (GermanQuAD chuẩn không có dạng này) → vẫn ghi QARecord với `reference_answers=[]`, `gold_evidences=[]`; tự động nằm ngoài `N_eval` (3.3), đếm trong `chunk_coverage_report.json`.
- QARecord không map được đáp án nào vào chunk → ghi với `gold_evidences=[]` (KHÔNG bỏ khỏi file — V1.0 build đã bỏ 184 dòng, sai quy tắc này).
- Dòng train bị loại vì leakage (2.3.e) → không ghi vào file split nào, `query_id` liệt kê trong `split_report.json`.
- Hai passage trùng hash khác offset → xử lý theo 2.3.a.
- Distractor trùng hash với passage gốc → dedup trước khi cắt tiền tố; lấy tiếp chunk hợp lệ để đủ tổng 50.000/150.000, không chấp nhận giảm tổng tier vì trùng lặp. Chuỗi thiếu thì xử lý theo lịch R ở 2.3.d. Nếu base đã vượt 50.000 thì dừng và báo mốc thiết kế không còn phù hợp, không cắt bớt base.

### 2.6 Kiểm thử
- Unit: hash deterministic; dedup không tạo trùng ID cho 2 text khác nhau.
- Unit: với 1 context giả lập + answer_start biết trước, `gold_evidences` sinh ra có `answer_start`/`answer_end` (local, trong `text` của passage_id đó) sao cho `passage_text[answer_start:answer_end] == answer_text` — xác nhận offset đúng NGAY TRONG chunk, không phải trong context gốc.
- Unit: GermanDPR — với positive context giả lập có answer xuất hiện 2 lần ở 2 chunk khác nhau, `positive_passage_ids` phải chứa ĐÚNG 2 passage_id đó, không chứa passage_id của chunk thứ 3 không có occurrence nào (xác nhận không gán tràn lan, và hợp nhất đúng nhiều occurrence).
- Unit: GermanDPR — record không có chunk nào chứa trọn occurrence nào bị loại hẳn khỏi output (không xuất hiện trong `retrieval_germandpr_*.jsonl`, không phải giữ lại dưới dạng text thô).
- Unit: hard-negative grouping — 1 hard-negative context dài tạo N chunk phải cho ra ĐÚNG 1 `hard_negative_groups` entry với `len(texts) == N` (không tách thành N entry riêng).
- Smoke: build corpus_base trên 200 mẫu, mọi `gold_passage_ids` (suy ra từ `gold_evidences`, khác rỗng) tồn tại trong corpus, mọi `gold_evidences[].answer_start/answer_end` khớp đúng substring tại `text` của chính `passage_id` đó (KHÔNG so với context gốc).
- Unit: split — dòng GermanQuAD train có key thuộc GermanDPR test phải bị loại (không có trong file nào), KHÔNG xuất hiện trong `qa_germanquad_test`; dòng gốc train được đưa vào dev giữ `query_id` `gq-train-…` với `source_split="dev"`.
- Unit: nguồn corpus (D6) — chunk hard negative không trùng chunk base nào thì KHÔNG có trong corpus; chunk hard negative trùng 1 chunk context GermanQuAD thì CÓ trong base với `source="germanquad"`; mọi chunk của positive context (kể cả chunk không chứa đáp án) đều có trong base.
- Unit: chuỗi distractor — đảo thứ tự bài trong fixture (có chunk trùng giữa bài và hash_order giả lập trùng) vẫn ra cùng chuỗi `PassageRecord`, gồm text/title/provenance; phần distractor của mid là tiền tố của large; không có chunk distractor trùng `passage_id` với base. Thiếu chunk ở một mức R thì tăng đúng lịch, không nối bù; tải lại artifact đã đóng băng dựng được tier mà không đọc dump.
- Unit: P4 — title từ occurrence base không được giữ sau dedup vẫn có trong tập loại trừ; chuẩn hóa `_`, Unicode, casing và whitespace đúng; title thiếu không bị suy đoán; lọc title áp dụng giống nhau cho mid/large và số bài bị lọc/thiếu title được báo cáo.
- Nghiệm thu: mọi `QARecord` có `gold_evidences` khác rỗng đều trỏ tới passage tồn tại với offset đúng trong CHÍNH chunk đó; mọi gold `passage_id` (QARecord và GermanDPR `positive_passage_ids`) nằm trong base; không passage nào của corpus có `source` ngoài 3 nguồn được phép; base ⊆ mid ⊆ large, `|mid|` = 50.000, `|large|` = 150.000; 0 `question_key` bị gán 2 split khác nhau; `qa_germanquad_test` chứa đúng và đủ các dòng GermanQuAD test gốc (mọi `query_id` bắt đầu bằng `gq-test-`), `retrieval_germandpr_test` chỉ chứa dòng GermanDPR test gốc; tổng số dòng GermanQuAD trong 3 file + số dòng bị loại vì leakage = 13.722; chunk coverage rate ≥98% trên từng file; 0 GermanDPR record gán positive cho chunk không chứa đáp án.

### 2.7 Artifact
`corpus_{base,mid,large}.jsonl`, `qa_germanquad_{split}.jsonl`, `retrieval_germandpr_{split}.jsonl`; `wikipedia_distractors.jsonl` (toàn bộ `PassageRecord` của chuỗi đã đóng băng, đúng thứ tự); `base_title_exclusions.json` (danh sách title chuẩn hóa khác rỗng đã sort, duy nhất); và 5 báo cáo:
- `corpus_manifest.json` — danh sách `passage_id` đóng băng của base; chuỗi distractor theo đúng thứ tự (danh sách `passage_id`) cùng R và lịch các mức đã thử, tên seed, snapshot dump đã ghim; đường dẫn và SHA-256 nội dung của corpus từng tier, `wikipedia_distractors.jsonl`, `base_title_exclusions.json`; điểm cắt và tổng thật của từng tầng; số bài Wikipedia đã duyệt/đã chọn/bị bỏ theo P4 và số title thiếu theo 2.3.d; ngày build. Đủ để dựng lại chính xác mọi tier từ artifact đã lưu và để Evaluation kiểm tra gold/kết quả retrieval thuộc corpus (3.3).
- `split_report.json` — số key/dòng mỗi split theo dataset, danh sách `query_id` bị loại vì leakage (2.3.e).
- `chunk_coverage_report.json` — theo từng file: số record, số record `gold_evidences=[]`, coverage rate.
- `ambiguous_offset_remap_report.json` — case string-search fallback mơ hồ (2.3.a).
- `germandpr_chunk_mapping_report.json` — xem dưới.

`germandpr_chunk_mapping_report.json` (xem 2.3.f/2.3.g) gồm:
- Số record có đúng 1 positive chunk / có nhiều positive chunk / bị loại hẳn (không chunk nào chứa occurrence).
- Phân phối số positive chunk/record (vd `{1: 8996, 2: 1172, 3: 90, ...}`).
- Xác nhận (bool + đếm) không còn record nào dùng context gốc chưa chunk hoặc `positive_passage_ids` chứa `None`.
- Số `hard_negative_groups`/query và phân phối số chunk/group (để kiểm tra context dài không áp đảo).

### 2.8 Tài nguyên
CPU-only, không cần GPU. Mỗi mức R cần một lượt streaming TOÀN BỘ snapshot tiếng Đức; có thể cần nhiều lượt nếu mức trước chưa đủ chunk (2.3.d). Dung lượng tải thực tế phụ thuộc snapshot và số lần đọc, phải ghi trong báo cáo chạy; chỉ giữ bài/chunk được chọn. Nội dung đã chọn được lưu ở `wikipedia_distractors.jsonl`, cấu hình và fingerprint ở `corpus_manifest.json`; không đọc lại dump khi mở rộng tier.

## 3. Module: Evaluation

### 3.1 Mục tiêu và phạm vi
V1. Chỉ tính metric từ dữ liệu đã lưu sẵn (không chạy model, không đo latency). Ba chế độ:
- `end_to_end` — `PredictionRecord` của pipeline retriever → [reranker] → reader: retrieval metrics, QA metrics, latency summary, error decomposition, manual review.
- `reader_only` — `PredictionRecord` của run `retriever="oracle"` (reader đọc thẳng gold passage, 8.3): QA metrics + latency. Nhãn báo cáo là "mốc năng lực reader trên gold passage", không phải trần tuyệt đối. Không có retrieval metrics hay error decomposition.
- `retrieval_only` — `RetrievalResult` rows (vd output BM25/Dense trước reader): retrieval metrics, không cần reader.

### 3.2 Input/Output
- Gold: `QARecord` JSONL (`qa_germanquad_test.jsonl` cho báo cáo chính thức; `qa_germanquad_dev.jsonl` trong lúc phát triển), hoặc — chỉ ở `retrieval_only` — `RetrievalTrainingRecord` JSONL (`retrieval_germandpr_test.jsonl`, gold = `positive_passage_ids`; sanity-check, P3) kèm `germandpr_chunk_mapping_report.json` để lấy số record bị loại lúc chuẩn bị dữ liệu.
- Prediction: `predictions_{run_id}.jsonl` (`PredictionRecord`) kèm entry của `run_id` trong `run_manifest.json` (nguồn cấu hình chuẩn — vẫn có khi file predictions rỗng, P1); hoặc `RetrievalResult` JSONL ở `retrieval_only`, kèm metadata chỉ rõ dataset/split và fingerprint của corpus/gold. Tên file retrieval ở 4.7/5.2 là tên logic; artifact của các run khác nhau phải nằm trong thư mục riêng theo `run_id` để không ghi đè.
- Corpus: `corpus_manifest.json` — xác định tier đang đánh giá và tập `passage_id` của tier đó, để kiểm tra gold ⊆ corpus (3.3).
- Output: `metrics_{run_id}.json` (hoặc theo tên file retrieval results ở `retrieval_only`).

### 3.3 Quy tắc xử lý

**Tập đánh giá & mẫu số (D2)**: tập đánh giá chính thức là `qa_germanquad_test.jsonl` = đúng GermanQuAD test gốc (2.3.e); chỉ tập này có cả gold passage lẫn gold answer span nên là tập DUY NHẤT cho funnel retrieval → selection → read → error decomposition. **Một mẫu số duy nhất `N_eval`** = số QARecord trong file gold có `gold_evidences` khác rỗng (theo dữ liệu hiện tại dự kiến 2.196/2.204; số thật lấy từ `chunk_coverage_report.json`):
- Mọi metric **chất lượng** — Recall@k, MRR@10, GoldCoverage@k, EM/F1, error decomposition — dùng chung mẫu số `N_eval`. Query thiếu prediction hoặc `status="error"` vẫn nằm trong mẫu số (retrieval tính như danh sách rỗng hoặc danh sách đã có; EM=F1=0) — không âm thầm bỏ khỏi mẫu số.
- Latency KHÔNG dùng `N_eval` mà dùng `n_latency` = số record `status="ok"` thuộc `N_eval` có giá trị ở giai đoạn đó; luôn báo `n_latency` cạnh số liệu.
- Câu gold rỗng KHÔNG vào bất kỳ mẫu số nào; số lượng báo cáo riêng (`n_excluded_empty_gold`). Mọi bảng kết quả ghi rõ `N_eval / tổng số câu của file` (vd "2.196/2.204").

GermanDPR test dùng riêng như retrieval-only sanity-check (gold = `positive_passage_ids`, P3), không vào funnel/báo cáo chính; mẫu số = số record của `retrieval_germandpr_test` (mọi record đều có positive). Mọi positive `passage_id` phải tồn tại trong corpus đang đánh giá — thiếu thì dừng và báo lỗi, không âm thầm bỏ khỏi gold. Output báo riêng số record GermanDPR test bị loại lúc chuẩn bị dữ liệu (lấy từ `germandpr_chunk_mapping_report.json`) cạnh mẫu số.

**Retrieval — Recall@k CHỈ có 1 định nghĩa chính thức** (bản trước dùng lẫn lộn 2 công thức dưới cùng tên "Recall@k" — sửa lại cho rõ):

```
Recall@k = (1/N_eval) · Σ_i  1{ top-k(query_i) ∩ gold(query_i) ≠ ∅ }
```

(top-k lấy từ `retrieved` — danh sách của retriever TRƯỚC rerank.) Tức trung bình trên `N_eval` query của chỉ số nhị phân "tìm được ÍT NHẤT 1 gold_passage_id trong top-k" — về mặt công thức, đây chính là Hit@k, và **đây là định nghĩa duy nhất được gọi là "Recall@k" trong toàn bộ spec/báo cáo**. Công thức phân số kiểu `|top-k ∩ gold| / |gold|` (hữu ích khi 1 query có nhiều `gold_passage_ids` — vd overlapping chunk ở 2.3.c) **KHÔNG được gọi là Recall@k** — nếu cần báo cáo, dùng tên riêng **`GoldCoverage@k`** và nói rõ đây là metric phụ, không phải metric chính.

MRR@10 = trung bình 1/rank của **gold passage đầu tiên xuất hiện trong top-10** (rank tính trên toàn bộ danh sách top-10 đã sắp theo score giảm dần), 0 nếu không có gold nào trong top-10.

**QA**: normalize trước so sánh — NFC → lowercase → **xóa** (không thay bằng khoảng trắng) ký tự khớp regex `[^\w\s]` → gộp khoảng trắng. Giữ ä/ö/ü/ß (`\w` của `re` trên str là Unicode-aware); không casefold (`ß` ≠ `ss`); không bỏ mạo từ tiếng Đức (khác script MLQA — số EM không so trực tiếp được với kết quả dùng MLQA). Tokenize whitespace-split sau normalize. EM/F1 lấy MAX qua toàn bộ `reference_answers` của `QARecord`. **"Không có prediction"** = không có record, hoặc `status="error"`, hoặc 5 field đáp án `None` (1.5) → EM=0, F1=0, đếm riêng `no_prediction`. **Oracle (`reader_only`) = "mốc năng lực reader trên gold passage"**, không gọi là trần tuyệt đối.

**Latency (D5)**: Evaluation KHÔNG đo và KHÔNG lọc warm-up — chỉ tóm tắt `latency_ms` đã ghi trong `PredictionRecord`: mean, p50, p95 cho từng giai đoạn (`retrieval`, `rerank`, `read`, `total`), percentile nội suy tuyến tính giữa 2 rank gần nhất, trên `n_latency` record (định nghĩa ở trên). Điều kiện đo (thiết bị, batch=1, warm-up, GPU sync, điểm bắt đầu/kết thúc) thuộc Integration, xem 8.3 — nhờ vậy kết quả không phụ thuộc thứ tự dòng trong file.

**Error decomposition — 2 tầng độc lập** (V1.0 có lỗi logic: nhóm thứ 5 "annotation_or_other" nằm sau 4 nhóm đã phủ hết 100% nên không bao giờ có case nào — sửa thành 2 tầng tách biệt).

**Tầng 1 — bucket tự động, loại trừ lẫn nhau, phủ đúng 100% `N_eval` query, gán theo thứ tự** (dùng retrieval pool / reader input đã định nghĩa ở 1.5):
1. `execution_failure` — không có record, hoặc `status="error"` (8.5).
2. `retrieval_miss` — gold không có trong **retrieval pool** = top-`retrieval_k` của retriever.
3. `selection_failure` (D3, thay `reranking_failure` của V1.0) — gold có trong retrieval pool nhưng không có trong **reader input** = `reader_k` passage được chọn: do reranker loại (bật reranker) HOẶC do cắt thẳng top-`reader_k` theo rank retriever (tắt reranker, tức gold ở hạng (`reader_k`, `retrieval_k`]).
4. `reader_failure` — gold có trong reader input nhưng câu trả lời cuối sai (EM=0, kể cả khi reader không trả đáp án).
5. `correct` — gold có trong reader input và EM=1.

Vì retrieval pool như nhau dù bật hay tắt reranker, với cùng retriever + corpus thì `retrieval_miss` giống nhau giữa 2 cấu hình — trừ khi 2 run có `execution_failure` khác nhau (bucket 1 được gán trước nên query lỗi ở run này có thể là `retrieval_miss` ở run kia). Chênh lệch `selection_failure` giữa "bật" và "tắt" chỉ phản ánh khả năng **giữ gold** trong `reader_k` passage, không phải toàn bộ lợi ích của reranker: đánh giá reranker vẫn phải so EM/F1 và latency.

Với MỌI query vẫn lưu riêng `final_correct` (EM=1 hay không), `gold_in_retrieval_pool`, `gold_in_reader_input` (cả 2 là `None` với `execution_failure`) và `first_gold_rank` — hệ thống có thể mất gold nhưng vẫn trả lời đúng từ passage khác (vd đáp án là thực thể phổ biến), nên `final_correct` KHÔNG được suy từ bucket và ngược lại.

**Tầng 2 — cờ đánh giá thủ công, cắt ngang tầng 1 (không phải bucket thứ 6)**: `manual_review_suggested` (bool) = true khi F1>0 nhưng EM=0 (khả năng lệch do chuẩn hóa/annotation chứ không phải lỗi hệ thống thật). Chọn tối đa 100 case có cờ (random seed 42 trên danh sách `query_id` đã sort) để review tay, kết luận ghi thành ghi chú định tính trong báo cáo, KHÔNG dùng để đổi % của tầng 1.

"Retrieval là bottleneck" là **giả thuyết** cần bảng tầng 1 xác nhận bằng số liệu, không viết như kết luận mặc định.

**Kiểm tra nhất quán trước khi tính** (vi phạm → dừng, không ghi output):
- Mọi record đạt schema 1.2/1.3/1.4/1.5; không trùng `query_id`; mọi `query_id` của predictions có trong gold.
- Mọi gold `passage_id` (suy ra từ `gold_evidences`, hoặc `positive_passage_ids` với gold GermanDPR) và mọi `passage_id` trong kết quả retrieval/reranking tồn tại trong tập ID của tier đang đánh giá theo `corpus_manifest.json`, kể cả kết quả giữ lại trong record lỗi. Không áp dụng yêu cầu này cho `hard_negative_groups`.
- Mọi record của 1 file predictions có cùng `run_id`/`config_id`/`corpus_id`/`retrieval_k`/`reader_k` và khớp entry của `run_id` trong `run_manifest.json`; `corpus_id` khớp tier đang đánh giá; ở end-to-end `1 ≤ reader_k ≤ retrieval_k`.
- Đọc mode, dataset/split, retriever và bật/tắt reranker từ manifest; các field này phải khớp config và fingerprint SHA-256 của file gold/corpus thực tế (8.7), không suy từ tên file hay từ việc `reranked` có giá trị. Các danh sách kết quả có rank liên tục từ 1, score hữu hạn không tăng, cùng query_id và không trùng passage_id.
- Với `status="ok"` ở end-to-end: nếu bật reranker, `reranked` phải là list (được rỗng khi pool rỗng), là subset của retrieval pool và có độ dài `min(reader_k, len(retrieval pool))`; nếu tắt reranker, `reranked` bắt buộc `None`. Chỉ kiểm tra `predicted_passage_id` thuộc reader input khi field khác `None`; 5 field đáp án cùng `None` vẫn hợp lệ.
- Ở `reader_only`: retriever phải là oracle, reranker tắt, `retrieved=[]`, `reranked=None`; chỉ kiểm tra `predicted_passage_id` thuộc gold khi field khác `None`. Gold rỗng hoặc reader không tìm được span cho phép 5 field `None`.

### 3.4 Cấu hình
| Tham số | Giá trị mặc định |
|---|---|
| Recall@k, k | 1, 5, 10, 20 |
| MRR cutoff | 10 |
| ngưỡng "correct" trong error decomposition | EM=1 |
| `retrieval_k`, `reader_k` | đọc từ `run_manifest.json`, đối chiếu với từng `PredictionRecord` (P1) — không phải tham số của Evaluation |
| manual review sample | tối đa 100 query, seed 42 |

### 3.5 Trường hợp biên
- Query thiếu hẳn trong file predictions HOẶC có dòng nhưng `status="error"` → bucket `execution_failure`, EM=F1=0, vẫn nằm trong `N_eval`.
- File predictions rỗng → vẫn tính được: mọi query là `execution_failure`, cấu hình lấy từ `run_manifest.json`, `n_latency=0`.
- Gold không có reference answer (không xảy ra với GermanQuAD chuẩn) → theo 1.2 thì `gold_evidences` cũng rỗng → nằm ngoài `N_eval`, đếm trong `n_excluded_empty_gold`; không chặn cả file.
- Prediction cho query có `gold_evidences=[]` → hợp lệ nhưng không tính vào metric nào.
- Vi phạm bất kỳ kiểm tra nhất quán nào ở 3.3 (gold `passage_id` không có trong corpus, sai `corpus_id`, trộn run hoặc lệch `run_manifest.json`, `reader_k > retrieval_k`, `query_id` lạ hoặc trùng, `predicted_passage_id` ngoài reader input, `reranked` không hợp lệ, `predicted_answer_text=""`) → dừng với thông báo lỗi liệt kê toàn bộ vi phạm, không âm thầm tính sai.

### 3.6 Kiểm thử
- Unit: tính tay 3-5 ví dụ EM/F1 (vd "Einstein" vs "Albert Einstein" → F1≈0,667) và Recall@k/MRR (gold hạng 2 → Recall@1=0, Recall@5=1, RR@10=0,5), assert code ra đúng số; query 2 gold chunk → Recall@k theo Hit@k, GoldCoverage@k theo tỉ lệ, ở cả hàm lẻ lẫn tầng tổng hợp.
- Unit: tắt reranker, gold ở hạng 8 với `reader_k`=5, `retrieval_k`=20 → `selection_failure` (không phải `retrieval_miss`); bật reranker, reranker loại gold hạng 8 → `selection_failure`; gold hạng 25 → `retrieval_miss` ở cả 2 cấu hình.
- Unit: mất gold nhưng trả lời đúng → bucket theo giai đoạn mất gold, `final_correct=true`.
- Unit: reranker tắt và 5 field đáp án `None` vẫn hợp lệ; oracle không có đáp án vẫn hợp lệ; reranker bật với pool rỗng trả `reranked=[]`; passage ID không có trong corpus bị từ chối ở cả retrieval-only và predictions (kể cả record lỗi).
- Unit: file predictions rỗng cùng manifest hợp lệ → mọi query đủ gold là execution_failure; manifest sai mode/dataset/split hoặc fingerprint gold/corpus không khớp bị từ chối. Các record có cùng tier nhưng thuộc corpus build khác không được đánh giá chung.
- Smoke: 20 prediction giả lập đủ cả 5 bucket tầng 1 (kể cả `execution_failure` do thiếu record và do `status="error"`), tổng = `N_eval`; vài case có F1>0/EM=0 phải được gắn `manual_review_suggested=true`.
- Nghiệm thu: cùng file predictions + cùng config luôn ra cùng kết quả (deterministic, không phụ thuộc thứ tự dòng trong file).

### 3.7 Artifact
`metrics_{run_id}.json`: mọi số liệu ở 3.3 + `counts` (`N_eval`, `n_excluded_empty_gold`, `n_latency` theo từng giai đoạn, số thiếu prediction, số `status="error"`) + `per_query` (bucket, `gold_in_retrieval_pool`, `gold_in_reader_input`, `first_gold_rank`, EM, F1, `final_correct`, `manual_review_suggested`) để phân tích robustness theo độ dài/loại câu hỏi mà không chạy lại. `predictions_*.jsonl` giữ nguyên làm nguồn tính lại không cần chạy model.

### 3.8 Tài nguyên
CPU-only.

## 4. Module: BM25

### 4.1 Mục tiêu & phạm vi
V1. Nhận corpus đã build (không tự dedup), trả `RetrievalResult` top-N/query.

### 4.2 Input/Output
Input: `corpus_{base,mid,large}.jsonl`, `qa_germanquad_{split}.jsonl` (`dev` trong lúc phát triển, `test` chỉ cho báo cáo cuối). Output: `retrieval_bm25_{tier}_{split}.jsonl` (top-50/query).

### 4.3 Quy tắc xử lý
Dùng `bm25s` (thống nhất mọi tầng corpus, xem 3.3). Tokenize trên `normalized_text` (đã chuẩn hóa ở module Data): whitespace split + loại token toàn ký tự không phải chữ/số. Query qua đúng normalize pipeline trước khi tokenize.

### 4.4 Cấu hình
top-N trả về = 50 (≥ `retrieval_k`=20 và ≥ k lớn nhất của Recall@k).

### 4.5 Trường hợp biên
- Query rỗng sau normalize → trả danh sách rỗng theo quy ước, không raise lỗi, log warning.
- Toàn bộ query token OOV → điểm 0 cho mọi passage; vẫn trả top-N theo thứ tự `passage_id` ổn định (deterministic, không random).
- Trùng `passage_id` trong kết quả → không thể xảy ra do corpus đã dedup; nếu xảy ra là bug, unit test phải bắt.

### 4.6 Kiểm thử
- Unit: query trùng gần nguyên văn 1 passage → passage đó rank 1.
- Smoke: 50 query mẫu, không lỗi, rank từ 1, không trùng `passage_id` trong cùng 1 kết quả.
- Nghiệm thu: kết quả đúng thứ tự giảm dần score, không trùng ID, query rỗng xử lý đúng quy ước.

### 4.7 Artifact
`retrieval_bm25_{tier}_{split}.jsonl` + metadata (`bm25s` version, `corpus_manifest` tương ứng).

### 4.8 Tài nguyên
CPU-only. Index 150K passage (tier lớn nhất): vài trăm MB RAM, build vài chục giây — chạy local hoàn toàn được.

## 5. Module: Dense Retriever

### 5.1 Mục tiêu & phạm vi
V1. Fine-tune dual-encoder (question encoder + passage encoder độc lập) theo phương pháp DPR trên GermanDPR train (sau loại leakage). Sau train: encode corpus → FAISS index → trả `RetrievalResult`.

### 5.2 Input/Output
- Train input: `retrieval_germandpr_train.jsonl` (`RetrievalTrainingRecord`, xem 1.3: `positive_texts` là chunk — dùng trực tiếp làm input passage_encoder; `hard_negative_groups[].texts` cũng là chunk, nhưng phải qua bước chọn ≤1 chunk/group/epoch ở 5.3 trước khi vào batch).
- Train output: checkpoint 2 tower (`question_encoder/`, `passage_encoder/`).
- Inference input: `corpus_{tier}.jsonl` + `qa_germanquad_{split}.jsonl`.
- Inference output: `retrieval_dense_{tier}_{split}.jsonl` + `faiss_index_{tier}.bin` + `faiss_id_map_{tier}.json` (mapping FAISS int index → `passage_id`, bắt buộc lưu kèm, không suy ra ngầm).

### 5.3 Quy tắc xử lý
- Kiến trúc: 2 tháp độc lập, khởi tạo từ `deepset/gbert-base`, **CLS pooling** (vector token `[CLS]` sau lớp cuối, đúng phương pháp DPR gốc — không mean-pooling).
- **Không normalize embedding** — giữ dot-product thô (đúng thiết kế DPR gốc, không cosine).
- Similarity = dot(question_vec, passage_vec).
- max_length: question_encoder=64 token, passage_encoder=300 token (điều chỉnh theo phân phối độ dài thật đo ở module Data nếu cần).
- **Seed ổn định bằng SHA-256, KHÔNG dùng Python `hash()`** — `hash()` built-in của Python áp dụng salt ngẫu nhiên mỗi process (`PYTHONHASHSEED` mặc định random) nên `hash(query_id)` cho kết quả KHÁC nhau giữa các lần chạy/process, phá vỡ tính reproducible. Dùng hàm chung cho MỌI lựa chọn ngẫu nhiên có seed trong module này (chọn positive, chọn hard-negative chunk/group):
  ```python
  def stable_seed(*parts: str) -> int:
      key = "|".join(parts)
      return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)

  rng = random.Random(base_seed + epoch_index + stable_seed(query_id))
  ```
  Cùng `(base_seed, epoch_index, query_id)` → luôn ra cùng giá trị seed, bất kể process/runtime nào (SHA-256 không phụ thuộc `PYTHONHASHSEED`).
- **Chính sách chọn 1 positive/query/epoch (single-target DPR NLL)**: `positive_texts` của 1 query có thể có >1 phần tử (xem 2.3.f — 12,5% record thật có nhiều positive chunk). Loss DPR gốc là **single-target NLL** (softmax trên 1 tập candidate, đúng 1 index là target) — KHÔNG hỗ trợ nhiều target cùng lúc trong 1 lần forward. Do đó: đầu mỗi epoch, với mỗi query có `len(positive_texts) > 1`: `idx = rng.randrange(len(positive_texts))` với `rng` khởi tạo như trên (`stable_seed(query_id)`). Lựa chọn có thể đổi giữa các epoch khác nhau (dùng hết thông tin của mọi positive qua nhiều epoch thay vì bỏ phí).
- **Chính sách chọn ≤1 chunk/hard-negative-group/query/epoch** — cùng lý do và cùng cơ chế seed: với mỗi group trong `hard_negative_groups` của query i, `chunk_idx = Random(base_seed + epoch_index + stable_seed(query_id, group.source_context_id)).randrange(len(group.texts))` → chọn ĐÚNG 1 chunk đại diện cho group đó epoch này (không phải cả `len(group.texts)` chunk). Nhờ vậy 1 hard-negative context dài (nhiều chunk) đóng góp CÙNG SỐ LƯỢNG passage vào batch như context ngắn (1 chunk) — không có trọng số lớn hơn chỉ vì dài hơn.
- **Batch construction**: micro-batch B câu hỏi; mỗi câu hỏi góp 1 positive (đã chọn) + tối đa `len(hard_negative_groups)` hard-negative (1 chunk đã chọn/group, KHÔNG hard-code luôn-3 group — data audit xác nhận 8/9.275 mẫu train có 1-2 hard-negative context, xem 5.5) → số passage/query trong batch ổn định = `1 + len(hard_negative_groups)`, không phụ thuộc số chunk mỗi context có.
- **In-batch negative & mask (tổng quát cho nhiều positive)**: gọi `gold_set(i)` = tập TOÀN BỘ `positive_passage_ids` của câu hỏi i (không chỉ phần tử được chọn làm target epoch này — các positive chưa được chọn epoch này vẫn là gold thật, không được dùng làm negative). Với câu hỏi i trong batch:
  1. **Trước khi chọn chunk/group** (bước trên): loại khỏi `hard_negative_groups` của i mọi group/chunk có `passage_id ∈ gold_set(i)` (hard-negative bị gán nhầm trùng gold — hiếm nhưng phải loại, xem 5.5) — nếu loại hết chunk của 1 group, group đó không đóng góp gì epoch này.
  2. Loại khỏi tập candidate-negative của i mọi passage (đã vào batch, từ positive-đã-chọn HOẶC hard-negative-chunk-đã-chọn của câu hỏi khác) có `passage_id ∈ gold_set(i)` — kể cả khi passage đó là positive-đã-chọn của câu hỏi khác j≠i trong batch (tránh false negative: passage này đúng là liên quan tới câu hỏi i, không được dùng làm negative dù nó đang đóng vai trò positive cho j).
  3. Mọi passage KHÁC (kể cả là positive-đã-chọn hoặc hard-negative-chunk-đã-chọn của câu hỏi j≠i, miễn không thuộc `gold_set(i)`) vẫn là in-batch negative hợp lệ cho i — đây là cơ chế in-batch negative chuẩn của DPR, không phải lỗi cần loại bỏ.
- **Gradient accumulation ≠ tăng in-batch negative**: in-batch negative chỉ tính trong phạm vi 1 micro-batch forward cùng lúc, không cross micro-batch. "Effective contrastive batch size" = micro-batch size thật, không phải batch size sau accumulation.
- Loss: NLL/contrastive kiểu DPR gốc (softmax similarity giữa câu hỏi và toàn bộ candidate trong batch sau chọn+mask, index của positive-đã-chọn epoch này là target).

### 5.4 Cấu hình
| Tham số | Giá trị dự kiến | Tune trên |
|---|---|---|
| micro-batch size | 16 (giảm nếu OOM trên T4) | theo VRAM, không chọn theo metric |
| learning rate | 2e-5 | dev (metric bên dưới) |
| epochs | 3 (checkpoint cuối mỗi epoch) | dev (metric bên dưới) |
| max_length (q/passage) | 64/300 | cố định, điều chỉnh nếu data-driven check khác |
| FAISS index | `IndexFlatIP` | cố định (corpus đủ nhỏ cho exact search) |
| `base_seed` (chọn positive/hard-negative chunk) | 42 | cố định, ghi vào `training_config.json` (xem 5.7) |

**Chọn checkpoint retriever (P2)**: metric = **Recall@20 trên danh sách `retrieved` (trước selection)**, định nghĩa 3.3, trên `qa_germanquad_dev` với tier base. Metric bằng nhau → chọn checkpoint sớm hơn. Checkpoint được chọn đóng băng khi đánh giá mid/large. `retrieval_germandpr_dev` chỉ dùng theo dõi loss khi train, không dùng chọn checkpoint.

Batch size là **giá trị dự kiến, chưa test trên VRAM thật** — OOM trên T4 thì giảm 8 hoặc 4 kèm gradient accumulation, effective batch size ghi rõ trong config log.

### 5.5 Trường hợp biên
- Số `hard_negative_groups` không phải luôn = 3: data audit xác nhận 8/9.275 mẫu train có 1-2 hard-negative context thay vì 3 (`germandpr_train_hard_negative_count_dist` trong `results/data_audit_report.json`). Code đọc `len(hard_negative_groups)` động, KHÔNG hard-code giả định luôn có đúng 3.
- Hard negative trùng BẤT KỲ gold positive nào của cùng câu hỏi (không chỉ trùng positive-đã-chọn epoch này — lỗi dữ liệu hiếm) → loại khỏi `hard_negative_groups` (group hoặc chunk cụ thể) TRƯỚC khi chọn chunk/epoch, log (xem 5.3 bước mask-trước-khi-chọn).
- Group sau khi loại hard-negative trùng gold còn 0 chunk → group đó không đóng góp negative nào epoch này (không lỗi, không cần fallback).
- Query có `len(positive_texts) > 1` (nhiều answer-bearing chunk, xem 2.3.f) → áp dụng chính sách chọn 1 positive/epoch ở 5.3, KHÔNG train đồng thời nhiều target trong 1 loss call.
- Record GermanDPR không map được chunk nào (0,20% thật, xem 2.3.f) đã bị loại hẳn ở bước data prep (module 2) — module 5 sẽ KHÔNG bao giờ thấy record kiểu này trong `retrieval_germandpr_train.jsonl`, không cần xử lý riêng.
- Câu hỏi/passage vượt max_length → truncate, log số lượng bị truncate (>5% tổng train thì cảnh báo tăng max_length).
- Corpus rỗng lúc build FAISS → raise lỗi rõ ràng (không để FAISS tạo index rỗng âm thầm).

### 5.6 Kiểm thử
- Unit: gradient chảy qua cả 2 tower sau 1 step backward (`.grad` không None ở cả hai).
- Unit: hàm mask negative — với batch giả lập có passage X là positive-đã-chọn của câu hỏi j nhưng KHÔNG thuộc `gold_set(i)`, X phải vẫn còn trong tập negative-candidate của i (không bị loại nhầm).
- Unit (false negative): với batch giả lập có passage X ∈ `gold_set(i)` NHƯNG X cũng là positive-đã-chọn của câu hỏi j≠i trong cùng batch, X phải bị loại khỏi tập negative-candidate của i (không được dùng làm negative cho i dù đang là target của j).
- Unit (hard negative trùng gold): với record có 1 hard-negative-group chứa chunk trùng `passage_id` với 1 gold positive của CHÍNH câu hỏi đó, chunk đó (và group nếu hết chunk) phải bị loại khỏi batch trước khi tính loss.
- Unit (nhiều positive, reproducible cùng process): với query có `positive_texts` độ dài 3, chạy `chọn positive` 2 lần cùng `(base_seed, epoch, query_id)` → phải ra đúng cùng 1 index; đổi `epoch_index` → được phép ra index khác.
- **Unit (reproducible XUYÊN process, dùng `subprocess`)**: chạy hàm chọn positive (và hàm chọn hard-negative chunk/group) trong 2 process Python riêng biệt (`subprocess.run([sys.executable, "-c", ...])`, mỗi process set `PYTHONHASHSEED` khác nhau hoặc để mặc định random) với CÙNG `(base_seed, epoch, query_id)` → phải ra ĐÚNG CÙNG 1 index ở cả 2 process — xác nhận SHA-256 seed không phụ thuộc `PYTHONHASHSEED`/salt của Python `hash()`.
- Unit (hard-negative grouping — 1 chunk/group): với 1 `hard_negative_groups` entry giả lập có `len(texts)=5` (context dài), chạy chọn chunk/epoch → phải trả về ĐÚNG 1 chunk, không phải cả 5.
- Unit (context dài không có trọng số lớn hơn): với 2 record giả lập — record A có 1 hard-negative context ngắn (1 chunk), record B có 1 hard-negative context dài (5 chunk) — sau bước chọn chunk/group, cả A và B đều đóng góp ĐÚNG 1 passage vào batch cho hard-negative đó (không phải B đóng góp 5).
- Smoke: train 1 epoch/50 mẫu, similarity(question, positive) tăng so với trước train.
- Save/load: encode lại 10 passage sau save+load, similarity chênh lệch <1e-5 so với trước save.
- Nghiệm thu: cả 9 điều trên pass; grep code xác nhận KHÔNG còn lời gọi `hash(` (built-in) nào dùng để tạo seed trong module này.

### 5.7 Artifact
`question_encoder/`, `passage_encoder/` (HF format), `faiss_index_{tier}.bin`, `faiss_id_map_{tier}.json`, `training_config.json` — BẮT BUỘC gồm `base_seed`, chính sách chọn positive/hard-negative chunk (tên hàm `stable_seed` + công thức seed dùng, xem 5.3) cùng mọi hyperparameter thật đã dùng — đủ để 1 người khác tái lập chính xác thứ tự chọn positive/hard-negative của từng epoch.

### 5.8 Tài nguyên
Train: Colab T4, fp16, checkpoint mỗi epoch lên Google Drive (lưu: weights + optimizer state + epoch/step counter + RNG state) để resume nếu bị ngắt phiên. Encode corpus (tối đa 150K ở tier large): chạy Colab T4 (nhanh) hoặc local RTX 3050 (chậm hơn, chỉ forward pass, vẫn khả thi). **Khi mở rộng tier** (base→mid→large), CHỈ encode embedding cho passage mới (đã có sẵn từ tier trước thì tái sử dụng, xem 2.3.d) — merge vào FAISS index cũ thay vì encode lại toàn bộ corpus.

## 6. Module: Reader

### 6.1 Mục tiêu & phạm vi
V1. Fine-tune extractive QA trên GermanQuAD train (sau loại leakage). Chọn đáp án cuối từ reader input (1.5): `reader_k` passage ở end-to-end, hoặc toàn bộ gold passage của query ở reader-only (oracle).

### 6.2 Input/Output
- Train input: `qa_germanquad_train.jsonl` — đọc `gold_evidences` (mỗi entry đã có `passage_id`/`answer_text`/`answer_start`/`answer_end` LOCAL đúng trong chunk đó, xem 1.2/2.3.c) làm cặp (passage_text, answer, offset) để train; KHÔNG dùng `normalized_text` của passage (dùng `text` gốc của passage, để offset khớp chính xác). 1 query có nhiều `gold_evidences` → sinh nhiều training example (1 example/evidence), không cần chọn "passage đại diện" như bản schema cũ.
- Train output: checkpoint `AutoModelForQuestionAnswering`.
- Khởi tạo: `deepset/gbert-base` với QA head mới; model và tokenizer ghim revision, seed khởi tạo head ghi vào `training_config.json`.
- Inference input: 1 hoặc nhiều `(query, passage_text)`.
- Inference output: bổ sung field vào `PredictionRecord`.

### 6.3 Quy tắc xử lý
- max_seq_length=384, doc_stride=128 (`return_overflowing_tokens=True`).
- Ràng buộc span hợp lệ: `start ≤ end`, cả hai trong vùng context, `end-start+1 ≤ max_answer_length` (=30 token).
- **Tập ứng viên và điểm thô**: xét TOÀN BỘ cặp token start/end hợp lệ theo các ràng buộc trên trong từng window, không cắt trước theo top-n start/end. Điểm thô = `start_logit[start] + end_logit[end]`. Map span về offset ký tự trong `text` gốc; chỉ giữ span có `0 ≤ char_start < char_end ≤ len(text)`.
- **Gộp overflow windows (cùng 1 passage)**: dedup theo `(passage_id, char_start, char_end)`. Một char-span xuất hiện trong nhiều window thì lấy MAX điểm thô của các occurrence làm điểm duy nhất; không cộng điểm hoặc giữ occurrence đầu. Sau dedup, toàn bộ span của passage là một tập ứng viên chung, sắp theo `(char_start, char_end)` để phép tổng không phụ thuộc thứ tự window.
- **Điểm span**: với điểm thô đã gộp `s_i`, tính `p_i = exp(s_i - max(s)) / sum_j exp(s_j - max(s))` trên TOÀN BỘ tập ứng viên đã dedup của chính passage đó; `reader_span_prob` là p của span được chọn. **Đây là scoring heuristic, không phải xác suất đã hiệu chuẩn để so sánh tuyệt đối giữa các passage** (passage có ít candidate hợp lệ có thể có softmax mass cao hơn). V1 cố định quy tắc này; so sánh với "raw logit sum cao nhất" là ablation riêng trên dev, không đổi quy tắc chính giữa chừng. NaN/Inf ở logits là lỗi inference (`status="error"`, `error_stage="read"`), không phải trường hợp không có span.
- **Chọn đáp án cuối khi đọc nhiều passage** (mọi passage của reader input): so `reader_span_prob` giữa các passage, chọn cao nhất. V1 KHÔNG dùng lại điểm reranker (λ=1 cố định — reranker đã hoàn thành vai trò chọn `reader_k` passage); thử λ·reader + (1-λ)·reranker (nếu làm) là ablation báo cáo riêng, không phải hyperparameter được tune.
- **Đồng điểm**: ở cả chọn span trong passage và chọn đáp án giữa các passage, nếu xác suất bằng nhau thì chọn khóa `(passage_id, char_start, char_end)` nhỏ nhất (ID so theo chuỗi, offset theo số nguyên); không phụ thuộc thứ tự reader input/window. Text đáp án = `passage.text[char_start:char_end]`.
- Lưu đủ: answer text, passage_id, offset (trong `text` gốc) — phục vụ truy vết lỗi ở Evaluation.

### 6.4 Cấu hình
| Tham số | Giá trị dự kiến | Tune trên |
|---|---|---|
| batch size | 8 (giảm nếu OOM) | theo VRAM, không chọn theo metric |
| learning rate | 3e-5 | dev (metric bên dưới) |
| epochs | 3 (checkpoint cuối mỗi epoch) | dev (metric bên dưới) |
| max_answer_length | 30 token | cố định |
| doc_stride | 128 | cố định |

**Chọn checkpoint reader** (cùng nguyên tắc P2): metric = F1 ở chế độ `reader_only` (reader đọc gold passage) trên `qa_germanquad_dev`; bằng nhau → checkpoint sớm hơn. Reader-only không dùng retrieval nên không phụ thuộc tier; checkpoint đóng băng cho mọi tier.

### 6.5 Trường hợp biên
- Đáp án vắt ranh giới overflow window → chỉ window chứa TRỌN đáp án được gán label lúc train (chuẩn xử lý HF), window kia dùng CLS index làm target (convention SQuAD).
- Context/câu hỏi rỗng → raise lỗi ở input validation, không chạy model trên input rác.
- Không passage nào có span hợp lệ → `status="ok"`, cả 5 field đáp án = `None` (D4 — không dùng `""`), log riêng; Evaluation đếm vào `no_prediction` và xếp bucket theo 3.3 (thường là `reader_failure` nếu gold có trong reader input).

### 6.6 Kiểm thử
- Unit: input context+question+answer_start đã biết, sau 1 bước train nhỏ, span dự đoán nằm trong `[0, len(context))`.
- Smoke: 5 mẫu context dài (>384 token) có đáp án ở window thứ 2, vẫn tìm đúng.
- Unit hậu xử lý với logits giả lập: cùng char-span ở 2 window lấy đúng max điểm; softmax trên toàn bộ tập span sau dedup khớp tính tay, tổng bằng 1 trong sai số số thực; đảo thứ tự window/passage không đổi đáp án; đồng điểm chọn đúng khóa quy định; logits NaN/Inf tạo lỗi read.
- Nghiệm thu: span luôn hợp lệ; xử lý đúng mẫu context-dài/overflow trong bộ test kiểm thử.

### 6.7 Artifact
Checkpoint HF format, `training_config.json`.

### 6.8 Tài nguyên
Train: Colab T4, fp16, batch 8 (+gradient accumulation nếu cần). Inference: chạy được cả local RTX 3050 4GB (chỉ forward pass).

## 7. Module: Reranker (Cross-Encoder)

### 7.1 Mục tiêu & phạm vi
V1. Nhận retrieval pool (top-`retrieval_k` của retriever), KHÔNG tự truy vấn corpus, trả `reader_k` passage đã sắp xếp lại (subset của input, không thêm passage mới).

### 7.2 Input/Output
Input: `retrieval_{bm25,dense}_{tier}_{split}.jsonl` (lấy top-`retrieval_k`=20). Output: `reranked_{retriever}_{tier}_{split}.jsonl` (`reader_k`=5/query), schema `RetrievalResult` với `retriever_name="cross_encoder"`.

### 7.3 Quy tắc xử lý
- Train pair: cùng nguồn dữ liệu với dense retriever (`retrieval_germandpr_train.jsonl`, schema 1.3): positive=1 (từ `positive_texts`, áp dụng cùng chính sách chọn 1 positive/epoch nếu >1, xem 5.3), hard-negative=0 (từ `hard_negative_groups` — áp dụng CÙNG chính sách chọn ≤1 chunk/group/epoch như 5.3, lý do tương tự: tránh 1 hard-negative context dài đóng góp nhiều cặp (question, passage) hơn context ngắn trong cùng epoch). Số `hard_negative_groups`/query thay đổi, KHÔNG hard-code 3 — nhất quán mask false-negative với 5.3 (loại group/chunk trùng gold positive TRƯỚC khi chọn).
- Checkpoint khởi tạo: `deepset/gbert-base` (nhất quán họ model). Alternative: cross-encoder đa ngữ pretrained trên MS MARCO — so bằng đúng metric chọn checkpoint reranker ở 7.4; cao hơn thì đổi default và log lý do, bằng nhau thì giữ `gbert-base`.
- Input: `[CLS] question [SEP] passage [SEP]`, max_length=384 (truncate passage trước, giữ nguyên câu hỏi).
- Loss: binary cross-entropy.
- Hai tham số **tách riêng, không dùng chung tên "k"**: `retrieval_k`=20 (số candidate retriever/reranker xem xét) và `reader_k`=5 (số passage CUỐI CÙNG đưa vào reader — áp dụng như nhau dù có bật reranker hay không, xem 8.3). Reranker nhận vào `retrieval_k`=20 candidate, xuất ra đúng `reader_k`=5.
- Đồng điểm: giữ thứ tự rank gốc từ retriever làm tie-breaker (stable sort).
- **Kiểm soát khi đo lợi ích reranker**: so "có reranker" (`reader_k`=5 sau rerank từ `retrieval_k`=20 candidate) với baseline "không reranker" phải dùng CÙNG `reader_k`=5 passage vào reader — baseline = top-`reader_k` trực tiếp từ retriever theo rank gốc (KHÔNG phải toàn bộ `retrieval_k`=20), để tách hiệu ứng reranking khỏi hiệu ứng "đọc nhiều context hơn". Trong error decomposition (3.3), chênh lệch `selection_failure` giữa 2 cấu hình chỉ cho biết reranker giữ gold tốt hơn bao nhiêu; lợi ích cuối cùng phải đọc từ EM/F1 end-to-end, cùng chi phí latency của giai đoạn rerank. Chi tiết áp dụng thống nhất ở Integration, xem 8.3.

### 7.4 Cấu hình
| Tham số | Giá trị dự kiến | Tune trên |
|---|---|---|
| retrieval_k (input) | 20 | cố định |
| reader_k (output) | 5 | cố định ở V1 — không tune (xem dưới) |
| batch size train | 16 | theo VRAM, không chọn theo metric |
| learning rate | 2e-5 | dev (metric bên dưới) |
| epochs | 2-3 (checkpoint cuối mỗi epoch) | dev (metric bên dưới) |

**Chọn checkpoint reranker (P2)**: metric = **Recall@5 (≡ Hit@5, định nghĩa 3.3) trên danh sách sau rerank**, với candidate cố định = top-20 của checkpoint retriever đã chọn (5.4) trên `qa_germanquad_dev`, tier base. Khi so các checkpoint reranker, checkpoint retriever và corpus giữ nguyên. Metric bằng nhau → chọn checkpoint sớm hơn. Checkpoint được chọn đóng băng khi đánh giá mid/large.

**Không tune `reader_k` bằng Hit@`reader_k`**: trên cùng một danh sách xếp hạng, tăng số passage giữ lại không bao giờ làm Hit giảm, nên metric này luôn chọn `reader_k` lớn nhất. V1 cố định `reader_k`=5. Nếu sau này quét `reader_k` (vd 3/5/10), đó là ablation báo cáo riêng bằng end-to-end EM/F1 cùng latency, không phải hyperparameter được chọn trên dev.

### 7.5 Trường hợp biên
- Retrieval pool < `retrieval_k` (retriever trả ít hơn) → chạy trên số lượng có sẵn, không pad giả; xuất `min(reader_k, len(pool))` passage.
- Toàn bộ candidate điểm thấp → vẫn trả đủ `reader_k` theo rank tương đối (quyết định "có đáp án hay không" là việc của reader, không phải reranker).

### 7.6 Kiểm thử
- Unit: reranker chỉ xuất `passage_id` đã có trong retrieval pool.
- Smoke: 20 query, `len(reranked) == min(reader_k, len(pool))` với mọi query.
- Nghiệm thu: chỉ sắp xếp/chọn từ retrieval pool, số kết quả = `min(reader_k, len(pool))`.

### 7.7 Artifact
Checkpoint HF format, `reranked_{retriever}_{tier}_{split}.jsonl`.

### 7.8 Tài nguyên
Train: Colab T4. Inference rerank 20 candidate/query: nhẹ, chạy được cả local.

## 8. Module: Integration (Pipeline end-to-end)

### 8.1 Mục tiêu & phạm vi
V1. Ghép Retriever → [Reranker] → Reader thành 1 pipeline chạy theo query, sinh `PredictionRecord` hoàn chỉnh, đo latency từng bước (Integration là module DUY NHẤT đo latency, xem mục 0).

### 8.2 Input/Output
Input: `qa_germanquad_{split}.jsonl` (`dev` trong lúc phát triển, `test` cho báo cáo cuối), checkpoint các module đã train, corpus+index 1 trong 3 tầng. Output: `predictions_{run_id}.jsonl`.

### 8.3 Quy tắc xử lý
Chạy tuần tự: retriever (BM25/Dense/Hybrid) trả tối đa top-N (N cấu hình ≥ `retrieval_k`) → retrieval pool = top-`retrieval_k`=20 → **reader đọc `min(reader_k, len(pool))` passage với `reader_k`=5, dù bật reranker hay không**; khi đủ candidate thì đọc đúng 5, thiếu thì không pad (khi tắt reranker mà reader đọc hết 20 thì baseline không còn so sánh được như 7.3 yêu cầu):
  - Bật reranker: reranker nhận retrieval pool → xuất tối đa `reader_k`=5 đã sắp xếp lại → reader đọc các passage đó.
  - Tắt reranker: lấy trực tiếp tối đa top-`reader_k`=5 theo rank gốc của retriever (KHÔNG phải cả 20) → reader đọc các passage đó.
  - Reader-only (oracle, `retriever="oracle"`, `config_id` `oracle_{tier}_rerank-off`): không retrieval (`retrieved=[]`, `reranked=None`, `latency_ms.retrieval=0`, `latency_ms.rerank=None`), reader đọc toàn bộ `gold_passage_ids` của query; `retrieval_k`/`reader_k` vẫn ghi theo config nhưng không được dùng; đánh giá bằng chế độ `reader_only` (3.1). Nếu có đáp án thì `predicted_passage_id` thuộc gold; không có span hoặc gold rỗng thì 5 field đáp án `None`.
- Chọn span theo 6.3 → ghi `PredictionRecord` kèm `retrieval_k`, `reader_k` (P1), `latency_ms` từng giai đoạn, `status`/`error_stage` (xem 8.5), `corpus_id`, `config_id` và `run_id`.
- **Đo latency (D5)** — điều kiện thống nhất cho mọi run:
  - Thiết bị: số chính thức đo trên Colab T4; số đo trên máy khác chỉ tham khảo, thiết bị ghi trong `run_manifest.json`.
  - Batch = 1 query; mỗi query đo đúng 1 lần.
  - Warm-up: trước lượt đo chính thức, chạy 1 **lượt chạy thử** gồm 10 query đầu (theo `query_id` đã sort) của `qa_germanquad_dev.jsonl` qua toàn bộ pipeline và bỏ kết quả của lượt đó. Chỉ bỏ lượt chạy thử, KHÔNG loại vĩnh viễn 10 query này: nếu chúng thuộc tập đang chạy (vd run trên dev), chúng vẫn được chạy lại trong lượt chính thức và có đúng 1 record đo như mọi query khác. Nhờ vậy mọi record trong predictions đều là số đo của lượt chính thức và Evaluation không cần biết warm-up.
  - Mốc thời gian: retrieval = từ nhận câu hỏi tới có kết quả top-N (tính cả query encoding, KHÔNG tính build/load index); rerank = từ nhận pool tới có passage được chọn; read = từ nhận reader input tới có đáp án cuối; total = từ nhận câu hỏi tới có đáp án cuối. Giai đoạn chạy GPU gọi `torch.cuda.synchronize()` ngay trước cả mốc bắt đầu và mốc kết thúc.
  - Dùng cùng implementation BM25 (`bm25s`) cho mọi tầng corpus — đổi thư viện giữa các tầng là đổi implementation, không phải chỉ đổi corpus size.

### 8.4 Cấu hình
`config_id` composite, người-đọc-được: `{retriever}_{tier}_rerank-{on|off}` với retriever ∈ bm25|dense|hybrid|oracle (vd `dense_mid_rerank-on`). `run_id` = `{config_id}__{dataset}-{split}__ckpt-{checkpoint_bundle_hash[:12]}__seed{seed}__{run_uuid}` (hash rút gọn chỉ để tên file ngắn hơn). `run_uuid` là UUID4 mới cho MỖI lần chạy mới, lưu trước khi inference; dataset/split lấy từ cấu hình tường minh, không parse query_id. `checkpoint_bundle_hash` là SHA-256 của JSON canonical (`sort_keys=True`, `ensure_ascii=False`, `separators=(",", ":")`, mã hóa UTF-8) ánh xạ tên từng thành phần sang fingerprint checkpoint của nó; thành phần không có checkpoint ghi null. Full hash và UUID lưu trong manifest. Hai run dev/test hoặc hai lần chạy mới cùng config/checkpoint/seed không ghi đè nhau.

Mỗi run có 1 file `predictions_{run_id}.jsonl` riêng. Các artifact retrieval/reranked/metrics có tên logic không chứa UUID phải nằm trong thư mục riêng theo `run_id`. Nếu tiếp tục một run bị ngắt, giữ run_id và manifest cũ, kiểm tra fingerprint không đổi và chỉ ghi query còn thiếu; không sinh hai record cho một query. Evaluation không thêm timestamp ngẫu nhiên vào metrics để giữ tính tất định khi tính lại cùng run.

| Tham số | Giá trị |
|---|---|
| `retrieval_k` / `reader_k` | 20 / 5 (7.4); end-to-end bắt buộc `1 ≤ reader_k ≤ retrieval_k`; ghi vào `run_manifest.json` và mọi `PredictionRecord` |
| batch khi chạy pipeline | 1 query |
| warm-up | 1 lượt chạy thử 10 query dev (theo `query_id` đã sort), bỏ kết quả lượt đó; các query này vẫn được đo ở lượt chính thức nếu thuộc tập đang chạy |
| số lần đo mỗi query | 1 |

### 8.5 Trường hợp biên
Module lỗi giữa chừng (vd reader OOM ở 1 query) → log lỗi, ghi `PredictionRecord` với `status="error"`, `error_stage` = giai đoạn lỗi ("retrieval"/"rerank"/"read"). Cả 5 field đáp án là `None`; giữ định danh/cấu hình, các kết quả retrieval/rerank và latency đã hoàn thành theo 1.5 (không xóa mọi field còn lại). Không crash toàn run. Evaluation đọc `status="error"` và xếp query đó vào bucket `execution_failure`.

### 8.6 Kiểm thử
- Smoke: full pipeline trên 20 query với mọi tổ hợp config (BM25/Dense × rerank on/off × 1 tier) + 1 run oracle, không crash, output đạt schema 1.5 và qua được kiểm tra nhất quán của Evaluation (3.3).
- Unit: kết quả lượt warm-up không được ghi; khi chạy trên dev, mỗi query trong 10 query warm-up vẫn có đúng 1 record (từ lượt chính thức); mọi record và manifest có cùng `retrieval_k`/`reader_k`; config với `reader_k > retrieval_k` bị từ chối trước khi chạy; tắt reranker và có đáp án thì `predicted_passage_id` thuộc `retrieved[:reader_k]`, không có đáp án thì 5 field `None` vẫn hợp lệ.
- Unit: hai run mới cùng config/checkpoint/seed và hai run dev/test có ID/path khác nhau; manifest vẫn đầy đủ khi predictions rỗng; resume giữ run_id, từ chối fingerprint đổi và không tạo query_id trùng.
- Nghiệm thu: chạy hết test set không lỗi với ít nhất 1 config đầy đủ (dense + rerank + corpus_base) trước khi mở rộng tier lớn hơn.

### 8.7 Artifact
`predictions_{run_id}.jsonl`, `run_manifest.json`. Manifest là object ánh xạ `run_id` → cấu hình, ghi entry TRƯỚC khi chạy query đầu tiên, kể cả khi predictions cuối cùng rỗng. Mỗi entry bắt buộc gồm:
- `run_id`, `run_uuid`, `config_id`, `mode` (`end_to_end` / `reader_only` / `retrieval_only`), `dataset` (`germanquad` / `germandpr`), `split`, `retriever`, `reranker_enabled`, `corpus_id`, `retrieval_k`, `reader_k`, `seed`. GermanDPR chỉ dùng `retrieval_only`; oracle chỉ dùng `reader_only`, reranker tắt.
- `gold_path`, `gold_sha256`, `corpus_path`, `corpus_sha256`, `corpus_manifest_path`, `corpus_manifest_sha256`, `predictions_path` hoặc `retrieval_results_path`. Fingerprint là SHA-256 đầy đủ của byte nội dung file artifact, không chỉ tên hoặc tier; đường dẫn tương đối resolve theo thư mục chứa manifest. Đổi thứ tự dòng tạo fingerprint file mới: fixture kiểm tra tính tất định phải cập nhật fingerprint tương ứng, còn các giá trị metric không được đổi.
- `checkpoints`: ánh xạ tên thành phần (`question_encoder`, `passage_encoder`, `reader`, `reranker`) → `path`, model/tokenizer revision và fingerprint; thành phần không dùng ghi null. Fingerprint thư mục checkpoint là SHA-256 của JSON canonical ánh xạ đường dẫn file tương đối đã sort sang SHA-256 nội dung từng file; ghi thêm `checkpoint_bundle_hash` theo 8.4.
- Điều kiện đo: thiết bị/GPU/CPU, phiên bản thư viện, batch query=1, danh sách 10 warm-up query_id và fingerprint file dev dùng warm-up, số lượt đo chính thức/query=1. Run retrieval_only không đo latency thì các field điều kiện đo/warm-up ghi null.

Evaluation nhận `run_id` tường minh để chọn entry, đọc cấu hình từ manifest và đối chiếu artifact (3.2–3.3); không suy cấu hình từ dòng prediction đầu tiên. `corpus_manifest.json` vẫn là nguồn tập ID từng tier; run manifest xác định chính xác phiên bản corpus/gold được dùng.

### 8.8 Tài nguyên
Model đã train sẵn (chỉ inference) → chạy được local RTX 3050 cho tier base/mid; tier large (150K) nên chạy Colab để encode/search nhanh hơn.

## 9. Bảng tiêu chí nghiệm thu tổng hợp

| Module | Điều kiện nghiệm thu |
|---|---|
| Schema/Data | Mọi `gold_evidences` khác rỗng đều trỏ tới passage tồn tại với offset đúng trong CHÍNH chunk đó; mỗi `question_key` chỉ thuộc đúng 1 split (train/dev/test) hoặc bị loại vì leakage; file test của mỗi dataset chỉ chứa đúng dòng test gốc của dataset đó; `query_id` mã hoá split gốc; QARecord gold rỗng vẫn được ghi; GermanDPR dùng `RetrievalTrainingRecord` (`positive_passage_ids`/`hard_negative_groups[].passage_ids` LUÔN khác None — record không map được đã bị loại ở nguồn; `hard_negative_groups[].passage_ids` không bắt buộc có trong corpus), không lẫn vào `QARecord` |
| Corpus | passage_id duy nhất; chỉ gồm 3 nguồn được phép, không passage nào vào corpus chỉ vì là hard negative; base (số thật) ⊆ mid (50.000) ⊆ large (150.000); phần distractor là tiền tố cùng chuỗi đã lọc P4/dedup; title loại trừ lấy trước dedup base; chuỗi và occurrence đại diện không phụ thuộc thứ tự dump, lưu đủ text để dựng tier không đọc lại dump; mọi gold ID nằm trong base, offset khớp text; chunking không phụ thuộc vị trí đáp án; coverage ≥98% trên từng file; GermanDPR positive là chunk chứa occurrence thật và nhóm hard negative đúng theo context gốc; tỷ lệ mapping đo lại theo split, số audit 2.3.f chỉ tham chiếu |
| BM25 | Kết quả đúng thứ tự giảm dần score; không trùng passage_id/query; query rỗng xử lý đúng quy ước, không crash |
| Dense | Gradient chảy qua cả 2 tower; seed dùng SHA-256 (`stable_seed`), KHÔNG dùng Python `hash()` ở bất kỳ đâu trong module; chọn positive VÀ chọn hard-negative-chunk/group đều reproducible xuyên process (test bằng `subprocess`); mỗi hard-negative-group đóng góp tối đa 1 chunk/epoch (context dài không có trọng số lớn hơn); mask loại đúng TOÀN BỘ `gold_set(i)` của câu hỏi đang xét (không chỉ positive-đã-chọn epoch này), không loại nhầm positive-đã-chọn của câu hỏi khác trừ khi nó cũng thuộc `gold_set(i)`; 0 hard-negative trùng gold positive lọt vào batch; similarity(positive) tăng sau vài step train batch nhỏ; save/load giữ kết quả trong sai số 1e-5; checkpoint chọn theo Recall@20 trước selection trên dev, tier base, bằng nhau → sớm hơn, ghi trong `training_config.json` |
| Reader | Khởi tạo gbert-base với QA head, ghim revision/seed; span hợp lệ, xét toàn bộ ứng viên; điểm thô start_logit + end_logit, char-span trùng lấy max rồi softmax trên tập dedup của passage; đảo window/input không đổi kết quả, đồng điểm theo khóa 6.3; không có span → 5 field None; checkpoint chọn theo F1 reader-only trên dev, bằng nhau → sớm hơn |
| Reranker | Chỉ sắp xếp/chọn từ retrieval pool; số kết quả = `min(reader_k, len(pool))`; checkpoint chọn theo Recall@5 sau rerank trên top-20 cố định của retriever đã đóng băng (dev, tier base), bằng nhau → sớm hơn |
| Evaluation | Metrics khớp ví dụ tính tay; Recall@k = Hit@k, GoldCoverage@k là metric phụ; metric chất lượng dùng N_eval, latency dùng n_latency; 5 bucket cộng đúng N_eval; gold ở hạng (reader_k, retrieval_k] khi tắt reranker là selection_failure; gold hoặc kết quả retrieval/rerank ngoài corpus, sai mode/split/fingerprint/manifest thì dừng; no-answer None và reranked=None khi tắt hợp lệ; predictions rỗng vẫn tính được; cùng đầu vào hợp lệ luôn cho cùng giá trị metric, không phụ thuộc thứ tự dòng |
| Integration | Chạy hết test không crash với ≥1 run đầy đủ; end-to-end reader đọc min(reader_k, số candidate sẵn có), dù bật/tắt reranker; oracle đọc toàn bộ gold; 1 ≤ reader_k ≤ retrieval_k, record khớp manifest; run_id phân biệt split/lần chạy, manifest được ghi trước inference và chứa fingerprint artifact/checkpoint; latency theo 8.3, warm-up không bỏ query khỏi lượt chính thức, GPU sync trước/sau |
