# RAG — Sinh câu trả lời có trích dẫn (V2) — Thiết kế

- Ngày: 2026-09-12. Trạng thái: **DESIGN — chờ người dùng duyệt file này**, chưa có
  implementation plan, chưa có code.
- Tên đầy đủ: Mở rộng hệ thống hỏi đáp mở tiếng Đức bằng sinh câu trả lời (RAG) có trích
  dẫn nguồn kiểm chứng được, cộng web app local để kiểm thử trực quan.
- **V1 baseline:** hệ thống hỏi đáp trích xuất hiện tại — Retriever (BM25/Dense) →
  Selection (reranker/top-5) → Reader trích xuất (gbert-base), theo
  [module spec V1.1 — frozen 2026-09-11](../../../docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md).
  **Tài liệu này không sửa V1.1.**
- **V2 extension (nội dung file này):** thêm một bộ trả lời sinh câu (LLM) có trích dẫn,
  đặt cạnh Reader trích xuất qua một interface chung, cộng RQ4 và một web app local.
- Không có số liệu thực nghiệm nào trong tài liệu này. Mọi bảng dưới đây mô tả **giao thức
  đo**, chưa có kết quả — chưa nhánh nào của V2 được chạy.
- Phụ thuộc: [Design V1 (RQ1–RQ3)](../../../docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md),
  [Constitution v1.0.0](../../../.specify/memory/constitution.md).

## 0. Phạm vi và thứ tự với V1

V2 phụ thuộc đầu ra của V1 (candidate top-20 đã khóa, corpus 3 tier, run infra), nên
**không chặn cổng hoàn tất V1**. Phần không phụ thuộc corpus/checkpoint thật (prompt,
parser, bộ kiểm chứng trích dẫn) làm được sớm trên fixture — xem §7.

Quyết định phạm vi đã chốt qua brainstorming (2026-09-12):

| # | Chủ đề | Quyết định |
|---|---|---|
| RG1 | Mục đích | Nhánh nghiên cứu là lõi (đo bằng metric, thêm RQ4); demo dùng lại đúng pipeline đó. Nghiên cứu làm trước |
| RG2 | Model sinh câu trả lời | Model mở chạy trên Colab T4 là nhánh chính (tái lập được); API (Claude) là nhánh đối chứng |
| RG3 | Trích dẫn | Trích nguyên văn rồi kiểm chứng bằng code (không dùng citation của API làm sự thật tuyệt đối, không tự suy vị trí từ model) |
| RG4 | Output | Hai trường: `short_answer` (chấm EM/F1) và `answer` (câu đầy đủ, hiển thị) |
| RG5 | V1 hay V2 | V2 — không chặn V1, cần amendment/spec riêng, không sửa V1.1 |
| RG6 | Web | FastAPI (chỉ `127.0.0.1`) + giao diện HTML/CSS/JS thuần riêng, không dùng framework frontend, không deploy |
| RG7 | Chức năng web | 1–5 (hỏi đáp, chọn cấu hình, xem chi tiết từng bước, chế độ kiểm thử trên dev, so sánh 2 cấu hình). Không làm lưu lịch sử/xuất kết quả (mục 6) |

## 1. Kiến trúc và thành phần

```
Trình duyệt ─► app/static (HTML/CSS/JS) ─► FastAPI app/ (chỉ bind 127.0.0.1)
                                                  │
Batch runner nghiên cứu (scripts/) ────────────────┤   hai lối vào, cùng một pipeline lõi
                                                  ▼
             src/integration.py — pipeline chung, đã có ở V1
             Retriever (BM25 | Dense+FAISS) → Selection (rerank | top-5)
                                                  │  ≤5 passage
                                                  ▼
             src/answerer.py — interface Answerer (MỚI, V2)
               ├─ ExtractiveAnswerer : bọc Reader V1, KHÔNG đổi hành vi Reader
               ├─ LLMAnswerer + ApiClient        (Claude, citations gốc)
               └─ LLMAnswerer + OpenModelClient  (Transformers: batch trên T4,
                                                  hoặc endpoint Colab cho web)
                                                  │
             src/citation.py — kiểm chứng trích dẫn, tính vị trí ký tự (MỚI, V2)
                                                  ▼
  Batch: predictions + run_manifest → Evaluation mở rộng (RQ4, §3)
  Web:   JSON trả về → giao diện hiển thị (§4)
```

**Ranh giới các thành phần (MỚI trong V2):**

- **`Answerer`** nhận câu hỏi và ≤5 passage (`passage_id`, `text`, `title`), trả về câu
  trả lời + trạng thái theo schema ở §2. Pipeline phía trước (Retriever/Selection) không
  biết bên dưới `Answerer` là model nào.
- **`citation.py`** chỉ gồm hàm thuần (không gọi model): kiểm tra `quote` có là chuỗi con
  của passage, tính vị trí ký tự, kiểm tra `short_answer` nằm trong `quote`.
- **LLM client** (`llm_clients.py`) là nơi duy nhất import `anthropic` hoặc code sinh văn
  bản của Transformers. Đổi model không đụng phần còn lại.
- **`app/`** mỏng: service nạp model/index theo tier một lần lúc khởi động, giữ trong bộ
  nhớ. Route không chứa logic nghiệp vụ.
- **Ranh giới read-only (quan trọng):** hàm pipeline lõi (`integration.py`) **không tự
  ghi `run_manifest`**. Ghi manifest là một lớp mỏng riêng, chỉ có ở batch runner nghiên
  cứu (`scripts/`), không nằm trong đường gọi của web. Nhờ vậy web dùng chung pipeline mà
  vẫn read-only — xem §4.

**Ràng buộc bắt buộc với V1:**
- Không sửa module spec V1.1.
- Khi `ExtractiveAnswerer` bọc Reader V1: output span, `passage_id`, offset và trạng thái
  phải **giữ nguyên** so với Reader V1 chạy cùng input (test hồi quy, §7 — hiện chưa có
  benchmark V1 bất biến để so bit-for-bit, nên test so trên frozen fixture + checkpoint,
  không phải so với một file benchmark đã đóng băng).

**File dự kiến khi triển khai** (chưa tạo — đây là kế hoạch, đặt tại vị trí chuẩn trong
repo `src/`/`app/`/`tests/`, không nằm trong `future_RAG/`):

- `src/answerer.py`, `src/llm_clients.py`, `src/citation.py`
- `src/prompts/answer_de_v1.txt` (prompt `rag`/`oracle_rag`, version hóa, hash ghi vào manifest/config)
- `src/prompts/answer_de_closed_book_v1.txt` (prompt `closed_book`, xem §2.1)
- `src/prompts/answer_de_claude_v1.txt` (prompt riêng cho `model_backend=claude`, xem §2.1 —
  phát hiện là thiếu khi triển khai Task 8, bổ sung vào đây ngày 2026-09-12)
- `src/schemas_rag.py` (schema PredictionRecord mở rộng cho RAG, tách khỏi `src/schemas.py` của V1)
- `app/main.py`, `app/routes.py`, `app/service.py`, `app/static/`
- Test tương ứng trong `tests/`

## 2. Bộ trả lời và cơ chế trích dẫn

### 2.1 Đầu vào cho LLM

≤5 passage được đánh số `[P1]..[P5]` theo thứ tự selection, kèm title và text. Model
**không thấy `passage_id`**; code tự map số thứ tự sang ID. Prompt ra lệnh bằng tiếng Anh,
bắt trả lời bằng tiếng Đức. Prompt là file có version và **hash SHA-256**, tối đa 2–3 biến
thể, chỉ so sánh trên dev (không tune theo test — theo Constitution II).

**Prompt riêng cho `closed_book`:** không có passage nào để đưa vào, nên dùng một file
prompt khác (`answer_de_closed_book_v1.txt`, cũng version + hash riêng), chỉ yêu cầu
`KURZ`/`ANTWORT`, **không có dòng `QUELLE`/`ZITAT`** trong định dạng output — vì không có
gì để trích, đòi hỏi model điền hai dòng đó là vô nghĩa. Answerer wrapper tự gán
`citation_status=null` cho mọi câu trả lời của `closed_book` (không đi qua nhánh
`missing` của parser, vì `missing` ngụ ý "đáng lẽ phải có citation nhưng thiếu"; ở đây
citation không áp dụng ngay từ đầu). `rag` và `oracle_rag` dùng chung một prompt có danh
sách passage (khác nhau ở nguồn passage: qua retrieval hay lấy thẳng gold).

**Prompt riêng cho `model_backend=claude`:** cũng dùng passage như `rag`/`oracle_rag`,
nhưng passage không được nhúng vào text prompt — mỗi passage gửi như một `document`
riêng của request (§2.3), và citation (tương đương `QUELLE`/`ZITAT`) do chính API trả về
qua cơ chế `citations`, không phải model tự gõ. Vì vậy prompt cho nhánh này
(`answer_de_claude_v1.txt`, cũng version + hash riêng) chỉ yêu cầu `KURZ`/`ANTWORT`,
**không có placeholder danh sách passage và không có dòng `QUELLE`/`ZITAT`** — khác cả
`answer_de_v1.txt` (có placeholder passage, đủ 4 nhãn) lẫn `answer_de_closed_book_v1.txt`
(không passage, không citation). Tổng cộng V2 có **3 file prompt**, không phải 2 — điểm
này bị bỏ sót ở bản thiết kế đầu, phát hiện khi ghép `LLMAnswerer` (Task 8) vì nhánh
Claude không thể dùng chung template có placeholder `{passages}` với nhánh model mở.

### 2.2 Định dạng output

Dùng nhãn dòng cố định, không dùng JSON (model nhỏ hay làm hỏng JSON khi gặp dấu nháy
tiếng Đức). Đúng hai dạng hợp lệ, parser không đoán field nào là `None`:

```
Có đáp án:                           Không có đáp án:
KURZ: <chép nguyên văn>              KURZ: KEINE ANTWORT
ANTWORT: <một câu tiếng Đức>         ANTWORT: None
QUELLE: P<k>                         QUELLE: None
ZITAT: "<nguyên văn từ P<k>>"        ZITAT: None
```

Dạng không có đáp án phải khớp **đúng** cả 4 giá trị trên; lệch bất kỳ đâu (kể cả có kèm
`QUELLE`/`ZITAT` có giá trị) → `generation_status=error, error_type=format_violation`.

**Dạng có đáp án chỉ bắt buộc `KURZ`/`ANTWORT` có giá trị.** Nếu model bỏ qua hoàn toàn
dòng `QUELLE`/`ZITAT` (hoặc ghi rỗng/`None`), đây **vẫn là một parse hợp lệ**, không phải
`format_violation` — nhưng thiếu bất kỳ nửa nào của cặp `QUELLE`/`ZITAT` đều dẫn tới
`citation_status=missing` (không suy đoán nửa còn thiếu từ nửa kia). Tóm lại,
`format_violation` chỉ áp dụng khi: (a) một nhãn xuất hiện nhiều hơn 1 lần, (b) `KURZ`
hoặc `ANTWORT` bị thiếu/rỗng, hoặc (c) dạng `KEINE ANTWORT` mà `ANTWORT`/`QUELLE`/`ZITAT`
không đúng cả ba giá trị `None`.

**Giới hạn đã biết:** quy tắc (c) chỉ ràng buộc dạng `KEINE ANTWORT`. Dạng có đáp án không
cấm `ANTWORT` là đúng literal chuỗi `"None"` (model tiếng Đức khó tự sinh ra từ này nên
rủi ro thấp trong thực tế, nhưng parser không chặn) — không coi đây là lỗi cần sửa ngay,
chỉ ghi nhận để không nhầm là bug khi gặp.

**Hai backend phải dùng chung một định nghĩa `format_violation`** (nhãn lặp, `KURZ`/
`ANTWORT` thiếu/rỗng): `model_backend=open` áp dụng quy tắc trên cho cả 4 nhãn;
`model_backend=claude` áp dụng đúng cùng quy tắc đó nhưng chỉ trên 2 nhãn `KURZ`/
`ANTWORT` (không có `QUELLE`/`ZITAT` dạng text — xem §2.3). Hai nhánh **không được** có
hai cách hiểu khác nhau về nhãn rỗng hay nhãn lặp, nếu không tỉ lệ `format_violation` ở
§3.5 sẽ không so sánh được giữa hai backend.

### 2.3 Nhánh model (`model_backend`)

`answerer` (loại bộ trả lời) và `model_backend` (nhánh model bên dưới) là hai chiều tách
biệt:

```
answerer      ∈ {extractive, closed_book, rag, oracle_rag}
model_backend ∈ {open, claude}   # bắt buộc khi answerer ≠ extractive
```

`closed_book` và `oracle_rag` cũng gọi LLM (chỉ khác nội dung passage đưa vào — không đưa
gì, hoặc đưa đúng gold passage), nên cũng cần `model_backend`.

- **`model_backend=open`:** giải mã greedy (tất định), ghim model revision và cấu hình
  lượng tử hóa. Chạy batch trên Colab T4, hoặc qua endpoint Colab khi phục vụ web.
- **`model_backend=claude`:** mỗi passage gửi dưới dạng một document, bật `citations`.
  Model vẫn trả hai dòng `KURZ`/`ANTWORT`; `cited_text` của API dùng làm input cho `ZITAT`
  nhưng **vẫn bị kiểm chứng lại** như mọi nhánh khác (§2.4 điểm 4). Trong research run:
  **tắt fallback sang model khác** — một run không được trộn hai model; model từ chối trả
  lời thì ghi `generation_status=error`.

### 2.4 Mô hình trạng thái

`generation_status` (tầng gọi model) và `citation_status` (tầng kiểm chứng) **tách biệt
hoàn toàn**, không trộn lẫn:

```
generation_status = error   (lỗi model/API, timeout, refusal, sai định dạng)
  → answered = null, citation_status = null, answer_supported = null
  → error_type ghi loại lỗi cụ thể (vd format_violation)

generation_status = success
  → answered = false   (đúng dạng KEINE ANTWORT)
    → citation_status = null, answer_supported = null
  → answered = true
    → citation_status ∈ {verified, wrong_passage, quote_not_found,
                          out_of_range, missing, multiple_sources}
    → answer_supported:
        true  khi citation_status = verified VÀ short_answer nằm trong quote
        false khi citation_status = verified NHƯNG short_answer không nằm trong quote
        null  khi citation_status ≠ verified
```

Quy trình kiểm chứng trong `citation.py`, áp dụng như nhau cho mọi `model_backend`:

1. **Parse** theo §2.2. Không đọc được → `generation_status=error`.
2. **Kiểm tra `QUELLE`.** Số passage ngoài khoảng 1..n → `out_of_range`.
   Nhiều passage (`P1, P3`, hoặc API trả nhiều citation) → `citation_status=multiple_sources`,
   **không tự chọn một cái**: `citations=[]`, giữ nguyên `raw_citation_refs=["P1","P3"]` để
   audit, `answer_supported=null`.
3. **Tìm `ZITAT`/`cited_text` trong đúng passage được trích.** Hai mức khớp: chuỗi con
   chính xác, hoặc khớp sau chuẩn hóa NFC + gộp khoảng trắng rồi map lại vị trí gốc.
   **Không dùng fuzzy match.** Xuất hiện nhiều lần → lấy lần đầu, ghi
   `quote_occurrence_count`.
4. Thấy ở **passage khác** (không phải passage được trích) → `wrong_passage`, **không tự
   sửa** số passage. Không thấy ở đâu → `quote_not_found`. Với `model_backend=claude`, vị
   trí ký tự do API trả (`api_char_start/end`) chỉ lưu để audit
   (`api_offset_agrees`); **vị trí do code tự tính mới là kết quả chính thức**.
5. **Kiểm tra `short_answer` nằm trong `quote`** bằng cùng cách chuẩn hóa như EM/F1 của
   V1 (`normalize_answer`). Khớp chuỗi con chính xác → tính `short_answer_start/end`.

Field audit lưu cho mỗi citation: `passage_id`, `quote`, `quote_start`, `quote_end`,
`quote_match_method` ∈ {`exact`, `normalized`}, `quote_occurrence_count`,
`short_answer_start`/`short_answer_end` (null nếu không khớp), cộng — chỉ khi
`model_backend=claude` — `api_char_start`/`api_char_end` (vị trí API trả về, lưu nguyên
văn, không sửa) và `api_offset_agrees` (bool: có khớp với `quote_start`/`quote_end` do
code tự tính hay không). Ba field cuối **chỉ để audit**, không bao giờ dùng thay cho
`quote_start`/`quote_end` — đây là bản sửa của bản thiết kế đầu, vốn nhắc tới
`api_offset_agrees` ở bước 4 phía trên nhưng quên liệt kê field tương ứng ở đây (phát
hiện ở review cuối cùng của batch triển khai đầu tiên). Với `model_backend=open` ba field
này không áp dụng (null).

**Không dùng field `grounded`.** Chỉ dùng `answer_supported` — nó chỉ chứng minh
`short_answer` có nguồn, **không** chứng minh toàn bộ câu `answer` không chứa thông tin
ngoài `quote` (V2 chưa làm claim-level verification; đây là giới hạn phải nêu trong báo
cáo, không gọi nhầm là "grounded" tuyệt đối).

### 2.5 Ánh xạ giữa các loại answerer

| | `extractive` (Reader V1) | `rag` / `closed_book` / `oracle_rag` |
|---|---|---|
| `short_answer` | span trích ra | đáp án model chép nguyên văn |
| `answer` | = span | câu tiếng Đức do model sinh |
| `quote` | = span | đoạn nguồn được trích |
| `citation_status` | `verified` theo cách xây dựng | theo bộ kiểm chứng ở §2.4 |
| Không có đáp án | `answered=false` (D4 của V1) | dạng `KEINE ANTWORT` |

`closed_book` không có passage nào để trích → `citation_status=null` toàn bộ.

## 3. Đánh giá nghiên cứu

### 3.1 RQ4 (thêm, không sửa RQ1–RQ3 của V1)

> **RQ4 — Khi corpus tăng từ base lên large, answer correctness, answer-supported rate và
> citation accuracy của RAG thay đổi thế nào, so với mốc closed-book và oracle?**

Bảng 4 (answerer × tier × metric), Hình 3 (answer-supported rate và EM theo tier), theo
đúng khuôn RQ1–RQ3 đã có trong design V1. Chưa có số liệu — bảng/hình chỉ tồn tại sau khi
chạy thật.

### 3.2 Baseline

| Baseline | Input cho LLM | Mục đích |
|---|---|---|
| `closed_book` | Không đưa passage nào | Đo model "tự biết" bao nhiêu, tách khỏi đóng góp của retrieval |
| `rag` | ≤5 passage từ pipeline V1 (retriever × rerank × tier) | Cấu hình chính |
| `oracle_rag` | Đúng gold passage (`gold_evidences`), không qua retrieval | Mốc năng lực sinh câu trả lời, giống oracle của Reader V1 |

### 3.3 Ma trận thực nghiệm

`closed_book` và `oracle_rag` không phụ thuộc retrieval/rerank/tier, nên **không nhân
bản** theo tier — ghi `tier = N/A` (một hàng dùng chung), không tạo 2 dòng `base`/`large`
với cùng một kết quả.

| Answerer | Số run (`model_backend=open`) | Tier |
|---|---|---|
| `rag` | **4** = Dense × {rerank bật, tắt} × {base, large} | base, large (run riêng) |
| `closed_book` | **1** | N/A (dùng chung) |
| `oracle_rag` | **1** | N/A (dùng chung) |
| `extractive` | 0 run mới — lấy lại kết quả Reader V1 đã có | — |

→ **6 run chính** trên `model_backend=open`. `model_backend=claude` chạy thêm **1 run** ở
cấu hình `rag` có **EM/F1 cao nhất trên dev** trong 4 cấu hình ở bảng trên (cùng tiêu chí
chọn cấu hình như P2 của V1: chọn trên dev, tie thì chọn cấu hình sớm hơn theo thứ tự
liệt kê) — đối chứng cách trích dẫn, không lặp lại toàn bộ ma trận, không chạy
`closed_book`/`oracle_rag` trên nhánh claude.

### 3.4 Bucket lỗi — 7 nhánh, loại trừ nhau theo thứ tự ưu tiên

Mở rộng 5 bucket của V1, chỉ áp dụng khi `answerer ≠ extractive` (không sửa bucket của
V1). Gán theo **đúng thứ tự ưu tiên sau** — một query có nhiều dấu hiệu lỗi cùng lúc thì
chỉ dừng ở bucket đầu tiên khớp, không tiếp tục kiểm các bucket sau (ví dụ:
`generation_failure` thì không kiểm citation nữa):

```
execution_failure
  → retrieval_miss
    → selection_failure
      → generation_failure   (generation_status = error)
        → grounding_failure  (answered=true AND
                               (citation_status ≠ verified OR answer_supported=false))
          → answer_failure   (answered=false)
                          OR (answer_supported=true AND EM<1)
            → correct        (answer_supported=true AND EM=1)
```

`final_correct` (EM=1) vẫn lưu riêng, không suy từ bucket và ngược lại (đúng nguyên tắc
D-series của V1): một câu có thể `final_correct=true` dù rơi vào `grounding_failure`
(đoán đúng nhờ tri thức tham số, trích dẫn sai).

### 3.5 Metric và mẫu số

| Metric | Mẫu số | Ghi chú |
|---|---|---|
| EM/F1 | `N_eval` | Dùng nguyên `normalize_answer`/`f1_score` của `src/evaluation.py`, không viết lại |
| Answer-supported rate | các câu `answered=true` | |
| Citation accuracy | các câu `answered=true` | Chấp nhận `quote` nằm trong **bất kỳ** passage nào thuộc `gold_evidences` của câu đó, không chỉ một passage cố định |
| Hallucination rate (proxy) | các câu `answered=true` | Proxy ở mức trích dẫn (`wrong_passage`/`quote_not_found`), không phải hallucination mức từng claim |
| Format violation rate | toàn bộ `N_eval` | Mọi lượt gọi generate, kể cả lỗi |
| Latency | `n_latency`, chỉ `generation_status=success` | Tách riêng bước `generate` và `verify_citation` |

Dev chọn prompt/model/lượng tử hóa; test chỉ chạy một lần sau khi khóa (Constitution II).

## 4. Web

FastAPI, chỉ bind `127.0.0.1`, phục vụ luôn file giao diện (cùng origin, không CORS,
không build step) — giữ đúng phong cách website tĩnh hiện có.

### 4.1 Route

| Route | `answerer` cho phép | Ghi chú |
|---|---|---|
| `GET /api/config` | — | Liệt kê `public_answerers` (`extractive`, `rag`, `closed_book`) và `dev_only_answerers` (thêm `oracle_rag`); mỗi mục kèm `available: bool` + field cấu hình ở §5.3 |
| `POST /api/ask` | `extractive`, `rag`, `closed_book` | `oracle_rag` → `422` (không có gold cho câu hỏi tự nhập) |
| `GET /api/dev-questions?tier=` | — | Danh sách câu hỏi **dev**, không lộ đáp án; tự gán `split=dev`, không nhận tham số split |
| `GET /api/dev-questions/{query_id}?tier=` | `extractive`, `rag`, `oracle_rag`, `closed_book` | Chấm 1 câu dev: chạy answerer nội bộ + cộng `gold`, `em`, `f1`, `bucket` |
| `POST /api/compare` | theo route tương ứng, tùy có `query_id` hay không | Chỉ ghép cấu hình hợp lệ với từng loại |

`closed_book`/`oracle_rag` không nhận retriever/rerank/tier — bị **ép null ở backend**
(không chỉ ẩn ở UI), dù request gửi kèm gì:

```
closed_book: retriever=null, rerank=false, tier=null, retrieved=[], selected=[]
oracle_rag:  retriever=null, rerank=false, tier=null, retrieved=[],
             selected=[gold passage của query_id]   # chỉ có ở route dev
```

### 4.2 Read-only

> Web không ghi artifact nghiên cứu, không tạo/cập nhật `run_manifest`, không sửa corpus
> hay checkpoint. Cache runtime (embedding câu hỏi, model đã nạp) và log lỗi tạm thời được
> phép.

Cơ chế: `integration.py` không tự ghi manifest (§1); ghi manifest chỉ có ở batch runner.

### 4.3 Đồng thời và timeout

`GET /api/dev-questions/{query_id}` là GET nhưng chạy inference:
- Timeout theo `model_backend` (model mở qua Colab cần dài hơn API).
- Nếu cùng `(query_id, config)` đang chạy → **`HTTP 429`**,
  `error_type=duplicate_inflight_request` (không chờ, không giữ connection lâu).
- Response luôn có `run_mode` ∈ {`ask`, `dev_evaluation`} để không lẫn nguồn khi debug log.

### 4.4 Giao diện

- Form chọn cấu hình + ô nhập câu hỏi. `closed_book`/`oracle_rag` tự ẩn ô retriever/tier.
- Khu kết quả: `answer` nổi bật; `short_answer`/EM-F1 thu nhỏ, chỉ hiện ở chế độ dev.
  Passage hiển thị dạng thẻ, `quote` tô nền, `short_answer` bên trong viền đậm hơn.
- Trạng thái hiển thị bằng nhãn màu kèm chữ (không chỉ màu): `verified` xanh,
  `wrong_passage`/`quote_not_found` đỏ, `missing` xám.
- Tab "So sánh" gọi `/api/compare`, hai cột. Tab "Kiểm thử" gọi `/api/dev-questions`, chỉ
  bật khi tier tương ứng đã build.
- Không có trong V2: WebSocket/streaming, lưu lịch sử, xác thực.

## 5. Xử lý lỗi và cấu hình

### 5.1 HTTP contract — tách theo thời điểm lỗi

```
503 — lỗi TRƯỚC KHI inference bắt đầu
      (index/checkpoint/endpoint chưa sẵn sàng cho tier/backend yêu cầu)

200 — inference ĐÃ bắt đầu; kết quả ghi vào generation_status=error
      (timeout, refusal, rate limit, format violation)

422 — input sai (câu hỏi rỗng, tier chưa build, answerer không tồn tại,
      hoặc oracle_rag gọi qua /api/ask)

429 — trùng (query_id, config) đang chạy (error_type=duplicate_inflight_request)
```

### 5.2 Chính sách retry, theo loại run

```
research run: không retry, không fallback, đúng 1 generation_status=error được ghi
web demo (V2 hiện tại): cũng không tự động retry

Muốn thử lại trên web → người dùng bấm nút, tạo request /api/ask MỚI,
không sửa/ghi đè kết quả của lượt trước.
```

### 5.3 Cấu hình đủ để tái lập

`GET /api/config` và mỗi response của `/api/ask`/`/api/dev-questions/{query_id}` đều kèm:

```
prompt_version, prompt_path, prompt_sha256          # chung mọi backend

model_backend = open:
  model_revision, quantization_config

model_backend = claude:
  model_id, api_version, citations_enabled
```

### 5.4 Tier khả dụng — lấy từ artifact thật, không lấy từ `.env`

Lúc startup, service tự kiểm tra `corpus_manifest.json`, FAISS index, checkpoint reranker
cho từng tier, đối chiếu fingerprint như run_manifest của V1. Tier chỉ `available=true`
khi artifact tồn tại **và** fingerprint khớp. `.env` (`CORPUS_TIERS_AVAILABLE`) chỉ là
**allowlist thu hẹp thêm**, không bao giờ dùng để bật một tier chưa có artifact thật.

`.env` khác (đọc một lần lúc startup, không đọc lại giữa chừng):

```
ANTHROPIC_API_KEY=            # rỗng → model_backend=claude bị disable, không lỗi khi start
RAG_OPEN_MODEL_ENDPOINT=      # URL Colab, rỗng → model_backend=open bị disable trên web
RAG_PROMPT_VERSION=answer_de_v1
CORPUS_TIERS_AVAILABLE=base   # allowlist thu hẹp, không phải nguồn sự thật
```

Thiếu `.env` hoàn toàn → web vẫn chạy được với `extractive`; `rag`/`closed_book` báo
`available=false`, không lỗi khi khởi động.

## 6. Ngoài phạm vi V2

- Deploy public, xác thực, Docker/CI/CD (không cần vì chỉ chạy local).
- Lưu lịch sử/xuất kết quả trên web (RG7) — batch runner đã lưu artifact kèm manifest.
- Claim-level hallucination detection (chỉ có `answer_supported` ở mức trích dẫn).
- Retry/fallback tự động ở bất kỳ tầng nào.
- Hybrid retrieval, hard-negative mining, Haystack (thuộc V2 khác, không liên quan RAG).

## 7. Kiểm thử và thứ tự làm

### 7.1 Kiểm thử — không gọi mạng thật trong test suite

| Lớp | Test |
|---|---|
| `citation.py` | Khớp chuỗi con chính xác, khớp sau chuẩn hóa, nhiều lần xuất hiện, không tìm thấy, `wrong_passage`, `short_answer` không nằm trong `quote` |
| Parser output | Hai dạng hợp lệ; mọi kiểu sai định dạng → `format_violation`; `multiple_sources` giữ `raw_citation_refs` |
| `ExtractiveAnswerer` | Test trên **frozen fixture + checkpoint V1** (chưa có benchmark V1 bất biến để so bit-for-bit): output span, `passage_id`, offset và trạng thái phải giữ nguyên so với Reader V1 khi chạy cùng input |
| `LLMAnswerer` | Dùng fake client trả sẵn text theo kịch bản (đúng định dạng, sai định dạng, `KEINE ANTWORT`, nhiều nguồn) — không gọi API thật |
| Bucket 7 nhánh | Fixture đủ case phủ hết 7 bucket, gồm case `verified` nhưng `answer_supported=false`, **và** case một query có nhiều dấu hiệu lỗi cùng lúc để xác nhận đúng thứ tự ưu tiên §3.4 (vd lỗi generation thì không được tiếp tục kiểm citation) |
| Route FastAPI | `TestClient`: `oracle_rag` bị chặn ở `/api/ask` (422); tier chưa build → 503; input rỗng → 422; `closed_book`/`oracle_rag` bị ép null dù request gửi kèm; trùng `(query_id, config)` → 429 |

### 7.2 Thứ tự làm

*Làm được ngay, trên fixture, không cần đợi V1:*
1. `citation.py` + test
2. `src/prompts/answer_de_v1.txt` + cơ chế hash
3. `llm_clients.py` (API thật + fake client cho test) + parser
4. `LLMAnswerer` ghép citation + parser, test bằng fake client

*Phải chờ phần tương ứng của V1:*
5. `ExtractiveAnswerer` — chờ Reader V1 (Chặng 8) có checkpoint
6. Bucket 7 nhánh mở rộng — chờ `evaluation.py` V1.1 ổn định (Chặng 6)
7. `app/` (FastAPI + static) — khung route/service dựng sớm được với pipeline giả lập,
   nối thật vào Retriever/Selection chờ Integration (Chặng 9)
8. Thực nghiệm RQ4 thật (ma trận 6+1 run, chọn prompt/model trên dev) — chỉ chạy sau khi
   V1 qua Chặng 9 và candidate top-20 đã khóa

## 8. Nhật ký cập nhật

2026-09-12: Viết design doc đầu tiên, gộp Phần 1–6 sau khi từng phần được duyệt qua
brainstorming — quyết định phạm vi RG1–RG7, 4 vòng sửa trên Phần 2+3 (2 vòng mỗi phần),
3 vòng sửa trên Phần 4/5/6 (1 vòng mỗi phần). Không sửa module spec V1.1. Chưa có
implementation plan, chưa có code, chưa có kết quả thực nghiệm.
Tự rà lại sau khi viết: bổ sung prompt/định dạng output riêng cho `closed_book` (không có
`QUELLE`/`ZITAT` vì không có passage để trích), làm rõ tiêu chí chọn cấu hình `rag` tốt
nhất cho run đối chứng API (EM/F1 trên dev, cùng kiểu P2 của V1), sửa số vòng sửa ở trên
cho khớp lịch sử hội thoại.

2026-09-12 (sau khi triển khai Task 1–8 qua subagent-driven-development, review cuối trên
model mạnh nhất): sửa 3 điểm phát hiện ở review cuối cùng của batch triển khai đầu tiên —
(1) §1/§2.1: thêm `src/prompts/answer_de_claude_v1.txt` — nhánh `model_backend=claude` cần
prompt riêng (chỉ KURZ/ANTWORT, không placeholder passage) vì passage đi qua `documents`
của API, không nhúng vào text; thiếu sót này chỉ lộ ra khi ghép `LLMAnswerer` ở Task 8,
đã vá tại chỗ trong lúc triển khai, giờ ghi lại vào spec cho khớp. (2) §2.2: nêu rõ hai
backend (`open`/`claude`) phải dùng chung một định nghĩa `format_violation`, và ghi nhận
giới hạn "ANTWORT: None" dạng literal không bị chặn ở nhánh có đáp án. (3) §2.4: bổ sung
`api_char_start`/`api_char_end`/`api_offset_agrees` vào danh sách field audit (trước đó
nhắc ở bước 4 nhưng quên liệt kê) — 3 field audit-only, chưa có trong `CitationDetail`
của code hiện tại, để lại cho người làm `schemas_rag.py` sau. Không sửa module spec V1.1.
