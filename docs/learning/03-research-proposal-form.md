---
trang_thai: draft — chưa duyệt
pham_vi: Form phân tích rút từ Research_Proposal_Corruption_Hierarchical_Learning_Diffusion_LM.pdf, áp dụng cho project German QA retrieval (V1)
ngay_tao: 2026-09-25
---

# 03 — Form phân tích Research Proposal, áp dụng cho German QA retrieval

Tài liệu này có hai phần: (A) rút ra cấu trúc chung ("form") từ
`Research_Proposal_Corruption_Hierarchical_Learning_Diffusion_LM.pdf`, và (B) điền form đó
bằng nội dung thật của project trong repo này. Phần B không phải tài liệu spec mới — mọi nội
dung tham chiếu spec/design hiện có, không thay thế chúng.

## A. Form (cấu trúc rút ra từ PDF mẫu)

PDF mẫu là một research proposal 7 mục + 1 mục "epistemic status" ở cuối:

| # | Mục | Câu hỏi mà mục đó trả lời |
|---|---|---|
| 1 | Research Problem & Motivation | Vì sao câu hỏi này đáng nghiên cứu? Cái gì đã biết, cái gì chưa? |
| 2 | Related Work & Research Gap | Ai đã làm gì gần nhất, và khoảng trống cụ thể còn lại là gì? (chia nhỏ theo nhóm: nền tảng / quy mô lớn gần đây / công trình gần nhất) |
| 3 | Research Question & Hypothesis | Câu hỏi nghiên cứu chính xác là gì? Giả thuyết — kèm CẢ các khả năng trái chiều, không chỉ khả năng mong muốn |
| 4 | Proposed Experiment | Dataset/setting, model, **reference condition** (baseline cố định), **biến can thiệp** (biến độc lập), **biến kiểm soát** (mọi thứ khác giữ cố định) |
| 5 | Evaluation & Analysis Plan | Metric chính, metric phân giải mịn hơn (theo mức/nhóm), cách đo "tốc độ"/độ khó, error analysis theo cấu trúc lỗi (không chỉ 1 con số), ablation |
| 6 | Scope & Feasibility | In scope / Out of scope tường minh, bảng tái sử dụng hạ tầng sẵn có, feasibility (compute, pilot trước khi chạy chính) |
| 7 | References | Trích dẫn từng công trình đã dùng ở mục 2 |
| 8 | Epistemic Status | Chú thích mọi câu khẳng định bằng 1 trong 3 nhãn nguồn gốc |

Đặc điểm quan trọng nhất của form này — không phải là các mục (khá chuẩn cho mọi proposal),
mà là **kỷ luật gắn nhãn nguồn gốc cho từng câu**, dùng xuyên suốt cả 7 mục:

- `[Literature-supported]` — có công trình đã công bố chứng minh/hỗ trợ.
- `[Proposed]` (hoặc biến thể `[Project design]`, `[Proposed interpretation]`) — quyết định
  thiết kế của chính project, xây trên literature nhưng chưa ai kiểm chứng cho case này.
- `[To be tested]` — giả thuyết/claim chưa có bằng chứng, sẽ được experiment trả lời.

Nhờ nhãn này, proposal phân biệt rõ "cái tôi biết" / "cái tôi chọn làm" / "cái tôi chưa biết
và sẽ đo" — không lẫn ba loại vào nhau như văn phong review thông thường.

## B. Áp form vào project German QA retrieval

Điền theo đúng 8 mục ở trên. Nguồn: `docs/superpowers/specs/2026-09-10-german-qa-retrieval-*`
(V1.1 frozen) và memory dự án; phần literature dùng tên công trình đã biết công khai, không
tra cứu lại trong phiên này — cần tự kiểm tra trích dẫn chính xác (số trang, venue) trước khi
đưa vào báo cáo chính thức.

### 1. Research Problem & Motivation

`[Proposed]` Open-domain extractive QA tiếng Đức thường được đánh giá ở một quy mô corpus
cố định, nên không thấy rõ **thành phần nào trong pipeline (retriever thưa, retriever dày,
reranker, reader) là điểm nghẽn khi corpus lớn dần** — một câu hỏi khác với "model nào SOTA
trên GermanQuAD/GermanDPR ở kích thước công bố gốc".

`[Literature-supported]` Möller, Risch & Pietsch (2021) giới thiệu GermanQuAD (extractive QA)
và GermanDPR (retrieval), báo cáo kết quả ở corpus kích thước cố định như bộ dữ liệu gốc công
bố — không phải một ablation theo corpus scale.

`[Literature-supported]` Reimers & Gurevych (ACL-IJCNLP 2021, "The Curse of Dense
Low-Dimensional Information Retrieval for Large Index Sizes") cho thấy retriever dày (dense,
chiều thấp) có thể suy giảm tương đối so với retriever thưa (BM25) khi index lớn dần — nhưng
trên tiếng Anh, không phải trên bộ GermanQuAD/GermanDPR, và không đi kèm reranker + reader
để xem lỗi truyền xuống downstream ra sao.

→ Khoảng trống cụ thể cho project: **chưa có phân tích corpus-scale + phân rã lỗi theo từng
tầng pipeline (retrieval miss → selection failure → reader failure)** cho riêng cặp
GermanQuAD/GermanDPR.

### 2. Related Work & Research Gap

**2.1 Nền tảng từng thành phần** `[Literature-supported]`
- BM25 — Robertson & Zaragoza (2009), hàm xếp hạng thưa dựa trên term frequency, dùng làm
  retriever nền và làm mốc so sánh cho dense retriever.
- DPR (Dense Passage Retrieval) — Karpukhin et al. (2020), hai-tháp encoder độc lập
  (question/passage), huấn luyện bằng in-batch negatives + hard negatives; project dùng đúng
  kiến trúc này (CLS pooling, không chuẩn hóa embedding, theo quyết định khóa của V1.1).
- Cross-encoder reranking — Nogueira & Cho (2019), chấm điểm lại cặp (query, passage) bằng
  một encoder chung thay vì hai tháp độc lập, để tăng precision ở top-k nhỏ sau retrieval.
- Extractive reader — kiến trúc span-extraction kiểu BERT (start/end logits), ở đây dùng
  `deepset/gbert-base`.

**2.2 Công trình gần nhất, cùng bộ dữ liệu** `[Literature-supported]`
- Möller, Risch & Pietsch (2021) — GermanQuAD/GermanDPR gốc. Đây là "reference paper": project
  tái tạo pipeline retrieval→rerank→read của họ nhưng thêm trục corpus-scale mà bản gốc không
  có.

**2.3 Corpus-scale / index-size** `[Literature-supported]`
- Reimers & Gurevych (2021) — bằng chứng gián tiếp (tiếng Anh, không có reranker/reader) rằng
  dense retriever có thể chịu ảnh hưởng khác BM25 khi index lớn.

**2.4 Research Gap** `[Proposed interpretation]`
Hai dòng trên cho thấy: (a) pipeline retrieval→rerank→read đã có baseline vững trên
GermanQuAD/GermanDPR ở quy mô gốc; (b) corpus-scale có thể ảnh hưởng khác nhau lên dense vs
sparse retrieval, quan sát trên tiếng Anh. Nhưng **chưa có nghiên cứu nào kết hợp cả hai**:
đo corpus-scale ablation có kiểm soát, kèm phân rã lỗi 4–5 nhánh xuyên suốt cả pipeline, trên
GermanQuAD/GermanDPR. Đây là khoảng trống project nhắm tới — không phải "chưa ai retrieval
tiếng Đức", mà là "chưa ai định vị điểm nghẽn theo tầng khi corpus tiếng Đức phình to".

### 3. Research Question & Hypothesis

**RQ** `[Proposed]`
> Khi corpus tăng từ base (~7.5K) → mid (50K) → large (150K), retrieval recall, reranking
> gain và reader accuracy trong pipeline German QA thay đổi thế nào — và điểm nghẽn chính (nơi
> phần lớn câu hỏi bị mất) dịch chuyển giữa các tầng retrieval/selection/reader ra sao?

**Hypothesis (H1)** `[To be tested]` — không giả định trước hướng ảnh hưởng, nêu cả hai khả
năng như PDF mẫu làm:
- *Khả năng 1*: thứ tự tầng nào chiếm ưu thế lỗi vẫn ổn định qua các tier (ví dụ
  `retrieval_miss` luôn là bucket lớn nhất), corpus scale chỉ đổi **độ lớn** của từng bucket,
  không đổi bucket nào chiếm ưu thế.
- *Khả năng 2*: corpus scale đổi luôn **bucket chiếm ưu thế** — ví dụ ở base, `reader_failure`
  chiếm nhiều nhất (do reader đọc sai dù retrieval đúng), nhưng ở large, `retrieval_miss`/
  `selection_failure` vượt lên vì corpus lớn làm retrieval khó hơn.

`[Literature-supported — indirect evidence]` Reimers & Gurevych gợi ý khả năng 2 đối với
riêng bước retrieval (dense so với sparse). `[To be tested]` Project kiểm tra liệu chênh lệch
đó có truyền xuống bucket lỗi end-to-end hay bị reranker/reader hấp thụ bớt.

### 4. Proposed Experiment

**4.1 Dataset / corpus tiers** `[Proposed — khóa trong spec V1.1]`
GermanQuAD (extractive) + GermanDPR (retrieval) hợp nhất qua `question_key`; corpus theo
allowed-sources (mọi chunk của GermanQuAD context + GermanDPR positive context = base) cộng
Wikipedia distractor cho mid/large. Ba tier: base ≈7.5K, mid = 50,000, large = 150,000 passage
(sau chunk, ràng buộc `parent_document_id`/offset ký tự).

**4.2 Model** `[Proposed — khóa trong spec V1.1]`
- Retriever thưa: BM25 (`bm25s`).
- Retriever dày: dual-encoder DPR trên `deepset/gbert-base`, CLS pooling, không chuẩn hóa
  embedding, huấn luyện bằng in-batch + hard negatives (nhóm theo `source_context_id`).
- Reranker: cross-encoder (kiến trúc theo Nogueira & Cho), nhận `retrieval_k=20` từ retriever.
- Reader: extractive span-extraction trên `deepset/gbert-base`, luôn nhận đúng `reader_k=5`
  bất kể reranker bật/tắt — để so sánh reranker có kiểm soát.

**4.3 Reference condition** `[Proposed]`
BM25 @ tier base, không rerank — mốc tham chiếu rẻ nhất, không phụ thuộc huấn luyện, dùng để
so mọi cấu hình khác.

**4.4 Experimental intervention (biến độc lập)** `[Proposed]`
Ma trận chính: `{BM25, Dense} × {rerank bật, tắt} × {base, mid, large}`. Không nhân bản những
gì không phụ thuộc tier (ví dụ nếu có baseline "closed-book" kiểu reader-only sau này, ghi
`tier = N/A` một dòng, đúng nguyên tắc PDF mẫu ở mục 3.3, thay vì lặp 3 dòng giống hệt).

**4.5 Controlled variables (giữ cố định)** `[Proposed — khóa trong spec V1.1]`

| Giữ cố định | Thay đổi |
|---|---|
| Query set (GermanQuAD test, 2,196 câu gốc) | Corpus tier |
| `reader_k = 5` | Retriever type (BM25 / Dense) |
| `retrieval_k = 20` | Rerank on/off |
| Kiến trúc reader, checkpoint chọn trên dev | |
| Seed cố định (SHA-256 `stable_seed()`) cho mọi lấy mẫu hard-negative | |
| Checkpoint retriever/reranker: chọn trên `qa_germanquad_dev`, tie → checkpoint sớm hơn, đóng băng cho mọi tier | |

### 5. Evaluation & Analysis Plan

**5.1 Metric chính (retrieval)** `[Proposed — khóa trong spec V1.1]` Recall@k (Hit@k-style),
GoldCoverage@k, MRR@10 — trên từng tier, từng retriever.

**5.2 Metric end-to-end** `[Proposed]` EM/F1 trên `N_eval` (mẫu số = số câu có gold non-empty),
tính bằng đúng `normalize_answer`/`f1_score` đã có trong `src/evaluation.py`.

**5.3 "Tốc độ/độ khó" tương tự `t*_ℓ(α)` của PDF mẫu** `[Proposed]` — PDF đo thời điểm đạt
threshold theo training step; project không có trục training step (mọi model train 1 lần),
nên tương tự gần nhất là: **tier nhỏ nhất mà tại đó một retriever/cấu hình đạt được một
ngưỡng Recall@k cho trước** (ví dụ Recall@20 ≥ 0.8) — so sánh tier này giữa BM25 và Dense cho
biết cấu hình nào "cần corpus nhỏ hơn" để đạt cùng chất lượng, tương tự tinh thần
time-to-threshold.

**5.4 Error analysis (bucket lỗi)** `[Proposed — khóa trong spec V1.1]` 5 bucket loại trừ lẫn
nhau theo thứ tự ưu tiên: `execution_failure → retrieval_miss → selection_failure →
reader_failure → correct`, cộng cờ độc lập `manual_review_suggested` cho nghi ngờ lệch chuẩn
hóa đáp án. Phân tích theo tier: bucket nào phình to nhất khi corpus tăng, có khác nhau giữa
BM25 và Dense không.

**5.5 Robustness / ablation** `[Proposed]` 3 tier (không chỉ 2 đầu mút) để xem xu hướng có
tuyến tính hay có "điểm gãy" ở một khoảng kích thước corpus nhất định — cùng tinh thần
`α ∈ {0, 0.2, ..., 1.0}` của PDF mẫu thay vì chỉ so 2 điểm cực trị.

### 6. Scope & Feasibility

**In scope** `[Proposed]` BM25, Dense (DPR-style), cross-encoder reranker, extractive reader;
3 corpus tier; error decomposition 5 bucket; GermanQuAD test làm query set chính thức.

**Out of scope (tường minh, giống mục 6 của PDF mẫu)** `[Proposed]`
- LLM answerer sinh câu có trích dẫn (đây là nhánh V2, xem `future_RAG/README.md`) — **không
  chặn V1**, chỉ bắt đầu sau khi V1 qua Chặng 9 (Integration).
- Thiết kế kiến trúc encoder/reader mới — dùng nguyên `gbert-base`.
- Pretraining hoặc fine-tune trên dữ liệu ngoài GermanQuAD/GermanDPR.
- Tinh chỉnh `reader_k` theo Hit@reader_k (bị cấm tường minh trong spec vì đơn điệu, gây
  leakage thiết kế).

**Feasibility** `[Proposed]`

| Thành phần | Tình trạng |
|---|---|
| BM25 (`bm25s`) | Đã chạy trên dev, cần chạy lại trên corpus sau khi Milestone A rebuild xong |
| Dense (DPR) | Chưa train — cần GPU (Colab T4), local chỉ CPU |
| Reranker | Chưa train |
| Reader | Chưa train |
| Evaluation harness | Đã có (`src/evaluation.py`, V1.0 semantics — cần cập nhật theo V1.1: bucket `selection_failure`) |
| Hạ tầng dữ liệu (schema, splits, corpus) | Đang rebuild theo V1.1 (T5.0–T5.5, nhánh `feature/v1.1-data-pipeline`) |

Máy local RTX 3050 4GB, `torch` cài local là bản CPU-only → huấn luyện Dense/reranker/reader
cần Colab T4; local chỉ dùng để code/debug/test đơn vị.

### 7. References

1. Möller, T., Risch, J., & Pietsch, M. (2021). *GermanQuAD and GermanDPR: Improving
   Non-English Question Answering and Passage Retrieval*. MRQA Workshop, EMNLP 2021.
2. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020).
   *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020.
3. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and
   Beyond*. Foundations and Trends in Information Retrieval.
4. Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT*. arXiv:1901.04085.
5. Reimers, N., & Gurevych, I. (2021). *The Curse of Dense Low-Dimensional Information
   Retrieval for Large Index Sizes*. ACL-IJCNLP 2021.

*Lưu ý:* danh sách trên viết theo trí nhớ chung về các công trình, chưa tra lại số trang/venue
chính xác trong phiên này — kiểm tra lại trước khi đưa vào báo cáo nộp.

### 8. Epistemic Status

Áp dụng đúng 3 nhãn của PDF mẫu cho mọi câu khẳng định khi viết báo cáo/proposal chính thức
của project này:

- `[Literature-supported]` — có công trình đã công bố (mục 7) hỗ trợ trực tiếp.
- `[Proposed]` — quyết định đã khóa trong spec V1.1 (`docs/superpowers/specs/2026-09-10-
  german-qa-retrieval-module-specs.md`) hoặc quyết định thiết kế của project, xây trên
  literature nhưng chưa có số liệu riêng của project xác nhận.
- `[To be tested]` — giả thuyết (H1 ở mục 3) hoặc bất kỳ con số nào chưa chạy thật; **không
  được viết số liệu dự kiến vào báo cáo như thể đã đo** (đúng nguyên tắc "bằng chứng trước
  khẳng định" mà project đã tự áp cho chính mình ở các tài liệu `docs/learning/00-02`).

Mục đích của form này là ép mọi phát biểu trong báo cáo cuối kỳ phải khai rõ nó đến từ đâu —
literature, quyết định thiết kế, hay giả thuyết chưa kiểm chứng — thay vì trộn lẫn ba loại khi
viết văn xuôi.
