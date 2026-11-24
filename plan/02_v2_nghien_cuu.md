# V2 — Kế hoạch phát triển lên nghiên cứu khoa học

Cập nhật tài liệu: 2026-10-14. Trạng thái: hướng nghiên cứu ứng viên, chưa xác nhận novelty và chưa có kết quả thực nghiệm.

[Mục lục](README.md) · [V1 phần A](01a_v1_phan_a_baseline.md) · [V1 phần B](01b_v1_phan_b_cai_tien.md)

## 1. Mục tiêu và giới hạn của kế hoạch

Phát triển hệ thống V1 hoàn chỉnh (giai đoạn 0–18) thành một nghiên cứu có vấn đề thực nghiệm, đóng góp khác biệt, đối chiếu công bằng, phân tích cơ chế và khả năng tái lập. Mục tiêu công bố quốc tế không được bảo đảm chỉ bằng số module hoặc mức F1 của một cấu hình.

Hướng đề xuất:

> Dùng tín hiệu từ lexical và dense retrieval để chọn ngân sách reranking theo từng câu hỏi, nhằm giảm chi phí trong khi giữ chất lượng đáp án và bằng chứng của QA trích xuất tiếng Đức.

Đóng góp được quyết định sau pilot và literature review. RRF, cross-encoder, hard-negative mining, adaptive routing và robustness tiếng Đức đều có tiền lệ. Không tự nhận việc kết hợp các thành phần hoặc đổi ngôn ngữ là novelty.

Trong thiết kế đầu tiên, mọi query đều chạy BM25 và dense. Router chỉ thay đổi ngân sách cross-encoder; không tuyên bố tiết kiệm chi phí truy hồi chưa hề bỏ qua.

## 2. Bản đồ 10 câu hỏi nghiên cứu

| Câu hỏi | Công việc giải quyết |
|---|---|
| Problem: thất bại ở đâu? | GĐ19 pilot và phân tích lỗi |
| Literature: đã biết gì? | GĐ20 rà soát và verification |
| Gap: hạn chế có bằng chứng? | GĐ19 + GĐ20 |
| Novelty: khác biệt nào? | GĐ20 + GĐ23, còn cần kiểm chứng |
| RQ/Hypothesis: kiểm chứng gì? | GĐ21 |
| Baseline: so với ai? | GĐ24 |
| Experiment: kết quả đáng tin? | GĐ22 + GĐ25 |
| Analysis: vì sao được/thất bại? | GĐ26 |
| Reproducibility: chạy lại được? | GĐ27 |
| Publication: phù hợp nơi nào? | GĐ28 |

## Giai đoạn 19 — Pilot để xác định vấn đề thật

### Mục tiêu và đầu vào

Kiểm tra có tiềm năng chọn ngân sách theo query hay không. Input: V1 (giai đoạn 0–18) ổn định, dev có gold/qrels, traces và timing instrumentation.

### Công việc

- Quét ngân sách reranking, ví dụ 0/10/20/50; reader nhận tối đa 5 passages.
- Candidate pool phải đủ cho ngân sách lớn nhất.
- Ngân sách 0 lấy top-5 RRF trực tiếp; ngân sách m chấm m passages rồi chọn top-5 trong số đã chấm.
- Ghi F1/EM, evidence retention, latency, pair/token/window counts cho từng query–action.
- Phân tích query được lợi, không đổi và bị hại khi tăng ngân sách.
- Tách lỗi retrieval, selection và reader, kèm mức chắc chắn của nhãn lỗi.
- Đo phần tỷ trọng chi phí reranking trong tổng latency.

### Phương pháp

Budget sweep, paired query analysis, quality–cost curves, Pareto frontier và oracle analysis.

Oracle dùng gold để chọn hành động tốt nhất cho từng query trong tập hành động đã thử. Nó là mốc tham chiếu có thông tin đáp án, không triển khai được và không phải giới hạn tuyệt đối cho mọi phương pháp có thể có.

### Kiến thức cần nắm

Chi phí cố định/biến đổi, non-monotonic performance, conditional benefit và sự khác nhau giữa tiềm năng oracle với router có thể học. Bất đồng retrievers cao không mặc nhiên nghĩa là query khó hoặc cần nhiều reranking.

### Đầu ra và điểm quyết định

Pilot report, query–action outcome matrix, error taxonomy ban đầu và quality–cost plot.

Chỉ tiếp tục hướng router nếu có khoảng cách có ý nghĩa giữa cấu hình tĩnh mạnh và lựa chọn theo query, chi phí tối ưu đáng kể, cùng tín hiệu có thể dùng trước reranking. Nếu không, đổi trọng tâm sang phân tích lỗi/robustness hoặc vấn đề khác có bằng chứng; không ép pilot thành kết quả tích cực.

## Giai đoạn 20 — Literature và xác nhận gap

### Mục tiêu

Xác định đóng góp ứng viên khác công trình gần nhất ở đâu và cần đối chiếu gì.

### Phạm vi tìm kiếm

- Hybrid retrieval, rank fusion và dense retrieval.
- Adaptive reranking, dynamic depth, ranked-list truncation.
- Query performance prediction và cost-aware routing.
- Retrieval/reranking theo chất lượng QA cuối.
- German extractive QA, robustness, cross-domain evaluation.
- Efficient cross-encoders, early exit và budgeted candidate selection.

### Quy trình

1. Ghi ngày tìm, database/site, chuỗi tìm kiếm và tiêu chí giữ/loại.
2. Dùng primary papers, official proceedings và author repositories.
3. Đọc phương pháp, experimental setup và limitations; không chỉ abstract.
4. Truy vết backward/forward citations của bài gần nhất.
5. Xác minh title/authors/year/venue/DOI, phân biệt peer-reviewed với preprint.
6. Ghi bằng chứng hỗ trợ và cả công trình làm suy yếu giả thuyết.
7. Lập novelty matrix trước khi dùng câu “chưa có nghiên cứu”.

### Ma trận literature

```text
Paper → Task → Language/domain → Features → Adaptive action
      → Objective → Data/splits → Baselines → Results → Limitations
      → Difference from proposed method → Verification status
```

### Các nguồn khởi đầu đã được đối chiếu trong cuộc trao đổi

| Công trình | Liên quan và giới hạn của tuyên bố mới |
|---|---|
| [GermanQuAD/GermanDPR, MRQA 2021](https://aclanthology.org/2021.mrqa-1.4/) | Đã có dữ liệu/model cho QA và dense retrieval tiếng Đức |
| [RRF, SIGIR 2009](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/) | Fusion thứ hạng đã có |
| [RocketQA, NAACL 2021](https://aclanthology.org/2021.naacl-main.466/) | Đã nghiên cứu hard-negative training và lọc nhiễu |
| [Robustness Evaluation of German Extractive QA, COLING 2025](https://aclanthology.org/2025.coling-main.121/) | Đã khảo sát tám model và perturbations tiếng Đức |
| [Optimal Re-Ranking Depth, SIGIR 2026](https://eprints.gla.ac.uk/383304/) | Độ sâu phù hợp thay đổi theo query; dự đoán không đơn giản |
| [Adaptive Re-Ranking, 2026](https://arxiv.org/abs/2606.25249) | Preprint về cost-aware query routing |
| [AdaPCR, 2025](https://arxiv.org/abs/2507.04069) | Preprint về adaptive passage combinations, liên hệ downstream QA |
| [BoundaryMORPH, 2026](https://arxiv.org/abs/2609.27213) | Preprint về phân bổ CE calls để xác định top-k membership |

Trạng thái xuất bản cần kiểm tra lại khi hoàn tất literature và trước submission. Danh sách này chưa phải systematic review hay chứng minh đã bao phủ mọi công trình.

### Gap ứng viên

Liệu bất đồng lexical–dense cùng tín hiệu giữ bằng chứng hỗ trợ chọn ngân sách CE theo chất lượng extractive QA, vượt các phương pháp tĩnh/thích nghi gần nhất và còn hiệu quả dưới biến đổi đầu vào?

Khác biệt nhiệm vụ, tín hiệu hoặc ngôn ngữ chỉ là giả thuyết về đóng góp; phải có so sánh trực tiếp và thí nghiệm chứng minh ý nghĩa.

### Đầu ra và điều kiện chuyển tiếp

Verified bibliography, literature matrix, gap statement có nguồn, danh sách nearest baselines. Nếu gần trùng phương pháp có trước, sửa thiết kế/đóng góp hoặc chuyển thành nghiên cứu phân tích với phát hiện riêng.

## Giai đoạn 21 — Chốt RQ, hypothesis và protocol

### Câu hỏi và giả thuyết

| RQ | Hypothesis ứng viên | Bằng chứng cần |
|---|---|---|
| RQ1: Disagreement có dự báo lợi ích reranking? | Hữu ích hơn query length hoặc tín hiệu một retriever | Held-out prediction, feature ablation, calibration nếu dự đoán xác suất |
| RQ2: Adaptive policy cải thiện quality–cost? | Giữ F1 trong biên cho phép và giảm chi phí | Non-inferiority cùng latency/Pareto comparison |
| RQ3: Lợi ích giữ dưới noise/shift? | Lợi ích còn trong các điều kiện định trước | Paired perturbations và domain transfer |

### Quyết định phải chốt trước test

- Metric chính/phụ và cách normalize F1.
- Action space, candidate pools, reader top-k.
- Non-inferiority margin có biện minh; ví dụ 1 điểm F1/100 chỉ là ứng viên.
- Cost definition: CE pairs, measured latency và end-to-end latency; không đánh đồng chúng.
- Cách chọn model/threshold trên dev và số cấu hình được thử.
- Seeds, thống kê, primary comparisons và cách xử lý nhiều comparisons.
- Perturbations, group bins, domain datasets và tiêu chí hợp lệ.
- Quy tắc dừng/thay đổi hướng và cách ghi exploratory analyses phát sinh.

### Đầu ra

Research protocol có version, hypothesis và acceptance/rejection criteria. Có thể đăng ký trước protocol nếu phù hợp; không mô tả là đã preregister khi chưa thực hiện.

## Giai đoạn 22 — Dữ liệu nghiên cứu và chống leakage

### Mục tiêu

Chuẩn bị train/router/dev/test và dữ liệu robustness/transfer phản ánh tổng quát hóa.

### Công việc

- Audit lại overlap query/context/article GermanQuAD–GermanDPR.
- Audit checkpoint training history; ghi unknown exposure thay vì khẳng định sạch khi không xác minh được.
- Tách base-model training, router-label creation, tuning và final evaluation.
- Giữ mọi paraphrase/perturbation của cùng query gốc trong cùng split.
- Group theo article/context; giữ ID mapping nguồn.
- Không dùng gold/qrels test để mining hoặc chọn hành động cho training.
- Ghi corpus construction, judged relevance coverage và chính sách document holdout.

### Tạo dữ liệu router tránh in-sample optimism

Nếu reader/retriever đã học query, outcomes của query đó có thể quá thuận lợi. Ưu tiên một trong hai thiết kế:

1. Dành một phần train cho router, base components không fine-tune trên phần đó.
2. Tạo out-of-fold outcomes bằng base components được học trên các folds khác, nếu đủ tài nguyên.

Chọn một protocol khả thi, ghi rõ chi phí và nguồn labels. Khi dùng public checkpoints, báo cáo lịch sử exposure đã biết và hạn chế chưa xác định.

### Dữ liệu chuyển miền

[M2QA](https://aclanthology.org/2024.findings-emnlp.365/) có phần tiếng Đức ở reviews, news và creative writing, theo dạng SQuAD 2.0. Nếu dùng:

- Quy định xử lý unanswerable: đánh giá riêng với abstention hoặc giới hạn answerable subset từ trước và thu hẹp tuyên bố.
- Nếu tạo corpus retrieval từ contexts, công bố đây là protocol do dự án xây và xác định qrels đầy đủ nhất có thể.
- Không dùng corpus-derived distractors để tuyên bố đã đánh giá tìm kiếm toàn web.
- Đánh giá transfer không tuning và adaptation có target-dev phải là hai điều kiện riêng.

### Đầu ra và nghiệm thu

Split manifests, exposure/overlap audit, router-label dataset, robustness dataset, transfer protocol. Test chưa tham gia lựa chọn; nếu benchmark test cũ đã được xem/khai thác, dùng kiểm tra độc lập cho xác nhận và công bố giới hạn.

## Giai đoạn 23 — Thiết kế và cài đặt phương pháp

### Pipeline đề xuất

```text
Câu hỏi
   ├── BM25 ── candidates ──┐
   └── Dense ─ candidates ──┤
                           ↓
                     Hợp nhất RRF
                           ↓
                    Trích đặc trưng
                           ↓
                   Router chọn ngân sách
              ┌────────────┼─────────────┐
              ↓            ↓             ↓
          Bỏ rerank    Rerank ít     Rerank nhiều
              └────────────┼─────────────┘
                           ↓
                        Top-5
                           ↓
                        Reader
                           ↓
                 Đáp án + bằng chứng
```

### 23.1. Đặc trưng trước reranking

| Nhóm | Ví dụ | Lưu ý |
|---|---|---|
| Đồng thuận | Overlap/Jaccard của top lists | Không tự đồng nhất agreement với đúng |
| Bất đồng rank | Rank differences trong tập ứng viên | Định nghĩa rank cho IDs vắng mặt |
| Độ tập trung | RRF top-score gaps, phân bố ranks | Không gọi là confidence đã calibrated |
| Query | Token count, đặc điểm từ vựng | Tránh proxy ngôn ngữ/domain không kiểm soát |
| Retriever riêng | Score margins của từng nhánh | Fit scaling chỉ trên train, không trộn raw scales tùy ý |

### 23.2. Hành động

| Hành động | Xử lý |
|---|---|
| A0 | Lấy top-5 RRF, bỏ CE |
| A1 | CE top-10 RRF, lấy top-5 trong 10 |
| A2 | CE top-20 RRF, lấy top-5 trong 20 |
| A3 | CE top-50 RRF, lấy top-5 trong 50 |

Fusion phải giữ ít nhất 50 candidates khi có đủ corpus. Candidate depths của retrievers phải được cố định/ghi lại. Không trộn CE scores của passages đã chấm với RRF scores của passages chưa chấm để sort một danh sách chung thiếu quy tắc.

### 23.3. Tạo supervision

Chạy các actions offline trên dữ liệu router hợp lệ, ghi F1, evidence retention và cost. Hai hướng thiết kế có thể chọn:

- Label action rẻ nhất đạt chất lượng theo ngưỡng đã định; quy định tie-break và fallback khi không action đạt.
- Học dự đoán utility/outcomes từng action rồi chọn theo quality/cost constraint.

Nếu đưa evidence retention vào target, định nghĩa nhãn và mức đầy đủ qrels; không giả định gold passage duy nhất là mọi bằng chứng đúng. Utility weights và thresholds chọn bằng dev, có ablation. Không đảm bảo mục tiêu huấn luyện biến thành ràng buộc chất lượng được bảo đảm lúc inference.

### 23.4. Thuật toán khởi đầu

1. Heuristic overlap/score-margin thresholds làm baseline dễ diễn giải.
2. Decision tree hoặc multinomial logistic regression.
3. Gradient-boosted trees hoặc predictors theo action nếu dữ liệu/tài nguyên cho phép.

Model router phổ biến không phải novelty. Khác biệt phải đến từ formulation, objective/features, protocol hoặc findings có bằng chứng đối chiếu.

### Kiểm tra cài đặt

- Inference router không được đọc gold answers/qrels hoặc CE/reader outputs của actions chưa chạy.
- Chi phí trích features và router được tính trong latency.
- Feature order/scaling/schema train và serving giống nhau.
- So action trace với config, kiểm tra skip CE thực sự không gọi CE.
- Không chạy tất cả actions ở benchmark online rồi chỉ báo thời gian action được chọn.

### Đầu ra và nghiệm thu

Method specification, feature schema, label generator, router checkpoint, config và inference traces. Chạy được chỉ từ query/candidates trước reranking; có phân biệt chi phí offline label creation và online serving.

## Giai đoạn 24 — Baselines mạnh và công bằng

### Các nhóm đối chiếu

| Baseline | Mục đích |
|---|---|
| BM25 + reader | Lexical anchor |
| Dense + reader | Semantic anchor |
| Hybrid + reader | Fusion anchor |
| Hybrid + CE ở nhiều fixed depths | Đối thủ chính; dựng static Pareto frontier |
| Query-length router | Kiểm tra tín hiệu rẻ |
| Overlap/score-margin heuristic | Kiểm tra lợi ích model học |
| Phương pháp adaptive gần nhất | Đối chiếu prior art |
| Random allocation cùng budget trung bình, nếu phù hợp | Kiểm tra quyết định có thông tin |
| Oracle trên actions đã thử | Mốc phân tích offline |

### Quy tắc cài đặt

- Cùng corpus/query sets/reader và tiền xử lý trong thí nghiệm chính.
- Đối chiếu nhiều mức budget, không chỉ fixed top-20 yếu.
- Tính overhead cho mọi router/baseline tương ứng.
- Cùng tuning budget hoặc công bố sai khác.
- Adapt prior method cho extractive QA phải mô tả phần thay đổi.
- Nếu code/model prior work không có hoặc không tái hiện được, báo rõ giới hạn; không âm thầm thay bằng heuristic rồi gắn tên bài gốc.
- Chọn điểm vận hành bằng dev; test dùng đánh giá các lựa chọn đã chốt.

### Đầu ra và nghiệm thu

Baseline registry với paper/code revision, configs, tuning logs, dev curves. Có nearest adaptive baseline phù hợp và static frontier mạnh để đánh giá contribution.

## Giai đoạn 25 — Thí nghiệm chính và kiểm định

### Ma trận thí nghiệm

| Mã | Nội dung | RQ |
|---|---|---|
| E1 | Predictive utility của disagreement/features | RQ1 |
| E2 | Clean QA quality–cost | RQ2 |
| E3 | Paired perturbations đã kiểm tra nghĩa | RQ3 |
| E4 | Domain transfer theo protocol | RQ3 |
| E5 | Thay encoder/reader, ít nhất khi cần cho tuyên bố tổng quát | RQ2/RQ3 |

### Metrics

- EM/F1; evidence retention và source/span validity.
- Candidate Hit/Recall@N; MRR nếu phù hợp qrels.
- CE pairs/query và thực tế token/windows được chấm.
- End-to-end latency mean, p50, p95; feature/router timings riêng.
- RAM/VRAM, startup và offline index/training costs tách riêng.
- Tần suất chọn actions, action regret so với offline oracle khi có nhãn.

### Độ tin cậy

- Chạy nhiều seeds khi có stochastic training; số seeds theo protocol và ngân sách, ví dụ 3–5 nếu khả thi.
- Paired bootstrap chênh lệch trên cùng query; cluster theo article hoặc original question khi nhiều variants phụ thuộc.
- Không coi các variants cùng nguồn hoặc nhiều seeds là các mẫu độc lập tùy ý.
- Tiền định non-inferiority margin; kiểm tra cận CI phù hợp. “Không significant” không chứng minh tương đương.
- Tiền định primary comparisons; hiệu chỉnh nhiều kiểm định khi cần.
- Báo effect size, CI, sample size và tất cả seeds, không chỉ run tốt nhất.

### Đo hiệu năng

Cùng hardware/precision/batching/concurrency/cache/warm-up. GPU timing cần xử lý bất đồng bộ đúng khi đo. Báo cả CE-call reduction và end-to-end latency; CE count giảm chưa chắc thời gian giảm vì padding, batch efficiency và router overhead.

### Đầu ra và nghiệm thu

Main results, uncertainty estimates và curves; từng RQ có kết luận được/không được hỗ trợ. Không sửa phương pháp theo test rồi báo như một phép đánh giá chưa được xem.

## Giai đoạn 26 — Ablation, cơ chế và thất bại

### Ablations

- Bỏ disagreement, chỉ query features.
- Chỉ features một retriever.
- Heuristic vs learned router.
- Relevance-based target vs downstream QA-based target.
- Bỏ evidence objective nếu phương pháp có dùng.
- Thay action set/candidate depth trong phạm vi protocol.
- Nếu thêm mining, tách tác động mining khỏi router bằng thí nghiệm riêng.

### Phân tích router

```text
Chọn ngân sách thấp
   ├── Giữ đáp án đúng → Tiết kiệm có ích
   └── Sai, action đắt có thể sửa → Cấp thiếu tài nguyên

Chọn ngân sách cao
   ├── Chất lượng tốt hơn → Chi phí có ích
   └── Không tốt hơn     → Chi phí chưa tạo lợi ích
```

Không giả định ngân sách cao luôn sửa được lỗi. Candidate pool thiếu evidence thì router không thể khắc phục chỉ bằng reranking.

### Phân tích lỗi QA

Chỉ phân loại đáp án cuối sai khi đủ bằng chứng: evidence vắng candidates; evidence mất khỏi reader inputs; evidence có nhưng reader sai. Unjudged evidence hoặc đáp án đúng từ nguồn không phù hợp được đánh dấu kiểm tra riêng.

### Kiểm tra cơ chế

- Router có chỉ học độ dài/domain proxy không?
- Disagreement có dự báo marginal benefit hay chỉ tương quan với lỗi?
- Nhóm nào bị suy giảm trong khi mean toàn bộ tăng?
- Nhiễu gây lỗi retriever, router, CE hay reader?
- Lợi ích còn khi thay model/miền hay phụ thuộc một checkpoint?
- Gold-context reader/oracle evidence injection dùng để tách bottleneck, chỉ offline.

### Đầu ra và nghiệm thu

Ablation tables, error taxonomy, representative successes/failures và limitations. Các diễn giải nhân quả phải dựa kiểm soát phù hợp; vài ví dụ hoặc correlation chưa đủ chứng minh cơ chế.

## Giai đoạn 27 — Tái lập và công bố artifacts

### Nội dung cần có

- Data processing, split IDs, corpus/qrels manifests và checksums.
- Feature extraction, action definitions, label-generation code.
- Model/tokenizer/index revisions, checkpoint hashes.
- Train/inference/evaluation configs và dependency lock.
- Router decisions, per-query predictions, evidence/timing traces.
- Perturbation generation code, seeds, validity annotations và rejection rules.
- Scripts tái tạo bảng/hình; mô tả hardware và chi phí chạy.
- Giấy phép và cách tải/tạo lại thành phần không được phân phối trực tiếp.

### Kiểm tra độc lập

```text
Môi trường sạch → Chuẩn bị dữ liệu → Tải/tạo artifacts
              → Chạy smoke experiment → Tạo lại bảng/hình
              → Đối chiếu kết quả và giải thích sai khác
```

Smoke run kiểm tra pipeline, không thay full benchmark. Không làm tròn/biến đổi dữ liệu để ép khớp số. Ghi sai khác do nondeterminism/hardware và phạm vi tái lập đã thực sự kiểm tra.

### Đầu ra và nghiệm thu

Reproducibility package, README, environment specs và reproduction log. Một lượt chạy độc lập kiểm tra được claims chính trong phạm vi tài nguyên đã nêu.

## Giai đoạn 28 — Viết bài, rà soát và chọn venue

### Cấu trúc bản thảo

```text
Introduction → Vấn đề, động cơ, đóng góp có bằng chứng
Related Work → Công trình gần nhất và khác biệt
Method → Features, actions, objective, router và complexity
Experimental Setup → Data, baselines, metrics, protocol, hardware
Results → Trả lời RQs
Analysis → Ablation, robustness, failures
Limitations → Data exposure, qrels, models, domains, resources
Conclusion → Kết luận trong phạm vi dữ liệu đo
Supplement → Chi tiết tái lập và bổ sung
```

### Kiểm tra claims

- Mỗi contribution gắn bảng/hình/phân tích hỗ trợ.
- Phân biệt oracle vs deployable policy, CE count vs actual latency.
- Báo cả nhóm/hardware/miền không cải thiện.
- Không nhận first/SOTA khi chưa review và đối chiếu đủ.
- Không nhận “evidence-preserving” nếu chỉ tăng answer F1 mà chưa đo evidence retention đáng tin cậy.
- Phân biệt hypothesis bác bỏ với lỗi triển khai; kết quả âm có giá trị nhưng cần framing và bằng chứng phù hợp.
- Kiểm tra citations/attribution, data/model licenses và chính sách disclosure của venue.

### Định hướng nơi công bố

| Trọng tâm đóng góp | Hướng cân nhắc |
|---|---|
| Ranking, budget allocation, efficiency | SIGIR/ECIR hoặc venue IR phù hợp |
| QA, ngôn ngữ, robustness và end-task behavior | ACL/EMNLP hoặc venue NLP phù hợp |
| Nghiên cứu IR mở rộng, phân tích toàn diện | Journal như ACM TOIS/IP&M sau khi kiểm tra scope hiện hành |

Đây là định hướng fit, không bảo đảm được nhận. Tham khảo [SIGIR full-paper scope](https://sigir2027.org/pages/submit-full.html); kiểm tra lại CFP, track, thời hạn và chính sách vào thời điểm nộp.

Q1/Q2 là quartile tạp chí theo hệ thống/năm/category, không phải tên mức chất lượng chung cho mọi hội thảo. Không gắn quartile hiện tại cho journal khi chưa xác minh theo yêu cầu cơ quan. Tham khảo giải thích [Clarivate về rank/quartile theo category](https://clarivate.com/academia-government/blog/a-primer-on-ties-in-the-jcr/).

### Đầu ra và nghiệm thu

Manuscript, supplementary materials, reproducibility package, claim–evidence checklist và danh sách venue được xác minh. Việc gửi bài thực hiện khi người dùng yêu cầu; tài liệu này chỉ lập kế hoạch.

## Lộ trình và điểm quyết định

```text
19 Pilot → 20 Literature/gap → 21 RQ/protocol → 22 Data
                                             ↓
23 Method → 24 Baselines → 25 Experiments → 26 Analysis
                                             ↓
                                      27 Reproduction → 28 Paper
```

Literature và kiểm tra tính mới được cập nhật trong suốt dự án, đặc biệt trước chốt method và trước nộp bài.

| Gate | Điều kiện | Nếu chưa đạt |
|---|---|---|
| Sau GĐ20 | Vấn đề có bằng chứng, gap cụ thể | Thu hẹp/đổi hướng hoặc chọn contribution phân tích |
| Sau GĐ24 | Method khác biệt, baselines đủ mạnh | Sửa formulation và đối chiếu; không tăng độ phức tạp để che thiếu novelty |
| Sau GĐ26 | Claims được dữ liệu hỗ trợ, biết giới hạn | Báo kết quả âm/hạn chế hoặc thực hiện kiểm chứng độc lập cần thiết |
| Sau GĐ28 | Bản thảo đúng bằng chứng và tái lập được | Sửa claims/artifacts trước submission |

## Kiến thức cần học theo thứ tự

1. Experimental design và leakage → đảm bảo dữ liệu/đối chiếu đúng.
2. Quality–cost trade-off, Pareto frontier và oracle → biết có vấn đề đáng tối ưu.
3. Literature reading, gap và claim–evidence mapping → tránh nhầm tích hợp với novelty.
4. Feature engineering, classification/regression và calibration → xây router dễ kiểm tra.
5. Contrastive/mining nếu mở rộng, tách khỏi thí nghiệm router.
6. Paired/cluster bootstrap, CI, non-inferiority → diễn giải đúng kết quả.
7. Robustness, domain shift, ablation → hiểu cơ chế và giới hạn.
8. Reproducibility và scientific writing → công bố đủ bằng chứng.

## Checklist V2

- [ ] Pilot cho thấy tiềm năng có ý nghĩa trong action space đã chọn.
- [ ] Literature matrix xác minh, gần nhất và có phản chứng.
- [ ] Novelty viết như ứng viên cho tới khi đối chiếu đầy đủ.
- [ ] RQs/hypotheses/protocol tiền định trước final test.
- [ ] Splits/checkpoints/router labels đã audit leakage.
- [ ] Router chỉ dùng features có ở thời điểm ra quyết định.
- [ ] Static frontier và nearest adaptive baselines được đối chiếu.
- [ ] Tính đủ chi phí, có uncertainty và phù hợp giả định thống kê.
- [ ] Robustness variants hợp lệ, không coi subgroup analysis là toàn bộ robustness.
- [ ] Có ablation, negative cases, oracle controls và limitations.
- [ ] Gói tái lập qua một lượt chạy kiểm tra độc lập.
- [ ] Manuscript chỉ khẳng định điều kết quả hỗ trợ.

## Ghi chú nguồn và hỗ trợ AI

Bản kế hoạch được tổng hợp với hỗ trợ AI từ cuộc trao đổi và các nguồn sơ cấp đã tra cứu. Đây chưa phải literature review có độ bao phủ đủ để xác nhận novelty. Các nhóm cần đọc bài gốc, kiểm tra lại trạng thái xuất bản, cập nhật nguồn gần thời điểm nộp, tự kiểm chứng mã/kết quả và tuân thủ chính sách disclosure của venue.
