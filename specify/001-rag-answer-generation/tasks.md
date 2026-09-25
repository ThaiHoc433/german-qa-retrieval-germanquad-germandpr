# Tasks — RAG answer generation V2

Ngày lập: 2026-09-12. Nguồn chuẩn: [spec](spec.md), [plan](plan.md),
[data model](data-model.md), [pipeline contract](contracts/pipeline.md),
[API contract](contracts/api.md), [research](research.md), [quickstart](quickstart.md)
và [constitution](../../.specify/memory/constitution.md).

## Cách thực hiện và nghiệm thu

Đây là danh sách công việc, **không phải báo cáo đã triển khai**. Mọi checkbox chưa
được đánh dấu; file hiện có không tự động đồng nghĩa với task hoàn tất. Đọc code mới
nhất trước khi sửa, bổ sung test cho gap rồi sửa tối thiểu; nếu hành vi đã đúng thì
giữ code và ghi bằng chứng test. Không ghi đè theo implementation plan cũ.

Paths dưới đây tính từ repo root. Tài liệu do Spec Kit sinh nằm trong `specify/`;
source/test/artifact thực thi vẫn ở vị trí theo plan. Không sửa spec V1.1 đóng băng.
Mỗi task ghi đầu vào/phụ thuộc bằng `Deps`, file đích và điều kiện đạt. Task test
chạy trước implementation tương ứng; không cố tạo test đỏ giả cho code đã đúng.
Chạy test bằng interpreter đã xác minh, lưu lệnh và kết quả thật vào
`specify/001-rag-answer-generation/validation/implementation.md` khi thực hiện.

`[P]` chỉ cho phép song song trong các wave nêu cuối tài liệu, sau khi mọi Deps đã
xong; không cho hai người sửa cùng file. Không chạy API trả phí, tải model hay test
benchmark chỉ từ việc duyệt danh sách này. Gate thiếu bằng chứng phải để unchecked.

## Phase 1 — Setup (M0)

- [ ] T001 Audit `src/answerer.py`, `src/citation.py`, parser/clients/prompts và tests hiện có; ghi bảng reuse/gap, git status, interpreter và gate V1 trong `specify/001-rag-answer-generation/validation/inventory.md`. Deps: không. Đạt khi phân biệt code tồn tại với hành vi đã test, không sửa file người dùng ngoài scope.
- [ ] T002 Tạo dependency profiles `requirements-rag-core.txt`, `requirements-rag-api.txt`, `requirements-rag-gpu.txt` và hướng dẫn môi trường trong `specify/001-rag-answer-generation/quickstart.md`. Deps: T001. Pin phiên bản tương thích bằng môi trường riêng; core chạy không torch/SDK/API key, GPU lock ghi từ smoke thật, không đổi môi trường V1.
- [ ] T003 Tạo fixtures dùng chung và chặn mạng mặc định cho suite RAG trong `tests/rag_fixtures.py` và `tests/test_rag_offline.py`. Deps: T002. Fake client/reader/corpus cho kết quả xác định; test chứng minh network không được gọi và fixture không chứa dữ liệu test thật hay secrets.

## Phase 2 — Foundational schemas (M1)

Đầu ra độc lập: các record hợp lệ round-trip; record sai bị từ chối, không sửa ngầm.

- [ ] T004 Viết contract tests trong `tests/test_schemas_rag.py` cho toàn bộ bảng field/trạng thái ở `data-model.md` §1–3. Deps: T003. Bao gồm enum sai, null sai nhánh, offsets ngoài text, selected không thuộc retrieved và metadata lộ secret; round-trip mọi nhánh hợp lệ.
- [ ] T005 Tạo `src/schemas_rag.py`, reuse dataclass hiện có bằng adapter thay vì duy trì hai schema mâu thuẫn. Deps: T004. Config giữ "extractive / rag / closed_book / oracle_rag", backend "open / claude cho LLM; null cho extractive", baseline "closed/oracle: null/null/false", k "20/5 cho rag/extractive; null khi không áp dụng; oracle_limit=5 riêng"; prompt "version, repo-relative path, SHA-256 byte file; null cho extractive; closed_book dùng prompt riêng". Validate đủ identity/budget/capability theo §1; T004 pass.
- [ ] T006 Hoàn thiện context/citation/answer/prediction validators trong `src/schemas_rag.py`. Deps: T005. Giữ "title (nullable)", "selection_rank (1-based)", "prompt_label (P1..Pn)", score hữu hạn; quote offsets "code-point offset [start,end) trong passage gốc; null nếu không match", method "exact / normalized / null", occurrence "0 nếu không tìm thấy", short offsets "offset gốc trong passage, không phải tương đối quote; null nếu không khớp chuỗi chính xác", API offsets nullable/agrees bool hoặc null. "citations luôn là list 0 hoặc1 CitationDetail"; multiple_sources giữ raw refs/trùng lặp; error/null và success branches đúng bảng §3, execution error không thành generation error. Giữ "schema_version=rag-v2-draft", "run_mode=research/ask/dev_evaluation; run_id bắt buộc research, null cho web", stage chưa chạy null, generation_attempted bool; toàn T004 pass, không sửa schema V1.

## Phase 3 — US1: Giữ nguyên extractive (P1; M5)

Mục tiêu: Reader đi qua interface mới nhưng không đổi đáp án. Nghiệm thu độc lập:
fake Reader luôn chạy được; checkpoint smoke là gate riêng, không cản US2 trên fixture.

- [ ] T007 [P] [US1] Viết adapter regression tests trong `tests/test_extractive_answerer.py`. Deps: T006. Fixture span/no-answer/read-error đối chiếu Reader trực tiếp với wrapper theo text, passage ID, offsets, status; quote bằng span, không áp dụng token-support RAG để đổi V1.
- [ ] T008 [US1] Thêm ExtractiveAnswerer vào `src/answerer.py` bằng dependency injection, giữ LLMAnswerer đang có. Deps: T007. Có span → verified/support true theo offset nguồn; không span → answered=false; read error thuộc execution; T007 pass, không import model khi chạy fake.
- [ ] T009 [US1] Viết pipeline regression tests trong `tests/test_rag_pipeline.py`. Deps: T008. Retriever top20 → selection top5 → Reader đúng thứ tự, rerank-off cũng đọc5; lỗi từng stage giữ kết quả đã xong, lõi không ghi file.
- [ ] T010 [US1] Tạo lõi inject được `src/rag_pipeline.py` cho extractive và các stage adapter. Deps: T009. Trả PredictionRagRecord đúng schema, checkpoint metadata và timing stage thực, không tự ghi manifest; T009 pass bằng fake.
- [ ] T011 [US1] Tạo/chạy kiểm chứng Reader thật qua `scripts/check_extractive_regression.py`, lưu bằng chứng tại `specify/001-rag-answer-generation/validation/v1-regression.md`. Deps: T010; gate Reader Chặng8, Integration Chặng9 và artifact V1 đã nghiệm thu. Cùng checkpoint/input/hardware cho span/ID/offset/status giống Reader trực tiếp; không đòi logits bit-for-bit giữa hardware. Thiếu checkpoint ghi chưa chạy, không đánh dấu xong.

## Phase 4 — US2: Sinh câu tiếng Đức có citation (P1; M1–M3)

Nghiệm thu độc lập: fake clients phủ parser, sáu citation states, null invariants,
budget và call count; không cần retriever/Reader thật hoặc mạng.

- [ ] T012 [P] [US2] Bổ sung `tests/test_citation.py`: exact/NFC/whitespace với mapping source, quote rỗng, overlap occurrences, dấu nháy ngoài, wrong passage, offset short answer tuyệt đối, API offset lệch. Deps: T006. Einstein/Ein, normalized-empty và normalized-only có expected rõ theo FR-008 provisional; không fuzzy match.
- [ ] T013 [P] [US2] Bổ sung grammar tests ở `tests/test_answer_parser.py`. Deps: T006. Open context4 dòng, closed/Claude2 dòng, nhãn duplicate/sai case, extra text, no-answer đúng mode, thiếu QUELLE/ZITAT → missing, nhiều refs giữ thứ tự/trùng lặp; no-answer kèm nguồn → format_violation.
- [ ] T014 [US2] Sửa `src/citation.py` theo T012: exact trước normalized toàn chuỗi có source map, đếm cả overlap ở mức match thành công và lấy offset nhỏ nhất. Deps: T012. Giữ source model chọn, wrong_passage offsets null; multiple_sources citations=[] và raw đủ; token support là dãy token chuẩn hóa không rỗng liên tiếp, offsets chỉ exact source. T012 pass, không sửa normalize_answer V1.
- [ ] T015 [US2] Sửa `src/answer_parser.py` theo grammar từng mode/backend, giữ raw/has_extra_text. Deps: T013. Không repair/JSON/fuzzy parse; mọi T013 pass, không biến citation missing thành generation error.
- [ ] T016 [US2] Viết prompt/context tests ở `tests/test_prompt_loader.py`. Deps: T014, T015. Hash byte file, ID không vào model, thứ tự P1..Pn, ≤5 không pad, closed context rỗng, oracle unique lexical IDs lấy5 và audit omitted; passage giữ nguyên text, chỉ dẫn trong passage là dữ liệu.
- [ ] T017 [US2] Sửa `src/prompt_loader.py` và `src/prompts/answer_de_v1.txt`, `src/prompts/answer_de_closed_book_v1.txt`, `src/prompts/answer_de_claude_v1.txt`. Deps: T016. Prompt tiếng Anh yêu cầu trả tiếng Đức, render theo mode; oracle dùng with-context prompt, giữ identity/hash và audit theo T016; không tạo hơn3 bộ prompt để tune dev.
- [ ] T018 [US2] Viết `tests/test_context_budget.py` và bổ sung `tests/test_llm_clients.py` với fake transport/count client. Deps: T017. Đúng budget gọi generate1 lần; vượt1 token gọi0; token-count khác generation; SDK timeout/refusal/rate-limit không retry; Claude ghép tất cả text blocks không thêm newline, lấy mọi citation object kể cả trùng nguồn.
- [ ] T019 [US2] Tạo `src/context_budget.py` và chuẩn hóa interface fake/open/Claude trong `src/llm_clients.py`. Deps: T018. Budget input+output+margin khóa theo config; không truncate/drop/retry; overflow giữ context audit và generation_attempted=false. Lazy SDK import, max_retries=0, timeout rõ; metadata model/revision/tokenizer/template/quantization/API đúng data-model, không lộ secret; T018 pass.
- [ ] T020 [US2] Viết trạng thái integration tests ở `tests/test_answerer.py`. Deps: T019, T014, T015. Cả open/Claude × rag/oracle/closed (dù research API chỉ chọn rag), lỗi chuẩn bị/parser/generate, verifier runtime error, multiple_sources, no-answer và raw audit; assert null invariants và call counts.
- [ ] T021 [US2] Ghép budget→client→parser→citation trong `src/answerer.py`. Deps: T020, T008. Giữ ExtractiveAnswerer; closed_book không gọi nhánh open khi backend=claude; tách generate/verify timing và nguyên nhân lỗi; T020 và T007 cùng pass.
- [ ] T022 [US2] Mở rộng `src/rag_pipeline.py` và `tests/test_rag_pipeline.py` cho các LLM answerer. Deps: T021, T010. Closed không retrieval; oracle gold chỉ từ caller được phép, tier=null nhưng provenance corpus thật; rag selected thuộc top20; lỗi verify là execution, overflow là generation; test spy không ghi manifest.
- [ ] T023 [US2] Tạo `scripts/smoke_rag.py`, `configs/rag/dev-smoke.json` và ghi hợp đồng endpoint mở trong `specify/001-rag-answer-generation/contracts/open-endpoint.md`. Deps: T022. CLI validate config trước gọi, fake mặc định tường minh, timeout/token-count/identity/no-retry kiểm bằng fake endpoint; config không chứa secret hay hash bịa.
- [ ] T024 [US2] Chạy live dev smoke khi được cấp tài nguyên/quyền qua `scripts/smoke_rag.py`; lưu `specify/001-rag-answer-generation/validation/backend-smoke.md` và lock môi trường tương ứng. Deps: T023. Xác nhận model license/revision, greedy/quantization/budget, endpoint và API response metadata thực; GPU/API chưa có thì ghi gate chưa đạt, không tự tải model/gọi trả phí.

## Phase 5 — US3: Đánh giá answerer và RQ4 (P2; M4/M7)

Nghiệm thu độc lập: arithmetic fixtures đúng, bucket phủ N_eval và manifest mismatch
bị từ chối; nghiên cứu thật có gate riêng sau V1 và dev lock.

- [ ] T025 [P] [US3] Tạo `tests/test_evaluation_rag.py` với fixture A–H trong pipeline contract và cases bổ sung format/empty gold/zero denominator/closed/oracle. Deps: T006. Kiểm EM2/8, support2/4, citation3/4, proxy1/4 là số fixture; bucket rag7/oracle5/closed4, final_correct độc lập, verified/support=false không lọt correct.
- [ ] T026 [US3] Tạo `src/evaluation_rag.py`, reuse normalize/EM/F1/percentile V1, không sửa `src/evaluation.py`. Deps: T025. N_eval từ gold; missing/error/no-answer0; ratio numerator/denominator/value nullable, denominator0 và closed citation=N/A. Citation accuracy verified và ID thuộc bất kỳ gold evidence; errors format chia N_eval; T025 pass.
- [ ] T027 [US3] Bổ sung timing tests và xử lý trong `tests/test_evaluation_rag.py`, `src/evaluation_rag.py`. Deps: T026. Chỉ execution ok/generation success với timing hữu hạn không âm; no-answer success vẫn tính, stage null không thành0; trả n/mean/p50/p95 tuyến tính/device/scope, total đo trực tiếp không cộng p95.
- [ ] T028 [P] [US3] Tạo `tests/test_rag_run.py` cho manifest-before-query, prediction rỗng, đổi1 byte gold/corpus/checkpoint/prompt, duplicate/ID ngoài corpus, resume đã có error, đường dẫn manifest-relative. Deps: T006. Fake runner đếm lượt, hash sai phải dừng trước inference/evaluation, không ghi vào manifest V1.
- [ ] T029 [US3] Tạo `src/rag_run.py` với manifest schema §4 và validators. Deps: T028. Namespace `results/rag/<run_id>/run_manifest.json`, UUID/config SHA, seed/gold/corpus/candidate/selected/checkpoint bundle/prompt/model/environment/protocol hash; closed corpus null, oracle corpus thật dù tier null; API không bịa weights hash/seed. Dùng helper V1 nếu đã có, cô lập adapter fixture nếu chưa có; không tuyên bố gate V1 đạt.
- [ ] T030 [US3] Hoàn thiện batch append/resume trong `src/rag_run.py`. Deps: T029, T022. Manifest ghi trước warmup/query; resume chỉ query thiếu record, không retry error; bugfix run mới predecessor/reason, không overwrite; timing batch1/10 dev warmup sorted, fresh hoặc reused_candidates với retrieval/rerank null đúng scope; T028 pass.
- [ ] T031 [US3] Tạo CLI `scripts/run_rag.py`, `scripts/evaluate_rag.py` và tests `tests/test_rag_cli.py`. Deps: T030, T027. Fake end-to-end sinh manifest→predictions→metrics traceable, chấm không gọi model; empty predictions vẫn chấm; lỗi fingerprint trả exit khác0.
- [ ] T032 [US3] Tạo config matrix ở `configs/rag/experiment-matrix.json`, `configs/rag/open-base-off-dev.json` và selection tests trong `tests/test_rag_selection.py`. Deps: T031. Bốn Dense rag × base/large × rerank on/off, closed1, oracle1; API thêm1 cấu hình; test chọn dev F1→EM→fixed order bằng full precision, không nhân baseline theo tier.
- [ ] T033 [US3] Tạo dev selection/lock script `scripts/lock_rag_protocol.py` và `tests/test_rag_protocol.py`. Deps: T032. Khóa candidate/checkpoint/config/prompt tối đa3 bộ/model/budget/quantization/support rule; kiểm không lấy test để chọn. FR-008 còn provisional phải bị chặn benchmark lock cho tới quyết định review được ghi, không tự duyệt.
- [ ] T034 [US3] Review fixture/dev của token-support và ghi quyết định trong `specify/001-rag-answer-generation/validation/support-rule-review.md`. Deps: T024, T033. Nêu false positives/negatives, Ein/Einstein/empty, rule revision và người duyệt; nếu cần đổi semantics mở amendment trước, không thay luật sau xem test.
- [ ] T035 [US3] Kiểm gate V1 và chạy dev selection bằng CLI, ghi `specify/001-rag-answer-generation/validation/benchmark-readiness.md`. Deps: T011, T024, T034. Yêu cầu Evaluation/shared infra Chặng6 và Integration Chặng9 thật, hash dữ liệu/candidates/checkpoints đúng; lưu cấu hình API thắng dev và protocol lock, chưa dùng test.
- [ ] T036 [US3] Chạy ma trận test6+1 bằng `scripts/run_rag.py` sau quyền chạy/tài nguyên và lock, tạo artifacts trong `results/rag/`. Deps: T035. Một lượt/config, không fallback/retry/tune từ test; extractive tái sử dụng run V1 tương ứng, thiếu đối chứng báo thiếu chứ không bịa; giữ mọi query lỗi trong N_eval.
- [ ] T037 [US3] Xuất Bảng4/Hình3 và kiểm truy vết trong `scripts/report_rag.py`, `specify/001-rag-answer-generation/validation/rq4-report.md`. Deps: T036. Rag tách rerank/tier, closed/oracle tier=N/A một hàng; ghi mẫu số, scope latency, oracle5 khác Reader oracle, API khác cả model/citation, 1 seed và gold theo nhãn; kết luận mô tả từ metrics thật.

## Phase 6 — US4: Web kiểm thử local (P3; M6)

Nghiệm thu độc lập: TestClient/fake service đủ route, UI fake ghi rõ mô phỏng; nối thật
chỉ sau gate tương ứng. Web không cần chờ RQ4 chạy xong.

- [ ] T038 [P] [US4] Viết capability/lifespan tests trong `tests/test_rag_service.py`. Deps: T006. Thiếu FAISS không vô hiệu BM25, rerank-off không cần reranker, env chỉ thu hẹp artifact đã hash đúng; checkpoint dùng chung, no secret/no research writes, missing .env không crash.
- [ ] T039 [US4] Tạo `app/service.py` và `app/main.py` với injectable lifespan/fake mode. Deps: T038, T022. Nạp mỗi tài nguyên khả dụng một lần, availability theo effective config, checked_at/reason rõ; real không fallback fake, một worker localhost, validate Host/Origin; T038 pass.
- [ ] T040 [US4] Tạo `tests/test_rag_api.py` cho GET config/dev list/dev detail, POST ask/compare. Deps: T039. Oracle public422, unknown tier422/unbuilt503, unknown dev404, split/path422, baseline null config, generation errors200/execution tách tầng; compare cùng câu và giữ kết quả bên A nếu B lỗi.
- [ ] T041 [US4] Tạo `app/routes.py` theo API contract. Deps: T040, T026. Map dev thật không dùng ID prefix; ask không lộ gold/metric; normalize closed/oracle trước availability; compare validate hai config rồi chạy tuần tự, HTTP envelope/per-side đúng; T040 pass.
- [ ] T042 [US4] Thêm inflight/timeout/no-store tests và implementation tại `tests/test_rag_api.py`, `app/service.py`, `app/routes.py`. Deps: T041. Cùng query_id+effective config hash trả429 và release finally; endpoint chết sau startup xử lý đúng giai đoạn; không retry, compare dùng chung khóa, GET inference no-store và web không tạo manifest.
- [ ] T043 [US4] Tạo UI semantic trong `app/static/index.html` và `app/static/styles.css`. Deps: T042. Form cấu hình, answer/short answer/citation, top20/selected5, dev/compare tabs, fake banner và trạng thái không chỉ màu; font≥16px, focus, reduced-motion, layout375/768/1440 không overflow.
- [ ] T044 [US4] Tạo `app/static/script.js` với ask/dev/compare tường minh theo nút bấm. Deps: T043. Không prefetch/auto retry/history; render text an toàn, offsets dùng Array.from Unicode code points, normalized-only không highlight giả; closed citation N/A, lỗi từng phía không mất kết quả còn lại.
- [ ] T045 [US4] Tạo browser checks `app/qa/check.cjs` và lưu ảnh desktop/mobile cùng checklist vào `specify/001-rag-answer-generation/validation/web/`. Deps: T044. Fake server kiểm bàn phím, nút/tab, quote Unicode ngoài BMP/HTML payload, 375/768/1440, không overlap/overflow và không ghi artifact nghiên cứu; sửa lỗi trước khi đánh dấu.
- [ ] T046 [US4] Kiểm web real bằng service đã khóa, ghi `specify/001-rag-answer-generation/validation/web-real.md`. Deps: T045, T011, T024. Chỉ config có artifact thật available; thử ask/dev/compare với quyền API nếu cần, giữ read-only, ghi backend chưa chạy thay vì coi fake là nghiệm thu thật.

## Phase 7 — Polish và bàn giao

- [ ] T047 Rà cross-artifact FR-001–FR-017/SC và chạy toàn suite offline từ `specify/001-rag-answer-generation/quickstart.md`; ghi `specify/001-rag-answer-generation/validation/acceptance.md`. Deps: T031, T033, T045. Đối chiếu null/offset/HTTP/mẫu số, V1 regression tests và frozen spec hash; live gates báo riêng, không sửa V1 ngầm.
- [ ] T048 Cập nhật `specify/001-rag-answer-generation/quickstart.md` với lệnh thực đã chạy, dependency locks, fake/real instructions và giới hạn. Deps: T047. Không còn lệnh trỏ file chưa tạo trong phần đã nghiệm thu; secrets/tài nguyên/model license được hướng dẫn rõ, không biến số fixture thành kết quả.
- [ ] T049 Bàn giao tổng trạng thái trong `specify/001-rag-answer-generation/validation/handoff.md`. Deps: T037, T046, T048. Liên kết tests/screenshots/run manifests/metrics/lock/review; mọi checkbox có bằng chứng. Chỉ tuyên bố hoàn tất V2 khi gate thật đều đạt; không tự commit/push/deploy.

## Dependencies và thứ tự giao hàng

```text
T001–T006: nền chung
  ├─ US1 fake T007–T010 ── V1 Ch8/9 ── T011
  ├─ US2 parser/citation → Answerer → pipeline → smoke T012–T024
  ├─ US3 fixture T025–T029 → pipeline/batch T030–T033
  │                         → review + V1 gates T034–T035 → benchmark/report T036–T037
  └─ US4 service fixtures T038 → pipeline/web T039–T045 → live T046
Offline acceptance T047–T048; đủ research + web live mới handoff T049
```

US1 có thứ tự ưu tiên P1 nhưng T011 không chặn US2 fixture. US2 T021 chờ T008
vì cùng sửa `answerer.py`; T022 chờ T010 vì cùng pipeline. US3 không cần US4;
US4 fake không cần benchmark US3, chỉ evaluator T026. Mỗi dependency cụ thể đã ghi
trên task; diagram là tóm tắt, không thay danh sách Deps.

### Parallel opportunities

Sau T006, có thể chọn các task test độc lập T007 (US1), T012/T013 (US2), T025/T028
(US3), T038 (US4). Chúng sửa các file khác nhau, cùng đọc schema đã ổn định.
Đây là ví dụ theo từng story, không phải yêu cầu tự spawn agent. Implementation
trong cùng module chạy tuần tự; không gắn [P] cho task đang chờ task test chưa xong.

### MVP và chiến lược thực hiện

1. T001–T010: nền tảng + MVP bảo toàn Reader qua fake adapter; T011 xác nhận MVP
   bằng checkpoint khi V1 sẵn sàng. Không gọi fake adapter là baseline thật đã đạt.
2. T012–T023: increment RAG ngoại tuyến có citation và overflow kiểm chứng được.
3. T025–T033 và T038–T045: evaluator/batch fixtures và web mô phỏng; không cần chờ
   GPU để học và kiểm thử các phần này.
4. Live smoke, review FR-008, V1 gates, dev lock rồi mới test6+1; cuối cùng bàn giao.

## Coverage và tổng số

| Yêu cầu | Tasks chính |
|---|---|
| FR-001 / FR-009: interface, record, preserve V1 | T004–T011, T020–T022 |
| FR-002: context/selection/oracle | T016–T017, T022 |
| FR-003: prompt/model provenance | T005, T017–T019, T023–T024, T029 |
| FR-004–FR-008: grammar/state/citation/support | T004–T006, T012–T015, T020–T021, T034 |
| FR-010–FR-012: metric/bucket/matrix | T025–T037 |
| FR-013–FR-014: web/capability | T038–T046 |
| FR-015–FR-017: no retry, offline tests, overflow | T003, T018–T024, T030, T042 |

49 tasks: Setup3 + Foundation3 + US1:5 + US2:13 + US3:13 + US4:9 + Polish3.
6 task test có nhãn [P]. Không có task triển khai nào được đánh dấu hoàn tất bởi
việc tạo tài liệu này. Bước kế tiếp: `speckit-analyze` kiểm chéo spec/plan/tasks,
rồi mới `speckit-implement` theo phạm vi được duyệt.
