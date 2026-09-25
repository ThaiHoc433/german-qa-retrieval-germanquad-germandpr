# Data model — RAG V2

Nguồn: [spec](spec.md). Đây là schema thiết kế cho src/schemas_rag.py, không sửa
PredictionRecord V1. JSON dùng null; Python dùng None. Mọi field có mặc định/null
được validator xác nhận, không suy trạng thái bằng truthiness của string.

## 1. Config và identity

| Entity/field | Kiểu và ràng buộc |
|---|---|
| answerer | extractive / rag / closed_book / oracle_rag |
| model_backend | open / claude cho LLM; null cho extractive |
| retriever, tier, reranker_enabled | rag/extractive: bm25 hoặc dense, tier đã build, bool; closed/oracle: null/null/false |
| retrieval_k, reader_k | 20/5 cho rag/extractive; null khi không áp dụng; oracle_limit=5 riêng |
| prompt | version, repo-relative path, SHA-256 byte file; null cho extractive; closed_book dùng prompt riêng |
| open model | model_id, model_revision, tokenizer_revision, chat_template_sha256, quantization_config, generation_config |
| Claude | canonical model_id, response_model_id, api_version, sdk_version, citations_enabled, request_id |
| budget | operational_context_limit, output_tokens, safety_margin_tokens, count_method/version; tất cả khóa trước test |
| capability | available, reason_code, checked_at; readiness snapshot, không bảo đảm endpoint sống mãi |

Read-only demo giữ request_id/effective_config, không giả run_manifest. Batch dùng
run_id riêng, config hash và manifest. API key/token và URL có secret không được xuất
trong /api/config, prediction hoặc manifest. Endpoint fingerprint là hash cấu hình
đã bỏ credential, không hash secret để công bố.

## 2. Context và Citation

ContextPassage: passage_id, title (nullable), text, selection_rank (1-based),
prompt_label (P1..Pn), source_document metadata nếu có. Model chỉ thấy label/title/text.
Giữ nguyên source text cho offset. RetrievedItem có passage_id, rank, score hữu hạn.
Selected IDs phải thuộc top20 trước selection ở rag; oracle lấy unique sorted gold IDs.

OracleAudit có oracle_selected_passage_ids và oracle_omitted_passage_count; không
suy corpus fingerprint từ tier=null. Manifest oracle ghi corpus chứa gold cụ thể.

CitationDetail:

| Field | Ràng buộc |
|---|---|
| passage_id | ID được map từ đúng source model chọn, không sửa sang source khác |
| quote | quote model đưa sau loại cặp nháy ngoài/trim; raw_quote giữ nguyên trong raw output |
| quote_start/end | code-point offset [start,end) trong passage gốc; null nếu không match |
| quote_match_method | exact / normalized / null |
| quote_occurrence_count | số occurrence tại mức match thành công; 0 nếu không tìm thấy |
| short_answer_start/end | offset gốc trong passage, không phải tương đối quote; null nếu không khớp chuỗi chính xác |
| api_char_start/end | nullable; metadata API để audit |
| api_offset_agrees | bool khi đủ thông tin đối chiếu; null nếu không có offset/match |

citations luôn là list 0 hoặc1 CitationDetail. multiple_sources: citations=[], giữ
raw_citation_refs có thứ tự và trùng lặp, cùng raw_citations/raw_generation cho audit.
out_of_range: không bịa passage_id, citations=[] và giữ ref. wrong_passage giữ ID
được model trích với quote offsets=null; có thể ghi matched_other_labels trong audit,
không đổi prediction nguồn. Missing giữ raw fields nhưng citations=[].

Verified chỉ nói quote có trong passage đã trích. Short answer support là phép kiểm
token riêng (FR-008). Offset và support không được suy lẫn nhau. Với normalized quote,
phải dùng bản đồ code points để tìm vị trí trên source; không cộng offset trên chuỗi
đã collapse whitespace. Bằng chứng chưa gán gold vẫn có thể verified nhưng citation
accuracy theo nhãn bằng0.

## 3. AnswerResult và PredictionRagRecord

AnswerResult chứa generation_status, error_type, answered, short_answer, answer,
citation_status, answer_supported, citations, raw_citation_refs, raw_generation,
has_extra_text, backend audit và context audit. generation_status mô tả toàn bước
answerer, kể cả chuẩn bị context/parser; không phải chỉ HTTP call tới LLM.

| Trường hợp | generation_status | answered | citation_status | answer_supported | answer/short_answer |
|---|---|---|---|---|---|
| Generation/preparation/parser lỗi | error | null | null | null | null |
| KEINE ANTWORT hợp lệ | success | false | null | null | null |
| closed_book có đáp án | success | true | null | null | string |
| rag/oracle quote verified, tokens match | success | true | verified | true | string |
| rag/oracle quote verified, tokens không match | success | true | verified | false | string |
| rag/oracle citation khác verified | success | true | một lỗi citation | null | string |

Reader adapter giữ span và status V1; khi có span, quote=short_answer=answer=span,
citation verified theo offset được kiểm trực tiếp. answer_supported của Reader
theo bằng chứng span, không dùng phép token RAG để thay đổi kết quả V1. Không có span
thì answered=false. Read error đi vào execution_failure theo V1, không generation_failure.

PredictionRagRecord bổ sung:

- schema_version=rag-v2-draft, query_id, question_key, question, dataset/split rõ ràng.
- execution_status=ok/error: chỉ lỗi pipeline retrieval/selection/read/verify hoặc
  hạ tầng nội bộ; error_stage và execution_error_type riêng. Generation error vẫn
  có execution_status=ok để không bị nuốt vào execution_failure.
- answer_result nullable nếu execution lỗi trước khi trả AnswerResult; giữ retrieved,
  selected và latency hoàn thành. Canonical answer fields null khi execution lỗi;
  raw generation còn có thể giữ trong audit nếu verifier lỗi sau generation.
- retrieved, selected, passage_texts từ corpus đã khóa; oracle retrieved=[],
  closed_book cả retrieved/selected=[].
- run_mode=research/ask/dev_evaluation; run_id bắt buộc research, null cho web;
  request_id cho web, effective_config mọi mode.
- latency_ms: retrieval, rerank, read, prepare_context, generate, verify_citation,
  total. Stage chưa chạy/không áp dụng là null; không nhập số0 giả vào mean.
- generation_attempted bool và token_count audit: count call không phải generate call.
- per_query metrics chỉ được Evaluation/web-dev thêm: EM,F1,bucket,final_correct,
  gold_in_retrieval_pool,gold_in_selected. Ask không có gold/bucket/EM/F1.

Illegal trạng thái do file corrupted là validation error khiến Evaluation dừng;
output model sai format là một record lỗi generation hợp lệ. Không tự chữa record.

## 4. Run manifest extension

RAG manifest ở results/rag/<run_id>/run_manifest.json, không ghi schema V2 vào manifest
V1 hiện có. Reuse thuật toán hash/UUID của V1 qua helpers, thêm:

- run_id/run_uuid, schema_version, answerer, backend, config_sha256, seed,
  dataset/split, N_eval definition, protocol_revision, support_rule_revision.
- gold file/path/hash, corpus + corpus_manifest path/hash khi dùng corpus; null cho
  closed_book (gold vẫn dùng để chấm); oracle tier=null nhưng corpus path/hash có thật.
- frozen candidate artifact path/hash/run_id; selected artifact path/hash hoặc
  deterministic selection config; null khi không dùng retrieval.
- checkpoints/bundle hash của thành phần thực dùng, model/tokenizer revisions,
  prompt/template hashes, quantization/decoding/context config, environment lock hash.
- warmup dev path/hash và danh sách10 query_id sorted, batch1, một lượt đo chính thức,
  CPU/GPU/library versions, measurement_mode (fresh/reused_candidates), stage timing scope.
- predictions_path, metric input fingerprint; đường dẫn resolve theo thư mục manifest.
- API model là provider identity, không bịa checkpoint weights SHA hoặc seed API.

Manifest ghi trước query đầu tiên, kể cả prediction rỗng. Resume giữ run_id và chỉ
xử lý query chưa có record; error record đã ghi không tự retry. Fingerprint mismatch,
duplicate query_id hoặc ID ngoài corpus dừng. Sửa bug cần run mới có predecessor/reason,
không overwrite kết quả cũ.

## 5. Metric output

Mỗi metric ratio có numerator, denominator, value nullable. N_eval lấy từ gold file,
không lấy len(predictions). Missing/error/no-answer nhận EM=F1=0. Empty gold đếm riêng.
Chất lượng có thể chấm cả verified=false, vì final_correct không suy từ citation.

Bucket sets: rag7, oracle5, closed_book4; unsupported bucket cho baseline hiển thị
N/A, không vẽ như một số0 đã được đo. Citation metrics closed_book N/A. Oracle/passages
khác nhau được ghi rõ trong bảng đối chiếu Reader.

Timing chỉ dùng record execution_status=ok/generation_status=success có measurement
hợp lệ. Với extractive dùng status V1. Từng stage trả n,mean,p50,p95,device/scope.
No-answer thành công vẫn thuộc mẫu latency; format/context errors không thuộc mẫu.
prepare_context có token count network và render/count CPU; generate đo riêng call
và inference. verify_citation CPU đo cả khi phát hiện quote lỗi, nếu generation success.
