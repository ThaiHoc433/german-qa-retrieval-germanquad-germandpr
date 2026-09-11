# German QA — Danh mục task và khuôn hướng dẫn

Ngày: 2026-09-11. Đây là danh mục phụ thuộc toàn dự án, **chưa phải tất cả thẻ thực thi**.
Chi tiết mỗi chặng được viết khi chặng bắt đầu, theo yêu cầu học từng bước của sinh viên.
[Roadmap](2026-09-11-german-qa-learning-roadmap.md) phân bổ 32 buổi, Chặng 5 có 6 buổi.
[Spec V1.1](../specs/2026-09-10-german-qa-retrieval-module-specs.md) là chuẩn kỹ thuật;
[V1.2 draft](../specs/2026-09-11-german-qa-retrieval-v1.2-amendments-draft.md) chưa hiệu lực.

## Khuôn một thẻ task sẵn sàng thực thi

1. ID/tên/trạng thái/buổi học và task phụ thuộc.
2. Đang làm gì; vì sao cần; spec reference và decision D/P liên quan.
3. Files: sửa/tạo/test, đường dẫn cụ thể; kiểm tra file đã tồn tại trước khi tạo.
4. Input: schema + fixture thật nhỏ. Output: interface/hàm/artifact cụ thể.
5. Từng bước thao tác: giải thích → test hành vi mong muốn → quan sát lỗi hiện tại →
   sửa tối thiểu → chạy lại → kiểm tra artifact. Chỉ dùng red/green khi có thay đổi code.
6. Lệnh kiểm tra, kết quả mong đợi và acceptance độc lập với score cao/thấp.
7. CPU/GPU, nơi lưu, điều kiện resume và bằng chứng hoàn tất.

Danh mục dưới chưa chốt tên/signature hàm mới. Khi viết thẻ, định nghĩa interface
consumes/produces chính xác, fixture/test cụ thể trước code; không bắt người thực hiện tự đoán.
Không giao toàn danh mục này cho agent chạy tự động khi chưa chi tiết hóa.

## Phụ thuộc toàn dự án

T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10.
Trong T5: validators trước producers; chunk/provenance trước gold; split + gold + D6 trước base build.
T6 run infrastructure trước lần chấm đầu. T8 oracle trước chọn reader checkpoint;
candidate retriever cố định trước chọn reranker. T9 mở rộng tier sau base gate.

| ID | Việc/đầu ra kiểm chứng | Phụ thuộc | Spec/nguồn |
|---|---|---|---|
| T1.1 | Khảo sát cây repo, pipeline tự vẽ | Không | Design, §0 |
| T1.2 | RQ → bảng/hình/artifact map | T1.1 | Design RQ1–RQ3 |
| T2.1 | Venv, interpreter, dependencies, baseline test log | T1.1 | Imports/code hiện có |
| T2.2 | Colab/Drive smoke môi trường, kiểm tra root Spec Kit nlp | T2.1 | §5.8/6.8, constitution |
| T3.1 | Đọc JSONL/Parquet và QA record, kiểm offset | T2.1 | §1.2/2.2 |
| T3.2 | Đọc DPR positive/negative, ví dụ nhóm | T3.1 | §1.3/2.3.f–g |
| T3.3 | Dataset card/paper/license/revision/cache inventory | T3.1 | Dataset gốc, §2.2 |
| T4.1 | Audit rows/keys/length/overlap theo dataset | T3.1–T3.3 | §2 |
| T4.2 | Requirement → code/test/artifact matrix D1–D6/P1–P4 | T4.1 | Decision log, §9 |
| T6.1 | Run identity, file/checkpoint hash, bundle hash, manifest + tests | T5.8 | §3.2/8.4/8.7 |
| T6.2 | Metric/mẫu số/QA normalization tests | T6.1 | §3.3–3.6 |
| T6.3 | Selection buckets, mode/manifest validation, CLI | T6.2 | §3.3/3.5/3.7 |
| T6.4 | BM25 base/dev retrieval-only run + metrics | T6.3 | §4 |
| T7.1 | Sampling/mask/dual encoder tests | T6.4 | §5.3/5.6 |
| T7.2 | Train smoke, save/load/resume, dev checkpoint selection | T7.1 | §5.4–5.8/P2 |
| T7.3 | Encode base, FAISS + ID map + retrieval run | T7.2 | §5.2/5.8 |
| T8.1 | Reader examples/window/span postprocessing tests | T7.3 | §6.2/6.3/6.6 |
| T8.2 | Oracle runner + reader train/checkpoint selection | T8.1,T6.1 | §6.4/8.3/P2 |
| T8.3 | Reranker pairs/sampling/subset tests | T8.2,T7.1 | §7.3/7.6 |
| T8.4 | Candidate cố định, post-rerank dev selection | T8.3,T7.3 | §7.4; draft A3 phải chốt |
| T9.1 | End-to-end orchestration, latency, lỗi, resume | T8.4 | §8 |
| T9.2 | Smoke 4 config + oracle, lock config, full base gate | T9.1 | §8.6/§9 |
| T9.3 | Stream/freeze distractor, mid/large, matrix benchmark | T9.2,T5.7 | §2.3.d/2.6/8.6 |
| T10.1 | RQ tables/figures + review/giới hạn | T9.3 | Design RQ, §3.7 |
| T10.2 | Reproduction guide, report, website cập nhật có nguồn | T10.1 | Constitution IV/VII |

## Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi

Các file mới là đề xuất kiến trúc; không tạo module trùng trách nhiệm của CorpusBuilder hiện có.
Khi bắt đầu chặng, chốt việc di chuyển/reuse và cập nhật imports trong thẻ task trước sửa code.

| ID | Việc | File chính dự kiến | Test/gate chính | Phụ thuộc; buổi |
|---|---|---|---|---|
| T5.0 | Validators V1.1: Passage/QA/Training/Prediction | sửa src/schemas.py; tests/test_schemas.py | fields k, provenance, null, evidence/schema hợp lệ | T4.2; 11 |
| T5.1 | Chunk, parent_document_id, offsets và title | sửa src/corpus.py; tạo tests/test_corpus.py | substring chunk đúng, chunk cuối; title nguồn | T5.0; 11 |
| T5.2 | Split D1 theo nguồn dòng | sửa src/splits.py; tạo tests/test_splits.py | train trùng test bị loại, test gốc giữ, ổn định | T5.0; 12 |
| T5.3 | QA evidence offsets + đầy đủ references/gold rỗng | tạo src/qa_records.py; tests/test_qa_records.py | text[start:end], nhiều overlap evidence, record rỗng giữ | T5.1,T5.2; 13 |
| T5.4 | Dedup collision fallback/report mơ hồ | src/qa_records.py; tests/test_qa_records.py | 0/1/>1 occurrence, relative position, log | T5.3; 13 |
| T5.5 | DPR positive union và negative groups | tạo src/retrieval_records.py; tests/test_retrieval_records.py | 2 occurrences đúng chunk; unmappable loại; 1 context = 1 group | T5.1,T5.2; 14 |
| T5.6 | Base D6 + title exclusions P4 | src/corpus.py hoặc tách src/corpus_builder.py; tests/test_corpus_builder.py | nguồn D6; title lấy trước dedup; missing title report | T5.4,T5.5; 15 |
| T5.7 | Chuỗi distractor trên fixture | tạo src/distractors.py; tests/test_distractors.py | R schedule, order/collision/dedup, prefix, title P4 | T5.6; 15 |
| T5.8 | Entrypoint build, reports, base acceptance | sửa notebooks/01_build_corpus_base.py; tạo scripts/check_data_acceptance.py và tests/test_data_acceptance.py | base schema/offset/source/split/coverage; chưa full mid/large | T5.6,T5.7; 16 |

### T5.2 — Thẻ mẫu: Gán split theo D1

- Trạng thái: thiết kế task; chưa triển khai.
- Vì sao: chặn câu test vào train/dev và không đẩy dòng train vào test.
- Spec: §2.3.e, §2.6, Schema/Data §9.
- Files: src/splits.py, tests/test_splits.py; build entrypoint cập nhật caller ở T5.8.
- Input: bốn split gốc; record minh họa do task tạo (không phải trích dẫn dataset):
  gq/train/id=51870/question="Frage A". Cùng key ở gdpr/test thì dòng gq train bị loại.
- Output: quyết định theo dòng + source dataset/split gốc, assigned split hoặc loại leakage;
  gq train đi dev vẫn có query_id=gq-train-51870; split report liệt kê query_id bị loại.
- Fixture 6 dòng: gq train A, gq train B, gq train C, gq test D, gdpr train B, gdpr test A.
  Gq A bị loại; gdpr A giữ test; gq/gdpr B cùng split; gq D giữ test;
  chạy hai lần cùng seed/config cho cùng quyết định. Thêm test riêng gdpr train trùng gdpr test.
- Lệnh: python -m pytest tests/test_splits.py -v (trong venv của chặng 2).
- Bước: viết assertions ở trên → chạy thấy thiếu/hành vi cũ sai → sửa quyết định theo dòng
  và stable query_id → chạy lại → kiểm source_split khác split gốc khi vào dev.
- Trên dữ liệu thật: báo riêng overlap keys và removed rows theo dataset/revision.
  Audit 1.025 key GQ train giao GDPR test và 1 key GDPR train giao test là mốc đối chiếu;
  không cộng thành tổng bắt buộc 1.026, không suy số dòng từ số key.
- Acceptance: không câu test vào train/dev, không train vào test; GermanQuAD
  train+dev+test+rows loại leakage = tổng rows gốc (13.722 với dataset hiện dùng).
  Record gold rỗng không bị loại trong phương trình này.
- CPU local. Học ở buổi 9, làm ở buổi 12. Chi tiết signature/test code sẽ chốt khi bắt đầu chặng.

## T5.8 — Phạm vi báo cáo và nghiệm thu

Namespace đề xuất: data/processed/v1.1/. Giữ nguyên data/processed hiện tại để audit.
Ở base: corpus base, 6 split files, base title exclusions và báo cáo áp dụng gồm corpus_manifest,
split_report, chunk_coverage_report, ambiguous_offset_remap_report, germandpr_chunk_mapping_report.
Manifest base không bịa fingerprint/ID mid hoặc large chưa tồn tại; ghi rõ trạng thái chưa build
theo schema manifest được chốt ở thẻ task. Chuỗi Wikipedia thật nghiệm thu ở T9.3.

Base gate: mọi gold ID trong base, offset đúng, nguồn hợp lệ, test đúng dòng gốc,
coverage ≥98% trên từng file. Fixture distractor qua P4/order/R/prefix là gate logic,
không phải chứng nhận tổng mid=50.000/large=150.000 trên dữ liệu thật.

## Phân tách file dùng chung được dự kiến ở chặng 6–9

- src/run_manifest.py + tests/test_run_manifest.py: identity, file/checkpoint fingerprints,
  bundle hash, manifest I/O/validation; runner gọi dùng chung.
- src/evaluation.py + tests/test_evaluation.py: metric và validator, không đo latency.
- src/reader_runner.py + tests/test_reader_runner.py: oracle sớm; dùng protocol đo §8.3.
- src/integration.py + tests/test_integration.py: orchestration đủ, không định nghĩa lại manifest.

Đường dẫn là kế hoạch, chưa phải file hiện hữu. Khi lập thẻ triển khai, chốt signature,
payload thực và error cases, kể cả adapter post_rerank cần draft A3 được phê chuẩn.
