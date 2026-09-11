# German QA — Lộ trình học và triển khai

> **For agentic workers:** Dùng superpowers:executing-plans khi thực hiện task chi tiết
> đã sẵn sàng. Đây là master roadmap theo hình thức học có hướng dẫn, không phải lệnh
> triển khai toàn bộ hay cho phép tự train/deploy. Chưa tự tạo hoặc giao subagent.

**Goal:** Giúp sinh viên hiểu dữ liệu và hoàn thiện pipeline đúng V1.1, rồi trả lời RQ bằng artifact.
**Architecture:** Data sở hữu corpus/split; retriever → selection → reader. Evaluation chấm artifact;
hạ tầng run dùng chung có trước BM25, oracle runner có cùng Reader, orchestration hoàn thiện sau.
**Tech Stack:** Python, pytest, pandas/Hugging Face, bm25s, PyTorch/Transformers, FAISS; local + Colab T4.
**Spec:** [V1.1 frozen](../specs/2026-09-10-german-qa-retrieval-module-specs.md).

## Global Constraints

- [Constitution v1.0.0](../../../.specify/memory/constitution.md) quản trị mọi chặng.
- [Design/RQ1–RQ3](../specs/2026-09-10-german-qa-retrieval-design.md) định hướng báo cáo.
- [V1.2 draft](../specs/2026-09-11-german-qa-retrieval-v1.2-amendments-draft.md) chưa hiệu lực;
  metadata mới A3 phải được chốt trước production run phụ thuộc nó.
- Base số thật sau build, dự kiến ~7,5K; mid=50.000; large=150.000, lồng nhau theo D6/P4.
- Test gốc không đổi; train trùng test bị loại; gold rỗng vẫn lưu.
- Chunk 100 từ / overlap 20; quality dùng N_eval, latency dùng n_latency.
- Recall@k=Hit@k; selection_failure áp dụng cả bật/tắt reranker; final_correct độc lập.
- retrieval_k=20, reader_k=5 cố định; thiếu không pad; oracle đọc toàn bộ gold.
- P2: Dense Recall@20 trước selection dev/base; Reader F1 oracle dev;
  Reranker Recall@5 sau rerank trên candidate cố định dev/base; tie chọn sớm hơn.
- Khóa checkpoint/config khi đánh giá test và mở rộng tier. Không tune theo test.
- Không sửa V1.1 frozen; giữ artifact cũ. Kết luận mô tả, không mặc định Dense thắng.

## Trạng thái và cách sử dụng

Ngày lập: 2026-09-11. Nội dung lộ trình đã được người dùng yêu cầu áp dụng sau review F1/F2/C1/D1.
Các checkbox đều chưa đánh dấu: tài liệu được lưu không đồng nghĩa chặng đã học hoặc code đã nghiệm thu.
51 test hiện có pass ở lần kiểm tra trước cập nhật tài liệu; có kỳ vọng V1.0, không là chứng nhận V1.1.

[Danh mục task](2026-09-11-german-qa-task-catalog.md) liệt kê phụ thuộc toàn dự án từ đầu.
Viết thẻ thực thi chi tiết khi bắt đầu chặng; không bắt sinh viên đọc mọi chi tiết GPU ngay buổi đầu.
Trạng thái task: chưa làm → đang làm → có code → qua test → nghiệm thu artifact.
Mỗi lần chuyển trạng thái ghi lệnh, kết quả và đường dẫn bằng chứng; không đánh dấu bằng cảm giác.

## Nhịp học và lịch dự kiến

32 buổi × 1,5–2,5 giờ; 3 buổi/tuần tương đương khoảng 11 tuần, dự phòng thành 11–12 tuần.
Thời gian tải Wikipedia/train GPU là thời gian máy chạy riêng, chưa hứa lịch hoàn tất.
Chốt lại sau khảo sát môi trường, fixture dữ liệu và smoke T4; không ép qua gate để kịp lịch.

| Chặng | Buổi toàn khóa | Số buổi | Môi trường |
|---|---|---:|---|
| 1. Hiểu đề tài/project | 1–2 | 2 | Đọc tài liệu/local |
| 2. Setup môi trường | 3–4 | 2 | Local, kiểm tra Colab |
| 3. Làm quen dataset | 5–7 | 3 | CPU/cache |
| 4. Data audit và quyết định spec | 8–10 | 3 | CPU |
| 5. Schema và dữ liệu V1.1 | 11–16 | 6 | CPU/fixture |
| 6. Run infra, Evaluation, BM25 | 17–19 | 3 | CPU |
| 7. Dense Retriever | 20–22 | 3 | Fixture CPU, smoke/train T4 |
| 8. Reader và Reranker | 23–26 | 4 | CPU/T4 |
| 9. Integration và benchmark | 27–29 | 3 | CPU/T4/streaming |
| 10. Phân tích và bàn giao | 30–32 | 3 | CPU/tài liệu |

Mỗi buổi: giải thích bằng ví dụ → xem input → thao tác từng lệnh → kiểm tra output →
bài tự giải thích và nhật ký. Buổi đầu khảo sát project và hai record minh họa;
cài đầy đủ môi trường ở chặng 2, không tải lại dataset chỉ để minh họa.

## Chặng 1 — Hiểu đề tài và khảo sát project

- [ ] Buổi 1: phân biệt closed-context/open-domain và extractive QA; ví dụ Einstein;
  khảo sát src/tests/notebooks/data/results/docs; mở 1 record mỗi dataset nếu đọc được bằng công cụ hiện có.
- [ ] Buổi 2: đọc RQ1 scale, RQ2 bottleneck, RQ3 selection; nối câu hỏi với bảng kết quả cần có;
  lập bản đồ module và hiện trạng giữ/sửa/viết mới/build lại.

Input: design, spec, cây repo và record mẫu. Output: ghi chú khảo sát, pipeline sinh viên tự vẽ.
Kiểm tra: giải thích vì sao tìm được passage và trả lời đúng là hai việc khác nhau.
Gate: biết nguồn chuẩn và chưa nhầm website/spec/code/metrics. Tasks T1.1–T1.2.

## Chặng 2 — Setup môi trường tái lập

- [ ] Buổi 3: kiểm tra Git/Python thật/venv; đọc imports để cài dependency dữ liệu/BM25;
  ghi phiên bản, chạy bộ test cũ và phân loại kỳ vọng đúng/sai V1.1.
- [ ] Buổi 4: kích hoạt lại môi trường từ terminal mới; thiết lập notebook Colab,
  kiểm tra GPU thực được cấp và nơi lưu checkpoint; kiểm tra Spec Kit root nlp.

Input: code/imports và máy local. Output: dependency được ghim, setup guide, log kiểm tra.
Kiểm tra: đóng/mở terminal vẫn chạy cùng interpreter và test; không dùng Windows Store alias.
Gate: môi trường và lệnh thực thi được ghi rõ. Constitution đã có, chỉ kiểm tra/reuse,
không init đè lần nữa. Tasks T2.1–T2.2.

## Chặng 3 — Tìm hiểu dataset từ từng record

- [ ] Buổi 5: JSON/object/list, JSONL và Parquet; dòng dữ liệu khác question_key duy nhất.
- [ ] Buổi 6: GermanQuAD question/context/answers/answer_start; nhiều references;
  tự kiểm tra context[start:start+len(answer)] == answer.
- [ ] Buổi 7: GermanDPR positive/hard negative; đọc dataset card, paper, license,
  revisions; kiểm tra cache; hiểu Wikipedia snapshot và streaming trước khi tải thêm.

Input: 4 split Parquet đã cache nếu hợp lệ. Output: notebook khám phá và bảng nguồn/revision/license.
Kiểm tra: chỉ đúng field của từng dataset; giải thích vì sao không gán mọi chunk positive context là gold.
Gate: hiểu split, offset và role của negative. Tasks T3.1–T3.3.

## Chặng 4 — Audit và đối chiếu V1.1

- [ ] Buổi 8: đếm rows/keys, độ dài context/answer, phân phối positive/negative.
- [ ] Buổi 9: giao question_key giữa 4 split; fixture train trùng test; diễn giải D1,
  báo riêng key/dòng theo dataset, không cộng máy móc các giao nhau.
- [ ] Buổi 10: đối chiếu D1–D6/P1–P4 với code/artifact; phân biệt audit cũ và nghiệm thu mới.

Input: dataset gốc, artifact cũ, spec. Output: audit và requirement traceability matrix.
Kiểm tra: dự đoán trước mỗi record giữ/loại/đi split nào rồi đối chiếu script.
Gate: biết các lỗi phải sửa và revision dữ liệu dùng làm căn cứ. Tasks T4.1–T4.2.

## Chặng 5 — Phân tích hợp đồng và build base V1.1

- [ ] Buổi 11: T5.0 schema validators, T5.1 chunk/provenance/title.
- [ ] Buổi 12: T5.2 split D1 và bảo toàn rows theo từng dataset.
- [ ] Buổi 13: T5.3 gold_evidences/reference_answers, T5.4 collision fallback.
- [ ] Buổi 14: T5.5 positive occurrence union và hard-negative groups.
- [ ] Buổi 15: T5.6 base D6/title exclusions P4; T5.7 distractor fixture.
- [ ] Buổi 16: T5.8 build base, các báo cáo áp dụng và checker nghiệm thu base.

Input: raw splits, fixture nhỏ, validators. Output: data/processed/v1.1/ và báo cáo.
Kiểm tra: offset đúng từng chunk, test gốc đúng, gold rỗng được lưu, negative không làm nguồn corpus,
coverage ≥98% theo file; train+dev+test+rows loại leakage = tổng GermanQuAD gốc.
Gate: base và logic distractor trên fixture đạt; chưa tuyên bố mid/large thật đã nghiệm thu.
Nếu không đạt coverage, dừng báo vấn đề; thay overlap cần quy trình phiên bản, không sửa frozen.

## Chặng 6 — Hạ tầng run, Evaluation và BM25

- [ ] Buổi 17: T6.1 run_id/UUID, file + checkpoint fingerprints, bundle hash và manifest;
  thiết kế đủ field ngay đầu; test bản sao gold sửa 1 byte bị từ chối.
- [ ] Buổi 18: T6.2–T6.3 tính tay Recall/GoldCoverage/MRR/EM/F1, mẫu số, bucket và
  validator; file prediction rỗng có manifest vẫn chấm được.
- [ ] Buổi 19: T6.4 BM25 base/dev qua run retrieval_only; không đo thì latency/warm-up
  metadata null; ghi entry trước chạy; lưu top-50 và metrics trong namespace run riêng.

Input: base V1.1 và fixtures. Output: module run infrastructure có test, Evaluation,
manifest entry BM25/dev và retrieval/metrics. Gate: đúng giao thức, không yêu cầu score cao.
Run helpers được triển khai sớm để đáp ứng §3.2/§8.7, không đổi pipeline V1.1.

## Chặng 7 — Dense Retriever

- [ ] Buổi 20: vector/dot product/CLS; tự vẽ batch; target, negative groups và mask toàn gold_set.
- [ ] Buổi 21: test seed xuyên process, gradient hai tower, save/load/resume; smoke T4 để đo VRAM.
- [ ] Buổi 22: train, chọn checkpoint Recall@20 GermanQuAD dev/base qua hạ tầng run;
  mỗi checkpoint được chấm có fingerprint riêng; encode corpus, lưu FAISS + ID map.

Output: checkpoint hai tower, config, index và dev runs. Gate theo §5.6/P2.
Kiểm tra: giải thích gradient accumulation không tăng in-batch negative của một forward.
Tasks T7.1–T7.3; thời gian train riêng ngoài giờ hướng dẫn.

## Chặng 8 — Reader và Reranker

- [ ] Buổi 23: Reader evidence examples, char/token offsets, overflow; kiểm thử logits giả,
  max raw score khi dedup và softmax theo §6.3, tie deterministic.
- [ ] Buổi 24: runner oracle tối thiểu: reader_only/oracle, reranker off, retrieved=[],
  reranked=null, latency.retrieval=0; đủ fingerprint/điều kiện đo từ đầu;
  train và chọn reader checkpoint F1 oracle dev.
- [ ] Buổi 25: Cross-Encoder/BCE; sampling/mask như Dense; test subset/min(5,pool), tie rank.
- [ ] Buổi 26: khóa candidate top-20, chọn reranker theo Recall@5 sau rerank;
  stage provenance riêng theo draft A3 sau khi được chốt; không lẫn metric retriever.

Output: checkpoint và artifact oracle/post-rerank dev có nguồn rõ. Gate theo §6/§7/P2.
Tasks T8.1–T8.4. Nếu A3 chưa chốt, tiếp tục fixtures/training logic độc lập;
production checkpoint selection theo adapter mở rộng đợi giao thức được phê chuẩn.

## Chặng 9 — Integration và thực nghiệm chính thức

- [ ] Buổi 27: ghép pipeline, mở rộng cách sử dụng manifest chung cho end-to-end;
  fingerprint đủ mọi checkpoint, total latency, query error, resume, không trùng record.
- [ ] Buổi 28: smoke 4 cấu hình + oracle; khóa checkpoint/config; nghiệm thu đầy đủ
  dense+rerank+base trên test theo §8.6 trước mở tier lớn. Lỗi phải log, không tune theo test.
- [ ] Buổi 29: chạy streaming toàn snapshot theo lịch R, lọc P4, dedup và đóng băng chuỗi;
  dựng mid/large, kiểm tra đúng tổng/tập con; reuse embeddings; hoàn tất ma trận benchmark.

Input: modules/checkpoints đạt dev gates. Output: 12 cấu hình end-to-end,
retrieval metrics, oracle và manifests; RQ1 dùng 6 cấu hình reranker off trong cùng ma trận.
Gate: SHA-256/config khớp, n_latency rõ, query lỗi không âm thầm bỏ; base/mid/large nghiệm thu thật.
Tasks T9.1–T9.3. Tải dữ liệu/GPU có thể cần nhiều phiên ngoài buổi học.

## Chặng 10 — Phân tích và bàn giao

- [ ] Buổi 30: Bảng 1/Hình 1 cho RQ1; Bảng 2/Hình 2 cho RQ2; oracle gap và final_correct.
- [ ] Buổi 31: Bảng 3 cho RQ3, paired config/latency; manual review tối đa 100 case;
  ghi giới hạn nhãn, seed, tài nguyên và phân tích phụ từ per_query.
- [ ] Buổi 32: viết kết luận mô tả; đối chiếu contributions với bằng chứng; README tái lập,
  cập nhật website bằng kết quả có provenance; rehearsal giải thích một số từ báo cáo về run.

Output: báo cáo, bảng/hình, manifest index và hướng dẫn bàn giao. Gate: mọi số truy ngược được,
không gọi oracle là trần tuyệt đối, không kết luận thống kê chưa được kiểm chứng.
Tasks T10.1–T10.2. Push/deploy là thao tác riêng khi người dùng yêu cầu.

## Sau mỗi buổi

Ghi ngày, task, câu hỏi đã hiểu/chưa hiểu, lệnh chạy, output path, lỗi gặp và bước tiếp theo.
Nếu lỗi gate: phân loại code/data/spec/environment; chỉ sửa trong phạm vi được duyệt,
không đổi giao thức để làm score đẹp. V2 hybrid/mining/Haystack không chặn hoàn tất V1.
