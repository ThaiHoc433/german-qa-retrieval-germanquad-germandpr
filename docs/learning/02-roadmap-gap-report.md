---
trang_thai: chưa làm
pham_vi: Giai đoạn B; đề xuất chờ duyệt
ngay_doi_chieu: 2026-09-19
---
# 02 — Báo cáo đối chiếu roadmap 32 buổi

DỪNG DUYỆT SAU TÀI LIỆU NÀY. Chưa tạo bài học Giai đoạn C. Tất cả buổi/task/gate của sinh viên vẫn **chưa làm**. Roadmap và spec không bị sửa.

## 1. Cách đọc báo cáo

- **Thứ tự hợp lý trong plan** không đồng nghĩa **đầu vào đã có trong repo**. Phân tích A chỉ ra producer/schema/artifact cũ cần sửa/build lại trước các run thật.
- Các khoảng phút ở bảng là **ước lượng sư phạm của người soạn**, chưa đo trên sinh viên. Ước lượng bao gồm giải thích, dự đoán, thao tác và kiểm tra/nhật ký; không tính thời gian máy train/tải/encode/benchmark, cũng không giả định sinh viên đã chuẩn bị miễn phí ngoài giờ.
- Mốc 90–150 phút là từ roadmap 1,5–2,5 giờ. Khoảng vượt mốc có nghĩa cần đề xuất tách hoặc giới hạn phạm vi, không phải thời lượng bài học đã duyệt. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Nhịp học và lịch dự kiến]
- Trong bảng, “gate” là điều kiện nguồn cần đối chiếu, chưa là rubric buổi đã được viết hoặc nghiệm thu.
- Ước lượng giả định sinh viên biết Python cơ bản nhưng chưa quen NLP/IR/PyTorch theo yêu cầu người dùng; không đủ căn cứ hứa hoàn tất implementation và benchmark trong 32 buổi.

Thuật ngữ: **tiền đề** là kiến thức/đầu vào cần trước bài; **fixture** là bộ input nhỏ có thể tính tay; **production selection** là chọn checkpoint bằng run thật thay vì ví dụ giả lập. Các thuật ngữ khác xem [glossary.md](glossary.md).

## 2. Kiểm tra lịch gốc

| Chặng đúng tên bảng roadmap | Buổi | Số buổi |
|---|---|---:|
| 1. Hiểu đề tài/project | 1–2 | 2 |
| 2. Setup môi trường | 3–4 | 2 |
| 3. Làm quen dataset | 5–7 | 3 |
| 4. Data audit và quyết định spec | 8–10 | 3 |
| 5. Schema và dữ liệu V1.1 | 11–16 | 6 |
| 6. Run infra, Evaluation, BM25 | 17–19 | 3 |
| 7. Dense Retriever | 20–22 | 3 |
| 8. Reader và Reranker | 23–26 | 4 |
| 9. Integration và benchmark | 27–29 | 3 |
| 10. Phân tích và bàn giao | 30–32 | 3 |

Tổng 32, không lặp buổi. Nguồn: [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Nhịp học và lịch dự kiến].

Phân bổ tuần gợi ý khi xếp tuần tự 3 buổi/tuần: tuần 1=1–3; 2=4–6; 3=7–9; 4=10–12; 5=13–15; 6=16–18; 7=19–21; 8=22–24; 9=25–27; 10=28–30; 11=31–32; 12=dự phòng không có buổi cố định. Thứ 2/4/6 là gợi ý do người dùng yêu cầu ở lượt xây HTML, không phải dữ kiện trong roadmap. Chưa gán ngày bắt đầu hay tuần hiện tại theo lịch thật.

## 3. Đối chiếu từng buổi 1–32

Nguồn catalog chung cho phụ thuộc và acceptance: [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án]; riêng T5 dùng [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi]. Việc gắn task vào buổi khi roadmap chỉ ghi ở cấp chặng là **ánh xạ sư phạm đề xuất**, được nêu rõ ở §4.

| Buổi | Task phủ | Tiền đề/đầu vào: thứ tự và thiếu hụt | Ước lượng phút hướng dẫn | Tải và rủi ro | Gate/output cần đối chiếu; nguồn |
|---|---|---|---|---|---|
| 1 | T1.1 | Không; chỉ Python cơ bản. Cache mẫu có, xem sổ nguồn A §5.1. | 90–120 | Vừa: khái niệm QA và cây repo; không đọc GPU. | Tự vẽ pipeline; nhận diện nguồn chuẩn. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 1 — Hiểu đề tài và khảo sát project] |
| 2 | T1.2 | T1.1/buổi 1; cần phân biệt metric với kết quả đã đo. | 90–120 | Vừa nếu chỉ nối RQ với bảng/hình, không dạy hết công thức. | RQ → Bảng 1/Hình 1, Bảng 2/Hình 2, Bảng 3. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 1 — Hiểu đề tài và khảo sát project] |
| 3 | T2.1 | T1.1; interpreter thật đã tìm được, dependency chưa ghim thành file. | 120–180 | Có thể quá tải vì môi trường/test cũ; buổi này trong roadmap có chạy test, còn nhiệm vụ soạn chỉ collect. | Môi trường/lệnh/log rõ; không dùng WindowsApps alias. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 2 — Setup môi trường tái lập] |
| 4 | T2.2 | T2.1; quyền Colab/Drive và GPU thực chưa xác minh. | 90–150 | Có điều kiện; thời gian chờ cấp GPU không phải giờ giải thích. | Kích hoạt lại môi trường; notebook/nơi lưu/root Spec Kit đúng. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 2 — Setup môi trường tái lập] |
| 5 | T3.1 | T2.1; cần đọc dict/list và sample Parquet. | 90–120 | Vừa; raw/cache có field thật, tránh đưa processed cũ làm schema chuẩn. | Phân biệt row/key và field. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 3 — Tìm hiểu dataset từ từng record] |
| 6 | T3.1 | Buổi 5; Python slice; context và answer_start raw có trong cache. | 90–150 | Vừa; offset đầu-cuối và nhiều reference là trọng tâm. | Tính và giải thích context[start:start+len(answer)] == answer. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 3 — Tìm hiểu dataset từ từng record] |
| 7 | T3.2, T3.3 | T3.1; DPR struct chứa list song song đã xác minh; thiếu đủ card/paper/license. | 150–210 | Quá tải tiềm năng: cấu trúc DPR và khảo sát nguồn trong cùng buổi. | Đúng field/role negative; inventory revision/license còn phải xác minh. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 3 — Tìm hiểu dataset từ từng record] |
| 8 | T4.1 | T3.1–T3.3; nếu buổi 7 chưa xong inventory phải ghi phần còn thiếu. | 90–150 | Vừa có điều kiện; chỉ số cũ không thay audit revision hiện dùng. | Dự đoán rows/keys/length rồi đối chiếu; ghi nguồn. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 4 — Audit và đối chiếu V1.1] |
| 9 | T4.1; làm quen T5.2 | T4.1 bắt đầu ở buổi 8; tập hợp và nguồn dòng. | 120–150 | Vừa nhưng cần fixture 6 dòng có sẵn trong catalog. | Dự đoán giữ/loại/split, không cộng key thành rows. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 4 — Audit và đối chiếu V1.1] |
| 10 | T4.2 | T4.1; đã đọc D1; D2–D6/P1–P4 cần giới thiệu theo ví dụ. | 120–180 | Có thể quá tải nếu audit toàn bộ code trong một buổi; dùng bảng phân tích A để định hướng, sinh viên vẫn tự giải thích. | Requirement → code/test/artifact; chỉ rõ revision/các lỗi cần sửa. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 4 — Audit và đối chiếu V1.1] |
| 11 | T5.0, T5.1 | T4.2; validators phải có trước producer ngay trong buổi; Prediction schema nhắc khái niệm k/None chưa học sâu. | 180–240 | Quá tải: 4 schema cùng chunk/provenance/title và test chưa có đủ. | Field/provenance và substring chunk đúng; chưa phải nghiệm thu base. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1] |
| 12 | T5.2 | T5.0 từ 11; fixture D1 ở 9; caller build còn cũ. | 120–180 | Có thể quá tải khi sửa theo dòng + stable ID + test mới; chữ ký chưa chốt. | Train trùng test loại; test gốc giữ; ID gốc ổn định. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1] |
| 13 | T5.3, T5.4 | T5.1,T5.2; T5.4 tiếp sau T5.3 trong cùng buổi. | 180–270 | Quá tải rõ: local offsets, references, gold rỗng và collision 0/1/>1 occurrences. | text[start:end] đúng; giữ record rỗng; remap mơ hồ có log. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1] |
| 14 | T5.5 | T5.1,T5.2; positive/negative ở 7; nhắc mask gold_set, chưa cần gradient. | 150–210 | Quá tải tiềm năng: occurrence union + nhóm + unmappable + false-negative. | Đúng answer-bearing chunks, 1 context=1 group. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1] |
| 15 | T5.6, T5.7 | T5.4,T5.5; T5.7 sau T5.6 ngay trong buổi. | 180–270 | Quá tải rõ: D6/P4 trước dedup, order/collision, lịch R/prefix. Chỉ fixture, không tải Wikipedia. | Đúng nguồn/title và logic distractor fixture. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1] |
| 16 | T5.8 | T5.6,T5.7; đồng thời mọi schema/split/gold producer trước đó phải hợp lệ. | 150–210 | Quá tải tiềm năng: entrypoint, nhiều report và checker chưa có; thời gian build phụ thuộc máy. | Base và coverage từng file đạt; không nhận mid/large thật. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1] |
| 17 | T6.1 | T5.8; hash/schema có từ 11; JSON canonical/bundle/UUID còn mới. | 180–240 | Quá tải: identity + file/checkpoint hash + manifest + validation + resume metadata. | Manifest trước run, kể cả rỗng; sửa byte gold phải bị từ chối. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 6 — Hạ tầng run, Evaluation và BM25] |
| 18 | T6.2, T6.3 | T6.1; metric trước bucket/CLI cùng buổi; phải bỏ kỳ vọng D3 cũ. | 180–270 | Quá tải rõ: tính tay nhiều metric, N_eval/n_latency, 5 bucket, manifest/mode/CLI. | Đúng giao thức và tính tay, không cần score cao. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 6 — Hạ tầng run, Evaluation và BM25] |
| 19 | T6.4 | T6.3 và base V1.1; tokenizer/runner hiện tại còn lệch. | 120–180 | Có thể quá tải nếu vừa sửa tokenizer vừa chạy/chấm BM25; phải dạy token/OOV trước. | BM25/dev retrieval_only với entry trước chạy, top-50 và metrics riêng run. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 6 — Hạ tầng run, Evaluation và BM25] |
| 20 | T7.1 | T6.4; thiếu tiền đề vector/tensor/softmax-NLL với sinh viên mới PyTorch. | 180–240 | Quá tải: hai tower, target và mask gold_set; cần block nền tảng CPU. | Tính tay batch/target/mask; logic sampling đúng theo §5.3. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 7 — Dense Retriever] |
| 21 | T7.2; kiểm thử T7.1 | T7.1; cần dạy gradient/optimizer/RNG trước save/load/resume. | 180–240 | Quá tải: seed xuyên process, gradient hai tower, smoke T4 và VRAM. | Gate §5.6/P2; smoke không thay training chính thức. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 7 — Dense Retriever] |
| 22 | T7.2, T7.3 | T7.2 đã smoke; chọn checkpoint cần encode/search dev trước bước FAISS cuối. | 150–210 | Quá tải và phụ thuộc nội bộ chưa rõ; train/encode máy chạy riêng. | Recall@20 dev/base trước selection; checkpoint fingerprint + FAISS ID map. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 7 — Dense Retriever] |
| 23 | T8.1 | T7.3; offset từ 6/13; cần token mapping, logits, softmax. | 180–240 | Quá tải rõ: overflow, toàn bộ span, dedup max, tie deterministic. | Gate §6.6: char-span hợp lệ, hậu xử lý logits giả đúng. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 8 — Reader và Reranker] |
| 24 | T8.2 | T8.1,T6.1; oracle runner chưa có, không dùng LLMAnswerer thay thế. | 180–270 | Quá tải rõ: runner/đo + train reader + chọn F1; train chạy riêng. | Oracle cùng Reader, toàn gold, fingerprint/đo đầy đủ; F1 dev/P2. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 8 — Reader và Reranker] |
| 25 | T8.3 | T8.2,T7.1; nếu Reader training chưa xong thì chưa qua phụ thuộc task thật. | 150–210 | Quá tải tiềm năng: Cross-Encoder/BCE và sampling/subset tests; CPU fixture có thể học độc lập nếu được duyệt. | Gate §7.6: subset, min(5,pool), tie; chưa nghiệm thu checkpoint. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 8 — Reader và Reranker] |
| 26 | T8.4 | T8.3,T7.3; bắt buộc chờ chốt A3 cho production selection. | 150–210 | Quá tải tiềm năng: train/rerank/checkpoint selection/provenance; thời gian train máy riêng. | Top-20 cố định; Recall@5 sau rerank; §7.4/P2, chờ chốt A3. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 8 — Reader và Reranker] |
| 27 | T9.1 | T8.4; manifest/oracle/core modules đã qua gate tương ứng mới ghép thật. | 180–240 | Quá tải: orchestration, partial errors, latency, resume/dedup record. | PredictionRecord và provenance đủ; không âm thầm bỏ lỗi. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 9 — Integration và thực nghiệm chính thức] |
| 28 | T9.2 | T9.1; checkpoint/config phải khóa trước test. | 150–210 | Quá tải tiềm năng: 4 config + oracle và toàn test/base cần thời gian thực thi riêng. | §8.6: dense+rerank+base toàn test trước tier lớn; log lỗi, không tune test. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 9 — Integration và thực nghiệm chính thức] |
| 29 | T9.3 | T9.2,T5.7; chuỗi distractor thật/mid/large hiện chưa thấy trong repo. | 150–240 | Quá tải ngay cả khi không tính streaming/encode/benchmark; máy có thể cần nhiều phiên. | Nguồn/P4/prefix/tổng/hash đúng, đủ ma trận; không hứa xong trong một buổi. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 9 — Integration và thực nghiệm chính thức] |
| 30 | T10.1 | T9.3; cần predictions/metrics/manifests chính thức, hiện chưa có. | 120–180 | Có thể quá tải khi vừa tổng hợp vừa giải thích RQ1/RQ2; bảng trống chỉ là mẫu. | Bảng 1/Hình 1 và Bảng 2/Hình 2 có nguồn, oracle gap/final_correct đúng. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 10 — Phân tích và bàn giao] |
| 31 | T10.1 | T10.1 bắt đầu ở 30; paired runs và sampling manual review. | 150–240 | Quá tải tiềm năng: 100 case là tối đa, chưa có tốc độ đọc tiếng Đức; không mặc định review đủ trong buổi. | Bảng 3 + giới hạn; không đổi bucket bằng review tay. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 10 — Phân tích và bàn giao] |
| 32 | T10.2 | T10.1; mọi số báo cáo đã có provenance, không chỉ ảnh chụp/bảng chép tay. | 150–210 | Quá tải tiềm năng: report, reproduction guide, website và rehearsal. | Số truy về run; kết luận mô tả; push/deploy là việc riêng. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 10 — Phân tích và bàn giao] |

## 4. Coverage task và gate

Catalog có **34 ID task riêng** khi tính cả T5.0; T1.1–T10.2 là cách gọi dải theo chặng, không phải một dãy số liên tục. Không thấy task catalog nào hoàn toàn không có chỗ trong roadmap, nhưng nhiều task chỉ được liệt kê chung cho chặng, chưa định nghĩa gate riêng của từng buổi. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án] [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Chặng 5 — Danh mục chi tiết và phân bổ sáu buổi]

| Chặng | Ánh xạ để duyệt | Mức tường minh / chỗ còn thiếu |
|---|---|---|
| 1 | T1.1→1; T1.2→2 | Đề xuất theo nội dung; roadmap ghi dải task ở cuối chặng |
| 2 | T2.1→3; T2.2→4 | Đề xuất theo nội dung; gate môi trường cấp chặng |
| 3 | T3.1→5–6; T3.2→7; T3.3→7 | Cần duyệt: T3.1 bao gồm QA offset nên không gán máy móc mỗi task một buổi |
| 4 | T4.1→8–9; T4.2→10; T5.2 được học khái niệm ở 9 | Catalog ghi rõ T5.2 “học ở buổi 9, làm ở buổi 12” |
| 5 | T5.0/T5.1→11; T5.2→12; T5.3/T5.4→13; T5.5→14; T5.6/T5.7→15; T5.8→16 | Tường minh, nhưng 9 task trong 6 buổi; nhiều phụ thuộc nằm cùng buổi |
| 6 | T6.1→17; T6.2/T6.3→18; T6.4→19 | Tường minh trong nội dung buổi |
| 7 | T7.1→20 và kiểm gradient ở 21; T7.2→21–22; T7.3→22 | Đề xuất theo nội dung; cần làm rõ encode/search trước quyết định checkpoint |
| 8 | T8.1→23; T8.2→24; T8.3→25; T8.4→26 | Đề xuất theo nội dung; chờ chốt A3 ở T8.4 |
| 9 | T9.1→27; T9.2→28; T9.3→29 | Đề xuất theo nội dung; base gate trước thật ở mid/large |
| 10 | T10.1→30–31; T10.2→32 | Đề xuất theo nội dung; không ép 2 task thành 2 buổi |

Nguồn: các chặng tương ứng trong roadmap; [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T5.2 — Thẻ mẫu: Gán split theo D1].

**Khoảng trống coverage nghiệm thu:**

1. **P3** (GermanDPR test retrieval_only) có trong spec nhưng roadmap chưa chỉ buổi/run cụ thể. Đề xuất đặt kiểm validator/fixture ở T6.3 (buổi 18) và artifact sanity-check thật ở T9.3 (buổi 29), báo riêng ở T10.1. Chỉ là bổ sung ánh xạ của hợp đồng đã có; chưa tạo task mới. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §3.2 Input/Output] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §3.3 Quy tắc xử lý]
2. **Gate riêng mỗi buổi** chưa có đầy đủ: hiện chỉ có gate chặng và acceptance task. Giai đoạn C phải viết điều kiện học tập đo được, dẫn về gate gốc; không tự đặt ngưỡng Recall/F1. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Trạng thái và cách sử dụng] [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Khuôn một thẻ task sẵn sàng thực thi]
3. **Files/signatures dự kiến** chưa là input hiện hữu. Các tên mới trong catalog phải ghi “dự kiến/chưa có”; lệnh gọi chúng phải là LỆNH MẪU – chưa xác minh. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phân tách file dùng chung được dự kiến ở chặng 6–9]
4. **Môi trường/nguồn**: T3.3 yêu cầu card/paper/license nhưng hiện mới đọc được card GermanQuAD cache; chưa đủ vật liệu cho toàn buổi 7. Xem [00-project-analysis.md §5.1 Cache đã đọc bằng đường dẫn local, không gọi Hub].

## 5. Phụ thuộc chéo và mâu thuẫn cần xử lý

### B1 — Code/test/artifact không đồng bộ với spec

- Schema Prediction thiếu k; QA processed lại cũ hơn cả validator QA; vì vậy không thể nối ngay processed cũ với CLI Evaluation hiện tại để minh họa “pipeline chuẩn”. [src/schemas.py §PREDICTION_RECORD_FIELDS] [notebooks/01_build_corpus_base.py §build_qa_records] [tests/test_schemas.py §test_old_round1_schema_is_rejected_with_specific_fields]
- Tắt reranker mà gold hạng 8: spec yêu cầu selection_failure, code/test cũ nói retrieval_miss. Bài tính tay phải theo spec; test cũ được dùng như ví dụ phát hiện sai hợp đồng. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §3.6 Kiểm thử] [src/evaluation.py §_stage_sets] [tests/test_evaluation.py §test_rerank_off_gold_beyond_reader_k_is_retrieval_miss]
- Corpus/QA test cũ không đúng D6/D1; coverage tổng không thay coverage từng file. Không nghiệm thu bằng việc đọc thấy một số lớn hơn 98%. [data/processed/corpus_manifest.json §source_breakdown] [data/processed/chunk_coverage_report.json §coverage_rate] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.6 Kiểm thử]
- BM25 tokenizer hiện tại khác §4.3; runner có RUN_ID cố định, ghi file cũ, chưa có manifest chuẩn. Đây là phần sửa trong tương lai, không làm trong nhiệm vụ học liệu. [src/bm25_retriever.py §BM25Retriever.search] [notebooks/02_run_bm25.py §RUN_ID] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §8.4 Cấu hình]

### B2 — Run infra trước BM25 là đúng, không phải đổi pipeline

Roadmap đặt T6.1 trước T6.4; catalog và constitution IV hỗ trợ hạ tầng manifest chung. Không được trì hoãn hash/bundle/oracle measurement đến Integration. A1/A2 là draft làm rõ; cơ sở đang hiệu lực còn nằm ở §3.2/§8.7 và constitution, không cần tự tuyên bố toàn V1.2 đã hiệu lực. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án] [.specify/memory/constitution.md §IV. Tái lập và truy vết được] [docs/superpowers/specs/2026-09-11-german-qa-retrieval-v1.2-amendments-draft.md §A1 — Hạ tầng run có trước lần đánh giá đầu tiên]

### B3 — T7.2/P2 cần encode/search trước kết quả cuối T7.3

T7.2 bao gồm dev checkpoint selection, T7.3 encode base/FAISS phụ thuộc T7.2. Nhưng Recall@20 cho mỗi checkpoint cần encode/search dưới checkpoint đó trước khi chọn. Không phải vòng phụ thuộc của mô hình, mà là catalog chưa tách **đánh giá candidate checkpoint** với **đóng gói index cuối của checkpoint được chọn**. Đề xuất ghi substep tạm trong thẻ T7.2, rồi T7.3 nghiệm thu index/ID map cuối; chủ project cần duyệt cách phân ranh. Không giả định tên hàm mới. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.2 Input/Output] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.4 Cấu hình]

### B4 — A3 là gate thật của production, không chặn học fixture

Criterion reranker đã có trong P2/§7.4. Metadata phân biệt retrieval/post_rerank và candidate fingerprint là draft A3: **chờ chốt A3**. Nếu chưa chốt, chỉ soạn/học fixture độc lập; không đi tiếp T9.1 production qua phụ thuộc T8.4 bằng cách đánh dấu xong giả. [docs/superpowers/specs/2026-09-11-german-qa-retrieval-v1.2-amendments-draft.md §A3 — Phân biệt điểm retriever và điểm sau rerank] [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án]

### B5 — Nghiệm thu base khác nghiệm thu đủ ba tier

Spec §2.6/§9 mô tả đích cuối cả ba tier; catalog T5.8 giới hạn base và fixture distractor, T9.3 mới làm chuỗi thật. Đây là nghiệm thu theo giai đoạn, không mâu thuẫn nếu ghi rõ phạm vi. Manifest base chưa có mid/large không được bịa hash/ID của file chưa tồn tại; cấu trúc trạng thái chưa build cần chốt. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T5.8 — Phạm vi báo cáo và nghiệm thu] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.7 Artifact]

### B6 — D1 chuẩn hóa question_key cần xác nhận chi tiết

src/splits.py gộp whitespace; §2.3.e nêu lowercase+strip+NFC, không ghi bước gộp khoảng trắng nội bộ. Đề xuất chủ project xác nhận cách hiểu chính xác qua quyết định tường minh; không tự mở rộng nghĩa spec hoặc sửa frozen. [src/splits.py §question_key] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.3 Quy tắc xử lý]

### B7 — Tuần học không đảm bảo gate theo thời gian máy

Buổi 22→23 phụ thuộc checkpoint/index; 24→25 phụ thuộc oracle/Reader; 26→27 phụ thuộc A3/reranker; 28→29 phụ thuộc full base test. Không thể dùng số thứ tự tuần để suy các gate đã đạt. Các bài đọc/fixture độc lập có thể được duyệt học trước, nhưng trạng thái task production vẫn chưa làm. [docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án] [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Nhịp học và lịch dự kiến]

### B8 — Thiếu bậc nền tảng với người mới PyTorch

Roadmap bắt đầu vector/dot/CLS tại 20 và gradient/save-load tại 21; chưa phân thời lượng riêng cho tensor, loss, backward, optimizer và RNG. Bản đồ khái niệm đề xuất đưa đúng lúc 20–21 thay vì nhồi buổi 1. Reader 23 cần thêm token offset, logits và softmax trước hậu xử lý. Đây là đề xuất sư phạm, không sửa kiến trúc. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 7 — Dense Retriever] [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 8 — Reader và Reranker]

## 6. Chạy máy riêng và công việc ngoài giờ hướng dẫn

Không ước lượng giờ tải/train khi chưa có smoke. “Máy chạy riêng” không biến thành quyền chạy trong nhiệm vụ này.

| Buổi/task | Công việc máy theo nguồn | Điều kiện trước chạy / ảnh hưởng lịch |
|---|---|---|
| 7 / T3.3 | Hiểu Wikipedia snapshot/streaming, kiểm cache; không bắt buộc tải tại buổi | Card/revision chưa đủ thì xác minh nguồn trước; không gọi loader tự tải |
| 15 / T5.7 | Fixture lịch R/order/prefix | Không chạy snapshot thật; giữ khác biệt fixture và streaming |
| 16 / T5.8 | Build base/báo cáo khi bước triển khai tương lai được phép | Chưa đo thời gian local; không ép checker/báo cáo hoàn tất trong phần giải thích |
| 21–22 / T7.2–T7.3 | Smoke/train Dense, checkpoint, encode corpus, FAISS | GPU thật, bộ dữ liệu V1.1, P2/dev/manifest; thời gian train riêng |
| 24 / T8.2 | Train Reader, oracle dev selection | Reader đúng offset/window trước; fingerprint/đo đã có |
| 25–26 / T8.3–T8.4 | Train/rerank candidate và chọn checkpoint | CPU fixture trước T4; **chờ chốt A3** trước selection production |
| 28 / T9.2 | Smoke 4 config + oracle, full test base | Khóa cấu hình; có thể cần thời gian thực thi ngoài giờ, không coi đã nghiệm thu do hết buổi |
| 29 / T9.3 | Streaming toàn snapshot mỗi mức R; freeze; encode phần mới; ma trận benchmark | T9.2 đạt trước, có thể nhiều phiên tải/GPU; không đổi R theo metric |

Nguồn: [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 5 — Phân tích hợp đồng và build base V1.1]; [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 7 — Dense Retriever]; [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 8 — Reader và Reranker]; [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 9 — Integration và thực nghiệm chính thức]; [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §2.8 Tài nguyên]; [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.8 Tài nguyên].

**Công việc của người, không được gọi là máy chạy riêng:** đọc paper/license, sửa code/test, debug môi trường, review tới 100 case, viết báo cáo/README và rehearsal. Nếu vượt 150 phút, phải tách/gộp/chuyển có ghi nhận; không ẩn vào “train ngoài giờ”. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 10 — Phân tích và bàn giao]

## 7. Rủi ro học tập và cách xử lý đề xuất

| Rủi ro | Điểm kẹt dự kiến | Cách xử lý trong học liệu, chưa triển khai |
|---|---|---|
| “Có file = đúng spec” | Buổi 1–3,10 | Cho hai cột hiện trạng/mục tiêu; đọc một case test cũ trái spec |
| Nhầm row/key | Buổi 5,8–9,12 | Tính tay fixture trùng key; báo removed rows theo dataset |
| Dùng normalized_text cho offset | Buổi 6,11,13,23 | Dự đoán slice gốc trước; phân biệt offset parent/chunk/token |
| Xóa hard negative bằng blacklist | Buổi 10,14–15 | Dạy nguồn D6: chunk trùng nguồn hợp lệ vẫn được giữ |
| Coi mọi chunk positive là gold | Buổi 7,14 | Fixture nhiều occurrence và một chunk không có answer |
| Đọc test xanh theo V1.0 | Buổi 3,18 | Đối chiếu test function cụ thể với §3.6; không dùng số test thay gate |
| Nhầm task có file dự kiến thành có code | Buổi 11–29 | Ghi hiện hữu/dự kiến; chưa có signature thì chưa viết lệnh chắc chắn |
| Nhầm accumulation với tăng negative | Buổi 20–21 | Vẽ một forward và tập candidate; chưa cần GPU để giải thích |
| Dùng điểm Reader như xác suất hiệu chuẩn | Buổi 23–24 | Tính logits/dedup/softmax bằng fixture, nêu heuristic |
| Tune test / báo “Dense tốt hơn” trước số liệu | Buổi 28–32 | Checklist nguồn run; khóa dev; chỉ kết luận mô tả |
| Lệnh audit tưởng chỉ-đọc | Buổi 1–10 | Đọc main/import scope; chỉ dùng đọc file/metadata, không chạy notebook |

Bằng chứng cho các rủi ro kỹ thuật được chỉ rõ trong [00-project-analysis.md §4. Hiện trạng module: giữ / sửa / viết mới / build lại] và các điều khoản tương ứng của spec; mức “người học hay kẹt” là nhận định sư phạm, chưa phải dữ liệu đo trên sinh viên này.

## 8. CẦN XÁC MINH

1. **CẦN XÁC MINH: phạm vi học mỗi buổi** — chủ project xác nhận 32 buổi là hướng dẫn/fixture hay phải hoàn thành implementation cùng giờ; nếu cả hai, cần duyệt thêm buổi.
2. **CẦN XÁC MINH: thời gian tự học và khả năng đọc tiếng Đức/tài liệu kỹ thuật** — sinh viên/giảng viên xác nhận trước chọn số case manual review; không đặt mặc định 100 case trong 150 phút.
3. **CẦN XÁC MINH: interpreter/cache trên máy học và môi trường Colab/Drive** — sinh viên kiểm buổi 3–4; paths cache trong A là của máy khảo sát.
4. **CẦN XÁC MINH: revision chuẩn của dữ liệu/model, card/license GermanDPR/Wikipedia và paper local** — chủ project cung cấp/chốt nguồn ở buổi 7. Bốn Parquet cache đã đọc được, không cần tải lại để minh họa.
5. **CẦN XÁC MINH: quyết định A3** — chủ project phê chuẩn amendment; hiện **chờ chốt A3**, việc duyệt học liệu không tự phê chuẩn A3.
6. **CẦN XÁC MINH: question_key có gộp whitespace nội bộ hay không** — chủ project/kiến trúc sư đối chiếu §2.3.e với code và ghi quyết định; không sửa V1.1.
7. **CẦN XÁC MINH: hợp đồng manifest base chưa có tier lớn** — kiến trúc sư chốt tại thẻ T5.8/T6.1, không bịa fingerprint.
8. **CẦN XÁC MINH: phân ranh T7.2/T7.3** — duyệt encode/search tạm để chấm từng checkpoint rồi đóng gói index cuối.
9. **CẦN XÁC MINH: checkpoint/artifact nghiệm thu ngoài repo** — chủ project cung cấp nếu có; hiện trạng “chưa thấy” chỉ giới hạn cây đã đọc.
10. **CẦN XÁC MINH: ngày bắt đầu/thứ học thực** — sinh viên chốt sau; Thứ 2/4/6 chỉ là gợi ý, không chặn viết Chặng 1.

## 9. Đề xuất thay đổi roadmap — chỉ đề xuất, chưa áp dụng

1. **PĐ1 — Chọn phạm vi 32 buổi.** Khuyến nghị giữ 32 buổi làm khung hướng dẫn, mỗi bài 90–150 phút, nêu rõ phần hiểu/fixture và gate project còn chờ. Không hứa xây toàn bộ hệ thống từ đầu trong giờ học. Nếu muốn học + implement từ đầu trong buổi, chọn phương án mở rộng lịch và duyệt số buổi sau phân rã.
2. **PĐ2 — Giảm tải trước chặng 5.** Ở buổi 7 tách đọc DPR với inventory nguồn; đề xuất đưa phần inventory vào một phiên bù riêng thay vì lấy giờ train. Buổi 10 chốt bảng đối chiếu, không bắt tự audit cả repo từ trang trắng. Không đánh dấu T3.3 xong nếu chưa đủ nguồn.
3. **PĐ3 — Tách các cặp chặng 5 khi cần implementation.** Đề xuất 11a=T5.0, 11b=T5.1; 13a=T5.3, 13b=T5.4; 15a=T5.6, 15b=T5.7. Đây là ba phiên bổ sung tiềm năng, **không phải buổi đã thêm vào lịch**. T5.2/T5.5/T5.8 cũng cần đo tải sau fixture trước khi hứa đủ 150 phút.
4. **PĐ4 — Chặng 6 chia hợp đồng và phép chấm.** Đề xuất tách 17 thành identity/file hash và checkpoint/manifest; 18 thành metric/mẫu số và bucket/validator/CLI. Có thể chuyển phần nhắc metric cơ bản vào cuối 16 chỉ nếu 16 còn thời lượng sau gate; không làm ngược phụ thuộc production T6.1 trước T5.8. Mọi chuyển phải được duyệt.
5. **PĐ5 — Chặng 7 thêm nền tảng đúng lúc.** Ưu tiên block vector/tensor/loss/mask bằng CPU ở 20 và gradient/RNG ở 21; tách smoke T4 khỏi phần hiểu nếu quá giờ. Chốt rõ substep encode/search candidate trong T7.2, T7.3 giữ nghiệm thu index cuối. Train/encode chạy riêng, không chọn checkpoint trước khi có dev metric.
6. **PĐ6 — Chặng 8 chia CPU và T4.** Đề xuất tách 23 thành examples/token offsets và hậu xử lý logits/window; 24 thành oracle runner/đo và Reader training-selection; 25/26 giữ fixture pairs/subset riêng với run train/selection. Production T8.4 vẫn chờ chốt A3; không lấy việc hoàn tất fixture làm bằng chứng qua task.
7. **PĐ7 — Gắn rõ P3.** Bổ sung ánh xạ P3 vào buổi 18 fixture/mode, 29 sanity-check thật, 30–31 báo riêng; không trộn GermanDPR test vào RQ chính.
8. **PĐ8 — Chặng 9–10 theo gate, không theo đồng hồ.** Buổi 29 có thể cần nhiều phiên máy; buổi 30 chưa có artifact thì chỉ học mẫu bảng có nhãn minh họa và giữ gate chưa làm. Chia review/báo cáo/rehearsal thành phiên bù khi vượt giờ. Tuần 12 chỉ là dự phòng; không hứa đủ chứa tất cả phiên bổ sung PĐ2–PĐ6.

Không đề xuất gộp thêm các buổi đang quá tải. Chỉ cân nhắc gộp phần ôn trùng sau khi đo năng lực sinh viên; không gộp gate của task phụ thuộc.

## 10. Kiểm tra tài liệu A–B trước điểm dừng

- Đã đối chiếu 32 hàng buổi, 10 chặng, tổng chặng khớp roadmap; ánh xạ 34 task được liệt kê, không đánh dấu task hoàn thành.
- Các file/hàm được khẳng định có đã đọc trong repo; các module mới của catalog được ghi là dự kiến/chưa thấy.
- Test chỉ **collect-only**: 110 test được thu thập, không chạy test/train/benchmark; không công bố pass.
- Nguồn raw cache và artifact cũ được phân biệt; số audit không biến thành kết quả V1.1.
- Chưa tạo lessons/, index.json hoặc README Giai đoạn D. Các kiểm tra đủ 32 **bài học** của Giai đoạn D chưa áp dụng.
- Chỉ cho phép ghi trong docs/learning/; đối chiếu phạm vi thay đổi ở cuối lượt soạn, không sửa roadmap/spec/code.

Kết quả kiểm tra chỉ-đọc cuối lượt soạn: 32 hàng buổi theo đúng thứ tự, 34 task catalog đều có ánh xạ; 115 tham chiếu file/section riêng biệt đã kiểm đường dẫn và đối chiếu heading Markdown, symbol Python hoặc khóa JSON khi áp dụng. Không thấy tham chiếu hỏng, checkbox đã đánh dấu hay ký tự điều khiển bất thường. Có 58 mục khái niệm trong bản đồ và glossary. Kiểm tra này xác nhận cấu trúc/truy vết tài liệu, không xác nhận sinh viên qua gate.

So sánh SHA-256 của 121 file ngoài docs/learning/ trước và sau bước viết tài liệu: không có file thay đổi hoặc phát sinh trong tập được kiểm kê; phép so sánh loại trừ .git, .agents, .codex, __pycache__ và .pytest_cache. git diff không có thay đổi tracked file; các file untracked từ trước được giữ nguyên. Chỉ tạo bốn file Markdown của Giai đoạn A–B trong docs/learning/.

## 11. Câu hỏi duyệt để sang Giai đoạn C

1. Bạn chọn **giữ 32 buổi hướng dẫn, tách phần implementation/máy chạy theo gate** (PĐ1), hay mở rộng lịch để làm implementation trong các phiên riêng?
2. Bạn duyệt các đề xuất PĐ2–PĐ8 và ánh xạ task §4, hay cần sửa mục nào? Có thể duyệt từng mục; không có nghĩa A3 được phát hành.
3. Bạn có nguồn paper/card/license hoặc artifact/checkpoint ngoài repo cần bổ sung không? Nếu chưa, có thể giữ các mục đó “CẦN XÁC MINH” và bắt đầu Chặng 1 từ nguồn đã đọc.

Chỉ khi người dùng trả lời **“duyệt”** (kèm chỉnh sửa nếu có), mới viết buổi 1–2 của **Chặng 1** rồi dừng chờ xem. Đây là điểm dừng do người dùng yêu cầu tại Giai đoạn B, không phải do thiếu quyền sửa code.
