---
chang: 7
buoi: [20, 21, 22]
trang_thai_hoc_vien: chưa làm
---

# Kiểm chứng học liệu Chặng 7 — chưa thực thi Dense

Ngày 2026-09-25. Phạm vi là ba bài theo mẫu 16 mục, lệnh chỉ đọc/fixture trong RAM. Không triển khai T7, cài dependency, tải model, train, encode corpus thật hoặc thay gate dự án bằng kết quả fixture.

## 1. Nguồn và hiện trạng

Nguồn bắt buộc được tìm thấy/đọc được, đối chiếu theo thứ tự yêu cầu:

| Nguồn | Tình trạng và phần dùng |
|---|---|
| `docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md` | Đọc được; Chặng 7, buổi 20–22 và gate |
| `docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md` | Đọc được; T7.1–T7.3, T6.4 → T7.1 và T7.2 → T7.3 |
| `docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md` | Đọc được; V1.1 frozen, §3/§5/§8 |
| `docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md` | Đọc được; RQ1/RQ2, giới hạn kết luận |
| `docs/superpowers/specs/2026-09-11-german-qa-retrieval-v1.2-amendments-draft.md` | Đọc được; draft chưa hiệu lực, A3 chờ chốt cho post-rerank |
| `.specify/memory/constitution.md` | Đọc được; nguồn chuẩn, kỷ luật test, truy vết, nghiệm thu |
| `src/`, `tests/`, `notebooks/`, `data/`, `results/`, `docs/` | Đọc cấu trúc; schema/corpus/split/QA/DPR và Evaluation cũ hiện hữu, chưa có module/test/notebook Dense |
| README | Root không có README.md (đã ghi trong phân tích A); `website/README.md` đọc được, không dùng làm spec model |

Không thiếu sáu tài liệu chuẩn. Đã đọc thêm concept map, báo cáo gap và log QA/DPR để giữ tiền đề/số liệu đúng phạm vi. Root README vắng không phải bằng chứng project chưa có code. [docs/learning/00-project-analysis.md §1. Kiểm kê nguồn trước phân tích]

Kiểm trực tiếp filesystem và metadata `.venv`: interpreter tồn tại; torch/transformers/faiss-cpu không có distribution và module torch/transformers/faiss không tìm thấy. Có numpy 2.5.3, pytest 9.1.1, pandas 2.2.3, pyarrow 23.0.1. `data/processed/v1.1/` và `src/run_manifest.py` chưa có; chưa có bằng chứng mở gate thực thi T7. Kiểm này không kết luận về Colab hoặc máy ngoài repo.

Raw DPR train cache đọc được ở revision `d0b867a49c45a44d172041351b5df73a87714372`; record đầu được tái kiểm bằng đúng lệnh buổi 20. Các số QA dev 1.047/16 lấy từ log producer trước đây, **không chạy lại toàn producer QA trong lượt soạn này**. [../chang-05/kiem-chung-T5.3-T5.4.md §3. Cache thật và hiệu chỉnh mốc D1]

## 2. Kiểm lệnh và expected

Trích các khối `powershell` trực tiếp từ ba file Markdown, bỏ đúng thụt lề danh sách bằng textwrap.dedent để dấu here-string về cột 0. Chạy từng khối bằng PowerShell NoProfile/NonInteractive tại repo root, dùng `.venv/Scripts/python.exe -B`; ghi exit code/stdout/stderr. Ba cell Python runtime chưa có được giữ nhãn mẫu, không chạy.

| Bài/khối | Kết quả đối chiếu |
|---|---|
| 20/1 | Interpreter True, namespace V1.1 False; ba distribution ABSENT; cây src chưa có Dense |
| 20/2 | Raw 1 positive/3 HN context; positive index [2], ID d20519f819d35708; HN sizes [3,3,2] |
| 20/3 | Group giữa rỗng; qA góp 3 vị trí; keep qA=[0,1,2], qB=[0,1,2,3,4]; NLL 0.169846/1.573172 |
| 21/1 | Đọc công thức seed/RNG/accumulation/checkpoint trong riêng module 5, không lẫn batch Reader |
| 21/2 | Hai process PYTHONHASHSEED=1/9876: seeds 3588503323/786639495; choices [[0,0,1],[1,2,0]], giống nhau |
| 21/3 | Loss 0.313262; dq [-0.268941,0.268941], dp [[-0.268941,0],[0.268941,0]] |
| 21/4 | Prefix [5,0]; resume [0,5,2,1,1] khớp; restart từ seed [5,0,0,5,2] không khớp |
| 22/1 | Hai path hiện False; nguồn P2 đúng Recall@20/GQ dev/base/tie sớm hơn |
| 22/2 | N_eval=3/4; Recall 2/3,1,1; Coverage 1/2,5/6,5/6; chọn epoch 2 |
| 22/3 | Dot [2,1,-1] → pZ,pA; map sort riêng sai thành pA,pM; cosine đổi thứ tự; chỉ pN là ID mới |

Không gọi các output này là training, nghiệm thu FAISS hay benchmark V1.1. Fixture vector/ID chữ cái không phải record schema thật. Script seed kiểm công thức spec trong stdlib, chưa kiểm một sampler production chưa tồn tại. Save/resume chỉ là RNG trong RAM, không phải checkpoint encoder/optimizer.

Ba cell **LỆNH MẪU – chưa xác minh trên runtime thật**: autograd torch, `!nvidia-smi` Colab, FAISS IndexFlatIP. Expected torch/FAISS suy từ phép tính tay; không ghi rằng runtime đã trả các giá trị đó. Không có lệnh CLI giả cho trainer chưa tồn tại.

## 3. Kiểm cấu trúc và phạm vi

- Buổi 20/21/22 lần lượt tuần 7 Thứ 4 / tuần 7 Thứ 6 / tuần 8 Thứ 2, lịch gợi ý có thể đổi; tên chặng Dense Retriever.
- Tasks: 20=T7.1; 21=T7.1 kiểm thử tiếp + T7.2; 22=T7.2 selection + T7.3. Phân bổ khớp buổi trong roadmap và bảng gap, không chuyển toàn bộ selection sang T7.3 sau khi đã chọn checkpoint.
- Mỗi bài đủ 16 mục và 150 phút hướng dẫn/fixture. Tải 180–240/180–240/150–210 trong gap được nêu rõ; nhánh thực thi và máy chạy riêng có task/thứ tự/gate, không hứa đã làm trong 150 phút.
- Output học viên có đường dẫn đề xuất dưới lessons/chang-07/thuc-hanh, chưa tạo các file này. Mọi trạng thái học viên chưa làm; không đánh dấu checklist đạt.
- Trích dẫn phải đối chiếu exact heading/symbol và đường dẫn thật; API/file tương lai không được trích như nguồn hiện hữu.
- Chỉ tạo ba bài và log này trong docs/learning/lessons/chang-07; kiểm hash tập file nền, git diff và file mới ở bước bàn giao. Không chạy lại full pytest vì không đổi code/test; số 398 trước đây không được trình bày như test vừa chạy cho Chặng 7.

Kết quả kiểm ngày 2026-09-25: 10/10 khối PowerShell exit 0, stderr rỗng và expected khớp; đã chạy lại khối đọc spec sau khi giới hạn riêng module 5. Có 64 lượt trích dẫn (28 cặp file/section riêng biệt), đều khớp đường dẫn/heading/symbol. Toàn bộ buổi 1–22 duy nhất, đủ 16 mục, trạng thái chưa làm; ba bài mới đúng lịch và timeline 150 phút, không có checkbox đã tick; lệnh tìm nội dung dùng PowerShell sẵn có.

So SHA-256 150 file nền được kiểm kê (file Git theo dõi và học liệu đã có): không file nào thay đổi/mất; git diff và staged diff rỗng, diff --check sạch, không có .pytest-tmp. Ngoài bốn file do lượt soạn này tạo, kiểm kê cuối thấy thêm `docs/learning/03-research-proposal-form.md`; file đó không thuộc các thao tác ghi của lượt này, được giữ nguyên và không được tính vào kết quả Chặng 7. Không commit hoặc sửa file ngoài bốn học liệu Chặng 7.

## 4. Điểm còn chờ duyệt hoặc kiểm thật

- Duyệt học liệu Chặng 7 trước soạn Chặng 8. Duyệt học liệu không cho phép tự thực thi T7 hoặc phê chuẩn thẻ T5.6 đang chờ.
- Base T5.8 và T6.1–T6.4 chưa nghiệm thu; cần artifact/report/run thật trước T7.1.
- Thẻ T7.1/T7.2/T7.3 phải chốt file/API/test/CLI, checkpoint/resume layout và bước encode/search candidate trước khi viết code; danh mục hiện chỉ có task/phụ thuộc.
- Runtime T4, model/tokenizer revision, dependency GPU/FAISS và nơi lưu checkpoint còn cần xác minh. Không mặc định máy luôn được cấp T4.
- Spec §5.6 liệt kê 10 mục kiểm trước dòng “cả 9 điều”; bài yêu cầu kiểm từng mục, không sửa frozen hoặc bỏ bớt mục. Chỉnh câu chữ nếu cần thuộc amendment của chủ project.
- Metadata post-rerank A3 còn draft; không chặn fixture Dense độc lập, vẫn chặn phần production phụ thuộc A3 ở chặng sau.
