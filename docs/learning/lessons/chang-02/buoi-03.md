---
buoi: 3
chang: 2
ten_chang: Setup môi trường tái lập
tuan_goi_y: 1
thu_goi_y: Thứ 6 (lịch gợi ý, có thể đổi)
tasks: [T2.1]
moi_truong: [Local]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 2 — Setup môi trường tái lập
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T2.1
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §0. Nguyên tắc chung & phân chia trách nhiệm module
  - docs/learning/00-project-analysis.md §7. Sổ lệnh đã đối chiếu và kết quả chỉ-đọc
trang_thai: chưa làm
---

# Buổi 3 — Xác định interpreter và baseline môi trường

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- phân biệt Python interpreter, venv và dependency;
- xác định interpreter Python thật đang được dùng trên máy hiện tại;
- đọc imports để lập danh sách dependency quan sát được mà không cài đặt;
- thu thập test bằng `pytest --collect-only` và giải thích vì sao “collected” không có nghĩa là “passed”;
- lập bảng phân loại kỳ vọng test cũ theo đúng / lệch / CẦN XÁC MINH so với V1.1.

## 2. Vị trí trong bức tranh lớn

Buổi 3 mở Chặng 2 và phục vụ T2.1. Môi trường đúng giúp các fixture, test và lệnh sau này chạy cùng interpreter; nó chưa chứng minh code khớp V1.1. Roadmap yêu cầu kiểm tra Git/Python thật/venv, đọc imports, ghi phiên bản, phân loại kỳ vọng test cũ đúng/sai V1.1, ghim dependency và chạy bộ test cũ. Trong lượt học liệu chỉ-đọc này, phần ghim dependency và chạy test thật được đặt thành **nhánh thực thi của chính T2.1 sau baseline**; không chạy trong lúc soạn bài và không gán lựa chọn `collect-only` cho roadmap. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 2 — Setup môi trường tái lập]

## 3. Tiền đề

- Buổi 1: biết thư mục `src/` và `tests/`.
- Buổi 2: biết test là bằng chứng hành vi cụ thể, không phải chứng nhận toàn project.
- Biết mở PowerShell và đọc output; chưa cần biết cách tạo venv hoặc cài dependency.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| Interpreter | File Python cụ thể thực sự chạy lệnh; trong lần khảo sát hiện tại là Python 3.13.3 ở đường dẫn cài đặt thật. | Tin `python` trên PATH dù nó trỏ WindowsApps alias. |
| venv | Môi trường Python tách biệt cho project và dependency của project. | Tưởng cài package vào máy là đủ tái lập. |
| Dependency | Gói mà module import, ví dụ `pytest`, `pandas`, `pyarrow`, `bm25s`, `huggingface-hub`. | Ghi package chỉ vì roadmap nhắc tới nhưng chưa thấy import hoặc metadata local. |
| Test collection | Pytest đọc và liệt kê test function mà chưa chạy assertion. | Viết “110 test pass” khi mới chạy `--collect-only`. |
| Baseline | Ghi nhận trạng thái môi trường và test trước khi học tiếp. | Sửa code để làm baseline đẹp hơn. |
| Reproducible | Người khác biết interpreter, dependency, lệnh và output cần dùng để lặp lại kiểm tra. | Chỉ ghi “chạy pytest” mà không ghi thư mục và interpreter. |

Các phiên bản package đã quan sát trên máy khảo sát là pytest 9.1.1, pandas 2.2.3, pyarrow 23.0.1, bm25s 0.3.11 và huggingface-hub 1.31.0; đây là metadata local, không phải yêu cầu cứng cho mọi máy. [docs/learning/00-project-analysis.md §7.1 Lệnh đã chạy, không ghi artifact project]

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | So sánh interpreter thật, alias, venv và dependency bằng chính repo. |
| 20–40 | Xem input | Đọc imports của các module hiện có và danh sách test. |
| 40–70 | Thao tác từng lệnh | Xác định interpreter, Git status, `.venv`, phiên bản package và collect-only. |
| 70–100 | Kiểm tra output | Lập baseline log; phân loại “đã quan sát”, “chưa xác minh”, “lệch spec”; lập bảng kỳ vọng test cũ so với V1.1. |
| 100–120 | Tự giải thích + nhật ký | Nói lại vì sao không được cài hoặc chạy thêm để làm đẹp baseline. |

## 6. Input

- `src/corpus.py`, `src/splits.py`, `src/evaluation.py`, `src/bm25_retriever.py`.
- `tests/` hiện có các file test đã được thu thập.
- Interpreter local: `C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe` — đường dẫn này cần xác minh trên máy sinh viên.

## 7. Thao tác từng bước

1. Xác nhận thư mục hiện tại:

   ```powershell
   Get-Location
   ```

   Kỳ vọng: root project `D:\Project\2026-2027-HK1\nlp`. Nếu khác, chuyển tới root đã được giảng viên cung cấp; không đoán đường dẫn.
2. Xem Python trên PATH và interpreter đã đối chiếu:

   ```powershell
   Get-Command python,py -ErrorAction SilentlyContinue | Select-Object Name,Source
   & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B -c "import sys; print(sys.executable); print(sys.version)"
   ```

   Kỳ vọng trên máy khảo sát: `sys.executable` là đường dẫn Python313 và phiên bản bắt đầu bằng 3.13. Nếu `python` trỏ `WindowsApps`, ghi cảnh báo; không sửa PATH trong buổi này.
3. Kiểm tra package metadata mà không import module project:

   ```powershell
   & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B -c "import importlib.metadata as m; print({n:m.version(n) for n in ['pytest','pandas','pyarrow','bm25s','huggingface-hub']})"
   ```

   Kỳ vọng: in được phiên bản hoặc báo package thiếu. Nếu thiếu, ghi `CẦN XÁC MINH: dependency chưa có — giảng viên quyết định cách cài ở phiên setup`, không chạy lệnh cài.
4. Kiểm tra trạng thái repo và venv bằng lệnh chỉ-đọc:

   ```powershell
   git status --short
   Test-Path -LiteralPath '.venv'
   ```

   Kỳ vọng: `git status --short` chỉ là ảnh chụp trạng thái, không commit hay sửa file; `Test-Path` trả `True` hoặc `False`. Nếu `.venv` chưa có, ghi `CẦN XÁC MINH`, không tự tạo trong buổi này.
5. Đọc imports thật:

   ```powershell
   Get-ChildItem -LiteralPath src,tests,notebooks -Recurse -File -Filter '*.py' | Select-String -Pattern '^(import |from )'
   ```

   Kỳ vọng: thấy dependency Python trong file thật; phân biệt import top-level với package chỉ được nhắc trong spec.
6. Thu thập test, không chạy test:

   ```powershell
   & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B -m pytest tests --collect-only -q -p no:cacheprovider
   ```

   Kỳ vọng trên lần khảo sát: `110 tests collected`, exit code 0. Ghi chú phép đếm: `110 = 51 + 59`, trong đó nhóm test schema/evaluation/BM25 được đếm `21 + 25 + 5 = 51`, còn năm file test RAG được đếm tổng `59`; đây là phép tính từ các file đã quan sát, không phải phân loại V1/V2 do project công bố. Vẫn là collected, không phải passed. Dùng collect-only trong lượt chỉ-đọc để không tạo cache/artifact và không biến kỳ vọng cũ thành bằng chứng nghiệm thu V1.1; đây là lựa chọn của phiên học liệu, không phải câu trích từ roadmap. Nếu collection lỗi, chép lỗi vào nhật ký và phân loại environment/import/test; không sửa code.
7. Lập bảng **kỳ vọng test cũ so với V1.1** với các cột `test/file`, `hành vi đang kiểm tra`, `mục spec`, `phù hợp / lệch / CẦN XÁC MINH`, `lý do`. Chỉ cần bắt đầu với tám file test và hai test lệch đã biết, không rà thủ công 110 test. Điền hai hàng mẫu đã đối chiếu:

   | test/file | Hành vi đang kiểm tra | Mục spec | Phân loại | Lý do |
   |---|---|---|---|---|
   | `tests/test_evaluation.py:213` — `test_rerank_off_gold_beyond_reader_k_is_retrieval_miss` | Gold ở hạng 8 khi tắt reranker | V1.1 D3, §3.3 | Lệch | Với `retrieval_k=20`, `reader_k=5`, gold nằm trong pool nhưng ngoài reader input nên là `selection_failure`. |
   | `tests/test_evaluation.py:224` — `test_rerank_on_gold_dropped_by_reranker_is_reranking_failure` | Gold bị loại sau rerank | V1.1 D3, §3.3 | Lệch tên bucket | D3 đổi `reranking_failure` thành `selection_failure`, cả bật và tắt reranker. |

   Không dùng nhãn V1/V2 thay cho đúng/sai V1.1; các hàng còn lại ghi `CẦN XÁC MINH` nếu chưa đối chiếu nguồn.
8. Lập baseline log gồm: ngày, thư mục, `git status --short`, trạng thái `.venv`, `sys.executable`, `sys.version`, package version, lệnh collect-only, số test collected, bảng kỳ vọng và lỗi nếu có. Không tạo file trong `results/` và không ghi đè log cũ.
9. **Nhánh thực thi T2.1 sau baseline (chưa chạy trong lượt chỉ-đọc):** giảng viên cung cấp lệnh tạo/kích hoạt `.venv`, ghim dependency theo profile đã thống nhất, rồi chạy bộ test cũ bằng interpreter đã chọn. Kết quả phải ghi riêng `passed/failed/error/skipped`, đối chiếu từng nhóm với bảng bước 7 và không dùng số pass để tuyên bố V1.1 đạt. Mốc thực hiện: **trước Buổi 11**, khi Chặng 5 bắt đầu viết code/test. Nếu chưa có lệnh repo-verified hoặc chưa được cấp quyền cài đặt, ghi `LỆNH MẪU – chưa xác minh` và `CẦN XÁC MINH`, không tự cài.

## 8. Dự đoán trước, kiểm tra sau

1. Nếu `python` và Python313 explicit cho hai đường dẫn khác nhau, lệnh pytest nào đáng tin hơn trong buổi này? Vì sao?
2. Nếu có 110 test collected, có thể kết luận 110 test pass không?
3. Nếu `pandas` thiếu nhưng chỉ khảo sát tree repo, có cần dừng toàn bộ buổi không?
4. Vì sao lượt chỉ-đọc này dùng collect-only thay vì chạy toàn bộ test?

Gợi ý: dùng interpreter đã xác định bằng `sys.executable`; collected không phải passed; phần đọc tài liệu/cây repo vẫn làm được nhưng task cần package phải ghi thiếu dependency; collect-only giữ baseline chỉ-đọc, còn chạy test thật nằm ở nhánh thực thi T2.1 sau baseline và chưa biến test cũ thành chứng nhận V1.1.

## 9. Output cần có

- Baseline log cá nhân gồm interpreter, package metadata, lệnh và output.
- Bảng “đã xác minh / chưa xác minh / không chạy” cho môi trường.
- Bảng kỳ vọng test cũ so với V1.1, kèm `git status --short` và trạng thái `.venv`.
- Danh sách test collection; không gọi là kết quả pass.

Tên file log cụ thể ngoài `docs/learning/` chưa được project chốt; nếu cần lưu artifact chung, ghi `CẦN XÁC MINH` thay vì tự tạo namespace.

## 10. Kiểm tra hiểu bài

1. Interpreter khác venv thế nào? **Đáp án gợi ý:** interpreter là chương trình Python chạy lệnh; venv là môi trường tách biệt thường chứa interpreter/package liên quan. **Chấm:** phân biệt được hai vai trò.
2. Vì sao `WindowsApps` alias là rủi ro? **Đáp án gợi ý:** nó có thể không phải interpreter project đang dùng. **Chấm:** biết kiểm tra `Source` và `sys.executable`.
3. `--collect-only` kiểm tra điều gì? **Đáp án gợi ý:** pytest tìm và thu thập test; chưa chạy assertion. **Chấm:** không dùng từ “pass”.
4. Khi package thiếu, bài học yêu cầu làm gì? **Đáp án gợi ý:** ghi lại, phân loại environment và chờ quyết định setup; không tự cài. **Chấm:** giữ nguyên nguyên tắc chỉ đọc.
5. Tại sao đọc imports quan trọng trước khi lập dependency profile? **Đáp án gợi ý:** profile dựa trên code thật, không dựa vào danh sách mong muốn. **Chấm:** nêu được tính truy vết.
6. `git status --short` và `Test-Path .venv` giúp phân biệt điều gì? **Đáp án gợi ý:** trạng thái thay đổi của repo và sự tồn tại của venv; cả hai chỉ quan sát. **Chấm:** không coi `True` là venv đã cấu hình đúng.
7. Vì sao một test xanh không có nghĩa là khớp V1.1? **Đáp án gợi ý:** test chỉ chứng minh hành vi mà nó viết ra; nếu tên bucket hoặc kỳ vọng cũ lệch D3 thì test vẫn có thể xanh nhưng không phải chuẩn V1.1. **Chấm:** viện dẫn được test mẫu và §3.3/D3.

## 11. Bài tự giải thích

Viết một checklist 8 dòng cho bạn cùng lớp để kiểm tra “mở terminal mới vẫn dùng đúng interpreter”. Mỗi dòng phải có một bằng chứng quan sát được.

Rubric 4 điểm:

- 1 điểm: có kiểm tra thư mục root;
- 1 điểm: có `sys.executable` và phiên bản;
- 1 điểm: có package/test collection và bảng kỳ vọng test cũ với ít nhất hai hàng mẫu;
- 1 điểm: phân biệt lỗi environment với lỗi code và không tự sửa ngoài phạm vi.

## 12. Gate của buổi

Qua buổi học khi môi trường và lệnh kiểm tra được ghi rõ, có thể phân biệt interpreter thật với alias, log dùng chữ “collected” chính xác, và có bảng phân loại kỳ vọng test cũ theo V1.1. Nhánh thực thi T2.1 nhận phần ghim dependency và chạy bộ test cũ sau baseline; nếu chưa được cấp quyền thì trạng thái là `CẦN XÁC MINH`, không coi gate đã đạt. Gate Chặng 2 yêu cầu môi trường/lệnh thực thi được ghi; task là T2.1. Constitution đã có, chỉ kiểm tra/reuse, không init đè lần nữa. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 2 — Setup môi trường tái lập]

## 13. Lỗi thường gặp & cách gỡ

- **Environment:** `python` là WindowsApps alias → dùng explicit interpreter sau khi giảng viên xác minh đường dẫn.
- **Environment:** pytest collection lỗi import → ghi traceback, kiểm tra package/interpreter, không sửa module.
- **Scope:** chạy `pytest` không có `--collect-only` trong lượt chỉ-đọc → dừng; test thật thuộc nhánh thực thi T2.1 sau baseline.
- **Reproducibility:** quên thư mục hiện tại → ghi `Get-Location` vào log.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T2.1
Interpreter / sys.executable:
Python version:
Package metadata:
Lệnh collect-only và số test collected:
Ghi chú phép đếm: 110 = 51 + 59 (21 + 25 + 5 ở schema/evaluation/BM25; 59 ở năm file RAG); collected ≠ passed
Git status --short:
.venv (Test-Path):
Bảng kỳ vọng test cũ ↔ V1.1:
Hiểu được:
Chưa hiểu:
Lỗi (code / data / spec / environment):
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc phần buổi 4 trong roadmap; chuẩn bị quyền truy cập Colab/Drive nếu project yêu cầu. Không tạo checkpoint và không chạy GPU.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: interpreter thật và package profile trên máy sinh viên; đường dẫn Python313 chỉ là máy khảo sát.
- CẦN XÁC MINH: cách tạo/kích hoạt venv được chủ project chốt; buổi này không tự tạo venv.
- CẦN XÁC MINH: số test collection có thay đổi sau khi môi trường được thiết lập.
- CẦN XÁC MINH: lệnh ghim dependency, cách tạo/kích hoạt `.venv` và lệnh chạy bộ test cũ do giảng viên/chủ project cung cấp cho nhánh thực thi T2.1 trước Buổi 11.
