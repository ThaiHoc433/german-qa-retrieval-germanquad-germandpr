---
buoi: 4
chang: 2
ten_chang: Setup môi trường tái lập
tuan_goi_y: 2
thu_goi_y: Thứ 2 (lịch gợi ý, có thể đổi)
tasks: [T2.2]
moi_truong: [Local, Colab T4]
thoi_luong: 120 phút
chay_may_rieng: không
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 2 — Setup môi trường tái lập
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §T2.2
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.8 Tài nguyên
  - .specify/memory/constitution.md §Phạm vi và nguồn chuẩn
  - docs/learning/00-project-analysis.md §8. Điểm cần xác minh trước học liệu chi tiết
trang_thai: chưa làm
---

# Buổi 4 — Kiểm tra môi trường tái lập và Colab

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- mô tả checklist mở terminal mới và kiểm tra lại interpreter đã ghi ở buổi 3;
- phân biệt kiểm tra GPU được cấp với việc train mô hình;
- xác định nơi lưu checkpoint dự kiến và trạng thái root Spec Kit mà không init đè;
- ghi rõ điều nào đã quan sát, điều nào chưa có quyền xác minh và điều nào cần chờ smoke T4.

## 2. Vị trí trong bức tranh lớn

Buổi 4 hoàn tất phần setup của Chặng 2 và phục vụ T2.2. Nó bám roadmap ở các mục kích hoạt môi trường từ terminal mới, thiết lập notebook Colab, kiểm tra GPU thực được cấp, nơi lưu checkpoint và Spec Kit root `nlp`; trong lượt học liệu này chỉ quan sát và ghi checklist. Nhánh thực thi T2.2 của chính buổi này nhận việc đóng/mở terminal, chạy lại interpreter và bộ test, thiết lập notebook/Colab, xác minh GPU và chốt checkpoint location sau khi T2.1 đã cung cấp môi trường. Không chạy train và không tải Wikipedia trong lượt soạn học liệu. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 2 — Setup môi trường tái lập]

## 3. Tiền đề

- Buổi 3: có baseline interpreter/package/collection hoặc đã ghi rõ lỗi môi trường.
- Biết checkpoint là file/trạng thái sẽ được lưu sau train; chưa cần biết nội dung optimizer/RNG.
- Có tài khoản/quyền Colab và Drive chỉ khi chủ project cung cấp; nếu chưa có, học theo checklist local.

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ cụ thể | Sai lầm hay gặp |
|---|---|---|
| Terminal mới | Một phiên terminal mới dùng để xem setup có phụ thuộc trạng thái phiên trước không. | Chỉ kiểm tra trong terminal cũ đã có biến môi trường ngầm. |
| Colab | Môi trường notebook tách khỏi local; cần ghi interpreter và nơi lưu artifact. | Coi mở được notebook là đã có GPU T4. |
| GPU thực được cấp | Thiết bị mà runtime hiện tại báo; phải quan sát bằng lệnh kiểm tra, không suy từ tên notebook. | Ghi “T4” chỉ vì spec nói train trên T4. |
| Checkpoint location | Nơi dự kiến lưu checkpoint và resume; chưa phải checkpoint đã tồn tại. | Tạo hoặc tải checkpoint trong buổi setup. |
| Spec Kit root | Root `.specify`/project mà tài liệu dùng; cần kiểm tra/reuse. | Init đè constitution hoặc tạo root mới khi root đã có. |
| Smoke | Kiểm tra nhỏ để biết môi trường có khả năng chạy; không phải train chính thức hay gate điểm số. | Dùng smoke để tuyên bố Dense/Reader đã đạt P2. |

Spec ghi train Dense/Reader trên Colab T4 và encode có thể dùng T4/local RTX 3050, nhưng đây là tài nguyên mục tiêu; máy học phải kiểm tra thực tế. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.8 Tài nguyên]

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | Phân biệt local, Colab, GPU được cấp, checkpoint và smoke. |
| 20–40 | Xem input | Đọc T2.2, constitution và trạng thái `.specify` hiện có. |
| 40–70 | Thao tác từng lệnh | Kiểm tra root/interpreter local; đọc metadata `.specify`; nếu có runtime được cấp thì chạy lệnh mẫu quan sát GPU. |
| 70–100 | Kiểm tra output | Hoàn thành setup checklist và đánh dấu “chưa xác minh” đúng chỗ. |
| 100–120 | Tự giải thích + nhật ký | Nói vì sao setup pass không đồng nghĩa pipeline pass. |

## 6. Input

- `docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md` — Chặng 2.
- `.specify/memory/constitution.md` — nguồn chuẩn và quy trình.
- Thư mục `.specify/` hiện có trong project.
- Local interpreter từ buổi 3; Colab/Drive chỉ khi quyền truy cập được cung cấp.

## 7. Thao tác từng bước

1. Mở terminal mới tại root và ghi:

   ```powershell
   Get-Location
   Get-Command python -ErrorAction SilentlyContinue | Select-Object Name,Source
   ```

   Kỳ vọng: vị trí là root project và source được ghi rõ. Nếu khác buổi 3, ghi khác biệt; không sửa môi trường âm thầm.
2. Đối chiếu interpreter explicit:

   ```powershell
   & 'C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe' -B -c "import sys; print(sys.executable); print(sys.version)"
   ```

   Kỳ vọng: kết quả có thể lặp lại buổi 3. Nếu đường dẫn không tồn tại trên máy khác, ghi `CẦN XÁC MINH` và dừng lệnh đó.
3. Kiểm tra root Spec Kit bằng lệnh chỉ-đọc:

   ```powershell
   Get-ChildItem -LiteralPath '.specify' -Force | Select-Object Name,Mode
   Get-Content -LiteralPath '.specify/memory/constitution.md' -Encoding UTF8 -TotalCount 25
   ```

   Kỳ vọng: thấy root `.specify` và constitution; không chạy script init, không sửa constitution.
4. Nếu có một runtime Colab được chủ project cấp, chỉ chạy các lệnh quan sát trong notebook/runtime đó:

   ```python
   import sys
   print(sys.executable)
   print(sys.version)
   ```

   Kỳ vọng: ghi interpreter runtime. **LỆNH MẪU – chưa xác minh trên runtime thật:**

   ```python
   !nvidia-smi
   ```

   Kỳ vọng: nếu runtime cho phép, output hiển thị GPU đang được cấp; nếu lệnh không có hoặc không có GPU, ghi nguyên văn lỗi và `CẦN XÁC MINH`. Không xem ví dụ này là bằng chứng Colab T4 đã sẵn sàng và không chạy lệnh này trên local PowerShell.
5. Lập bảng setup:

   | Mục | Đã quan sát | Chưa xác minh |
   |---|---|---|
   | Local root/interpreter |  |  |
   | `.specify`/constitution |  |  |
   | Colab interpreter |  |  |
   | GPU thực được cấp |  |  |
   | Nơi lưu checkpoint |  |  |

   Kỳ vọng: GPU/checkpoint có thể còn “chưa xác minh”; không điền T4/Drive theo suy đoán.
6. Ở **nhánh thực thi T2.2 sau baseline**, đóng và mở lại terminal/notebook rồi chạy lại interpreter và bộ test bằng lệnh đã thống nhất ở Buổi 3; ghi `passed/failed/error/skipped` riêng, không gọi mở được terminal là test pass. Trong notebook, thiết lập runtime/Drive theo hướng dẫn của chủ project, chạy lại `!nvidia-smi`, ghi GPU thực tế và checkpoint location. **Mốc trước Buổi 20 là quyết định tổ chức học liệu**, đặt để Chặng 7 có môi trường smoke/train; roadmap chỉ nêu việc cần làm, không quy định ngày này. Các lệnh cài đặt/thiết lập chưa được đối chiếu runtime thật phải mang nhãn `LỆNH MẪU – chưa xác minh`.
7. Ghi điều kiện chuyển tiếp: có thể học Chặng 3 bằng CPU/cache nếu raw fixture đọc được; train GPU và nơi lưu checkpoint chỉ được dùng sau khi nhánh thực thi T2.2 ghi được bằng chứng xác minh.

## 8. Dự đoán trước, kiểm tra sau

1. Nếu local có Python đúng nhưng Colab chưa có GPU, có thể học toàn bộ Chặng 3 không?
2. Nếu `.specify/memory/constitution.md` đã tồn tại, vì sao không chạy init lại?
3. Nếu runtime báo một GPU khác T4, bài học nên ghi gì?

Gợi ý: Chặng 3 đọc record/cache có thể tiếp tục nếu input hợp lệ; root đã có thì kiểm tra/reuse, không init đè; ghi thiết bị quan sát thực tế và `CẦN XÁC MINH` tác động tới smoke/train, không đổi spec để khớp mong muốn.

## 9. Output cần có

- Setup checklist local/Colab với trạng thái “đã quan sát/chưa xác minh”.
- Log root `.specify` và constitution đã đọc, không chỉnh sửa.
- Ghi chú nơi lưu checkpoint dự kiến hoặc mục `CẦN XÁC MINH` nếu chủ project chưa cung cấp.
- Nếu đã thực hiện nhánh T2.2: bằng chứng đóng/mở terminal vẫn giữ interpreter và kết quả bộ test; log GPU/checkpoint với run date. Nếu chưa thực hiện, ghi rõ `CẦN XÁC MINH`.

Không tạo checkpoint, không tải model/dataset, không init Spec Kit.

## 10. Kiểm tra hiểu bài

1. Smoke khác train chính thức thế nào? **Đáp án gợi ý:** smoke kiểm tra khả năng chạy nhỏ; train tạo checkpoint và cần điều kiện/tài nguyên/gate riêng. **Chấm:** không gọi smoke là nghiệm thu P2.
2. Vì sao phải mở terminal mới? **Đáp án gợi ý:** phát hiện setup chỉ đúng nhờ trạng thái cũ. **Chấm:** nêu được tính tái lập.
3. Constitution đã có thì thao tác đúng là gì? **Đáp án gợi ý:** đọc và reuse; không init đè. **Chấm:** bảo vệ file nguồn chuẩn.
4. Nếu chưa có Drive, có nên tự chọn thư mục checkpoint? **Đáp án gợi ý:** ghi chưa xác minh và chờ quyết định; không bịa artifact path. **Chấm:** phân biệt kế hoạch với hiện trạng.
5. T4 trong spec có chứng minh máy hiện tại được cấp T4 không? **Đáp án gợi ý:** không; phải kiểm runtime thực. **Chấm:** nhận biết mục tiêu tài nguyên khác bằng chứng.

## 11. Bài tự giải thích

Giải thích bằng sơ đồ trạng thái: `chưa xác minh → kiểm tra local → kiểm tra runtime → đủ điều kiện smoke / cần hỗ trợ`. Đặt một ví dụ lỗi vào mỗi nhánh.

Rubric 4 điểm:

- 1 điểm: có trạng thái local và runtime riêng;
- 1 điểm: có bằng chứng quan sát cho mỗi chuyển trạng thái;
- 1 điểm: không biến T4/checkpoint dự kiến thành hiện trạng;
- 1 điểm: chỉ ra hành động an toàn khi thiếu quyền hoặc thiếu thiết bị.

## 12. Gate của buổi

Qua buổi học khi sinh viên có setup checklist, biết root Spec Kit đã tồn tại, và ghi rõ kết quả quan sát. Nhánh thực thi T2.2 phải được gắn trạng thái đã làm hoặc `CẦN XÁC MINH` cho việc đóng/mở terminal vẫn giữ interpreter và bộ test, thiết lập Colab/GPU, checkpoint location; kế hoạch học liệu đặt mốc hoàn tất trước Buổi 20 nhưng không coi đó là gate mới của roadmap. Không được suy đoán T4/Drive. Không init đè constitution. Đây là gate môi trường của Chặng 2 và task T2.2, chưa phải gate train hoặc benchmark. [docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 2 — Setup môi trường tái lập]

## 13. Lỗi thường gặp & cách gỡ

- **Environment:** Colab không có runtime/GPU → ghi chưa xác minh; tiếp tục phần local nếu độc lập.
- **Environment:** terminal mới dùng interpreter khác → so `Source` và `sys.executable` với buổi 3.
- **Environment:** terminal mới mở được nhưng test chưa chạy → chạy lại bộ test ở nhánh thực thi T2.2 bằng interpreter đã ghim; ghi trạng thái thật, không đánh dấu pass theo suy đoán.
- **Spec:** muốn sửa constitution để hợp môi trường → không sửa file frozen/quản trị.
- **Scope:** tạo checkpoint hoặc tải model trong setup → dừng; đó là công việc máy riêng ở chặng sau.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T2.2
Local root/interpreter:
.specify/constitution đã đọc:
Colab interpreter/GPU (nếu có):
Nơi lưu checkpoint: đã xác minh / CẦN XÁC MINH
Bằng chứng đóng/mở terminal và chạy lại test:
Thiết lập notebook/Drive: đã xác minh / CẦN XÁC MINH
Hiểu được:
Chưa hiểu:
Lỗi (code / data / spec / environment):
Bước tiếp theo:
Trạng thái: chưa làm
```

## 15. Việc chuẩn bị cho buổi sau

Đọc phần Chặng 3 và chuẩn bị mở một record Parquet đã cache nếu đường dẫn còn tồn tại. Không tải lại dataset chỉ để minh họa.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: quyền Colab/Drive và GPU thực được cấp.
- CẦN XÁC MINH: nơi lưu checkpoint và chính sách resume của project.
- CẦN XÁC MINH: đường dẫn cache Parquet/interpreter trên máy sinh viên.
- CẦN XÁC MINH: có cần một smoke runtime riêng hay không; không tự thêm gate điểm số.
