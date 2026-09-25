---
buoi: 21
chang: 7
ten_chang: Dense Retriever
tuan_goi_y: 7
thu_goi_y: Thứ 6 (lịch gợi ý, có thể đổi)
tasks: [T7.1, T7.2]
moi_truong: [Đọc tài liệu, Local, CPU, Colab T4]
thoi_luong: 150 phút hướng dẫn và fixture; smoke/triển khai là nhánh riêng
chay_may_rieng: có; smoke train 1 epoch trên 50 mẫu và kiểm runtime T4 sau khi đủ gate, chưa đo thời gian
nguon:
  - docs/superpowers/plans/2026-09-11-german-qa-learning-roadmap.md §Chặng 7 — Dense Retriever
  - docs/superpowers/plans/2026-09-11-german-qa-task-catalog.md §Phụ thuộc toàn dự án
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.3 Quy tắc xử lý
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.6 Kiểm thử
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.7 Artifact
  - docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.8 Tài nguyên
  - .specify/memory/constitution.md §IV. Tái lập và truy vết được
trang_thai: chưa làm
---

# Buổi 21 — Seed xuyên process, gradient và checkpoint có thể resume

## 1. Mục tiêu học tập

Sau buổi này bạn có thể:

- tính seed/chọn positive và HN theo epoch, đối chiếu kết quả trong hai process khác PYTHONHASHSEED;
- tính gradient toy ở cả hai phía query/passage và phân biệt nó với kiểm encoder thật;
- giải thích vì sao resume cần trạng thái optimizer/RNG/tiến độ, không chỉ weights;
- lập tiêu chí smoke T4, save/load và log micro-batch/accumulation trước khi chạy thật.

## 2. Vị trí trong bức tranh lớn

Buổi 21 kiểm tiếp sampler T7.1 và chuẩn bị T7.2: nếu lựa chọn candidate hoặc resume sai, so sánh retriever cho RQ1 sẽ thiếu khả năng tái lập. SHA-256 seed là quy tắc project; số index fixture phụ thuộc cả `random.Random` của runtime được ghi nhận. Chưa có trainer hoặc test model Dense trong repo, nên các phép tính dưới đây là học liệu độc lập, không thay nghiệm thu T7.2. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.3 Quy tắc xử lý] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-design.md §RQ1 — Ảnh hưởng của quy mô corpus]

## 3. Tiền đề

- Buổi 20: target đơn, softmax/NLL, mask toàn gold_set, một chunk/group sau lọc.
- Python function/dict và làm tròn số; chưa giả định đã biết autograd. Ôn đạo hàm qua công thức §7 bước 3.
- Gate thực thi: T7.1 có code/test đạt và tiền đề T6.4; Colab/GPU/nơi lưu checkpoint từ T2.2 còn phải kiểm trên runtime thật. Metadata local tại lúc soạn chưa có torch; không tự cài chỉ để đọc bài. [kiem-chung-chang-07.md §1. Nguồn và hiện trạng]

## 4. Khái niệm cốt lõi

| Khái niệm | Ví dụ minh họa | Sai lầm hay gặp |
|---|---|---|
| Stable seed / RNG | Seed SHA-256 cố định đầu vào cho bộ sinh số giả ngẫu nhiên RNG; qid và context ID xác định lựa chọn. | Dùng Python hash() có salt khác process. |
| Epoch / sampling | Epoch là một lượt qua tập train; epoch 0 và 1 được phép chọn positive khác nhau. | Bắt buộc mọi epoch chọn khác, dù ngẫu nhiên có thể trùng. |
| Gradient / autograd | Gradient là đạo hàm loss theo tham số; autograd tính đạo hàm theo chuỗi phép tính. | Chỉ thấy loss có số rồi kết luận hai tower đều nhận gradient. |
| Optimizer / checkpoint / resume | Optimizer dùng gradient cập nhật weights; checkpoint lưu trạng thái để tiếp tục từ chỗ dừng. | Chỉ lưu weights nhưng gọi là resume đầy đủ. |
| Gradient accumulation | Cộng gradient qua nhiều micro-batch trước một lần cập nhật. | Gọi micro-batch 4 × accumulation 4 là có 16 query in-batch negative trong một forward. |
| Smoke / VRAM | Smoke là lượt chạy nhỏ kiểm hoạt động; VRAM là bộ nhớ GPU, phải đo trên máy được cấp. | Ghi micro-batch 16 chắc chắn vừa T4 mà chưa đo. |

Spec dự kiến micro-batch 16, có thể giảm 8/4 nếu OOM; fp16 trên T4 và checkpoint mỗi epoch lên Drive. Đây là cấu hình dự kiến cần log thật, không phải phép đo hiện có. [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.4 Cấu hình] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.8 Tài nguyên]

## 5. Kế hoạch buổi

| Phút | Khối | Việc làm |
|---:|---|---|
| 0–20 | Giải thích bằng ví dụ | Gradient, RNG và resume. |
| 20–45 | Xem input | Spec seed/checkpoint và fixture hai candidate. |
| 45–95 | Thao tác từng lệnh | Seed xuyên process, gradient tay, RNG resume trong RAM. |
| 95–125 | Kiểm tra output | Ma trận bằng chứng smoke/save-load, điều kiện T4. |
| 125–150 | Tự giải thích + nhật ký | Rubric và các gate còn chờ. |

Gap ước lượng 180–240 phút cho tải đầy đủ. Phần triển khai trainer/test, setup runtime và máy chạy smoke không nằm trong 150 phút này; phải có log/phiên thực thi riêng, không đánh dấu đã hoàn thành. [docs/learning/02-roadmap-gap-report.md §3. Đối chiếu từng buổi 1–32]

## 6. Input

- Spec §5.3–5.8 và fixture minh họa: `base_seed=42`, qid `fixture-q0`, context ID `fixture-group`, 3 positive/5 HN sau mask; các ID chỉ là nhãn cho phép tính seed, không phải record schema thật.
- Fixture gradient: q=`[1,0]`, p0=`[1,0]`, p1=`[0,1]`, target=0; không model hoặc dataset train.
- `.venv/Scripts/python.exe` hiện có. Runtime local dùng Python 3.13.3 trong log; mọi lệnh bên dưới tự đủ biến. [kiem-chung-chang-07.md §2. Kiểm lệnh và expected]
- Chưa có checkpoint hai tower/optimizer state để đọc. Chưa có trainer CLI được xác minh; không đưa tên module giả như thể đã có.

## 7. Thao tác từng bước

PowerShell tại repo root; bỏ thụt lề để here-string mở/đóng ở cột 0. Các khối PowerShell chỉ tính trong RAM, không train/ghi checkpoint.

1. **Đọc đúng công thức nguồn**:

   ```powershell
   $specLines = Get-Content 'docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md' -Encoding UTF8
   $start = ($specLines | Select-String '^### 5\.3 ').LineNumber
   $end = ($specLines | Select-String '^## 6\. Module:').LineNumber
   $specLines[($start-1)..($end-2)] | Select-String 'def stable_seed|rng = random.Random|chunk_idx = Random|Gradient accumulation|weights \+ optimizer state' | ForEach-Object { $_.Line }
   ```

   Kỳ vọng: công thức positive dùng `stable_seed(query_id)`, HN dùng thêm `source_context_id`; seed cộng base_seed và epoch. Nếu không thấy, kiểm đúng frozen spec/path, không tự thay công thức.

2. **Dự đoán cùng seed khác process có đổi index không**, rồi chạy:

   ```powershell
   @'
   import json, os, subprocess, sys
   child = """
   import hashlib, json, random
   def stable_seed(*parts):
       return int(hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:8],16)
   qid, gid = 'fixture-q0', 'fixture-group'
   rows = []
   for epoch in (0,1):
       pos = random.Random(42+epoch+stable_seed(qid)).randrange(3)
       hn = random.Random(42+epoch+stable_seed(qid,gid)).randrange(5)
       rows.append([epoch,pos,hn])
   print(json.dumps({'seeds':[stable_seed(qid),stable_seed(qid,gid)],'choices':rows},sort_keys=True))
   """
   import textwrap
   outputs = []
   for salt in ('1','9876'):
       env = dict(os.environ, PYTHONHASHSEED=salt)
       result = subprocess.run([sys.executable,'-B','-c',textwrap.dedent(child)],env=env,text=True,capture_output=True,check=True)
       outputs.append(json.loads(result.stdout))
   print(outputs[0])
   print('same_across_processes', outputs[0] == outputs[1])
   assert outputs[0] == outputs[1]
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng trên Python 3.13.3: seeds `[3588503323,786639495]`; choices `[[0,0,1],[1,2,0]]`; `same_across_processes True`. SHA-256 seed cố định; index của `random.Random` phải được đối chiếu khi đổi Python/runtime. Epoch khác **được phép**, không bắt buộc đổi lựa chọn. Nếu khác giữa hai process cùng runtime: kiểm công thức, thứ tự list sau mask, nguồn RNG và PYTHONHASHSEED; không đổi expected để che lỗi. HN đã lọc gold trước khi lấy index, đúng buổi 20.

3. **Tính gradient bằng tay trước**, không chạy optimizer. Score `[1,0]`, `p0=exp(1)/(exp(1)+1)`, loss=`-log(p0)`. Đạo hàm theo score là `[p0-1,1-p0]`; vì score=q·p nên `dL/dq=Σ_j (dL/dscore_j)*p_j`, `dL/dp_j=(dL/dscore_j)*q`.

   ```powershell
   @'
   import math
   prob = math.exp(1)/(math.exp(1)+1)
   ds = [prob-1, 1-prob]
   dq = [ds[0],ds[1]]
   dp = [[ds[0],0.0],[ds[1],0.0]]
   print('loss', f'{-math.log(prob):.6f}')
   print('dq', [round(v,6) for v in dq])
   print('dp', [[round(v,6) for v in row] for row in dp])
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: loss `0.313262`; dq `[-0.268941,0.268941]`; dp `[[-0.268941,0.0],[0.268941,0.0]]`. Cả hai phía có đạo hàm khác zero trong ví dụ; đây chưa kiểm `.grad` của encoder thật. Nếu khác, kiểm target=0, dấu của `p-target` và dot product không normalize.

   **LỆNH MẪU – chưa xác minh trên runtime thật:** cell Python khi đã có PyTorch được duyệt. Không chạy tại local hiện thiếu torch; không tải model. Dự kiến autograd cho cùng số bước 3, không có bước train/optimizer:

   ```python
   import torch
   q = torch.tensor([1.0,0.0], requires_grad=True)
   p = torch.tensor([[1.0,0.0],[0.0,1.0]], requires_grad=True)
   loss = -torch.log_softmax(p @ q, dim=0)[0]
   loss.backward()
   print(loss.item(), q.grad, p.grad)
   ```

   Sai khác nhỏ float32 so với tính tay float64 là có thể; `.grad=None` thì kiểm `requires_grad`, detach/no_grad hoặc đường tính loss. Không gọi hai tensor toy là hai tower đã nghiệm thu; test T7.1 phải kiểm model thật theo §5.6.

4. **Dự đoán resume RNG thiếu state**, rồi phục hồi state trong RAM. Sampler bước 2 khởi tạo RNG theo query/epoch; fixture dưới minh họa RNG tuần tự của một lượt chạy, không đề nghị thay sampler bằng RNG toàn cục:

   ```powershell
   @'
   import random
   rng = random.Random(42)
   prefix = [rng.randrange(7) for _ in range(2)]
   state = rng.getstate()
   expected = [rng.randrange(7) for _ in range(5)]
   restored = random.Random(0)
   restored.setstate(state)
   actual = [restored.randrange(7) for _ in range(5)]
   restarted = random.Random(42)
   wrong = [restarted.randrange(7) for _ in range(5)]
   print('prefix', prefix)
   print('uninterrupted', expected, 'resumed', actual, 'match', actual==expected)
   print('restart_from_seed', wrong, 'match', wrong==expected)
   '@ | & '.venv/Scripts/python.exe' -B -
   ```

   Kỳ vọng: prefix `[5,0]`; uninterrupted/resumed `[0,5,2,1,1]`, match True; restart `[5,0,0,5,2]`, match False. Khôi phục seed đầu không đồng nghĩa khôi phục tiến độ. Đây chỉ là RNG Python; checkpoint thật cần mọi RNG/trạng thái thực sự được dùng. Nếu khác, kiểm chỗ chụp state và số lần draw trước/sau, không gọi extra draw để in debug.

5. **Lập bảng bằng chứng trước chạy T4**, mỗi hàng ghi input, expected và trạng thái chưa làm:

   | Kiểm | Expected/điều kiện theo spec | Bằng chứng cần lưu |
   |---|---|---|
   | Hai tower | Gradient tới tham số cả question/passages sau backward | Test log model thật; tensor toy không thay thế |
   | Sampling/mask | Toàn gold_set, ≤1/group, hai process cùng seed cùng lựa chọn | Test T7.1, phiên bản/runtime |
   | Smoke | 1 epoch/50 mẫu, similarity positive tăng so trước | Cấu hình, mẫu đã chọn từ train, số trước/sau, VRAM/lỗi |
   | Save/load | Encode lại 10 passage; chênh similarity <1e-5 | Đúng checkpoint/input/mode, số sai lệch thực đo |
   | Resume | Weights, optimizer state, epoch/step và RNG state khôi phục | Mốc dừng/tiếp tục và đối chiếu lượt không ngắt |

   Resume chi tiết và tên file cần chốt ở thẻ T7.2; không bịa format optimizer/checkpoint. Ghi model/tokenizer revision, base_seed/công thức sampler, hyperparameter, micro-batch thật và accumulation riêng. **4×4 không tạo 16 query negative trong một forward.** [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.6 Kiểm thử] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.7 Artifact] [docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md §5.8 Tài nguyên]

6. **Nhánh thực thi T7.2 tại buổi này:** sau T7.1 pass, duyệt thẻ trainer/test/runtime trước cài đặt hoặc smoke. Kiểm GPU thật và nơi lưu checkpoint có thể đọc lại sau kết nối; chưa có bằng chứng thì gate runtime vẫn chờ. Cell Colab sau là **LỆNH MẪU – chưa xác minh trên runtime thật**, không chạy dấu `!` trong PowerShell:

   ```python
   !nvidia-smi
   ```

   Kỳ vọng là bảng thiết bị thực được cấp, **không hứa chắc T4**. Không có GPU hoặc khác dự kiến: ghi lại và chốt tài nguyên trước smoke; không tự thay đổi spec để vượt gate. Nếu được chạy smoke theo thẻ, lưu đầy đủ hàng bước 5. Training đầy đủ/chọn checkpoint chuyển buổi 22, không coi smoke là kết quả P2.

## 8. Dự đoán trước, kiểm tra sau

1. Hai PYTHONHASHSEED khác nhau có làm stable_seed đổi không? **Không vì dùng SHA-256; hai lựa chọn fixture cùng runtime phải giống.** Chấm: phân biệt hash built-in và SHA-256.
2. Epoch 0/1 cùng chọn một index có tự động là lỗi không? **Không; ngẫu nhiên được phép trùng.** Chấm: kiểm công thức trước, không yêu cầu khác bằng mọi giá.
3. Chỉ khởi tạo lại seed=42 có tái hiện phần tiếp theo của RNG không? **Không trong fixture đã draw hai lần; cần đúng state/tiến độ.** Chấm: chỉ ra prefix và kết quả restart sai.

## 9. Output cần có

Đường dẫn **đề xuất cho người học, chưa tạo**: `docs/learning/lessons/chang-07/thuc-hanh/buoi-21-nhat-ky.md`, chứa:

- hai process salts, seeds/choices, Python version và kết quả so sánh;
- loss/gradient tính tay, trạng thái cell autograd nếu chưa chạy;
- RNG prefix/state/resume expected và bảng năm nhóm bằng chứng bước 5;
- danh sách checkpoint state, micro-batch/accumulation, thiết bị/path/revision còn cần xác minh.

Sau thực thi được duyệt mới có test/smoke log và checkpoint có thể resume ở nơi đã chốt; không sinh file checkpoint giả từ bảng ghi chú.

## 10. Kiểm tra hiểu bài

1. Tại sao HN seed có source_context_id? **Để lựa chọn theo từng group/context đúng công thức, không dùng một lựa chọn chung cho mọi group.** Chấm: viết đủ các thành phần seed.
2. Gold bị lọc trước hay sau chọn HN? **Trước; group hết chunk không đóng góp.** Chấm: không randrange trên group rỗng.
3. `.grad` của một tower None cần kiểm gì? **Đường tính loss, detach/no_grad, requires_grad và tham số thực được dùng.** Chấm: nêu hai kiểm tra thay vì kết luận từ loss hữu hạn.
4. Accumulation 4 với micro-batch 4 cho bao nhiêu query cùng forward? **4, không phải 16.** Chấm: tách tổng gradient khỏi negative trong forward.
5. Save/load và resume khác gì? **Save/load kiểm output mô hình sau lưu/nạp; resume còn cần optimizer, RNG và tiến độ để tiếp tục train.** Chấm: nêu cả hai phạm vi.
6. Similarity toy tăng có đủ qua smoke không? **Chưa: cần model thật, 50 mẫu train/1 epoch và log; chưa phải P2.** Chấm: không gán kết quả minh họa cho project.

## 11. Bài tự giải thích

Vẽ điểm ngắt giữa một epoch: đánh dấu cái gì đã lưu, lần draw/step nào tiếp theo, và điều gì sai nếu chỉ nạp weights. Giải thích riêng vì sao accumulation không bổ sung negative giữa các micro-batch.

Rubric 4 điểm: seed/group đúng; gradient hai tower; đủ state/tiến độ; giới hạn CPU toy/smoke/P2. Mỗi tiêu chí 1 điểm, chưa đánh dấu đạt.

## 12. Gate của buổi

Gate hiểu bài: tái hiện seed xuyên process và RNG resume, tính đúng gradient, lập đủ bảng bằng chứng và ghi các cell chưa chạy. Gate thực thi: T7.1 pass, runtime/checkpoint path được kiểm, smoke/save-load/resume có bằng chứng theo spec. **Fixture RNG/gradient hoặc lệnh nvidia-smi riêng lẻ không mở gate train chính thức.** Nghiệm thu Chặng 7 còn chờ P2/index ở buổi 22.

## 13. Lỗi thường gặp & cách gỡ

- **Code:** dùng hash() cho seed → đối chiếu SHA-256 và kiểm hai process khác salt.
- **Data:** lấy index HN trước lọc gold → lọc trước sampling; bỏ group rỗng.
- **Code:** resume lệch lượt → kiểm optimizer/RNG/epoch-step/cursor ở đúng điểm ngắt theo thẻ.
- **Environment:** Colab không có T4/Drive path chưa bền → ghi thiết bị/path thực, chưa chạy smoke khi thiếu tiền đề.
- **Spec:** gọi accumulation là tăng negative → vẽ riêng từng forward, không gộp candidate sau phép cộng gradient.

## 14. Mẫu nhật ký buổi học

```text
Ngày:
Task: T7.1 kiểm seed/mask; T7.2 gradient/smoke/resume
Hiểu/chưa hiểu:
Lệnh và runtime đã chạy; lệnh mẫu chưa chạy:
Seed, epoch, query/context ID, hai process outputs:
Gradient tính tay và autograd (nếu đã chạy):
State/điểm dừng/resume:
Thiết bị/VRAM, micro-batch/accumulation (chưa đo thì ghi chưa đo):
Output path/checkpoint path:
Lỗi và cách gỡ:
Gate hiểu bài / gate thực thi: chưa làm
Bước tiếp theo:
```

## 15. Việc chuẩn bị cho buổi sau

Ôn Recall@20 và fingerprint ở buổi 17–18. Mang bảng smoke/resume; thiếu gate thì buổi 22 chỉ làm fixture selection/index, không train hoặc chấm checkpoint thật.

## 16. Điểm cần xác minh

- CẦN XÁC MINH: thẻ T7.1/T7.2, file/API/test/CLI và format checkpoint/resume — người triển khai chốt trước code, không suy từ tensor toy.
- CẦN XÁC MINH: Colab/GPU/Drive, bộ phiên bản tương thích và model/tokenizer revision — chủ project/người vận hành kiểm runtime thật; không suy từ .venv CPU.
- CẦN XÁC MINH: log smoke/save-load/resume trên model thật — thực hiện trong nhánh được duyệt trước training chính thức.
- Spec §5.6 có câu “cả 9 điều” nhưng liệt kê 10 mục kiểm trước dòng nghiệm thu; dùng từng yêu cầu làm checklist, không bỏ mục để khớp số đếm. Nếu cần sửa câu chữ, chủ project xử lý amendment, không sửa frozen.
