# Lịch sử hội thoại (xuất từ transcript)


---

## 🧑 USER #1

<pasted_content id="d211">

Hàm C: count_check(name, ds)

┌───────────┬───────────────────────────────────────────────┐
│    Mục    │                   Nội dung                    │
├───────────┼───────────────────────────────────────────────┤
│ Giai đoạn │ Giai đoạn 1, Bước 1                           │
├───────────┼───────────────────────────────────────────────┤
│ Kỹ thuật  │ Đối chiếu số liệu thực tế với số liệu kỳ vọng │
└───────────┴───────────────────────────────────────────────┘

def count_check(name, ds):
    rows = []
    for split, expected_n in DATASETS[name].expected.items():
        actual_n = len(ds[split]) nếu split tồn tại, ngược lại None
        rows.append({
          dataset, split, expected: expected_n, actual: actual_n,
          diff: actual_n - expected_n,
          status: "OK" nếu diff == 0, "LỆCH" nếu khác, "THIẾU SPLIT" nếu None
        })
    # liệt kê thêm các split có thật nhưng không nằm trong expected (vd: validation)
    return rows

Giải trình: đừng dùng assert làm dừng chương trình khi lệch. Hãy ghi LỆCH vào bảng rồi chạy tiếp. Lệch là một phát hiện cần đưa vào báo cáo, không phải lỗi chương trình.
    for split, expected_n in DATASETS[name].expected.items():
        actual_n = len(ds[split]) nếu split tồn tại, ngược lại None
        rows.append({
          dataset, split, expected: expected_n, actual: actual_n,
          diff: actual_n - expected_n,
          status: "OK" nếu diff == 0, "LỆCH" nếu khác, "THIẾU SPLIT" nế
        })
    # liệt kê thêm các split có thật nhưng không nằm trong expected (vd
    return rows

Giải trình: đừng dùng assert làm dừng chương trình khi lệch. Hãy ghi LỆCH vào bảng rồi chạy tiếp. Lệch là một phát hiện cần đưa vào báo cáo, không phải lỗi chương trình.
</pasted_content id="d211">

, tao đang code hàm này cho project, giải thích lỗi sa i


---

## 🤖 CLAUDE #2

Đoạn bạn dán là pseudocode, nên có mấy chỗ không chạy được nếu chép nguyên sang Python.

## Lỗi sai

1. **`nếu ... ngược lại` không phải cú pháp Python.** Phải viết `len(ds[split]) if split in ds else None`.
2. **`diff = actual_n - expected_n` sẽ nổ `TypeError` khi `actual_n` là `None`.** Đây là lỗi nghiêm trọng nhất. Trường hợp "THIẾU SPLIT" chính là lúc `actual_n = None`, nên phải xử lý `None` trước khi trừ.
3. **Dict viết sai cú pháp.** `{dataset, split, expected: expected_n, ...}` thiếu key dạng chuỗi, và `dataset, split` đứng một mình sẽ thành set hoặc lỗi cú pháp.
4. **Phần "liệt kê split dư" chỉ là comment, chưa có code.** Hàm sẽ bỏ sót các split như `validation` có trong `ds` nhưng không có trong `expected`.
5. **`DATASETS[name].expected` dùng truy cập thuộc tính.** Nó chỉ chạy nếu `DATASETS[name]` là object/dataclass. Nếu là dict thì phải viết `DATASETS[name]["expected"]`.
6. **Đoạn dán bị lặp và cắt cụt** (`"THIẾU SPLIT" nế`, `(vd`). Đừng copy phần lặp đó vào code.

## Bản đúng

```python
def count_check(name, ds):
    rows = []
    expected = DATASETS[name].expected  # hoặc DATASETS[name]["expected"]

    for split, expected_n in expected.items():
        actual_n = len(ds[split]) if split in ds else None

        if actual_n is None:
            diff, status = None, "THIẾU SPLIT"
        else:
            diff = actual_n - expected_n
            status = "OK" if diff == 0 else "LỆCH"

        rows.append({
            "dataset": name, "split": split,
            "expected": expected_n, "actual": actual_n,
            "diff": diff, "status": status,
        })

    # split có thật nhưng không nằm trong expected (vd: validation)
    for split in ds:
        if split not in expected:
            rows.append({
                "dataset": name, "split": split,
                "expected": None, "actual": len(ds[split]),
                "diff": None, "status": "NGOÀI KỲ VỌNG",
            })

    return rows
```

Hàm không dùng `assert`, nên khi lệch nó vẫn ghi vào bảng và chạy tiếp, đúng như phần "Giải trình" yêu cầu.

Nếu bạn có code thật đang chạy và báo lỗi, dán code hoặc traceback lên, tôi sẽ chỉ chính xác dòng sai.


---

## 🧑 USER #3

Hàm này chức năng để làm gì


---

## 🤖 CLAUDE #4

`count_check` **kiểm tra xem số lượng mẫu thực tế của từng split trong dataset có khớp với số lượng bạn kỳ vọng không**, và trả về kết quả dưới dạng bảng.

## Nó làm gì

Khi load một dataset (ví dụ từ HuggingFace), bạn thường biết trước số mẫu mà tài liệu hoặc paper công bố, ví dụ `train = 8000`, `test = 1000`. Hàm này so sánh số liệu đó với số thực tế:

| split | expected | actual | diff | status |
|---|---|---|---|---|
| train | 8000 | 8000 | 0 | OK |
| test | 1000 | 987 | -13 | LỆCH |
| dev | 500 | None | None | THIẾU SPLIT |
| validation | None | 500 | None | NGOÀI KỲ VỌNG |

## Tại sao cần

- **Phát hiện dữ liệu bị sai hoặc thiếu**: tải lỗi, phiên bản dataset khác, lọc nhầm, mất mẫu.
- **Bước đầu tiên của pipeline**: đây là "Giai đoạn 1, Bước 1", tức kiểm tra tính toàn vẹn dữ liệu trước khi huấn luyện hay đánh giá.
- **Phục vụ báo cáo**: kết quả lệch là một phát hiện cần ghi lại, không phải lỗi chương trình. Vì vậy hàm không dùng `assert` để dừng, mà ghi trạng thái vào bảng rồi chạy tiếp.

## Đầu vào và đầu ra

- **Input**: `name` là tên dataset (để tra `DATASETS[name].expected`), và `ds` là dataset đã load (dict các split).
- **Output**: danh sách các dòng (`rows`), mỗi dòng ứng với một split. Có thể chuyển thẳng sang `pandas.DataFrame` để in hoặc đưa vào báo cáo.


---

## 🧑 USER #5

<pasted_content id="d211">
Hàm D: write_manifest(...)

┌───────────┬───────────────────────────────────────────┐
│    Mục    │                 Nội dung                  │
├───────────┼───────────────────────────────────────────┤
│ Giai đoạn │ Giai đoạn 1, Bước 1                       │
├───────────┼───────────────────────────────────────────┤
│ Kỹ thuật  │ Lưu manifest (siêu dữ liệu tải) dạng JSON │
└───────────┴───────────────────────────────────────────┘

def write_manifest(all_info):
    manifest = {
      "download_time": thời điểm hiện tại (ISO),
      "datasets": {
         name: { repo_id, revision_sha, splits: {split: số dòng},
                 columns: {split: tên cột + kiểu} }
      },
      "library_versions": { datasets, huggingface_hub, python }
    }
    ghi ra REPORT_DIR / "step1_manifest.json"

Ghi thêm cột columns (tên cột và kiểu, lấy từ ds[split].features) vì Bước 2 sẽ dùng lại.
</pasted_content id="d211">

, hãy hướng dẫn tao thực hiện cài đặt hàm này và giải thích ý nghĩa của hàm trong step của project


---

## 🤖 CLAUDE #6

## Ý nghĩa trong project

`write_manifest` ghi lại **"biên lai" của lần tải dữ liệu**: tải cái gì, từ đâu, phiên bản nào, lúc nào, bằng thư viện nào. Hàm này quan trọng vì ba lý do:

| Mục trong manifest | Vì sao cần |
|---|---|
| `revision_sha` | Dataset trên HuggingFace có thể bị cập nhật. SHA ghim đúng phiên bản bạn đã dùng, nên 3 tháng sau vẫn tái lập được kết quả. |
| `splits` (số dòng) | Bằng chứng số liệu thực tế, đối chiếu với kết quả của `count_check`. |
| `columns` (tên + kiểu) | **Bước 2 đọc lại file này** để biết schema, nên không cần load lại dataset chỉ để xem cột. |
| `library_versions`, `download_time` | Phục vụ tái lập thí nghiệm và phần "môi trường" trong báo cáo. |

Cách bố trí: Bước 1 (tải + kiểm tra) sinh ra `step1_manifest.json`, Bước 2 đọc file đó làm đầu vào. Manifest là "hợp đồng" giữa hai bước.

## Cài đặt

**1. Quyết định cấu trúc `all_info`** (đây là chỗ pseudocode không nói rõ). Tôi đề xuất:

```python
all_info = {
    "imdb": {"repo_id": "stanfordnlp/imdb", "ds": ds_imdb},   # ds là DatasetDict
    ...
}
```

**2. Lấy `revision_sha`**: lấy từ `HfApi`, không lấy từ `ds`:

```python
from huggingface_hub import HfApi
sha = HfApi().dataset_info(repo_id).sha
```

**3. Hàm hoàn chỉnh**:

```python
import json, platform
from datetime import datetime, timezone
from pathlib import Path

import datasets
import huggingface_hub
from huggingface_hub import HfApi

REPORT_DIR = Path("reports")  # đổi theo project của bạn

def write_manifest(all_info):
    api = HfApi()
    manifest = {
        "download_time": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
        "library_versions": {
            "datasets": datasets.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "python": platform.python_version(),
        },
    }

    for name, info in all_info.items():
        ds = info["ds"]
        manifest["datasets"][name] = {
            "repo_id": info["repo_id"],
            "revision_sha": api.dataset_info(info["repo_id"]).sha,
            "splits": {split: len(ds[split]) for split in ds},
            "columns": {
                split: {col: str(feat) for col, feat in ds[split].features.items()}
                for split in ds
            },
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "step1_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out
```

## Lưu ý

- **`str(feat)`** cho chuỗi dễ đọc như `Value('string')` hay `ClassLabel(names=[...])`. Nếu Bước 2 cần parse lại thành schema, dùng `ds[split].features.to_dict()` thay thế, vì nó là dict JSON-serializable đầy đủ.
- **`ensure_ascii=False`** giữ nguyên tiếng Việt nếu có trong tên.
- **`mkdir(parents=True, exist_ok=True)`** tránh lỗi khi thư mục `REPORT_DIR` chưa tồn tại.
- **`revision_sha` cần mạng.** Nếu dataset load từ cache offline, lời gọi này sẽ lỗi. Bạn nên bọc `try/except` và ghi `None` kèm cảnh báo, theo đúng tinh thần "lệch là phát hiện chứ không phải crash" của `count_check`.
- **Nên gọi sau `count_check`**: bạn có thể nhúng luôn kết quả `rows` của `count_check` vào manifest nếu muốn một file duy nhất cho báo cáo.


---

## 🧑 USER #7

<pasted_content id="d211">
Bước 1 chi tiết: Tải dữ liệu, ghi dấu vân tay phiên bản, đối chiếu số liệu

Bước này gồm 4 việc nhỏ (1.1 đến 1.4). Nguyên tắc chung: code nằm trong src/, file chạy nằm trong scripts/, dữ liệu tải về nằm trong data/raw/, kết quả ghi nhận nằm trong reports/. Tách như vậy để các bước sau (EDA, tiền xử lý, huấn luyện) import lại được mà không phải viết lại.

1.0 Cây thư mục cần tạo (làm một lần)

Tạo trong D:\Project\2026-2027-HK1\nlp_v1\:

nlp_v1/
├── requirements.txt              <- (1.1) danh sách thư viện
├── src/
│   ├── __init__.py               <- file rỗng, để Python coi src là package
│   ├── config.py                 <- (1.2) hằng số: tên dataset, số liệu kỳ vọng, đường dẫn
│   └── data/
│       ├── __init__.py           <- file rỗng
│       └── download.py           <- (1.3) hàm tải + ghi manifest + đối chiếu
├── scripts/
│   └── 01_download_data.py       <- (1.4) file CHẠY, chỉ gọi hàm trong src
├── data/
│   └── raw/                      <- dữ liệu gốc, KHÔNG sửa tay
└── reports/                      <- nơi ghi manifest và bảng đối chiếu

Lệnh tạo nhanh (PowerShell, đứng ở thư mục nlp_v1):

mkdir src\data, scripts, data\raw, reports
ni src\__init__.py, src\data\__init__.py

---

1.1 File requirements.txt (thư mục gốc nlp_v1/)

┌───────────┬───────────────────────────────────────────┐
│    Mục    │                 Nội dung                  │
├───────────┼───────────────────────────────────────────┤
│ Giai đoạn │ Giai đoạn 1, Bước 1 (chuẩn bị môi trường) │
├───────────┼───────────────────────────────────────────┤
│ Kỹ thuật  │ Cố định phiên bản thư viện để tái lập     │
└───────────┴───────────────────────────────────────────┘

Nội dung cần có (ghi phiên bản cụ thể sau khi bạn cài và chạy pip freeze):

datasets        # tải dataset từ Hugging Face Hub
huggingface_hub # lấy commit hash (revision) của dataset
pandas
numpy
matplotlib
transformers    # dùng ở Bước 4 để đếm token
pyyaml          # (tuỳ chọn) nếu muốn đưa config ra YAML

Cài: pip install -r requirements.txt

Lưu ý: bản datasets mới (4.x) không còn chạy dataset dùng loading script. Nếu bước 1.4 báo lỗi kiểu "scripts are no longer supported" với deepset/germandpr, xem mục "Xử lý sự cố" cuối bài.

---

1.2 File src/config.py

┌───────────┬──────────────────────────────────────────────────────────────────┐
│    Mục    │                             Nội dung                             │
├───────────┼──────────────────────────────────────────────────────────────────┤
│ Giai đoạn │ Giai đoạn 1, Bước 1 (khai báo hằng số)                           │
├───────────┼──────────────────────────────────────────────────────────────────┤
│ Kỹ thuật  │ Tập trung cấu hình ở một nơi, không rải số "ma thuật" trong code │
└───────────┴──────────────────────────────────────────────────────────────────┘

Pseudo-code:

# --- đường dẫn: tính từ vị trí file config.py để chạy ở đâu cũng đúng ---
ROOT       = thư mục cha của thư mục cha của file này     # = nlp_v1/
RAW_DIR    = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

# --- danh sách dataset ---
DATASETS = {
  "germanquad": { repo_id: "deepset/germanquad", expected: {"train": 11518, "test": 2204} },
  "germandpr" : { repo_id: "deepset/germandpr",  expected: {"train": 9275,  "test": 1025} },
}

Giải trình:
- expected là số liệu từ đề bài, dùng làm mốc để đối chiếu ở 1.3. Đây chỉ là mốc, không phải sự thật. Số đếm được sau khi tải mới là sự thật.
- Dùng ROOT tính từ chính file này để tránh lỗi đường dẫn tương đối khi chạy từ thư mục khác.

---

1.3 File src/data/download.py

Đây là file chính. Gồm 4 hàm, mỗi hàm một việc:

Hàm A: download_dataset(name)

┌───────────┬─────────────────────────────────────────────────┐
│    Mục    │                    Nội dung                     │
├───────────┼─────────────────────────────────────────────────┤
│ Giai đoạn │ Giai đoạn 1, Bước 1                             │
├───────────┼─────────────────────────────────────────────────┤
│ Kỹ thuật  │ Tải dataset và lưu nguyên bản ra đĩa (snapshot) │
└───────────┴─────────────────────────────────────────────────┘

def download_dataset(name):
    cfg = DATASETS[name]
    ds  = load_dataset(cfg.repo_id)          # trả về DatasetDict {split: Dataset}
    ds.save_to_disk(RAW_DIR / name)          # lưu bản gốc
    return ds

Giải trình: save_to_disk giúp các bước sau đọc lại từ máy, không phụ thuộc mạng và không bị đổi phiên bản giữa chừng.

Hàm B: get_revision(name)

┌───────────┬─────────────────────────────────────────────────────┐
│    Mục    │                      Nội dung                       │
├───────────┼─────────────────────────────────────────────────────┤
│ Giai đoạn │ Giai đoạn 1, Bước 1                                 │
├───────────┼─────────────────────────────────────────────────────┤
│ Kỹ thuật  │ Ghi "dấu vân tay" phiên bản (commit SHA) để tái lập │
└───────────┴─────────────────────────────────────────────────────┘

def get_revision(name):
    info = HfApi().dataset_info(DATASETS[name].repo_id)
    return info.sha          # chuỗi commit hash

Giải trình: dataset trên Hub có thể bị cập nhật. Ghi SHA lại thì sau này bạn chứng minh được mình dùng đúng phiên bản nào.

Hàm C: count_check(name, ds)

┌───────────┬───────────────────────────────────────────────┐
│    Mục    │                   Nội dung                    │
├───────────┼───────────────────────────────────────────────┤
│ Giai đoạn │ Giai đoạn 1, Bước 1                           │
├───────────┼───────────────────────────────────────────────┤
│ Kỹ thuật  │ Đối chiếu số liệu thực tế với số liệu kỳ vọng │
└───────────┴───────────────────────────────────────────────┘

def count_check(name, ds):
    rows = []
    for split, expected_n in DATASETS[name].expected.items():
        actual_n = len(ds[split]) nếu split tồn tại, ngược lại None
        rows.append({
          dataset, split, expected: expected_n, actual: actual_n,
          diff: actual_n - expected_n,
          status: "OK" nếu diff == 0, "LỆCH" nếu khác, "THIẾU SPLIT" nếu None
        })
    # liệt kê thêm các split có thật nhưng không nằm trong expected (vd: validation)
    return rows

Giải trình: đừng dùng assert làm dừng chương trình khi lệch. Hãy ghi LỆCH vào bảng rồi chạy tiếp. Lệch là một phát hiện cần đưa vào báo cáo, không phải lỗi chương trình.

Hàm D: write_manifest(...)

┌───────────┬───────────────────────────────────────────┐
│    Mục    │                 Nội dung                  │
├───────────┼───────────────────────────────────────────┤
│ Giai đoạn │ Giai đoạn 1, Bước 1                       │
├───────────┼───────────────────────────────────────────┤
│ Kỹ thuật  │ Lưu manifest (siêu dữ liệu tải) dạng JSON │
└───────────┴───────────────────────────────────────────┘

def write_manifest(all_info):
    manifest = {
      "download_time": thời điểm hiện tại (ISO),
      "datasets": {
         name: { repo_id, revision_sha, splits: {split: số dòng},
                 columns: {split: tên cột + kiểu} }
      },
      "library_versions": { datasets, huggingface_hub, python }
    }
    ghi ra REPORT_DIR / "step1_manifest.json"

Ghi thêm cột columns (tên cột và kiểu, lấy từ ds[split].features) vì Bước 2 sẽ dùng lại.

---
</pasted_content id="d211">

, giải thích ý nghĩa của giai đoạn, kiểm tra lại các hàm mà ta đã sửa và kiểm tra các hàm chưa chỉnh sửa hãy giải thích và chỉnh sửa lại cho đúng 

ĐẶC BIỆT: khi giải thích, bạn phải là giảng viên đang giải thích cho sinh viên hiểu vấn đề code đang làm gì phục vụ giai đoạn, phải làm sao cho sinh viên hiểu được code và cấu trúc sẽ được xây dựng


---

## 🤖 CLAUDE #8

# Giai đoạn 1: "Nhập kho" dữ liệu trước khi dùng

Hãy hình dung bạn là thủ kho nhận một lô hàng. Bạn không đem hàng đi bán ngay mà làm bốn việc:

1. **Nhận hàng** (tải dataset về máy).
2. **Ghi số lô, ngày nhập** (phiên bản, thời điểm, thư viện đã dùng).
3. **Đếm lại so với phiếu giao hàng** (số mẫu thực tế so với số liệu đề bài).
4. **Lập phiếu nhập kho** (file manifest) để người dùng sau tra cứu.

Giai đoạn này chưa "làm NLP". Nó trả lời câu hỏi nền tảng: **dữ liệu của tôi là dữ liệu gì, đúng phiên bản nào, có đủ không?** Nếu bỏ qua, mọi kết quả huấn luyện phía sau đều không có gì để bảo chứng. Đây cũng là phần "Dữ liệu" mà báo cáo khoa học buộc phải có.

## Vì sao cấu trúc thư mục tách như vậy

| Thư mục | Vai trò | Ví dụ trong kho |
|---|---|---|
| `src/` | **Bộ công cụ**: hàm tái sử dụng | Dụng cụ đếm, cân |
| `scripts/` | **Nút bấm**: chỉ gọi công cụ theo thứ tự | Quy trình nhập kho |
| `data/raw/` | **Hàng gốc**, không sửa tay | Kệ hàng nguyên seal |
| `reports/` | **Giấy tờ ghi nhận** | Phiếu nhập kho |

Điểm cốt lõi là Bước 2 (EDA) có thể `import` lại hàm của Bước 1 mà không phải viết lại, vì logic nằm ở `src/`, còn `scripts/` chỉ là lớp mỏng điều phối.

## Kiểm tra từng thành phần

Có **5 lỗi cần sửa** trong phần bạn dán, cộng với 2 chỗ trong bản `write_manifest` tôi đưa trước đó.

### `config.py`: dict nhưng bị truy cập như object

Pseudocode viết `cfg.repo_id` và `DATASETS[name].expected`, nhưng `DATASETS` được khai báo bằng dict `{...}`. Dict không có thuộc tính `.repo_id`, nên chạy sẽ báo `AttributeError`.

**Sửa:** dùng `dataclass`, vừa truy cập được bằng `.`, vừa có cấu trúc rõ ràng.

### Hàm A `download_dataset`: tải bản mới nhất, không ghim phiên bản

`load_dataset(repo_id)` luôn lấy **commit mới nhất**. Hàm B hỏi SHA ở một lời gọi riêng. Nếu chủ dataset cập nhật giữa hai lời gọi, bạn ghi SHA của phiên bản này nhưng dữ liệu lại là phiên bản kia. Manifest khi đó **nói dối**.

**Sửa:** lấy SHA trước, rồi truyền `revision=sha` vào `load_dataset`. Như vậy SHA ghi lại chắc chắn là SHA đã tải.

### Hàm B `get_revision`: đúng, chỉ cần đổi vị trí gọi

`HfApi().dataset_info(repo_id).sha` là cách lấy đúng. Lỗi nằm ở **thứ tự gọi**, đã nêu ở trên.

### Hàm C `count_check`: đã sửa, giữ nguyên

Bản đã sửa xử lý `None` trước khi trừ, và có thêm các split ngoài kỳ vọng. Ý quan trọng với sinh viên là **lệch không phải lỗi chương trình, lệch là một phát hiện**, nên hàm ghi lại rồi chạy tiếp.

### Hàm D `write_manifest`: sửa lại so với bản trước

Bản tôi đưa lúc trước gọi `api.dataset_info()` lần thứ hai bên trong hàm, lặp lại đúng lỗi của hàm A. Nó còn lấy `repo_id` từ `all_info`, trong khi nguồn sự thật đã là `config.py`.

**Sửa:** nhận SHA đã có sẵn từ bên ngoài, và lấy `repo_id` từ `DATASETS`.

### Lỗi ẩn ở file chạy: `import src` thất bại

Khi chạy `python scripts/01_download_data.py`, Python chỉ thêm `scripts/` vào đường dẫn tìm module, không thêm thư mục gốc. Dòng `from src.config import ...` sẽ báo `ModuleNotFoundError`.

**Sửa:** chèn thư mục gốc vào `sys.path` ở đầu file chạy.

## Code hoàn chỉnh

**`src/config.py`**

```python
from dataclasses import dataclass
from pathlib import Path

# Tính từ vị trí file này (src/config.py) -> lên 2 cấp = nlp_v1/
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    expected: dict  # số mẫu kỳ vọng theo split (mốc từ đề bài)


DATASETS = {
    "germanquad": DatasetSpec("deepset/germanquad", {"train": 11518, "test": 2204}),
    "germandpr": DatasetSpec("deepset/germandpr", {"train": 9275, "test": 1025}),
}
```

**`src/data/download.py`**

```python
import json
import platform
from datetime import datetime, timezone

import datasets
import huggingface_hub
from datasets import load_dataset
from huggingface_hub import HfApi

from src.config import DATASETS, RAW_DIR, REPORT_DIR


def get_revision(name):
    """Lấy commit SHA mới nhất của dataset trên Hub."""
    return HfApi().dataset_info(DATASETS[name].repo_id).sha


def download_dataset(name, revision):
    """Tải đúng phiên bản `revision` và lưu bản gốc ra đĩa."""
    spec = DATASETS[name]
    ds = load_dataset(spec.repo_id, revision=revision)  # ghim phiên bản
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(RAW_DIR / name))
    return ds


def count_check(name, ds):
    """Đối chiếu số mẫu thực tế với kỳ vọng. Lệch -> ghi lại, không dừng."""
    rows = []
    expected = DATASETS[name].expected

    for split, expected_n in expected.items():
        actual_n = len(ds[split]) if split in ds else None
        if actual_n is None:
            diff, status = None, "THIẾU SPLIT"
        else:
            diff = actual_n - expected_n
            status = "OK" if diff == 0 else "LỆCH"
        rows.append({"dataset": name, "split": split, "expected": expected_n,
                     "actual": actual_n, "diff": diff, "status": status})

    for split in ds:  # split có thật nhưng không nằm trong kỳ vọng
        if split not in expected:
            rows.append({"dataset": name, "split": split, "expected": None,
                         "actual": len(ds[split]), "diff": None,
                         "status": "NGOÀI KỲ VỌNG"})
    return rows


def write_manifest(all_info):
    """all_info = {name: {"revision_sha": str, "ds": DatasetDict}}"""
    manifest = {
        "download_time": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            name: {
                "repo_id": DATASETS[name].repo_id,
                "revision_sha": info["revision_sha"],
                "splits": {s: len(info["ds"][s]) for s in info["ds"]},
                "columns": {
                    s: {c: str(f) for c, f in info["ds"][s].features.items()}
                    for s in info["ds"]
                },
            }
            for name, info in all_info.items()
        },
        "library_versions": {
            "datasets": datasets.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "python": platform.python_version(),
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "step1_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out
```

**`scripts/01_download_data.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # để import được src

import pandas as pd

from src.config import DATASETS, REPORT_DIR
from src.data.download import count_check, download_dataset, get_revision, write_manifest


def main():
    all_info, all_rows = {}, []
    for name in DATASETS:
        sha = get_revision(name)          # 1. ghi số lô TRƯỚC
        ds = download_dataset(name, sha)  # 2. tải đúng lô đó
        all_info[name] = {"revision_sha": sha, "ds": ds}
        all_rows += count_check(name, ds)  # 3. đếm lại

    df = pd.DataFrame(all_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "step1_count_check.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    print("Manifest:", write_manifest(all_info))  # 4. lập phiếu nhập kho


if __name__ == "__main__":
    main()
```

## Luồng chạy (để sinh viên nhớ)

```
get_revision ──► download_dataset ──► count_check ──► write_manifest
 (ghi số lô)      (nhập đúng lô)      (đếm so phiếu)   (lập phiếu kho)
                       │                   │                │
                  data/raw/<name>    step1_count_check.csv  step1_manifest.json
```

Chạy từ thư mục `nlp_v1`:

```powershell
pip install -r requirements.txt
python scripts/01_download_data.py
```

## Xử lý sự cố thường gặp

- **Lỗi "scripts are no longer supported" với `germandpr`:** `datasets` 4.x bỏ hỗ trợ dataset dùng loading script. Cách nhanh nhất là `pip install "datasets<4"`, rồi ghi đúng phiên bản đó vào `requirements.txt`.
- **`ModuleNotFoundError: No module named 'src'`:** kiểm tra `src/__init__.py` và `src/data/__init__.py` đã tồn tại (file rỗng), và dòng `sys.path.insert` ở đầu script.
- **Kết quả `LỆCH`:** đừng sửa code cho khớp. Hãy ghi vào báo cáo và tìm nguyên nhân (đề bài ghi số của phiên bản khác, hay dataset đã được cập nhật).

Bạn muốn tôi tạo luôn các file này trong `D:\Project\2026-2027-HK1\nlp_v1` không?


---

## 🧑 USER #9

(venv) PS D:\Project\2026-2027-HK1\nlp_v1> python scripts/01_download_data.py
(venv) PS D:\Project\2026-2027-HK1\nlp_v1> 
kiểm tra xem tại sao nó không hoạt động


---

## 🤖 CLAUDE #10

**Nguyên nhân: cả 4 file code đều rỗng (0 byte).** Python chạy một file rỗng thì không có lệnh nào để thực thi, nên nó thoát im lặng, không in gì và cũng không báo lỗi.

| File | Kích thước |
|---|---|
| `scripts/01_download_data.py` | 0 |
| `src/config.py` | 0 |
| `src/data/download.py` | 0 |
| `requirements.txt` | 0 |

Có lẽ bạn đã tạo file nhưng chưa dán nội dung, hoặc dán rồi quên lưu (Ctrl+S).

Ngoài ra còn hai chỗ thiếu:

- **Thiếu `src/__init__.py` và `src/data/__init__.py`.** Python 3 vẫn chạy được nhờ namespace package, nhưng nên tạo để đúng cấu trúc đã thiết kế.
- **`requirements.txt` rỗng**, nên các thư viện (`datasets`, `huggingface_hub`, `pandas`) có thể chưa được cài.

Bạn muốn tôi ghi nội dung đã sửa vào 4 file này, tạo hai file `__init__.py` rồi chạy thử luôn không?


---

## 🧑 USER #11

Hãy hướng dẫn chính xác lỗi ở đâu và sửa như thế nào


---

## 🤖 CLAUDE #12

## Lỗi ở đâu

Chương trình **không hề lỗi**. Nó không in gì vì **file rỗng, không có dòng code nào để chạy**.

Bạn có thể tự kiểm chứng. Chạy trong PowerShell, đứng ở `nlp_v1`:

```powershell
Get-ChildItem src\config.py, src\data\download.py, scripts\01_download_data.py, requirements.txt | Select-Object Name, Length
```

Cột `Length` đang là `0` ở cả 4 file. Dù bạn đã mở file trong editor và nhìn thấy code, nếu **chưa lưu (Ctrl+S)** thì trên đĩa vẫn là 0 byte.

## Cách sửa (làm lần lượt)

**Bước 1: tạo 2 file `__init__.py` còn thiếu**

```powershell
ni src\__init__.py, src\data\__init__.py
```

Hai file này để rỗng, đó là đúng.

**Bước 2: ghi nội dung vào 4 file rồi lưu**

| File | Dán nội dung từ |
|---|---|
| `src/config.py` | khối `src/config.py` trong tin nhắn trước |
| `src/data/download.py` | khối `src/data/download.py` trong tin nhắn trước |
| `scripts/01_download_data.py` | khối `scripts/01_download_data.py` trong tin nhắn trước |
| `requirements.txt` | nội dung bên dưới |

`requirements.txt`:

```
datasets<4
huggingface_hub
pandas
numpy
matplotlib
transformers
```

Dòng `datasets<4` là bắt buộc, vì `deepset/germandpr` dùng loading script mà `datasets` 4.x không còn hỗ trợ.

**Bước 3: kiểm tra đã lưu thật**

Chạy lại lệnh ở trên, cột `Length` của cả 4 file phải **lớn hơn 0**.

**Bước 4: cài thư viện** (venv đang bật sẵn)

```powershell
pip install -r requirements.txt
```

**Bước 5: kiểm tra import trước khi chạy thật**

```powershell
python -c "from src.config import DATASETS; print(DATASETS)"
```

Nếu in ra hai `DatasetSpec` thì cấu trúc package đã đúng. Nếu báo `ModuleNotFoundError`, quay lại Bước 1.

**Bước 6: chạy**

```powershell
python scripts/01_download_data.py
```

## Kết quả đúng sẽ trông như thế nào

Lần đầu mất vài phút vì tải dataset. Sau đó terminal in bảng đối chiếu, ví dụ:

```
   dataset split  expected  actual  diff status
germanquad train     11518   11518     0     OK
germanquad  test      2204    2204     0     OK
 germandpr train      9275    9275     0     OK
 germandpr  test      1025    1025     0     OK
Manifest: ...\reports\step1_manifest.json
```

Đồng thời có thêm: `data/raw/germanquad/`, `data/raw/germandpr/`, `reports/step1_count_check.csv`, `reports/step1_manifest.json`.

## Nếu vẫn lỗi

Sau Bước 6 mà có traceback, dán nguyên văn lỗi cho tôi. Cách phân biệt nhanh:

- `ModuleNotFoundError: src` → thiếu `__init__.py` hoặc thiếu dòng `sys.path.insert` ở đầu script.
- `ModuleNotFoundError: datasets` → chưa chạy Bước 4 trong đúng venv.
- Lỗi mạng/`ConnectionError` → không kết nối được HuggingFace Hub.
- Vẫn không in gì → file chưa lưu, quay lại Bước 3.


---

## 🧑 USER #13

Tao vẫn chưa hiểu là thiếu cái gì


---

## 🤖 CLAUDE #14

Có hai chuyện khác nhau, mày đang nhầm lẫn giữa chúng.

## 1. Lần chạy trước im lặng vì file rỗng

Lúc mày chạy, `01_download_data.py` là 0 byte. Giờ file đã có nội dung rồi (374, 3281, 1125 byte), nên vấn đề đó đã hết.

## 2. Giờ code đã có nhưng còn lỗi, chạy sẽ báo traceback

Tao vừa đọc 3 file của mày. Dưới đây là những chỗ sai, theo thứ tự Python sẽ vấp phải.

### `src/config.py`

| Dòng | Sai | Sửa |
|---|---|---|
| đầu file | **Thiếu import** `Path` và `dataclass` → `NameError` ngay dòng 1 | Thêm `from dataclasses import dataclass` và `from pathlib import Path` |
| 9, 10 | `{ repo_id:"...", expected: {...} }` là **dict viết sai**, `repo_id` và `expected` là tên biến chưa định nghĩa | Dùng class đã khai báo: `DatasetSpec("deepset/germanquad", {...})` |
| 9, 10 | `{"train: 11518"}` là **một chuỗi** nằm trong set, không phải dict | `{"train": 11518, "test": 2204}` (dấu `"` đóng sau `train`) |
| 10 | `"train: 10000"` là số bịa. Đề bài ghi germandpr: train 9275, test 1025 | `{"train": 9275, "test": 1025}` |

Bản đúng:

```python
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    expected: dict

DATASETS = {
    "germanquad": DatasetSpec("deepset/germanquad", {"train": 11518, "test": 2204}),
    "germandpr":  DatasetSpec("deepset/germandpr",  {"train": 9275,  "test": 1025}),
}
```

### `src/data/download.py`

| Dòng | Sai | Sửa |
|---|---|---|
| 12 | `def download_dataset(name)` **thiếu tham số `revision`**, nhưng dòng 15 dùng `revision` → `NameError`. Script lại gọi `download_dataset(name, sha)` → `TypeError` | `def download_dataset(name, revision):` |
| 21 | `DATASETS[name]["repo_id"]`: `DATASETS[name]` là dataclass, không index bằng `[]` được → `TypeError` | `DATASETS[name].repo_id` |
| 29 | `DATASETS[name]["expected"]` sai tương tự | `DATASETS[name].expected` |
| 35 | `diff` gán một chuỗi `"Missing split..."`, trong khi nơi khác là số. Cột bị lẫn kiểu | `diff, status = None, "THIẾU SPLIT"` |
| 78-80 | `datasets`, `huggingface_hub`, `python` **bị lặp 2 lần** trong dict | Xoá 3 dòng thừa |

Chỗ dễ nhầm nhất là dòng 21, 29 so với dòng 15, 64: file đang **trộn hai kiểu truy cập** (`["x"]` và `.x`). Đã dùng dataclass thì chỉ dùng `.x`.

### `scripts/01_download_data.py`

| Dòng | Sai | Sửa |
|---|---|---|
| 3 | `from turtle import pd`: `turtle` là module vẽ hình, không có `pd` → `ImportError` | `import pandas as pd` |
| 7 | `src.congig` **gõ sai chính tả** → `ModuleNotFoundError` | `src.config` |
| 17 và 24 | `write_manifest(all_info)` **gọi 2 lần**, ghi file hai lần | Giữ dòng 24, xoá dòng 17 |

### Chưa có

- `src/__init__.py` và `src/data/__init__.py`: cả hai chưa tồn tại. Python 3 vẫn chạy được nhưng nên tạo bằng `ni src\__init__.py, src\data\__init__.py`.
- `requirements.txt` vẫn **0 byte**. Dán vào: `datasets<4`, `huggingface_hub`, `pandas`.

## Tóm lại, "thiếu cái gì"

1. Thiếu **import** trong `config.py`.
2. Thiếu tham số **`revision`** ở hàm `download_dataset`.
3. Thiếu `__init__.py` và nội dung `requirements.txt`.
4. Có 3 chỗ **gõ sai** (`congig`, `turtle`, `"train: 11518"`) và 2 chỗ truy cập sai kiểu (`["repo_id"]` thay vì `.repo_id`).

Mày muốn tao ghi thẳng các sửa này vào 3 file rồi chạy thử, hay mày tự sửa theo bảng trên?


---

## 🧑 USER #15

<pasted_content id="d211">
Giai đoạn 2: Tiền xử lý, chia tập và corpus

Giai đoạn này sinh ra bộ dữ liệu chuẩn mà mọi giai đoạn sau (retriever, reader, đánh giá) sẽ dùng. Sai ở đây thì mọi kết quả phía sau đều không đáng tin, vì vậy thứ tự làm việc quan trọng hơn tốc độ.

▎ Trước khi bắt đầu, bạn cần có danh sách vấn đề và quyết định xử lý của Giai đoạn 1 (reports/eda.md). Giai đoạn 2 là nơi thực thi các quyết định đó.

---

Bước 2.0: Chốt "quy tắc ghi trước" (viết ra giấy, chưa viết code)

Đề bài nhấn mạnh "quy tắc ghi trước" vì nếu bạn quyết định sau khi thấy kết quả thì dễ vô tình thiên vị. Hãy tạo file docs/preprocessing_rules.md và điền các quyết định sau trước khi chạy bất kỳ phép lọc nào:

┌─────┬─────────────────────────────────────────────────────────────┬────────────────────────────────────────────────┐
│  #  │                      Câu hỏi cần chốt                       │                 Gợi ý lựa chọn                 │
├─────┼─────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ R1  │ Offset sai nhưng đáp án xuất hiện đúng 1 lần trong context? │ Sửa bằng tìm chuỗi, ghi mapping                │
├─────┼─────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ R2  │ Đáp án xuất hiện nhiều lần, hoặc không tìm thấy?            │ Cách ly (quarantine), không đoán               │
├─────┼─────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ R3  │ Mẫu train có câu hỏi/context trùng với test?                │ Loại khỏi train (không bao giờ loại khỏi test) │
├─────┼─────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ R4  │ Hai passage giống hệt về văn bản?                           │ Gộp thành 1, gộp provenance                    │
├─────┼─────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ R5  │ Tỷ lệ train/dev?                                            │ 90/10, theo nhóm (xem 2.6)                     │
├─────┼─────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ R6  │ Có đưa passage DPR vào corpus không?                        │ Có, nếu Giai đoạn 1 xác nhận hợp lệ            │
├─────┼─────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│ R7  │ Seed chia tập?                                              │ Một số cố định, ví dụ 42                       │
└─────┴─────────────────────────────────────────────────────────────┴────────────────────────────────────────────────┘

---

Bước 2.1: Tạo cấu trúc file

Mở rộng cấu trúc từ Giai đoạn 1:

nlp_v1/
├── docs/preprocessing_rules.md        <- (2.0) quy tắc ghi trước
├── src/
│   ├── config.py                      <- thêm: PROCESSED_DIR, SEED, SPLIT_RATIO
│   └── data/
│       ├── io_utils.py                <- đọc/ghi jsonl, json, hash
│       ├── text_utils.py              <- chuẩn hóa văn bản, tạo retrieval_text
│       ├── validate_spans.py          <- kiểm tra và sửa offset
│       ├── build_corpus.py            <- dựng corpus + passage_id
│       ├── build_qa.py                <- dựng QA + qrels
│       ├── split.py                   <- chia train/dev theo nhóm
│       └── quality_report.py          <- tổng hợp báo cáo chất lượng
├── scripts/
│   └── 02_preprocess.py               <- file chạy, gọi tuần tự các module
├── data/
│   ├── raw/                           <- BẤT BIẾN (từ Giai đoạn 1)
│   └── processed/v1/                  <- đầu ra của giai đoạn này
└── tests/
    └── test_phase2_acceptance.py      <- (2.9) kiểm tra nghiệm thu

Thư mục v1/ giúp bạn tạo v2/ sau này mà không ghi đè (ứng với "corpus version" trong manifest).

---

Bước 2.2: Schema mục tiêu (đây là "hợp đồng" của dữ liệu)

Viết các schema này vào docs/schema.md để cả nhóm dùng chung:

Corpus (corpus.jsonl, một dòng một passage):

┌────────────────┬──────────────────────────────────────────────────────────────┐
│     Trường     │                           Ý nghĩa                            │
├────────────────┼──────────────────────────────────────────────────────────────┤
│ passage_id     │ ID nội bộ ổn định (xem 2.4)                                  │
├────────────────┼──────────────────────────────────────────────────────────────┤
│ original_text  │ Văn bản gốc, không sửa, dành cho reader                      │
├────────────────┼──────────────────────────────────────────────────────────────┤
│ title          │ Tiêu đề bài viết                                             │
├────────────────┼──────────────────────────────────────────────────────────────┤
│ document_id    │ ID bài viết/nhóm (nếu thiếu, dùng title hoặc hash của title) │
├────────────────┼──────────────────────────────────────────────────────────────┤
│ source         │ Danh sách nguồn: [{dataset, split, source_id, role}]         │
├────────────────┼──────────────────────────────────────────────────────────────┤
│ retrieval_text │ Bản phục vụ tìm kiếm (hạ chữ thường...), tách riêng          │
└────────────────┴──────────────────────────────────────────────────────────────┘

QA (qa_*.jsonl):

┌─────────────┬────────────────────────────────────────────────┐
│   Trường    │                    Ý nghĩa                     │
├─────────────┼────────────────────────────────────────────────┤
│ question_id │ ID ổn định của câu hỏi                         │
├─────────────┼────────────────────────────────────────────────┤
│ question    │ Văn bản câu hỏi                                │
├─────────────┼────────────────────────────────────────────────┤
│ passage_id  │ Passage chứa đáp án (khóa sang corpus)         │
├─────────────┼────────────────────────────────────────────────┤
│ answers     │ [{text, start}], start tính trên original_text │
└─────────────┴────────────────────────────────────────────────┘

Qrels (qrels_*.json): {question_id: [passage_id, ...]}.

---

Bước 2.3: Xử lý văn bản (src/data/text_utils.py)

┌───────────┬──────────────────────────────────────────────────┐
│    Mục    │                     Nội dung                     │
├───────────┼──────────────────────────────────────────────────┤
│ Giai đoạn │ 2, Xử lý văn bản (mục 1 đến 3, 7 trong đề)       │
├───────────┼──────────────────────────────────────────────────┤
│ Kỹ thuật  │ Tách văn bản cho reader và văn bản cho retrieval │
└───────────┴──────────────────────────────────────────────────┘

Nguyên tắc cốt lõi: original_text không bao giờ bị sửa. Mọi chuẩn hóa chỉ xảy ra trên retrieval_text.

def make_retrieval_text(original_text):
    t = unicodedata.normalize("NFC", original_text)
    t = gộp mọi khoảng trắng liên tiếp thành 1
    t = t.strip()
    t = t.lower()           # chỉ ở đây mới hạ chữ thường
    return t

def make_dedup_key(original_text):
    return sha1(original_text.encode("utf-8"))     # khóa trùng CHÍNH XÁC trên raw

Giải trình:
- Nếu sửa original_text (ví dụ NFC làm đổi độ dài ký tự), mọi answer_start sẽ lệch. Đây là lý do đề bài yêu cầu "mapping" khi sửa văn bản ảnh hưởng vị trí. Cách an toàn nhất là không sửa.
- Bản chuẩn hóa dùng để dò gần trùng (mục 7 của đề) chỉ để báo cáo, không dùng để gộp passage, trừ khi có mapping offset.

---

Bước 2.4: Kiểm tra và xử lý offset (src/data/validate_spans.py)

┌───────────┬───────────────────────────────────────────────────────────────┐
│    Mục    │                           Nội dung                            │
├───────────┼───────────────────────────────────────────────────────────────┤
│ Giai đoạn │ 2, Xử lý văn bản (mục 3 và 6)                                 │
├───────────┼───────────────────────────────────────────────────────────────┤
│ Kỹ thuật  │ So khớp chuỗi theo offset, cách ly mẫu không sửa đáng tin cậy │
└───────────┴───────────────────────────────────────────────────────────────┘

def validate(context, answer_text, answer_start):
    if context[start : start+len(text)] == text:   return "OK", start
    positions = tất cả vị trí text xuất hiện trong context
    if len(positions) == 1:  return "FIXED", positions[0]     # sửa đáng tin cậy
    if len(positions) > 1:   return "QUARANTINE_AMBIGUOUS", None
    else:                    return "QUARANTINE_NOT_FOUND", None

Với mỗi mẫu, ghi lại một dòng log: question_id, trạng thái, start_cũ, start_mới. Đó chính là mapping mà đề yêu cầu.

┌──────────────┬──────────────────────────────────────────────────────────┐
│  Trạng thái  │                        Hành động                         │
├──────────────┼──────────────────────────────────────────────────────────┤
│ OK           │ Giữ nguyên                                               │
├──────────────┼──────────────────────────────────────────────────────────┤
│ FIXED        │ Dùng start mới, ghi vào log                              │
├──────────────┼──────────────────────────────────────────────────────────┤
│ QUARANTINE_* │ Đưa vào data/processed/v1/quarantine.jsonl, không vào QA │
└──────────────┴──────────────────────────────────────────────────────────┘

Lưu ý với GermanQuAD test: test có nhiều đáp án cho mỗi câu hỏi. Chỉ cách ly từng đáp án lỗi; chỉ cách ly cả câu hỏi khi không còn đáp án hợp lệ nào. Với test, mọi quyết định phải theo quy tắc R1/R2 đã ghi trước, và bạn cần báo cáo số mẫu test bị ảnh hưởng (xem 2.7).

Lưu ý với GermanDPR: DPR thường chỉ có văn bản đáp án, không có offset. Với mẫu từ DPR, hoặc (a) dùng cho retrieval/qrels (không cần offset), hoặc (b) tìm đáp án trong positive passage bằng chính quy tắc trên. Hãy ghi rõ lựa chọn trong preprocessing_rules.md.

---

Bước 2.5: Dựng corpus (src/data/build_corpus.py)

┌───────────┬────────────────────────────────────────────────────────────────┐
│    Mục    │                            Nội dung                            │
├───────────┼────────────────────────────────────────────────────────────────┤
│ Giai đoạn │ 2, Corpus + Xử lý văn bản (mục 4 và 5)                         │
├───────────┼────────────────────────────────────────────────────────────────┤
│ Kỹ thuật  │ Hợp nhất, loại trùng bằng hash, gán ID ổn định, giữ provenance │
└───────────┴────────────────────────────────────────────────────────────────┘

corpus = {}      # key = dedup_key
for mỗi context trong GermanQuAD (train+test):
    thêm_vào_corpus(text, title, source={dataset:"germanquad", split, role:"context"})
for mỗi passage trong GermanDPR (positive + hard_negative, train+test):
    thêm_vào_corpus(text, title, source={dataset:"germandpr", split, role:"positive"|"hard_negative", source_id})

def thêm_vào_corpus(text, title, source):
    key = make_dedup_key(text)
    nếu key chưa có: tạo bản ghi mới
    bản ghi.source.append(source)      # KHÔNG bỏ provenance khi trùng

Gán passage_id ổn định:

passage_id = "p_" + key[:16]       # sinh từ hash văn bản gốc

Giải trình:
- Hash theo nội dung nghĩa là chạy lại bao nhiêu lần cũng ra cùng ID, và không phụ thuộc thứ tự duyệt.
- ID nguồn (nếu có) không dùng làm khóa chính, vì đề nói nó có thể rỗng hoặc trùng giữa các dataset. Nó được giữ trong source[].source_id.
- Cần kiểm tra va chạm: hai văn bản khác nhau cho cùng 16 ký tự đầu của hash gần như không xảy ra, nhưng hãy assert số ID duy nhất bằng số bản ghi duy nhất.

Điểm cần chú ý: hai bản có cùng văn bản nhưng khác title thì gộp, và lưu cả hai title vào source. Hai bản gần giống nhưng khác ký tự (mục 7 của đề) không gộp, chỉ ghi vào báo cáo.

---

Bước 2.6: Dựng QA và qrels (src/data/build_qa.py)

┌───────────┬───────────────────────────────────────────────────────────┐
│    Mục    │                         Nội dung                          │
├───────────┼───────────────────────────────────────────────────────────┤
│ Giai đoạn │ 2, Schema QA + Qrels                                      │
├───────────┼───────────────────────────────────────────────────────────┤
│ Kỹ thuật  │ Ánh xạ context sang passage_id, tạo qrels từ nhãn đã biết │
└───────────┴───────────────────────────────────────────────────────────┘

for mỗi mẫu QuAD hợp lệ (sau 2.4):
    pid = "p_" + hash(context)[:16]
    qa.append({question_id, question, passage_id: pid, answers})
    qrels[question_id] = [pid]

for mỗi mẫu DPR:
    pid = id của positive passage
    qrels[question_id] ∪= {pid}

Giải trình về qrels: "relevant_passage_ids đã biết" nghĩa là chỉ những gì dataset xác nhận. Nếu một câu hỏi xuất hiện ở cả hai dataset, hợp nhất các passage liên quan vào một mục qrels, và dùng cùng một question_id (suy ra từ hash của câu hỏi đã chuẩn hóa). Cảnh báo: passage không có trong qrels chưa chắc là không liên quan, hãy ghi giới hạn này vào báo cáo.

Hard negatives: theo đề, negative gắn với query, và passage negative của câu A có thể là đáp án đúng của câu B. Vì vậy:
- Không gắn nhãn "negative" lên passage trong corpus.
- Nếu cần cho huấn luyện retriever, lưu riêng hardneg_train.json dạng {question_id: [passage_id,...]} và chỉ cho tập train.

---

Bước 2.7: Chia tập (src/data/split.py)

┌───────────┬───────────────────────────────────────────────────────────┐
│    Mục    │                         Nội dung                          │
├───────────┼───────────────────────────────────────────────────────────┤
│ Giai đoạn │ 2, Chia tập                                               │
├───────────┼───────────────────────────────────────────────────────────┤
│ Kỹ thuật  │ Chia theo nhóm (group split) để chống rò rỉ, cố định seed │
└───────────┴───────────────────────────────────────────────────────────┘

Nguyên tắc: test gốc giữ nguyên. Chỉ tách train thành train/dev.

# 1. Xác định "nhóm" cho mỗi câu hỏi
group_of(q) = document_id / title nếu có
              nếu không có -> hash(context)        # công bố giới hạn này

# 2. Gộp nhóm liên thông (nếu 2 câu hỏi chung context hoặc chung câu hỏi giữa 2 dataset)
dùng union-find trên: question <-> passage_id

# 3. Chia theo nhóm, không chia theo câu hỏi
groups = danh sách nhóm, sắp xếp ổn định rồi xáo bằng random(seed=SEED)
dev_groups = lấy dần nhóm cho đến khi ~10% số câu hỏi
train_groups = còn lại

# 4. Lưu split IDs ra file, không tái chia ở lần sau

Xử lý xung đột theo quy tắc R3:

for q trong train/dev:
    nếu q.question_hash hoặc q.passage_id nằm trong tập test:
        loại q khỏi train/dev, ghi vào "excluded_conflict" kèm lý do
# Tuyệt đối không có bước nào loại mẫu khỏi test vì "khó"

Giải trình: chia theo câu hỏi thì các câu hỏi cùng context sẽ rơi vào cả train và dev, khiến dev quá dễ và điểm cao giả. Số 90/10 chỉ là ứng viên; hãy xem phân phối kích thước nhóm từ Giai đoạn 1 (Bước 5) rồi mới chốt, đặc biệt nếu có một vài nhóm rất lớn làm tỷ lệ thực tế lệch.

---

Bước 2.8: Xuất file, manifest, báo cáo chất lượng

2.8a. Các file đầu ra (vào data/processed/v1/)

corpus.jsonl
qa_train.jsonl   qa_dev.jsonl   qa_test.jsonl
qrels_dev.json   qrels_test.json        (thêm qrels_train.json nếu cần huấn luyện)
split_manifest.json
data_quality_report.json
quarantine.jsonl                         (mẫu bị cách ly, kèm lý do)
span_fix_log.jsonl                       (mapping offset đã sửa)

2.8b. split_manifest.json

{
  "corpus_version": "v1",
  "raw_revisions": { từ step1_manifest.json },
  "seed": 42, "ratio": "90/10",
  "preprocessing_config": { các quy tắc R1-R7, phiên bản code },
  "split_ids": { "train": [question_id...], "dev": [...], "test": [...] }
}

2.8c. data_quality_report.json

Số liệu tối thiểu: số mẫu đầu vào/đầu ra mỗi split, số mẫu OK/FIXED/QUARANTINE, số passage trước/sau dedup, số passage có nhiều provenance, số mẫu bị loại do xung đột, overlap giữa các split sau khi chia (phải bằng 0 trong những chỗ bắt buộc).

---

Bước 2.9: Nghiệm thu (tests/test_phase2_acceptance.py)

Viết các phép kiểm tra tự động, mỗi phép tương ứng một dòng trong "Nghiệm thu" của đề:

┌─────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────┐
│        Phép kiểm tra        │                                         Điều kiện đạt                                         │
├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Spans hợp lệ                │ Với mọi QA: passage.original_text[start:start+len(text)] == text                              │
├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Qrels đầy đủ                │ Mọi passage_id trong qrels và QA đều có trong corpus.jsonl                                    │
├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Không overlap trái protocol │ question_id train ∩ dev = ∅, train ∩ test = ∅, dev ∩ test = ∅; không nhóm nào nằm ở hai split │
├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ ID duy nhất                 │ passage_id không trùng; question_id không trùng trong một split                               │
├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Tái tạo được                │ Chạy lại script từ data/raw/ cho ra file có cùng hash (so sánh checksum)                      │
├─────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
│ Mẫu bị cách ly có lý do     │ Mọi dòng trong quarantine.jsonl có trường reason                                              │
└─────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────┘

Lệnh chạy: python scripts/02_preprocess.py rồi pytest tests/test_phase2_acceptance.py.

---

Những bẫy dễ mắc

1. Sửa original_text rồi quên sửa offset. Giữ raw nguyên vẹn là cách tránh triệt để.
2. Chia theo câu hỏi thay vì theo nhóm. Đây là nguồn rò rỉ phổ biến nhất.
3. Dùng passage_id của DPR làm khóa chung. ID nguồn có thể rỗng hoặc trùng giữa dataset; hãy dùng hash nội dung.
4. Loại mẫu khó khỏi test. Chỉ được xử lý theo quy tắc ghi trước, và phải báo cáo số lượng.
5. Dùng qrels hoặc nhãn test khi tuning. Dữ liệu test nằm trong corpus được, nhưng câu hỏi và nhãn test không được dùng để học hay chỉnh tham số.
6. Chia lại mỗi lần chạy. Lưu split IDs vào manifest rồi đọc lại.

Khi bạn làm xong từng phần, hãy gửi lên nội dung data_quality_report.json và 2 hoặc 3 dòng của quarantine.jsonl; tôi sẽ giúp bạn kiểm tra xem các quyết định có nhất quán với quy tắc đã ghi trước không.
</pasted_content id="d211">

, hướng dẫn cho tao cài đặt


---

## 🤖 CLAUDE #16

Đang dựng xong các module Giai đoạn 2 trong thư mục tạm, giờ chạy thử trên dữ liệu thật.


---

## 🤖 CLAUDE #17

# Cài đặt Giai đoạn 2, theo thứ tự làm

Tao đã dựng toàn bộ Giai đoạn 2 trong thư mục tạm và chạy trên `data/raw` thật của mày. **Chưa đụng vào project của mày.** Code dưới đây đã chạy được, và chạy hai lần cho ra 13 file giống hệt nhau (hash trùng).

Có hai điều cần biết trước khi làm.

1. **`reports/eda.md` chưa tồn tại.** Đề yêu cầu Giai đoạn 2 thực thi các quyết định từ EDA. Tao đã chạy các phép kiểm tra tương đương (bên dưới) nên mày có số liệu để viết quy tắc, nhưng mày vẫn nên ghi chúng vào `eda.md` cho đủ hồ sơ.
2. **Dữ liệu có một phát hiện quan trọng** cần quyết định trước khi chốt quy tắc (mục "Phát hiện").

## Bức tranh tổng thể (giảng viên nói)

Hãy nghĩ Giai đoạn 2 như một **dây chuyền có 4 trạm**, mỗi trạm là một file:

```
data/raw (bất biến)
   │
   ├─► build_corpus.py ─► corpus.jsonl      "kho passage, mỗi passage một ID cố định"
   ├─► build_qa.py     ─► qa + qrels        "câu hỏi ↔ passage đúng, đã kiểm offset"
   ├─► split.py        ─► train/dev/test    "chia theo NHÓM, không theo câu hỏi"
   └─► scripts/02_preprocess.py             "người điều phối, chỉ gọi 3 trạm trên"
```

Hai nguyên tắc sống còn mà mọi dòng code đều phục vụ:

- **Không bao giờ sửa `original_text`**, vì sửa là lệch `answer_start`. Chuẩn hóa chỉ làm trên bản `retrieval_text` riêng.
- **ID sinh từ hash nội dung**, nên chạy lại bao nhiêu lần vẫn ra cùng ID.

## Thứ tự làm

| # | Việc | File |
|---|---|---|
| 1 | Viết quy tắc ghi trước (**chưa có code**) | `docs/preprocessing_rules.md` |
| 2 | Thêm hằng số | `src/config.py` |
| 3 | Công cụ đọc/ghi + hash | `src/data/io_utils.py` |
| 4 | Chuẩn hóa văn bản, sinh ID | `src/data/text_utils.py` |
| 5 | Kiểm tra offset | `src/data/validate_spans.py` |
| 6 | Dựng corpus | `src/data/build_corpus.py` |
| 7 | Dựng QA + qrels | `src/data/build_qa.py` |
| 8 | Chia tập | `src/data/split.py` |
| 9 | Điều phối | `scripts/02_preprocess.py` |
| 10 | Nghiệm thu | `tests/test_phase2_acceptance.py` |

## Bước 1: `docs/preprocessing_rules.md`

Phải viết **trước khi chạy** bất kỳ phép lọc nào. Nội dung đề xuất (đã khớp với code):

```markdown
# Quy tắc tiền xử lý (ghi trước khi chạy)
R1  Offset sai, đáp án xuất hiện đúng 1 lần  -> sửa bằng tìm chuỗi, ghi span_fix_log.jsonl
R2  Đáp án xuất hiện nhiều lần / không thấy   -> cách ly vào quarantine.jsonl, không đoán
R3  Mẫu train trùng câu hỏi/passage với test  -> loại khỏi train, không bao giờ loại khỏi test
R4  Hai passage giống hệt văn bản             -> gộp 1, giữ đủ provenance trong source[]
R5  Tỷ lệ train/dev                           -> 90/10, chia theo nhóm liên thông
R6  Passage DPR vào corpus                    -> có: positive + hard_negative
R7  Seed                                      -> 42
R8  Câu hỏi trùng văn bản trong cùng split    -> gộp đáp án nếu cùng passage;
                                                 khác passage thì chỉ hợp nhất vào qrels
R9  GermanDPR không có offset                 -> tìm đáp án trong positive passage;
                                                 duy nhất -> lấy vị trí; mơ hồ -> start=None (không cách ly)
```

## Bước 2: thêm vào `src/config.py`

```python
PROCESSED_DIR = ROOT / "data" / "processed" / "v1"
SEED = 42
DEV_RATIO = 0.10
```

## Bước 3: `src/data/io_utils.py`

Một "hộp dụng cụ" để mọi module dùng chung cách ghi file, nên chỉ có một nơi quy định `ensure_ascii=False`, `sort_keys=True`. Chính `sort_keys` làm hai lần chạy ra file giống hệt nhau.

```python
import hashlib
import json
from pathlib import Path


def sha1_hex(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```

## Bước 4: `src/data/text_utils.py`

Đây là nơi thể hiện nguyên tắc "tách văn bản cho reader và cho retrieval". Hàm `make_retrieval_text` mới được hạ chữ thường, còn `original_text` đi thẳng vào corpus không đổi.

```python
import re
import unicodedata

from src.data.io_utils import sha1_hex

_WS = re.compile(r"\s+")


def _normalize(text):
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def make_retrieval_text(original_text):
    """Bản phục vụ tìm kiếm. original_text KHÔNG bị sửa."""
    return _normalize(original_text).lower()


def make_dedup_key(original_text):
    """Khóa trùng CHÍNH XÁC trên văn bản gốc."""
    return sha1_hex(original_text)


def passage_id_from_key(key):
    return "p_" + key[:16]


def make_question_id(question):
    return "q_" + sha1_hex(_normalize(question).lower())[:16]


def make_document_id(title, fallback_key):
    t = _normalize(title).lower() if title else ""
    if t:
        return "d_" + sha1_hex(t)[:12]
    return "d_" + fallback_key[:12]
```

**Lưu ý về `document_id`:** GermanQuAD **không có cột title**, nên context của QuAD tự thành một "bài" riêng. Đây là giới hạn phải ghi vào báo cáo; nhóm chỉ gộp được qua context chung và qua title của DPR.

## Bước 5: `src/data/validate_spans.py`

Hàm này là "máy soi offset". Chú ý `start=None` dành cho GermanDPR (không có offset).

```python
def find_all(context, text):
    positions, i = [], context.find(text)
    while i != -1:
        positions.append(i)
        i = context.find(text, i + 1)
    return positions


def validate(context, text, start):
    """Trả về (status, start_đúng).
    start=None nghĩa là dataset không cung cấp offset (GermanDPR)."""
    if not text:
        return "QUARANTINE_EMPTY", None
    if start is not None and start >= 0 and context[start:start + len(text)] == text:
        return "OK", start
    pos = find_all(context, text)
    if len(pos) == 1:
        return ("FIXED" if start is not None else "LOCATED"), pos[0]
    if len(pos) > 1:
        return "QUARANTINE_AMBIGUOUS", None
    return "QUARANTINE_NOT_FOUND", None
```

## Bước 6: `src/data/build_corpus.py`

Mỗi passage được dò theo hash văn bản. Khi trùng, **không vứt provenance** mà nối thêm vào `source[]`.

```python
from src.data.text_utils import (make_dedup_key, make_document_id,
                                 make_retrieval_text, passage_id_from_key)


def _add(corpus, text, title, source):
    key = make_dedup_key(text)
    rec = corpus.get(key)
    if rec is None:
        rec = {
            "passage_id": passage_id_from_key(key),
            "original_text": text,
            "title": title or None,
            "document_id": make_document_id(title, key),
            "source": [],
            "retrieval_text": make_retrieval_text(text),
        }
        corpus[key] = rec
    elif title and not rec["title"]:
        rec["title"] = title
        rec["document_id"] = make_document_id(title, key)
    rec["source"].append(dict(source, title=title or None))


def build_corpus(raw):
    """raw = {"germanquad": DatasetDict, "germandpr": DatasetDict}"""
    corpus, n_input = {}, 0

    for split, ds in raw["germanquad"].items():
        for row in ds:
            n_input += 1
            _add(corpus, row["context"], None,
                 {"dataset": "germanquad", "split": split,
                  "source_id": str(row["id"]), "role": "context"})

    for split, ds in raw["germandpr"].items():
        for row in ds:
            for role, field in (("positive", "positive_ctxs"),
                                ("hard_negative", "hard_negative_ctxs")):
                c = row[field]
                for title, text, pid in zip(c["title"], c["text"], c["passage_id"]):
                    n_input += 1
                    _add(corpus, text, title,
                         {"dataset": "germandpr", "split": split,
                          "source_id": pid, "role": role})

    records = sorted(corpus.values(), key=lambda r: r["passage_id"])
    ids = [r["passage_id"] for r in records]
    assert len(set(ids)) == len(ids), "va chạm passage_id"
    stats = {
        "passages_before_dedup": n_input,
        "passages_after_dedup": len(records),
        "passages_multi_provenance": sum(1 for r in records if len(r["source"]) > 1),
    }
    return records, stats
```

`positive_ctxs` của DPR là **dict các list** (cột-chủ-đạo), nên phải `zip(title, text, passage_id)`, không duyệt như list các dict.

## Bước 7: `src/data/build_qa.py`

```python
from src.data.text_utils import (make_dedup_key, make_question_id,
                                 passage_id_from_key)
from src.data.validate_spans import validate


def _pid(text):
    return passage_id_from_key(make_dedup_key(text))


def _merge_answers(prev, new):
    have = {a["text"] for a in prev["answers"]}
    for a in new:
        if a["text"] not in have:
            prev["answers"].append(a)
            have.add(a["text"])


def _add_row(pool, qrels, row, pids, stats):
    qid = row["question_id"]
    if qid not in pool:
        pool[qid] = row
        qrels[qid] = set(pids)
        return
    prev = pool[qid]
    if prev["passage_id"] == row["passage_id"]:
        _merge_answers(prev, row["answers"])
        stats["duplicate_same_passage_merged"] += 1
    else:
        stats["duplicate_other_passage_to_qrels_only"] += 1
    qrels[qid] |= set(pids)


def build_qa(raw):
    """Trả về pools[split] = {qid: row}, qrels[split] = {qid: set(pid)},
    hardneg[split] = {qid: [pid]}, quarantine, fixlog, stats."""
    pools = {"train": {}, "test": {}}
    qrels = {"train": {}, "test": {}}
    hardneg = {"train": {}, "test": {}}
    quarantine, fixlog = [], []
    stats = {"duplicate_same_passage_merged": 0,
             "duplicate_other_passage_to_qrels_only": 0,
             "span_status": {}, "input": {}, "kept": {}}

    def count(status):
        stats["span_status"][status] = stats["span_status"].get(status, 0) + 1

    # --- GermanQuAD trước (offset đáng tin hơn), rồi tới GermanDPR ---
    for split, ds in raw["germanquad"].items():
        stats["input"][f"germanquad/{split}"] = len(ds)
        for row in ds:
            ctx, qid = row["context"], make_question_id(row["question"])
            pid = _pid(ctx)
            good, seen = [], set()
            for t, s in zip(row["answers"]["text"], row["answers"]["answer_start"]):
                status, new = validate(ctx, t, s)
                count(status)
                if status == "OK":
                    a = (t, s)
                elif status == "FIXED":
                    fixlog.append({"question_id": qid, "dataset": "germanquad",
                                   "split": split, "text": t,
                                   "old_start": s, "new_start": new})
                    a = (t, new)
                else:
                    quarantine.append({"question_id": qid, "dataset": "germanquad",
                                       "split": split, "scope": "answer",
                                       "reason": status, "text": t, "start": s})
                    continue
                if a not in seen:
                    seen.add(a)
                    good.append({"text": a[0], "start": a[1]})
            if not good:
                quarantine.append({"question_id": qid, "dataset": "germanquad",
                                   "split": split, "scope": "question",
                                   "reason": "NO_VALID_ANSWER"})
                continue
            _add_row(pools[split], qrels[split],
                     {"question_id": qid, "question": row["question"],
                      "passage_id": pid, "answers": good,
                      "origin": "germanquad"}, [pid], stats)

    for split, ds in raw["germandpr"].items():
        stats["input"][f"germandpr/{split}"] = len(ds)
        for row in ds:
            qid = make_question_id(row["question"])
            pos = row["positive_ctxs"]["text"]
            if not pos:
                quarantine.append({"question_id": qid, "dataset": "germandpr",
                                   "split": split, "scope": "question",
                                   "reason": "NO_POSITIVE"})
                continue
            pids = [_pid(t) for t in pos]
            answers, seen = [], set()
            for t in row["answers"]:
                status, new = validate(pos[0], t, None)
                count("DPR_" + status)
                start = new if status == "LOCATED" else None
                if (t, start) not in seen:
                    seen.add((t, start))
                    answers.append({"text": t, "start": start})
            _add_row(pools[split], qrels[split],
                     {"question_id": qid, "question": row["question"],
                      "passage_id": pids[0], "answers": answers,
                      "origin": "germandpr"}, pids, stats)
            hn = [_pid(t) for t in row["hard_negative_ctxs"]["text"]]
            if hn:
                hardneg[split].setdefault(qid, [])
                for p in hn:
                    if p not in hardneg[split][qid]:
                        hardneg[split][qid].append(p)

    for split in pools:
        stats["kept"][split] = len(pools[split])
    return pools, qrels, hardneg, quarantine, fixlog, stats
```

Tại sao **QuAD xử lý trước DPR**: khi cùng một câu hỏi xuất hiện ở cả hai, bản QuAD có offset đáng tin hơn nên được giữ làm bản chính. DPR chỉ bổ sung đáp án và qrels.

## Bước 8: `src/data/split.py`

Phần quan trọng nhất. Chia theo **nhóm liên thông** bằng union-find: câu hỏi ↔ passage ↔ bài viết. Hai câu hỏi cùng context luôn rơi vào cùng một split.

```python
import random


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def remove_conflicts_with_test(train_pool, train_qrels, test_pool, test_qrels):
    """R3: loại khỏi TRAIN mẫu trùng câu hỏi/passage với test. Không đụng test."""
    test_qids = set(test_pool)
    test_pids = {p for ps in test_qrels.values() for p in ps}
    kept, excluded = {}, []
    for qid, row in train_pool.items():
        if qid in test_qids:
            excluded.append({"question_id": qid, "reason": "QUESTION_IN_TEST"})
        elif train_qrels[qid] & test_pids:
            excluded.append({"question_id": qid, "reason": "PASSAGE_IN_TEST"})
        else:
            kept[qid] = row
    return kept, excluded


def make_groups(pool, qrels, pid_to_doc):
    """Nhóm liên thông: câu hỏi <-> passage <-> bài viết."""
    uf = UnionFind()
    for qid in pool:
        for pid in qrels[qid]:
            uf.union("q:" + qid, "p:" + pid)
            uf.union("p:" + pid, "d:" + pid_to_doc[pid])
    groups = {}
    for qid in pool:
        groups.setdefault(uf.find("q:" + qid), []).append(qid)
    return list(groups.values())


def split_train_dev(pool, qrels, pid_to_doc, seed, dev_ratio):
    groups = make_groups(pool, qrels, pid_to_doc)
    groups = [sorted(g) for g in groups]
    groups.sort(key=lambda g: g[0])             # thứ tự ổn định
    random.Random(seed).shuffle(groups)
    target = round(dev_ratio * len(pool))
    dev = []
    for g in groups:
        if len(dev) >= target:
            break
        if len(dev) + len(g) > target * 1.2:    # bỏ qua nhóm quá lớn
            continue
        dev.extend(g)
    dev_set = set(dev)
    train = [q for g in groups for q in g if q not in dev_set]
    sizes = sorted((len(g) for g in groups), reverse=True)
    info = {"n_groups": len(groups), "largest_groups": sizes[:5],
            "target_dev": target}
    return sorted(train), sorted(dev), info
```

Dòng `groups.sort(...)` **trước** `shuffle` là chỗ then chốt cho tái lập: thứ tự duyệt dict có thể khác nhau giữa các lần, nhưng sắp xếp trước thì `shuffle(seed)` luôn cho cùng kết quả.

## Bước 9: `scripts/02_preprocess.py`

```python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk

from src.config import (DATASETS, DEV_RATIO, PROCESSED_DIR, RAW_DIR,
                        REPORT_DIR, SEED)
from src.data.build_corpus import build_corpus
from src.data.build_qa import build_qa
from src.data.io_utils import read_json, write_json, write_jsonl
from src.data.split import remove_conflicts_with_test, split_train_dev


def main(out_dir):
    out = Path(out_dir)

    raw = {name: load_from_disk(str(RAW_DIR / name)) for name in DATASETS}

    corpus, corpus_stats = build_corpus(raw)
    pid_to_doc = {r["passage_id"]: r["document_id"] for r in corpus}

    pools, qrels, hardneg, quarantine, fixlog, qa_stats = build_qa(raw)

    # mọi passage_id trong qrels phải có trong corpus
    for split in qrels:
        for ps in qrels[split].values():
            assert ps <= set(pid_to_doc), "qrels trỏ tới passage không có trong corpus"

    train_pool, excluded = remove_conflicts_with_test(
        pools["train"], qrels["train"], pools["test"], qrels["test"])
    train_ids, dev_ids, split_info = split_train_dev(
        train_pool, qrels["train"], pid_to_doc, SEED, DEV_RATIO)

    qa = {
        "train": [train_pool[q] for q in train_ids],
        "dev": [train_pool[q] for q in dev_ids],
        "test": [pools["test"][q] for q in sorted(pools["test"])],
    }
    qr = {
        "train": {q: sorted(qrels["train"][q]) for q in train_ids},
        "dev": {q: sorted(qrels["train"][q]) for q in dev_ids},
        "test": {q: sorted(qrels["test"][q]) for q in sorted(pools["test"])},
    }

    write_jsonl(out / "corpus.jsonl", corpus)
    for s in qa:
        write_jsonl(out / f"qa_{s}.jsonl", qa[s])
        write_json(out / f"qrels_{s}.json", qr[s])
    write_json(out / "hardneg_train.json",
               {q: hardneg["train"][q] for q in train_ids if q in hardneg["train"]})
    write_jsonl(out / "quarantine.jsonl", quarantine)
    write_jsonl(out / "span_fix_log.jsonl", fixlog)
    write_jsonl(out / "excluded_conflict.jsonl", excluded)

    step1 = read_json(REPORT_DIR / "step1_manifest.json")
    write_json(out / "split_manifest.json", {
        "corpus_version": "v1",
        "raw_revisions": {n: d["revision_sha"] for n, d in step1["datasets"].items()},
        "seed": SEED, "ratio": f"{round((1 - DEV_RATIO) * 100)}/{round(DEV_RATIO * 100)}",
        "preprocessing_config": {
            "R1": "offset sai, đáp án xuất hiện đúng 1 lần -> sửa + ghi log",
            "R2": "đáp án nhiều lần/không thấy -> quarantine",
            "R3": "train trùng test -> loại khỏi train",
            "R4": "passage trùng văn bản -> gộp, giữ provenance",
            "R5": "dev 10% theo nhóm liên thông",
            "R6": "DPR positive + hard_negative vào corpus",
            "R7": f"seed={SEED}",
        },
        "split_ids": {"train": train_ids, "dev": dev_ids,
                      "test": [r["question_id"] for r in qa["test"]]},
    })

    ids = {s: {r["question_id"] for r in qa[s]} for s in qa}
    write_json(out / "data_quality_report.json", {
        "corpus": corpus_stats,
        "qa": qa_stats,
        "excluded_conflict": {
            "count": len(excluded),
            "by_reason": {r: sum(1 for e in excluded if e["reason"] == r)
                          for r in {e["reason"] for e in excluded}}},
        "split": dict(split_info, sizes={s: len(qa[s]) for s in qa}),
        "overlap": {"train&dev": len(ids["train"] & ids["dev"]),
                    "train&test": len(ids["train"] & ids["test"]),
                    "dev&test": len(ids["dev"] & ids["test"])},
        "quarantine_total": len(quarantine),
        "span_fixed": len(fixlog),
    })
    print("Đã ghi vào", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    main(ap.parse_args().out)
```

Tham số `--out` để test nghiệm thu chạy lại vào thư mục tạm mà không ghi đè bản thật.

## Bước 10: `tests/test_phase2_acceptance.py`

```python
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from src.config import PROCESSED_DIR
from src.data.io_utils import file_sha1, read_json, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "dev", "test")


@pytest.fixture(scope="module")
def corpus():
    return {r["passage_id"]: r for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}


@pytest.fixture(scope="module")
def qa():
    return {s: read_jsonl(PROCESSED_DIR / f"qa_{s}.jsonl") for s in SPLITS}


def test_spans_valid(corpus, qa):
    for rows in qa.values():
        for r in rows:
            text = corpus[r["passage_id"]]["original_text"]
            for a in r["answers"]:
                if a["start"] is not None:
                    assert text[a["start"]:a["start"] + len(a["text"])] == a["text"]


def test_qrels_in_corpus(corpus):
    for s in SPLITS:
        for pids in read_json(PROCESSED_DIR / f"qrels_{s}.json").values():
            assert set(pids) <= set(corpus)


def test_no_overlap(qa):
    ids = {s: {r["question_id"] for r in qa[s]} for s in SPLITS}
    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["test"]
    assert not ids["dev"] & ids["test"]


def test_unique_ids(corpus, qa):
    for rows in qa.values():
        assert len({r["question_id"] for r in rows}) == len(rows)


def test_quarantine_has_reason():
    for r in read_jsonl(PROCESSED_DIR / "quarantine.jsonl"):
        assert r.get("reason")


def test_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "02_preprocess.py"),
                        "--out", tmp], check=True, cwd=ROOT)
        for name in ("corpus.jsonl", "qa_train.jsonl", "qa_dev.jsonl",
                     "qa_test.jsonl", "split_manifest.json"):
            assert file_sha1(Path(tmp) / name) == file_sha1(PROCESSED_DIR / name)
```

Thêm `pytest` vào `requirements.txt`. Chạy:

```powershell
python scripts/02_preprocess.py
python -m pytest tests -q
```

Dùng `python -m pytest` (không phải `pytest` trần) để thư mục gốc nằm trong `sys.path`, nếu không `from src...` sẽ lỗi.

> **Về độ tin cậy:** pipeline thì tao đã chạy thật; các phép kiểm tra tương đương `test_spans_valid`, `test_qrels_in_corpus`, `test_no_overlap`, `test_unique_ids` cũng đã chạy và đạt. Riêng **file pytest ở trên chưa được chạy** vì tao chưa cài `pytest` trong venv của mày.

## Kết quả mày sẽ thấy (đã chạy trên dữ liệu thật)

| Chỉ số | Giá trị |
|---|---|
| Passage trước / sau dedup | 54 908 → **26 864** |
| Passage có nhiều nguồn | 6 424 |
| Span kiểm tra: đúng / sai | 16 065 / **0** |
| Cách ly / sửa offset | 0 / 0 |
| `qa_train` / `qa_dev` / `qa_test` | **9 368 / 1 067 / 3 228** |
| Overlap train∩dev, train∩test, dev∩test | 0, 0, 0 |
| Bị loại khỏi train do trùng test | 1 042 |
| Số nhóm liên thông | 567 (lớn nhất 212 câu hỏi) |

## Phát hiện cần ghi vào báo cáo (và cần mày quyết định)

1. **GermanDPR test nằm trọn trong GermanQuAD train.** Cả 1 025/1 025 câu hỏi của DPR test đều có mặt ở QuAD train, trong khi DPR test ∩ QuAD test = 0. Nếu không chặn, mô hình huấn luyện trên QuAD train sẽ đã thấy đúng câu hỏi test của DPR. Quy tắc R3 đã loại 1 025 câu này khỏi train, cộng 17 câu trùng passage.
2. **Quyết định của mày:** hiện `qa_test` = QuAD test (2 204) + DPR test (1 025) − 1 trùng = **3 228**. Nếu muốn test thuần QuAD để so sánh công bằng với paper GermanQuAD thì phải đổi. Phải chốt và ghi vào `preprocessing_rules.md` *trước khi* xem điểm.
3. **GermanQuAD sạch về offset**: 18 054/18 054 đáp án `OK`, không cần sửa hay cách ly. Đó chính là một kết luận đáng đưa vào báo cáo, không phải "thiếu việc".
4. **615 đáp án DPR mơ hồ** (xuất hiện nhiều lần trong positive passage) được giữ với `start = None`, **không cách ly**, vì câu hỏi vẫn dùng được cho retrieval.
5. **41 câu hỏi QuAD train trùng văn bản** (11 518 → 11 477 duy nhất), và 29 trường hợp cùng câu hỏi nhưng khác passage được hợp nhất vào qrels.

## Giới hạn cần ghi (theo đề)

- GermanQuAD không có `title` → nhóm chia tập chỉ gộp được qua context/qrels chung, không gộp theo bài viết thật.
- Passage không có trong qrels **chưa chắc không liên quan**.
- `negative_ctxs` (negative thường) của DPR chưa đưa vào corpus, chỉ `positive` và `hard_negative`.

Mày muốn tao copy các file đã kiểm chứng này vào đúng vị trí trong `nlp_v1` (và cài `pytest`, chạy nghiệm thu), hay mày tự gõ theo hướng dẫn?


---

## 🧑 USER #18

Mày hãy trình bày lại và giải thích từng hàm đang cài đặt sẽ làm nhiệm vụ gì, mày đưa tao quá nhiều code nhưng không giải thích


---

## 🤖 CLAUDE #19

Mày nói đúng, tao đưa code nhiều mà ít giải thích. Lần này tao đi từng hàm, theo thứ tự dữ liệu chảy qua, và chỉ giữ code ở chỗ cần minh họa.

## Ví dụ xuyên suốt

Mọi hàm bên dưới đều xử lý cùng một thứ: **một câu hỏi, đáp án của nó, và đoạn văn chứa đáp án**.

```
question : "Wer gründete Rom?"
context  : "... Romulus gründete Rom im Jahr 753 v. Chr. ..."
answer   : text="Romulus", answer_start=4
```

Giai đoạn 2 biến hàng chục nghìn bộ ba như vậy thành dữ liệu sạch, có ID cố định và đã chia train/dev/test.

## Nhóm 1: `io_utils.py` (đọc/ghi file)

Nhóm này không có logic nghiệp vụ, chỉ để **mọi module ghi file cùng một cách**.

| Hàm | Làm gì | Vì sao cần |
|---|---|---|
| `sha1_hex(text)` | Biến một chuỗi thành "dấu vân tay" 40 ký tự hex. Cùng chuỗi luôn cho cùng dấu vân tay. | Nền tảng để sinh ID từ nội dung. |
| `file_sha1(path)` | Dấu vân tay của cả một file. | Test nghiệm thu dùng nó để chứng minh chạy hai lần ra file giống hệt. |
| `write_jsonl` / `read_jsonl` | Ghi/đọc file mà **mỗi dòng là một bản ghi JSON**. | Corpus và QA có hàng chục nghìn dòng, đọc từng dòng nhẹ hơn nạp một file khổng lồ. |
| `write_json` / `read_json` | Ghi/đọc một file JSON nguyên khối. | Dùng cho qrels, manifest, báo cáo. |

Hai tuỳ chọn đáng nhớ: `ensure_ascii=False` để tiếng Đức (ä, ö, ü) không bị biến thành `\u00e4`, và `sort_keys=True` để thứ tự khoá cố định, nhờ đó hai lần chạy ra file **giống từng byte**.

## Nhóm 2: `text_utils.py` (tạo ID và chuẩn hóa)

Nguyên tắc cốt lõi của file này: **`original_text` không bao giờ bị sửa**, vì sửa là lệch `answer_start`. Mọi chuẩn hóa chỉ áp dụng cho bản sao.

**`_normalize(text)`** là hàm phụ nội bộ. Nó đưa chữ về dạng Unicode NFC (cùng một chữ "ü" có thể được lưu bằng một hoặc hai ký tự, NFC thống nhất lại), gộp nhiều khoảng trắng thành một, rồi cắt khoảng trắng hai đầu.

**`make_retrieval_text(original_text)`** gọi `_normalize` rồi hạ chữ thường. Kết quả là bản dành cho công cụ tìm kiếm (BM25 về sau). Reader vẫn đọc `original_text`. Tách đôi như vậy để tìm kiếm không phân biệt hoa/thường mà offset đáp án vẫn đúng.

**`make_dedup_key(original_text)`** trả về hash của văn bản **gốc, nguyên văn**. Hai passage chỉ coi là trùng khi giống hệt từng ký tự. Cố ý không dùng bản đã chuẩn hóa, vì gộp hai passage "gần giống" mà không có ánh xạ offset sẽ làm hỏng đáp án.

**`passage_id_from_key(key)`** lấy 16 ký tự đầu của hash, thêm tiền tố: `p_3fa9c1...`. Vì ID sinh từ nội dung nên chạy lại bao nhiêu lần vẫn ra cùng ID, và không phụ thuộc thứ tự duyệt file.

**`make_question_id(question)`** tương tự, nhưng hash trên câu hỏi đã chuẩn hóa và hạ chữ thường: `q_8b21...`. Nhờ vậy cùng một câu hỏi xuất hiện ở cả QuAD lẫn DPR sẽ nhận **cùng một ID**, và đó là cách hai dataset được nối lại với nhau.

**`make_document_id(title, fallback_key)`** tạo ID "bài viết" từ tiêu đề. Nếu không có tiêu đề thì dùng hash của passage. GermanQuAD **không có cột title**, nên mỗi context của QuAD tự thành một "bài" riêng. Đây là giới hạn phải ghi vào báo cáo.

## Nhóm 3: `validate_spans.py` (kiểm tra đáp án có nằm đúng chỗ không)

**`find_all(context, text)`** trả về mọi vị trí mà `text` xuất hiện trong `context`. Với "Romulus" xuất hiện hai lần thì nó trả `[4, 120]`.

**`validate(context, text, start)`** là "máy soi offset". Nó trả về một cặp `(trạng_thái, start_đúng)` theo thứ tự sau:

| Điều kiện | Trạng thái | Ý nghĩa |
|---|---|---|
| Đáp án rỗng | `QUARANTINE_EMPTY` | Không có gì để kiểm. |
| `context[start : start+len] == text` | `OK` | Offset đúng, giữ nguyên. |
| Offset sai nhưng chuỗi xuất hiện **đúng 1 lần** | `FIXED` (hoặc `LOCATED` nếu vốn không có offset) | Sửa được, chắc chắn. |
| Chuỗi xuất hiện **nhiều lần** | `QUARANTINE_AMBIGUOUS` | Không biết lần nào là đúng, **không đoán**. |
| Không thấy chuỗi | `QUARANTINE_NOT_FOUND` | Dữ liệu lỗi. |

Quy tắc tương ứng R1/R2 đã ghi trước. Tham số `start=None` dành cho GermanDPR vì DPR chỉ có chữ đáp án, không có vị trí, nên hàm tự đi tìm.

## Nhóm 4: `build_corpus.py` (kho passage)

**`_add(corpus, text, title, source)`** thêm một passage vào kho, có hai nhánh:

- **Passage chưa có**: tạo bản ghi mới gồm `passage_id`, `original_text`, `title`, `document_id`, `retrieval_text`, và danh sách `source` rỗng.
- **Passage đã có** (trùng văn bản): **không tạo bản mới**, chỉ bổ sung thêm tiêu đề nếu bản cũ chưa có.

Cả hai nhánh đều **nối thêm nguồn gốc vào `source`**. Một passage có thể xuất hiện ở QuAD train, DPR train và DPR test cùng lúc, nên bản ghi sau khi gộp sẽ ghi đủ cả ba nơi. Đó là "provenance".

**`build_corpus(raw)`** duyệt toàn bộ dữ liệu thô và gọi `_add` cho từng passage:

1. Mỗi `context` của GermanQuAD (train + test).
2. Mỗi passage `positive` và `hard_negative` của GermanDPR (train + test).

Cuối cùng nó sắp theo `passage_id` (để thứ tự ổn định), kiểm tra `assert` rằng không có hai passage khác nhau mà trùng ID, rồi trả về danh sách bản ghi cùng thống kê.

Kết quả thực tế: **54 908 passage thô → 26 864 sau khi gộp**, trong đó 6 424 passage có nhiều nguồn.

## Nhóm 5: `build_qa.py` (câu hỏi ↔ passage)

**`_pid(text)`** là đường tắt: hash văn bản rồi đổi thành `passage_id`. Dùng để biết một `context` tương ứng passage nào trong corpus.

**`_merge_answers(prev, new)`** gộp danh sách đáp án mới vào bản ghi cũ, **bỏ đáp án đã có** (so theo chữ). Tránh việc "Romulus" bị ghi hai lần.

**`_add_row(pool, qrels, row, pids, stats)`** xử lý câu hỏi trùng ID, vì cùng một câu hỏi có thể đến từ nhiều dòng. Nó rẽ ba nhánh:

- **ID chưa có**: thêm mới, và ghi các passage liên quan vào `qrels`.
- **ID đã có, cùng passage**: gộp đáp án vào bản cũ (đếm vào `duplicate_same_passage_merged`).
- **ID đã có, khác passage**: **không** đè lên bản cũ (đáp án của passage này không có nghĩa trên passage kia), chỉ thêm passage mới vào `qrels` (đếm vào `duplicate_other_passage_to_qrels_only`).

**`build_qa(raw)`** là hàm lớn nhất, làm hai phần.

*Phần GermanQuAD.* Với mỗi câu hỏi, nó gọi `validate` cho từng đáp án rồi quyết định:

- `OK` → giữ nguyên.
- `FIXED` → dùng vị trí mới và ghi vào `span_fix_log` (đây chính là "mapping" đề yêu cầu).
- Còn lại → ghi vào `quarantine` và bỏ đáp án đó. Chỉ khi **không còn đáp án hợp lệ nào** thì mới cách ly cả câu hỏi. Điều này quan trọng với tập test của QuAD, vì mỗi câu có nhiều đáp án.

*Phần GermanDPR.* DPR không có offset, nên nó lấy passage `positive` đầu tiên làm passage chính rồi gọi `validate(..., start=None)` để **tự tìm** đáp án. Tìm được duy nhất thì ghi `start`; mơ hồ thì để `start=None` nhưng **vẫn giữ câu hỏi** vì nó vẫn dùng được cho retrieval. Nó cũng gom `hard_negative` của từng câu vào `hardneg`.

Thứ tự **QuAD trước, DPR sau** là cố ý: khi trùng câu hỏi, bản QuAD có offset đáng tin hơn nên làm bản chính.

Đầu ra gồm: các "bể" câu hỏi theo split, `qrels` (câu hỏi nào ↔ passage nào là đúng), `hardneg`, danh sách cách ly, log sửa offset, và thống kê.

## Nhóm 6: `split.py` (chia tập)

Đây là phần tinh tế nhất. Bài toán: nếu chia **theo câu hỏi**, hai câu hỏi cùng hỏi về một đoạn văn có thể rơi vào train và dev khác nhau. Mô hình đã "thuộc" đoạn đó từ train, nên điểm dev cao giả. Cách đúng là chia **theo nhóm**.

**`UnionFind`** là cấu trúc dữ liệu "gom nhóm". Hai phương thức:

- `union(a, b)`: ghi nhận "a và b cùng một nhóm".
- `find(x)`: cho biết x thuộc nhóm nào (trả về đại diện của nhóm).

Nó gom được cả các quan hệ gián tiếp. Nếu A cùng nhóm B, B cùng nhóm C thì A, C tự động cùng nhóm.

**`remove_conflicts_with_test(...)`** thực thi quy tắc R3. Với mỗi câu hỏi của tập train, nếu câu hỏi đó **hoặc passage của nó** có mặt trong test thì loại khỏi train và ghi lý do (`QUESTION_IN_TEST` hay `PASSAGE_IN_TEST`). Tập test thì **không bao giờ bị đụng đến**. Đây là điểm đạo đức quan trọng: không được bỏ mẫu khó khỏi test để điểm đẹp hơn.

**`make_groups(pool, qrels, pid_to_doc)`** dùng `UnionFind` để nối câu hỏi ↔ passage ↔ bài viết. Kết quả là các **nhóm liên thông**: mỗi nhóm là tập câu hỏi có liên hệ với nhau qua passage hoặc bài viết chung. Thực tế ra 567 nhóm, nhóm lớn nhất có 212 câu hỏi.

**`split_train_dev(...)`** chia cả nhóm vào dev cho đến khi đủ khoảng 10%:

1. Sắp xếp các nhóm theo thứ tự ổn định.
2. Xáo trộn bằng `random.Random(seed=42)`.
3. Lấy dần nhóm vào dev cho tới khi đạt mục tiêu.
4. Nhóm nào làm dev vượt quá 120% mục tiêu thì bỏ qua (tránh một nhóm khổng lồ làm lệch tỷ lệ).
5. Phần còn lại là train.

Bước **sắp xếp trước khi xáo** là then chốt để tái lập: thứ tự dict có thể khác nhau giữa các lần chạy, nhưng sắp xếp trước thì xáo với cùng seed luôn cho cùng kết quả. Vì dev lấy theo cả nhóm nên số thực tế (1 067) hơi lệch mục tiêu (1 044), đó là bình thường.

## Nhóm 7: `scripts/02_preprocess.py` → hàm `main`

`main` không có logic riêng, chỉ **điều phối theo thứ tự**:

```
đọc dữ liệu thô  →  dựng corpus  →  dựng QA/qrels  →  kiểm tra qrels ⊂ corpus
→  loại mẫu train trùng test  →  chia train/dev  →  ghi 13 file  →  in báo cáo
```

Có một câu `assert` ở giữa: mọi `passage_id` trong qrels phải có trong corpus. Nếu vi phạm thì chương trình **dừng ngay**, vì đó là lỗi cấu trúc thật chứ không phải "phát hiện cần báo cáo" như số lệch ở Giai đoạn 1.

Tham số `--out` cho phép chạy ghi vào thư mục khác, để test tái lập không ghi đè dữ liệu thật.

## Nhóm 8: `tests/` (nghiệm thu)

Mỗi test là một "câu hỏi kiểm tra" đối với dữ liệu đầu ra:

| Test | Câu hỏi nó trả lời |
|---|---|
| `test_spans_valid` | Cắt đúng đoạn theo `start` có ra đúng chữ đáp án không? |
| `test_qrels_in_corpus` | Mọi passage mà qrels nhắc đến có thật trong corpus không? |
| `test_no_overlap` | Train, dev, test có chung câu hỏi nào không? |
| `test_unique_ids` | Có ID nào bị trùng trong một split không? |
| `test_quarantine_has_reason` | Mọi mẫu bị cách ly đều có lý do chứ? |
| `test_reproducible` | Chạy lại từ đầu có ra file giống hệt không? |

## Tóm tắt bằng một sơ đồ

```
validate_spans   : "đáp án này có nằm đúng chỗ không?"
text_utils       : "gán ID cố định cho passage và câu hỏi"
build_corpus     : "gom mọi passage, bỏ trùng, nhớ nguồn gốc"
build_qa         : "gắn câu hỏi với passage đúng, đã kiểm đáp án"
split            : "chia theo nhóm, loại rò rỉ sang test"
test             : "chứng minh các điều trên là thật"
```

Mày thấy hàm nào vẫn còn mơ hồ không? Tao có thể đi sâu một hàm bằng một ví dụ chạy bằng tay (ví dụ `split_train_dev` hay `_add_row`, hai hàm khó hình dung nhất).


---

## 🧑 USER #20

Mày trình bày lại code và giải thích phía dưới code


---

## 🤖 CLAUDE #21

Mỗi hàm bên dưới là một khối code, và phần giải thích nằm ngay phía dưới khối đó. Code giống bản đã chạy thử, chỉ thêm vài dấu `# (1)`, `# (2)` ở hai hàm dài để phần giải thích tham chiếu được.

---

# File 1: `src/data/io_utils.py`

```python
import hashlib
import json
from pathlib import Path


def sha1_hex(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
```

**Giải thích:** Hàm nhận một chuỗi và trả về "dấu vân tay" dài 40 ký tự hex. Cùng một chuỗi thì dấu vân tay luôn giống nhau, còn đổi dù một ký tự thì khác hoàn toàn. Đây là nền tảng để sinh ID từ nội dung ở các file sau. `.encode("utf-8")` là bắt buộc vì hash làm việc trên byte chứ không phải chữ.

```python
def file_sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()
```

**Giải thích:** Cũng là dấu vân tay, nhưng của **cả một file** (đọc toàn bộ byte rồi hash). Chỉ test nghiệm thu dùng nó để so sánh hai lần chạy có ra file giống hệt không.

```python
def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
```

**Giải thích:** Ghi một đối tượng Python ra file JSON. Mấy chi tiết quan trọng:
- `mkdir(...)` tự tạo thư mục cha nếu chưa có, nên khỏi lo lỗi "không tìm thấy thư mục".
- `ensure_ascii=False` giữ nguyên chữ ä, ö, ü thay vì biến thành `\u00e4`.
- `sort_keys=True` cố định thứ tự khoá, nhờ đó hai lần chạy cho file giống từng byte.
- `newline="\n"` ép xuống dòng kiểu Unix, tránh Windows chèn `\r\n` làm hash khác giữa các máy.

```python
def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
```

**Giải thích:** Đọc ngược lại một file JSON thành đối tượng Python. Chỉ cần nhớ luôn truyền `encoding="utf-8"`, nếu không Windows sẽ đọc bằng bảng mã mặc định và vỡ tiếng Đức.

```python
def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
```

**Giải thích:** Ghi một danh sách bản ghi theo định dạng **JSON Lines**: mỗi bản ghi một dòng. Corpus có hơn 26 000 passage, định dạng này cho phép đọc từng dòng mà không phải nạp cả file khổng lồ vào bộ nhớ. Các tuỳ chọn có ý nghĩa y như `write_json`.

```python
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```

**Giải thích:** Đọc lại file JSONL thành danh sách. `if line.strip()` bỏ qua dòng trống để không lỗi khi file có dòng cuối rỗng.

---

# File 2: `src/data/text_utils.py`

```python
import re
import unicodedata

from src.data.io_utils import sha1_hex

_WS = re.compile(r"\s+")


def _normalize(text):
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()
```

**Giải thích:** Hàm phụ (dấu `_` đầu tên nghĩa là "dùng nội bộ"). Nó làm ba việc liên tiếp: đưa Unicode về dạng NFC (cùng một chữ "ü" có thể được lưu bằng một hoặc hai ký tự, NFC thống nhất chúng), gộp mọi dãy khoảng trắng/xuống dòng thành **một dấu cách**, rồi cắt khoảng trắng hai đầu. `_WS` được biên dịch một lần ở ngoài hàm để không phải tạo lại mỗi lần gọi.

```python
def make_retrieval_text(original_text):
    """Bản phục vụ tìm kiếm. original_text KHÔNG bị sửa."""
    return _normalize(original_text).lower()
```

**Giải thích:** Tạo **bản sao** dành cho công cụ tìm kiếm: chuẩn hóa rồi hạ chữ thường. Điểm cốt lõi: hàm này *không đụng* vào `original_text`. Nếu sửa bản gốc thì độ dài thay đổi và mọi `answer_start` bị lệch. Tách thành hai bản giúp tìm kiếm không phân biệt hoa/thường mà đáp án vẫn đúng vị trí.

```python
def make_dedup_key(original_text):
    """Khóa trùng CHÍNH XÁC trên văn bản gốc."""
    return sha1_hex(original_text)
```

**Giải thích:** Hash của văn bản **gốc, nguyên văn**. Hai passage chỉ bị coi là trùng khi giống hệt từng ký tự. Cố ý không hash bản đã chuẩn hóa: nếu gộp hai passage "gần giống" mà không có ánh xạ offset thì sẽ làm hỏng vị trí đáp án.

```python
def passage_id_from_key(key):
    return "p_" + key[:16]
```

**Giải thích:** Lấy 16 ký tự đầu của hash và thêm tiền tố `p_`, ra dạng `p_3fa9c1b2e4d5a601`. Vì ID sinh từ nội dung nên chạy lại bao nhiêu lần vẫn ra cùng ID, không phụ thuộc thứ tự duyệt file hay thứ tự máy.

```python
def make_question_id(question):
    return "q_" + sha1_hex(_normalize(question).lower())[:16]
```

**Giải thích:** ID cho câu hỏi, hash trên câu hỏi đã chuẩn hóa và hạ chữ thường. Hệ quả hay: cùng một câu hỏi xuất hiện ở cả GermanQuAD lẫn GermanDPR sẽ nhận **cùng một ID**. Đó là cách hai dataset được nối với nhau mà không cần ID gốc (vốn có thể rỗng hoặc trùng).

```python
def make_document_id(title, fallback_key):
    t = _normalize(title).lower() if title else ""
    if t:
        return "d_" + sha1_hex(t)[:12]
    return "d_" + fallback_key[:12]
```

**Giải thích:** ID cho "bài viết". Có tiêu đề thì hash tiêu đề, nên các passage cùng bài sẽ cùng `document_id`. Không có tiêu đề thì dùng hash của chính passage. GermanQuAD **không có cột title**, nên mỗi context của QuAD tự thành một "bài" riêng. Đây là giới hạn phải ghi vào báo cáo.

---

# File 3: `src/data/validate_spans.py`

```python
def find_all(context, text):
    positions, i = [], context.find(text)
    while i != -1:
        positions.append(i)
        i = context.find(text, i + 1)
    return positions
```

**Giải thích:** Trả về **mọi vị trí** mà `text` xuất hiện trong `context`. `str.find` chỉ cho vị trí đầu tiên và trả `-1` nếu không thấy, nên hàm lặp: tìm lần đầu, rồi tìm tiếp từ `i + 1` trở đi, dừng khi hết. Ví dụ "Romulus" xuất hiện ở vị trí 4 và 120 thì trả `[4, 120]`. Dùng `i + 1` (không phải `i + len`) để bắt được cả các lần xuất hiện chồng lên nhau.

```python
def validate(context, text, start):
    """Trả về (status, start_đúng).
    start=None nghĩa là dataset không cung cấp offset (GermanDPR)."""
    if not text:
        return "QUARANTINE_EMPTY", None
    if start is not None and start >= 0 and context[start:start + len(text)] == text:
        return "OK", start
    pos = find_all(context, text)
    if len(pos) == 1:
        return ("FIXED" if start is not None else "LOCATED"), pos[0]
    if len(pos) > 1:
        return "QUARANTINE_AMBIGUOUS", None
    return "QUARANTINE_NOT_FOUND", None
```

**Giải thích:** Đây là "máy soi offset". Nó kiểm tra theo thứ tự ưu tiên, dòng nào khớp trước thì trả luôn:

| Điều kiện | Trả về | Nghĩa |
|---|---|---|
| Đáp án rỗng | `QUARANTINE_EMPTY` | Không có gì để kiểm. |
| Cắt `context[start : start+len]` ra đúng chữ đáp án | `OK` | Offset đúng, giữ nguyên. |
| Offset sai/không có, nhưng chuỗi xuất hiện **đúng 1 lần** | `FIXED` (nếu vốn có offset) hoặc `LOCATED` (nếu không có offset) | Sửa được chắc chắn. |
| Chuỗi xuất hiện **nhiều lần** | `QUARANTINE_AMBIGUOUS` | Không biết lần nào đúng, **không đoán**. |
| Không thấy chuỗi | `QUARANTINE_NOT_FOUND` | Dữ liệu lỗi. |

Kết quả luôn là một cặp `(trạng_thái, vị_trí_đúng)`; vị trí là `None` khi bị cách ly. Điều kiện `start is not None` là để dùng được cho GermanDPR, nơi không có offset.

---

# File 4: `src/data/build_corpus.py`

```python
from src.data.text_utils import (make_dedup_key, make_document_id,
                                 make_retrieval_text, passage_id_from_key)


def _add(corpus, text, title, source):
    key = make_dedup_key(text)
    rec = corpus.get(key)
    if rec is None:
        rec = {
            "passage_id": passage_id_from_key(key),
            "original_text": text,
            "title": title or None,
            "document_id": make_document_id(title, key),
            "source": [],
            "retrieval_text": make_retrieval_text(text),
        }
        corpus[key] = rec
    elif title and not rec["title"]:
        rec["title"] = title
        rec["document_id"] = make_document_id(title, key)
    rec["source"].append(dict(source, title=title or None))
```

**Giải thích:** Thêm một passage vào kho (`corpus` là dict với khoá là hash văn bản). Hàm rẽ hai nhánh:

- **Passage chưa có** (`rec is None`): tạo bản ghi mới với đủ trường theo schema, kể cả `retrieval_text`.
- **Passage đã có** (trùng văn bản): *không* tạo bản mới. Chỉ khi bản cũ chưa có tiêu đề mà lần này có thì mới bổ sung tiêu đề và tính lại `document_id`.

Dòng cuối chạy ở **cả hai nhánh**: nối thêm nguồn gốc vào `source`. Nhờ vậy một passage xuất hiện ở QuAD train, DPR train và DPR test sẽ được ghi đủ ba nơi (provenance), dù kho chỉ giữ một bản.

```python
def build_corpus(raw):
    """raw = {"germanquad": DatasetDict, "germandpr": DatasetDict}"""
    corpus, n_input = {}, 0

    for split, ds in raw["germanquad"].items():
        for row in ds:
            n_input += 1
            _add(corpus, row["context"], None,
                 {"dataset": "germanquad", "split": split,
                  "source_id": str(row["id"]), "role": "context"})

    for split, ds in raw["germandpr"].items():
        for row in ds:
            for role, field in (("positive", "positive_ctxs"),
                                ("hard_negative", "hard_negative_ctxs")):
                c = row[field]
                for title, text, pid in zip(c["title"], c["text"], c["passage_id"]):
                    n_input += 1
                    _add(corpus, text, title,
                         {"dataset": "germandpr", "split": split,
                          "source_id": pid, "role": role})

    records = sorted(corpus.values(), key=lambda r: r["passage_id"])
    ids = [r["passage_id"] for r in records]
    assert len(set(ids)) == len(ids), "va chạm passage_id"
    stats = {
        "passages_before_dedup": n_input,
        "passages_after_dedup": len(records),
        "passages_multi_provenance": sum(1 for r in records if len(r["source"]) > 1),
    }
    return records, stats
```

**Giải thích:** Duyệt toàn bộ dữ liệu thô, gọi `_add` cho từng passage, theo hai vòng:

1. **GermanQuAD:** mỗi `context` là một passage. Không có tiêu đề nên truyền `None`; `source_id` là id của câu hỏi gốc.
2. **GermanDPR:** mỗi dòng có hai nhóm passage là `positive` và `hard_negative`. Chú ý `c["title"]`, `c["text"]`, `c["passage_id"]` là **ba danh sách song song** (cột-chủ-đạo), nên phải `zip` chúng lại để lấy từng bộ ba.

Sau khi duyệt xong, hàm sắp kho theo `passage_id` để thứ tự ổn định, rồi `assert` hai điều: không có hai passage khác nhau mà trùng ID (xác suất cực nhỏ nhưng cần chắc chắn). Cuối cùng trả về danh sách bản ghi và thống kê. Kết quả thực tế là 54 908 passage thô → 26 864 sau khi gộp, trong đó 6 424 có nhiều nguồn.

---

# File 5: `src/data/build_qa.py`

```python
from src.data.text_utils import (make_dedup_key, make_question_id,
                                 passage_id_from_key)
from src.data.validate_spans import validate


def _pid(text):
    return passage_id_from_key(make_dedup_key(text))
```

**Giải thích:** Đường tắt hai bước: văn bản → hash → `passage_id`. Dùng để biết một `context` tương ứng với passage nào trong corpus. Vì cùng công thức như lúc dựng corpus nên ID chắc chắn khớp.

```python
def _merge_answers(prev, new):
    have = {a["text"] for a in prev["answers"]}
    for a in new:
        if a["text"] not in have:
            prev["answers"].append(a)
            have.add(a["text"])
```

**Giải thích:** Gộp danh sách đáp án `new` vào bản ghi cũ `prev`, **bỏ những đáp án đã có** (so theo chữ). Tránh việc "Romulus" bị ghi hai lần khi cùng một câu hỏi đến từ hai dataset.

```python
def _add_row(pool, qrels, row, pids, stats):
    qid = row["question_id"]
    if qid not in pool:
        pool[qid] = row
        qrels[qid] = set(pids)
        return
    prev = pool[qid]
    if prev["passage_id"] == row["passage_id"]:
        _merge_answers(prev, row["answers"])
        stats["duplicate_same_passage_merged"] += 1
    else:
        stats["duplicate_other_passage_to_qrels_only"] += 1
    qrels[qid] |= set(pids)
```

**Giải thích:** Xử lý việc thêm một câu hỏi vào "bể" `pool`, đề phòng câu hỏi trùng ID. Ba tình huống:

- **ID chưa có:** thêm bản ghi mới, và ghi các passage liên quan vào `qrels`.
- **ID đã có, cùng passage:** gộp đáp án vào bản cũ.
- **ID đã có, khác passage:** *không* đè lên bản cũ, vì vị trí đáp án của passage này vô nghĩa trên passage kia. Chỉ ghi passage mới vào `qrels`.

Dòng `qrels[qid] |= set(pids)` chạy ở hai nhánh sau (`|=` là hợp tập hợp). Hai bộ đếm trong `stats` cho báo cáo biết có bao nhiêu trường hợp mỗi loại.

```python
def build_qa(raw):
    """Trả về pools[split] = {qid: row}, qrels[split] = {qid: set(pid)},
    hardneg[split] = {qid: [pid]}, quarantine, fixlog, stats."""
    pools = {"train": {}, "test": {}}
    qrels = {"train": {}, "test": {}}
    hardneg = {"train": {}, "test": {}}
    quarantine, fixlog = [], []
    stats = {"duplicate_same_passage_merged": 0,
             "duplicate_other_passage_to_qrels_only": 0,
             "span_status": {}, "input": {}, "kept": {}}

    def count(status):
        stats["span_status"][status] = stats["span_status"].get(status, 0) + 1

    # --- GermanQuAD trước (offset đáng tin hơn), rồi tới GermanDPR ---
    for split, ds in raw["germanquad"].items():                      # (1)
        stats["input"][f"germanquad/{split}"] = len(ds)
        for row in ds:
            ctx, qid = row["context"], make_question_id(row["question"])
            pid = _pid(ctx)
            good, seen = [], set()
            for t, s in zip(row["answers"]["text"], row["answers"]["answer_start"]):
                status, new = validate(ctx, t, s)                    # (2)
                count(status)
                if status == "OK":
                    a = (t, s)
                elif status == "FIXED":
                    fixlog.append({"question_id": qid, "dataset": "germanquad",
                                   "split": split, "text": t,
                                   "old_start": s, "new_start": new})
                    a = (t, new)
                else:
                    quarantine.append({"question_id": qid, "dataset": "germanquad",
                                       "split": split, "scope": "answer",
                                       "reason": status, "text": t, "start": s})
                    continue
                if a not in seen:                                    # (3)
                    seen.add(a)
                    good.append({"text": a[0], "start": a[1]})
            if not good:                                             # (4)
                quarantine.append({"question_id": qid, "dataset": "germanquad",
                                   "split": split, "scope": "question",
                                   "reason": "NO_VALID_ANSWER"})
                continue
            _add_row(pools[split], qrels[split],
                     {"question_id": qid, "question": row["question"],
                      "passage_id": pid, "answers": good,
                      "origin": "germanquad"}, [pid], stats)

    for split, ds in raw["germandpr"].items():                       # (5)
        stats["input"][f"germandpr/{split}"] = len(ds)
        for row in ds:
            qid = make_question_id(row["question"])
            pos = row["positive_ctxs"]["text"]
            if not pos:
                quarantine.append({"question_id": qid, "dataset": "germandpr",
                                   "split": split, "scope": "question",
                                   "reason": "NO_POSITIVE"})
                continue
            pids = [_pid(t) for t in pos]
            answers, seen = [], set()
            for t in row["answers"]:                                 # (6)
                status, new = validate(pos[0], t, None)
                count("DPR_" + status)
                start = new if status == "LOCATED" else None
                if (t, start) not in seen:
                    seen.add((t, start))
                    answers.append({"text": t, "start": start})
            _add_row(pools[split], qrels[split],
                     {"question_id": qid, "question": row["question"],
                      "passage_id": pids[0], "answers": answers,
                      "origin": "germandpr"}, pids, stats)
            hn = [_pid(t) for t in row["hard_negative_ctxs"]["text"]]   # (7)
            if hn:
                hardneg[split].setdefault(qid, [])
                for p in hn:
                    if p not in hardneg[split][qid]:
                        hardneg[split][qid].append(p)

    for split in pools:
        stats["kept"][split] = len(pools[split])
    return pools, qrels, hardneg, quarantine, fixlog, stats
```

**Giải thích:** Hàm lớn nhất, đọc theo các dấu đánh số:

- **(1) Vòng GermanQuAD.** Với mỗi câu hỏi, tính `qid` từ câu hỏi và `pid` từ context.
- **(2) Soi từng đáp án.** Mỗi đáp án qua `validate` rồi rẽ: `OK` giữ nguyên, `FIXED` dùng vị trí mới và **ghi vào `fixlog`** (đây là "mapping offset" đề yêu cầu), còn lại thì ghi vào `quarantine` kèm lý do và bỏ riêng đáp án đó (`continue`). Hàm `count` chỉ đếm số lượng từng trạng thái cho báo cáo.
- **(3) Khử đáp án trùng.** `seen` chặn cùng một cặp (chữ, vị trí) bị ghi hai lần. Test của GermanQuAD có nhiều đáp án mỗi câu nên hay gặp.
- **(4) Cách ly cả câu hỏi chỉ khi cần.** Chỉ khi *không còn đáp án hợp lệ nào* mới cách ly cả câu hỏi (`scope: "question"`). Còn nếu chỉ vài đáp án hỏng thì chỉ cách ly riêng chúng (`scope: "answer"`). Đúng như đề dặn với tập test.
- **(5) Vòng GermanDPR.** Chạy **sau** QuAD, vì khi trùng câu hỏi thì bản QuAD (offset đáng tin) được giữ làm bản chính. Câu nào không có passage `positive` thì cách ly (`NO_POSITIVE`).
- **(6) Tự tìm đáp án cho DPR.** DPR không có offset, nên gọi `validate(pos[0], t, None)` trên passage positive đầu tiên. Chỉ khi `LOCATED` (tìm thấy duy nhất) mới ghi `start`; còn lại `start = None`, nhưng **không cách ly**, vì câu hỏi vẫn dùng được cho retrieval. `pids` giữ *tất cả* passage positive để đưa vào qrels.
- **(7) Gom hard negative.** Lưu riêng vào `hardneg` (theo từng câu hỏi), không gắn nhãn "negative" lên passage trong corpus, vì negative của câu A có thể là đáp án đúng của câu B.

Cuối cùng ghi số câu hỏi giữ lại mỗi split và trả về cả sáu thứ.

---

# File 6: `src/data/split.py`

```python
import random


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)
```

**Giải thích:** Cấu trúc dữ liệu "gom nhóm". Mỗi phần tử trỏ tới một "cha"; những phần tử cùng nhóm sẽ lần theo cha tới cùng một "đại diện".

- `find(x)`: tìm đại diện của nhóm chứa `x`. `setdefault` tự thêm `x` thành nhóm riêng nếu lần đầu gặp. Dòng `parent[x] = parent[parent[x]]` rút ngắn đường đi để lần sau nhanh hơn.
- `union(a, b)`: nhập nhóm của `a` và `b` làm một.

Điểm mạnh là quan hệ gián tiếp tự được gom: nối A–B và B–C thì A, C tự cùng nhóm.

```python
def remove_conflicts_with_test(train_pool, train_qrels, test_pool, test_qrels):
    """R3: loại khỏi TRAIN mẫu trùng câu hỏi/passage với test. Không đụng test."""
    test_qids = set(test_pool)
    test_pids = {p for ps in test_qrels.values() for p in ps}
    kept, excluded = {}, []
    for qid, row in train_pool.items():
        if qid in test_qids:
            excluded.append({"question_id": qid, "reason": "QUESTION_IN_TEST"})
        elif train_qrels[qid] & test_pids:
            excluded.append({"question_id": qid, "reason": "PASSAGE_IN_TEST"})
        else:
            kept[qid] = row
    return kept, excluded
```

**Giải thích:** Thực thi quy tắc R3. Đầu tiên gom hai tập tra cứu từ test: các câu hỏi test (`test_qids`) và mọi passage liên quan đến test (`test_pids`). Rồi với **mỗi câu hỏi của train**: nếu chính câu hỏi đó có trong test thì loại (`QUESTION_IN_TEST`); nếu passage của nó giao với passage của test (`&` là giao tập hợp) cũng loại (`PASSAGE_IN_TEST`); ngược lại giữ. Hàm trả về cả danh sách giữ và danh sách loại kèm lý do. Tập test không hề bị sửa: hàm chỉ đọc nó.

```python
def make_groups(pool, qrels, pid_to_doc):
    """Nhóm liên thông: câu hỏi <-> passage <-> bài viết."""
    uf = UnionFind()
    for qid in pool:
        for pid in qrels[qid]:
            uf.union("q:" + qid, "p:" + pid)
            uf.union("p:" + pid, "d:" + pid_to_doc[pid])
    groups = {}
    for qid in pool:
        groups.setdefault(uf.find("q:" + qid), []).append(qid)
    return list(groups.values())
```

**Giải thích:** Dựng các **nhóm liên thông**. Với mỗi câu hỏi và mỗi passage liên quan, nối ba loại nút: câu hỏi (`q:`) ↔ passage (`p:`) ↔ bài viết (`d:`). Các tiền tố `q:`, `p:`, `d:` để ba loại ID không đè lên nhau. Sau khi nối xong, gom các câu hỏi theo đại diện nhóm (`uf.find`). Kết quả là danh sách các nhóm, mỗi nhóm là tập câu hỏi có liên hệ với nhau. Hai câu hỏi cùng hỏi về một đoạn văn luôn nằm cùng nhóm, nên sẽ không bị tách sang hai split khác nhau.

```python
def split_train_dev(pool, qrels, pid_to_doc, seed, dev_ratio):
    groups = make_groups(pool, qrels, pid_to_doc)
    groups = [sorted(g) for g in groups]
    groups.sort(key=lambda g: g[0])             # thứ tự ổn định
    random.Random(seed).shuffle(groups)
    target = round(dev_ratio * len(pool))
    dev = []
    for g in groups:
        if len(dev) >= target:
            break
        if len(dev) + len(g) > target * 1.2:    # bỏ qua nhóm quá lớn
            continue
        dev.extend(g)
    dev_set = set(dev)
    train = [q for g in groups for q in g if q not in dev_set]
    sizes = sorted((len(g) for g in groups), reverse=True)
    info = {"n_groups": len(groups), "largest_groups": sizes[:5],
            "target_dev": target}
    return sorted(train), sorted(dev), info
```

**Giải thích:** Chia train/dev **theo cả nhóm**:

1. Sắp xếp bên trong mỗi nhóm rồi sắp xếp các nhóm với nhau. Bước này *quan trọng cho tái lập*: thứ tự duyệt dict có thể khác nhau giữa các lần chạy, nhưng sắp xếp trước thì xáo trộn theo cùng seed luôn ra cùng kết quả.
2. `random.Random(seed).shuffle(groups)` xáo thứ tự các nhóm bằng seed cố định (42).
3. `target` là số câu hỏi dev mong muốn (khoảng 10%).
4. Lấy dần từng nhóm vào dev cho tới khi đủ. Nhóm nào làm dev vượt quá 120% mục tiêu thì bỏ qua (`continue`), để một nhóm khổng lồ không làm lệch tỷ lệ. Vì lấy nguyên nhóm nên số thực tế hơi lệch mục tiêu (1 067 so với 1 044), đó là bình thường.
5. Phần còn lại là train. Hàm trả về danh sách ID train, dev (đã sắp xếp) và thông tin về nhóm để đưa vào báo cáo.

---

# File 7: `scripts/02_preprocess.py`

```python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk

from src.config import (DATASETS, DEV_RATIO, PROCESSED_DIR, RAW_DIR,
                        REPORT_DIR, SEED)
from src.data.build_corpus import build_corpus
from src.data.build_qa import build_qa
from src.data.io_utils import read_json, write_json, write_jsonl
from src.data.split import remove_conflicts_with_test, split_train_dev


def main(out_dir):
    out = Path(out_dir)

    raw = {name: load_from_disk(str(RAW_DIR / name)) for name in DATASETS}   # (1)

    corpus, corpus_stats = build_corpus(raw)                                 # (2)
    pid_to_doc = {r["passage_id"]: r["document_id"] for r in corpus}

    pools, qrels, hardneg, quarantine, fixlog, qa_stats = build_qa(raw)      # (3)

    # mọi passage_id trong qrels phải có trong corpus
    for split in qrels:                                                      # (4)
        for ps in qrels[split].values():
            assert ps <= set(pid_to_doc), "qrels trỏ tới passage không có trong corpus"

    train_pool, excluded = remove_conflicts_with_test(                       # (5)
        pools["train"], qrels["train"], pools["test"], qrels["test"])
    train_ids, dev_ids, split_info = split_train_dev(                        # (6)
        train_pool, qrels["train"], pid_to_doc, SEED, DEV_RATIO)

    qa = {
        "train": [train_pool[q] for q in train_ids],
        "dev": [train_pool[q] for q in dev_ids],
        "test": [pools["test"][q] for q in sorted(pools["test"])],
    }
    qr = {
        "train": {q: sorted(qrels["train"][q]) for q in train_ids},
        "dev": {q: sorted(qrels["train"][q]) for q in dev_ids},
        "test": {q: sorted(qrels["test"][q]) for q in sorted(pools["test"])},
    }

    write_jsonl(out / "corpus.jsonl", corpus)                                # (7)
    for s in qa:
        write_jsonl(out / f"qa_{s}.jsonl", qa[s])
        write_json(out / f"qrels_{s}.json", qr[s])
    write_json(out / "hardneg_train.json",
               {q: hardneg["train"][q] for q in train_ids if q in hardneg["train"]})
    write_jsonl(out / "quarantine.jsonl", quarantine)
    write_jsonl(out / "span_fix_log.jsonl", fixlog)
    write_jsonl(out / "excluded_conflict.jsonl", excluded)

    step1 = read_json(REPORT_DIR / "step1_manifest.json")                    # (8)
    write_json(out / "split_manifest.json", {
        "corpus_version": "v1",
        "raw_revisions": {n: d["revision_sha"] for n, d in step1["datasets"].items()},
        "seed": SEED, "ratio": f"{round((1 - DEV_RATIO) * 100)}/{round(DEV_RATIO * 100)}",
        "preprocessing_config": {
            "R1": "offset sai, đáp án xuất hiện đúng 1 lần -> sửa + ghi log",
            "R2": "đáp án nhiều lần/không thấy -> quarantine",
            "R3": "train trùng test -> loại khỏi train",
            "R4": "passage trùng văn bản -> gộp, giữ provenance",
            "R5": "dev 10% theo nhóm liên thông",
            "R6": "DPR positive + hard_negative vào corpus",
            "R7": f"seed={SEED}",
        },
        "split_ids": {"train": train_ids, "dev": dev_ids,
                      "test": [r["question_id"] for r in qa["test"]]},
    })

    ids = {s: {r["question_id"] for r in qa[s]} for s in qa}                 # (9)
    write_json(out / "data_quality_report.json", {
        "corpus": corpus_stats,
        "qa": qa_stats,
        "excluded_conflict": {
            "count": len(excluded),
            "by_reason": {r: sum(1 for e in excluded if e["reason"] == r)
                          for r in {e["reason"] for e in excluded}}},
        "split": dict(split_info, sizes={s: len(qa[s]) for s in qa}),
        "overlap": {"train&dev": len(ids["train"] & ids["dev"]),
                    "train&test": len(ids["train"] & ids["test"]),
                    "dev&test": len(ids["dev"] & ids["test"])},
        "quarantine_total": len(quarantine),
        "span_fixed": len(fixlog),
    })
    print("Đã ghi vào", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    main(ap.parse_args().out)
```

**Giải thích:** `main` không có logic riêng, chỉ điều phối các module theo đúng thứ tự:

- **(1)** Đọc hai dataset thô từ `data/raw` (bản đã lưu ở Giai đoạn 1, không cần mạng).
- **(2)** Dựng corpus, rồi tạo bảng tra `pid_to_doc` (passage → bài viết) cho bước chia tập.
- **(3)** Dựng QA, qrels, hard negative, danh sách cách ly và log sửa offset.
- **(4)** Kiểm tra cấu trúc: mọi passage mà qrels nhắc tới phải có trong corpus. Đây là chỗ dùng `assert` hợp lý, vì vi phạm là lỗi cấu trúc thật, khác với số liệu lệch ở Giai đoạn 1.
- **(5)** Loại khỏi train các mẫu trùng test (R3).
- **(6)** Chia phần train còn lại thành train/dev theo nhóm.
- **(7)** Ghi các file đầu ra. Test lấy nguyên từ `pools["test"]` đã sắp xếp, không qua bộ lọc nào.
- **(8)** Ghi `split_manifest.json`: phiên bản dữ liệu thô (lấy từ manifest Giai đoạn 1), seed, tỷ lệ, quy tắc R1–R7 và **danh sách ID từng split** để lần sau đọc lại chứ không chia lại.
- **(9)** Tính overlap giữa các split (phải bằng 0) rồi ghi báo cáo chất lượng.

Khối `if __name__ == "__main__":` cho phép chạy `python scripts/02_preprocess.py`, hoặc thêm `--out thư_mục` để ghi sang chỗ khác (test tái lập dùng cách này để không ghi đè dữ liệu thật).

---

# File 8: `tests/test_phase2_acceptance.py`

```python
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from src.config import PROCESSED_DIR
from src.data.io_utils import file_sha1, read_json, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "dev", "test")


@pytest.fixture(scope="module")
def corpus():
    return {r["passage_id"]: r for r in read_jsonl(PROCESSED_DIR / "corpus.jsonl")}


@pytest.fixture(scope="module")
def qa():
    return {s: read_jsonl(PROCESSED_DIR / f"qa_{s}.jsonl") for s in SPLITS}
```

**Giải thích:** Phần chuẩn bị chung. Hai `fixture` nạp corpus (thành dict tra theo `passage_id`) và QA của ba split **một lần** cho cả file test (`scope="module"`), để các test dưới đây dùng lại mà không phải đọc file nhiều lần.

```python
def test_spans_valid(corpus, qa):
    for rows in qa.values():
        for r in rows:
            text = corpus[r["passage_id"]]["original_text"]
            for a in r["answers"]:
                if a["start"] is not None:
                    assert text[a["start"]:a["start"] + len(a["text"])] == a["text"]
```

**Giải thích:** Với mọi đáp án có vị trí, cắt đúng đoạn từ `original_text` của passage rồi so với chữ đáp án. Bỏ qua đáp án `start=None` (DPR mơ hồ). Test này chứng minh việc không sửa văn bản gốc đã giữ đúng offset.

```python
def test_qrels_in_corpus(corpus):
    for s in SPLITS:
        for pids in read_json(PROCESSED_DIR / f"qrels_{s}.json").values():
            assert set(pids) <= set(corpus)
```

**Giải thích:** Mọi passage được qrels nhắc tới phải tồn tại trong corpus. Nếu không, bước đánh giá retrieval về sau sẽ tham chiếu tới passage "ma".

```python
def test_no_overlap(qa):
    ids = {s: {r["question_id"] for r in qa[s]} for s in SPLITS}
    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["test"]
    assert not ids["dev"] & ids["test"]
```

**Giải thích:** Hai split bất kỳ không được chung câu hỏi nào (`&` là giao tập hợp, `not` tập rỗng thì đạt). Đây là phép kiểm chống rò rỉ chính. Lưu ý nó kiểm theo câu hỏi, chưa kiểm theo nhóm; nếu muốn chặt hơn có thể thêm kiểm "không nhóm nào nằm ở hai split".

```python
def test_unique_ids(corpus, qa):
    for rows in qa.values():
        assert len({r["question_id"] for r in rows}) == len(rows)
```

**Giải thích:** Trong mỗi split, số ID duy nhất phải bằng số dòng, tức không có ID lặp. (Tham số `corpus` ở đây không dùng đến, bạn có thể bỏ.)

```python
def test_quarantine_has_reason():
    for r in read_jsonl(PROCESSED_DIR / "quarantine.jsonl"):
        assert r.get("reason")
```

**Giải thích:** Mọi mẫu bị cách ly phải ghi lý do. Hiện tại `quarantine.jsonl` rỗng nên test này đạt "suông", nhưng nó sẽ bắt lỗi khi dữ liệu mới có mẫu lỗi.

```python
def test_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "02_preprocess.py"),
                        "--out", tmp], check=True, cwd=ROOT)
        for name in ("corpus.jsonl", "qa_train.jsonl", "qa_dev.jsonl",
                     "qa_test.jsonl", "split_manifest.json"):
            assert file_sha1(Path(tmp) / name) == file_sha1(PROCESSED_DIR / name)
```

**Giải thích:** Chạy lại toàn bộ pipeline vào một thư mục tạm, rồi so dấu vân tay của các file chính với bản thật. Giống nhau nghĩa là kết quả tái lập được. `sys.executable` đảm bảo dùng đúng Python của venv đang chạy test; `check=True` làm test lỗi nếu script lỗi.

---

**Lưu ý về độ tin cậy:** pipeline (các file 1 đến 7) tao đã chạy thật trên dữ liệu của mày, và hai lần chạy cho hash giống hệt. Các phép kiểm tương đương bốn test đầu cũng đã chạy và đạt. Riêng **file pytest này chưa được chạy**, vì `pytest` chưa được cài trong venv của mày.

Mày muốn tao đi tiếp bằng một ví dụ chạy tay cho hàm nào khó hình dung nhất, hay chuyển sang việc đưa code vào project và cài `pytest` để chạy nghiệm thu?


---

## 🧑 USER #22

Hãy mô tả lại tổng quan quá trình từ đầu đến bước bây giờ đi, tao vẫn chưa hiểu tao đang làm gì


---

## 🤖 CLAUDE #23

# Mày đang làm gì: bức tranh tổng thể

## 1. Mục tiêu cuối cùng của project

Mày đang xây một **hệ thống hỏi đáp tiếng Đức kiểu "trích xuất"**. Người dùng gõ một câu hỏi, hệ thống tìm trong kho văn bản đoạn có chứa câu trả lời, rồi chỉ ra chính xác câu trả lời nằm ở đâu trong đoạn đó.

```
Câu hỏi ─► [Tìm đoạn văn liên quan] ─► [Đọc đoạn, chỉ ra đáp án] ─► Đáp án + bằng chứng
             (Retriever: BM25)           (Reader)
```

Tức là hệ thống có **hai bộ phận**:

- **Retriever** (người thủ thư): trong hàng chục nghìn đoạn văn, tìm vài đoạn có khả năng chứa đáp án.
- **Reader** (người đọc): đọc các đoạn đó và khoanh đúng cụm từ là câu trả lời.

Kế hoạch trong `plan/README.md` chia thành ba phiên bản:

| Phiên bản | Nội dung | Ví dụ |
|---|---|---|
| **V1** (mày đang ở đây) | Baseline: BM25 + reader, rồi làm thành ứng dụng web | Bản đơn giản nhất chạy được |
| **V2** | Cải tiến: thêm dense retrieval, hybrid, reranking | Làm bản V1 chính xác hơn |
| **V3** | Hướng nghiên cứu: thử một phương pháp mới | Để viết bài báo |

## 2. Vì sao phải làm "dữ liệu" trước

Hãy nghĩ như xây nhà. Retriever và reader là **tòa nhà**, còn dữ liệu là **cái móng**. Móng lệch thì tầng nào xây lên cũng nghiêng, và tệ hơn là mày **không biết nó nghiêng**, vì con số đo ra vẫn trông hợp lý.

Ví dụ một lỗi móng: nếu câu hỏi trong tập test cũng nằm trong tập train, mô hình đã "học thuộc" đáp án, điểm sẽ cao giả. Mày sẽ tưởng hệ thống tốt trong khi thật ra nó chưa giỏi. Vì thế hai giai đoạn đầu chỉ làm một việc: **chuẩn bị dữ liệu sạch và đáng tin**. Chưa có dòng code NLP nào cả.

## 3. Đã đi qua những gì

### Giai đoạn 1: Nhập kho dữ liệu

**Bài toán:** dữ liệu nằm trên HuggingFace. Phải tải về, và chắc chắn rằng nó đúng, đủ, đúng phiên bản.

Kho hàng gồm hai dataset tiếng Đức:

- **GermanQuAD**: mỗi mẫu gồm câu hỏi + đoạn văn + đáp án (kèm vị trí đáp án trong đoạn).
- **GermanDPR**: mỗi mẫu gồm câu hỏi + đoạn đúng (positive) + đoạn gây nhiễu (hard negative). Nó **không có vị trí đáp án**.

Bốn hàm của giai đoạn này, theo ví dụ "thủ kho nhận hàng":

| Hàm | Việc thủ kho làm |
|---|---|
| `get_revision` | Ghi số lô hàng (phiên bản dataset trên Hub) |
| `download_dataset` | Nhận đúng lô hàng đó và cất vào `data/raw/` |
| `count_check` | Đếm lại: số mẫu thực tế có khớp phiếu giao hàng không? |
| `write_manifest` | Lập phiếu nhập kho (`step1_manifest.json`) |

Quá trình làm thực tế của mày:

1. Mày dán hàm C, D và hỏi sao code sai. Tao chỉ ra các lỗi (ví dụ trừ với `None`, dict viết sai cú pháp).
2. Chạy script thì **im lặng, không in gì**. Nguyên nhân: các file lúc đó đều **rỗng (0 byte)**.
3. Mày điền code, nhưng còn nhiều lỗi (thiếu import, gõ `congig`, `from turtle import pd`, thiếu tham số `revision`...). Tao liệt kê từng dòng.
4. Mày sửa và **chạy thành công**. Kết quả có trên đĩa: `data/raw/` (cả hai dataset), `reports/step1_count_check.csv`, `reports/step1_manifest.json`.

Số liệu kiểm tra khớp hoàn toàn đề bài: GermanQuAD 11 518 train / 2 204 test, GermanDPR 9 275 train / 1 025 test.

### Giai đoạn 2: Tiền xử lý, chia tập, dựng kho passage

**Bài toán:** dữ liệu thô chưa dùng được ngay. Cần biến nó thành "bộ dữ liệu chuẩn" mà mọi giai đoạn sau (retriever, reader, đánh giá) cùng dùng. Cụ thể cần ba thứ:

- **Corpus**: kho tất cả đoạn văn, mỗi đoạn một ID cố định, không trùng lặp. Đây là "thư viện" mà BM25 sẽ tìm trong đó.
- **QA + qrels**: danh sách câu hỏi, và với mỗi câu hỏi thì đoạn nào là đoạn đúng (qrels là "đáp án chuẩn" cho phần tìm kiếm).
- **Chia tập train / dev / test**: train để học, dev để chọn cấu hình, test để chấm điểm cuối cùng và **không được nhìn trước**.

Các hàm trong giai đoạn này, theo thứ tự dữ liệu chảy:

```
validate_spans  →  kiểm tra đáp án có nằm đúng vị trí không
build_corpus    →  gom mọi đoạn văn, bỏ trùng, nhớ nguồn gốc
build_qa        →  gắn câu hỏi với đoạn đúng
split           →  chia theo NHÓM (không theo câu hỏi), loại mẫu rò rỉ sang test
02_preprocess   →  chạy tất cả theo thứ tự, ghi 13 file đầu ra
```

**Chia theo nhóm** là ý tưởng quan trọng nhất. Nếu hai câu hỏi cùng hỏi về một đoạn văn mà một câu rơi vào train, câu kia vào dev, thì mô hình đã thấy đoạn đó khi học và điểm dev cao giả. Nên các câu hỏi liên quan nhau được giữ chung một split.

Tao đã dựng và **chạy thử toàn bộ giai đoạn này** trên dữ liệu thật của mày, và kết quả như sau:

| Chỉ số | Giá trị |
|---|---|
| Passage trong corpus | 54 908 → **26 864** sau khi bỏ trùng |
| Số đáp án kiểm tra offset | 16 065, **sai: 0** |
| Câu hỏi train / dev / test | **9 368 / 1 067 / 3 228** |
| Trùng giữa các tập | **0** |
| Chạy hai lần | Ra 13 file **giống hệt** nhau |

## 4. Hiện giờ mày đang ở đâu

Đây là chỗ cần rõ ràng nhất, vì dễ nhầm giữa "đã thiết kế" và "đã nằm trong project của mày".

| Hạng mục | Trạng thái thật |
|---|---|
| Giai đoạn 1 | ✅ **Xong, nằm trong project** của mày, đã chạy |
| Code Giai đoạn 2 | ⚠️ Tao đã viết và chạy thử, nhưng nằm trong **thư mục tạm**, **chưa có trong `nlp_v1` của mày** |
| `docs/preprocessing_rules.md` | ❌ Chưa viết (đề yêu cầu viết *trước khi chạy* lọc) |
| `reports/eda.md` | ❌ Chưa có (đề giả định Giai đoạn 1 đã tạo nó) |
| `tests/test_phase2_acceptance.py` | ❌ Chưa có, và `pytest` chưa cài |

Theo bảng mốc của kế hoạch, mày đang hướng tới **V1-M1: "Dữ liệu, split, corpus, qrels và offsets hoàn tất"**. Mốc này đạt khi Giai đoạn 2 nằm trong project, chạy xong và qua nghiệm thu.

## 5. Phát hiện quan trọng đang chờ mày quyết định

Khi chạy thử, tao thấy **toàn bộ 1 025 câu hỏi của GermanDPR test đều nằm sẵn trong GermanQuAD train**. Nếu không xử lý, mô hình học trên QuAD train đã thấy các câu hỏi test của DPR. Code đã tự loại 1 025 câu này khỏi train.

Kéo theo một quyết định thuộc về mày: tập test hiện gồm **QuAD test + DPR test = 3 228 câu**. Nếu mày muốn so sánh với paper GermanQuAD thì có thể chỉ cần test của QuAD (2 204 câu). Cần chốt điều này và ghi vào `preprocessing_rules.md` **trước khi xem điểm số**, để không bị thiên vị.

## 6. Việc tiếp theo, theo thứ tự

1. Quyết định về tập test (mục 5).
2. Viết `docs/preprocessing_rules.md` (các quy tắc R1 đến R9).
3. Đưa code Giai đoạn 2 vào project (tao có thể làm giúp).
4. Chạy `02_preprocess.py`, cài `pytest`, chạy nghiệm thu.
5. Xong thì sang V1 tiếp: dựng BM25 (retriever) rồi reader.

## Tóm lại bằng một câu

Mày đang xây hệ thống hỏi đáp tiếng Đức; hiện tại mày đang ở giai đoạn **chuẩn bị dữ liệu** (xong bước tải và kiểm tra, đang bước làm sạch và chia tập), chứ chưa bắt đầu phần "thông minh" của hệ thống.

Phần nào ở trên mày vẫn thấy mơ hồ nhất? Hoặc nếu muốn đi tiếp, mày chọn quyết định ở mục 5 trước, rồi tao đưa code vào project giúp mày.


---

## 🧑 USER #24

Cái tao muốn là phân tích chi tiết từ tập dữ liệu -> cấu trúc dữ liệu -> tập dữ liệu nào sẽ được ứng dụng cho vấn đề nào của project -> từ đó tiến hành tiền xử lý dữ liệu (trong bước này sẽ thực hiện những bước tiền xử lý nào) -> xây dựng corpus -> xây dựng cho retrieval | xây dựng cho quad ...


---

## 🤖 CLAUDE #25

Để phân tích có căn cứ, tao sẽ đọc trực tiếp dữ liệu thật của mày trước, rồi mới kết luận.


---

## 🤖 CLAUDE #26

Tao vừa đọc trực tiếp dữ liệu thật của mày (hai script kiểm tra nằm trong thư mục tạm, chưa đụng vào project). Kết quả **làm thay đổi một vài điều tao nói trước đó**, nên tao nêu rõ ở mục 6.

# 1. Hai tập dữ liệu là gì

| | GermanQuAD | GermanDPR |
|---|---|---|
| Bản chất | Tập hỏi đáp trích xuất (kiểu SQuAD) | Tập huấn luyện cho tìm kiếm đoạn văn (kiểu DPR) |
| Nguồn | Wikipedia tiếng Đức | Wikipedia tiếng Đức |
| Quy mô | train 11 518 / test 2 204 | train 9 275 / test 1 025 |
| Mỗi mẫu có | câu hỏi, 1 đoạn văn, đáp án + vị trí | câu hỏi, đáp án (chữ), 1 đoạn đúng, 3 đoạn gây nhiễu |

**Phát hiện then chốt:** GermanDPR **không phải dữ liệu độc lập**. Đây là phiên bản "đóng gói lại" của GermanQuAD train:

- Cả 9 275 câu hỏi DPR train **và** 1 025 câu hỏi DPR test đều nằm trong GermanQuAD train.
- Mọi đoạn "đúng" (positive) của DPR đều là context của QuAD, nguyên văn (2 281/2 281 ở train, 192/192 ở test).
- Với DPR train, đáp án giống hệt QuAD ở 9 250/9 275 câu.

Vậy DPR không thêm câu hỏi mới hay đáp án mới. Thứ nó thêm là **các đoạn gây nhiễu (hard negative)**.

# 2. Cấu trúc dữ liệu

## GermanQuAD (4 cột, phẳng)

```
id        : số nguyên
question  : "Von welchem Gesetzt stammt das Amerikanische ab? "
context   : "Recht_der_Vereinigten_Staaten\n\n=== Amerikanisches Common Law ===\nObwohl die ..."
answers   : {text: ["britischen Common Laws"], answer_start: [146]}
```

| Đặc điểm | Train | Test |
|---|---|---|
| Số đáp án / câu hỏi | **1** (cả 11 518) | **3** (2 128 câu) hoặc 2 (76 câu) |
| Context duy nhất | 2 540 (≈ 4,5 câu hỏi / context) | 474 (≈ 4,6 câu / context) |
| Độ dài context (ký tự) | p50 1 294, **max 2 496** | p50 1 996, p90 5 788, **max 11 647** |
| Context > 2 500 ký tự | 0 | **884 / 2 204 (40%)** |

Hai dòng cuối là chỗ đáng lo: reader sẽ học trên đoạn ngắn nhưng bị chấm trên đoạn dài hơn nhiều.

## GermanDPR (5 cột, lồng nhau)

```
question            : str
answers             : [str]                       ← chỉ có CHỮ, không có vị trí
positive_ctxs       : {title:[..], text:[..], passage_id:[..]}   ← mỗi câu đúng 1 đoạn
negative_ctxs       : {...}                       ← LUÔN RỖNG (0 đoạn ở cả 10 300 mẫu)
hard_negative_ctxs  : {...}                       ← 3 đoạn / câu (9 267 trên 9 275 câu)
```

Hai điểm bẫy: `passage_id` **rỗng ở toàn bộ 10 300 mẫu** (nên không thể dùng làm khoá, đúng như thiết kế dùng hash nội dung), và các cột `*_ctxs` là **dict của các list song song**, không phải list của dict.

## Cấu trúc ngầm trong văn bản (quan trọng, nhưng không nằm ở cột nào)

Mọi context đều có dạng:

```
Tiêu_đề_bài_viết \n\n === Tên mục === \n nội dung ...
```

Dòng đầu là **tiêu đề bài viết Wikipedia** (khớp mẫu này ở 92% train, 95% test). Vì vậy GermanQuAD *có* thông tin bài viết, chỉ là nằm trong văn bản chứ không có cột `title`.

# 3. Tập nào phục vụ vấn đề nào của project

| Vấn đề trong project | Dữ liệu dùng | Lý do |
|---|---|---|
| **Huấn luyện reader** | GermanQuAD train (tách ra train/dev) | Có vị trí đáp án (nhãn span) |
| **Đánh giá reader (cuối cùng)** | GermanQuAD test | Có 3 đáp án tham chiếu, chưa ai nhìn thấy |
| **Kho tìm kiếm (corpus)** | Context của QuAD (train + test) **+** các đoạn DPR | Retriever phải tìm trong kho đủ lớn |
| **Đánh giá retriever (BM25)** | Câu hỏi QuAD + đoạn đúng làm đáp án chuẩn (qrels) | Đoạn đúng của mỗi câu hỏi đã biết |
| **Huấn luyện dense retriever (V2)** | DPR hard negative (+ positive) | Đây là thứ DPR có mà QuAD không có |
| **Làm test** | **Chỉ GermanQuAD test** (khuyến nghị) | Xem mục 6 |

Thông tin quan trọng về kho: corpus có 26 864 đoạn, nhưng chỉ ≈3 014 đoạn là context QuAD. **Khoảng 23 850 đoạn còn lại đến từ hard negative của DPR**; 98% hard negative là đoạn mới, không phải context QuAD. Chính chúng làm kho "đủ lớn và khó" để việc tìm kiếm có ý nghĩa.

# 4. Tiền xử lý: làm những gì, vì sao

Mỗi bước dưới đây có một **bằng chứng từ dữ liệu**, không phải làm theo thói quen.

| # | Bước | Bằng chứng | Cách xử lý |
|---|---|---|---|
| P1 | **Giữ nguyên `original_text`** | Offset đáp án chỉ đúng trên văn bản gốc | Không sửa; chuẩn hóa chỉ làm trên bản sao `retrieval_text` |
| P2 | Kiểm tra offset QuAD | Cần chắc nhãn đúng | Đã đo: 18 054/18 054 đáp án **đúng**, không cần sửa |
| P3 | Chuẩn hóa `retrieval_text` | **5 874 context** (≈43%) chứa ký tự lạ (`\xa0`, `\u200b`, `\ufeff`); 1 074 có khoảng trắng kép | NFC, gộp khoảng trắng, **thêm: xoá ký tự zero-width**, hạ chữ thường |
| P4 | Làm sạch câu hỏi | **2 004 câu hỏi** kết thúc bằng dấu cách thừa | `strip()` khi tạo truy vấn (câu hỏi không có offset nên sửa an toàn) |
| P5 | Khử trùng passage | Cùng một passage nằm ở nhiều nơi (54 908 → 26 864) | Gộp theo hash văn bản gốc, giữ nguồn gốc trong `source[]` |
| P6 | Khử câu hỏi trùng | QuAD train lặp 40 câu, test 1, DPR train 21 | Gộp theo `question_id` (hash câu hỏi chuẩn hóa) |
| P7 | Rút tiêu đề bài viết | Dòng đầu context là tiêu đề; có thể gom theo bài | Lấy dòng đầu làm `title` khi khớp mẫu; đổi `_` thành khoảng trắng |
| P8 | Loại mẫu rò rỉ sang test | DPR test ⊂ QuAD train | Loại khỏi train; **không bao giờ loại khỏi test** |
| P9 | Chia train/dev **theo nhóm** | ≈4,5 câu hỏi dùng chung một context | Nhóm theo context/bài viết, seed 42 |
| P10 | Bỏ `negative_ctxs` | Cột này luôn rỗng | Bỏ qua, ghi vào báo cáo |

# 5. Xây dựng các bộ dữ liệu

```
 GermanQuAD ──► context ─────────────┐
 (train+test)                         ├─► [Khử trùng + gán passage_id] ─► CORPUS (26 864 đoạn)
 GermanDPR ──► positive + hard_neg ──┘
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
   BỘ CHO RETRIEVAL                                                  BỘ CHO READER (QuAD)
   câu hỏi → đoạn đúng                                               câu hỏi + đoạn + vị trí đáp án
```

## 5.1 Corpus (`corpus.jsonl`)

Mỗi dòng là một đoạn văn duy nhất: `passage_id` (hash), `original_text` (cho reader), `retrieval_text` (cho BM25), `title`, `document_id`, `source[]` (đoạn này đến từ dataset/split/vai trò nào).

Đơn vị của kho là **nguyên cả context**, không cắt nhỏ, để qrels đơn giản (một câu hỏi ↔ một đoạn). Hệ quả: test có đoạn dài tới 11 647 ký tự. Việc có cắt nhỏ hay không là quyết định cho giai đoạn BM25, nên ghi vào quy tắc.

## 5.2 Bộ cho Retrieval

| File | Nội dung | Dùng cho |
|---|---|---|
| `qa_*.jsonl` (cột `question`) | Truy vấn | Đưa vào BM25 |
| `qrels_train/dev/test.json` | `{question_id: [passage_id đúng]}` | Tính Recall@k, MRR |
| `hardneg_train.json` | `{question_id: [passage_id gây nhiễu]}` | Chỉ huấn luyện dense (V2), **chỉ ở train** |

Hai lưu ý cần ghi vào báo cáo: passage không có trong qrels *chưa chắc* không liên quan; và hard negative của DPR được chọn cho **câu hỏi train**, nên quanh đoạn của câu hỏi test có ít "mồi nhiễu" hơn, khiến việc tìm kiếm trên test có thể dễ hơn thực tế.

## 5.3 Bộ cho Reader

Mỗi mẫu: `question`, `passage_id` (lấy `original_text` từ corpus), `answers[{text, start}]`. Điều kiện: chỉ dùng mẫu có `start` hợp lệ.

| Split | Đáp án / câu | Vai trò |
|---|---|---|
| train | 1 | Huấn luyện |
| dev | 1 (vì cắt từ train) | Chọn cấu hình |
| test | 3 | Chấm cuối (EM/F1 lấy max theo 3 đáp án) |

Hai hệ quả phải nhớ:
- **Điểm dev sẽ thấp hơn điểm test một cách hệ thống**, vì dev chỉ có 1 đáp án tham chiếu còn test có 3. Đừng kết luận "mô hình tốt lên ở test" từ chênh lệch này.
- Reader cần **cửa sổ trượt** (sliding window) khi suy luận, vì 40% đoạn test vượt độ dài ≈512 token mà lúc huấn luyện chưa từng thấy. Kích thước cửa sổ và bước trượt là tham số thí nghiệm, chọn trên dev.

# 6. Điều chỉnh so với code đã đưa trước

Dữ liệu cho thấy ba chỗ tao nói chưa đúng hoặc cần đổi:

1. **"GermanQuAD không có title nên không gom được theo bài" là sai.** Tiêu đề nằm ở dòng đầu context. Cần sửa `build_corpus` để rút tiêu đề ở đó (P7), nếu không việc chia theo nhóm yếu hơn mức có thể.
2. **Quyết định về tập test nay nên rõ hơn.** Tao đã hỏi mày nên dùng "QuAD test + DPR test" (3 228 câu) hay chỉ QuAD test. Vì DPR ⊂ QuAD train, tao khuyến nghị **test = chỉ GermanQuAD test (2 204 câu)**, và DPR chỉ đóng góp *đoạn văn + hard negative*, không đóng góp *câu hỏi*. Cách này còn làm code đơn giản hơn: bỏ được phần "tự tìm đáp án DPR" và 615 đáp án mơ hồ.
3. **`retrieval_text` cần thêm bước xoá ký tự zero-width** (P3).

Một giới hạn **không sửa được, chỉ báo cáo**: khoảng 223/233 bài viết của test cũng xuất hiện trong train (qua 461/474 đoạn test; con số xấp xỉ vì lấy theo dòng đầu context). Test và train khác *đoạn* nhưng cùng *bài viết*, và đây là cách chia gốc của tác giả dataset.

# 7. Việc cần mày quyết định

1. **Test = chỉ QuAD test** (khuyến nghị) hay giữ cả DPR test?
2. **Kho có cắt nhỏ đoạn dài không?** V1 tối thiểu là giữ nguyên.
3. Mày muốn tao **sửa code Giai đoạn 2 theo mục 6 rồi đưa vào project**, hay mày muốn đọc/chỉnh `preprocessing_rules.md` trước?


---

## 🧑 USER #27

Các tiến trình hoàn chỉnh cho xử lý project này là gì


---

## 🤖 CLAUDE #28

Để liệt kê đúng các tiến trình hoàn chỉnh, tao đọc kế hoạch gốc của mày trong `plan/`.


---

## 🤖 CLAUDE #29

# Toàn bộ tiến trình của project

Theo `plan/` của mày, project có **29 giai đoạn (0 đến 28)**, chia thành 3 phiên bản nối tiếp nhau. Mỗi phiên bản làm trên nền của phiên bản trước.

```
V1 (GĐ 0–10)          V2 (GĐ 11–18)              V3 (GĐ 19–28)
Hệ thống cơ bản   ─►  Làm hệ thống tốt hơn   ─►  Biến thành nghiên cứu
BM25 + Reader         + Dense, Hybrid, Rerank    + Router, thí nghiệm, bài báo
```

## V1: xây hệ thống cơ bản (GĐ 0–10)

Đích của V1: người dùng gõ câu hỏi trên web và nhận được đáp án kèm bằng chứng.

| GĐ | Tên | Làm gì | Đầu ra chính |
|---|---|---|---|
| 0 | Dựng khung | Chốt hợp đồng dữ liệu (request/response), cấu trúc thư mục, seed, cấu hình | Khung project, schema |
| 1 | Thu thập + phân tích dữ liệu | Tải hai dataset, ghi phiên bản, EDA, đối chiếu số liệu | Raw data, manifest, báo cáo EDA |
| 2 | Tiền xử lý + chia tập + corpus | Làm sạch, kiểm offset, khử trùng, chia theo nhóm, dựng corpus và qrels | `corpus.jsonl`, `qa_*.jsonl`, `qrels_*.json` |
| 3 | BM25 + metrics tìm kiếm | Xây bộ tìm kiếm từ khóa, đo Hit/Recall/MRR@k | Index BM25, bảng đo trên dev |
| 4 | Reader + fine-tuning | Huấn luyện mô hình khoanh vùng đáp án (GBERT/GELECTRA), cửa sổ trượt | Checkpoint, EM/F1 khi có sẵn đoạn đúng |
| 5 | Ghép pipeline | BM25 top-20 → lấy top-5 → reader → chọn đáp án cuối | Pipeline chạy từ đầu đến cuối |
| 6 | Dense (tùy chọn) | Tìm kiếm theo ngữ nghĩa bằng embedding + FAISS | Có thể chuyển sang V2 |
| 7 | Đánh giá + phân tích lỗi | Đo ba tầng (retrieval, reader, end-to-end); phân loại lỗi sai | Bảng lỗi, query-level predictions |
| 8 | Backend FastAPI | API `/qa`, `/retrieve`, `/health` | API chạy được |
| 9 | Giao diện React | Ô nhập, hiển thị đáp án, tô sáng vị trí đáp án | Web UI |
| 10 | Docker + triển khai | Đóng gói, chạy trên máy chủ, đo tài nguyên | Website hoạt động |

Thứ tự phụ thuộc: `0 → 1 → 2 → [3 song song 4] → 5 → 7 → 8 → 9 → 10`.

## V2: cải tiến chất lượng (GĐ 11–18)

Chỉ bắt đầu khi V1 chạy được và baseline đã được lưu. Ý tưởng chính là thêm tìm kiếm ngữ nghĩa và xếp hạng lại, rồi **đo xem có thật sự tốt hơn không**.

| GĐ | Tên | Làm gì |
|---|---|---|
| 11 | Đóng băng baseline | Lưu mọi thứ của V1 làm mốc đối chiếu (B0 = BM25 → reader) |
| 12 | Dense retriever | Bi-encoder + FAISS (B1 = Dense → reader) |
| 13 | Hybrid bằng RRF | Trộn thứ hạng BM25 và dense (H0) |
| 14 | Cross-encoder reranking | Chấm lại top-20, lấy top-5 (H1) |
| 15 | So sánh đóng góp | Ma trận B0/B1/H0/H1, đo ba tầng và độ trễ. **Đây là mốc hoàn thành V2** |
| 16 | Hard-negative mining | Huấn luyện dense phân biệt đoạn dễ nhầm (bổ sung) |
| 17 | Phân tích nhóm + robustness | Yếu ở nhóm nào; chịu lỗi gõ/diễn đạt lại ra sao (bổ sung) |
| 18 | Tích hợp ứng dụng | Đưa cấu hình tốt nhất vào API/UI, triển khai lại |

Quy tắc xuyên suốt: mỗi so sánh chỉ đổi **một** thành phần (giữ nguyên corpus, reader, evaluator), nếu không sẽ không biết cải thiện đến từ đâu.

## V3: nghiên cứu khoa học (GĐ 19–28)

Hướng ứng viên là một "router" chọn nên rerank bao nhiêu đoạn cho từng câu hỏi để cân bằng chất lượng và tốc độ. Kế hoạch ghi rõ đây là **phương án cần kiểm chứng, chưa phải đóng góp mới đã xác nhận**.

| GĐ | Tên | Làm gì |
|---|---|---|
| 19 | Pilot | Xác định vấn đề có thật bằng thực nghiệm nhỏ |
| 20 | Literature + gap | Rà soát tài liệu, xác nhận khoảng trống |
| 21 | Chốt RQ, giả thuyết, protocol | Quyết định trước khi chạm test |
| 22 | Dữ liệu nghiên cứu + chống leakage | Dữ liệu cho router, dữ liệu chuyển miền |
| 23 | Thiết kế + cài đặt phương pháp | Đặc trưng, hành động, supervision, thuật toán |
| 24 | Baseline mạnh và công bằng | Các hệ thống để so sánh |
| 25 | Thí nghiệm chính | Ma trận thí nghiệm, độ bất định, đo hiệu năng |
| 26 | Ablation + phân tích | Bỏ từng phần để xem phần nào quan trọng; phân tích thất bại |
| 27 | Tái lập + công bố artifacts | Gói mã/dữ liệu để người khác chạy lại |
| 28 | Viết bài + chọn venue | Bản thảo, kiểm tra claims |

## Các mốc nghiệm thu

| Mốc | Điều kiện |
|---|---|
| **V1-M1** | Dữ liệu, split, corpus, qrels, offsets xong |
| V1-M2 | BM25 và reader chạy độc lập, metrics được kiểm tra |
| V1-M3 | Pipeline BM25 + reader có kết quả end-to-end + phân tích lỗi |
| V1-M4 | API, React, deploy hoạt động |
| V2-M1…M4 | Dense/hybrid → rerank → chốt cấu hình → ứng dụng cải tiến |
| V3-M1…M4 | Có gap cụ thể → chốt protocol → có thí nghiệm → gói tái lập + bản thảo |

## Mày đang ở đâu (đối chiếu thực tế)

| GĐ | Trạng thái thật |
|---|---|
| 0 | ⚠️ Một phần. Mày đã có `src/`, `scripts/`, `data/`, `reports/`, nhưng kế hoạch đề xuất `src/germanqa/{data,retrieval,reader,evaluation}` và có `configs/`, `notebooks/`, `tests/`... Chưa khớp hoàn toàn; cần quyết định có đổi cấu trúc hay không |
| 1 | ✅ Tải, manifest, đối chiếu số liệu đã chạy. ⚠️ **Báo cáo EDA chưa được viết thành file** (tao đã chạy phân tích ở lượt trước nhưng chưa nằm trong project) |
| 2 | ⚠️ Đã thiết kế và chạy thử đúng, nhưng code **chưa nằm trong project** của mày |
| 3 đến 28 | ❌ Chưa bắt đầu |

Mày đang hướng tới mốc **V1-M1** (xong khi Giai đoạn 2 hoàn tất).

## Đường đi ngắn nhất tới một sản phẩm chạy được

Nếu chỉ tính V1 tối thiểu (không có dense):

```
GĐ 2 (xong dữ liệu) → GĐ 3 BM25 ┐
                                 ├→ GĐ 5 ghép → GĐ 7 đánh giá → GĐ 8 API → GĐ 9 UI → GĐ 10 deploy
                     GĐ 4 Reader ┘
```

Việc cần làm **ngay bây giờ** vẫn là hoàn tất Giai đoạn 2. Cụ thể: chốt quyết định về tập test (khuyến nghị chỉ dùng QuAD test), sửa code theo ba điều chỉnh đã nêu, đưa vào project, rồi chạy nghiệm thu.

Mày muốn tao làm tiếp việc đó, hay muốn trao đổi trước về cấu trúc thư mục ở Giai đoạn 0 (vì nó ảnh hưởng đến chỗ đặt code của mọi giai đoạn sau)?


---

## 🧑 USER #30

Ví dụ: Tiến trình 1 - tiền xử lý dữ liệu (tao chỉ nhớ mỗi tiến trình này - hãy giải thích các tiến trình tiếp theo khi xây dựng project nlp)


---

## 🤖 CLAUDE #31

# Các tiến trình của một project NLP, giải thích theo project của mày

Tiến trình 1 (tiền xử lý) là bước **chuẩn bị nguyên liệu**. Các tiến trình sau là **chế biến, kiểm tra và phục vụ**. Mày cứ nhớ quy luật chung này:

```
Dữ liệu sạch → Xây các bộ phận → Ghép lại → Đo xem tốt không → Cải tiến → Đưa cho người dùng
   (TT1)          (TT2, TT3)       (TT4)        (TT5)            (TT6)        (TT7)
```

Ví dụ xuyên suốt: câu hỏi **"Wer gründete Rom?"** (Ai lập ra Rome?).

## Tiến trình 1: Tiền xử lý dữ liệu (mày đã biết)

Biến dữ liệu thô thành **kho đoạn văn sạch + danh sách câu hỏi có đáp án chuẩn + ba tập train/dev/test**.
Mọi tiến trình sau đều dùng lại kết quả này, nên sai ở đây thì sai tất cả.

## Tiến trình 2: Xây bộ tìm kiếm (Retriever, bản BM25)

**Câu hỏi nó trả lời:** trong 26 864 đoạn văn, đoạn nào *có khả năng* chứa đáp án?

Hình dung: mày hỏi thủ thư, thủ thư không đọc hết sách mà tra mục lục theo từ khóa. BM25 làm đúng việc đó. Nó chấm điểm từng đoạn dựa trên việc đoạn có chứa các từ của câu hỏi không, từ nào hiếm thì quan trọng hơn ("Rom" quý hơn "wer"), và đoạn quá dài thì bị trừ bớt.

- **Vào:** câu hỏi + kho đoạn văn.
- **Ra:** danh sách top-k đoạn xếp theo điểm, ví dụ top-20.
- **Dùng dữ liệu của TT1:** `retrieval_text` (bản chữ thường để tìm kiếm) và `qrels` (đoạn đúng của mỗi câu hỏi, để chấm).
- **Đo bằng:** *Recall@k* ("đoạn đúng có nằm trong top-k không?") và *MRR* ("đoạn đúng đứng thứ mấy?").
- **Lỗi hay gặp:** dùng test để chỉnh tham số, hoặc câu hỏi rỗng/đồng điểm không được xử lý.

## Tiến trình 3: Xây bộ đọc (Reader, fine-tune mô hình)

**Câu hỏi nó trả lời:** cho một câu hỏi và *một đoạn văn đã biết là đúng*, đáp án nằm ở đâu?

Đây là phần "học máy" thật sự. Mày lấy một mô hình ngôn ngữ đã biết tiếng Đức (GBERT hoặc GELECTRA) rồi dạy nó thêm bằng các mẫu (câu hỏi, đoạn, vị trí đáp án). Mô hình học cách chọn *ký tự bắt đầu* và *ký tự kết thúc* của đáp án.

- **Vào:** `qa_train.jsonl` (câu hỏi + đoạn + `answer_start`).
- **Ra:** một checkpoint mô hình, và với mỗi câu hỏi là vị trí đáp án trong đoạn.
- **Điểm khó:** mô hình chỉ đọc được tối đa khoảng 512 token một lần. Đoạn dài hơn phải **cắt thành các cửa sổ chồng lấn** (sliding window) rồi gộp kết quả. Dữ liệu của mày có đoạn test dài tới 11 647 ký tự, nên bước này bắt buộc.
- **Đo bằng:** *EM* (khớp đúng hoàn toàn) và *F1* (khớp một phần).
- **Dev dùng để:** chọn checkpoint tốt nhất. **Test chỉ chạm một lần, lúc cuối.**

## Tiến trình 4: Ghép thành pipeline

TT2 và TT3 hiện là hai bộ phận rời. Bây giờ nối lại:

```
"Wer gründete Rom?" → BM25 lấy top-20 → giữ top-5 → Reader đọc 5 đoạn → chọn đáp án tốt nhất
                                                                         → "Romulus" + đoạn nguồn + vị trí
```

Điểm quan trọng: **khi chạy thật không được dùng đoạn đúng có sẵn**. Reader phải tự dựa vào những gì BM25 tìm ra. Nếu BM25 tìm sai đoạn thì reader dù giỏi cũng vô ích.

## Tiến trình 5: Đánh giá và phân tích lỗi

**Câu hỏi nó trả lời:** hệ thống tốt đến đâu, và khi sai thì sai ở đâu?

Đo ở ba tầng riêng biệt, vì chỉ nhìn điểm cuối sẽ không biết lỗi từ đâu:

| Tầng | Hỏi | Chỉ số |
|---|---|---|
| Retrieval | BM25 có tìm được đoạn đúng không? | Recall@k, MRR |
| Reader | Cho đoạn đúng thì đọc ra đáp án đúng không? | EM, F1 |
| Toàn hệ thống | Hỏi vào, đáp án ra có đúng không? | EM, F1, thời gian |

Khi đáp án cuối sai, phân loại nguyên nhân:

- Đoạn đúng **không có trong top-20** → lỗi tìm kiếm.
- Có trong top-20 nhưng **rớt khỏi top-5** → lỗi chọn lọc.
- Có trong top-5 nhưng **đọc sai** → lỗi của reader.

Nhờ vậy lần cải tiến sau mày biết nên sửa bộ phận nào, thay vì đoán.

## Tiến trình 6: Cải tiến tìm kiếm

BM25 chỉ so khớp **từ**, nên bỏ lỡ câu diễn đạt khác nhưng cùng nghĩa ("gründete" và "Gründer"). Ba cải tiến:

1. **Dense retrieval:** biến câu hỏi và đoạn văn thành vector số, tìm theo "gần nghĩa". Dùng nhiều dữ liệu hard negative của DPR ở đây.
2. **Hybrid (RRF):** trộn thứ hạng của BM25 và dense, lấy ưu điểm cả hai.
3. **Reranking (cross-encoder):** một mô hình đọc kỹ từng cặp (câu hỏi, đoạn) để xếp lại top-20 chính xác hơn.

Luật vàng: mỗi lần **chỉ đổi một thứ** (giữ nguyên reader, kho, cách chấm), nếu không sẽ không biết điểm tăng nhờ đâu. Và không có gì đảm bảo cải tiến luôn tốt hơn, nên phải đo rồi mới kết luận.

## Tiến trình 7: Đóng gói thành ứng dụng

Đến đây mày có mô hình chạy trong notebook, nhưng người dùng cần một cái web:

- **Backend (FastAPI):** nhận câu hỏi qua mạng, chạy pipeline, trả đáp án + bằng chứng.
- **Frontend (React):** ô nhập câu hỏi, hiển thị đáp án, tô sáng chỗ đáp án trong đoạn nguồn.
- **Docker + triển khai:** đóng gói để chạy trên máy chủ, đo RAM và tốc độ thực tế.

Chi tiết dễ sai: vị trí tô sáng trên web tính theo UTF-16 còn Python tính theo ký tự, nên với tiếng Đức (ä, ö, ü, ß) phải chuyển đổi cẩn thận. Đây là lý do TT1 đã dặn **không bao giờ sửa văn bản gốc**.

## Tiến trình 8: Nghiên cứu (nếu muốn công bố)

Sau khi có hệ thống tốt, mày hỏi: "Có điều gì *mới* mà chưa ai làm không?" Hướng ứng viên trong kế hoạch là một "router" quyết định nên rerank mấy đoạn tùy độ khó của câu hỏi, để tiết kiệm thời gian mà vẫn chính xác. Quy trình: chạy thử nhỏ, đọc tài liệu xem đã có ai làm chưa, chốt giả thuyết, làm thí nghiệm có kiểm định, rồi viết bài. Kế hoạch ghi rõ đây là **ý tưởng cần kiểm chứng**, chưa phải đóng góp đã khẳng định.

## Bảng nhớ nhanh

| TT | Tên | Một câu |
|---|---|---|
| 1 | Tiền xử lý | Chuẩn bị dữ liệu sạch, chia tập đúng |
| 2 | Retriever (BM25) | Tìm đoạn văn liên quan |
| 3 | Reader | Khoanh vùng đáp án trong đoạn |
| 4 | Ghép pipeline | Nối tìm kiếm với đọc |
| 5 | Đánh giá | Đo ba tầng, tìm nguồn lỗi |
| 6 | Cải tiến | Dense, hybrid, rerank |
| 7 | Ứng dụng | API, web, triển khai |
| 8 | Nghiên cứu | Tìm điểm mới và công bố |

Mày đang ở TT1 (gần xong). Bước kế tiếp trực tiếp là TT2 và TT3, hai việc này độc lập nên làm song song được.

Mày muốn tao đi sâu vào tiến trình nào trước, theo cách giảng như ở TT1 (từng hàm, có code và giải thích bên dưới)?


---

## 🧑 USER #32

Đây là mày cho tiến trình của bài toán liên quan đến retrieval rồi, giả sử các bài toán phân loại cảm xúc ... thì sao, cái tao muốn là tiến trình tổng quan


---

## 🤖 CLAUDE #33

# Tiến trình tổng quan của một project NLP (áp dụng cho mọi bài toán)

Mày nói đúng, bảng tao đưa lúc nãy là riêng cho bài hỏi đáp. Phần bên dưới là **khung chung**. Phân loại cảm xúc, hỏi đáp, nhận diện thực thể (NER), dịch máy đều đi qua những bước này, chỉ khác ở nội dung từng bước.

## Bức tranh chung

```
 1. Định nghĩa     2. Dữ liệu      3. Khám phá      4. Tiền xử lý
    bài toán    →   thu thập   →    (EDA)      →    + chia tập
                                                        │
 8. Giám sát   ←  7. Triển khai ← 6. Đánh giá  ←  5. Biểu diễn + mô hình
                                       │                  ▲
                                       └── 6b. Cải tiến ──┘   (vòng lặp)
```

Điểm quan trọng nhất: đây **không phải đường thẳng mà là vòng lặp**. Đánh giá xong thường phải quay lại sửa dữ liệu hoặc mô hình.

## Từng tiến trình

### 1. Định nghĩa bài toán và tiêu chí thành công

**Trả lời:** mình đang giải quyết cái gì, và thế nào là "tốt"?

- Đầu vào là gì, đầu ra là gì (một nhãn? một đoạn văn bản? một chuỗi nhãn?).
- Chọn **chỉ số đo** *trước khi* huấn luyện, để không chọn số đẹp sau khi thấy kết quả.

| Bài toán | Vào → Ra | Chỉ số thường dùng |
|---|---|---|
| Phân loại cảm xúc | Câu → tích cực/tiêu cực | Accuracy, F1 |
| Hỏi đáp trích xuất | Câu hỏi + đoạn → đáp án | EM, F1 |
| Nhận diện thực thể | Câu → nhãn từng từ | F1 theo thực thể |
| Dịch máy | Câu nguồn → câu đích | BLEU, chrF |

### 2. Thu thập dữ liệu

**Trả lời:** dữ liệu lấy từ đâu, có đủ và đúng không?

- Dùng dataset có sẵn, tự thu thập, hoặc tự gán nhãn.
- Ghi lại **nguồn, phiên bản, ngày lấy, giấy phép**. Dataset trên mạng có thể đổi, nên không ghi thì không tái lập được.
- Đối chiếu số liệu thực tế với mô tả (đây chính là Giai đoạn 1 của mày).

### 3. Khám phá dữ liệu (EDA)

**Trả lời:** dữ liệu trông như thế nào, có gì bất thường?

- Cấu trúc các cột, kiểu dữ liệu, giá trị thiếu.
- Phân phối nhãn: cảm xúc có bị lệch 90% tích cực không? (Lệch thì accuracy cao vẫn vô nghĩa.)
- Phân phối độ dài văn bản, ngôn ngữ, ký tự lạ.
- Trùng lặp, nhãn sai, mẫu bất thường (đọc bằng mắt vài chục mẫu).

Đầu ra: một báo cáo ghi **vấn đề phát hiện được và quyết định xử lý**. Bước sau sẽ thi hành các quyết định đó.

### 4. Tiền xử lý và chia tập

**Trả lời:** làm sao để dữ liệu sạch, nhất quán, và đánh giá được một cách trung thực?

Gồm hai việc:

**Làm sạch:** chuẩn hóa mã hóa ký tự, xử lý khoảng trắng, loại trùng, cách ly mẫu hỏng. Chỉ làm những gì có bằng chứng từ EDA, không làm theo thói quen (ví dụ không phải lúc nào cũng nên xóa dấu câu hay stopword; với phân loại cảm xúc, dấu "!" và từ "không" mang nhiều thông tin).

**Chia tập:** đây là phần dễ sai nhất.

| Tập | Dùng để | Quy tắc |
|---|---|---|
| Train | Học | Thoải mái xem |
| Dev (validation) | Chọn cấu hình, tinh chỉnh | Xem nhiều lần được |
| Test | Chấm điểm cuối | **Chỉ chạm một lần** |

Phải chống **rò rỉ dữ liệu** (data leakage): cùng một mẫu, hoặc mẫu gần giống, nằm ở cả train và test thì điểm cao là giả. Ví dụ cùng một khách hàng viết 5 review mà bị chia vào cả hai tập. Cố định seed và lưu danh sách ID để không chia lại mỗi lần chạy.

### 5. Biểu diễn văn bản và xây mô hình

**Trả lời:** biến chữ thành số thế nào, và dùng mô hình gì?

Máy chỉ hiểu số, nên cần **biểu diễn** văn bản:

| Cách | Ý tưởng | Hợp khi |
|---|---|---|
| Bag-of-words / TF-IDF | Đếm từ | Baseline nhanh, dữ liệu nhỏ |
| Word embedding | Mỗi từ là một vector | Cần hiểu nghĩa gần nhau |
| Transformer (BERT...) | Vector phụ thuộc ngữ cảnh | Hiện là lựa chọn mạnh nhất cho đa số bài |

**Quy tắc vàng: làm baseline đơn giản trước** (ví dụ TF-IDF + Logistic Regression). Lý do: nó cho một con số mốc. Nếu mô hình Transformer to lớn chỉ hơn baseline 1%, mày biết cái giá phải trả có đáng không, và còn phát hiện được lỗi dữ liệu sớm.

Rồi mới huấn luyện mô hình mạnh hơn: chọn kiến trúc, hàm mất mát, tối ưu, theo dõi loss trên dev, **dừng đúng lúc** để tránh học thuộc lòng (overfitting).

### 6. Đánh giá và phân tích lỗi

**Trả lời:** tốt đến đâu, và sai ở đâu?

- Tính chỉ số đã chọn ở bước 1 trên dev (chọn cấu hình) rồi trên test (kết quả cuối).
- **Đừng chỉ nhìn một con số.** Xem ma trận nhầm lẫn, hiệu năng theo từng nhóm (câu ngắn/dài, từng chủ đề), và đọc kỹ các mẫu sai.
- Phân loại nguyên nhân lỗi: dữ liệu sai nhãn? thiếu ví dụ loại đó? mô hình yếu? Mỗi nguyên nhân có cách chữa khác nhau.
- So sánh công bằng: đổi **một** yếu tố mỗi lần, để biết điểm tăng nhờ đâu.

### 6b. Cải tiến (quay lại vòng lặp)

Dựa vào phân tích lỗi để chọn đòn bẩy phù hợp, theo thứ tự thường hiệu quả nhất:

1. Sửa/bổ sung **dữ liệu** (thường có lợi nhất).
2. Đổi **biểu diễn/mô hình**.
3. Điều chỉnh **siêu tham số**.
4. Thêm kỹ thuật (tăng cường dữ liệu, ensemble...).

Sau mỗi thay đổi lại đánh giá trên dev. Chỉ khi chốt xong cấu hình mới chạy test.

### 7. Triển khai

**Trả lời:** làm sao người khác dùng được mô hình?

- Lưu mô hình + bộ tokenizer + phiên bản tiền xử lý **đi cùng nhau** (mô hình huấn luyện với tiền xử lý A mà lúc dùng thật lại xử lý kiểu B sẽ cho kết quả sai).
- Đóng gói thành API hoặc ứng dụng.
- Đo **tài nguyên và tốc độ thực tế** (RAM, độ trễ), không suy đoán.
- Kiểm tra đầu vào bất thường: câu rỗng, quá dài, ngôn ngữ khác.

### 8. Giám sát và bảo trì

**Trả lời:** mô hình còn tốt theo thời gian không?

Ngôn ngữ và người dùng thay đổi (từ lóng mới, chủ đề mới), điểm số ngoài đời dần tụt (data drift). Cần ghi nhật ký dự đoán, lấy mẫu kiểm tra định kỳ, và huấn luyện lại khi cần.

## Những nguyên tắc chạy xuyên suốt mọi bước

| Nguyên tắc | Ý nghĩa |
|---|---|
| **Tái lập được** | Lưu seed, phiên bản dữ liệu/thư viện/cấu hình để chạy lại ra cùng kết quả |
| **Không nhìn test** | Test chỉ để chấm cuối; chọn cấu hình dựa trên dev |
| **Baseline trước** | Luôn có mốc đơn giản để so |
| **Dữ liệu trước mô hình** | Dữ liệu tệ thì mô hình xịn cũng vô ích |
| **Ghi lại quyết định** | Viết ra *vì sao* làm, để sau này giải thích được |

## Đối chiếu với project của mày

| Khung chung | Trong project hỏi đáp tiếng Đức |
|---|---|
| 1. Định nghĩa bài toán | Hỏi đáp trích xuất; chỉ số EM/F1 |
| 2. Thu thập | Giai đoạn 1: tải GermanQuAD + GermanDPR |
| 3. EDA | Phân tích cấu trúc, độ dài, trùng lặp (tao đã chạy) |
| 4. Tiền xử lý + chia tập | Giai đoạn 2, **tức "Tiến trình 1" mà mày nhớ** |
| 5. Mô hình | BM25 + reader (có thêm bước tìm kiếm vì bài hỏi đáp cần tìm đoạn trước) |
| 6. Đánh giá + cải tiến | Đo ba tầng, rồi thêm dense/hybrid/rerank |
| 7–8. Triển khai, bảo trì | API, web, Docker |

Hiểu như vậy thì "Tiến trình 1" của mày thật ra gồm các bước **2, 3, 4** trong khung chung, tức toàn bộ phần chuẩn bị dữ liệu. Bài hỏi đáp chỉ có thêm bước retrieval vì nó có hai bộ phận (tìm + đọc) thay vì một mô hình duy nhất như phân loại cảm xúc.


---

## 🧑 USER #34

Vậy chúng ta đã đến bước cài đặt nào rồi


---

## 🤖 CLAUDE #35

# Mày đang ở đâu

Tao vừa đọc lại toàn bộ project. Mày đang ở **Giai đoạn 2, bước viết các module tiền xử lý**: đã gõ được 6 trên 10 hạng mục, nhưng **chưa có module nào chạy được trọn vẹn**.

| # | Hạng mục (theo thứ tự đã đưa) | Trạng thái |
|---|---|---|
| 1 | `docs/preprocessing_rules.md` | ✅ Có (R1–R9). ⚠️ Chưa ghi quyết định về tập test |
| 2 | Thêm hằng số vào `config.py` | ❌ Chưa (thiếu `PROCESSED_DIR`, `SEED`, `DEV_RATIO`) |
| 3 | `io_utils.py` | ⚠️ Import được, nhưng tên hàm lệch |
| 4 | `text_utils.py` | ❌ Lỗi ngay khi import |
| 5 | `validate_spans.py` | ⚠️ Import được, nhưng logic sai |
| 6 | `build_corpus.py` | ❌ Không import được (do #4) và lỗi cấu trúc |
| 7 | `build_qa.py` | ❌ Không import được và lỗi cấu trúc |
| 8 | `split.py` | ⚠️ Mới có 1 trên 4 phần |
| 9 | `scripts/02_preprocess.py` | ❌ File rỗng (0 byte) |
| 10 | `tests/` | ❌ Chưa có |

Giai đoạn 1 vẫn xong. `reports/eda.md` vẫn chưa có.

## Lỗi cần sửa, theo từng file

Tao đã chạy import từng module và kiểm tra `UnionFind` (hai kết quả được ghi "đã chạy"). Các lỗi còn lại tao đọc ra từ code, chưa chạy vì module chưa import được.

### `text_utils.py`
| Dòng | Lỗi | Sửa |
|---|---|---|
| 6 | `re.complie` (đã chạy, báo lỗi) | `re.compile` |
| 27, 31 | Gọi `_normalize`, nhưng mày đã đổi tên hàm thành `normalize_text` → `NameError` | Dùng `normalize_text` |
| 27 | `"q_ "` có dấu cách, không `.lower()`, không cắt `[:16]` → ID thành `q_ <40 ký tự>` | `"q_" + sha1_hex(normalize_text(question).lower())[:16]` |
| 31 | Khi `title` là `None` (mọi context QuAD) thì `normalize_text(None)` sẽ lỗi | `t = normalize_text(title).lower() if title else ""` |

### `validate_spans.py`
| Dòng | Lỗi | Sửa |
|---|---|---|
| 18 | `if len(pos) == 0:` rồi `pos[0]` → khi không tìm thấy thì `IndexError`, và trường hợp "đúng 1 lần" không bao giờ vào | `== 1` |

### `io_utils.py`
| Dòng | Lỗi | Sửa |
|---|---|---|
| 21, 28 | Tên `write_json1` / `read_json1` (số **1**), trong khi các module khác gọi `write_jsonl` / `read_jsonl` (chữ **l**) | Đổi thành `jsonl` |
| 11 | `write_json(data, path)` đảo thứ tự tham số so với `write_jsonl(path, rows)` | Thống nhất `(path, obj)` cho cả hai, tránh gọi nhầm |

### `build_corpus.py`
| Dòng | Lỗi | Sửa |
|---|---|---|
| 35–53 | **Thụt lề sai**: vòng DPR, `records`, `return` nằm *bên trong* vòng QuAD. Hàm trả về ngay sau split `train` của QuAD, nên QuAD `test` không bao giờ được xử lý | Dedent từ dòng 35 xuống ngang `for split...` ở dòng 28 |
| 33 | `row[id]` (biến built-in `id`) | `row["id"]` |
| 42 | `_add(corpus, row["context"], None, ...)` trong vòng DPR: DPR không có `context` | `_add(corpus, text, title, ...)` |
| 44 | `"role": "role"` (chuỗi cố định) | `"role": role` |

### `build_qa.py` (nhiều lỗi nhất)
| Dòng | Lỗi | Sửa |
|---|---|---|
| 13 | `prev["answser"]` | `"answers"` |
| 49 | Khoá `"duplicate_other_passage_to_qrels_only:"` có dấu `:` lọt vào trong chuỗi → `KeyError` | Bỏ dấu `:` |
| 53–54 | `def count` thụt lề ở cột 0 nên nằm **ngoài** `build_qa`; toàn bộ phần sau bị nuốt vào thân `count`, còn `build_qa` trả về `None`. Dòng 54 còn viết `status["span_status"]` | Thụt vào trong `build_qa`; dòng 54 dùng `stats[...]` |
| 57 | `.itimes()` | `.items()` |
| 69 | `"question_id:"` | `"question_id"` |
| 80 | `{"text: a[0]", "start[1]"}` là một **set hai chuỗi**, không phải dict | `{"text": a[0], "start": a[1]}` |
| 83, 98 | `"score"` | `"scope"` |
| 88, 112 | Khoá `"answer"` | `"answers"` (đúng schema) |
| 105 | `count("DPR_ + status")` truyền một chuỗi cố định | `count("DPR_" + status)` |
| 121–123 | `for split in pools` và `return` thụt quá sâu, nằm trong vòng DPR → trả về sau split `train` của DPR, `test` bị bỏ | Dedent về ngang `for split, ds in ...` ở dòng 91 |

### `split.py`
| Dòng | Lỗi | Sửa |
|---|---|---|
| 12 | `return x` nằm **trong** vòng `while`. Tao đã chạy thử: `find('a')` và `find('c')` đều trả `None` (đáng lẽ bằng nhau) | Dedent `return x` ra ngang `while` |
| — | Thiếu 3 hàm: `remove_conflicts_with_test`, `make_groups`, `split_train_dev` | Viết tiếp |

## Điều cần nói thẳng

Hầu hết lỗi trên thuộc **cùng một loại**: thụt lề sai, và gõ sai chính tả tên biến/khoá. Python không báo lỗi lúc đọc file với kiểu sai thụt lề, nó chỉ làm sai lặng lẽ (ví dụ trả về sau nửa dữ liệu). Với nhóm lỗi này, **chạy rồi nhìn kết quả** mới phát hiện được, nên cần có bước kiểm tra số lượng ngay sau khi chạy.

## Gợi ý thứ tự làm tiếp

1. Sửa 5 file có lỗi theo bảng (bắt đầu từ `text_utils.py` vì các file khác phụ thuộc vào nó).
2. Thêm hằng số vào `config.py`, viết nốt `split.py`, rồi `02_preprocess.py`.
3. Chạy và **đối chiếu với các con số tham chiếu** tao đã đo: 26 864 passage sau gộp, 16 065 span kiểm tra không sai, overlap bằng 0. Lệch con số nào thì biết ngay còn bug ở đâu.
4. Trước khi chạy: **chốt quyết định tập test** và ghi vào `preprocessing_rules.md`, cùng ba điều chỉnh (tiêu đề từ dòng đầu context, xoá ký tự zero-width, test chỉ QuAD).

Mày muốn tao sửa thẳng các lỗi trên vào file cho mày, hay mày tự sửa theo bảng rồi gọi tao kiểm tra lại?


---

## 🧑 USER #36

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

<tools-used-before-summary>Bash, Glob, Grep, PowerShell, Read, Write</tools-used-before-summary>
Summary:
## 1. Primary Request and Intent

The user (a Vietnamese-speaking student, writes informally with "tao/mày") is building a **German extractive QA system** (GermanQuAD + GermanDPR; BM25 retriever + fine-tuned reader, later dense/hybrid/rerank, API/React/Docker, optional research V3). Project dir: `D:\Project\2026-2027-HK1\nlp_v1` (Windows, PowerShell/Git Bash, venv at `.\venv`, Python 3.13.3, `datasets` 5.0.1, `huggingface_hub` 1.31.0). Plans live in `plan/` (README, 01 V1 stages 0–10, 02 V2 stages 11–18, 03 V3 stages 19–28) – read in full for V1/V2.

Request history (all in Vietnamese; user wants a **teacher-style** explanation so the student understands what the code does and why):
1. Explained errors in pseudocode `count_check` (Hàm C); then asked its purpose.
2. Asked how to implement `write_manifest` (Hàm D) and its meaning in Step 1.
3. Pasted full "Bước 1" spec; asked for the phase's meaning, review of fixed (C, D) and unfixed (A, B) functions, in lecturer style.
4. Ran `python scripts/01_download_data.py` → silent output; asked why. Cause found: all files were 0 bytes (empty). Later asked "hướng dẫn chính xác lỗi ở đâu và sửa", then "vẫn chưa hiểu là thiếu cái gì" (re-inspection showed files now had content but many bugs: missing imports, `src.congig`, `from turtle import pd`, missing `revision` param, `["repo_id"]` vs `.repo_id`, wrong dict syntax in config, etc.). User fixed and Phase 1 ran successfully.
5. Pasted "Giai đoạn 2" spec (tiền xử lý, chia tập, corpus; steps 2.0–2.9) and asked for installation guidance.
6. Complained "quá nhiều code nhưng không giải thích" → asked for per-function explanation; then asked to re-present code with explanation **below each code block** (done, full 8-file walkthrough).
7. Asked for an overall description from the beginning to now ("tao vẫn chưa hiểu tao đang làm gì").
8. Asked for detailed analysis: dataset → data structure → which dataset serves which project problem → preprocessing steps → build corpus → build retrieval set | build QA/reader set. (Done with real data inspection.)
9. Asked "Các tiến trình hoàn chỉnh cho xử lý project này là gì" (done: V1/V2/V3 stage tables from plan files).
10. Said he only remembers "Tiến trình 1 - tiền xử lý dữ liệu" and asked to explain subsequent processes; then clarified he wants a **general NLP project lifecycle** (e.g., sentiment classification), not retrieval-specific (done: 8-stage generic framework + mapping to project).
11. **Latest message: "Vậy chúng ta đã đến bước cài đặt nào rồi"** — asking what implementation step they have reached. I was in the middle of checking project state when summary was requested.

No security-relevant constraints were stated by the user. (System guidance: don't act outward-facing/irreversible without confirming; user has never explicitly approved me writing files into the project — I only offered, and the user typed code themselves.)

## 2. Key Technical Concepts
- Extractive QA = Retriever (BM25 → later dense/hybrid RRF/cross-encoder) + Reader (GBERT/GELECTRA span start/end, sliding windows); metrics Hit/Recall/MRR@k, EM/F1.
- Data hygiene: immutable raw data; `original_text` never modified (offsets), separate `retrieval_text`; content-hash IDs (`p_`+sha1[:16], `q_`+hash of normalized lowercase question[:16], `d_`+hash(title)[:12]); dedup with provenance (`source[]`); span validation statuses OK/FIXED/LOCATED/QUARANTINE_*; quarantine instead of assert; group split via union-find (question↔passage↔document), fixed seed 42, 90/10; conflict removal from train only (R3); test never filtered; reproducibility (sorted keys, hashes identical across runs).
- Generic NLP lifecycle: define task/metric → collect → EDA → preprocess+split → baseline → model → eval/error analysis → improve loop → deploy → monitor.

## 3. Files and Code Sections

**Phase 1 (done, working, in project):** `src/config.py` (DatasetSpec dataclass, DATASETS, ROOT/RAW_DIR/REPORT_DIR), `src/data/download.py` (get_revision, download_dataset(name, revision), count_check, write_manifest), `scripts/01_download_data.py`, `requirements.txt` (270 bytes), outputs `data/raw/{germanquad,germandpr}`, `reports/step1_count_check.csv`, `reports/step1_manifest.json` (GermanQuAD sha fff05cea…, GermanDPR sha 5129d024…; counts all OK: QuAD 11518/2204, DPR 9275/1025). Also user-created `scripts/export_samples.py` and `data/samples/*.json`.

**Verified reference implementation (in scratchpad, NOT in project):** `C:\Users\PC\AppData\Local\Temp\claude\D--Project-2026-2027-HK1-nlp-v1\8911bc9d-dc92-40e7-913e-28245915de94\scratchpad\proj\` with `src/config.py` (adds PROCESSED_DIR, SEED=42, DEV_RATIO=0.10), `src/data/{io_utils,text_utils,validate_spans,build_corpus,build_qa,split}.py`, `scripts/02_preprocess.py` (argparse `--out`). Ran on real data: corpus 54,908→26,864 passages (6,424 multi-provenance); 16,065 spans checked, 0 wrong; QuAD 18,054/18,054 answers OK; qa_train/dev/test = 9,368/1,067/3,228; overlaps 0; 1,042 removed from train (1,025 QUESTION_IN_TEST + 17 PASSAGE_IN_TEST); 567 groups (largest 212); two runs → all 13 output files hash-identical. pytest file (`tests/test_phase2_acceptance.py`) drafted in chat but never run (pytest not installed in venv).

**Data findings (from inspect scripts in scratchpad `inspect_data.py`, `inspect2.py`):**
- GermanQuAD cols: id, context, question, answers{text[],answer_start[]}. Train: 1 answer/q, 2,540 unique contexts (~4.5 q/ctx), context ≤2,496 chars; Test: 3 answers (2,128 q) or 2 (76 q), 474 contexts, context p50 1,996, p90 5,788, max 11,647; 884/2,204 (40%) > 2,500 chars → reader needs sliding windows; dev (from train) has only 1 reference so dev EM < test EM systematically.
- Contexts have form `Title\n\n=== Section ===\ntext`; first line = article title (matches ~92% train, 95% test) — so QuAD title can be extracted (my earlier claim "QuAD has no title" was wrong). ≈223/233 test articles also appear in train (approximate, by first line).
- GermanDPR: cols question, answers[str], positive_ctxs/negative_ctxs/hard_negative_ctxs as dict-of-parallel-lists {title,text,passage_id}; 1 positive/q; `negative_ctxs` always empty; 3 hard negatives/q; **passage_id empty for all**; answers single string.
- **DPR ⊂ QuAD train**: all 9,275 DPR-train and 1,025 DPR-test questions exist in QuAD train; all DPR positives are QuAD contexts verbatim; answers identical 9,250/9,275. DPR adds mainly hard negatives (~98% new passages; only 5% share title with positive); ~23,850 of 26,864 corpus passages come from DPR hard negatives. Hard negatives were mined for train questions → test retrieval may be easier (limitation).
- Noise: 5,874 contexts contain \xa0/\u200b/\ufeff; 1,074 double spaces; 2,004 questions end with whitespace; repeated questions: QuAD train 40, test 1, DPR train 21.
- My recommendations (pending user decision): test = **QuAD test only (2,204)**; DPR contributes passages + hard negatives, not QA rows; extract QuAD title from first line for document_id; strip zero-width chars in retrieval_text; strip questions; keep whole contexts as corpus units in V1.

**Current state of user's project files (read just before summary; user typing Phase 2 by hand):**
- `docs/preprocessing_rules.md` (1,126 B): R1–R9 copied from my text; no decision recorded yet about test set / title extraction.
- `src/config.py`: still lacks PROCESSED_DIR, SEED, DEV_RATIO.
- `src/data/io_utils.py`: `write_json(data, path)` has reversed arg order vs my version `(path, obj)`; functions named `write_json1`/`read_json1` (should be `write_jsonl`/`read_jsonl`).
- `src/data/text_utils.py`: `re.complie` typo (import-time AttributeError, confirmed by running); `_normalize` used but function is named `normalize_text` (NameError); `make_question_id` returns `"q_ " + sha1_hex(...)` (space, no lower(), no `[:16]`); `make_document_id` calls normalize on possibly-None title (crash) and no lower(); comments still say QuAD has no title.
- `src/data/validate_spans.py`: bug `if len(pos) == 0:` returns `pos[0]` – must be `== 1` (IndexError otherwise).
- `src/data/build_corpus.py`: DPR loop, `records/stats/return` wrongly indented inside the QuAD split loop (returns after first split); `str(row[id])` should be `row["id"]`; DPR `_add(corpus, row["context"], None, …)` should be `_add(corpus, text, title, …)`; `"role": "role"` should be `role`; stats key names `passage_before_dedup` vs earlier `passages_before_dedup`.
- `src/data/build_qa.py`: `prev["answser"]` typo; stats key `"duplicate_other_passage_to_qrels_only:"` has stray colon; `def count` is at module level (col 0), swallowing the loops into it, uses `status["span_status"]` instead of `stats[...]`; `.itimes()`; `"question_id:"` key typo; `good.append({"text: a[0]", "start[1]"})` (is a set of strings); `"score"` instead of `"scope"` (two places); `"answer"` instead of `"answers"` (QuAD and DPR rows); `count("DPR_ + status")`; final `for split in pools … return` indented inside the DPR loop (returns after first DPR split).
- `src/data/split.py`: only `UnionFind` present (384 B); `return x` in `find` is wrongly indented inside the while loop (verified: after union a-b-c, find('a') and find('c') both return None); missing `remove_conflicts_with_test`, `make_groups`, `split_train_dev`.
- `scripts/02_preprocess.py`: **0 bytes (empty)**. `tests/` and `reports/eda.md` do not exist; pytest not installed.
- Import check results: io_utils OK, validate_spans OK (but logic bug), split OK (but logic bug); text_utils, build_corpus, build_qa fail at import due to `re.complie`.

## 4. Errors and Fixes
- Silent script run: files empty (0 bytes) → told user to save/paste content.
- Earlier user-code bugs (Phase 1) enumerated by line and fixed by user.
- My own failed Bash heredoc batch (parse error with `<<'EOF'`) → recreated files with the Write tool instead.
- My earlier statements corrected: (a) "GermanQuAD has no title" → title is in first line of context; (b) test composition (QuAD+DPR 3,228) now recommended to change to QuAD-test-only; (c) `write_manifest` should receive SHA from outside and take `repo_id` from `DATASETS`.
- No user pushback beyond: "quá nhiều code nhưng không giải thích" (fixed by per-function explanations, then code-with-explanation-below format).

## 5. Problem Solving
Phase 1 solved and running. Phase 2 design fully verified in scratchpad; user is re-implementing manually and has numerous bugs (listed above). Open design decisions awaiting user: test set composition, chunking of long passages, whether to restructure folders per plan (`src/germanqa/...`) vs current `src/data`, `scripts/`.

## 6. All User Messages
1. Pasted "Hàm C: count_check(name, ds)" spec + "tao đang code hàm này cho project, giải thích lỗi sa i"
2. "Hàm này chức năng để làm gì"
3. Pasted "Hàm D: write_manifest(...)" + "hãy hướng dẫn cho tao thực hiện cài đặt hàm này và giải thích ý nghĩa của hàm trong step của project"
4. Pasted full "Bước 1 chi tiết" (1.0–1.3, funcs A–D) + "giải thích ý nghĩa của giai đoạn, kiểm tra lại các hàm mà ta đã sửa và kiểm tra các hàm chưa chỉnh sửa hãy giải thích và chỉnh sửa lại cho đúng ĐẶC BIỆT: khi giải thích, bạn phải là giảng viên đang giải thích cho sinh viên hiểu vấn đề code đang làm gì phục vụ giai đoạn, phải làm sao cho sinh viên hiểu được code và cấu trúc sẽ được xây dựng"
5. "(venv) PS …> python scripts/01_download_data.py (venv) PS …> kiểm tra xem tại sao nó không hoạt động"
6. "Hãy hướng dẫn chính xác lỗi ở đâu và sửa như thế nào"
7. "Tao vẫn chưa hiểu là thiếu cái gì"
8. Pasted "Giai đoạn 2: Tiền xử lý, chia tập và corpus" (2.0–2.9) + "hướng dẫn cho tao cài đặt"
9. "Mày hãy trình bày lại và giải thích từng hàm đang cài đặt sẽ làm nhiệm vụ gì, mày đưa tao quá nhiều code nhưng không giải thích"
10. "Mày trình bày lại code và giải thích phía dưới code"
11. "Hãy mô tả lại tổng quan quá trình từ đầu đến bước bây giờ đi, tao vẫn chưa hiểu tao đang làm gì"
12. "Cái tao muốn là phân tích chi tiết từ tập dữ liệu -> cấu trúc dữ liệu -> tập dữ liệu nào sẽ được ứng dụng cho vấn đề nào của project -> từ đó tiến hành tiền xử lý dữ liệu (trong bước này sẽ thực hiện những bước tiền xử lý nào) -> xây dựng corpus -> xây dựng cho retrieval | xây dựng cho quad ..."
13. "Các tiến trình hoàn chỉnh cho xử lý project này là gì"
14. "Ví dụ: Tiến trình 1 - tiền xử lý dữ liệu (tao chỉ nhớ mỗi tiến trình này - hãy giải thích các tiến trình tiếp theo khi xây dựng project nlp)"
15. "Đây là mày cho tiến trình của bài toán liên quan đến retrieval rồi, giả sử các bài toán phân loại cảm xúc ... thì sao, cái tao muốn là tiến trình tổng quan"
16. "Vậy chúng ta đã đến bước cài đặt nào rồi"

## 7. Pending Tasks
- Answer the latest question: report exactly which implementation step the user has reached (Giai đoạn 2, mid-way through step "2.1/2.3–2.7 writing modules"; mapping to my 10-step order: rules doc ✔ partial, config ✘, io_utils ✔ with bugs, text_utils ✔ with bugs, validate_spans ✔ with bug, build_corpus ✔ with bugs, build_qa ✔ with bugs, split ~25% (UnionFind only, with bug), scripts/02_preprocess.py ✘ empty, tests ✘, eda.md ✘), with the concrete bug list per file/line and recommended fix order.
- Open decisions to put to the user: test = QuAD-test-only?; extract QuAD title from first line; zero-width char removal; chunking; folder structure per plan; record decisions in `docs/preprocessing_rules.md` (and create `reports/eda.md`).
- Offer (not yet approved) to apply fixes / copy verified files into project and install pytest.

## 8. Current Work
Immediately before the summary request I was answering "Vậy chúng ta đã đến bước cài đặt nào rồi" by inspecting the live project: listed files (found scripts/02_preprocess.py = 0 bytes, split.py = 384 B, etc.), read `docs/preprocessing_rules.md`, `src/config.py`, `split.py`, `build_qa.py`, `build_corpus.py`, `text_utils.py`, `io_utils.py`, `validate_spans.py`, and ran import checks (text_utils/build_corpus/build_qa fail on `re.complie`; UnionFind.find returns None). I had not yet written the final reply to the user.

## 9. Optional Next Step
Write the reply to the user's latest question ("Vậy chúng ta đã đến bước cài đặt nào rồi"): state they are in **Giai đoạn 2 (tiền xử lý/chia tập/corpus), midway through writing the modules**, present a status table per file/step, list the specific bugs found (file + line + fix, as in section 3) in priority order (text_utils `re.complie`/`_normalize`/`make_question_id`; validate_spans `== 0`→`== 1`; build_corpus indentation + `row[id]` + DPR `_add` args; build_qa structure/typos; split `find` return indentation + missing functions; io_utils names `write_jsonl/read_jsonl`; config additions; empty 02_preprocess.py; no tests/eda.md), and ask whether they want to fix by hand per the list or have me apply the verified code into the project; also remind of the pending test-set decision. Do not modify project files without the user's go-ahead.

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: C:\Users\PC\.claude\projects\D--Project-2026-2027-HK1-nlp-v1\8911bc9d-dc92-40e7-913e-28245915de94.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.

The messages after this summary are the most recent messages from before compaction, kept verbatim. The summary was written without seeing them, so something it says has not happened yet may already have happened in them.


---

## 🧑 USER #37

ủa sao mất lịch sử chat phía trên rồi


---

## 🤖 CLAUDE #38

Login expired · Please run /login


---

## 🤖 CLAUDE #39

Lịch sử không mất. Cuộc trò chuyện dài quá nên đã vượt giới hạn ngữ cảnh, và Claude Code **tự tóm tắt** phần cũ để tao làm tiếp được. Giao diện chỉ hiện phần sau đoạn tóm tắt, còn tao vẫn nắm nội dung chính:

- **Dự án:** hệ thống QA tiếng Đức với GermanQuAD và GermanDPR.
- **Đã xong:** Giai đoạn 1 (tải dữ liệu, `count_check`, manifest).
- **Đang làm:** Giai đoạn 2, mày đang tự gõ các module tiền xử lý.
- **Phát hiện từ dữ liệu thật:** DPR là tập con của QuAD train, nên tao đề xuất test chỉ dùng QuAD test.

Toàn bộ bản gốc của cuộc trò chuyện vẫn nằm trong file transcript:
`C:\Users\PC\.claude\projects\D--Project-2026-2027-HK1-nlp-v1\8911bc9d-dc92-40e7-913e-28245915de94.jsonl`

Lệnh `/login` vừa rồi không ảnh hưởng gì đến việc này.

Tao vừa đọc lại `build_qa.py`, `split.py` và `preprocessing_rules.md`. Cả ba **chưa thay đổi** so với bảng lỗi tao đưa ở trên, nên bảng đó vẫn đúng.

Câu hỏi còn treo: mày muốn **tao sửa thẳng các lỗi vào file** hay **tự sửa theo bảng rồi gọi tao kiểm tra lại**? Nếu sửa thẳng thì tao cũng cần mày chốt luôn: test chỉ lấy QuAD test (2.204 mẫu), đúng không?


---

## 🧑 USER #40

Tao muốn mày xuất lại toàn bộ đoạn hội thoại
