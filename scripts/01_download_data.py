import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from germanqa.config import DATASETS, RAW_DIR, REPORT_DIR
from germanqa.data.download import download_dataset, get_revision, count_check, write_manifest

def main():
    all_info, all_rows = {}, []
    for name in DATASETS:
        sha = get_revision(name)        #1. ghi đúng phiên bản 
        ds = download_dataset(name, sha)      #2. tải dữ liệu về
        all_info[name] = {"revision_sha": sha, "ds": ds}
        all_rows += count_check(name, ds)  #3. kiểm tra số mẫu
    
    df = pd.DataFrame(all_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)      #4. ghi lại manifest
    df.to_csv(REPORT_DIR / "step1_count_check.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    print("Manifest và báo cáo đã được ghi vào thư mục reports.", write_manifest(all_info))
    # đã ghi lại manifest, báo cáo số mẫu, và SHA của dataset.

if __name__ == "__main__":
    main()