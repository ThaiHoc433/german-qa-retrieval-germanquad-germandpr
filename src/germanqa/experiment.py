### Thông tin môi trường và experiment_id, dùng để gắn vào mọi kết quả"
"""Công cụ tái lập: seed, mã thí nghiệm, ghi lại môi trường, đọc cấu hình YAML."""
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import yaml

from germanqa.config import CONFIG_DIR, RESULTS_DIR


def set_seed(seed):
    """Cố định mọi nguồn ngẫu nhiên đang dùng."""
    random.seed(seed)
    np.random.seed(seed)
    try:                       # torch chưa cần ở GĐ0-3, có thì seed luôn
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_config(name="baseline"):
    """Đọc configs/<name>.yaml thành dict."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def new_experiment_id(tag, config):
    """Mã thí nghiệm = thẻ + thời điểm + 8 ký tự băm của cấu hình.
    Cùng cấu hình -> cùng đuôi băm; khác cấu hình -> đuôi khác."""
    blob = json.dumps(config, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{tag}_{stamp}_{digest}"


def environment_info(packages=("numpy", "pandas", "datasets", "transformers", "PyYAML")):
    """Phiên bản Python và thư viện, để kết quả truy vết được."""
    versions = {}
    for p in packages:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def save_run(experiment_id, config, metrics, extra=None):
    """Ghi results/<experiment_id>.json gồm cấu hình, môi trường, số đo."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"experiment_id": experiment_id, "config": config,
              "environment": environment_info(), "metrics": metrics}
    if extra:
        record.update(extra)
    path = RESULTS_DIR / f"{experiment_id}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path
