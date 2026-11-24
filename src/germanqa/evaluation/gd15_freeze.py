"""Đóng băng cấu hình trước khi mở test một lần (GĐ15.6). Hàm thuần, test được: dựng bản đóng băng, kiểm lại, và kiểm điều kiện trước khi chạy.

Quy tắc đã chốt từ đầu phần B: chọn cấu hình CHỈ bằng dev; đóng băng; chạy test MỘT lần cho cả ma trận; không dùng lỗi test để thiết kế.
Bản đóng băng ghi SHA-1 của: cấu hình, manifest, kết quả quyết định (prereg, chọn, xác nhận, giả thuyết), các số đo dev đã dùng để phân tích, mã chạy pipeline, và hai tệp
của B0 được dùng lại (B0 đã chạy trên test từ 2026-11-04, nên không chạy lại)."""
from pathlib import Path

from germanqa.data.io_utils import file_sha1, read_json, read_jsonl
from germanqa.evaluation.manifest import load_manifest, verify_manifest_detail
from germanqa.experiment_rules import read_ledger

CONFIGS = ["configs/pipeline.yaml", "configs/reader.yaml", "configs/dense.yaml", "configs/hybrid.yaml", "configs/rerank.yaml"]
MANIFESTS = ["baselines/B0.json", "baselines/B1.json", "baselines/B1.v2.json", "baselines/H0.json", "baselines/H1.json"]
DECISIONS = ["results/13_prereg.json", "results/13_selection.json", "results/13_confirmation.json", "results/14_prereg.json", "results/14_selection.json",
             "results/14_confirmation.json", "results/15_hypotheses.json", "results/15_hypotheses_addendum.json", "results/select_question_ids.json"]
DEV_ANALYSIS = ["results/15_dev_matrix.json", "results/15_stage_timing.json"]
CODE = ["src/germanqa/pipeline.py", "src/germanqa/schemas.py", "src/germanqa/retrieval/hybrid.py", "src/germanqa/retrieval/rrf.py",
        "src/germanqa/retrieval/dense.py", "src/germanqa/retrieval/bm25.py", "src/germanqa/retrieval/bm25_store.py", "src/germanqa/rerank/cross_encoder.py",
        "src/germanqa/rerank/factory.py", "src/germanqa/reader/inference.py", "src/germanqa/reader/features.py", "src/germanqa/reader/qa_metrics.py",
        "src/germanqa/evaluation/gd15_hypotheses.py", "src/germanqa/evaluation/report.py",
        # chính script chạy test và hai script phân tích dev: bằng chứng chúng không bị sửa giữa lúc đóng băng và lúc chạy
        "scripts/GD15/15f_run_test.py", "scripts/GD15/15c_dev_matrix.py", "scripts/GD15/15d_stage_timing.py"]
B0_REUSE = ["results/eval_test_20261103T170542Z_93098dcb.json", "reports/pipeline_test_predictions.jsonl"]
TEST_CODES = {"B1", "H0", "H1"}


def all_frozen_files():
    return CONFIGS + MANIFESTS + DECISIONS + DEV_ANALYSIS + CODE + B0_REUSE


def build_freeze(root):
    root = Path(root)
    return {"files_sha1": {rel: file_sha1(root / rel) for rel in all_frozen_files()},
            "B0_reuse": {"eval_file": B0_REUSE[0], "predictions_file": B0_REUSE[1]}}


def verify_freeze(freeze, root):
    """Danh sách chỗ lệch giữa bản đóng băng và tệp hiện tại (rỗng = khớp)."""
    root, bad = Path(root), []
    for rel, want in freeze["files_sha1"].items():
        p = root / rel
        got = file_sha1(p) if p.exists() else None
        if got != want:
            bad.append(f"{rel}: đóng băng {want}, hiện tại {got}")
    return bad


def b0_reuse_problems(root):
    """B0 được dùng lại nên số cũ phải đúng là của cấu hình B0 hiện tại: so cấu hình lưu trong kết quả test với manifest đã đóng băng."""
    root, bad = Path(root), []
    ev = read_json(root / B0_REUSE[0])
    man = read_json(root / "baselines/B0.json")
    cfg, want = ev["config"]["pipeline_config"], man["effective_config"]["pipeline"]
    for key in ("bm25", "pipeline", "reader"):
        if cfg.get(key) != want.get(key):
            bad.append(f"B0: cấu hình '{key}' của lần chạy test cũ khác manifest hiện tại")
    if ev["config"]["checkpoint"] != Path(want["pipeline"]["checkpoint"]).name:
        bad.append("B0: checkpoint reader của lần chạy test cũ khác manifest hiện tại")
    corpus = file_sha1(root / "data" / "processed" / "corpus.jsonl")[:12]
    if ev["config"]["corpus_version"] != corpus:
        bad.append(f"B0: corpus của lần chạy test cũ ({ev['config']['corpus_version']}) khác corpus hiện tại ({corpus})")
    n_pred = sum(1 for _ in read_jsonl(root / B0_REUSE[1]))
    n_test = sum(1 for _ in read_jsonl(root / "data" / "processed" / "qa_test.jsonl"))
    if n_pred != n_test:
        bad.append(f"B0: {n_pred} dự đoán test đã lưu so với {n_test} câu test")
    return bad


def preflight_problems(root):
    """Mọi điều kiện phải đúng TRƯỚC khi mở test. Rỗng = được phép đóng băng/chạy."""
    root, bad = Path(root), []
    for rel in DECISIONS + DEV_ANALYSIS + B0_REUSE:
        if not (root / rel).exists():
            bad.append(f"thiếu {rel}")
    if bad:
        return bad
    runs = [r for r in read_ledger(root / "results" / "test_exposure_ledger.jsonl") if r["kind"] == "metric_run" and set(r.get("codes", [])) & TEST_CODES]
    if runs:
        bad.append(f"sổ phơi bày đã có lần tính chỉ số test cho {sorted({c for r in runs for c in r['codes']})}")
    for code in ("B0", "B1.v2", "H0", "H1"):
        try:
            mism, n = verify_manifest_detail(load_manifest(code, root), root)
        except (OSError, ValueError, KeyError) as e:           # thiếu/hỏng manifest phải thành một vấn đề được báo, không được làm cổng sập
            bad.append(f"manifest {code} không đọc được ({type(e).__name__})")
            continue
        if mism or n == 0:
            bad.append(f"manifest {code} không khớp tệp thật ({len(mism)} lệch, {n} SHA-1 đã kiểm)")
    try:
        bad += b0_reuse_problems(root)
    except (OSError, ValueError, KeyError) as e:
        bad.append(f"không kiểm được việc dùng lại B0 ({type(e).__name__})")
    return bad
