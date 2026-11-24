"""GĐ14.1: kiểm ba model ứng viên TRƯỚC khi chấm dữ liệu thật.
  (1) cấu hình: số nhãn, id2label, độ dài tối đa, tokenizer nhanh;
  (2) nhãn nào là 'liên quan': thử các cặp có đáp án biết trước (đoạn đúng phải điểm cao hơn đoạn lạc đề);
  (3) VRAM đỉnh và thời gian chấm 20 cặp (đúng kích thước pool) khi model chạy MỘT mình trên GPU.
Không dùng nhãn/độ đo của tập đánh giá: các cặp thử là câu viết tay và đoạn corpus ngẫu nhiên (seed cố định).
Ghi results/14a_probe.json (không ghi đè)."""
import argparse
import json
import random
import statistics
import time

import torch

from germanqa.config import PROCESSED_DIR, RESULTS_DIR, SEED
from germanqa.data.io_utils import read_json, read_jsonl, write_json
from germanqa.rerank.cross_encoder import CrossEncoderReranker
from germanqa.rerank.hub import local_snapshot

CANDIDATES = {
    "mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "gbert_dpr": "deepset/gbert-base-germandpr-reranking",
    "bge_m3": "BAAI/bge-reranker-v2-m3",
}
# (câu hỏi, đoạn đúng, đoạn lạc đề): viết tay, đáp án rõ ràng
HAND = [
    ("Was ist die Hauptstadt von Deutschland?",
     "Berlin ist die Hauptstadt der Bundesrepublik Deutschland und zugleich eines ihrer sechzehn Länder.",
     "Der Rhein ist ein Fluss in Mitteleuropa, der in den Schweizer Alpen entspringt und in die Nordsee mündet."),
    ("Wer schrieb den Faust?",
     "Johann Wolfgang von Goethe schrieb die Tragödie Faust, deren erster Teil 1808 erschien.",
     "Der Kölner Dom ist eine römisch-katholische Kirche und gehört seit 1996 zum Weltkulturerbe."),
    ("Wie hoch ist die Zugspitze?",
     "Die Zugspitze ist mit 2962 Metern über dem Meeresspiegel der höchste Berg Deutschlands.",
     "Der FC Bayern München wurde 1900 gegründet und ist der erfolgreichste deutsche Fußballverein."),
    ("Wann fiel die Berliner Mauer?",
     "Die Berliner Mauer fiel am 9. November 1989, als die Grenzübergänge geöffnet wurden.",
     "Die Currywurst ist ein Imbissgericht aus gebrühter und gebratener Wurst mit einer Soße aus Ketchup."),
    ("Welche Sprache spricht man in Österreich?",
     "In Österreich ist Deutsch die Amtssprache, daneben gibt es regionale Minderheitensprachen.",
     "Photosynthese ist ein biochemischer Prozess, bei dem Pflanzen Lichtenergie in chemische Energie umwandeln."),
]


def pick_label(rr, num_labels):
    """Mit model nhiều nhãn: chọn nhãn khiến đoạn đúng điểm cao hơn đoạn lạc đề trên cả 5 cặp; trả về (nhãn, bảng)."""
    table = {}
    for lab in range(num_labels):
        r = CrossEncoderReranker.__new__(CrossEncoderReranker)
        r.__dict__.update(rr.__dict__)
        r.relevant_label = lab
        wins = 0
        for q, good, bad in HAND:
            s = r.score_texts(q, [good, bad])
            wins += int(s[0] > s[1])
        table[lab] = wins
    best = [lab for lab, w in table.items() if w == len(HAND)]
    return (best[0] if len(best) == 1 else None), table


def main(batch_sizes):
    out_path = RESULTS_DIR / "14a_probe.json"
    if out_path.exists():
        raise SystemExit(f"{out_path.name} đã tồn tại: không ghi đè")
    corpus = read_jsonl(PROCESSED_DIR / "corpus.jsonl")
    texts = [c["original_text"] for c in corpus]
    qa = read_jsonl(PROCESSED_DIR / "qa_dev.jsonl")
    rnd = random.Random(SEED)
    res = {"seed": SEED, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__, "models": {}}
    for key, repo in CANDIDATES.items():
        path = local_snapshot(repo)
        cfg = read_json(path + "/config.json")
        info = {"repo": repo, "snapshot": path, "num_labels": len(cfg.get("id2label", {"0": 0})), "id2label": cfg.get("id2label"),
                "max_position_embeddings": cfg.get("max_position_embeddings"), "architectures": cfg.get("architectures")}
        max_len = min(512, (cfg.get("max_position_embeddings") or 514) - 2)
        info["max_length_used"] = max_len
        # nhãn: nạp tạm với nhãn 0 để chấm, rồi chọn
        rr = CrossEncoderReranker(path, max_length=max_len, batch_size=16, fp16=torch.cuda.is_available(),
                                  relevant_label=0 if info["num_labels"] > 1 else None)
        info["tokenizer_fast"] = rr.tokenizer.is_fast
        if info["num_labels"] > 1:
            lab, table = pick_label(rr, info["num_labels"])
            info["relevant_label"], info["label_check_wins_of_5"] = lab, table
            rr.relevant_label = lab if lab is not None else 0
        else:
            r1 = [rr.score_texts(q, [g, b]) for q, g, b in HAND]
            info["relevant_label"] = None
            info["label_check_wins_of_5"] = sum(int(s[0] > s[1]) for s in r1)
        # VRAM + tốc độ: 20 đoạn thật (1 đoạn đúng + 19 ngẫu nhiên), câu hỏi dev thật
        speed = {}
        if torch.cuda.is_available():
            for bs in batch_sizes:
                rr.batch_size = bs
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                times = []
                try:
                    for i in range(25):
                        ex = qa[i]
                        cands = rnd.sample(texts, 20)
                        t = time.perf_counter()
                        rr.score_texts(ex["question"], cands)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - t)
                    times = times[5:]                                    # bỏ 5 lần khởi động
                    speed[str(bs)] = {"median_s_per_20_pairs": round(statistics.median(times), 4),
                                      "p90_s": round(sorted(times)[int(0.9 * len(times)) - 1], 4),
                                      "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
                except torch.cuda.OutOfMemoryError:
                    speed[str(bs)] = {"error": "OOM"}
                    torch.cuda.empty_cache()
        info["speed"] = speed
        info["weights_vram_mb"] = round(sum(p.numel() * p.element_size() for p in rr.model.parameters()) / 2**20, 1)
        res["models"][key] = info
        print(key, json.dumps({k: v for k, v in info.items() if k not in ("snapshot", "id2label")}, ensure_ascii=False), flush=True)
        del rr
        torch.cuda.empty_cache()
    write_json(out_path, res)
    print("đã ghi", out_path.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16])
    main(ap.parse_args().batch_sizes)
