"""Vòng huấn luyện và suy luận cho reader (GĐ4.3)."""
import random
import time

import numpy as np
import torch

from germanqa.reader.postprocess import best_answers
from germanqa.reader.qa_metrics import evaluate_answers


def collate(batch, pad_id=0):
    """Ghép các cửa sổ thành một lô, đệm (pad) tới độ dài lớn nhất TRONG LÔ."""
    n = max(len(f["input_ids"]) for f in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    types = torch.zeros((len(batch), n), dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, f in enumerate(batch):
        k = len(f["input_ids"])
        ids[i, :k] = torch.tensor(f["input_ids"])
        types[i, :k] = torch.tensor(f["token_type_ids"])
        mask[i, :k] = 1
    out = {"input_ids": ids, "token_type_ids": types, "attention_mask": mask}
    if "start_positions" in batch[0]:
        out["start_positions"] = torch.tensor([f["start_positions"] for f in batch])
        out["end_positions"] = torch.tensor([f["end_positions"] for f in batch])
    return out


@torch.no_grad()
def predict_logits(model, feats, batch_size=32, device="cuda", fp16=True):
    """Chạy mô hình trên mọi cửa sổ; trả về (start_logits, end_logits) là danh sách mảng numpy,
    mỗi mảng có độ dài đúng bằng số token của cửa sổ đó (không dính phần đệm)."""
    model.eval()
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))   # nhóm theo độ dài
    starts, ends = [None] * len(feats), [None] * len(feats)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        batch = {k: v.to(device) for k, v in collate([feats[i] for i in idx]).items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16 and device == "cuda"):
            out = model(**batch)
        s, e = out.start_logits.float().cpu().numpy(), out.end_logits.float().cpu().numpy()
        for row, i in enumerate(idx):
            n = len(feats[i]["input_ids"])
            starts[i], ends[i] = s[row, :n], e[row, :n]
    return starts, ends


def evaluate(model, examples, feats, n_best, max_answer_tokens, batch_size, device, fp16=True):
    s, e = predict_logits(model, feats, batch_size, device, fp16)
    preds = best_answers(examples, feats, s, e, n_best, max_answer_tokens)
    return evaluate_answers(preds, examples), preds


def train(model, train_feats, dev_examples, dev_feats, cfg, seed, device, log):
    """AdamW + warmup tuyến tính rồi giảm tuyến tính; fp16 autocast + GradScaler;
    gradient accumulation; giữ checkpoint có EM+F1 dev tốt nhất. Trả về (best_state, history)."""
    t = cfg["train"]
    rng = random.Random(seed)
    model.to(device)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": t["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=t["lr"])
    steps_per_epoch = -(-len(train_feats) // (t["batch_size"] * t["grad_accum"]))
    total = steps_per_epoch * t["epochs"]
    warm = int(total * t["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else max(0.0, (total - s) / max(total - warm, 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=t["fp16"] and device == "cuda")

    best, best_state, history, step = -1.0, None, [], 0
    t0 = time.time()
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        order = list(range(len(train_feats)))
        rng.shuffle(order)
        batches = [order[i:i + t["batch_size"]] for i in range(0, len(order), t["batch_size"])]
        run_loss, n_loss = 0.0, 0
        for bi, idx in enumerate(batches, start=1):
            batch = {k: v.to(device) for k, v in collate([train_feats[i] for i in idx]).items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=t["fp16"] and device == "cuda"):
                loss = model(**batch).loss / t["grad_accum"]
            scaler.scale(loss).backward()
            run_loss += loss.item() * t["grad_accum"]
            n_loss += 1
            if bi % t["grad_accum"] == 0 or bi == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % t["log_every"] == 0:
                    log(f"epoch {epoch} step {step}/{total} loss {run_loss / n_loss:.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
                    run_loss, n_loss = 0.0, 0
        metrics, _ = evaluate(model, dev_examples, dev_feats, cfg["n_best"], cfg["max_answer_tokens"],
                              cfg["eval_batch_size"], device, t["fp16"])
        history.append({"epoch": epoch, "step": step, "dev": metrics, "seconds": round(time.time() - t0)})
        log(f"== epoch {epoch}: dev EM {metrics['em']:.4f} F1 {metrics['f1']:.4f}")
        if metrics["em"] + metrics["f1"] > best:
            best = metrics["em"] + metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, history


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
