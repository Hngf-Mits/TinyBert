"""train_distill.py - teacher -> student size sweep: CE-only vs KD for every size (resumable).

Usage (from inside SST-2/):
    python train_distill.py                      # all sizes, default seeds
    python train_distill.py --sizes bert_tiny_L2_H128 --seeds 42
    python train_distill.py --full-seeds         # 3 seeds for every size
Every finished run is saved to results/*.json, so re-running resumes instead of restarting.
"""
import argparse
import gc
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config as C
from dataset import get_columns, inspect_data, make_loaders
from model_student import build_student
from model_teacher import get_teacher
from utils import evaluate, n_params, save_json, set_seed


def kd_loss(student_logits, teacher_logits, labels, T=C.TEMPERATURE, alpha=C.ALPHA):
    ce = F.cross_entropy(student_logits, labels)
    kd = F.kl_div(F.log_softmax(student_logits / T, -1), F.softmax(teacher_logits / T, -1),
                  reduction="batchmean") * T * T
    return alpha * ce + (1 - alpha) * kd


def train(model, loaders, device, num_labels, *, epochs, lr, teacher_logits=None, T=C.TEMPERATURE, alpha=C.ALPHA,
          weight_decay=C.WEIGHT_DECAY, amp=True):
    """CE-only if teacher_logits is None, otherwise KD from cached logits. Best epoch (dev macro-F1) is restored."""
    use_kd = teacher_logits is not None
    if use_kd:
        teacher_logits = teacher_logits.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = epochs * len(loaders["train"])
    sched = get_linear_schedule_with_warmup(opt, int(C.WARMUP_RATIO * total), total)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_ep, best_state, hist = -1.0, 0, None, []
    for ep in range(epochs):
        model.train()
        run, t0 = 0.0, time.time()
        for b in loaders["train"]:
            idx = b.pop("idx").to(device)
            y = b.pop("labels").to(device)
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**b).logits
            logits = logits.float()
            loss = kd_loss(logits, teacher_logits[idx], y, T, alpha) if use_kd else F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        dev = evaluate(model, loaders["dev"], device, num_labels, amp)
        hist.append({"epoch": ep + 1, "loss": run / len(loaders["train"]), "dev_acc": dev["accuracy"],
                     "dev_f1": dev["macro_f1"], "sec": time.time() - t0})
        print(f"  ep{ep + 1:>2} loss {hist[-1]['loss']:.4f} | dev acc {dev['accuracy']:.2f} f1 {dev['macro_f1']:.2f}")
        if dev["macro_f1"] > best:
            best, best_ep = dev["macro_f1"], ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"best_epoch": best_ep, "best_dev_f1": best, "history": hist}


def run_student(size, mode, seed, *, tok, loaders, device, num_labels, teacher_logits, epochs):
    name = f"{size}__{mode}__s{seed}"
    path = os.path.join(C.RES_DIR, name + ".json")
    if os.path.exists(path):
        return
    print(f"\n===== {name} =====")
    set_seed(seed)
    t0 = time.time()
    model, lr = build_student(size, device, tok)
    out = train(model, loaders, device, num_labels, epochs=epochs, lr=lr,
                teacher_logits=teacher_logits if mode == "kd" else None)
    m = evaluate(model, loaders["test"], device, num_labels, amp=True)
    save_json({"kind": "run", "size": size, "mode": mode, "seed": seed, "lr": lr, "epochs": epochs,
               "params_M": n_params(model) / 1e6, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
               "f1_per_class": m["f1_per_class"], "best_epoch": out["best_epoch"], "history": out["history"],
               "train_min": (time.time() - t0) / 60}, path)
    print(f"  TEST acc {m['accuracy']:.2f} | macro-F1 {m['macro_f1']:.2f} | {(time.time() - t0) / 60:.1f} min")
    ck = os.path.join(C.CKPT_DIR, size)          # one checkpoint per architecture is enough for the benchmark
    if not os.path.exists(ck):
        model.save_pretrained(ck)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", default=C.SIZES, choices=C.SIZES)
    p.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    p.add_argument("--full-seeds", action="store_true", help="run all seeds for every size "
                   "(default: all seeds only for MULTI_SEED_SIZES, first seed for the rest)")
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", "| CPU cores:", os.cpu_count())
    seeds_for = lambda s: a.seeds if (a.full_seeds or s in C.MULTI_SEED_SIZES) else a.seeds[:1]
    print("planned student runs:", sum(2 * len(seeds_for(s)) for s in a.sizes))

    # ---- data
    cols, num_labels = get_columns()
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    loaders, _ = make_loaders(tok, cols)
    majority_acc = inspect_data(cols)["majority_acc"]

    # ---- teacher
    teacher_logits, teacher_res = get_teacher(tok, loaders, len(cols["train"]["y"]), majority_acc, device)
    y_tr = torch.tensor(cols["train"]["y"])
    probs = torch.softmax(teacher_logits, -1)
    print(f"TEACHER test: acc {teacher_res['accuracy']:.2f} | macro-F1 {teacher_res['macro_f1']:.2f}")
    print(f"teacher accuracy ON ITS OWN TRAIN SET: {(teacher_logits.argmax(1) == y_tr).float().mean().item() * 100:.2f}% "
          f"| mean max-prob {probs.max(1).values.mean().item():.3f}")
    print("-> if train acc is ~100% and max-prob ~1.0, the teacher memorised the train set: "
          "its soft labels carry little extra information, so KD has little to add.")

    # ---- students
    for size in a.sizes:
        for seed in seeds_for(size):
            for mode in ("ce", "kd"):
                run_student(size, mode, seed, tok=tok, loaders=loaders, device=device, num_labels=num_labels,
                            teacher_logits=teacher_logits, epochs=a.epochs)
    print("\nall student runs done")


if __name__ == "__main__":
    main()
