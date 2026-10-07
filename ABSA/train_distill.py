"""train_distill.py - step 2: train the TinyBERT student with CE only and with KD (same hyper-parameters).

Usage:  python train_distill.py                       # all seeds, both modes
        python train_distill.py --seeds 42 --modes kd  # a subset
        python train_distill.py --force                # re-run even if a train log already exists
Needs:  checkpoints/teacher_bert_base/  (run train_teacher.py first)
Output: checkpoints/tinybert_{ce,kd}_s<seed>/  and  results/tinybert_*_train_log.json
"""
import argparse
import gc
import glob
import json
import os
import time

import torch

from config import (ALPHA, CHECKPOINT_DIR, CONFLICT_POLICY, LR_STUDENT, PRECISION, RESULTS_DIR, SEEDS,
                    STUDENT_EPOCHS, STUDENT_MODES, TEMPERATURE, TEST_SOURCE, WEIGHT_DECAY)
from dataset import build_dataloaders, inspect_data
from model_student import build_student, check_same_vocab
from model_teacher import load_teacher
from trainer import train_model
from utils import get_device, save_json, set_seed


def run_student(mode, seed, teacher, tok_teacher, device, force=False):
    name = f"tinybert_{mode}_s{seed}"
    log_path = os.path.join(RESULTS_DIR, f"{name}_train_log.json")
    if os.path.exists(log_path) and not force:
        print(f"skip {name} (already trained; use --force to redo)")
        return
    print(f"\n===== {name} =====")
    set_seed(seed)
    model, tok_student = build_student(device)
    check_same_vocab(tok_student, tok_teacher)
    train_loader, dev_loader, _ = build_dataloaders(tok_student)

    t0 = time.time()
    out = train_model(model, train_loader, dev_loader, device, teacher=teacher if mode == "kd" else None,
                      epochs=STUDENT_EPOCHS, lr=LR_STUDENT, weight_decay=WEIGHT_DECAY,
                      temperature=TEMPERATURE, alpha=ALPHA, precision=PRECISION)

    save_dir = os.path.join(CHECKPOINT_DIR, name)
    model.save_pretrained(save_dir)
    tok_student.save_pretrained(save_dir)
    save_json({"name": name, "mode": mode, "seed": seed, **out, "lr": LR_STUDENT, "epochs": STUDENT_EPOCHS,
               "temperature": TEMPERATURE, "alpha": ALPHA, "precision": PRECISION,
               "train_minutes": (time.time() - t0) / 60.0}, log_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def print_training_curves():
    """Dev curves and the epoch that was kept, for every trained model."""
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_train_log.json"))):
        d = json.load(open(f))
        f1s = [round(h["dev_macro_f1"], 1) for h in d["history"]]
        print(f"{d['name']:<20} best epoch {d['best_epoch']:>2} | dev macro-F1/epoch {f1s} | {d['train_minutes']:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--modes", nargs="+", default=list(STUDENT_MODES), choices=["ce", "kd"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    inspect_data(test_source=TEST_SOURCE, conflict=CONFLICT_POLICY)   # same data check as the teacher run
    teacher, tok_teacher = load_teacher(device)

    for seed in args.seeds:
        for mode in args.modes:        # CE-only baseline AND distillation, same hyper-parameters
            run_student(mode, seed, teacher, tok_teacher, device, args.force)

    print("\nCheckpoints:", sorted(os.listdir(CHECKPOINT_DIR)))
    print_training_curves()


if __name__ == "__main__":
    main()
