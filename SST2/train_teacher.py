"""train_teacher.py - fine-tune bert-base-uncased on SST-2 (the teacher).

Usage (from inside SST-2/):
    python train_teacher.py                # config.TEACHER_EPOCHS epochs, lr config.LR_TEACHER
Saves checkpoints/teacher_bert_base/ and results/teacher_bert_base_train_log.json.
Skip this script entirely by setting config.TEACHER_PRETRAINED to a hub checkpoint.
"""
import argparse
import os
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as C
from dataset import get_columns, inspect_data, make_loaders
from model_teacher import teacher_dir
from train_distill import train
from utils import evaluate, save_json, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=C.TEACHER_EPOCHS)
    p.add_argument("--lr", type=float, default=C.LR_TEACHER)
    p.add_argument("--force", action="store_true", help="retrain even if a teacher checkpoint already exists")
    a = p.parse_args()

    if os.path.exists(os.path.join(teacher_dir(), "config.json")) and not a.force:
        print(f"teacher already trained ({teacher_dir()}); use --force to retrain")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== TEACHER: bert-base-uncased fine-tuned on SST-2 =====", "| device:", device)
    set_seed(42)
    cols, num_labels = get_columns()
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    loaders, _ = make_loaders(tok, cols)
    majority_acc = inspect_data(cols)["majority_acc"]

    teacher = AutoModelForSequenceClassification.from_pretrained(C.TEACHER_ID, num_labels=num_labels).to(device)
    t0 = time.time()
    out = train(teacher, loaders, device, num_labels, epochs=a.epochs, lr=a.lr)      # CE only, best dev epoch restored

    test = evaluate(teacher, loaders["test"], device, num_labels, amp=True)
    print(f"TEACHER test: acc {test['accuracy']:.2f} | macro-F1 {test['macro_f1']:.2f} "
          f"| per-class F1 {[round(x, 1) for x in test['f1_per_class']]}")
    if test["accuracy"] <= majority_acc:
        raise RuntimeError("Teacher is not better than always predicting the majority class: "
                           "something is wrong with data/labels/training. Do not continue.")

    teacher.save_pretrained(teacher_dir())
    tok.save_pretrained(teacher_dir())
    save_json({"name": C.TEACHER_NAME, **out, "test": test, "lr": a.lr, "epochs": a.epochs,
               "train_minutes": (time.time() - t0) / 60.0},
              os.path.join(C.RES_DIR, f"{C.TEACHER_NAME}_train_log.json"))
    print("saved:", teacher_dir())


if __name__ == "__main__":
    main()
