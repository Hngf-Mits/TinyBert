"""train_teacher.py - step 1: check the data, then fine-tune bert-base-uncased on ABSA (CE loss).

Usage:  python train_teacher.py
Output: checkpoints/teacher_bert_base/  and  results/teacher_bert_base_train_log.json
"""
import os
import time

from config import (CONFLICT_POLICY, LR_TEACHER, PRECISION, RESULTS_DIR, TEACHER_EPOCHS, TEACHER_NAME,
                    TEACHER_SEED, TEST_SOURCE)
from dataset import build_dataloaders, inspect_data, majority_accuracy
from model_teacher import build_teacher, freeze, save_teacher
from trainer import train_model
from utils import evaluate_model, get_device, save_json, set_seed


def main():
    device = get_device()
    print(f"Device: {device}")

    # Fails loudly if any split is unlabeled, has an unmappable label, or is missing a class.
    stats = inspect_data(test_source=TEST_SOURCE, conflict=CONFLICT_POLICY)
    majority_acc = majority_accuracy(stats)
    print(f"\nMajority-class accuracy on test (the floor any real model must beat): {majority_acc:.1f}%")

    print("\n===== TEACHER: bert-base-uncased fine-tuned on ABSA =====")
    set_seed(TEACHER_SEED)
    teacher, tokenizer = build_teacher(device)
    train_loader, dev_loader, test_loader = build_dataloaders(tokenizer)

    t0 = time.time()
    out = train_model(teacher, train_loader, dev_loader, device, teacher=None,
                      epochs=TEACHER_EPOCHS, lr=LR_TEACHER, precision=PRECISION)

    test = evaluate_model(teacher, test_loader, device)
    print(f"TEACHER test: acc {test['accuracy']:.2f} | macro-F1 {test['macro_f1']:.2f} "
          f"| per-class F1 {[round(x, 1) for x in test['f1_per_class']]}")
    if test["accuracy"] <= majority_acc:
        raise RuntimeError("Teacher is not better than always predicting the majority class: "
                           "something is wrong with data/labels/training. Do not continue.")

    save_teacher(teacher, tokenizer)
    save_json({"name": TEACHER_NAME, **out, "test": test, "lr": LR_TEACHER, "epochs": TEACHER_EPOCHS,
               "precision": PRECISION, "train_minutes": (time.time() - t0) / 60.0},
              os.path.join(RESULTS_DIR, f"{TEACHER_NAME}_train_log.json"))
    freeze(teacher)
    print("Teacher saved.")


if __name__ == "__main__":
    main()
