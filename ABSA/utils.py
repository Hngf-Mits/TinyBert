"""utils.py - seeding, device, size counting, json helpers, metrics and evaluation."""
import json
import os
import random

import numpy as np
import torch

from config import NUM_LABELS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size / (1024 * 1024)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classification_metrics(preds, labels, num_labels=NUM_LABELS):
    """Accuracy and macro-F1 (over ALL classes, absent classes count as 0), in %."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    acc = float((preds == labels).mean() * 100)
    f1s = []
    for c in range(num_labels):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom * 100) if denom > 0 else 0.0)
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "f1_per_class": f1s}


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("labels")
        preds.append(model(**batch).logits.argmax(-1).cpu())
        labels.append(y.cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def evaluate_model(model, dataloader, device):
    return classification_metrics(*predict(model, dataloader, device))
