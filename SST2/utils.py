"""utils.py - seeding, JSON I/O, metrics, prediction and parameter counting."""
import json
import os
import random

import numpy as np
import torch


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------- metrics
def metrics(preds, labels, num_labels):
    preds, labels = np.asarray(preds), np.asarray(labels)
    f1s = []
    for c in range(num_labels):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        d = 2 * tp + fp + fn
        f1s.append(float(2 * tp / d * 100) if d else 0.0)
    return {"accuracy": float((preds == labels).mean() * 100),
            "macro_f1": float(np.mean(f1s)), "f1_per_class": f1s}


@torch.no_grad()
def predict(model, loader, device, amp=False):
    model.eval()
    P, Y = [], []
    for b in loader:
        y = b.pop("labels")
        b.pop("idx")
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            P.append(model(**b).logits.argmax(-1).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def evaluate(model, loader, device, num_labels, amp=False):
    return metrics(*predict(model, loader, device, amp), num_labels)


# ---------------------------------------------------------- model helpers
def n_params(model):
    return sum(p.numel() for p in model.parameters())


def n_params_no_embedding(model):
    emb = model.bert.embeddings.word_embeddings.weight.numel() if hasattr(model, "bert") else 0
    return n_params(model) - emb
