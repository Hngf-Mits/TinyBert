"""dataset.py - SST-2 loading, tokenisation and dataloaders.

Splits: train / dev (checkpoint selection only) / test (= GLUE validation, touched once per run).
"""
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (BATCH_SIZE, DATA_SEED, DEV_FRACTION, EVAL_BATCH_SIZE, HF_DATASET, LABEL_NAMES, MAX_LENGTH,
                    MIN_PER_CLASS, NUM_LABELS, RES_DIR)
from utils import save_json


def get_columns(data_seed=DATA_SEED, dev_fraction=DEV_FRACTION):
    """Return ({split: {"a": [...], "b": None, "y": [...]}}, num_labels)."""
    from datasets import load_dataset
    ds = load_dataset(*HF_DATASET)
    s, y = list(ds["train"]["sentence"]), list(ds["train"]["label"])
    perm = np.random.RandomState(data_seed).permutation(len(s))
    n_dev = int(len(s) * dev_fraction)
    dev_i, tr_i = perm[:n_dev], perm[n_dev:]
    pick = lambda ids: {"a": [s[i] for i in ids], "b": None, "y": [y[i] for i in ids]}
    cols = {"train": pick(tr_i), "dev": pick(dev_i),
            "test": {"a": list(ds["validation"]["sentence"]), "b": None,
                     "y": list(ds["validation"]["label"])}}   # GLUE test labels are hidden
    return cols, NUM_LABELS


def inspect_data(cols, min_per_class=MIN_PER_CLASS):
    """Print label distribution per split, save it to results/label_stats.json, fail loudly if a split is broken."""
    stats = {"sizes": {}, "label_counts": {}}
    for name in ("train", "dev", "test"):
        counts = Counter(cols[name]["y"])
        dist = {LABEL_NAMES[c]: counts.get(c, 0) for c in range(NUM_LABELS)}
        if any(v < min_per_class for v in dist.values()):
            raise RuntimeError(f"Split '{name}' has a broken label distribution {dist} "
                               f"(need >= {min_per_class} samples per class); every metric on it would be meaningless.")
        n = len(cols[name]["y"])
        stats["sizes"][name], stats["label_counts"][name] = n, dist
        print(f"  {name:<5} n={n:<6} " + ", ".join(f"{k} {v} ({v / n * 100:.1f}%)" for k, v in dist.items()))
    stats["majority_acc"] = max(stats["label_counts"]["test"].values()) / stats["sizes"]["test"] * 100
    print(f"Majority-class accuracy on test (the floor any real model must beat): {stats['majority_acc']:.1f}%")
    save_json(stats, os.path.join(RES_DIR, "label_stats.json"))
    return stats


class Tokenized(Dataset):
    def __init__(self, tok, cols, max_length=MAX_LENGTH):
        enc = tok(cols["a"], cols["b"], truncation=True, max_length=max_length) if cols["b"] \
            else tok(cols["a"], truncation=True, max_length=max_length)
        self.ids = enc["input_ids"]
        self.tt = enc.get("token_type_ids") or [[0] * len(x) for x in self.ids]
        self.y = cols["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ids[i], self.tt[i], self.y[i], i


def collate(batch):
    n, L = len(batch), max(len(b[0]) for b in batch)
    ids = torch.zeros(n, L, dtype=torch.long)
    tt = torch.zeros_like(ids)
    am = torch.zeros_like(ids)
    for r, (a, t, _, _) in enumerate(batch):
        k = len(a)
        ids[r, :k] = torch.tensor(a)
        tt[r, :k] = torch.tensor(t)
        am[r, :k] = 1
    return {"input_ids": ids, "token_type_ids": tt, "attention_mask": am,
            "labels": torch.tensor([b[2] for b in batch]), "idx": torch.tensor([b[3] for b in batch])}


def make_loaders(tok, cols, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    ds = {k: Tokenized(tok, v, max_length) for k, v in cols.items()}
    train = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
    # un-shuffled copy of train: used once to cache teacher logits in dataset order
    train_fixed = DataLoader(ds["train"], batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate)
    dev = DataLoader(ds["dev"], batch_size=EVAL_BATCH_SIZE, collate_fn=collate)
    test = DataLoader(ds["test"], batch_size=EVAL_BATCH_SIZE, collate_fn=collate)
    return {"train": train, "train_fixed": train_fixed, "dev": dev, "test": test}, ds
