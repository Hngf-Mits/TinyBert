"""dataset.py - Aspect-Based Sentiment Analysis data (SemEval-2014 Task 4, restaurants).

Task   : 3-class aspect polarity (negative / neutral / positive); input = (aspect span, review text).
Safety : labels are mapped STRICTLY (unknown values raise), every split's label distribution is validated,
         and splitting is done per review sentence to avoid leakage between train / dev / test.
"""
from collections import Counter

import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

import os

from config import (BATCH_SIZE, CONFLICT_NAMES, CONFLICT_POLICY, DATA_SEED, DATASET_NAME, DEV_FRACTION,
                    LABEL_COLUMNS, LABEL_NAMES, LABEL_SYNONYMS, MAX_LENGTH, MIN_PER_CLASS, NUM_LABELS,
                    RESULTS_DIR, TEST_SOURCE)
from utils import save_json

CONFLICT_LABEL = "conflict"    # internal sentinel (was called CONFLICT in the notebook; renamed to avoid
                               # clashing with the CONFLICT_POLICY setting)


# ------------------------------------------------------ labels (strict + safe)
def normalize_label(value, label_names=None):
    """Return 0/1/2, or CONFLICT_LABEL. Raise ValueError for anything unrecognised (never guess)."""
    if label_names is not None and isinstance(value, (int, np.integer)):
        value = label_names[int(value)]
    if isinstance(value, str):
        key = value.strip().lower()
        if key in CONFLICT_NAMES:
            return CONFLICT_LABEL
        if key in LABEL_SYNONYMS:
            return LABEL_SYNONYMS[key]
    raise ValueError(f"unrecognised label {value!r}")


def _group_split(texts, fraction, seed):
    """Split row indices so that all rows of one review sentence land on the same side."""
    unique = sorted(set(texts))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    held_texts = set(unique[: max(1, int(round(len(unique) * fraction)))])
    rest = [i for i, t in enumerate(texts) if t not in held_texts]
    held = [i for i, t in enumerate(texts) if t in held_texts]
    return rest, held


def _take(cols, idx):
    return {k: [cols[k][i] for i in idx] for k in ("text", "span", "target")}


def _distribution(targets):
    counts = Counter(targets)
    return {LABEL_NAMES[c]: counts.get(c, 0) for c in range(NUM_LABELS)}


def _check_split(name, targets, min_per_class=MIN_PER_CLASS):
    dist = _distribution(targets)
    if any(v < min_per_class for v in dist.values()):
        raise RuntimeError(
            f"Split '{name}' has a broken label distribution {dist} (need >= {min_per_class} "
            f"samples per class). Every metric computed on it would be meaningless. "
            f"Inspect the raw label values printed by inspect_data(); if the official test split "
            f"is the problem, set TEST_SOURCE = 'from_train' as a workaround.")
    return dist


def prepare_splits(raw, conflict=CONFLICT_POLICY, test_source=TEST_SOURCE, dev_fraction=DEV_FRACTION,
                   seed=DATA_SEED, verbose=True):
    """Pure-python (testable) cleaning + splitting.

    raw: {split: {"text": [...], "span": [...], "label": [...], "label_names": list|None}}
    conflict: "drop" (default, standard 3-class ABSA) or "neutral" (merge into neutral).
    Returns (splits, stats); splits = {"train", "dev", "test"} -> {"text", "span", "target"}.
    """
    assert conflict in ("drop", "neutral"), "conflict must be 'drop' or 'neutral'"
    assert test_source in ("official", "from_train"), "test_source must be 'official'/'from_train'"
    stats = {"raw_label_counts": {}, "n_conflict_dropped": {}}
    cleaned = {}
    # With test_source="from_train" the official test split is ignored entirely.
    needed = ("train", "test") if test_source == "official" else ("train",)
    for split in needed:
        if split not in raw:
            raise KeyError(f"dataset has no '{split}' split (found {list(raw)})")
        cols = raw[split]
        stats["raw_label_counts"][split] = dict(Counter(map(repr, cols["label"])))
        if all(l is None or (isinstance(l, str) and not l.strip()) for l in cols["label"]):
            raise ValueError(
                f"Split '{split}' is UNLABELED: all {len(cols['label'])} labels are empty. "
                f"Set TEST_SOURCE = 'from_train' to carve a labelled test set out of train.")
        text, span, target, unknown, n_conflict = [], [], [], Counter(), 0
        for t, s, lab in zip(cols["text"], cols["span"], cols["label"]):
            try:
                m = normalize_label(lab, cols.get("label_names"))
            except ValueError:
                unknown[repr(lab)] += 1
                continue
            if m == CONFLICT_LABEL:
                n_conflict += 1
                if conflict == "drop":
                    continue
                m = 1
            text.append(t); span.append(s); target.append(m)
        if unknown:
            raise ValueError(
                f"Split '{split}' has label values that cannot be mapped to {LABEL_NAMES}: "
                f"{dict(unknown.most_common(10))}. Raw label counts: "
                f"{stats['raw_label_counts'][split]}")
        stats["n_conflict_dropped"][split] = n_conflict if conflict == "drop" else 0
        cleaned[split] = {"text": text, "span": span, "target": target}

    train = cleaned["train"]
    if test_source == "official":
        test = cleaned["test"]
    else:  # carve a test set out of train, grouped by sentence
        rest, test_idx = _group_split(train["text"], 0.15, seed)
        test, train = _take(train, test_idx), _take(train, rest)
    rest, dev_idx = _group_split(train["text"], dev_fraction, seed + 1)
    splits = {"train": _take(train, rest), "dev": _take(train, dev_idx), "test": test}

    stats["mapped_label_counts"] = {}
    for name, cols in splits.items():
        stats["mapped_label_counts"][name] = _check_split(name, cols["target"])
    stats["sizes"] = {k: len(v["target"]) for k, v in splits.items()}
    stats["conflict_policy"] = conflict
    stats["test_source"] = test_source

    if verbose:
        print(f"conflict policy = {conflict} | test source = {test_source}")
        for name in ("train", "dev", "test"):
            d = stats["mapped_label_counts"][name]
            n = stats["sizes"][name]
            pct = ", ".join(f"{k} {v} ({v / n * 100:.1f}%)" for k, v in d.items())
            print(f"  {name:<5} n={n:<5} {pct}")
    return splits, stats


_DATA_CACHE = {}


def _read_raw():
    ds = load_dataset(DATASET_NAME)
    label_col = next((c for c in LABEL_COLUMNS if c in ds["train"].column_names), None)
    if label_col is None:
        raise KeyError(f"no label column among {LABEL_COLUMNS}; columns = {ds['train'].column_names}")
    raw = {}
    for split in ds.keys():
        feat = ds[split].features[label_col]
        raw[split] = {
            "text": list(ds[split]["text"]),
            "span": list(ds[split]["span"]),
            "label": list(ds[split][label_col]),
            "label_names": list(feat.names) if hasattr(feat, "names") else None,
        }
    return raw


def load_absa(test_source=TEST_SOURCE, conflict=CONFLICT_POLICY, dev_fraction=DEV_FRACTION, verbose=False):
    key = (test_source, conflict, dev_fraction)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = prepare_splits(_read_raw(), conflict, test_source, dev_fraction, verbose=verbose)
    return _DATA_CACHE[key]


def inspect_data(test_source=TEST_SOURCE, conflict=CONFLICT_POLICY, dev_fraction=DEV_FRACTION):
    """Print raw + mapped label distributions per split, save them, and fail loudly if broken."""
    raw = _read_raw()
    print("RAW label values (before any mapping):")
    for split, cols in raw.items():
        print(f"  {split:<10} n={len(cols['label']):<5} {dict(Counter(map(repr, cols['label'])))}")
    splits, stats = prepare_splits(raw, conflict, test_source, dev_fraction, verbose=True)
    _DATA_CACHE[(test_source, conflict, dev_fraction)] = (splits, stats)
    save_json(stats, os.path.join(RESULTS_DIR, "label_stats.json"))
    return stats


def majority_accuracy(stats, split="test"):
    """Accuracy (%) of always predicting the most frequent class - the floor any real model must beat."""
    return max(stats["mapped_label_counts"][split].values()) / stats["sizes"][split] * 100


def build_dataloaders(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, test_source=TEST_SOURCE,
                      conflict=CONFLICT_POLICY, dev_fraction=DEV_FRACTION):
    """Return (train_loader, dev_loader, test_loader). Input = (aspect span, review text)."""
    splits, _ = load_absa(test_source, conflict, dev_fraction)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_fn(batch):
        return tokenizer(batch["span"], batch["text"], truncation=True, max_length=max_length)

    loaders = {}
    for name, cols in splits.items():
        ds = Dataset.from_dict({"text": cols["text"], "span": cols["span"], "labels": cols["target"]})
        ds = ds.map(tokenize_fn, batched=True, remove_columns=["text", "span"])
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"), collate_fn=collator)
    return loaders["train"], loaders["dev"], loaders["test"]
