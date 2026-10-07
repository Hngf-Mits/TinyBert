"""model_teacher.py - frozen BERT-base teacher + cached logits.

The teacher is either
  * the BERT-base fine-tuned on SST-2 by `train_teacher.py` (default, config.TEACHER_PRETRAINED = None), or
  * a pretrained hub checkpoint (config.TEACHER_PRETRAINED = "<hf id>").
It is frozen and in eval() mode, so its logits never change: compute them ONCE and reuse for every student/seed.
This makes KD runs cost about the same as CE-only runs.
"""
import gc
import os
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as C
from utils import evaluate, load_json, n_params, save_json


def teacher_dir():
    return os.path.join(C.CKPT_DIR, C.TEACHER_NAME)


def load_teacher(device, tok):
    if C.TEACHER_PRETRAINED:
        model = AutoModelForSequenceClassification.from_pretrained(C.TEACHER_PRETRAINED, num_labels=C.NUM_LABELS)
        assert AutoTokenizer.from_pretrained(C.TEACHER_PRETRAINED).get_vocab() == tok.get_vocab(), "teacher vocab differs"
        source = C.TEACHER_PRETRAINED
    else:
        if not os.path.exists(os.path.join(teacher_dir(), "config.json")):
            raise FileNotFoundError(f"no fine-tuned teacher in {teacher_dir()} - run `python train_teacher.py` first "
                                    f"(or set TEACHER_PRETRAINED in config.py)")
        model = AutoModelForSequenceClassification.from_pretrained(teacher_dir())
        source = "fine-tuned here (train_teacher.py)"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, source


@torch.no_grad()
def cache_teacher_logits(teacher, train_fixed, n_train, device, path, amp=True):
    if os.path.exists(path):
        return torch.load(path)
    teacher.eval()
    out = None
    for b in train_fixed:
        idx = b.pop("idx")
        b.pop("labels")
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            lg = teacher(**b).logits.float().cpu()
        if out is None:
            out = torch.zeros(n_train, lg.size(1))
        out[idx] = lg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    return out


def get_teacher(tok, loaders, n_train, majority_acc, device):
    """Return (teacher_logits, teacher_result_dict). Resumes from disk if already done."""
    res_t = os.path.join(C.RES_DIR, "teacher.json")
    cache_path = os.path.join(C.CACHE_DIR, "teacher_logits_sst2.pt")
    if os.path.exists(res_t) and os.path.exists(cache_path):
        print("resumed teacher from disk")
        return torch.load(cache_path), load_json(res_t)

    t0 = time.time()
    teacher, source = load_teacher(device, tok)
    test_m = evaluate(teacher, loaders["test"], device, C.NUM_LABELS, amp=True)
    if test_m["accuracy"] <= majority_acc:
        raise RuntimeError("teacher does not beat the majority-class baseline - stop and check data/labels")
    logits = cache_teacher_logits(teacher, loaders["train_fixed"], n_train, device, cache_path)
    if not os.path.exists(os.path.join(teacher_dir(), "config.json")):   # pretrained hub teacher: save for the benchmark
        teacher.save_pretrained(teacher_dir())
    res = {"kind": "run", "size": "teacher", "mode": "teacher", "seed": 42, "params_M": n_params(teacher) / 1e6,
           "accuracy": test_m["accuracy"], "macro_f1": test_m["macro_f1"], "f1_per_class": test_m["f1_per_class"],
           "train_min": (time.time() - t0) / 60, "source": source}
    save_json(res, res_t)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return logits, res
