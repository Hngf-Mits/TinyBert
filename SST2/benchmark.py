"""benchmark.py - hardware benchmark (latency, throughput, FLOPs, VRAM). Latency depends on the architecture, not the seed.

Run everything (one subprocess per model+device => clean VRAM, resumable):
    python benchmark.py --all
Benchmark a single model (this is what --all calls internally):
    python benchmark.py --name bert_tiny_L2_H128 --model checkpoints/bert_tiny_L2_H128 --device cpu --samples cache/bench_samples.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

import config as C
from utils import n_params, n_params_no_embedding, save_json


# ------------------------------------------------------------ worker (one model)
def bench_one(name, model_path, device, samples_path, threads=0):
    from transformers import AutoModelForSequenceClassification
    if threads:
        torch.set_num_threads(threads)
    dev = torch.device(device)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(dev).eval()
    S = json.load(open(samples_path))
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)

    def batchify(seqs):
        L = max(map(len, seqs))
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        am = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            am[i, :len(s)] = 1
        return {"input_ids": ids.to(dev), "attention_mask": am.to(dev), "token_type_ids": torch.zeros_like(ids).to(dev)}

    with torch.no_grad():
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        one = [batchify([s]) for s in S[:C.BENCH_LATENCY_RUNS]]
        for x in one[:C.BENCH_WARMUP]:
            model(**x)                                       # warm-up
        lat = []
        for x in one:
            sync()
            t = time.perf_counter()
            model(**x)
            sync()
            lat.append((time.perf_counter() - t) * 1e3)
        batches = [batchify(S[i:i + C.BENCH_BATCH]) for i in range(0, len(S), C.BENCH_BATCH)]
        sync()
        t = time.perf_counter()
        for x in batches:
            model(**x)
        sync()
        thr = len(S) / (time.perf_counter() - t)
        flops = None
        try:
            from torch.utils.flop_counter import FlopCounterMode
            x = batchify([[101] + [1000] * 30 + [102]])      # fixed length 32 -> comparable across models
            with FlopCounterMode(display=False) as fc:
                model(**x)
            flops = fc.get_total_flops() / 1e9
        except Exception as e:
            print("flop counter unavailable:", e)

    res = {"kind": "bench", "name": name, "device": device, "threads": torch.get_num_threads(),
           "params_M": n_params(model) / 1e6, "params_noemb_M": n_params_no_embedding(model) / 1e6,
           "size_MB": sum(p.nelement() * p.element_size() for p in model.parameters()) / 2 ** 20,
           "lat_ms_median": float(np.median(lat)), "lat_ms_p95": float(np.percentile(lat, 95)),
           "throughput_fps": thr, "gflops_len32": flops,
           "peak_vram_MB": torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0.0}
    save_json(res, os.path.join(C.RES_DIR, f"bench_{name}_{device}.json"))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})


# ------------------------------------------------------------ orchestrator
def make_samples():
    from transformers import AutoTokenizer
    from dataset import Tokenized, get_columns
    cols, _ = get_columns()
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    ids = Tokenized(tok, cols["test"]).ids[:C.BENCH_N_SAMPLES]
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    path = os.path.join(C.CACHE_DIR, "bench_samples.json")
    json.dump(ids, open(path, "w"))
    return path


def bench_all(sizes, run_cpu):
    sample_path = make_samples()
    devices = (["cuda"] if torch.cuda.is_available() else []) + (["cpu"] if run_cpu else [])
    for name in [C.TEACHER_NAME] + sizes:
        for dv in devices:
            if os.path.exists(os.path.join(C.RES_DIR, f"bench_{name}_{dv}.json")):
                continue
            print(f"-> {name} on {dv}")
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--name", name,
                                "--model", os.path.join(C.CKPT_DIR, name), "--device", dv,
                                "--samples", sample_path], capture_output=True, text=True)
            print("  ", (r.stdout.strip().splitlines() or [""])[-1][:300])
            if r.returncode != 0:
                print("FAILED:", r.stderr[-1000:])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="benchmark teacher + all students on every available device")
    p.add_argument("--sizes", nargs="+", default=C.SIZES, choices=C.SIZES)
    p.add_argument("--no-cpu", action="store_true")
    p.add_argument("--name"); p.add_argument("--model")
    p.add_argument("--device", choices=["cuda", "cpu"])
    p.add_argument("--samples", help="json: list of token-id lists from the test set")
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()
    if a.all:
        bench_all(a.sizes, C.RUN_CPU_BENCH and not a.no_cpu)
    else:
        if not (a.name and a.model and a.device and a.samples):
            p.error("single-model mode needs --name --model --device --samples (or use --all)")
        bench_one(a.name, a.model, a.device, a.samples, a.threads)
