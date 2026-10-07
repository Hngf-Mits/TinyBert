"""benchmark.py - step 3: efficiency + quality benchmark (test split) and the final summary tables.

Each model is benchmarked in its OWN process so that peak-VRAM numbers are meaningful.

Usage:  python benchmark.py all            # benchmark every checkpoint on GPU (+ CPU), one process per model
        python benchmark.py all --no-cpu   # GPU only
        python benchmark.py one --name tinybert_kd_s42 --model checkpoints/tinybert_kd_s42 --device cuda
        python benchmark.py summary        # build summary.csv, kd_gain.csv and results_absa.zip
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from config import BASE_DIR, CHECKPOINT_DIR, CONFLICT_POLICY, RESULTS_DIR, RUN_CPU_BENCH, TEACHER_NAME, TEST_SOURCE
from utils import count_parameters, evaluate_model, get_device, model_size_mb, save_json, set_seed


# ------------------------------------------------------------------ core
def _single_samples(dataloader, n):
    samples = []
    for batch in dataloader:
        for idx in range(batch["input_ids"].size(0)):
            seq_len = int(batch["attention_mask"][idx].sum())
            samples.append({k: v[idx:idx + 1, :seq_len] for k, v in batch.items() if k != "labels"})
            if len(samples) >= n:
                return samples
    return samples


@torch.no_grad()
def benchmark_model(model, test_loader, device, num_warmup=10, num_runs=200):
    """Run with ONLY this model in the process so peak VRAM is meaningful."""
    model.eval()
    use_cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if use_cuda else (lambda: None)
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    quality = evaluate_model(model, test_loader, device)               # 1) quality (test)

    total = 0                                                          # 2) throughput
    sync()
    start = time.perf_counter()
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        model(**inputs)
        total += inputs["input_ids"].size(0)
    sync()
    throughput = total / (time.perf_counter() - start)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_cuda else 0.0

    samples = _single_samples(test_loader, num_runs)                   # 3) latency, batch=1
    for s in samples[:num_warmup]:
        model(**{k: v.to(device) for k, v in s.items()})
    times = []
    for s in samples:
        s = {k: v.to(device) for k, v in s.items()}
        sync()
        t0 = time.perf_counter()
        model(**s)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "params_million": count_parameters(model) / 1e6,
        "weights_mb": model_size_mb(model),
        "accuracy": quality["accuracy"],
        "f1": quality["macro_f1"],
        "f1_per_class": quality["f1_per_class"],
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_median": float(np.median(times)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "throughput_fps": throughput,
        "peak_vram_mb": peak_vram,
        "device": str(device),
    }


# ------------------------------------------------------------ subcommands
def cmd_one(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from dataset import build_dataloaders

    set_seed(42)
    device = torch.device(args.device) if args.device else get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    _, _, test_loader = build_dataloaders(tokenizer, test_source=args.test_source, conflict=args.conflict)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    stats = benchmark_model(model, test_loader, device)
    stats["name"] = args.name
    suffix = "" if device.type == "cuda" else "_cpu"
    path = os.path.join(RESULTS_DIR, f"{args.name}{suffix}.json")
    save_json(stats, path)
    print(path, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items() if k != "f1_per_class"})


def cmd_all(args):
    models = {os.path.basename(os.path.normpath(d)): d for d in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*/")))}
    if not models:
        sys.exit(f"no checkpoints found in {CHECKPOINT_DIR} - train first")
    devices = ["cuda"] if torch.cuda.is_available() else []
    if RUN_CPU_BENCH and not args.no_cpu:
        devices.append("cpu")

    for name, path in models.items():
        for dev in devices:
            print(f"-> benchmarking {name} on {dev} ...")
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--name", name, "--model", path,
                                "--device", dev, "--test-source", TEST_SOURCE, "--conflict", CONFLICT_POLICY],
                               capture_output=True, text=True)
            print((r.stdout.strip().splitlines() or [""])[-1])
            if r.returncode != 0:
                print("FAILED:\n", r.stderr[-1500:])


def cmd_summary(_args):
    import pandas as pd

    stats = json.load(open(os.path.join(RESULTS_DIR, "label_stats.json")))
    majority_acc = max(stats["mapped_label_counts"]["test"].values()) / stats["sizes"]["test"] * 100

    rows = []
    for f in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        d = json.load(open(f))
        if "accuracy" in d and "device" in d:          # benchmark files only
            rows.append(d)
    df = pd.DataFrame(rows)
    df["method"] = df["name"].str.replace(r"_s\d+$", "", regex=True)
    df["seed"] = df["name"].str.extract(r"_s(\d+)$")[0]

    gpu = df[df["device"].str.startswith("cuda")]
    cpu = df[df["device"] == "cpu"][["name", "latency_ms_median"]].rename(columns={"latency_ms_median": "cpu_latency_ms"})
    gpu = gpu.merge(cpu, on="name", how="left")

    summary = gpu.groupby("method").agg(
        runs=("name", "count"), params_M=("params_million", "first"), size_MB=("weights_mb", "first"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        macro_f1_mean=("f1", "mean"), macro_f1_std=("f1", "std"),
        gpu_latency_ms=("latency_ms_median", "mean"), cpu_latency_ms=("cpu_latency_ms", "mean"),
        throughput_fps=("throughput_fps", "mean"), peak_vram_MB=("peak_vram_mb", "mean"),
    ).round(2)

    if TEACHER_NAME in summary.index:
        t = summary.loc[TEACHER_NAME]
        summary["acc_retained_%"] = (summary["acc_mean"] / t["acc_mean"] * 100).round(2)
        summary["size_reduction_%"] = ((1 - summary["size_MB"] / t["size_MB"]) * 100).round(1)
        summary["gpu_speedup_x"] = (t["gpu_latency_ms"] / summary["gpu_latency_ms"]).round(2)
        summary["cpu_speedup_x"] = (t["cpu_latency_ms"] / summary["cpu_latency_ms"]).round(2)

    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"Majority-class accuracy on test: {majority_acc:.1f}%\n")
    print(summary)

    # What does distillation add? Paired by seed: KD minus CE-only (same model, same hyper-parameters).
    sub = gpu[gpu["method"].isin(["tinybert_ce", "tinybert_kd"])]
    if sub["method"].nunique() == 2:
        pv = sub.pivot(index="seed", columns="method", values=["accuracy", "f1"])
        gain = pd.DataFrame({
            "acc_gain_pts": pv["accuracy"]["tinybert_kd"] - pv["accuracy"]["tinybert_ce"],
            "macro_f1_gain_pts": pv["f1"]["tinybert_kd"] - pv["f1"]["tinybert_ce"],
        }).round(2)
        print("\nKD minus CE-only, paired by seed:")
        print(gain)
        print("\nmean / std over seeds:")
        print(gain.agg(["mean", "std"]).round(2))
        gain.to_csv(os.path.join(RESULTS_DIR, "kd_gain.csv"))

    archive = shutil.make_archive(os.path.join(BASE_DIR, "results_absa"), "zip", RESULTS_DIR)
    print(f"\nArchive: {archive}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("one", help="benchmark ONE checkpoint (used internally by 'all')")
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--test-source", default=TEST_SOURCE)
    p.add_argument("--conflict", default=CONFLICT_POLICY)
    p.set_defaults(func=cmd_one)

    p = sub.add_parser("all", help="benchmark every checkpoint, one process per model")
    p.add_argument("--no-cpu", action="store_true")
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("summary", help="aggregate benchmark JSON files into tables")
    p.set_defaults(func=cmd_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
