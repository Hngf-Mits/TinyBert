"""report.py - aggregate results/*.json into summary.csv + plots, and answer "how small can it become?".

Usage:  python report.py [--zip]
"""
import argparse
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from utils import load_json


def main(make_zip=False):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(C.RES_DIR, "*.json"))]
    runs = pd.DataFrame([r for r in rows if r.get("kind") == "run"])
    bench = pd.DataFrame([r for r in rows if r.get("kind") == "bench"])
    teacher_res = load_json(os.path.join(C.RES_DIR, "teacher.json"))
    majority_acc = load_json(os.path.join(C.RES_DIR, "label_stats.json"))["majority_acc"]
    T_ACC = teacher_res["accuracy"]

    # ---- hardware table
    hw = bench.pivot_table(index="name", columns="device",
                           values=["lat_ms_median", "throughput_fps", "peak_vram_MB"], aggfunc="first")
    hw.columns = [f"{'gpu' if d == 'cuda' else 'cpu'}_{v}" for v, d in hw.columns]
    base = bench.drop_duplicates("name").set_index("name")[["params_M", "params_noemb_M", "size_MB", "gflops_len32"]]
    hw = base.join(hw)

    # ---- quality table (mean +- std over seeds)
    agg = (runs[runs["mode"].isin(["ce", "kd"])].groupby(["size", "mode"])
           .agg(n=("seed", "count"), acc=("accuracy", "mean"), acc_std=("accuracy", "std"),
                f1=("macro_f1", "mean"), f1_std=("macro_f1", "std")).reset_index())
    table = hw.loc[[s for s in C.SIZES if s in hw.index]].copy()
    for mode in ("ce", "kd"):
        a = agg[agg["mode"] == mode].set_index("size")
        table[f"{mode}_acc"] = a["acc"]; table[f"{mode}_std"] = a["acc_std"]
        table[f"{mode}_f1"] = a["f1"]; table[f"{mode}_n"] = a["n"]
    table["kd_gain"] = table["kd_acc"] - table["ce_acc"]
    table["retained_%_best"] = table[["ce_acc", "kd_acc"]].max(axis=1) / T_ACC * 100
    table["size_reduction_%"] = (1 - table["size_MB"] / hw.loc[C.TEACHER_NAME, "size_MB"]) * 100
    lat_col = "cpu_lat_ms_median" if "cpu_lat_ms_median" in table.columns else "gpu_lat_ms_median"
    table["speedup_x"] = hw.loc[C.TEACHER_NAME, lat_col] / table[lat_col]
    table = table.sort_values("params_M")
    table.round(2).to_csv(os.path.join(C.RES_DIR, "summary.csv"))

    print(f"teacher: acc {T_ACC:.2f} | {hw.loc['teacher', 'params_M']:.1f}M params | majority floor {majority_acc:.1f}% "
          f"| latency column used for speed-up: {lat_col}")
    show = ["params_M", "params_noemb_M", "gflops_len32", lat_col, "ce_acc", "ce_std", "kd_acc", "kd_std",
            "kd_gain", "ce_n", "retained_%_best", "speedup_x"]
    print(table[show].round(2).to_string())

    # ---- trade-off plot
    order = table.index.tolist()
    xs = {"params_M": "Parameters (M, log scale)",
          lat_col: f"{'CPU' if 'cpu' in lat_col else 'GPU'} latency, batch=1 (ms, log scale)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (xcol, xlabel) in zip(axes, xs.items()):
        for mode, color, mk, lab in (("ce", "tab:blue", "o", "CE only"), ("kd", "tab:red", "s", "KD")):
            ax.errorbar(table[xcol], table[f"{mode}_acc"], yerr=table[f"{mode}_std"].fillna(0),
                        color=color, marker=mk, capsize=3, label=lab)
        for s in order:
            ax.annotate(s.split("_")[1] if s.startswith("bert") else "tinybert", (table.loc[s, xcol], table.loc[s, "kd_acc"]),
                        textcoords="offset points", xytext=(4, 6), fontsize=8)
        ax.scatter([hw.loc[C.TEACHER_NAME, xcol]], [T_ACC], marker="*", s=220, color="gold", edgecolor="k", zorder=5, label="Teacher")
        ax.axhline(T_ACC, color="gold", ls="--", lw=1)
        ax.axhline(0.95 * T_ACC, color="gray", ls=":", lw=1, label="95% of teacher")
        ax.axhline(majority_acc, color="k", ls=":", lw=1, label="majority class")
        ax.set_xscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("Test accuracy (%)"); ax.grid(alpha=.3)
    axes[0].set_title("Quality vs size"); axes[1].set_title("Quality vs speed (Pareto view)"); axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(C.RES_DIR, "tradeoff.png"), dpi=150); plt.close()

    # ---- paired KD gain per size (same seed, same hyper-parameters)
    pv = runs[runs["mode"].isin(["ce", "kd"])].pivot_table(index=["size", "seed"], columns="mode",
                                                           values=["accuracy", "macro_f1"]).dropna()
    gain = pd.DataFrame({"acc_gain_pts": pv["accuracy"]["kd"] - pv["accuracy"]["ce"],
                         "macro_f1_gain_pts": pv["macro_f1"]["kd"] - pv["macro_f1"]["ce"]}).round(2)
    gain.to_csv(os.path.join(C.RES_DIR, "kd_gain.csv"))       # one row per (size, seed)
    g = gain.groupby("size")["acc_gain_pts"].agg(["mean", "std", "count"]).reindex(order)
    print("\nKD - CE accuracy gain per size (paired by seed):\n", g.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(range(len(g)), g["mean"], yerr=g["std"].fillna(0), capsize=4,
           color=["tab:green" if v > 0 else "tab:red" for v in g["mean"]])
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{s}\n{table.loc[s, 'params_M']:.1f}M" for s in g.index], fontsize=7)
    ax.set_ylabel("KD - CE accuracy (points)"); ax.set_title("Does distillation help, and for which sizes?")
    ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(C.RES_DIR, "kd_gain.png"), dpi=150); plt.close()

    # ---- "how small can it become?"
    print()
    for mode in ("ce", "kd"):
        ok = table[table[f"{mode}_acc"] >= 0.95 * T_ACC]
        if len(ok):
            s = ok["params_M"].idxmin()
            print(f"[{mode.upper()}] smallest size keeping >=95% of teacher accuracy: {s} ({table.loc[s, 'params_M']:.1f}M params, "
                  f"{table.loc[s, 'params_noemb_M']:.2f}M non-embedding, {table.loc[s, 'size_reduction_%']:.0f}% smaller, "
                  f"{table.loc[s, 'speedup_x']:.1f}x faster)")
        else:
            print(f"[{mode.upper()}] no student reaches 95% of the teacher accuracy")

    if make_zip:
        shutil.make_archive(os.path.join(C.BASE_DIR, "results_sst2"), "zip", C.RES_DIR)
        print("saved:", os.path.join(C.BASE_DIR, "results_sst2.zip"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip", action="store_true", help="also zip results/")
    main(p.parse_args().zip)
