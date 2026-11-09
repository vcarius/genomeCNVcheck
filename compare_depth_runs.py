#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_depth_runs.py — compare two analyze_cnv.py runs that differ in ONE setting
(e.g. supplementary alignments included vs excluded), window by window.

Usage
  python3 compare_depth_runs.py <data_dir_A> <data_dir_B> -o <out_prefix> [--label-a with_supp --label-b without_supp]

Reads window_cnv_calls.tsv and run_parameters.tsv from each data dir. Outputs
  <out>.summary.txt          baselines, CN-band shares, per-chromosome medians, depth lost from A to B
  <out>.per_chromosome.tsv   per chromosome: median CN in A and B, median fraction of depth lost
  <out>.pdf / .png           (1) relative CN density A vs B, (2) raw depth density A vs B,
                             (3) fraction of depth lost per window vs CN in A, (4) per-chromosome loss
Reading: if A reproduces the earlier single-peak picture and B the two-peak picture, the setting
alone explains the change. 'lost' = 1 - depth_B / depth_A (for supplementary: the share of the
depth carried by supplementary alignments).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

BANDS = [(-np.inf, 0.25, "<0.25"), (0.25, 0.75, "0.25-0.75"), (0.75, 1.25, "0.75-1.25"),
         (1.25, 1.75, "1.25-1.75"), (1.75, 2.5, "1.75-2.5"), (2.5, np.inf, ">2.5")]


def load(d):
    w = pd.read_csv(os.path.join(d, "window_cnv_calls.tsv"), sep="\t", dtype={"chrom": str}, na_values=["NA"])
    p = pd.read_csv(os.path.join(d, "run_parameters.tsv"), sep="\t").set_index("parameter")["value"]
    return w, float(p["baseline_depth"])


def density(ax, x, label, color, xmax):
    x = x[np.isfinite(x) & (x >= 0) & (x <= xmax)]
    xs = np.linspace(0, xmax, 600)
    ax.plot(xs, gaussian_kde(x)(xs), color=color, lw=2, label=label)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("-o", "--out-prefix", required=True)
    ap.add_argument("--label-a", default="with_supp")
    ap.add_argument("--label-b", default="without_supp")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)) or ".", exist_ok=True)

    wa, base_a = load(a.dir_a)
    wb, base_b = load(a.dir_b)
    m = wa.merge(wb, on=["chrom", "start", "end"], suffixes=("_a", "_b"))
    with np.errstate(divide="ignore", invalid="ignore"):
        m["lost"] = 1 - m["depth_eff_b"] / m["depth_eff_a"]
    m.loc[m["depth_eff_a"] <= 0, "lost"] = np.nan
    A, B = a.label_a, a.label_b

    L = ["=" * 78, f" {A} (A) vs {B} (B): {len(m)} shared windows "
                   f"({len(wa)} in A, {len(wb)} in B)", "=" * 78,
         f"Disomic baseline: A = {base_a:.1f}x | B = {base_b:.1f}x | B/A = {base_b / base_a:.3f}",
         f"Median raw depth: A = {m['depth_a'].median():.1f} | B = {m['depth_b'].median():.1f}",
         f"Depth lost from A to B per window: median {m['lost'].median():.3f}, "
         f"IQR {m['lost'].quantile(0.25):.3f}-{m['lost'].quantile(0.75):.3f}",
         f"Windows with depth > 0 in A but 0 in B: {int(((m['depth_a'] > 0) & (m['depth_b'] == 0)).sum())}",
         "-" * 78, "Share of windows per relative-CN band (CN relative to each run's own baseline):"]
    for lo, hi, lab in BANDS:
        fa = ((m["copy_number_a"] >= lo) & (m["copy_number_a"] < hi)).mean() * 100
        fb = ((m["copy_number_b"] >= lo) & (m["copy_number_b"] < hi)).mean() * 100
        L.append(f"  CN {lab:10s}  A {fa:5.1f}%   B {fb:5.1f}%")
    L.append("Is the loss uniform? (if yes, B is just A rescaled and the shape cannot change)")
    for lo, hi, lab in BANDS[1:5]:
        sel = (m["copy_number_a"] >= lo) & (m["copy_number_a"] < hi)
        if sel.any():
            L.append(f"  windows with CN_A {lab:10s}: median depth lost {m.loc[sel, 'lost'].median():.3f} (n={int(sel.sum())})")

    pc = m.groupby("chrom").agg(median_cn_a=("copy_number_a", "median"), median_cn_b=("copy_number_b", "median"),
                                median_depth_lost=("lost", "median"), n_windows=("lost", "size"))
    pc = pc.sort_values("median_depth_lost", ascending=False).round(3)
    pc.to_csv(f"{a.out_prefix}.per_chromosome.tsv", sep="\t")
    L += ["-" * 78, "Chromosomes losing the most depth from A to B:", pc.head(10).to_string(),
          "Chromosomes losing the least:", pc.tail(5).to_string(), "=" * 78]
    txt = "\n".join(L)
    print(txt)
    with open(f"{a.out_prefix}.summary.txt", "w") as fh:
        fh.write(txt + "\n")

    fig, ax = plt.subplots(1, 4, figsize=(24, 5.5))
    density(ax[0], m["copy_number_a"].to_numpy(), A, "#2f6db5", 3)
    density(ax[0], m["copy_number_b"].to_numpy(), B, "#d9772b", 3)
    for k in (0.5, 1.0, 1.5):
        ax[0].axvline(k, color="grey", ls=":", lw=1)
    ax[0].set_xlabel("Relative CN (each run's own baseline = 1.0)")
    ax[0].set_ylabel("Density")
    ax[0].set_title("1. Shape of the distribution", loc="left", fontweight="bold")
    ax[0].legend()

    xmax = float(np.nanquantile(m["depth_gc_a"], 0.98))
    density(ax[1], m["depth_gc_a"].to_numpy(), f"{A} (baseline {base_a:.0f}x)", "#2f6db5", xmax)
    density(ax[1], m["depth_gc_b"].to_numpy(), f"{B} (baseline {base_b:.0f}x)", "#d9772b", xmax)
    ax[1].set_xlabel("Depth (GC-corrected)")
    ax[1].set_title("2. Absolute depth", loc="left", fontweight="bold")
    ax[1].legend()

    ok = m["lost"].notna() & m["copy_number_a"].between(0, 3)
    hb = ax[2].hexbin(m.loc[ok, "copy_number_a"], m.loc[ok, "lost"].clip(-0.2, 1), gridsize=60, bins="log", cmap="viridis")
    fig.colorbar(hb, ax=ax[2], label="windows (log)")
    ax[2].set_xlabel(f"Relative CN in {A}")
    ax[2].set_ylabel(f"Fraction of depth lost in {B}")
    ax[2].set_title("3. Where the depth is lost", loc="left", fontweight="bold")

    top = pc["median_depth_lost"].sort_values()
    ax[3].barh(range(len(top)), top.values, color="#7a8392")
    ax[3].set_yticks(range(len(top)))
    ax[3].set_yticklabels(top.index, fontsize=6)
    ax[3].axvline(m["lost"].median(), color="black", ls="--", lw=1, label="genome median")
    ax[3].set_xlabel("Median fraction of depth lost")
    ax[3].set_title("4. Per chromosome", loc="left", fontweight="bold")
    ax[3].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{a.out_prefix}.pdf")
    plt.savefig(f"{a.out_prefix}.png", dpi=200)
    print(f"[OK] {a.out_prefix}.summary.txt, .per_chromosome.tsv, .pdf, .png")


if __name__ == "__main__":
    main()
