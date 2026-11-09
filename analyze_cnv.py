#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_cnv.py — copy-number (CN) estimation from long-read depth.

Designed for T. cruzi CL Brener reads mapped to the haploid Esmeraldo-like reference: both
haplotypes stack on the same sequence, so the dominant depth peak is the DISOMIC state
(relative CN 1.0 = 2 copies per cell; 1.5 = 3 copies; 0.5 = 1 copy).

Steps
  1. Windows with > --max-n N content are dropped; depth is rescaled to effective (non-N) bases.
     Windows with sequence but zero depth are kept (homozygous losses remain visible).
  2. GC correction: depth divided by the median depth of its GC bin (GC computed on ACGT only),
     each bin pooled with neighbours until it holds >= --min-gc-bin windows.
  3. Disomic baseline = dominant peak of a KDE of the corrected depth; the baseline obtained
     with other bandwidths is reported as a sensitivity check.
  4. Optional low-mappability mask (--compare: depth from a stricter-MAPQ pass).
  5. Segmentation: Gaussian HMM on log2(CN) with fixed CN states (Viterbi), then segments
     shorter than --min-seg-bp are merged into the most similar neighbour. States are called
     per SEGMENT (median CN), never per window.
  6. Per-chromosome summary with block-bootstrap CI of the median (windows are autocorrelated).

Usage
  python3 analyze_cnv.py <data_dir> <plot_dir> [options]

Inputs in <data_dir>
  coverage.regions.bed.gz   mosdepth --by output
  windows_gc.tsv            chrom start end gc_acgt frac_N (with header; written by run_cnv.sh)

Outputs (<suffix> from --suffix, empty by default)
  <data_dir>/window_cnv_calls<suffix>.tsv            one row per analysed window (header)
  <data_dir>/cnv_segments<suffix>.bed                segments; header line starts with '#'
  <data_dir>/chromosome_ploidy_summary<suffix>.tsv   per-chromosome summary
  <data_dir>/baseline_sensitivity<suffix>.tsv        baseline under different KDE bandwidths
  <data_dir>/run_parameters<suffix>.tsv              parameters and key estimates
  <plot_dir>/coverage_distribution<suffix>.pdf, cnv_manhattan_segments<suffix>.pdf
"""
import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# Relative CN states (1.0 = 2 copies) used by the HMM and for labelling
STATES = OrderedDict([
    ("loss_0copies", 0.05),
    ("loss_1copy",   0.50),
    ("disomic",      1.00),
    ("gain_3copies", 1.50),
    ("gain_4copies", 2.00),
    ("gain_5copies", 2.50),
    ("amplified",    4.00),   # broad state for collapsed multigene families / arrays
])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir")
    p.add_argument("plot_dir")
    p.add_argument("--max-n", type=float, default=0.10, help="Maximum N fraction per window (default 0.10)")
    p.add_argument("--bw", default="scott", help="KDE bandwidth: 'scott', 'silverman' or a number (default scott)")
    p.add_argument("--min-gc-bin", type=int, default=50, help="Minimum windows per GC bin (default 50)")
    p.add_argument("--switch-prob", type=float, default=1e-6,
                   help="HMM per-window probability of changing state (default 1e-6)")
    p.add_argument("--min-seg-bp", type=int, default=25000,
                   help="Segments shorter than this (analysed bp) are merged into a neighbour (default 25000)")
    p.add_argument("--compare", default=None,
                   help="coverage.regions.bed.gz from a stricter-MAPQ pass, to flag low-mappability windows")
    p.add_argument("--mappability-log2", type=float, default=0.5,
                   help="|log2(depth ratio main/strict)| above this = low mappability (default 0.5)")
    p.add_argument("--suffix", default="", help="Suffix for all output files (e.g. _focal)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.median(np.abs(x - np.median(x))) if x.size else np.nan


def block_bootstrap_ci(x, block=50, n_boot=1000, alpha=0.05, seed=42):
    """CI of the median resampling contiguous blocks (respects autocorrelation between windows)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.nan, np.nan
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        boots[i] = np.median(np.concatenate([x[s:s + block] for s in starts])[:n])
    return np.quantile(boots, alpha / 2), np.quantile(boots, 1 - alpha / 2)


def safe_chrom_label(chrom):
    return chrom.replace("ptg0000", "ptg_").replace("ntLink_", "ntL_")


def kde_peak(values, bw):
    values = values[np.isfinite(values) & (values > 0)]
    lo, hi = np.quantile(values, [0.005, 0.95])
    v = values[(values >= lo) & (values <= hi)]
    try:
        bw_method = float(bw)
    except ValueError:
        bw_method = bw
    kde = gaussian_kde(v, bw_method=bw_method)
    xs = np.linspace(0, hi, 2000)
    ys = kde(xs)
    peaks, props = find_peaks(ys, prominence=ys.max() * 0.05)
    if len(peaks) == 0:
        return float(np.median(v)), xs, ys
    return float(xs[peaks[np.argmax(props["prominences"])]]), xs, ys


def gc_correct(df, min_bin):
    """Divide depth by the median depth of its GC bin (only windows with depth > 0 define the
    medians). A bin with fewer than min_bin windows is widened symmetrically until it has enough."""
    d = df.copy()
    d["gc_bin"] = (d["gc_acgt"] * 100).round().clip(0, 100).astype(int)
    ref = d.loc[d["depth_eff"] > 0, ["gc_bin", "depth_eff"]]
    by_bin = {b: g["depth_eff"].values for b, g in ref.groupby("gc_bin")}
    medians = {}
    for b in range(101):
        width, vals = 0, []
        while True:
            vals = np.concatenate([by_bin.get(x, np.array([])) for x in range(b - width, b + width + 1)])
            if vals.size >= min_bin or width > 50:
                break
            width += 1
        medians[b] = np.median(vals) if vals.size else np.nan
    factor = d["gc_bin"].map(medians) / ref["depth_eff"].median()
    d["depth_gc"] = d["depth_eff"] / factor
    bad = ~np.isfinite(d["depth_gc"])
    d.loc[bad, "depth_gc"] = d.loc[bad, "depth_eff"]
    return d


def viterbi_segment(log2cn, sigma, switch_prob):
    """Gaussian HMM on log2(CN) with fixed states (STATES). Returns the state index per window."""
    mu = np.log2(np.array(list(STATES.values())))
    K = len(mu)
    sig = np.full(K, sigma)
    sig[-1] = max(sigma, 0.6)          # broad 'amplified' state
    sig[0] = max(sigma, 1.0)           # broad ~0-copy state (log2 of values near 0 is unstable)
    x = np.clip(log2cn, -6, 6)
    ll = -0.5 * ((x[:, None] - mu[None, :]) / sig[None, :]) ** 2 - np.log(sig[None, :])
    trans = np.full((K, K), np.log(switch_prob / (K - 1)))
    np.fill_diagonal(trans, np.log1p(-switch_prob))
    n = len(x)
    dp = np.empty((n, K))
    bp = np.zeros((n, K), dtype=int)
    dp[0] = ll[0] - np.log(K)
    for i in range(1, n):
        cand = dp[i - 1][:, None] + trans
        bp[i] = np.argmax(cand, axis=0)
        dp[i] = cand[bp[i], np.arange(K)] + ll[i]
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    return path


def runs(path):
    """State vector -> list of [start, end_exclusive, state]."""
    out, s = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[s]:
            out.append([s, i, path[s]])
            s = i
    return out


def merge_short(segs, cn, min_w):
    """Repeatedly merge the shortest segment (< min_w windows) into the neighbour with the closest
    median CN; finally merge adjacent segments that ended up with the same state."""
    segs = [list(s) for s in segs]
    while len(segs) > 1:
        lens = [e - s for s, e, _ in segs]
        i = int(np.argmin(lens))
        if lens[i] >= min_w:
            break
        s, e, _ = segs[i]
        med = np.nanmedian(cn[s:e])
        cands = []
        if i > 0:
            ps, pe, _ = segs[i - 1]
            cands.append((abs(np.nanmedian(cn[ps:pe]) - med), i - 1))
        if i < len(segs) - 1:
            ns, ne, _ = segs[i + 1]
            cands.append((abs(np.nanmedian(cn[ns:ne]) - med), i + 1))
        _, j = min(cands)
        a, b = sorted([i, j])
        segs[a] = [segs[a][0], segs[b][1], segs[j][2]]
        del segs[b]
    merged = [segs[0]]
    for s in segs[1:]:
        if s[2] == merged[-1][2]:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


def nearest_state(cn_value):
    if not np.isfinite(cn_value):
        return np.nan
    if cn_value >= 3.0:
        return "amplified"
    names = list(STATES.keys())
    vals = np.array(list(STATES.values()))
    return names[int(np.argmin(np.abs(np.log2(max(cn_value, 0.05)) - np.log2(vals))))]


def chrom_label(cn):
    if not np.isfinite(cn):
        return "NA"
    for target, label in [(1.0, "disomic (2 copies)"), (0.5, "monosomic (1 copy)"),
                          (1.5, "trisomic (3 copies)"), (2.0, "tetrasomic (4 copies)")]:
        if abs(cn - target) <= 0.2:
            return label
    return "amplified" if cn > 2.25 else "intermediate/mosaic"


def load_coverage(path):
    return pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end", "depth"],
                       compression="infer", dtype={"chrom": str})


def main():
    a = parse_args()
    os.makedirs(a.plot_dir, exist_ok=True)
    sfx = a.suffix
    out = lambda name, ext, d=a.data_dir: os.path.join(d, f"{name}{sfx}.{ext}")

    print("[PY] Loading depth and GC...")
    cov = load_coverage(os.path.join(a.data_dir, "coverage.regions.bed.gz"))
    gc = pd.read_csv(os.path.join(a.data_dir, "windows_gc.tsv"), sep="\t", na_values=["NA"], dtype={0: str})
    if not {"gc_acgt", "frac_N"}.issubset(gc.columns):
        sys.exit("[ERROR] windows_gc.tsv lacks gc_acgt/frac_N columns: run the updated run_cnv.sh.")
    gc = gc.rename(columns={gc.columns[0]: "chrom", gc.columns[1]: "start", gc.columns[2]: "end"})

    df = cov.merge(gc, on=["chrom", "start", "end"], how="left", validate="one_to_one")
    n_nogc = int(df["frac_N"].isna().sum())
    if n_nogc:
        print(f"[WARN] {n_nogc} windows without GC/N (contigs missing from the FASTA?); dropped.")
    n_total = len(df)
    df = df[df["frac_N"].notna() & (df["frac_N"] <= a.max_n) & df["gc_acgt"].notna()].copy()
    print(f"[PY] Windows: {n_total} total; {n_total - len(df)} dropped (N > {a.max_n:.0%} or no GC); "
          f"{len(df)} analysed.")
    df["depth_eff"] = df["depth"] / (1.0 - df["frac_N"])

    print("[PY] GC correction...")
    df = gc_correct(df, a.min_gc_bin)

    print("[PY] Disomic baseline from the dominant KDE peak...")
    base, xs, ys = kde_peak(df["depth_gc"].values, a.bw)
    sens = []
    for bw in ["0.01", "0.05", "scott", "silverman"]:
        b, _, _ = kde_peak(df["depth_gc"].values, bw)
        sens.append({"bw": bw, "baseline_depth": round(b, 2), "rel_diff_vs_chosen": round((b - base) / base, 4)})
    sens = pd.DataFrame(sens)
    sens.to_csv(out("baseline_sensitivity", "tsv"), sep="\t", index=False)
    print(f"[PY] Baseline (relative CN 1.0 = 2 copies): {base:.1f}x | bandwidth sensitivity: "
          + ", ".join(f"{r.bw}={r.baseline_depth:.1f}x ({r.rel_diff_vs_chosen:+.1%})" for r in sens.itertuples()))

    df["copy_number"] = df["depth_gc"] / base
    df["log2_cn"] = np.log2(np.clip(df["copy_number"], 0.01, None))

    df["low_mappability"] = False
    if a.compare:
        cmp_ = load_coverage(a.compare).rename(columns={"depth": "depth_cmp"})
        df = df.merge(cmp_, on=["chrom", "start", "end"], how="left")
        cmp_eff = df["depth_cmp"] / (1.0 - df["frac_N"])
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log2((df["depth_eff"] / df["depth_eff"].median()) / (cmp_eff / cmp_eff.median()))
        df["low_mappability"] = ~np.isfinite(lr) | (np.abs(lr) > a.mappability_log2)
        df = df.drop(columns=["depth_cmp"])
        print(f"[PY] Low-mappability windows: {int(df['low_mappability'].sum())} ({df['low_mappability'].mean():.1%})")

    # ------------------------------------------------------------------ segmentation
    print("[PY] HMM segmentation...")
    sigma = 1.4826 * mad(df.loc[np.abs(df["log2_cn"]) < 0.5, "log2_cn"])
    print(f"[PY] Robust SD of log2(CN) near the baseline: {sigma:.3f}")
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    win = int((df["end"] - df["start"]).median())
    min_w = max(1, int(round(a.min_seg_bp / win)))
    print(f"[PY] Typical window {win} bp -> minimum segment {min_w} windows")

    df["seg_id"] = -1
    seg_rows, seg_counter = [], 0
    for chrom, idx in df.groupby("chrom").groups.items():
        idx = np.asarray(sorted(idx))
        sub = df.loc[idx]
        use = ~sub["low_mappability"].values
        if use.sum() < 2:
            print(f"[WARN] {chrom}: fewer than 2 usable windows; not segmented.")
            continue
        path = np.full(len(sub), -1)
        path[use] = viterbi_segment(sub["log2_cn"].values[use], sigma, a.switch_prob)
        # masked windows inherit the state of the previous (or next) usable window
        path = pd.Series(np.where(path >= 0, path, np.nan)).ffill().bfill().astype(int).values
        cn = sub["copy_number"].values
        for s0, e0, _ in merge_short(runs(path), cn, min_w):
            rows = idx[s0:e0]
            ok = ~df.loc[rows, "low_mappability"].values
            vals = df.loc[rows, "copy_number"].values[ok]
            seg_cn = float(np.median(vals)) if vals.size else np.nan
            df.loc[rows, "seg_id"] = seg_counter
            seg_rows.append(OrderedDict([
                ("chrom", chrom),
                ("start", int(df.loc[rows[0], "start"])),
                ("end", int(df.loc[rows[-1], "end"])),
                ("seg_id", seg_counter),
                ("n_windows", len(rows)),
                ("bp_analyzed", int((df.loc[rows, "end"] - df.loc[rows, "start"]).sum())),
                ("median_cn", round(seg_cn, 3) if np.isfinite(seg_cn) else np.nan),
                ("copies_est", round(2 * seg_cn, 2) if np.isfinite(seg_cn) else np.nan),
                ("state", nearest_state(seg_cn)),
                ("frac_low_mappability", round(float(1 - ok.mean()), 3)),
            ]))
            seg_counter += 1
    segs = pd.DataFrame(seg_rows)
    seg_map = segs.set_index("seg_id")
    df["seg_cn"] = df["seg_id"].map(seg_map["median_cn"])
    df["seg_state"] = df["seg_id"].map(seg_map["state"])

    seg_path = out("cnv_segments", "bed")
    segs.rename(columns={"chrom": "#chrom"}).to_csv(seg_path, sep="\t", index=False, na_rep="NA")
    tot = segs["bp_analyzed"].sum()
    print("[PY] Segments by state (analysed Mb, % of analysed sequence; N gaps excluded):")
    for st, g in segs.groupby("state"):
        L = g["bp_analyzed"].sum()
        print(f"       {st:15s} {L / 1e6:7.2f} Mb  {L / tot:6.1%}  ({len(g)} segments)")

    # ------------------------------------------------------------------ per-chromosome summary
    summ = []
    for chrom, sub in df.groupby("chrom"):
        cn_ok = sub.loc[~sub["low_mappability"], "copy_number"].values
        med = float(np.median(cn_ok)) if len(cn_ok) else np.nan
        lo, hi = block_bootstrap_ci(cn_ok, block=50, seed=a.seed)
        cs = segs[segs["chrom"] == chrom]
        L = cs["bp_analyzed"]
        non_disomic = L[cs["state"] != "disomic"].sum() / L.sum() if L.sum() else np.nan
        summ.append(OrderedDict([
            ("chrom", chrom), ("chrom_label", safe_chrom_label(chrom)),
            ("n_windows", len(sub)),
            ("n_windows_zero_depth", int((sub["depth"] == 0).sum())),
            ("n_windows_low_mappability", int(sub["low_mappability"].sum())),
            ("median_depth_gc", float(np.median(sub["depth_gc"]))),
            ("median_copy_number", med),
            ("copies_est", 2 * med),
            ("mad_copy_number", mad(cn_ok)),
            ("copy_number_ci_low", lo), ("copy_number_ci_high", hi),
            ("n_segments", len(cs)),
            ("frac_length_non_disomic", non_disomic),
            ("classification", chrom_label(med)),
        ]))
    pd.DataFrame(summ).sort_values("median_copy_number", ascending=False) \
        .to_csv(out("chromosome_ploidy_summary", "tsv"), sep="\t", index=False)

    cols = ["chrom", "start", "end", "depth", "depth_eff", "depth_gc", "gc_acgt", "frac_N",
            "copy_number", "log2_cn", "low_mappability", "seg_id", "seg_cn", "seg_state"]
    df[cols].to_csv(out("window_cnv_calls", "tsv"), sep="\t", index=False, na_rep="NA")

    pd.DataFrame([
        ("max_n", a.max_n), ("bw", a.bw), ("min_gc_bin", a.min_gc_bin), ("switch_prob", a.switch_prob),
        ("min_seg_bp", a.min_seg_bp), ("compare", a.compare or ""), ("mappability_log2", a.mappability_log2),
        ("window_bp", win), ("windows_analysed", len(df)), ("baseline_depth", round(base, 2)),
        ("sigma_log2", round(sigma, 4)), ("n_segments", len(segs)),
        ("frac_non_disomic", round(segs.loc[segs["state"] != "disomic", "bp_analyzed"].sum() / tot, 4)),
    ], columns=["parameter", "value"]).to_csv(out("run_parameters", "tsv"), sep="\t", index=False)
    print(f"[PY] Outputs: {out('window_cnv_calls', 'tsv')}, {seg_path}, {out('chromosome_ploidy_summary', 'tsv')}")

    # ------------------------------------------------------------------ plots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["depth_gc"], bins=100, range=(0, df["depth_gc"].quantile(0.98)), density=True,
            alpha=0.5, color="gray", label="Depth (GC-corrected, N-adjusted)")
    ax.plot(xs, ys, color="#b4581a", lw=2, label=f"KDE (bw={a.bw})")
    ax.axvline(base, color="#2f6db5", ls="--", label=f"Disomic baseline, 2 copies ({base:.1f}x)")
    for k, lab in [(0.5, "1 copy"), (1.5, "3 copies"), (2.0, "4 copies")]:
        ax.axvline(base * k, color="#7a8392", ls=":", lw=1)
        ax.text(base * k, ax.get_ylim()[1] * 0.95, lab, rotation=90, va="top", ha="right", fontsize=8, color="#4a5463")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Density")
    ax.set_title("Global depth distribution")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out("coverage_distribution", "pdf", a.plot_dir))
    plt.close()

    # Same distribution, stacked by the CN state of each window's segment: shows which states
    # build the main peak, the left flank (fewer copies) and the right tail (gains / amplifications).
    state_order = [s for s in ["loss_0copies", "loss_1copy", "disomic", "gain_3copies", "gain_4copies",
                               "gain_5copies", "amplified"] if s in set(df["seg_state"].dropna())]
    state_col = {"loss_0copies": "#7f2704", "loss_1copy": "#d9772b", "disomic": "#2f6db5",
                 "gain_3copies": "#74a9cf", "gain_4copies": "#9e9ac8", "gain_5copies": "#756bb1",
                 "amplified": "#54278f"}
    hi = df["depth_gc"].quantile(0.98)
    bins = np.linspace(0, hi, 101)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for k, yscale in enumerate(["linear", "log"]):
        ax = axs[k]
        data = [df.loc[df["seg_state"] == s, "depth_gc"].clip(upper=hi) for s in state_order]
        labels = [f"{s} ({100 * (df['seg_state'] == s).mean():.1f}% of windows)" for s in state_order]
        ax.hist(data, bins=bins, stacked=True, color=[state_col[s] for s in state_order], label=labels)
        ax.axvline(base, color="black", ls="--", lw=1)
        for m in (0.5, 1.5, 2.0):
            ax.axvline(base * m, color="#7a8392", ls=":", lw=1)
        ax.set_yscale(yscale)
        ax.set_xlabel("Depth (GC-corrected, N-adjusted)")
        ax.set_ylabel("Windows" + (" (log scale)" if yscale == "log" else ""))
        ax.set_title(f"Depth by segment CN state ({yscale} y axis); dashed = disomic baseline {base:.1f}x",
                     fontsize=10)
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out("coverage_distribution_by_state", "pdf", a.plot_dir))
    plt.close()

    y_max = max(3.5, df["copy_number"].quantile(0.995) + 0.5)
    fig, ax = plt.subplots(figsize=(16, 6))
    order = df.groupby("chrom")["end"].max().sort_values(ascending=False).index.tolist()
    offset, ticks, labels = 0, [], []
    state_color = {"disomic": "#2f6db5", "loss_1copy": "#d9772b", "loss_0copies": "#8c2d04",
                   "gain_3copies": "#b4581a", "gain_4copies": "#b4581a", "gain_5copies": "#b4581a",
                   "amplified": "#7b1fa2"}
    for i, chrom in enumerate(order):
        sub = df[df["chrom"] == chrom]
        ax.scatter(sub["start"] + offset, sub["copy_number"], s=2, alpha=0.4,
                   color=["#2c3e50", "#95a5a6"][i % 2], edgecolors="none", rasterized=True)
        for r in segs[segs["chrom"] == chrom].itertuples():
            if np.isfinite(r.median_cn):
                ax.hlines(r.median_cn, r.start + offset, r.end + offset,
                          color=state_color.get(r.state, "red"), lw=2.6, zorder=4)
        size = sub["end"].max()
        ticks.append(offset + size / 2)
        labels.append(safe_chrom_label(chrom))
        offset += size
    for k in [0.5, 1.0, 1.5, 2.0]:
        ax.axhline(k, color="black", ls="--", lw=1, alpha=0.6 if k == 1.0 else 0.25)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, offset)
    ax.set_ylabel("Relative copy number (1.0 = 2 copies)")
    sec = ax.secondary_yaxis("right", functions=(lambda y: 2 * y, lambda y: y / 2))
    sec.set_ylabel("Estimated copies per cell")
    ax.set_title(f"Copy number vs. Esmeraldo-like reference (disomic baseline {base:.1f}x); lines = HMM segments")
    plt.tight_layout()
    plt.savefig(out("cnv_manhattan_segments", "pdf", a.plot_dir), dpi=300)
    plt.close()
    print("[PY] Done.")


if __name__ == "__main__":
    main()
