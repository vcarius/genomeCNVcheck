#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_events_by_type.py — molecules per locus by event type (origins / IZD, terminations / TZD) and CN state,
raw and normalised by copy number.

Panel E is one pooled rate per CN state, so it has no distribution to show as a boxplot. The per-locus
view does: for every locus (molecules of one type merged within --merge-distance, as in the paper),
  raw         = number of molecules
  normalised  = molecules x (disomic depth / local depth)  ->  molecules expected at disomic coverage
A zone (IZD/TZD) whose count is only high because the segment has more copies falls below the threshold
after normalisation.

INPUT
  -b  BED(s) of molecules: chrom start end type (one line per molecule; type e.g. origins, IZ, terminations, TZ)
  -w  window_cnv_calls.tsv from analyze_cnv.py (CN state and depth of each 1 kb window)
  -a  <prefix>_annotated.tsv from verifyRegionsSummary_V3.py: keeps only RETAINED molecules, as panels D/E

HOW EACH LOCUS IS COUNTED
  1. each molecule is resized to 2 kb around its midpoint (--standardize-bp), as in the paper
  2. each molecule takes the CN state of the window holding its midpoint (same counts as panels D/E)
  3. molecules of the same type within 1 kb are merged into loci (--merge-distance, = bedtools merge -d)
  4. raw = molecules in the locus (or only in its central --fixed-window bp)
     normalised = raw x disomic depth / local depth (molecules expected at disomic coverage)
  5. a locus is a zone when the count is > --threshold (more than 10: the paper's 10 IZD / 18 TZD)

RECOMMENDED RUN (MAPQ >= 20 depth, loci split at CN boundaries, counts over the central 2 kb = the IZD)
  python3 plot_events_by_type.py -b OriginsClassified.bed -w window_cnv_calls.tsv -a ORI_annotated.tsv \\
      --depth-bed coverage_mapq20.regions.bed.gz --split-at-cn-boundary --fixed-window 2000 \\
      --types origins IZ -o ORI_by_type

OUTPUT
  <o>_loci.tsv     one row per locus: type, molecules, CN state, local depth, span_bp, normalised molecules
                   (with --fixed-window, molecules over the whole merged span are in molecules_whole_span)
  <o>_summary.tsv  per type x CN state: molecules (each molecule's own midpoint, = panels D/E), loci,
                   molecules_in_loci, median raw / normalised, loci > threshold raw / normalised
  <o>.pdf / .png   boxplots (raw | normalised), log scale, threshold line
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATE_ORDER = ["loss_0copies", "loss_1copy", "disomic", "gain_3copies", "gain_4copies", "gain_5copies", "amplified"]
STATE_LABEL = {"loss_0copies": "0", "loss_1copy": "1 copy", "disomic": "2 (disomic)", "gain_3copies": "3 copies",
               "gain_4copies": "4 copies", "gain_5copies": "5 copies", "amplified": "amplified"}
PALETTE = ["#2F6DB5", "#C0392B", "#D9772B", "#7B3294", "#4A5463", "#1B9E77"]


def read_typed_bed(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.split()
                if len(f) < 4:
                    sys.exit(f"[ERROR] {p}: need 4 columns (chrom start end type): {line.strip()}")
                rows.append((f[0], int(f[1]), int(f[2]), f[3]))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "type"])


def merge_loci(ev, dist, split_cn=False):
    """Merge molecules of one type within dist (bedtools merge -d). With split_cn, a locus never spans two CN
    states (the CN state is part of the merge key)."""
    keys = ["type", "chrom"] + (["mol_state"] if split_cn else [])
    out = []
    for kv, g in ev.sort_values(keys + ["start", "end"]).groupby(keys, sort=False):
        t, c = kv[0], kv[1]
        cs = ce = None
        sts = []
        for s, e, ms in zip(g["start"].to_numpy(), g["end"].to_numpy(), g["mol_state"].to_numpy()):
            if cs is not None and s - ce <= dist:
                ce = max(ce, e)
                sts.append(ms)
            else:
                if cs is not None:
                    out.append((c, cs, ce, t, len(sts), "|".join(sts)))
                cs, ce, sts = int(s), int(e), [ms]
        out.append((c, cs, ce, t, len(sts), "|".join(sts)))
    loci = pd.DataFrame(out, columns=["chrom", "start", "end", "type", "molecules", "mol_states"])
    return loci.sort_values(["type", "chrom", "start"]).reset_index(drop=True)


def annotate(loci, w, col):
    depth = np.full(len(loci), np.nan)
    state = np.full(len(loci), "NA", dtype=object)
    for c, idx in loci.groupby("chrom").groups.items():
        g = w[w["chrom"] == c]
        if g.empty:
            continue
        st, en = g["start"].to_numpy(), g["end"].to_numpy()
        dv, sv = g[col].to_numpy(dtype=float), g["seg_state"].to_numpy(dtype=object)
        for k in np.asarray(idx):
            s, e = loci.at[k, "start"], loci.at[k, "end"]
            i0 = max(np.searchsorted(en, s, side="right"), 0)
            i1 = np.searchsorted(st, e, side="left")
            if i1 > i0:
                ov = np.minimum(en[i0:i1], e) - np.maximum(st[i0:i1], s)
                ok = np.isfinite(dv[i0:i1]) & (ov > 0)
                if ok.any():
                    depth[k] = np.average(dv[i0:i1][ok], weights=ov[ok])
            m = (s + e) // 2
            j = np.searchsorted(st, m, side="right") - 1
            if 0 <= j < len(st) and m < en[j] and isinstance(sv[j], str):
                state[k] = sv[j]
    loci["local_depth"] = np.round(depth, 1)
    # locus state = majority state of its molecules; a tie goes to the state at the locus midpoint
    cn = []
    for sts, mid_state in zip(loci["mol_states"], state):
        vc = pd.Series(sts.split("|")).value_counts()
        top = vc[vc == vc.max()].index.tolist()
        cn.append(mid_state if mid_state in top else top[0])
    loci["CN_State"] = cn
    loci["n_CN_states"] = [len(set(s.split("|"))) for s in loci["mol_states"]]
    return loci


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-b", "--bed", nargs="+", required=True,
                    help="BED(s) of molecules: chrom start end type (one line per molecule)")
    ap.add_argument("-w", "--windows", required=True, help="window_cnv_calls.tsv from analyze_cnv.py")
    ap.add_argument("-o", "--out-prefix", required=True, help="Output prefix (folder is created if needed)")
    ap.add_argument("--depth-bed", default=None,
                    help="mosdepth regions file of the same windows used as depth instead of --depth-col, e.g. "
                         "coverage_mapq20.regions.bed.gz (reads with MAPQ >= 20; recommended)")
    ap.add_argument("--depth-col", default="depth_eff",
                    help="Depth column of window_cnv_calls.tsv used when --depth-bed is not given (default depth_eff)")
    ap.add_argument("--merge-distance", type=int, default=1000,
                    help="Molecules of the same type closer than this (bp) form one locus, as bedtools merge -d "
                         "(default 1000, as in the paper)")
    ap.add_argument("--standardize-bp", type=int, default=2000,
                    help="Resize each molecule to this size around its midpoint before merging, as in the paper and "
                         "in panels D/E (default 2000; 0 keeps the raw forkSense interval)")
    ap.add_argument("-a", "--annotated", default=None,
                    help="<prefix>_annotated.tsv of verifyRegionsSummary_V3.py: keep only RETAINED molecules, as in "
                         "panels D/E (matched by midpoint)")
    ap.add_argument("--mid-tolerance", type=int, default=2, help="bp tolerance when matching midpoints (default 2)")
    ap.add_argument("--fixed-window", type=int, default=0,
                    help="Count only molecules whose midpoint lies within this window centred on each locus (e.g. 2000, "
                         "the paper's IZD size), so loci of different widths are compared over the same length")
    ap.add_argument("--split-at-cn-boundary", action="store_true",
                    help="Never merge molecules lying in different CN states into one locus (then every column of the "
                         "summary adds up to the per-molecule counts of panels D/E)")
    ap.add_argument("--threshold", type=float, default=10, help="Zone threshold: a locus is a zone when its count is GREATER than this (default 10, i.e. more than 10 molecules, which reproduces the paper's 10 IZD and 18 TZD)")
    ap.add_argument("--types", nargs="+", default=None, help="Order of types in the plot (default: as found)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)) or ".", exist_ok=True)

    w = pd.read_csv(a.windows, sep="\t", dtype={"chrom": str}, na_values=["NA"]).sort_values(["chrom", "start"])
    col = a.depth_col
    if a.depth_bed:
        d = pd.read_csv(a.depth_bed, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3],
                        names=["chrom", "start", "end", "depth_ext"], dtype={"chrom": str})
        w = w.merge(d, on=["chrom", "start", "end"], how="left")
        col = "depth_ext"
    w = w.reset_index(drop=True)
    base = float(np.nanmedian(w.loc[w["seg_state"] == "disomic", col]))
    print(f"[INFO] disomic depth (median of disomic windows, '{col}'): {base:.1f}x")

    ev = read_typed_bed(a.bed)
    n0 = len(ev)
    ev["mid"] = (ev["start"] + ev["end"]) // 2
    if a.standardize_bp > 0:
        # as in the paper (and in panels D/E): each molecule = std_bp centred on its midpoint
        ev["start"] = (ev["mid"] - a.standardize_bp // 2).clip(lower=0)
        ev["end"] = ev["start"] + a.standardize_bp
    if a.annotated:
        an = pd.read_csv(a.annotated, sep="\t", dtype={"Chrom": str})
        an["mid"] = (an["Start"] + an["End"]) // 2
        keep = set()
        ret = an[an["Final_Classification"] == "RETAINED"]
        for c, m in zip(ret["Chrom"], ret["mid"]):
            for dd in range(-a.mid_tolerance, a.mid_tolerance + 1):
                keep.add((c, m + dd))
        ok = np.array([(c, m) in keep for c, m in zip(ev["chrom"], ev["mid"])])
        print(f"[INFO] {int(ok.sum())} of {n0} molecules matched a RETAINED region in {os.path.basename(a.annotated)}; "
              f"{int((~ok).sum())} left out (discarded, unassessed or unmatched)")
        ev = ev[ok].reset_index(drop=True)
    # CN state of each molecule = state of the window holding its own midpoint (as in panels D/E)
    ev["mol_state"] = "NA"
    for c, idx in ev.groupby("chrom").groups.items():
        g = w[w["chrom"] == c]
        if g.empty:
            continue
        st, en = g["start"].to_numpy(), g["end"].to_numpy()
        sv = g["seg_state"].to_numpy(dtype=object)
        m = ev.loc[idx, "mid"].to_numpy()
        j = np.searchsorted(st, m, side="right") - 1
        jj = np.clip(j, 0, len(st) - 1)
        hit = (j >= 0) & (m < en[jj])
        vals = np.where(hit, sv[jj], "NA")
        ev.loc[idx, "mol_state"] = [v if isinstance(v, str) else "NA" for v in vals]
    print("[INFO] molecules per CN state (by molecule midpoint, as in panels D/E): "
          + ", ".join(f"{s}={n}" for s, n in ev["mol_state"].value_counts().items()))
    loci = annotate(merge_loci(ev, a.merge_distance, a.split_at_cn_boundary), w, col)
    cross = loci[loci["n_CN_states"] > 1]
    if len(cross):
        print(f"[INFO] {len(cross)} loci ({int(cross['molecules'].sum())} molecules) span a CN-state boundary "
              f"({', '.join(f'{k}: {v}' for k, v in cross['type'].value_counts().items())}); in the per-locus columns "
              "they take the majority state of their molecules. --split-at-cn-boundary splits them instead")
    loci["span_bp"] = loci["end"] - loci["start"]
    if a.fixed_window > 0:
        # count only molecules whose midpoint lies within +/- fixed_window/2 of the locus centre (same size for
        # every locus, like the paper's 2 kb IZD), and use the depth of that window
        half = a.fixed_window // 2
        loci["molecules_whole_span"] = loci["molecules"]
        cen = (loci["start"] + loci["end"]) // 2
        fixed = loci.copy()
        fixed["start"], fixed["end"] = (cen - half).clip(lower=0), cen + half
        fixed = annotate(fixed, w, col)
        cnt = np.zeros(len(loci), dtype=int)
        for (tp, c), idx in loci.groupby(["type", "chrom"]).groups.items():
            m = np.sort(ev.loc[(ev["type"] == tp) & (ev["chrom"] == c), "mid"].to_numpy())
            for k in idx:
                cnt[k] = np.searchsorted(m, cen[k] + half, side="left") - np.searchsorted(m, cen[k] - half, side="left")
        loci["molecules"] = cnt
        loci["local_depth"] = fixed["local_depth"].to_numpy()
        print(f"[INFO] counts restricted to a fixed {a.fixed_window} bp window at each locus centre "
              f"(molecules over the whole merged span kept in 'molecules_whole_span')")
    loci["molecules_CN_normalised"] = np.round(loci["molecules"] * base / loci["local_depth"], 2)
    loci.to_csv(f"{a.out_prefix}_loci.tsv", sep="\t", index=False)

    types = a.types or list(pd.unique(ev["type"]))
    states = [s for s in STATE_ORDER if s in set(loci["CN_State"])]
    summ = []
    for t in types:
        for s in states:
            g = loci[(loci["type"] == t) & (loci["CN_State"] == s)]
            if g.empty and not ((ev["type"] == t) & (ev["mol_state"] == s)).any():
                continue
            summ.append({"type": t, "CN_State": s,
                         "molecules": int(((ev["type"] == t) & (ev["mol_state"] == s)).sum()),
                         "loci": len(g), "molecules_in_loci": int(g["molecules"].sum()),
                         "median_raw": g["molecules"].median(),
                         "median_normalised": round(g["molecules_CN_normalised"].median(), 2),
                         f"loci_raw_gt_{a.threshold:g}": int((g["molecules"] > a.threshold).sum()),
                         f"loci_normalised_gt_{a.threshold:g}": int((g["molecules_CN_normalised"] > a.threshold).sum())})
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{a.out_prefix}_summary.tsv", sep="\t", index=False)
    print(summ.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    nt = len(types)
    width = 0.8 / nt
    rng = np.random.default_rng(1)
    for ax, ycol, title in [(axes[0], "molecules", "Raw: molecules per locus"),
                            (axes[1], "molecules_CN_normalised", "Normalised: molecules at disomic coverage")]:
        for j, t in enumerate(types):
            col_ = PALETTE[j % len(PALETTE)]
            data, pos = [], []
            for i, s in enumerate(states):
                v = loci.loc[(loci["type"] == t) & (loci["CN_State"] == s), ycol].dropna().to_numpy()
                if len(v):
                    data.append(v)
                    pos.append(i - 0.4 + width * (j + 0.5))
            if not data:
                continue
            bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=col_, alpha=0.35, edgecolor=col_)
            for k in ("whiskers", "caps", "medians"):
                for l in bp[k]:
                    l.set(color=col_)
            for v, p in zip(data, pos):
                ax.scatter(p + rng.uniform(-width * 0.3, width * 0.3, len(v)), v, s=6, color=col_, alpha=0.5,
                           linewidths=0, label=t if p == pos[0] and ax is axes[0] else None)
        ax.axhline(a.threshold, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels([STATE_LABEL.get(s, s) for s in states], rotation=45, ha="right")
        ax.set_xlabel("CN state of the segment")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Molecules per locus (log)")
    axes[0].legend(frameon=False, fontsize=8, markerscale=2)
    axes[1].text(0.99, 0.98, f"dashed: {a.threshold:g} molecules (zone = more than {a.threshold:g})\nnormalised = molecules × "
                 f"{base:.0f}× / local depth", transform=axes[1].transAxes, ha="right", va="top", fontsize=7.5,
                 color="#444444")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out_prefix}.{ext}", dpi=300)
    for s in ["_loci.tsv", "_summary.tsv", ".pdf", ".png"]:
        print(f"[OK] {a.out_prefix}{s}")


if __name__ == "__main__":
    main()
