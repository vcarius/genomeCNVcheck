#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_audit_results.py — publication figure for the structural audit (verifyRegions.py).

Panels
  A  Share of regions per structural status
  B  Diagnostic plane: depth ratio vs. % of reads with a core breakpoint. The ratio is the one used
     for classification (core / CN segment when verifyRegions.py ran with --cnv-windows, otherwise
     core / flanks)
  C  Clean spanning reads per status
  D  (optional, --oe)   Observed/expected events per CN state (verifyRegionsSummary_V3.py)
  E  (optional, --rate) Events per Mb of read sequence per CN state, relative to disomic

Input: the verifyRegions.py TSV or the <prefix>_annotated.tsv from verifyRegionsSummary_V3.py.
Outputs: <prefix>.pdf (vector) and <prefix>.png (300 dpi).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Display names (status codes written by verifyRegions.py are unchanged); old and new codes accepted
DISPLAY_NAMES = {
    "VALIDATED_STRUCTURE": "Validated structure",
    "INTACT_CORE_FLANK_CLIPPING": "Intact core, flank clipping",
    "INTACT_ROI_UNSTABLE_10KB_FLANK": "Intact core, flank clipping",
    "MODERATE_INSTABILITY_REVIEW": "Moderate instability (review)",
    "HETEROZYGOUS_SV_OR_HAPLOTYPE_DIVERGENCE": "Haplotype divergence / het. SV",
    "HOMOZYGOUS_LOCAL_INDEL_SPANNING_ROI": "Homozygous local indel",
    "REPEAT_COLLAPSE_IN_ROI": "Repeat collapse (discarded)",
    "WARNING_LOCAL_HIGH_COVERAGE_CNV": "Local hyper-coverage (discarded)",
    "CRITICAL_ERROR_CHIMERA_OR_BREAK_IN_ROI": "Break / chimera (discarded)",
    "WARNING_ZERO_SPANNING_READS": "No spanning reads (discarded)",
    "NO_CORE_COVERAGE": "No core coverage",
}
PALETTE = {
    "Validated structure": "#2ca02c",
    "Intact core, flank clipping": "#1f77b4",
    "Moderate instability (review)": "#ff7f0e",
    "Haplotype divergence / het. SV": "#9467bd",
    "Homozygous local indel": "#bcbd22",
    "Repeat collapse (discarded)": "#d62728",
    "Local hyper-coverage (discarded)": "#e377c2",
    "Break / chimera (discarded)": "#8c564b",
    "No spanning reads (discarded)": "#7f7f7f",
    "No core coverage": "#17becf",
}
CN_NAMES = {"loss_0copies": "0 copies", "loss_1copy": "1 copy", "disomic": "2 (disomic)",
            "gain_3copies": "3 copies", "gain_4copies": "4 copies", "gain_5copies": "5 copies",
            "amplified": "amplified"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="verifyRegions.py TSV or *_annotated.tsv")
    p.add_argument("-o", "--output-prefix", default="figure_structural_audit")
    p.add_argument("--oe", default=None, help="<prefix>_OE_by_CN_state.tsv -> panel D")
    p.add_argument("--rate", default=None, help="<prefix>_rate_by_CN_state.tsv -> panel E")
    p.add_argument("--high-depth-ratio", type=float, default=1.6, help="Same value used in verifyRegions.py")
    p.add_argument("--min-bp-perc-critical", type=float, default=30.0, help="Same value used in verifyRegions.py")
    return p.parse_args()


def depth_ratio_column(df):
    """Ratio used for classification, and an axis label."""
    if "Depth_Ratio_Used" in df.columns and (df["Depth_Ratio_Used"] == "segment").mean() > 0.5:
        seg = pd.to_numeric(df["Depth_Ratio_Core_vs_Segment"], errors="coerce")
        flank = pd.to_numeric(df["Depth_Ratio_Core_vs_Flank"], errors="coerce")
        return seg.where(df["Depth_Ratio_Used"] == "segment", flank), "Depth ratio (core / CN segment)"
    return pd.to_numeric(df["Depth_Ratio_Core_vs_Flank"], errors="coerce"), "Depth ratio (core / flanks)"


def main():
    a = parse_args()
    if not os.path.exists(a.input):
        sys.exit(f"[ERROR] Input '{a.input}' not found.")
    df = pd.read_csv(a.input, sep="\t")
    if "Structural_Status" not in df.columns:
        sys.exit("[ERROR] No 'Structural_Status' column: is this a verifyRegions.py output?")

    sns.set_theme(style="ticks", font_scale=1.0)
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    df["Status_Label"] = df["Structural_Status"].map(lambda s: DISPLAY_NAMES.get(s, s))
    palette = {lab: PALETTE.get(lab, "#cccccc") for lab in df["Status_Label"].unique()}
    order = df["Status_Label"].value_counts().index.tolist()
    df["ratio_x"], ratio_label = depth_ratio_column(df)

    n_panels = 3 + bool(a.oe) + bool(a.rate)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6.5))

    # A. status overview
    pct = (df["Status_Label"].value_counts(normalize=True) * 100).rename("Percentage").reset_index()
    pct.columns = ["Status_Label", "Percentage"]
    sns.barplot(data=pct, y="Status_Label", x="Percentage", hue="Status_Label", legend=False,
                palette=palette, ax=axes[0], edgecolor="black", linewidth=0.8)
    for bar in axes[0].patches:
        if bar.get_width() > 0:
            axes[0].annotate(f"{bar.get_width():.1f}%", (bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2),
                             ha="left", va="center", fontsize=9)
    axes[0].set_title(f"A. Structural status (n = {len(df)})", fontsize=12, fontweight="bold", loc="left")
    axes[0].set_xlabel("Regions (%)")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, 100)
    sns.despine(ax=axes[0])

    # B. diagnostic plane
    sns.scatterplot(data=df, x="ratio_x", y="Perc_Core_Reads_With_BP", hue="Status_Label", hue_order=order,
                    palette=palette, s=40, alpha=0.8, edgecolor="black", linewidth=0.4, ax=axes[1])
    axes[1].axvline(a.high_depth_ratio, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(a.min_bp_perc_critical, color="darkred", linestyle=":", linewidth=1)
    axes[1].text(0.99, 0.99,
                 f"dashed: depth ratio {a.high_depth_ratio}\ndotted: {a.min_bp_perc_critical:.0f}% reads with a core breakpoint\n"
                 "above the dotted line, regions are discarded\nonly when fewer than 3 reads span the core",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8, color="#444444")
    axes[1].set_title("B. Depth vs. core breakpoints", fontsize=12, fontweight="bold", loc="left")
    axes[1].set_xlabel(ratio_label)
    axes[1].set_ylabel("Reads with a core breakpoint (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    sns.despine(ax=axes[1])

    # C. physical support
    sns.boxplot(data=df, x="Status_Label", y="Clean_Spanning_Core", order=order, hue="Status_Label",
                hue_order=order, legend=False, palette=palette, ax=axes[2], fliersize=2, linewidth=1.0)
    sns.stripplot(data=df, x="Status_Label", y="Clean_Spanning_Core", order=order, color="black",
                  alpha=0.25, jitter=0.2, size=2.5, ax=axes[2])
    axes[2].set_title("C. Reads spanning the region", fontsize=12, fontweight="bold", loc="left")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Clean spanning reads")
    axes[2].tick_params(axis="x", labelrotation=45, labelsize=9)
    for lab in axes[2].get_xticklabels():
        lab.set_horizontalalignment("right")
    sns.despine(ax=axes[2])

    k = 3
    # D. O/E per CN state
    if a.oe:
        oe = pd.read_csv(a.oe, sep="\t")
        oe = oe[oe["Set"] == "retained"].copy()
        col = "OE_vs_null" if "OE_vs_null" in oe.columns else "OE_vs_perm"
        oe["label"] = oe["CN_State"].map(lambda s: CN_NAMES.get(s, s))
        cn_colors = {"loss_0copies": "#b35806", "loss_1copy": "#f1a340", "disomic": "#2f6db5",
                     "gain_3copies": "#d9c6e6", "gain_4copies": "#b391cf", "gain_5copies": "#8c5fb3",
                     "amplified": "#542788"}
        colors = [cn_colors.get(s, "#9a9a9a") for s in oe["CN_State"]]
        # 95% range of the circular-shift null, in O/E units (range / expected)
        lo, hi = [], []
        for r in oe.itertuples():
            try:
                a_, b_ = [float(x) for x in str(r.Null_95pct_range).split("-")]
                lo.append(a_ / r.Expected_null)
                hi.append(b_ / r.Expected_null)
            except (ValueError, ZeroDivisionError, AttributeError):
                lo.append(np.nan)
                hi.append(np.nan)
        x = np.arange(len(oe))
        axes[k].bar(x, oe[col], color=colors, edgecolor="black", linewidth=0.6)
        axes[k].vlines(x, lo, hi, color="#333333", linewidth=2.2, alpha=0.6, label="95% of the null")
        axes[k].set_xticks(x)
        axes[k].set_xticklabels(oe["label"])
        axes[k].axhline(1.0, color="black", linestyle="--", linewidth=1)
        for i, r in enumerate(oe.itertuples()):
            star = "*" if r.p_two_sided < 0.05 else ""
            top = np.nanmax([getattr(r, col), hi[i]])
            axes[k].annotate(f"{star}n={r.Observed}\np={r.p_two_sided:.3g}", (i, top),
                             xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        axes[k].legend(handles=[Patch(color="#f1a340", label="loss"), Patch(color="#2f6db5", label="disomic"),
                                Patch(color="#8c5fb3", label="gain (darker = more copies)"),
                                Line2D([0], [0], color="#333333", lw=2.2, alpha=0.6,
                                       label="95% range of the null (O/E = 1 expected)")],
                       fontsize=7, loc="upper left", frameon=False)
        oe[col] = oe[col].clip(lower=0)
        if np.isfinite(np.nanmax(hi)):
            oe[col] = oe[col].where(oe[col] > 0, 0)
        axes[k].set_ylim(0, max(1.6, np.nanmax(list(oe[col]) + hi) * 1.3))
        axes[k].set_title("D. Events per CN state (O/E)", fontsize=12, fontweight="bold", loc="left")
        axes[k].set_ylabel("Observed / expected (circular-shift null)")
        axes[k].set_xlabel("CN state of the segment")
        axes[k].tick_params(axis="x", labelrotation=45, labelsize=9)
        sns.despine(ax=axes[k])
        k += 1

    # E. rate per read Mb relative to disomic
    if a.rate:
        rt = pd.read_csv(a.rate, sep="\t")
        if "rate_vs_disomic" in rt.columns:
            rt["label"] = rt["CN_State"].map(lambda s: CN_NAMES.get(s, s))
            ci = rt["rate_vs_disomic_CI95"].str.split("-", expand=True).astype(float)
            err = [(rt["rate_vs_disomic"] - ci[0]).clip(lower=0), (ci[1] - rt["rate_vs_disomic"]).clip(lower=0)]
            axes[k].errorbar(rt["label"], rt["rate_vs_disomic"], yerr=err, fmt="o", color="#2f6db5",
                             ecolor="#4a5463", capsize=4, markersize=7)
            axes[k].axhline(1.0, color="black", linestyle="--", linewidth=1)
            axes[k].set_ylim(0, max(1.5, ci[1].max() * 1.1))
            axes[k].set_title("E. Events per read Mb (vs disomic)", fontsize=12, fontweight="bold", loc="left")
            axes[k].set_ylabel("Rate relative to disomic (95% CI, segment bootstrap)")
            axes[k].set_xlabel("CN state of the segment")
            axes[k].tick_params(axis="x", labelrotation=45, labelsize=9)
            sns.despine(ax=axes[k])

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(f"{a.output_prefix}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{a.output_prefix}.png", format="png", dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved: {a.output_prefix}.pdf (vector) and {a.output_prefix}.png (300 dpi)")


if __name__ == "__main__":
    main()
