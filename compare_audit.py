#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_audit.py — structural audit of the events vs. their control sets, with blocks as the unit.

Are origin/termination regions found in assembly artefacts more (or less) often than comparable control
regions? Regions are not independent: molecules at the same place get the same audit, and matched
controls may share positions. Every interval and test therefore resamples BLOCKS (regions whose
+/- --flank windows overlap, as in make_controls.py), never single regions.

INPUT  the *_annotated.tsv files written by verifyRegionsSummary_V3.py (columns Chrom, Start, End,
       Structural_Status, Final_Classification, CN_State), one for the events and one per control set.
       Run verifyRegionsSummary_V3.py on each control audit first.

OUTCOME  DISCARDED_STRUCTURAL among assessed regions (RETAINED + DISCARDED_STRUCTURAL); regions
         UNASSESSED_* (gap / no coverage) are counted and left out of the denominator.

STATISTICS (per control set, events vs control)
  - % discarded with 95% CI (block bootstrap), difference in percentage points and ratio with CI
  - permutation test: block labels (event / control) shuffled, number of blocks per set kept
  - Mann-Whitney U on the per-block fraction discarded (each block counts once)
  - CN-stratified difference: per-state differences combined with Mantel-Haenszel weights, block-bootstrap CI
  - per Structural_Status distribution (descriptive)

USAGE
  python3 compare_audit.py -t ORI_annotated.tsv \\
      -c controls/ORI.matched_annotated.tsv controls/ORI.shift_1_annotated.tsv controls/ORI.random_annotated.tsv \\
      --labels matched shift_1 random -o ORI_vs_controls

OUTPUTS  <o>_overall.tsv, <o>_by_CN_state.tsv, <o>_status.tsv, <o>_summary.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

STATE_ORDER = ["loss_0copies", "loss_1copy", "disomic", "gain_3copies", "gain_4copies",
               "gain_5copies", "amplified"]
ASSESSED = ["RETAINED", "DISCARDED_STRUCTURAL"]


def load(path, flank, tag):
    d = pd.read_csv(path, sep="\t", dtype={"Chrom": str})
    need = {"Chrom", "Start", "End", "Structural_Status", "Final_Classification", "CN_State"}
    if not need.issubset(d.columns):
        sys.exit(f"[ERROR] {path} lacks {sorted(need - set(d.columns))}: pass the *_annotated.tsv of "
                 "verifyRegionsSummary_V3.py")
    d = d.sort_values(["Chrom", "Start"]).reset_index(drop=True)
    blk, cur_c, cur_e, n = [], None, -1, 0
    for c, s, e in zip(d["Chrom"], d["Start"] - flank, d["End"] + flank):
        if c != cur_c or s > cur_e:
            n += 1
            cur_c, cur_e = c, e
        else:
            cur_e = max(cur_e, e)
        blk.append(f"{tag}_B{n:05d}")
    d["block"] = blk
    d["assessed"] = d["Final_Classification"].isin(ASSESSED)
    d["disc"] = d["Final_Classification"].eq("DISCARDED_STRUCTURAL")
    return d


def block_arrays(d, states):
    """Per block: assessed and discarded counts, overall and per CN state."""
    a = d[d["assessed"]]
    b = pd.unique(a["block"])
    bi = pd.Series(np.arange(len(b)), index=b)[a["block"]].to_numpy()
    si = pd.Series(np.arange(len(states)), index=states).reindex(a["CN_State"]).to_numpy()
    ok = ~np.isnan(si)
    N = np.zeros((len(b), len(states)))
    D = np.zeros((len(b), len(states)))
    np.add.at(N, (bi[ok], si[ok].astype(int)), 1)
    np.add.at(D, (bi[ok], si[ok].astype(int)), a["disc"].to_numpy()[ok])
    n_all = np.bincount(bi, minlength=len(b)).astype(float)
    d_all = np.bincount(bi, weights=a["disc"].to_numpy().astype(float), minlength=len(b))
    return n_all, d_all, N, D


def ci(x, digits=2):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "NA"
    return f"{np.quantile(x, 0.025):.{digits}f} to {np.quantile(x, 0.975):.{digits}f}"


def mh_diff(Nt, Dt, Nc, Dc):
    """Mantel-Haenszel-weighted difference in proportion across strata (columns), in percentage points."""
    w = np.where((Nt + Nc) > 0, Nt * Nc / np.maximum(Nt + Nc, 1), 0)
    pt = np.divide(Dt, Nt, out=np.zeros_like(Dt), where=Nt > 0)
    pc = np.divide(Dc, Nc, out=np.zeros_like(Dc), where=Nc > 0)
    return 100 * np.sum(w * (pt - pc), axis=-1) / np.sum(w, axis=-1)


def main():
    np.seterr(divide="ignore", invalid="ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--target", required=True, help="*_annotated.tsv of the events")
    ap.add_argument("-c", "--controls", nargs="+", required=True, help="*_annotated.tsv of each control set")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("-o", "--out-prefix", required=True)
    ap.add_argument("--target-label", default="events")
    ap.add_argument("--flank", type=int, default=10000, help="Block definition, as make_controls.py (default 10000)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(p).replace("_annotated.tsv", "") for p in a.controls]
    if len(labels) != len(a.controls):
        sys.exit("[ERROR] one label per control file")
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)) or ".", exist_ok=True)
    rng = np.random.default_rng(a.seed)

    T = load(a.target, a.flank, "T")
    C = {lab: load(p, a.flank, lab) for lab, p in zip(labels, a.controls)}
    states = [s for s in STATE_ORDER if s in set(T["CN_State"]) | set().union(*[set(x["CN_State"]) for x in C.values()])]
    tn, td, tN, tD = block_arrays(T, states)

    def boot_w(nb):
        return rng.multinomial(nb, np.full(nb, 1 / nb), size=a.n_boot).astype(float)

    Wt = boot_w(len(tn))
    pt_b = 100 * (Wt @ td) / (Wt @ tn)
    tN_b, tD_b = np.einsum("rb,bs->rs", Wt, tN), np.einsum("rb,bs->rs", Wt, tD)

    overall, by_state, status_rows, lines = [], [], [], []

    def describe(lab, d, n, dd, pb):
        return {"set": lab, "n_regions": len(d), "n_assessed": int(d["assessed"].sum()),
                "n_unassessed": int((~d["assessed"]).sum()), "n_blocks": len(n),
                "n_discarded": int(dd.sum()), "pct_discarded": round(100 * dd.sum() / n.sum(), 2),
                "pct_discarded_CI95": ci(pb)}

    overall.append(describe(a.target_label, T, tn, td, pt_b))
    for lab, d in C.items():
        cn, cd, cN, cD = block_arrays(d, states)
        Wc = boot_w(len(cn))
        pc_b = 100 * (Wc @ cd) / (Wc @ cn)
        r = describe(lab, d, cn, cd, pc_b)
        pt, pc = 100 * td.sum() / tn.sum(), 100 * cd.sum() / cn.sum()
        r["diff_events_minus_control_pp"] = round(pt - pc, 2)
        r["diff_CI95"] = ci(pt_b - pc_b)
        r["ratio_events_to_control"] = round(pt / pc, 3) if pc > 0 else np.nan
        r["ratio_CI95"] = ci(pt_b / pc_b, 3)
        # permutation of block labels
        n_all, d_all = np.concatenate([tn, cn]), np.concatenate([td, cd])
        nt = len(tn)
        obs = pt - pc
        perm = np.empty(a.n_perm)
        for k in range(a.n_perm):
            idx = rng.permutation(len(n_all))
            i1, i2 = idx[:nt], idx[nt:]
            perm[k] = 100 * (d_all[i1].sum() / n_all[i1].sum() - d_all[i2].sum() / n_all[i2].sum())
        r["p_permutation_blocks"] = round((np.sum(np.abs(perm) >= abs(obs)) + 1) / (a.n_perm + 1), 4)
        u = stats.mannwhitneyu(td / tn, cd / cn, alternative="two-sided")
        r["p_MannWhitney_block_fraction"] = float(f"{u.pvalue:.3g}")
        # CN-stratified (Mantel-Haenszel weights)
        cN_b, cD_b = np.einsum("rb,bs->rs", Wc, cN), np.einsum("rb,bs->rs", Wc, cD)
        r["CN_stratified_diff_pp"] = round(float(mh_diff(tN.sum(0), tD.sum(0), cN.sum(0), cD.sum(0))), 2)
        r["CN_stratified_diff_CI95"] = ci(mh_diff(tN_b, tD_b, cN_b, cD_b))
        overall.append(r)
        for j, s in enumerate(states):
            nt_s, dt_s, nc_s, dc_s = tN[:, j].sum(), tD[:, j].sum(), cN[:, j].sum(), cD[:, j].sum()
            p1 = 100 * dt_s / nt_s if nt_s else np.nan
            p2 = 100 * dc_s / nc_s if nc_s else np.nan
            by_state.append({"control_set": lab, "CN_State": s,
                             "events_assessed": int(nt_s), "events_blocks": int((tN[:, j] > 0).sum()),
                             "events_pct_discarded": round(p1, 2),
                             "control_assessed": int(nc_s), "control_blocks": int((cN[:, j] > 0).sum()),
                             "control_pct_discarded": round(p2, 2),
                             "diff_pp": round(p1 - p2, 2),
                             "diff_CI95": ci(100 * tD_b[:, j] / tN_b[:, j] - 100 * cD_b[:, j] / cN_b[:, j])})

    for lab, d in [(a.target_label, T)] + list(C.items()):
        vc = d["Structural_Status"].value_counts()
        for st, n in vc.items():
            status_rows.append({"set": lab, "Structural_Status": st, "n": int(n), "pct": round(100 * n / len(d), 2)})
    status = pd.DataFrame(status_rows).pivot_table(index="Structural_Status", columns="set", values="pct",
                                                   fill_value=0)[[a.target_label] + labels]

    ov, bs = pd.DataFrame(overall), pd.DataFrame(by_state)
    ov.to_csv(f"{a.out_prefix}_overall.tsv", sep="\t", index=False)
    bs.to_csv(f"{a.out_prefix}_by_CN_state.tsv", sep="\t", index=False)
    status.to_csv(f"{a.out_prefix}_status.tsv", sep="\t")
    pd.set_option("display.width", 250)
    lines = ["=" * 100, f"STRUCTURAL AUDIT: {a.target_label} vs controls (unit = block, flank {a.flank} bp; "
             f"{a.n_boot} bootstraps, {a.n_perm} permutations)", "=" * 100,
             "Outcome: DISCARDED_STRUCTURAL among assessed regions (unassessed = gap / no coverage, excluded)", "",
             ov.to_string(index=False), "", "By CN state:", bs.to_string(index=False), "",
             "Structural_Status (% of all regions):", status.to_string(), "",
             "diff > 0: events are discarded MORE often than the control; < 0: less often.",
             "CN_stratified_diff: difference within CN states, combined (Mantel-Haenszel weights).", "=" * 100]
    txt = "\n".join(lines)
    print(txt)
    with open(f"{a.out_prefix}_summary.txt", "w") as fh:
        fh.write(txt + "\n")
    for s in ["_overall.tsv", "_by_CN_state.tsv", "_status.tsv", "_summary.txt"]:
        print(f"[OK] {a.out_prefix}{s}")


if __name__ == "__main__":
    main()
