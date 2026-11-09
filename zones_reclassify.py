#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zones_reclassify.py - rebuild the paper's mergers (IZ/TZ) from the molecules themselves and say which zones
survive the structural audit and the copy-number correction.

WHY
  The zones are defined by the merge, not by a label. This script redoes the paper's procedure on the full set
  of molecules (all 2 kb regions, before the audit), so every zone keeps the identity it has in the paper, and
  only then counts, inside each zone, the molecules RETAINED by the audit and their CN-normalised number.
  (Re-merging only the molecules labelled IZ after the audit can split a zone or pick up a wrong label.)

STEPS
  1. molecules = all regions of the annotated file (2 kb each, as in the paper), RETAINED or not
  2. bedtools merge -d 1000 on all of them -> mergers; n_all = molecules per merger (paper count)
  3. zone (paper) = merger with n_all >= --zone-min
  4. n_retained = molecules of the merger RETAINED by the audit
  5. local depth = depth over the merger span (overlap-weighted, MAPQ>=20 with --depth-bed)
     normalised = n_retained x disomic depth / local depth   (molecules expected at disomic coverage)
     normalised_retained_segment uses the median depth of the CN segment instead (copy number only, no local
     fluctuation); --depth-scope chooses which one drives 'status' (status_local / status_segment are both kept)
  6. status of each paper zone (zone-min 11: counts >= 11, normalised > 10 = "more than 10", see --norm-gt):
       keeps                 n_retained >= zone-min and normalised > zone-min - 1
       lost_CN_correction    n_retained >= zone-min, normalised <= zone-min - 1 (high count only from extra copies)
       lost_audit            n_retained < zone-min (molecules discarded as structural artefacts)
     mergers that are not paper zones:
       new_candidate         normalised > zone-min - 1 and n_retained >= --min-raw (usually 1-copy segments)
       borderline            n_all = zone-min - 1

NOTE ON THE THRESHOLD
  On the 4,287 origins this gives the paper's 2,078 mergers; 13 have >= 10 molecules, 10 have >= 11. The
  paper reports 10 IZD, so the rule applied was "more than 10" (--zone-min 11, default). Use --zone-min 10 to
  follow the wording of the text ("threshold of 10").

INPUT
  -a  <prefix>_annotated.tsv of verifyRegionsSummary_V3.py (Chrom, Start, End, Final_Classification;
      one 2 kb region per molecule, all molecules)
  -w  window_cnv_calls.tsv (chrom start end seg_state + depth column)   --depth-bed coverage_mapq20.regions.bed.gz

OUTPUT
  <o>_zones.tsv        every merger that is a paper zone, a candidate or borderline, with counts and status
  <o>_zones_2kb.bed    paper zones trimmed to +-1 kb of the merger midpoint (IZD/TZD), name = status
  <o>_typed.bed        every molecule with its type from the merge (single / zone / zone_lost_audit for the
                       molecules of a zone removed by the audit, not plotted by --types single zone), input of
                       plot_events_by_type.py -b (labels consistent with the mergers)

EXAMPLE
  python3 zones_reclassify.py -a ORI_annotated.tsv -w window_cnv_calls.tsv \\
      --depth-bed coverage_mapq20.regions.bed.gz --type-names origins IZ -o ORI_zones
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-a", "--annotated", required=True, help="<prefix>_annotated.tsv (all molecules)")
    ap.add_argument("-w", "--windows", required=True, help="window_cnv_calls.tsv (seg_state + depth)")
    ap.add_argument("-o", "--out-prefix", required=True)
    ap.add_argument("--depth-bed", default=None, help="mosdepth regions file (e.g. coverage_mapq20.regions.bed.gz)")
    ap.add_argument("--depth-col", default="depth_eff", help="depth column of -w when --depth-bed is not given")
    ap.add_argument("--merge-distance", type=int, default=1000, help="bedtools merge -d (default 1000)")
    ap.add_argument("--zone-min", type=int, default=11,
                    help="molecules for a merger to be a zone (default 11 = 'more than 10', reproduces the paper's 10 IZD)")
    ap.add_argument("--norm-gt", type=float, default=None,
                    help="a normalised count passes when it is GREATER than this (default zone-min - 1 = 10, i.e. the "
                         "paper's 'more than 10' applied to a non-integer value)")
    ap.add_argument("--min-raw", type=int, default=5,
                    help="minimum retained molecules for a new candidate (default 5)")
    ap.add_argument("--depth-scope", choices=["local", "segment"], default="local",
                    help="depth that drives 'status': local = depth over the merger span (default; also corrects focal "
                         "collapses), segment = median depth of the CN segment (copy number only). Both are reported")
    ap.add_argument("--type-names", nargs=2, default=["origins", "IZ"], metavar=("SINGLE", "ZONE"),
                    help="type names for <o>_typed.bed (default origins IZ; use terminations TZ)")
    ap.add_argument("--fixed-window", type=int, default=0,
                    help="Statistical test: count retained molecules in the densest window of this size inside each "
                         "merger (e.g. 2000, the paper's IZD) instead of the whole merger (default 0 = whole merger). "
                         "Also centres <o>_zones_2kb.bed on that window instead of the merger midpoint")
    ap.add_argument("--ambiguous-ratio", type=float, default=3.0,
                    help="local depth / disomic depth >= this: coordinate ambiguous (collapsed copies fold onto one "
                         "reference position); such mergers are not tested and are left out of the background (default 3.0)")
    ap.add_argument("--tile-bp", type=int, default=10000,
                    help="Tile size used to estimate the negative-binomial overdispersion of the background (default "
                         "10000, about the span of a zone; uses --fixed-window when that is set)")
    ap.add_argument("--fdr", type=float, default=0.05, help="Benjamini-Hochberg level for status_stat (default 0.05)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)) or ".", exist_ok=True)

    # windows, depth and disomic baseline
    w = pd.read_csv(a.windows, sep="\t", dtype={"chrom": str}, na_values=["NA"])
    col = a.depth_col
    if a.depth_bed:
        d = pd.read_csv(a.depth_bed, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3],
                        names=["chrom", "start", "end", "depth_ext"], dtype={"chrom": str})
        w = w.merge(d, on=["chrom", "start", "end"], how="left")
        col = "depth_ext"
    w = w.sort_values(["chrom", "start"]).reset_index(drop=True)
    base = float(np.nanmedian(w.loc[w["seg_state"] == "disomic", col]))
    print(f"[INFO] disomic depth (median of disomic windows, '{col}'): {base:.1f}x")
    W = {c: (g["start"].to_numpy(), g["end"].to_numpy(), g[col].to_numpy(dtype=float),
             g["seg_state"].to_numpy(dtype=object)) for c, g in w.groupby("chrom")}
    has_seg = "seg_id" in w.columns
    if has_seg:
        seg_depth = w[w["seg_id"] >= 0].groupby(["chrom", "seg_id"])[col].median()
        SID = {c: g["seg_id"].to_numpy() for c, g in w.groupby("chrom")}
    elif a.depth_scope == "segment":
        sys.exit("[ERROR] --depth-scope segment needs a seg_id column in -w")

    def seg_depth_at(c, s, e):
        # median depth of the CN segment covering most of the merger (windows outside segments are ignored)
        if not has_seg or c not in W:
            return np.nan
        st, en, _, _ = W[c]
        i0, i1 = np.searchsorted(en, s, side="right"), np.searchsorted(st, e, side="left")
        if i1 <= i0:
            return np.nan
        ov = np.minimum(en[i0:i1], e) - np.maximum(st[i0:i1], s)
        sid = SID[c][i0:i1]
        ok = (sid >= 0) & (ov > 0)
        if not ok.any():
            return np.nan
        best = pd.Series(ov[ok]).groupby(sid[ok]).sum().idxmax()
        return float(seg_depth.get((c, best), np.nan))

    def state_at(c, m):
        if c not in W:
            return "NA"
        st, en, _, sv = W[c]
        j = np.searchsorted(st, m, side="right") - 1
        return sv[j] if 0 <= j < len(st) and m < en[j] and isinstance(sv[j], str) else "NA"

    def depth_over(c, s, e):
        if c not in W:
            return np.nan
        st, en, dv, _ = W[c]
        i0, i1 = np.searchsorted(en, s, side="right"), np.searchsorted(st, e, side="left")
        if i1 <= i0:
            return np.nan
        ov = np.minimum(en[i0:i1], e) - np.maximum(st[i0:i1], s)
        ok = np.isfinite(dv[i0:i1]) & (ov > 0)
        return float(np.average(dv[i0:i1][ok], weights=ov[ok])) if ok.any() else np.nan

    # molecules
    an = pd.read_csv(a.annotated, sep="\t", dtype={"Chrom": str})
    need = {"Chrom", "Start", "End", "Final_Classification"}
    if not need <= set(an.columns):
        sys.exit(f"[ERROR] {a.annotated} needs columns {sorted(need)}")
    m = an[["Chrom", "Start", "End", "Final_Classification"]].rename(
        columns={"Chrom": "chrom", "Start": "start", "End": "end", "Final_Classification": "audit"})
    m = m.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    m["mid"] = (m["start"] + m["end"]) // 2
    m["retained"] = m["audit"].eq("RETAINED")
    m["mol_state"] = [state_at(c, x) for c, x in zip(m["chrom"], m["mid"])]
    lens = (m["end"] - m["start"]).value_counts()
    if len(lens) > 1:
        print(f"[WARN] regions have {len(lens)} different lengths; the paper merged 2 kb regions")
    print(f"[INFO] {len(m)} molecules, {int(m['retained'].sum())} RETAINED")

    # merge all molecules (paper)
    mid_id = np.zeros(len(m), dtype=int)
    k = -1
    prev_c, ce = None, None
    for i, (c, s, e) in enumerate(zip(m["chrom"], m["start"], m["end"])):
        if c != prev_c or s - ce > a.merge_distance:
            k += 1
            prev_c, ce = c, e
        else:
            ce = max(ce, e)
        mid_id[i] = k
    m["merger"] = mid_id
    g = m.groupby("merger")
    mg = pd.DataFrame({"chrom": g["chrom"].first(), "start": g["start"].min(), "end": g["end"].max(),
                       "n_all": g.size(), "n_retained": g["retained"].sum().astype(int)})
    disc = m[~m["retained"]].groupby("merger")["audit"].agg(lambda s: ";".join(f"{k}:{v}" for k, v in s.value_counts().items()))
    mg["not_retained"] = disc.reindex(mg.index).fillna("")
    print(f"[INFO] {len(mg)} mergers; n_all >= 10: {int((mg['n_all'] >= 10).sum())}; "
          f">= {a.zone_min} (zones): {int((mg['n_all'] >= a.zone_min).sum())}")

    # CN state: majority over retained molecules (all molecules if none retained); tie -> state at merger midpoint
    def cn_of(sub, c, s, e):
        sts = sub.loc[sub["retained"], "mol_state"] if sub["retained"].any() else sub["mol_state"]
        vc = sts.value_counts()
        top = vc[vc == vc.max()].index.tolist()
        ms = state_at(c, (s + e) // 2)
        return (ms if ms in top else top[0]), len(vc)
    cn = [cn_of(m[m["merger"] == i], r.chrom, r.start, r.end) for i, r in mg.iterrows()]
    mg["CN_State"] = [x[0] for x in cn]
    mg["n_CN_states"] = [x[1] for x in cn]
    mg["span_bp"] = mg["end"] - mg["start"]
    mg["local_depth"] = [round(depth_over(r.chrom, r.start, r.end), 1) for r in mg.itertuples()]
    mg["normalised_all"] = np.round(mg["n_all"] * base / mg["local_depth"], 2)
    mg["normalised_retained"] = np.round(mg["n_retained"] * base / mg["local_depth"], 2)
    mg["segment_depth"] = [round(seg_depth_at(r.chrom, r.start, r.end), 1) for r in mg.itertuples()]
    mg["local_vs_segment"] = np.round(mg["local_depth"] / mg["segment_depth"], 2)
    mg["normalised_retained_segment"] = np.round(mg["n_retained"] * base / mg["segment_depth"], 2)
    norm_col = "normalised_retained" if a.depth_scope == "local" else "normalised_retained_segment"

    # ---- statistical zone call: retained molecules vs. the number expected from the local depth -------------------
    # expected = lambda x sum(depth x bp) over the counted span, lambda = retained molecules per read bp in the genome
    # outside low-mappability and ambiguous (collapsed) windows; NB overdispersion estimated on tiles of the genome.
    lowmap = w["low_mappability"].astype(str).str.lower().eq("true") if "low_mappability" in w.columns \
        else pd.Series(False, index=w.index)
    w["bg_ok"] = np.isfinite(w[col]) & ~lowmap & (w[col] / base < a.ambiguous_ratio)
    w["mol"] = 0
    ret = m[m["retained"]]
    for c, g in w.groupby("chrom"):
        mids = np.sort(ret.loc[ret["chrom"] == c, "mid"].to_numpy())
        st, en = g["start"].to_numpy(), g["end"].to_numpy()
        w.loc[g.index, "mol"] = np.searchsorted(mids, en, side="left") - np.searchsorted(mids, st, side="left")
    bg = w[w["bg_ok"]]
    lam = bg["mol"].sum() / float((bg[col] * (bg["end"] - bg["start"])).sum())
    tile = a.fixed_window if a.fixed_window > 0 else a.tile_bp
    t = bg.assign(tile=bg["start"] // tile, rb=bg[col] * (bg["end"] - bg["start"]), bp=bg["end"] - bg["start"])
    t = t.groupby(["chrom", "tile"]).agg(n=("mol", "sum"), rb=("rb", "sum"), bp=("bp", "sum"))
    t = t[t["bp"] >= tile / 2]
    mu_t = lam * t["rb"]
    alpha = max(0.0, float((((t["n"] - mu_t) ** 2 - t["n"]).sum()) / (mu_t ** 2).sum()))
    print(f"[INFO] background: {lam * 1e6:.3f} retained molecules per read Mb; NB overdispersion alpha = {alpha:.3f} "
          f"({len(t)} tiles of {tile} bp)")

    def densest(mids, W):
        mids = np.sort(mids)
        if len(mids) == 0:
            return 0, np.nan
        j = np.searchsorted(mids, mids + W, side="left")
        k = int(np.argmax(j - np.arange(len(mids))))
        inside = mids[k:j[k]]
        return int(j[k] - k), float((inside[0] + inside[-1]) // 2)

    n_test, s_test, e_test, cen_test = [], [], [], []
    for i, r in mg.iterrows():
        mids = m.loc[(m["merger"] == i) & m["retained"], "mid"].to_numpy()
        if a.fixed_window > 0:
            n_k, cc = densest(mids, a.fixed_window)
            cc = (r.start + r.end) // 2 if not np.isfinite(cc) else int(cc)
            n_test.append(n_k)
            s_test.append(cc - a.fixed_window // 2)
            e_test.append(cc + a.fixed_window // 2)
            cen_test.append(cc)
        else:
            n_test.append(len(mids))
            s_test.append(r.start)
            e_test.append(r.end)
            cen_test.append((r.start + r.end) // 2)
    mg["n_tested"] = n_test
    mg["test_center"] = cen_test
    dep_t = np.array([depth_over(c, s, e) for c, s, e in zip(mg["chrom"], s_test, e_test)])
    mg["expected"] = np.round(lam * dep_t * (np.array(e_test) - np.array(s_test)), 2)
    mg["obs_over_expected"] = np.round(mg["n_tested"] / mg["expected"], 2)
    amb = (mg["local_depth"] / base >= a.ambiguous_ratio).to_numpy()
    mu = mg["expected"].to_numpy(dtype=float)
    k = mg["n_tested"].to_numpy()
    if alpha > 0:
        pv = nbinom.sf(k - 1, 1 / alpha, 1 / (1 + alpha * mu))
    else:
        pv = poisson.sf(k - 1, mu)
    pv = np.where(np.isfinite(mu) & ~amb, pv, np.nan)
    q = np.full(len(pv), np.nan)
    ok = np.isfinite(pv)
    if ok.any():                                   # Benjamini-Hochberg over every tested merger
        o = np.argsort(pv[ok])
        ranked = pv[ok][o] * ok.sum() / np.arange(1, ok.sum() + 1)
        qq = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
        tmp = np.empty(ok.sum())
        tmp[o] = qq
        q[ok] = tmp
    mg["p_NB"] = [float(f"{x:.3g}") if np.isfinite(x) else np.nan for x in pv]
    mg["q_BH"] = [float(f"{x:.3g}") if np.isfinite(x) else np.nan for x in q]
    mg["status_stat"] = np.where(amb, "ambiguous_coordinate", np.where(q < a.fdr, "enriched", ""))
    print(f"[INFO] statistical call (q < {a.fdr:g}): " + ", ".join(
        f"{k_}={v}" for k_, v in mg.loc[mg["status_stat"] != "", "status_stat"].value_counts().items()))

    T = a.zone_min
    NT = a.norm_gt if a.norm_gt is not None else T - 1
    print(f"[INFO] zone rule: molecules >= {T} (integer counts); normalised > {NT:g}")
    paper = mg["n_all"] >= T

    def status(n_all, n_ret, nor):
        if n_all >= T:
            if n_ret < T:
                return "lost_audit"
            if not np.isfinite(nor):
                return "no_depth"
            return "keeps" if nor > NT else "lost_CN_correction"
        if np.isfinite(nor) and nor > NT and n_ret >= a.min_raw:
            return "new_candidate"
        if n_all == T - 1:
            return "borderline"
        return ""
    mg["status"] = [status(x, y, z) for x, y, z in zip(mg["n_all"], mg["n_retained"], mg[norm_col])]
    mg["status_local"] = [status(x, y, z) for x, y, z in zip(mg["n_all"], mg["n_retained"], mg["normalised_retained"])]
    mg["status_segment"] = [status(x, y, z) for x, y, z in
                            zip(mg["n_all"], mg["n_retained"], mg["normalised_retained_segment"])]
    print(f"[INFO] status driven by {a.depth_scope} depth (both normalisations reported)")
    mg["paper_zone"] = paper
    cols = ["chrom", "start", "end", "span_bp", "paper_zone", "status", "CN_State", "n_CN_states", "n_all",
            "n_retained", "not_retained", "local_depth", "segment_depth", "local_vs_segment", "normalised_all",
            "normalised_retained", "normalised_retained_segment", "status_local", "status_segment",
            "n_tested", "expected", "obs_over_expected", "p_NB", "q_BH", "status_stat"]
    out = mg[(mg["status_local"] != "") | (mg["status_segment"] != "") | (mg["status_stat"] != "")][cols].sort_values(["paper_zone", "status", "chrom", "start"],
                                                    ascending=[False, True, True, True])
    out.to_csv(f"{a.out_prefix}_zones.tsv", sep="\t", index=False)
    print(out.drop(columns=["not_retained", "status_local", "n_CN_states", "span_bp", "paper_zone"]).to_string(index=False))

    z = mg[paper]
    cen = z["test_center"].astype(int) if a.fixed_window > 0 else (z["start"] + z["end"]) // 2
    pd.DataFrame({"chrom": z["chrom"], "s": cen - 1000, "e": cen + 1000,
                  "name": z["status"] + "|" + z["CN_State"] + "|" + z["status_stat"].replace("", "not_enriched")}).to_csv(
        f"{a.out_prefix}_zones_2kb.bed", sep="\t", header=False, index=False)
    single, zone = a.type_names
    m["type"] = np.where(m["merger"].map(mg["n_all"]) >= T, zone, single)
    # molecules left from a zone that lost the audit: own label, so panel F does not plot the fragments as zones
    m.loc[m["merger"].map(mg["status"]).eq("lost_audit"), "type"] = f"{zone}_lost_audit"
    m[["chrom", "start", "end", "type"]].to_csv(f"{a.out_prefix}_typed.bed", sep="\t", header=False, index=False)
    print("[INFO] status of paper zones: " + ", ".join(f"{k}={v}" for k, v in z["status"].value_counts().items()))
    for s in ["_zones.tsv", "_zones_2kb.bed", "_typed.bed"]:
        print(f"[OK] {a.out_prefix}{s}")


if __name__ == "__main__":
    main()
