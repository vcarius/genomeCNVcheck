#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verifyRegionsSummary_V3.py — combine replication events with copy number (CN).

Input events (-a): either the TSV written by verifyRegions.py (structural audit) or a plain BED
(e.g. raw per-read origin/termination calls from DNAscent forkSense). Plain BEDs are treated as
'NOT_AUDITED'. Events are NEVER discarded because of their CN: CN is an annotation.

What it computes
  1. Annotation per event: CN of the overlapped segment(s) (overlap-weighted), CN state of the
     segment with the largest overlap, distance to the nearest N gap, final classification:
       DISCARDED_STRUCTURAL    collapse / hyper-coverage / break / no spanning reads (audit only)
       UNASSESSED_NO_COVERAGE  no read touched the core (audit only)
       UNASSESSED_IN_GAP       event midpoint inside an N gap or outside analysed windows
       RETAINED / NOT_AUDITED
  2. Observed/expected (O/E) events per CN state. Default null: CIRCULAR SHIFT — all events of a
     chromosome are shifted together by a random offset (>= --min-shift, wrapping around) in the
     gap-free space, preserving their number, order, spacing and clustering. Use for LOCI (one
     record per site): it tests whether sites are enriched or depleted in CN states.
  3. Rate per Mb of read sequence per CN state (events / sum of depth x window length), relative to
     the disomic state, with CIs from resampling CN segments. Use for PER-READ calls: if CN does not
     change firing per molecule, the rate is the same in every state even though raw counts scale
     with copy number. The denominator comes from the depth in window_cnv_calls.tsv; ideally that
     depth comes from the same reads analysed by DNAscent.

Usage
  python3 verifyRegionsSummary_V3.py -a ORI.audit.tsv|origins.bed -s cnv_segments.bed \
      -w window_cnv_calls.tsv -g windows_gc.tsv -o OUT/ORI [--perm-mode circular] [--n-perm 5000]
Outputs
  <o>_annotated.tsv, <o>_OE_by_CN_state.tsv, <o>_rate_by_CN_state.tsv, <o>_summary.txt
"""
import argparse
import sys

import numpy as np
import pandas as pd

STRUCTURAL_ARTIFACT = {
    "CRITICAL_ERROR_CHIMERA_OR_BREAK_IN_ROI",
    "WARNING_ZERO_SPANNING_READS",
    "REPEAT_COLLAPSE_IN_ROI",
    "WARNING_LOCAL_HIGH_COVERAGE_CNV",
}
UNASSESSED = {"NO_CORE_COVERAGE"}
STATE_ORDER = ["loss_0copies", "loss_1copy", "disomic", "gain_3copies", "gain_4copies",
               "gain_5copies", "amplified"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-a", "--audit", required=True, help="verifyRegions.py TSV, or a plain BED of events")
    p.add_argument("-s", "--segments", required=True, help="cnv_segments.bed (analyze_cnv.py)")
    p.add_argument("-w", "--windows", required=True, help="window_cnv_calls.tsv (analyze_cnv.py)")
    p.add_argument("-g", "--gc", default=None,
                   help="windows_gc.tsv (frac_N) to locate N gaps; without it, positions outside the "
                        "analysed windows are treated as gaps")
    p.add_argument("-o", "--out-prefix", required=True)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--perm-mode", choices=["circular", "event"], default="circular",
                   help="circular (default) keeps event clustering; event places each event independently "
                        "(ignores clustering; optimistic p-values)")
    p.add_argument("--min-shift", type=int, default=25000, help="Minimum circular shift, analysable bp (default 25000)")
    p.add_argument("--depth-col", default="depth_eff", help="Depth column for the rate denominator (default depth_eff)")
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstrap replicates for rate CIs (default 2000)")
    p.add_argument("--near-gap-bp", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------------------------- input
def read_events(path):
    """verifyRegions.py TSV (has a header with Chrom/Start/End) or a plain BED."""
    with open(path) as fh:
        first = next((l for l in fh if l.strip() and not l.startswith(("track", "browser"))), "")
    cols = first.rstrip("\n").lstrip("#").split("\t")
    if {"Chrom", "Start", "End"}.issubset(cols):
        ev = pd.read_csv(path, sep="\t", dtype={"Chrom": str})
        ev = ev.drop(columns=[c for c in ["Global_Copy_Number", "Global_CNV_State", "Final_Classification",
                                          "Segment_CN", "Copies_Est", "CN_State", "N_Segments_Overlapped",
                                          "Dist_To_Gap_bp", "Near_Gap", "In_Gap"] if c in ev.columns])
        if "Structural_Status" not in ev.columns:
            ev["Structural_Status"] = "NOT_AUDITED"
        return ev, True
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            rows.append((p[0], int(p[1]), int(p[2]), p[3] if len(p) > 3 else "."))
    ev = pd.DataFrame(rows, columns=["Chrom", "Start", "End", "Region_ID"])
    ev["Structural_Status"] = "NOT_AUDITED"
    return ev, False


def read_segments(path):
    s = pd.read_csv(path, sep="\t", na_values=["NA"])
    s = s.rename(columns={s.columns[0]: "chrom"})          # '#chrom' (new) or 'chrom' (old)
    s["chrom"] = s["chrom"].astype(str)
    return s.sort_values(["chrom", "start"]).reset_index(drop=True)


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def gap_intervals(windows, gc_path):
    gaps = {}
    if gc_path:
        gc = pd.read_csv(gc_path, sep="\t", na_values=["NA"], dtype={0: str})
        gc.columns = ["chrom", "start", "end"] + list(gc.columns[3:])
        for c, g in gc[gc["frac_N"] > 0.5].groupby("chrom"):
            m = merge(zip(g["start"].astype(int), g["end"].astype(int)))
            gaps[c] = (np.array([x for x, _ in m]), np.array([y for _, y in m]))
        return gaps
    for c, g in windows.groupby("chrom"):                    # holes between analysed windows
        g = g.sort_values("start")
        prev_end, nxt = g["end"].values[:-1], g["start"].values[1:]
        keep = nxt > prev_end
        gaps[c] = (prev_end[keep], nxt[keep])
    return gaps


# ----------------------------------------------------------------------------- annotation
def annotate(ev, segs, windows, gaps):
    n = len(ev)
    seg_cn = np.full(n, np.nan)
    seg_state = np.full(n, "gap_or_unassessed", dtype=object)
    n_ov = np.zeros(n, dtype=int)
    mid_seg = np.full(n, -1)
    mid_state = np.full(n, None, dtype=object)
    in_gap = np.zeros(n, dtype=bool)
    d_gap = np.full(n, np.inf)

    seg_by = {c: g for c, g in segs.groupby("chrom")}
    win_by = {c: g.sort_values("start") for c, g in windows.groupby("chrom")}
    for c, idx in ev.groupby("Chrom").groups.items():
        idx = np.asarray(idx)
        s = ev.loc[idx, "Start"].to_numpy()
        e = ev.loc[idx, "End"].to_numpy()
        mid = (s + e) // 2

        g = seg_by.get(c)
        if g is not None:
            gs, ge = g["start"].to_numpy(), g["end"].to_numpy()
            gcn, gst = g["median_cn"].to_numpy(), g["state"].to_numpy(dtype=object)
            i0 = np.searchsorted(ge, s, side="right")
            i1 = np.searchsorted(gs, e, side="left")
            for k in range(len(idx)):
                if i1[k] <= i0[k]:
                    continue
                sl = slice(i0[k], i1[k])
                w = np.minimum(ge[sl], e[k]) - np.maximum(gs[sl], s[k])
                ok = (w > 0) & np.isfinite(gcn[sl])
                if not ok.any():
                    continue
                seg_cn[idx[k]] = np.average(gcn[sl][ok], weights=w[ok])
                seg_state[idx[k]] = gst[sl][np.argmax(np.where(ok, w, -1))]
                n_ov[idx[k]] = int(ok.sum())

        wg = win_by.get(c)
        if wg is not None:
            st, en = wg["start"].to_numpy(), wg["end"].to_numpy()
            i = np.searchsorted(st, mid, side="right") - 1
            hit = (i >= 0) & (mid < en[np.clip(i, 0, len(en) - 1)])
            ii = np.clip(i, 0, len(en) - 1)
            mid_seg[idx[hit]] = wg["seg_id"].to_numpy()[ii[hit]]
            mid_state[idx[hit]] = wg["seg_state"].to_numpy(dtype=object)[ii[hit]]
            outside = ~hit
        else:
            outside = np.ones(len(idx), dtype=bool)

        gp = gaps.get(c)
        if gp is not None and len(gp[0]):
            gs_, ge_ = gp
            j = np.searchsorted(gs_, mid, side="right") - 1
            jj = np.clip(j, 0, len(gs_) - 1)
            mid_in_gap = (j >= 0) & (mid < ge_[jj])
            # region-to-gap distance (0 if overlapping)
            jr = np.searchsorted(gs_, e, side="left")          # first gap starting at/after region end
            jl = np.searchsorted(ge_, s, side="right") - 1      # last gap ending at/before region start
            dl = np.where(jl >= 0, s - ge_[np.clip(jl, 0, len(ge_) - 1)], np.inf)
            dr = np.where(jr < len(gs_), gs_[np.clip(jr, 0, len(gs_) - 1)] - e, np.inf)
            ovl = np.zeros(len(idx), dtype=bool)
            k0 = np.searchsorted(ge_, s, side="right")
            k1 = np.searchsorted(gs_, e, side="left")
            ovl = k1 > k0
            d_gap[idx] = np.where(ovl, 0, np.minimum(dl, dr))
        else:
            mid_in_gap = np.zeros(len(idx), dtype=bool)
        in_gap[idx] = mid_in_gap | outside

    ev = ev.copy()
    ev["Segment_CN"] = np.round(seg_cn, 3)
    ev["Copies_Est"] = np.round(2 * seg_cn, 2)
    ev["CN_State"] = np.where(in_gap, "gap_or_unassessed", seg_state)
    ev["N_Segments_Overlapped"] = n_ov
    ev["Midpoint_Seg_ID"] = mid_seg
    ev["Midpoint_CN_State"] = mid_state
    ev["In_Gap"] = in_gap
    ev["Dist_To_Gap_bp"] = d_gap
    return ev


def classify(row):
    if row["Structural_Status"] in STRUCTURAL_ARTIFACT:
        return "DISCARDED_STRUCTURAL"
    if row["Structural_Status"] in UNASSESSED:
        return "UNASSESSED_NO_COVERAGE"
    if row["In_Gap"]:
        return "UNASSESSED_IN_GAP"
    return "NOT_AUDITED" if row["Structural_Status"] == "NOT_AUDITED" else "RETAINED"


# ----------------------------------------------------------------------------- O/E
def _oe_rows(states, obs, null, n_used, genome_frac):
    rows = []
    for k, s in enumerate(states):
        o, e = obs[k], null[:, k].mean()
        p_hi = (1 + (null[:, k] >= o).sum()) / (len(null) + 1)
        p_lo = (1 + (null[:, k] <= o).sum()) / (len(null) + 1)
        lo, hi = np.quantile(null[:, k], [0.025, 0.975])
        rows.append({
            "CN_State": s, "Observed": int(o),
            "Observed_pct": round(100 * o / n_used, 2) if n_used else np.nan,
            "Genome_pct": round(100 * genome_frac[s], 2),
            "Expected_null": round(e, 1), "Null_95pct_range": f"{lo:.0f}-{hi:.0f}",
            "OE_vs_genome": round((o / n_used) / genome_frac[s], 3) if n_used and genome_frac[s] > 0 else np.nan,
            "OE_vs_null": round(o / e, 3) if e > 0 else np.nan,
            "p_two_sided": round(min(1.0, 2 * min(p_hi, p_lo)), 4),
        })
    return pd.DataFrame(rows)


def oe_test(events, windows, n_perm, seed, min_shift_bp, mode, chunk=250):
    """O/E per CN state. Observed and null both use the state of the window holding the event midpoint.
    circular: every event of a chromosome moves by the same number of analysed windows (with wrap-around).
    event:    every event is placed independently in a random analysed window of its chromosome."""
    rng = np.random.default_rng(seed)
    w = windows[windows["seg_state"].isin(STATE_ORDER)].sort_values(["chrom", "start"])
    states = [s for s in STATE_ORDER if s in set(w["seg_state"])]
    sidx = {s: i for i, s in enumerate(states)}
    win = int((w["end"] - w["start"]).median())
    kmin = max(1, int(np.ceil(min_shift_bp / win)))
    genome_frac = (w.groupby("seg_state").size() / len(w)).reindex(states).fillna(0)

    obs = np.zeros(len(states))
    null = np.zeros((n_perm, len(states)))
    n_used = n_dropped = 0
    for chrom, g in events.groupby("Chrom"):
        wc = w[w["chrom"] == chrom]
        if wc.empty:
            n_dropped += len(g)
            continue
        st_, en_ = wc["start"].to_numpy(), wc["end"].to_numpy()
        lab = np.array([sidx[x] for x in wc["seg_state"]])
        mids = ((g["Start"] + g["End"]) // 2).to_numpy()
        i = np.searchsorted(st_, mids, side="right") - 1
        ok = (i >= 0) & (mids < en_[np.clip(i, 0, len(en_) - 1)])
        n_dropped += int((~ok).sum())
        i = i[ok]
        if i.size == 0:
            continue
        n_used += i.size
        obs += np.bincount(lab[i], minlength=len(states))
        n_c = len(wc)
        if mode == "circular" and n_c <= 2 * kmin:
            null += np.bincount(lab[i], minlength=len(states))      # too short to shift: uninformative
            continue
        for c0 in range(0, n_perm, chunk):                           # chunks keep memory bounded
            m = min(chunk, n_perm - c0)
            if mode == "circular":
                d = rng.integers(kmin, n_c - kmin + 1, size=m)
                pos = (i[None, :] + d[:, None]) % n_c
            else:
                pos = rng.integers(0, n_c, size=(m, i.size))
            L = lab[pos]
            for k in range(len(states)):
                null[c0:c0 + m, k] += (L == k).sum(axis=1)
    if n_dropped:
        print(f"[INFO] {n_dropped} events with midpoint outside analysed windows were left out of the O/E test")
    return _oe_rows(states, obs, null, n_used, genome_frac)


# ----------------------------------------------------------------------------- rate per read Mb
def rate_by_state(events, windows, depth_col, n_boot, seed):
    """Events per Mb of read sequence in each CN state; CI by resampling CN segments within each state."""
    rng = np.random.default_rng(seed)
    w = windows[windows["seg_state"].isin(STATE_ORDER) & (windows["seg_id"] >= 0)].copy()
    if depth_col not in w.columns:
        sys.exit(f"[ERROR] column '{depth_col}' not found in window_cnv_calls.tsv")
    w["read_bp"] = w[depth_col].clip(lower=0) * (w["end"] - w["start"])
    seg = w.groupby("seg_id").agg(state=("seg_state", "first"), read_bp=("read_bp", "sum"))
    ev = events[events["Midpoint_Seg_ID"] >= 0]
    seg["n_events"] = ev.groupby("Midpoint_Seg_ID").size().reindex(seg.index).fillna(0)
    states = [s for s in STATE_ORDER if s in set(seg["state"])]

    def rates(table):
        t = table.groupby("state")[["n_events", "read_bp"]].sum()
        return (t["n_events"] / (t["read_bp"] / 1e6)).reindex(states)

    point = rates(seg)
    groups = {s: seg[seg["state"] == s] for s in states}
    boots = np.empty((n_boot, len(states)))
    for b in range(n_boot):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups.values()]
        boots[b] = rates(pd.concat(parts)).to_numpy()
    ref = states.index("disomic") if "disomic" in states else None
    rows = []
    for k, s in enumerate(states):
        r = {"CN_State": s, "n_segments": len(groups[s]),
             "n_events": int(groups[s]["n_events"].sum()),
             "read_Mb": round(groups[s]["read_bp"].sum() / 1e6, 2),
             "events_per_read_Mb": round(point[s], 4),
             "rate_CI95": f"{np.nanquantile(boots[:, k], 0.025):.4f}-{np.nanquantile(boots[:, k], 0.975):.4f}"}
        if ref is not None:
            rel = boots[:, k] / boots[:, ref]
            r["rate_vs_disomic"] = round(point[s] / point["disomic"], 3)
            r["rate_vs_disomic_CI95"] = f"{np.nanquantile(rel, 0.025):.3f}-{np.nanquantile(rel, 0.975):.3f}"
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    ev, audited = read_events(a.audit)
    ev["Chrom"] = ev["Chrom"].astype(str)
    segs = read_segments(a.segments)
    windows = pd.read_csv(a.windows, sep="\t", dtype={"chrom": str}, na_values=["NA"])
    if "seg_state" not in windows.columns:
        sys.exit("[ERROR] window_cnv_calls.tsv has no 'seg_state': use analyze_cnv.py output.")
    windows["seg_id"] = windows["seg_id"].fillna(-1).astype(int)

    ev = annotate(ev, segs, windows, gap_intervals(windows, a.gc))
    ev["Near_Gap"] = ev["Dist_To_Gap_bp"] <= a.near_gap_bp
    ev["Final_Classification"] = ev.apply(classify, axis=1)
    ev.to_csv(f"{a.out_prefix}_annotated.tsv", sep="\t", index=False)

    kept_label = "RETAINED" if audited else "NOT_AUDITED"
    assessed = ev[ev["Final_Classification"] != "UNASSESSED_IN_GAP"]
    kept = ev[ev["Final_Classification"] == kept_label]
    oe = pd.concat([oe_test(assessed, windows, a.n_perm, a.seed, a.min_shift, a.perm_mode).assign(Set="all_assessed"),
                    oe_test(kept, windows, a.n_perm, a.seed, a.min_shift, a.perm_mode).assign(Set="retained")],
                   ignore_index=True)
    oe = oe[["Set"] + [c for c in oe.columns if c != "Set"]]
    oe.to_csv(f"{a.out_prefix}_OE_by_CN_state.tsv", sep="\t", index=False)
    rate = rate_by_state(kept, windows, a.depth_col, a.n_boot, a.seed)
    rate.to_csv(f"{a.out_prefix}_rate_by_CN_state.tsv", sep="\t", index=False)

    n = len(ev)
    L = ["=" * 78, f" SUMMARY — {a.audit}", "=" * 78, f"Events: {n}"]
    for k, v in ev["Final_Classification"].value_counts().items():
        L.append(f"  {k:28s}: {v:7d} ({100 * v / n:5.1f}%)")
    if audited:
        near, disc = ev["Near_Gap"], ev["Final_Classification"] == "DISCARDED_STRUCTURAL"
        L.append(f"Structural discards within {a.near_gap_bp} bp of an N gap: {100 * disc[near].mean():.1f}% "
                 f"(n={int(near.sum())}) vs farther: {100 * disc[~near].mean():.1f}% (n={int((~near).sum())})")
    L += ["-" * 78, "CN state (segment) x final classification:",
          pd.crosstab(ev["CN_State"], ev["Final_Classification"], margins=True).to_string(),
          "-" * 78,
          f"Observed/expected per CN state ({a.n_perm} permutations, mode '{a.perm_mode}'"
          + (f", minimum shift {a.min_shift} bp" if a.perm_mode == "circular" else "") + ")",
          "  -> interpret for LOCI (one record per site)",
          oe.to_string(index=False),
          "-" * 78,
          f"Events per Mb of read sequence per CN state (set '{kept_label}', depth '{a.depth_col}', "
          f"{a.n_boot} segment bootstraps)",
          "  -> interpret for PER-READ calls: a constant rate means CN does not change firing per molecule",
          rate.to_string(index=False), "=" * 78]
    txt = "\n".join(L)
    print(txt)
    with open(f"{a.out_prefix}_summary.txt", "w") as fh:
        fh.write(txt + "\n")
    for suf in ["_annotated.tsv", "_OE_by_CN_state.tsv", "_rate_by_CN_state.tsv", "_summary.txt"]:
        print(f"[OK] {a.out_prefix}{suf}")


if __name__ == "__main__":
    main()
