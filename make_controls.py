#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_controls.py — control regions for replication origins/terminations, all in one place.

MODES (choose by the question the control must answer)
  matched  CONTEXT-MATCHED control (recommended for the STRUCTURAL AUDIT). Each target gets a
           partner of the same size in the same chromosome, CN state (segment), distance-to-N-gap class
           and mappability class, at least --margin bp (edge to edge, default 1 kb) from any event.
           Fallback L1 keeps every criterion but the chromosome (see MATCHED STRATA).
           Do NOT use it for the CN question: matching on CN removes the difference by design.
  shift    CIRCULAR SHIFT (recommended for the CN question and as audit sensitivity). The whole set
           of each chromosome moves by one random offset (>= --min-shift, wrapping around) in the
           gap-free space. Preserves number, size, order, spacing and clustering. K sets (--k).
  desert   EVENT DESERTS (descriptive: where no event is detected). Regions >= --margin (15 kb)
           from any event, >= --min-spacing apart.
  random   UNIFORM RANDOM outside N gaps and outside event cores (--margin 200 bp). Naive reference.
  all      All four with their default parameters, plus one balance report.

  coverage (optional, not in 'all') positions drawn proportionally to read bp (--depth-col/--depth-bed),
           overlaps allowed, events not avoided.

PER-MOLECULE TARGETS (e.g. forkSense calls, one region per read; no merging, no classes)
  Several molecules fire at the same place, so their regions overlap. Each set of mutually overlapping
  molecules (an "overlap group", connected by direct overlap only; nothing is merged or classified) is
  moved as a unit: the control receives one region per molecule, with the same sizes and the same
  offsets inside the group. Every control set therefore has the same number of regions, the same
  overlap pattern and the same multiplicity as the targets; only the position changes. matched pairs
  the group (same chromosome, CN state, gap-distance class and mappability as the group midpoint);
  shift moves every group of a chromosome by the same offset; random places each group once (same
  chromosome) in event-free space; desert places one region per group (descriptive only).
  Molecules of one group, and of one block, are not independent: resample by 'block'.
  --collapse yes merges the molecules into loci instead (bedtools merge -d --collapse-distance).

INPUTS
  -t  target BED (regions that need controls, e.g. origins.bed); sizes may vary
  -e  BED(s) of ALL events to avoid (origins and terminations); the target is always included
  -g  windows_gc.tsv from run_cnv.sh (chrom start end gc_acgt frac_N, with header)
  -w  window_cnv_calls.tsv from analyze_cnv.py (needed by 'matched'; used by the balance report)
  -o  output prefix (e.g. controls/ORI)

OUTPUTS
  <o>.<mode>.bed (shift: <o>.shift_<k>.bed), 8 columns:
      chrom start end name mode match_level_or_flag stratum block
    - name links a matched control to its target (ctrl_matched_T00012 <-> T00012)
    - block = cluster of regions whose flanks (+/- --flank) overlap. Regions in a block are NOT
      independent: use the block as the resampling unit in statistics.
    The first 4 columns can be passed directly to verifyRegions.py (-b).
  <o>.target_annotated.bed   the same annotation for the targets (Txxxxx, stratum, block)
  <o>.target_loci.bed        only with --collapse yes: the loci (with n_molecules)
  <o>.balance.tsv            genomic context of the targets vs. each control set

MATCHED STRATA
  stratum = chromosome | CN state | distance to the nearest N gap (<5, 5-10, 10-25, 25-50, 50-200,
  >200 kb) | mappability (low_mappability from analyze_cnv.py --compare). Candidate windows noisier
  in CN than 95% of the targets are excluded (--disp-support). Fallback L1 keeps every criterion but the
  chromosome; relaxing the gap class needs --relax-gap.
  Pass 1 gives every control its own position. When event-free sequence runs out (dense per-molecule
  events leave little of it), pass 2 lets the remaining targets use a position already taken by another
  control in the same stratum (matching with replacement; flag 'reused_position'), so the controls
  represent all targets rather than the subset that happened to fit. Shared positions fall in the same
  block, so block resampling accounts for them. --no-reuse keeps pass 1 only. 'random' does the same
  (flag 'reused_space').

EXAMPLES
  python3 make_controls.py all     -t origins.bed -e origins.bed terminations.bed -g windows_gc.tsv -w window_cnv_calls.tsv -o controls/ORI
  python3 make_controls.py matched -t origins.bed -e origins.bed terminations.bed -g windows_gc.tsv -w window_cnv_calls.tsv -o controls/ORI --ratio 2
  python3 make_controls.py shift   -t origins.bed -e origins.bed terminations.bed -g windows_gc.tsv -o controls/ORI --k 5 [--drop-overlapping]
"""
import argparse
import bisect
import os
import sys

import numpy as np
import pandas as pd

GAP_BINS = [-1, 5_000, 10_000, 25_000, 50_000, 200_000, np.inf]
GAP_LABELS = ["<5kb", "5-10kb", "10-25kb", "25-50kb", "50-200kb", ">200kb"]
STATE_ORDER = ["loss_0copies", "loss_1copy", "disomic", "gain_3copies", "gain_4copies",
               "gain_5copies", "amplified"]
OUT_COLS = ["chrom", "start", "end", "name", "mode", "match_or_flag", "stratum", "block"]


# ----------------------------------------------------------------------------- helpers
def read_bed(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            rows.append((p[0], int(p[1]), int(p[2]), p[3] if len(p) > 3 else "."))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "orig_name"])


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def collapse_loci(df, dist, std_bp=0):
    """Merge regions into loci with bedtools merge -d semantics (next start - current end <= dist;
    dist = -1 merges strict overlaps only). Keeps the merged span and the number of regions per locus;
    with std_bp > 0 the locus becomes std_bp centred on the span midpoint (the paper's +/- 1 kb)."""
    rows = []
    d = df.sort_values(["chrom", "start", "end"])
    for c, g in d.groupby("chrom", sort=False):
        cs = ce = None
        n = 0
        for s, e in zip(g["start"].to_numpy(), g["end"].to_numpy()):
            if cs is not None and s - ce <= dist:
                ce, n = max(ce, e), n + 1
            else:
                if cs is not None:
                    rows.append((c, cs, ce, n))
                cs, ce, n = int(s), int(e), 1
        if cs is not None:
            rows.append((c, cs, ce, n))
    loci = pd.DataFrame(rows, columns=["chrom", "span_start", "span_end", "n_molecules"])
    if std_bp > 0:
        mid = (loci["span_start"] + loci["span_end"]) // 2
        loci["start"] = (mid - std_bp // 2).clip(lower=0)
        loci["end"] = loci["start"] + std_bp
    else:
        loci["start"], loci["end"] = loci["span_start"], loci["span_end"]
    loci["orig_name"] = "n_molecules=" + loci["n_molecules"].astype(str)
    return loci


def n_overlapping(df):
    """Number of regions that overlap at least one other region."""
    ov = collapse_loci(df, -1)
    return int(ov.loc[ov["n_molecules"] > 1, "n_molecules"].sum())


def overlap_groups(df):
    """Group id per row: connected components of directly overlapping regions (same chromosome)."""
    order = np.lexsort((df["end"].to_numpy(), df["start"].to_numpy(), df["chrom"].to_numpy()))
    ch, st, en = df["chrom"].to_numpy(), df["start"].to_numpy(), df["end"].to_numpy()
    gid = np.empty(len(df), dtype=np.int64)
    cur_chr, cur_end, n = None, -1, -1
    for k in order:
        if ch[k] != cur_chr or st[k] >= cur_end:
            n += 1
            cur_chr, cur_end = ch[k], en[k]
        else:
            cur_end = max(cur_end, en[k])
        gid[k] = n
    return gid


def group_table(target):
    """One row per overlap group: span, size, members (name, offset start, offset end)."""
    rows = []
    for g, m in target.groupby("group", sort=True):
        gs, ge = int(m["start"].min()), int(m["end"].max())
        mem = list(zip(m["name"], (m["start"] - gs).astype(int), (m["end"] - gs).astype(int)))
        rows.append((m["chrom"].iloc[0], gs, ge, ge - gs, len(m), mem, f"G{g:05d}"))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "size", "n_members", "members", "gname"])


def expand(c, s, grp_row, mode, flag, suffix=""):
    """Control rows for every member of a group whose span starts at s."""
    return [(c, s + o0, s + o1, f"ctrl_{mode}_{nm}{suffix}", mode, flag) for nm, o0, o1 in grp_row.members]


def dist_to_intervals(x, starts, ends):
    """Distance from each point x to the nearest interval (0 inside). Intervals merged and sorted."""
    x = np.asarray(x)
    if len(starts) == 0:
        return np.full(x.shape, np.inf)
    starts, ends = np.asarray(starts), np.asarray(ends)
    i = np.searchsorted(starts, x, side="right") - 1
    il = np.clip(i, 0, len(starts) - 1)
    inside = (i >= 0) & (x < ends[il])
    d = np.full(x.shape, np.inf)
    d[inside] = 0
    left = (i >= 0) & ~inside
    d[left] = x[left] - ends[il][left] + 1
    ir = np.clip(i + 1, 0, len(starts) - 1)
    right = (i + 1 < len(starts)) & ~inside
    d[right] = np.minimum(d[right], starts[ir][right] - x[right])
    return d


def overlaps_any(starts, ends, s, e):
    """Do [s, e) and any of the merged, sorted intervals overlap?"""
    i = bisect.bisect_right(starts, e - 1) - 1
    return i >= 0 and ends[i] > s


def assign_blocks(df, flank, tag):
    """Clusters of regions whose +/- flank windows overlap (positional, safe with any index)."""
    order = np.lexsort((df["start"].to_numpy(), df["chrom"].to_numpy()))
    ch, st, en = df["chrom"].to_numpy(), df["start"].to_numpy(), df["end"].to_numpy()
    blocks = np.empty(len(df), dtype=object)
    cur_chr, cur_end, n = None, -1, 0
    for k in order:
        s, e = st[k] - flank, en[k] + flank
        if ch[k] != cur_chr or s > cur_end:
            n += 1
            cur_chr, cur_end = ch[k], e
        else:
            cur_end = max(cur_end, e)
        blocks[k] = f"{tag}_B{n:05d}"
    return blocks


class Genome:
    """Gaps, chromosome sizes, events and per-window context."""

    def __init__(self, gc_path, win_path, event_frames, event_pad, match_map=True):
        gc = pd.read_csv(gc_path, sep="\t", na_values=["NA"], dtype={0: str})
        gc.columns = ["chrom", "start", "end"] + list(gc.columns[3:])
        if "frac_N" not in gc.columns:
            sys.exit("[ERROR] windows_gc.tsv has no frac_N column (use the updated run_cnv.sh).")
        self.sizes = gc.groupby("chrom")["end"].max().astype(int).to_dict()
        self.gaps, self.analyzable = {}, {}
        for c, g in gc[gc["frac_N"] > 0.5].groupby("chrom"):
            m = merge(zip(g["start"].astype(int), g["end"].astype(int)))
            self.gaps[c] = ([a for a, _ in m], [b for _, b in m])
        for c, g in gc[gc["frac_N"] <= 0.5].groupby("chrom"):
            self.analyzable[c] = merge(zip(g["start"].astype(int), g["end"].astype(int)))

        ev = pd.concat(event_frames, ignore_index=True)
        self.events, self.events_padded = {}, {}
        for c, g in ev.groupby("chrom"):
            m = merge(zip(g["start"].astype(int), g["end"].astype(int)))
            self.events[c] = ([a for a, _ in m], [b for _, b in m])
            mp = merge([(max(0, a - event_pad), b + event_pad) for a, b in m])
            self.events_padded[c] = ([a for a, _ in mp], [b for _, b in mp])

        self.win = None
        if win_path:
            w = pd.read_csv(win_path, sep="\t", dtype={"chrom": str}, na_values=["NA"])
            if not {"chrom", "start", "end", "seg_state", "log2_cn"}.issubset(w.columns):
                sys.exit("[ERROR] window_cnv_calls.tsv lacks seg_state/log2_cn (use analyze_cnv.py).")
            w = w.sort_values(["chrom", "start"]).reset_index(drop=True)
            w["seg_state"] = w["seg_state"].where(w["seg_state"].isin(STATE_ORDER), np.nan)
            w["mid"] = (w["start"] + w["end"]) // 2
            # local CN dispersion (+/- 10 windows): median absolute deviation around the local median
            w["local_disp"] = w.groupby("chrom")["log2_cn"].transform(
                lambda s: (s - s.rolling(21, center=True, min_periods=5).median()).abs()
                .rolling(21, center=True, min_periods=5).median())
            w["d_gap"] = np.inf
            w["d_event"] = np.inf
            for c, g in w.groupby("chrom"):
                gs, ge = self.gaps.get(c, ([], []))
                es, ee = self.events.get(c, ([], []))
                w.loc[g.index, "d_gap"] = dist_to_intervals(g["mid"].values, gs, ge)
                w.loc[g.index, "d_event"] = dist_to_intervals(g["mid"].values, es, ee)
            w["gap_bin"] = pd.cut(w["d_gap"], GAP_BINS, labels=GAP_LABELS).astype(str)
            w["lowmap"] = (w["low_mappability"].astype(str).str.lower().eq("true")
                           if "low_mappability" in w.columns else False)
            w["map_class"] = np.where(w["lowmap"], "lowmap", "map") if match_map else "any"
            self.win = w

    def context_of(self, df):
        """Context of the window holding each region's midpoint (vectorised per chromosome)."""
        out = df.copy().reset_index(drop=True)
        n = len(out)
        mids = ((out["start"] + out["end"]) // 2).to_numpy()
        seg_state = np.full(n, "NA", dtype=object)
        map_class = np.full(n, "NA" if self.win is not None else "any", dtype=object)
        lowmap = np.zeros(n, dtype=bool)
        local_disp = np.full(n, np.nan)
        d_gap = np.full(n, np.inf)
        for c, idx in out.groupby("chrom").groups.items():
            idx = np.asarray(idx)
            m = mids[idx]
            gs, ge = self.gaps.get(c, ([], []))
            d_gap[idx] = dist_to_intervals(m, gs, ge)
            if self.win is None:
                continue
            g = self.win[self.win["chrom"] == c]
            if g.empty:
                continue
            st, en = g["start"].to_numpy(), g["end"].to_numpy()
            i = np.searchsorted(st, m, side="right") - 1
            ii = np.clip(i, 0, len(st) - 1)
            hit = (i >= 0) & (m < en[ii])
            ss = g["seg_state"].to_numpy(dtype=object)[ii]
            seg_state[idx[hit]] = np.where(pd.isna(ss[hit]), "NA", ss[hit])
            local_disp[idx[hit]] = g["local_disp"].to_numpy()[ii[hit]]
            map_class[idx[hit]] = g["map_class"].to_numpy(dtype=object)[ii[hit]]
            lowmap[idx[hit]] = g["lowmap"].to_numpy(dtype=bool)[ii[hit]]
        out["seg_state"] = seg_state
        out["map_class"] = map_class
        out["lowmap"] = lowmap
        out["local_disp"] = local_disp
        out["d_gap"] = d_gap
        out["gap_bin"] = pd.cut(out["d_gap"], GAP_BINS, labels=GAP_LABELS).astype(str)
        out["stratum"] = out["chrom"] + "|" + out["seg_state"] + "|" + out["gap_bin"] + "|" + out["map_class"]
        ov = np.zeros(n, dtype=bool)
        for k, (c, s, e) in enumerate(zip(out["chrom"], out["start"], out["end"])):
            ps, pe = self.events_padded.get(c, ([], []))
            ov[k] = overlaps_any(ps, pe, s, e)
        out["overlaps_event"] = ov
        return out


class Occupancy:
    """Accepted control intervals per chromosome; rejects overlaps closer than 'spacing'."""

    def __init__(self, spacing):
        self.spacing = spacing
        self.iv = {}

    def try_add(self, c, s, e):
        lst = self.iv.setdefault(c, [])
        i = bisect.bisect_left(lst, (s, e))
        if i > 0 and lst[i - 1][1] + self.spacing > s:
            return False
        if i < len(lst) and e + self.spacing > lst[i][0]:
            return False
        lst.insert(i, (s, e))
        return True


# ----------------------------------------------------------------------------- modes
def mode_matched(G, target, grp, a, rng):
    if G.win is None:
        sys.exit("[ERROR] 'matched' needs -w window_cnv_calls.tsv")
    w = G.win
    min_half = int(grp["size"].min()) // 2
    elig = w[(w["d_event"] >= a.margin + min_half) & (w["d_gap"] >= min_half) & w["seg_state"].notna()].copy()
    if a.exclude_low_mappability:
        elig = elig[~elig["lowmap"]]
    # common support on local CN noise: no candidate noisier than (almost) every target
    if a.disp_support > 0 and target["local_disp"].notna().any():
        cap = float(np.nanquantile(target["local_disp"], a.disp_support))
        n0 = len(elig)
        elig = elig[elig["local_disp"].notna() & (elig["local_disp"] <= cap)]
        print(f"[matched] {n0 - len(elig)} of {n0} candidate windows excluded: local CN dispersion > {cap:.3f} "
              f"(quantile {a.disp_support:g} of the targets)")
    elig["k_full"] = elig["chrom"] + "|" + elig["seg_state"] + "|" + elig["gap_bin"] + "|" + elig["map_class"]
    elig["k_sg"] = elig["seg_state"] + "|" + elig["gap_bin"] + "|" + elig["map_class"]
    pool_full = {k: list(g.index) for k, g in elig.groupby("k_full")}
    pool_sg = {k: list(g.index) for k, g in elig.groupby("k_sg")}
    for d in (pool_full, pool_sg):
        for k in d:
            rng.shuffle(d[k])
    W_chr, W_mid = w["chrom"].to_numpy(), w["mid"].to_numpy()
    W_dev, W_dgap = w["d_event"].to_numpy(), w["d_gap"].to_numpy()
    occ = Occupancy(a.min_spacing)

    def levels(r):
        keys = [("L0_chrom_state_gap_map", pool_full, f"{r.chrom}|{r.seg_state}|{r.gap_bin}|{r.map_class}")]
        if a.fallback:
            # every criterion but the chromosome
            keys.append(("L1_state_gap_map_anychrom", pool_sg, f"{r.seg_state}|{r.gap_bin}|{r.map_class}"))
            if a.relax_gap and r.gap_bin in GAP_LABELS:
                j = GAP_LABELS.index(r.gap_bin)
                nb = [GAP_LABELS[x] for x in (j - 1, j + 1) if 0 <= x < len(GAP_LABELS)]
                rng.shuffle(nb)
                keys += [("L2_chrom_state_adjacentgap", pool_full, f"{r.chrom}|{r.seg_state}|{b}|{r.map_class}")
                         for b in nb]
        return keys

    rows, unmatched_g, unmatched_m, left = [], 0, 0, []
    want = int(round(a.ratio))
    # largest groups first (they have the fewest candidate positions); random order within ties
    order = grp.assign(_u=rng.random(len(grp))).sort_values(["size", "_u"], ascending=[False, True])
    for r in order.itertuples():
        if r.seg_state == "NA":
            unmatched_g, unmatched_m = unmatched_g + 1, unmatched_m + r.n_members
            continue
        half, got, skipped = r.size // 2, 0, []
        for lvl, pools, key in levels(r):
            pool = pools.get(key, [])
            while got < want and pool:
                idx = pool.pop()
                # this candidate must also fit a group of THIS span
                if W_dev[idx] < a.margin + half or W_dgap[idx] < half:
                    skipped.append(idx)
                    continue
                c, s = W_chr[idx], int(W_mid[idx]) - half
                if s < 0 or s + r.size > G.sizes.get(c, 0) or not occ.try_add(c, s, s + r.size):
                    continue
                rows += expand(c, s, r, "matched", lvl, f"_{got + 1}" if want > 1 else "")
                got += 1
            pool.extend(skipped)          # too small for this group, may still fit a smaller one
            skipped = []
            if got >= want:
                break
        if got == 0:
            unmatched_g, unmatched_m = unmatched_g + 1, unmatched_m + r.n_members
            if r.seg_state != "NA":
                left.append(r)
    n1 = len(rows)
    print(f"[matched] pass 1 (distinct positions): {n1} controls for {len(target)} targets in {len(grp)} overlap "
          f"groups (ratio {want}); without a partner: {unmatched_m} targets ({unmatched_g} groups)")

    # pass 2: event-free sequence ran out. Groups still without a partner may use a position already taken
    # by another control (matching with replacement): same strata, same margin from every event.
    if a.reuse and left:
        pool_full = {k: list(g.index) for k, g in elig.groupby("k_full")}
        pool_sg = {k: list(g.index) for k, g in elig.groupby("k_sg")}
        still_g = still_m = 0
        for r in left:
            half, done = r.size // 2, False
            for lvl, pools, key in levels(r):
                pool = pools.get(key, [])
                for idx in rng.permutation(pool) if pool else []:
                    if W_dev[idx] < a.margin + half or W_dgap[idx] < half:
                        continue
                    c, s = W_chr[idx], int(W_mid[idx]) - half
                    if s < 0 or s + r.size > G.sizes.get(c, 0):
                        continue
                    rows += expand(c, s, r, "matched", f"{lvl};reused_position", "_1" if want > 1 else "")
                    done = True
                    break
                if done:
                    break
            if not done:
                still_g, still_m = still_g + 1, still_m + r.n_members
        print(f"[matched] pass 2 (positions may be shared): {len(rows) - n1} more controls; still without a "
              f"partner: {still_m + (unmatched_m - sum(x.n_members for x in left))} targets")
    out = pd.DataFrame(rows, columns=OUT_COLS[:6])
    n_distinct = len(out.drop_duplicates(["chrom", "start", "end"]))
    print(f"[matched] {len(out)} controls for {len(target)} targets; {n_distinct} distinct control regions; "
          f"levels: {out['match_or_flag'].value_counts().to_dict()}")
    return {"matched": out}


class Space:
    """Gap-free (compressed) coordinates of one chromosome."""

    def __init__(self, iv):
        self.iv = iv
        self.starts = [s for s, _ in iv]
        lens = [e - s for s, e in iv]
        self.cum = [0] + list(np.cumsum(lens)[:-1].astype(int))
        self.L = int(sum(lens))

    def to_comp(self, p):
        i = bisect.bisect_right(self.starts, p) - 1
        if i < 0:
            return 0
        s, e = self.iv[i]
        if p < e:
            return self.cum[i] + p - s
        return self.cum[i + 1] if i + 1 < len(self.iv) else self.L - 1

    def from_comp(self, x):
        i = bisect.bisect_right(self.cum, x) - 1
        return self.iv[i][0] + (x - self.cum[i]), i


def mode_shift(G, target, grp, a, rng):
    spaces = {c: Space(iv) for c, iv in G.analyzable.items() if iv}
    sets = {}
    for k in range(1, a.k + 1):
        placed, lost = [], 0
        for c, g in grp.groupby("chrom"):
            sp = spaces.get(c)
            if sp is None or sp.L < 2 * a.min_shift:
                lost += int(g["n_members"].sum())
                continue
            delta = int(rng.integers(a.min_shift, sp.L - a.min_shift + 1))
            for r in g.itertuples():
                x = (sp.to_comp((r.start + r.end) // 2) + delta) % sp.L
                nm, i = sp.from_comp(x)
                s_i, e_i = sp.iv[i]
                if e_i - s_i < r.size:                  # gap-free stretch shorter than the group
                    lost += r.n_members
                    continue
                ns = min(max(nm - r.size // 2, s_i), e_i - r.size)
                placed.append((c, ns, ns + r.size, r))
        # fitting a group inside a gap-free stretch can make two groups overlap: keep the first of each
        placed.sort(key=lambda x: (x[0], x[1]))
        rows, cur_chr, cur_end = [], None, -1
        for c, s, e, r in placed:
            if c != cur_chr or s >= cur_end:
                rows += expand(c, s, r, f"shift{k}", "")
                cur_chr, cur_end = c, e
            else:
                lost += r.n_members
        out = pd.DataFrame(rows, columns=OUT_COLS[:6])
        out["mode"] = f"shift_{k}"
        sets[f"shift_{k}"] = out
        print(f"[shift] set {k}: {len(out)} regions" + (f" ({lost} could not be placed)" if lost else ""))
    return sets


def mode_coverage(G, target, a, rng):
    """Positions drawn proportionally to read bp per window; overlaps allowed, events not avoided."""
    if G.win is None:
        sys.exit("[ERROR] 'coverage' needs -w window_cnv_calls.tsv")
    w = G.win
    if a.depth_bed:
        d = pd.read_csv(a.depth_bed, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3],
                        names=["chrom", "start", "end", "ext_depth"], dtype={"chrom": str})
        dep = w[["chrom", "start", "end"]].merge(d, on=["chrom", "start", "end"], how="left")["ext_depth"]
        if dep.isna().mean() > 0.01:
            sys.exit(f"[ERROR] windows not found in {a.depth_bed}: use the depth file of the same run_cnv.sh run")
        dep, label = dep.to_numpy(dtype=float), os.path.basename(a.depth_bed)
    else:
        if a.depth_col not in w.columns:
            sys.exit(f"[ERROR] column '{a.depth_col}' not in window_cnv_calls.tsv")
        dep, label = w[a.depth_col].to_numpy(dtype=float), a.depth_col
    wt = np.nan_to_num(dep, nan=0.0).clip(min=0) * (w["end"] - w["start"]).to_numpy()
    if "frac_N" in w.columns:
        wt[w["frac_N"].to_numpy() > 0.5] = 0
    n = int(round(len(target) * a.ratio))
    idx = rng.choice(len(w), size=n, p=wt / wt.sum())
    ws, we = w["start"].to_numpy()[idx], w["end"].to_numpy()[idx]
    pos = ws + (rng.random(n) * (we - ws)).astype(int)
    sizes = rng.choice(target["size"].to_numpy(), size=n, replace=True)
    chroms = w["chrom"].to_numpy()[idx]
    L = np.array([G.sizes.get(c, 0) for c in chroms])
    s = np.clip(pos - sizes // 2, 0, np.maximum(L - sizes, 0))
    out = pd.DataFrame({"chrom": chroms, "start": s, "end": s + sizes,
                        "name": [f"ctrl_coverage_{i + 1:06d}" for i in range(n)], "mode": "coverage",
                        "match_or_flag": f"weight={label}"})
    print(f"[coverage] {n} regions drawn proportionally to read bp ({label} x window length)")
    return out


def sample_free_space(G, grp, a, label, rng, keep_members=True, max_tries=500):
    """Random placement in free space (outside gaps and events +/- margin), same chromosome, no two
    groups closer than min_spacing (rejection sampling). keep_members: each group is placed with all its
    molecules (random); otherwise one region per group with the size of one of its molecules (desert)."""
    rows, deficit, want_tot = [], [], 0
    occ = Occupancy(a.min_spacing)
    for c, g in grp.groupby("chrom"):
        L = G.sizes.get(c, 0)
        n_want = int(round(len(g) * a.ratio))
        want_tot += int(g["n_members"].sum() * a.ratio) if keep_members else n_want
        excl = [(s, e) for s, e in zip(*G.gaps.get(c, ([], [])))]
        excl += [(max(0, s - a.margin), min(L, e + a.margin)) for s, e in zip(*G.events.get(c, ([], [])))]
        free, cur = [], 0
        for s, e in merge(excl):
            if s > cur:
                free.append((cur, s))
            cur = max(cur, e)
        if cur < L:
            free.append((cur, L))
        if a.ratio == 1:
            pick = rng.permutation(len(g))                       # every group used once
        else:
            pick = rng.choice(len(g), size=n_want, replace=True)
        templates = [g.iloc[i] for i in pick]
        uses, unplaced = {}, []
        placed = 0
        for r in sorted(templates, key=lambda x: -x["size"]):   # large groups first
            if keep_members:
                sz = int(r["size"])
            else:
                nm, o0, o1 = r["members"][int(rng.integers(0, len(r["members"])))]
                sz = int(o1 - o0)
            cand = [(s, e) for s, e in free if e - s >= sz]
            if not cand:
                continue
            lens = np.array([e - s - sz + 1 for s, e in cand], dtype=float)
            for _ in range(max_tries):
                s0, _e0 = cand[int(rng.choice(len(cand), p=lens / lens.sum()))]
                s = s0 + int(rng.integers(0, int(_e0 - s0 - sz) + 1))
                if occ.try_add(c, s, s + sz):
                    uses[r["gname"]] = uses.get(r["gname"], 0) + 1
                    suf = f"_{uses[r['gname']]}" if a.ratio != 1 else ""
                    if keep_members:
                        rows += [(c, s + o0, s + o1, f"ctrl_{label}_{nm}{suf}", label, f"margin={a.margin}")
                                 for nm, o0, o1 in r["members"]]
                    else:
                        rows.append((c, s, s + sz, f"ctrl_{label}_{r['gname']}{suf}", label, f"margin={a.margin}"))
                    placed += 1
                    break
            else:
                unplaced.append((r, sz))
        # event-free sequence ran out: the remaining groups may overlap other controls (still event-free)
        if a.reuse and keep_members:
            for r, sz in unplaced:
                cand = [(s, e) for s, e in free if e - s >= sz]
                if not cand:
                    continue
                lens = np.array([e - s - sz + 1 for s, e in cand], dtype=float)
                s0, _e0 = cand[int(rng.choice(len(cand), p=lens / lens.sum()))]
                s = s0 + int(rng.integers(0, int(_e0 - s0 - sz) + 1))
                uses[r["gname"]] = uses.get(r["gname"], 0) + 1
                suf = f"_{uses[r['gname']]}" if a.ratio != 1 else ""
                rows += [(c, s + o0, s + o1, f"ctrl_{label}_{nm}{suf}", label, f"margin={a.margin};reused_space")
                         for nm, o0, o1 in r["members"]]
                placed += 1
        if placed < n_want:
            deficit.append(f"{c}:{placed}/{n_want}")
    out = pd.DataFrame(rows, columns=OUT_COLS[:6])
    unit = "regions" if keep_members else "regions (one per overlap group)"
    print(f"[{label}] {len(out)} of {want_tot} {unit} (margin {a.margin} bp, spacing {a.min_spacing} bp)"
          + (f"; short on {len(deficit)} chromosomes" if deficit else ""))
    return out


# ----------------------------------------------------------------------------- reporting
def balance_row(label, df):
    n = len(df)
    r = {"set": label, "n": n, "n_chromosomes": df["chrom"].nunique() if n else 0,
         "median_size_bp": int(np.median(df["end"] - df["start"])) if n else 0}
    for s in STATE_ORDER:
        r[f"pct_{s}"] = round(100 * (df["seg_state"] == s).mean(), 1) if n else np.nan
    r["pct_state_NA"] = round(100 * (df["seg_state"] == "NA").mean(), 1) if n else np.nan
    r["pct_within_10kb_of_gap"] = round(100 * (df["d_gap"] < 10_000).mean(), 1) if n else np.nan
    r["pct_low_mappability"] = round(100 * df["lowmap"].mean(), 1) if n and "lowmap" in df else np.nan
    r["median_dist_to_gap_kb"] = round(np.nanmedian(df["d_gap"].replace(np.inf, np.nan)) / 1e3, 1) if n else np.nan
    r["median_local_CN_dispersion"] = round(np.nanmedian(df["local_disp"]), 3) if n and df["local_disp"].notna().any() else np.nan
    r["pct_overlapping_real_events"] = round(100 * df["overlaps_event"].mean(), 1) if n else np.nan
    r["n_overlap_groups"] = int(overlap_groups(df).max() + 1) if n else 0
    r["pct_in_multi_molecule_groups"] = round(100 * n_overlapping(df) / n, 1) if n else np.nan
    r["n_independent_blocks"] = df["block"].nunique() if n else 0
    return r


def write_set(G, df, path, a, tag):
    ann = G.context_of(df)
    ann["block"] = assign_blocks(ann, a.flank, tag) if len(ann) else []
    if len(ann):
        ann.loc[ann["overlaps_event"], "match_or_flag"] += ";overlaps_real_event"
    if a.drop_overlapping and tag.startswith("shift"):
        before = len(ann)
        ann = ann[~ann["overlaps_event"]]
        print(f"[{tag}] --drop-overlapping removed {before - len(ann)} regions on real events")
    ann.sort_values(["chrom", "start"])[OUT_COLS].to_csv(path, sep="\t", header=False, index=False)
    return ann


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["matched", "shift", "desert", "random", "coverage", "all"])
    ap.add_argument("-t", "--target", required=True)
    ap.add_argument("-e", "--events", nargs="+", required=True)
    ap.add_argument("-g", "--gc", required=True)
    ap.add_argument("-w", "--windows", default=None)
    ap.add_argument("-o", "--out-prefix", required=True)
    ap.add_argument("--margin", type=int, default=None,
                    help="Minimum edge-to-edge distance between a control and any event (bp). "
                         "Defaults: matched 1000, desert 15000, random 200")
    ap.add_argument("--ratio", type=float, default=1, help="Controls per target (matched/desert/random)")
    ap.add_argument("--min-spacing", type=int, default=None,
                    help="Minimum gap between controls (bp). Defaults: desert 20000, others 0")
    ap.add_argument("--collapse", choices=["auto", "yes", "no"], default="no",
                    help="Merge overlapping targets (per-molecule calls) into loci first: yes, no (default), "
                         "auto = only when targets overlap")
    ap.add_argument("--depth-col", default="depth",
                    help="coverage: window_cnv_calls.tsv column used as the sampling weight (default depth)")
    ap.add_argument("--depth-bed", default=None,
                    help="coverage: mosdepth regions file of the same windows as the weight instead, e.g. "
                         "coverage_mapq20.regions.bed.gz (closer to the reads DNAscent keeps)")
    ap.add_argument("--collapse-distance", type=int, default=1000,
                    help="Merge distance for --collapse, as bedtools merge -d (default 1000, the paper's mergers)")
    ap.add_argument("--standardize-bp", type=int, default=2000,
                    help="Size of each collapsed locus, centred on the merged-span midpoint (default 2000, "
                         "the paper's +/- 1 kb); 0 keeps the merged span")
    ap.add_argument("--no-reuse", dest="reuse", action="store_false",
                    help="matched/random: no second pass. By default, targets left without a control because "
                         "event-free sequence ran out may use a position already taken by another control")
    ap.add_argument("--no-fallback", dest="fallback", action="store_false",
                    help="matched: accept partners from the exact stratum only")
    ap.add_argument("--relax-gap", action="store_true",
                    help="matched: last fallback may use the adjacent gap-distance class (same chromosome)")
    ap.add_argument("--disp-support", type=float, default=0.95,
                    help="matched: exclude candidate windows whose local CN dispersion exceeds this quantile "
                         "of the targets' (default 0.95; 0 disables)")
    ap.add_argument("--no-match-mappability", dest="match_mappability", action="store_false",
                    help="matched: do not stratify by the low_mappability flag")
    ap.add_argument("--exclude-low-mappability", action="store_true",
                    help="matched: skip windows flagged as low mappability by analyze_cnv.py --compare")
    ap.add_argument("--k", type=int, default=5, help="shift: number of sets")
    ap.add_argument("--min-shift", type=int, default=25000, help="shift: minimum offset (gap-free bp)")
    ap.add_argument("--drop-overlapping", action="store_true", help="shift: drop regions lying on real events")
    ap.add_argument("--event-pad", type=int, default=200, help="Margin used to flag 'overlaps_real_event'")
    ap.add_argument("--flank", type=int, default=10000, help="Audit flank size (defines the blocks)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)) or ".", exist_ok=True)

    raw = read_bed(a.target)
    n_ov = n_overlapping(raw)
    print(f"[targets] {len(raw)} regions; {n_ov} ({100 * n_ov / max(len(raw), 1):.1f}%) overlap another target")
    collapsed = a.collapse == "yes" or (a.collapse == "auto" and n_ov > 0)
    if collapsed:
        target = collapse_loci(raw, a.collapse_distance, a.standardize_bp)
        nm = target["n_molecules"]
        print(f"[targets] collapsed into {len(target)} loci (merge distance {a.collapse_distance} bp, as bedtools "
              f"merge -d {a.collapse_distance}); regions per locus: median {int(nm.median())}, max {int(nm.max())}, "
              f"90th percentile {np.quantile(nm, 0.9):g}; loci with >= 10: {int((nm >= 10).sum())}")
        print("          " + (f"loci standardised to {a.standardize_bp} bp around the merged-span midpoint. "
                              if a.standardize_bp > 0 else "loci keep the merged span. ")
              + "Controls are built for the LOCI: audit the same loci (target_loci.bed).")
    else:
        target = raw
    target["size"] = (target["end"] - target["start"]).clip(lower=1)
    events = [read_bed(p) for p in a.events] + [raw[["chrom", "start", "end", "orig_name"]],
                                                 target[["chrom", "start", "end", "orig_name"]]]
    G = Genome(a.gc, a.windows, events, a.event_pad, a.match_mappability)
    missing = sorted(set(target["chrom"]) - set(G.sizes))
    if missing:
        print(f"[WARN] {len(missing)} target chromosome(s) absent from windows_gc.tsv: {missing[:5]}")
    target["name"] = [f"T{i:05d}" for i in range(len(target))]
    if collapsed:
        loci_path = f"{a.out_prefix}.target_loci.bed"
        with open(loci_path, "w") as fh:
            fh.write("#chrom\tstart\tend\tname\tn_molecules\tspan_start\tspan_end\n")
            target[["chrom", "start", "end", "name", "n_molecules", "span_start", "span_end"]].to_csv(
                fh, sep="\t", header=False, index=False)

    tctx = G.context_of(target)
    tctx["block"] = assign_blocks(tctx, a.flank, "target")
    tout = tctx.assign(mode="target", match_or_flag=tctx["orig_name"])
    tout.sort_values(["chrom", "start"])[OUT_COLS].to_csv(f"{a.out_prefix}.target_annotated.bed",
                                                          sep="\t", header=False, index=False)
    tctx = tctx.set_index("name", drop=False)
    tctx["group"] = overlap_groups(tctx)
    grp = G.context_of(group_table(tctx))
    if len(grp) < len(tctx):
        nm = grp["n_members"]
        print(f"[targets] {len(grp)} overlap groups (molecules per group: median {int(nm.median())}, max {int(nm.max())}; "
              f"span: median {int(grp['size'].median())} bp, max {int(grp['size'].max())} bp). Each group is moved "
              "as a unit: controls keep one region per molecule, with the same sizes and offsets")

    defaults = {"matched": (1000, 0), "desert": (15000, 20000), "random": (200, 0), "shift": (0, 0),
                "coverage": (0, 0)}
    modes = ["matched", "shift", "desert", "random"] if a.mode == "all" else [a.mode]
    user_margin, user_spacing = a.margin, a.min_spacing
    balance = [balance_row("target", tctx)]
    for m in modes:
        if m == "matched" and G.win is None:
            print("[matched] skipped: needs -w window_cnv_calls.tsv")
            continue
        single = a.mode != "all"
        a.margin = user_margin if (single and user_margin is not None) else defaults[m][0]
        a.min_spacing = user_spacing if (single and user_spacing is not None) else defaults[m][1]
        if m == "matched":
            sets = mode_matched(G, tctx, grp, a, rng)
        elif m == "shift":
            sets = mode_shift(G, tctx, grp, a, rng)
        elif m == "coverage":
            sets = {"coverage": mode_coverage(G, tctx, a, rng)}
        else:
            sets = {m: sample_free_space(G, grp, a, m, rng, keep_members=(m == "random"))}
        if m == "matched" and len(sets["matched"]):
            paired = sets["matched"]["name"].str.replace("ctrl_matched_", "", regex=False).str.replace(r"_\d+$", "", regex=True)
            sub = tctx[tctx["name"].isin(set(paired))]
            if len(sub) < len(tctx):
                balance.append(balance_row("target_with_partner", sub))
        shifted = []
        for label, df in sets.items():
            path = f"{a.out_prefix}.{label}.bed"
            ann = write_set(G, df, path, a, label)
            balance.append(balance_row(label, ann))
            if label.startswith("shift"):
                shifted.append(ann)
            print(f"        -> {path}")
        if len(shifted) > 1:
            pooled = balance_row("shift_all_sets", pd.concat(shifted, ignore_index=True))
            pooled["n_overlap_groups"] = pooled["pct_in_multi_molecule_groups"] = np.nan   # sets overlap each other
            balance.append(pooled)

    bal = pd.DataFrame(balance)
    bal["n_overlap_groups"] = bal["n_overlap_groups"].astype("Int64")
    bal.to_csv(f"{a.out_prefix}.balance.tsv", sep="\t", index=False)
    show = ["set", "n", "n_chromosomes", "median_size_bp", "pct_disomic", "pct_within_10kb_of_gap",
            "median_dist_to_gap_kb", "pct_low_mappability", "median_local_CN_dispersion",
            "pct_overlapping_real_events", "n_overlap_groups", "pct_in_multi_molecule_groups",
            "n_independent_blocks"]
    print("\nContext balance (target vs controls):")
    print(bal[[c for c in show if c in bal.columns]].to_string(index=False))
    print(f"\n[OK] {a.out_prefix}.balance.tsv\n[OK] {a.out_prefix}.target_annotated.bed"
          + (f"\n[OK] {a.out_prefix}.target_loci.bed  <- audit this file with verifyRegions.py" if collapsed else ""))


if __name__ == "__main__":
    main()
