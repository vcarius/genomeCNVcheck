#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verifyRegions.py — structural audit of replication origins/terminations (or any regions)
with long reads (ONT/HiFi).

For each region of interest (ROI):
  core  = ROI +/- --core-buffer-bp      (breakpoints here count as core breakpoints)
  flank = ROI +/- --max-distance-bp     (context: flank depth, flank breakpoints, split reads)

Per-ROI metrics (primary alignments with MAPQ >= -q unless stated otherwise)
  Reads_Touching_Core, Clean_Spanning_Core (reads crossing the whole ROI with no core breakpoint and
  no large indel), Core_Breakpoints (significant soft/hard clips inside the core), Core_Large_Indels,
  Core_Mean_Depth and Flank_Mean_Depth (aligned bases / length), Flank_Breakpoints, Flank_Split_Reads.

Depth ratio used for classification
  default         core depth / flank depth (legacy behaviour). The flank can contain neighbouring
                  events and N gaps (zero depth), which inflates the ratio next to gaps.
  --cnv-windows   core depth / expected depth of the copy-number segment containing the ROI
                  (median depth of that segment in window_cnv_calls.tsv from analyze_cnv.py).
                  Independent of the flank. Core depth for this ratio is computed exactly like
                  run_cnv.sh computes window depth (mosdepth -x -F 1796: read span, all MAPQ,
                  primary + supplementary). If run_cnv.sh was run with -S (supplementary excluded),
                  pass --span-exclude-supplementary here as well.

Status codes (in order of evaluation)
  NO_CORE_COVERAGE, CRITICAL_ERROR_CHIMERA_OR_BREAK_IN_ROI, WARNING_ZERO_SPANNING_READS,
  REPEAT_COLLAPSE_IN_ROI, WARNING_LOCAL_HIGH_COVERAGE_CNV, HOMOZYGOUS_LOCAL_INDEL_SPANNING_ROI,
  HETEROZYGOUS_SV_OR_HAPLOTYPE_DIVERGENCE, INTACT_CORE_FLANK_CLIPPING (formerly
  INTACT_ROI_UNSTABLE_10KB_FLANK), VALIDATED_STRUCTURE, MODERATE_INSTABILITY_REVIEW.
"""
import argparse
import os
import statistics
import sys

import numpy as np
import pandas as pd
import pysam

QUERY_CONSUMING_OPS = {0, 1, 4, 5, 7, 8}   # M, I, S, H, =, X (H included: full original read length)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-b", "--bed", required=True, help="BED with ROIs (first 3 columns; 4th = ROI name)")
    p.add_argument("-m", "--bam", nargs="+", required=True, help="Indexed long-read BAM file(s)")
    p.add_argument("-q", "--min-mapq", type=int, default=10, help="Minimum MAPQ (default 10)")
    p.add_argument("--core-buffer-bp", type=int, default=200,
                   help="Buffer around the ROI that defines the core (default 200)")
    p.add_argument("-d", "--max-distance-bp", type=int, default=10000,
                   help="Flank size on each side of the ROI (default 10000)")
    p.add_argument("-p", "--min-clip-perc", type=float, default=15.0,
                   help="Minimum clipped fraction of the read (%%) for a breakpoint (default 15)")
    p.add_argument("-c", "--min-clip-bp", type=int, default=500,
                   help="Clip length (bp) that counts as a breakpoint regardless of %% (default 500)")
    p.add_argument("--abs-floor-clip-bp", type=int, default=100,
                   help="Clips shorter than this are always ignored (default 100)")
    p.add_argument("-i", "--min-indel-bp", type=int, default=50,
                   help="Minimum indel size (bp) inside the core (default 50)")
    p.add_argument("--min-bp-perc-critical", type=float, default=30.0,
                   help="Core-breakpoint %% for the critical status (default 30)")
    p.add_argument("--min-bp-perc-het", type=float, default=20.0,
                   help="Core-breakpoint %% for haplotype divergence / heterozygous SV (default 20)")
    p.add_argument("--max-bp-perc-validated", type=float, default=15.0,
                   help="Maximum core-breakpoint %% for a validated structure (default 15)")
    p.add_argument("--high-depth-ratio", type=float, default=1.6,
                   help="Depth ratio for repeat collapse (default 1.6)")
    p.add_argument("--extreme-depth-ratio", type=float, default=1.8,
                   help="Depth ratio for local hyper-coverage (default 1.8)")
    p.add_argument("--het-indel-ratio", type=float, default=0.25,
                   help="Indel-read fraction for haplotype divergence / heterozygous SV (default 0.25)")
    p.add_argument("--cnv-windows", default=None,
                   help="window_cnv_calls.tsv (analyze_cnv.py): classify with core / segment depth")
    p.add_argument("--span-exclude-supplementary", action="store_true",
                   help="Exclude supplementary alignments from the core depth used against the CN segment "
                        "(only if run_cnv.sh was run with -S)")
    p.add_argument("-o", "--output", default="structural_audit.tsv", help="Output TSV")
    return p.parse_args()


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def has_internal_large_indel(read, start_pos, end_pos, min_indel_bp):
    """True if the read has an insertion or deletion >= min_indel_bp inside [start_pos, end_pos]."""
    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        if op in (0, 7, 8, 3):                       # M, =, X, N consume the reference
            ref_pos += length
        elif op == 2:                                # D
            if length >= min_indel_bp and max(start_pos, ref_pos) < min(end_pos, ref_pos + length):
                return True
            ref_pos += length
        elif op == 1:                                # I
            if length >= min_indel_bp and start_pos <= ref_pos <= end_pos:
                return True
    return False


def is_significant_clip(clip_len, read_len, a):
    if clip_len < a.abs_floor_clip_bp:
        return False
    return (100.0 * clip_len / read_len) >= a.min_clip_perc or clip_len >= a.min_clip_bp


def dist_to_roi(pos, s, e):
    return s - pos if pos < s else (pos - e if pos > e else 0)


class SegmentDepth:
    """Expected depth of the CN segment containing a position (median depth_eff of its windows)."""

    def __init__(self, path):
        w = pd.read_csv(path, sep="\t", dtype={"chrom": str}, na_values=["NA"])
        need = {"chrom", "start", "end", "depth_eff", "seg_id"}
        if not need.issubset(w.columns):
            sys.exit(f"[ERROR] {path} lacks {sorted(need - set(w.columns))}: use analyze_cnv.py output.")
        ok = np.isfinite(w["depth_eff"])
        if "low_mappability" in w.columns:
            ok &= ~w["low_mappability"].astype(str).str.lower().eq("true")
        self.seg_depth = w[ok & (w["seg_id"] >= 0)].groupby("seg_id")["depth_eff"].median().to_dict()
        self.by_chr = {c: (g["start"].to_numpy(), g["end"].to_numpy(), g["seg_id"].to_numpy())
                       for c, g in w.sort_values(["chrom", "start"]).groupby("chrom")}

    def expected(self, chrom, pos):
        t = self.by_chr.get(chrom)
        if t is None:
            return np.nan
        st, en, sid = t
        i = np.searchsorted(st, pos, side="right") - 1
        if i < 0 or pos >= en[i] or sid[i] < 0:
            return np.nan                         # position in a gap / excluded window
        return float(self.seg_depth.get(int(sid[i]), np.nan))


def classify(clean_spanning, indel_spanning, touching, perc_core_bp, ratio, core_indels, flank_bp, a):
    if touching == 0:
        return "NO_CORE_COVERAGE"
    spanning_ratio = clean_spanning / touching
    total_spanning = clean_spanning + indel_spanning
    indel_ratio = core_indels / touching
    ratio_ok = np.isfinite(ratio)

    # 1. continuity failure (true break / chimera / assembly error)
    if total_spanning < 3 and perc_core_bp >= a.min_bp_perc_critical:
        return "CRITICAL_ERROR_CHIMERA_OR_BREAK_IN_ROI"
    if total_spanning == 0:
        return "WARNING_ZERO_SPANNING_READS"
    # 2. collapsed repeat / hyper-coverage
    if ratio_ok and ratio >= a.high_depth_ratio and (perc_core_bp >= a.max_bp_perc_validated or indel_ratio >= 0.20):
        return "REPEAT_COLLAPSE_IN_ROI"
    if ratio_ok and ratio >= a.extreme_depth_ratio:
        return "WARNING_LOCAL_HIGH_COVERAGE_CNV"
    # 3. homozygous local indel / shifted repeat consensus
    if clean_spanning < 5 and total_spanning / touching >= 0.70 and indel_ratio >= 0.75 \
            and perc_core_bp < a.max_bp_perc_validated:
        return "HOMOZYGOUS_LOCAL_INDEL_SPANNING_ROI"
    # 4. haplotype divergence or heterozygous SV
    if 0.20 <= spanning_ratio <= 0.80 and (perc_core_bp >= a.min_bp_perc_het or indel_ratio >= a.het_indel_ratio):
        return "HETEROZYGOUS_SV_OR_HAPLOTYPE_DIVERGENCE"
    # 5. intact core, clipping concentrated in the flanks (descriptive; depends on flank size)
    if perc_core_bp < a.max_bp_perc_validated and spanning_ratio >= 0.50 and flank_bp >= 0.5 * touching:
        return "INTACT_CORE_FLANK_CLIPPING"
    # 6. validated
    if perc_core_bp < a.max_bp_perc_validated and ratio_ok and 0.6 <= ratio <= (a.high_depth_ratio - 0.1) \
            and clean_spanning >= 5:
        return "VALIDATED_STRUCTURE"
    return "MODERATE_INSTABILITY_REVIEW"


def read_rois(path):
    rois = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            rois.append((p[0], int(p[1]), int(p[2]), p[3] if len(p) > 3 else f"{p[0]}:{p[1]}-{p[2]}"))
    return rois


def audit_roi(sam, chrom, raw_start, raw_end, a, seg):
    chrom_len = sam.get_reference_length(chrom)
    start = max(0, min(raw_start, chrom_len))
    end = max(start, min(raw_end, chrom_len))
    roi_len = max(1, end - start)
    core_s, core_e = max(0, start - a.core_buffer_bp), min(chrom_len, end + a.core_buffer_bp)
    fl_s, fl_e = max(0, start - a.max_distance_bp), min(chrom_len, end + a.max_distance_bp)

    touching, core_bp, core_indel, clean_span, indel_span = set(), set(), set(), set(), set()
    flank_bp = flank_split = 0
    bp_dist = []
    core_bases = left_bases = right_bases = 0
    core_span_all = 0                                   # read-span bases, all MAPQ (run_cnv.sh/mosdepth definition)

    for read in sam.fetch(chrom, fl_s, fl_e):
        if read.is_unmapped or read.is_secondary or read.is_duplicate or read.is_qcfail or not read.cigartuples:
            continue
        if not (read.is_supplementary and a.span_exclude_supplementary):
            core_span_all += overlap(read.reference_start, read.reference_end, start, end)
        if read.mapping_quality < a.min_mapq:
            continue

        q = read.query_name
        r0, r1 = read.reference_start, read.reference_end
        touches_core = r0 <= core_e and r1 >= core_s
        left_in = fl_s <= r0 <= fl_e
        right_in = fl_s <= r1 <= fl_e
        first_op, first_len = read.cigartuples[0]
        last_op, last_len = read.cigartuples[-1]

        # split alignments (primary with SA tag, or supplementary) clipped inside the flank window
        if (read.is_supplementary or read.has_tag("SA")) and \
                ((left_in and first_op in (4, 5)) or (right_in and last_op in (4, 5))):
            flank_split += 1

        if read.is_supplementary:
            continue

        # aligned-base depth (M/=/X blocks), equivalent to pysam count_coverage on these reads
        for bs, be in read.get_blocks():
            core_bases += overlap(bs, be, start, end)
            left_bases += overlap(bs, be, fl_s, start)
            right_bases += overlap(bs, be, end, fl_e)

        has_indel = False
        if touches_core:
            touching.add(q)
            has_indel = has_internal_large_indel(read, core_s, core_e, a.min_indel_bp)
            if has_indel:
                core_indel.add(q)

        has_core_bp = has_flank_bp = False
        if left_in or right_in:
            read_len = sum(n for op, n in read.cigartuples if op in QUERY_CONSUMING_OPS)
            if read_len > 0:
                # 5' clip (ignored at the chromosome start)
                if left_in and first_op in (4, 5) and r0 > 0 and is_significant_clip(first_len, read_len, a):
                    bp_dist.append(dist_to_roi(r0, start, end))
                    if core_s <= r0 <= core_e:
                        has_core_bp = True
                    else:
                        has_flank_bp = True
                # 3' clip (ignored at the chromosome end)
                if right_in and last_op in (4, 5) and r1 < chrom_len and is_significant_clip(last_len, read_len, a):
                    bp_dist.append(dist_to_roi(r1, start, end))
                    if core_s <= r1 <= core_e:
                        has_core_bp = True
                    else:
                        has_flank_bp = True
                if has_core_bp and touches_core:
                    core_bp.add(q)
                elif has_flank_bp:
                    flank_bp += 1

        # reads crossing the whole ROI
        if touches_core and not has_core_bp and r0 <= start and r1 >= end:
            (indel_span if has_indel else clean_span).add(q)

    n_touch = len(touching)
    n_core_bp = len(core_bp)
    n_clean = len(clean_span - core_bp - core_indel)
    n_indel_span = len(indel_span - core_bp)
    core_depth = core_bases / (end - start) if end > start else 0.0
    fl_len = (start - fl_s) + (fl_e - end)
    flank_depth = (left_bases + right_bases) / fl_len if fl_len > 0 else 0.0
    ratio_flank = core_depth / flank_depth if flank_depth > 0 else 0.0
    perc_core_bp = 100.0 * n_core_bp / n_touch if n_touch else 0.0

    core_depth_all = core_span_all / (end - start) if end > start else 0.0
    exp_depth = seg.expected(chrom, (start + end) // 2) if seg else np.nan
    ratio_seg = core_depth_all / exp_depth if np.isfinite(exp_depth) and exp_depth > 0 else np.nan
    if seg and np.isfinite(ratio_seg):
        ratio_used, used = ratio_seg, "segment"
    else:
        ratio_used, used = ratio_flank, "flank"

    status = classify(n_clean, n_indel_span, n_touch, perc_core_bp, ratio_used, len(core_indel), flank_bp, a)
    return [chrom, start, end, roi_len, n_touch, n_clean, n_indel_span,
            f"{core_depth:.2f}", f"{flank_depth:.2f}", f"{ratio_flank:.2f}",
            f"{core_depth_all:.2f}", "NA" if not np.isfinite(exp_depth) else f"{exp_depth:.2f}",
            "NA" if not np.isfinite(ratio_seg) else f"{ratio_seg:.2f}", used,
            n_core_bp, f"{perc_core_bp:.2f}", len(core_indel), flank_bp, flank_split,
            f"{statistics.median(bp_dist):.0f}" if bp_dist else "NA", status]


HEADER = ["Sample", "Chrom", "Start", "End", "Region_ID", "ROI_Length_bp",
          "Reads_Touching_Core", "Clean_Spanning_Core", "Indel_Spanning_Core",
          "Core_Mean_Depth", "Flank_Mean_Depth", "Depth_Ratio_Core_vs_Flank",
          "Core_Span_Depth_AllMAPQ", "Segment_Expected_Depth", "Depth_Ratio_Core_vs_Segment", "Depth_Ratio_Used",
          "Core_Breakpoints", "Perc_Core_Reads_With_BP", "Core_Large_Indels",
          "Flank_Breakpoints", "Flank_Split_Reads", "Median_BP_Dist_To_ROI_bp", "Structural_Status"]


def main():
    a = parse_args()
    if not os.path.exists(a.bed):
        sys.exit(f"[ERROR] BED file '{a.bed}' not found.")
    rois = read_rois(a.bed)
    seg = SegmentDepth(a.cnv_windows) if a.cnv_windows else None
    print(f"[INFO] {len(rois)} ROIs | core ROI +/- {a.core_buffer_bp} bp | flanks +/- {a.max_distance_bp} bp | "
          f"MAPQ >= {a.min_mapq} | depth ratio for classification: {'CN segment' if seg else 'flanks'}")

    with open(a.output, "w") as out:
        out.write("\t".join(HEADER) + "\n")
        for bam in a.bam:
            sample = os.path.basename(bam)
            with pysam.AlignmentFile(bam, "rb") as sam:
                refs = set(sam.references)
                skipped = 0
                for k, (chrom, s, e, rid) in enumerate(rois, 1):
                    if chrom not in refs:
                        skipped += 1
                        continue
                    row = audit_roi(sam, chrom, s, e, a, seg)
                    out.write("\t".join(map(str, [sample] + row[:3] + [rid] + row[3:])) + "\n")
                    if k % 500 == 0:
                        print(f"[INFO] {sample}: {k}/{len(rois)} ROIs")
                if skipped:
                    print(f"[WARN] {sample}: {skipped} ROIs on chromosomes absent from the BAM were skipped.")
    print(f"[INFO] Audit completed: {a.output}")


if __name__ == "__main__":
    main()
