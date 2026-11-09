#!/usr/bin/env bash
################################################################################
# compare_supplementary.sh — controlled test: does excluding supplementary alignments, and
# nothing else, change the depth distribution?
#
# Same BAM, same windows, same MAPQ, same analysis parameters; only the mosdepth flag changes:
#   with_supp     -F 1796  (unmapped, secondary, QC-fail, duplicates removed; supplementary kept)
#   without_supp  -F 3844  (as above + supplementary removed)
# The low-mappability pass is deliberately left out so that only one setting differs.
#
# Usage: compare_supplementary.sh -b GENOME.bam -g 2_CNV/data/windows_gc.tsv -o SUPP_TEST [-t 16] [-w 1000] [-q 0]
#   -g must come from a run with the same window size (-w) on the same reference.
################################################################################
set -euo pipefail

THREADS=16; WINDOW=1000; MAPQ=0; OUT="SUPP_TEST"
while getopts "b:g:o:t:w:q:h" opt; do
    case "$opt" in
        b) BAM="$OPTARG" ;; g) GC="$OPTARG" ;; o) OUT="$OPTARG" ;;
        t) THREADS="$OPTARG" ;; w) WINDOW="$OPTARG" ;; q) MAPQ="$OPTARG" ;;
        *) sed -n '2,13p' "$0"; exit 1 ;;
    esac
done
[[ -n "${BAM:-}" && -n "${GC:-}" ]] || { sed -n '2,13p' "$0"; exit 1; }
HERE="$(dirname "$(realpath "$0")")"
for f in analyze_cnv.py compare_depth_runs.py; do
    [[ -f "${HERE}/$f" ]] || { echo "[ERROR] ${HERE}/$f not found" >&2; exit 1; }
done

mkdir -p "${OUT}"
{
    echo "BAM: $(realpath "${BAM}")"
    echo "windows_gc: $(realpath "${GC}") | window ${WINDOW} bp | MAPQ >= ${MAPQ}"
    echo "--- aligned bases (samtools stats, 'bases mapped (cigar)') ---"
    echo -n "primary only            : "; samtools stats -@ "${THREADS}" -F 0x904 "${BAM}" | awk -F'\t' '$2 ~ /bases mapped \(cigar\)/ {print $3}'
    echo -n "primary + supplementary : "; samtools stats -@ "${THREADS}" -F 0x104 "${BAM}" | awk -F'\t' '$2 ~ /bases mapped \(cigar\)/ {print $3}'
    echo -n "supplementary records   : "; samtools view -@ "${THREADS}" -c -f 0x800 -F 0x104 "${BAM}"
    echo -n "primary records         : "; samtools view -@ "${THREADS}" -c -F 0x904 "${BAM}"
} | tee "${OUT}/supplementary_stats.txt"

for run in with_supp:1796 without_supp:3844; do
    name="${run%%:*}"; flag="${run#*:}"
    mkdir -p "${OUT}/${name}/data" "${OUT}/${name}/plots"
    echo "[INFO] mosdepth ${name} (-F ${flag})"
    mosdepth -t "${THREADS}" -x -Q "${MAPQ}" -F "${flag}" --by "${WINDOW}" "${OUT}/${name}/data/coverage" "${BAM}"
    cp "${GC}" "${OUT}/${name}/data/windows_gc.tsv"
    python3 "${HERE}/analyze_cnv.py" "${OUT}/${name}/data" "${OUT}/${name}/plots" | tee "${OUT}/${name}/analyze.log"
done

python3 "${HERE}/compare_depth_runs.py" "${OUT}/with_supp/data" "${OUT}/without_supp/data" \
    -o "${OUT}/comparison" --label-a with_supp --label-b without_supp
echo "[INFO] Read ${OUT}/supplementary_stats.txt and ${OUT}/comparison.summary.txt; plots in ${OUT}/*/plots and ${OUT}/comparison.pdf"
