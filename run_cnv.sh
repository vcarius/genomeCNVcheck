#!/usr/bin/env bash
################################################################################
# run_cnv.sh — read-depth copy-number (CN) pipeline for long-read data
#
#   1. Uses a pre-aligned BAM (-b, recommended: the Winnowmap BAM) or aligns reads (-r, minimap2)
#   2. Window depth with mosdepth (primary + supplementary alignments, MAPQ >= -q)
#   3. Optional second depth pass at a stricter MAPQ (-M) to flag low-mappability windows
#   4. Per-window GC (ACGT only) and N fraction with bedtools nuc
#   5. analyze_cnv.py: GC correction, disomic baseline, HMM segmentation, summaries, plots
#
# Outputs (in <outdir>/data and <outdir>/plots): see analyze_cnv.py
################################################################################
set -euo pipefail

THREADS=32
WINDOW_SIZE=1000
MIN_MAPQ=0           # keep multi-mapping reads (MAPQ 0): needed for repeat-rich T. cruzi regions
MASK_MAPQ=20         # second pass to flag low-mappability windows; 0 disables it
OUT_DIR="OUTPUT_CNV"
ANALYZE_ARGS=""
# mosdepth exclusion flag. 1796 = unmapped + secondary + QC-fail + duplicate (mosdepth default):
# SUPPLEMENTARY alignments are KEPT. Each supplementary piece is a real molecule at that position;
# with ultra-long reads on a gapped haploid reference of a hybrid genome, excluding them (3844)
# removes depth unevenly and creates a spurious low-depth mode. Use -S only for sensitivity tests.
FLAG=1796

usage() {
    cat <<EOF
Usage: $0 -f <reference.fasta> (-b <aligned.bam> | -r <reads.fastq>) [options]

Required:
  -f   Reference FASTA. Must be the SAME file used to build the BAM
       (e.g. 1_MAPPING/00_input/genome.fasta, which includes decoys if any were used).
  -b   Pre-aligned, coordinate-sorted BAM (recommended)
  -r   Reads (FASTQ/FASTA) to align with minimap2 -x map-ont (alternative to -b)

Options:
  -o   Output directory                                   [${OUT_DIR}]
  -t   Threads                                            [${THREADS}]
  -w   Window size (bp)                                   [${WINDOW_SIZE}]
  -q   Minimum MAPQ for the main depth pass               [${MIN_MAPQ}]
  -M   MAPQ for the low-mappability pass (0 = disabled)   [${MASK_MAPQ}]
  -a   Extra arguments passed to analyze_cnv.py, quoted
       (e.g. -a "--min-seg-bp 5000 --suffix _focal")
  -S   Exclude supplementary alignments (mosdepth -F 3844); sensitivity tests only
  -h   Show this help
EOF
    exit "${1:-1}"
}

while getopts "f:r:b:o:t:w:q:M:a:Sh" opt; do
    case "$opt" in
        f) FASTA_IN="$OPTARG" ;;
        r) READS_IN="$OPTARG" ;;
        b) BAM_IN="$OPTARG" ;;
        o) OUT_DIR="$OPTARG" ;;
        t) THREADS="$OPTARG" ;;
        w) WINDOW_SIZE="$OPTARG" ;;
        q) MIN_MAPQ="$OPTARG" ;;
        M) MASK_MAPQ="$OPTARG" ;;
        a) ANALYZE_ARGS="$OPTARG" ;;
        S) FLAG=3844 ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done

if [[ -z "${FASTA_IN:-}" || ( -z "${READS_IN:-}" && -z "${BAM_IN:-}" ) ]]; then
    echo "[ERROR] Provide -f and either -b or -r." >&2
    usage 1
fi
if [[ -n "${READS_IN:-}" && -n "${BAM_IN:-}" ]]; then
    echo "[ERROR] Use either -b or -r, not both." >&2
    exit 1
fi

# --- dependencies -------------------------------------------------------------
DEPS=(samtools mosdepth bedtools python3)
[[ -n "${READS_IN:-}" ]] && DEPS+=(minimap2)
for cmd in "${DEPS[@]}"; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "[ERROR] Missing dependency: $cmd" >&2; exit 1; }
done

ANALYZE_SCRIPT="$(dirname "$(realpath "$0")")/analyze_cnv.py"
[[ -f "$ANALYZE_SCRIPT" ]] || { echo "[ERROR] $ANALYZE_SCRIPT not found (keep it next to run_cnv.sh)." >&2; exit 1; }

mkdir -p "${OUT_DIR}/data" "${OUT_DIR}/plots"
DATA="${OUT_DIR}/data"
LOG="${OUT_DIR}/run_cnv.log"
exec > >(tee -a "${LOG}") 2>&1

echo "[INFO] $(date '+%F %T') run_cnv.sh"
echo "[INFO] FASTA=${FASTA_IN} BAM=${BAM_IN:-<align ${READS_IN:-}>} window=${WINDOW_SIZE} MAPQ=${MIN_MAPQ} maskMAPQ=${MASK_MAPQ} mosdepthFlag=${FLAG} threads=${THREADS}"
echo "[INFO] samtools: $(samtools --version | head -1) | mosdepth: $(mosdepth --version 2>&1 | head -1) | bedtools: $(bedtools --version)"

[[ -f "${FASTA_IN}.fai" ]] || samtools faidx "${FASTA_IN}"

# --- 1/5 BAM ------------------------------------------------------------------
if [[ -n "${BAM_IN:-}" ]]; then
    echo "[INFO] 1/5 Using pre-aligned BAM: ${BAM_IN}"
    BAM_FILE="${BAM_IN}"
    if [[ ! -f "${BAM_FILE}.bai" && ! -f "${BAM_FILE%.bam}.bai" && ! -f "${BAM_FILE}.csi" ]]; then
        samtools index -@ "${THREADS}" "${BAM_FILE}"
    fi
else
    echo "[INFO] 1/5 Aligning with minimap2 (note: the main pipeline uses Winnowmap; prefer -b)"
    BAM_FILE="${DATA}/aligned.bam"
    minimap2 -ax map-ont -t "${THREADS}" "${FASTA_IN}" "${READS_IN}" \
        | samtools sort -@ "${THREADS}" -m 2G -o "${BAM_FILE}"
    samtools index -@ "${THREADS}" "${BAM_FILE}"
fi

# Fail fast if the FASTA is not the one used for the BAM (names AND lengths must match)
MISMATCH=$(LC_ALL=C join -t $'\t' -a1 \
    <(samtools view -H "${BAM_FILE}" | awk -v OFS='\t' '$1=="@SQ"{n="";l="";for(i=2;i<=NF;i++){if($i~/^SN:/)n=substr($i,4);if($i~/^LN:/)l=substr($i,4)};print n,l}' | LC_ALL=C sort -k1,1) \
    <(cut -f1,2 "${FASTA_IN}.fai" | LC_ALL=C sort -k1,1) \
    | awk -F'\t' 'NF<3 || $2!=$3' | wc -l)
if [[ "${MISMATCH}" -gt 0 ]]; then
    echo "[ERROR] ${MISMATCH} BAM contig(s) missing from the FASTA or with a different length." >&2
    echo "        Use the FASTA the BAM was built with (e.g. 1_MAPPING/00_input/genome.fasta)." >&2
    exit 1
fi

# --- 2/5 depth ----------------------------------------------------------------
# -F ${FLAG} (see top of script); -x = fast mode (read span)
echo "[INFO] 2/5 mosdepth (MAPQ >= ${MIN_MAPQ}, -F ${FLAG}, ${WINDOW_SIZE} bp windows)"
mosdepth -t "${THREADS}" -x -Q "${MIN_MAPQ}" -F "${FLAG}" --by "${WINDOW_SIZE}" "${DATA}/coverage" "${BAM_FILE}"

COMPARE_ARG=""
if [[ "${MASK_MAPQ}" -gt 0 && "${MASK_MAPQ}" -ne "${MIN_MAPQ}" ]]; then
    echo "[INFO] 3/5 mosdepth (MAPQ >= ${MASK_MAPQ}) for the low-mappability mask"
    mosdepth -t "${THREADS}" -x -Q "${MASK_MAPQ}" -F "${FLAG}" --by "${WINDOW_SIZE}" \
        "${DATA}/coverage_mapq${MASK_MAPQ}" "${BAM_FILE}"
    COMPARE_ARG="--compare ${DATA}/coverage_mapq${MASK_MAPQ}.regions.bed.gz"
else
    echo "[INFO] 3/5 low-mappability pass disabled"
fi

# --- 4/5 GC and N per window --------------------------------------------------
# bedtools nuc columns with a 3-column BED: 6-9 = A,C,G,T counts; 10 = N; 12 = sequence length
echo "[INFO] 4/5 GC (ACGT only) and N fraction per window"
zcat "${DATA}/coverage.regions.bed.gz" | cut -f1-3 > "${DATA}/windows.bed"
bedtools nuc -fi "${FASTA_IN}" -bed "${DATA}/windows.bed" \
  | awk -v OFS='\t' 'NR==1{print "chrom","start","end","gc_acgt","frac_N"; next}
        {acgt=$6+$7+$8+$9; gc=(acgt>0)?($7+$8)/acgt:"NA"; print $1,$2,$3,gc,$10/$12}' \
  > "${DATA}/windows_gc.tsv"
rm -f "${DATA}/windows.bed"

# --- 5/5 analysis -------------------------------------------------------------
echo "[INFO] 5/5 analyze_cnv.py ${COMPARE_ARG} ${ANALYZE_ARGS}"
# shellcheck disable=SC2086  # intentional word splitting of the optional arguments
python3 "${ANALYZE_SCRIPT}" "${DATA}" "${OUT_DIR}/plots" ${COMPARE_ARG} ${ANALYZE_ARGS}

echo "[INFO] Done: ${DATA}/{window_cnv_calls.tsv,cnv_segments.bed,chromosome_ploidy_summary.tsv}"
