#!/usr/bin/env bash
################################################################################
# mapping_V2.sh — repeat-aware long-read mapping with Winnowmap2
#
# High-frequency k-mers (Meryl, k=15, above the 0.9998 'distinct' fraction) are passed to
# Winnowmap with -W. They are DOWN-WEIGHTED during minimizer selection; no sequence is masked
# or removed, so repetitive regions remain alignable. k must equal Winnowmap's minimizer k.
# These are the parameters recommended by the Winnowmap2 authors (Jain et al.) for ONT/PacBio reads.
#
# Optional competitive decoys (-d) are concatenated to the target FASTA before indexing, so
# reads from sequences absent from the target do not mis-map onto it.
################################################################################
set -euo pipefail

# ------------------------------------------------------------------------------ defaults
FA_TC=""
READS=""
DECOYS=""
OUTPUT="OUTPUT"
PREFIX="GENOME"
THREADS=32
SORT_THREADS=16
KMER=15                   # must match Winnowmap's minimizer k (15 for ONT/PacBio)
REP_FRAC=0.9998           # Meryl 'distinct' fraction recommended in the Winnowmap2 protocol
PRESET="map-ont"          # map-ont | map-pb | map-pb-clr | splice | splice:hq | asm5 | asm10 | asm20
CONDA_ENV="genome_asm"    # empty string = do not activate any Conda environment
INTERACTIVE_MENU=false

usage() {
    cat <<EOF
================================================================================
  REPEAT-AWARE LONG-READ MAPPING WITH WINNOWMAP2
================================================================================
Usage:
  $0 -f <reference.fasta> -r <reads.fastq.gz> [options]
  $0 --menu        (interactive configuration)

Required:
  -f, --fasta <file>        Target reference FASTA
  -r, --reads <file>        Long reads, FASTQ(.gz) (ONT or PacBio)

Mapping:
  -d, --decoys <f1,f2>      Comma-separated decoy FASTA(s) for competitive mapping
  -o, --output <dir>        Output directory                         [${OUTPUT}]
  -p, --prefix <str>        Prefix for index and BAM files           [${PREFIX}]
  -x, --preset <str>        Winnowmap preset                         [${PRESET}]
  -k, --kmer <int>          k for Meryl/Winnowmap                    [${KMER}]
  -m, --rep-frac <float>    Meryl 'distinct' fraction; k-mers above it are down-weighted
                            during minimizer selection (nothing is masked)  [${REP_FRAC}]

Resources:
  -t, --threads <int>       Winnowmap threads                        [${THREADS}]
  -s, --sort-threads <int>  samtools threads                         [${SORT_THREADS}]
  -e, --conda-env <str>     Conda environment ('' = none)            [${CONDA_ENV}]
  -M, --menu                Interactive menu
  -h, --help                Show this help

Examples:
  $0 -f Esmeraldo_v68.fasta -r reads.fastq.gz -t 64 -o 1_MAPPING
  $0 -f target.fasta -d decoy1.fasta,decoy2.fasta -r reads.fastq.gz -k 15 -m 0.9998 -o 1_MAPPING_decoys
================================================================================
EOF
}

run_interactive_menu() {
    while true; do
        clear
        cat <<EOF
================================================================================
       INTERACTIVE MENU — WINNOWMAP2 MAPPING
================================================================================
  [1]  Target FASTA (-f)       : ${FA_TC:-"<NOT SET>"}
  [2]  Long reads (-r)         : ${READS:-"<NOT SET>"}
  [3]  Decoy FASTA(s) (-d)     : ${DECOYS:-"<none>"}
  [4]  Output directory (-o)   : ${OUTPUT}
  [5]  File prefix (-p)        : ${PREFIX}
  [6]  Winnowmap preset (-x)   : ${PRESET}
  [7]  k-mer size (-k)         : ${KMER}
  [8]  Meryl fraction (-m)     : ${REP_FRAC}
  [9]  Winnowmap threads (-t)  : ${THREADS}
  [10] samtools threads (-s)   : ${SORT_THREADS}
  [11] Conda environment (-e)  : ${CONDA_ENV:-"<none>"}
--------------------------------------------------------------------------------
  [R] RUN    [Q] Quit
================================================================================
EOF
        read -rp "Choose an option to edit, or R to run: " opt
        case "${opt}" in
            1) read -e -rp "Target FASTA: " FA_TC ;;
            2) read -e -rp "Reads FASTQ(.gz): " READS ;;
            3) read -e -rp "Decoy FASTA(s), comma-separated (Enter to clear): " DECOYS ;;
            4) read -e -rp "Output directory [${OUTPUT}]: " val; OUTPUT="${val:-$OUTPUT}" ;;
            5) read -rp "Prefix [${PREFIX}]: " val; PREFIX="${val:-$PREFIX}" ;;
            6) read -rp "Preset [${PRESET}]: " val; PRESET="${val:-$PRESET}" ;;
            7) read -rp "k-mer size [${KMER}]: " val; KMER="${val:-$KMER}" ;;
            8) read -rp "Meryl fraction [${REP_FRAC}]: " val; REP_FRAC="${val:-$REP_FRAC}" ;;
            9) read -rp "Winnowmap threads [${THREADS}]: " val; THREADS="${val:-$THREADS}" ;;
            10) read -rp "samtools threads [${SORT_THREADS}]: " val; SORT_THREADS="${val:-$SORT_THREADS}" ;;
            11) read -rp "Conda environment ('' for none) [${CONDA_ENV}]: " val; CONDA_ENV="${val:-$CONDA_ENV}" ;;
            [Rr])
                if [[ -z "${FA_TC}" || -z "${READS}" ]]; then
                    echo -e "\n[ERROR] Items [1] and [2] are required."
                    read -rp "Press Enter to continue..."
                else
                    break
                fi ;;
            [Qq]) echo "[INFO] Cancelled."; exit 0 ;;
            *) echo "[WARN] Invalid option."; sleep 1 ;;
        esac
    done
}

# ------------------------------------------------------------------------------ CLI
[[ $# -eq 0 ]] && INTERACTIVE_MENU=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--fasta)        FA_TC="$2"; shift 2 ;;
        -r|--reads)        READS="$2"; shift 2 ;;
        -d|--decoys)       DECOYS="$2"; shift 2 ;;
        -o|--output)       OUTPUT="$2"; shift 2 ;;
        -p|--prefix)       PREFIX="$2"; shift 2 ;;
        -x|--preset)       PRESET="$2"; shift 2 ;;
        -k|--kmer)         KMER="$2"; shift 2 ;;
        -m|--rep-frac)     REP_FRAC="$2"; shift 2 ;;
        -t|--threads)      THREADS="$2"; shift 2 ;;
        -s|--sort-threads) SORT_THREADS="$2"; shift 2 ;;
        -e|--conda-env)    CONDA_ENV="$2"; shift 2 ;;
        -M|--menu)         INTERACTIVE_MENU=true; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done
[[ "${INTERACTIVE_MENU}" == true ]] && run_interactive_menu

# ------------------------------------------------------------------------------ validation
[[ -n "${FA_TC}" && -n "${READS}" ]] || { echo "[ERROR] -f and -r are required." >&2; usage; exit 1; }
[[ -f "${FA_TC}" ]] || { echo "[ERROR] FASTA not found: ${FA_TC}" >&2; exit 1; }
[[ -f "${READS}" ]] || { echo "[ERROR] Reads not found: ${READS}" >&2; exit 1; }
case "${PRESET}" in
    map-ont|map-pb|map-pb-clr|splice|splice:hq|asm5|asm10|asm20) ;;
    *) echo "[ERROR] Invalid preset '${PRESET}'." >&2; exit 1 ;;
esac

# Absolute paths before changing directory
ABS_FA_TC="$(realpath "${FA_TC}")"
ABS_READS="$(realpath "${READS}")"
ABS_OUTPUT="$(realpath -m "${OUTPUT}")"
declare -a ABS_DECOYS=()
if [[ -n "${DECOYS}" ]]; then
    IFS=',' read -ra DECOY_LIST <<< "${DECOYS}"
    for dec in "${DECOY_LIST[@]}"; do
        [[ -f "${dec}" ]] || { echo "[ERROR] Decoy not found: ${dec}" >&2; exit 1; }
        ABS_DECOYS+=("$(realpath "${dec}")")
    done
fi

# ------------------------------------------------------------------------------ environment
if [[ -n "${CONDA_ENV}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        echo "[INFO] Activating Conda environment: ${CONDA_ENV}"
        set +u
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
        set -u
    else
        echo "[WARN] conda not found; using tools from the current PATH."
    fi
fi
for tool in samtools meryl winnowmap; do
    command -v "${tool}" >/dev/null 2>&1 || { echo "[ERROR] '${tool}' not found in PATH." >&2; exit 1; }
done

# ------------------------------------------------------------------------------ setup and logging
mkdir -p "${ABS_OUTPUT}"/{00_input,01_index,02_mapping,logs}
cd "${ABS_OUTPUT}"
LOG_FILE="logs/${PREFIX}_mapping.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================================"
echo "[INFO] Start            : $(date '+%F %T')"
echo "[INFO] Target FASTA     : ${ABS_FA_TC}"
echo "[INFO] Decoy FASTA(s)   : ${ABS_DECOYS[*]:-none}"
echo "[INFO] Reads            : ${ABS_READS}"
echo "[INFO] Output           : ${ABS_OUTPUT}"
echo "[INFO] Preset / k-mer   : ${PRESET} / k=${KMER} (distinct=${REP_FRAC})"
echo "[INFO] Threads map/sort : ${THREADS} / ${SORT_THREADS}"
# Tool versions, for the Methods section
echo "[INFO] winnowmap        : $(winnowmap --version 2>&1 | head -1)"
echo "[INFO] meryl            : $(meryl --version 2>&1 | head -1)"
echo "[INFO] samtools         : $(samtools --version | head -1)"
echo "================================================================================"

ln -sf "${ABS_READS}" 00_input/reads.fastq.gz
if [[ ${#ABS_DECOYS[@]} -eq 0 ]]; then
    ln -sf "${ABS_FA_TC}" 00_input/genome.fasta
else
    echo "[INFO] Concatenating target + ${#ABS_DECOYS[@]} decoy file(s) into 00_input/genome.fasta"
    rm -f 00_input/genome.fasta
    cat "${ABS_FA_TC}" "${ABS_DECOYS[@]}" > 00_input/genome.fasta
fi

# ------------------------------------------------------------------------------ step 1: index
echo "[INFO] Step 1: samtools faidx"
samtools faidx 00_input/genome.fasta

# ------------------------------------------------------------------------------ step 2: k-mer list
build_kmer_list () {
    local ref_fa="$1" prefix="$2"
    local meryl_db="01_index/${prefix}_k${KMER}.meryl"
    local rep_txt="01_index/${prefix}.repetitive_k${KMER}.txt"
    rm -rf "${meryl_db}"                       # stale database from an interrupted run
    meryl count k="${KMER}" threads="${THREADS}" output "${meryl_db}" "${ref_fa}"
    meryl print greater-than "distinct=${REP_FRAC}" threads="${THREADS}" "${meryl_db}" > "${rep_txt}"
    echo "[INFO] High-frequency k-mers listed: $(wc -l < "${rep_txt}")"
}
echo "[INFO] Step 2: high-frequency k-mers (Meryl) for Winnowmap down-weighting"
build_kmer_list 00_input/genome.fasta "${PREFIX}"

# ------------------------------------------------------------------------------ step 3: mapping
echo "[INFO] Step 3: Winnowmap (${PRESET})"
winnowmap -t "${THREADS}" -k "${KMER}" -W "01_index/${PREFIX}.repetitive_k${KMER}.txt" -ax "${PRESET}" \
    00_input/genome.fasta 00_input/reads.fastq.gz \
    | samtools sort -@ "${SORT_THREADS}" -o "02_mapping/${PREFIX}.bam"
samtools index -@ "${SORT_THREADS}" "02_mapping/${PREFIX}.bam"

# ------------------------------------------------------------------------------ step 4: QC
echo "[INFO] Step 4: BAM QC (requested by the reviewer: mapping rates and MAPQ)"
BAM="02_mapping/${PREFIX}.bam"
samtools stats    -@ "${SORT_THREADS}" "${BAM}" > "logs/${PREFIX}.stats.txt"
samtools flagstat -@ "${SORT_THREADS}" "${BAM}" > "logs/${PREFIX}.flagstat.txt"
samtools idxstats "${BAM}"                        > "logs/${PREFIX}.idxstats.txt"   # reads per contig (target vs decoys)
samtools view -@ "${SORT_THREADS}" -F 0x904 "${BAM}" | cut -f5 | sort -n | uniq -c \
                                                  > "logs/${PREFIX}.mapq_dist.txt"  # primary alignments only

echo "================================================================================"
echo "[INFO] Finished       : $(date '+%F %T')"
echo "[INFO] BAM            : ${ABS_OUTPUT}/${BAM}"
echo "[INFO] QC             : ${ABS_OUTPUT}/logs/${PREFIX}.{stats,flagstat,idxstats,mapq_dist}.txt"
echo "================================================================================"
