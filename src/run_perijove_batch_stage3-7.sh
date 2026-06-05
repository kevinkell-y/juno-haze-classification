#!/usr/bin/env bash

set -euo pipefail

PJ="${1:?Usage: ./run_perijove_batch_stage3-7.sh <PJ_ID>   Example: ./run_perijove_batch_stage3-7.sh PJ05}"

# Determine repository root automatically from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUNO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PJDIR="$JUNO_ROOT/data/$PJ"
RAW_DIR="$PJDIR/raw"
SRC_DIR="$JUNO_ROOT/src"
CUB_ROOT="$PJDIR/cub"
ANALYSIS_ROOT="$PJDIR/analysis"
LOG_ROOT="$PJDIR/logs"

STAGE3_PY="$SRC_DIR/03.rectify_limb.py"
STAGE4_PY="$SRC_DIR/04.plot_rectified_perpendiculars.py"
STAGE4B_PY="$SRC_DIR/04b.slant_distance_detectability.py"
STAGE5_PY="$SRC_DIR/05.graph_profiles.py"
STAGE6_PY="$SRC_DIR/06.peakfinder.py"
STAGE7_PY="$SRC_DIR/07.haze_analysis.py"

SMOOTH_WINDOW=5
BIN_WIDTH_DEG=1.0
MIN_FRAGMENTS_PER_BIN=15

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

log() {
  echo "[$(timestamp)] $*"
}

line() {
  printf '%0.s=' {1..72}
  echo
}

banner() {
  local msg="$1"
  echo
  line
  echo "$msg"
  line
}

subbanner() {
  local msg="$1"
  echo
  echo "----------------------------------------------------------------"
  echo "$msg"
  echo "----------------------------------------------------------------"
}

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    log "ERROR: Missing file: $f"
    exit 1
  fi
}

require_dir() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    log "ERROR: Missing directory: $d"
    exit 1
  fi
}

require_command() {
  local c="$1"
  if ! command -v "$c" >/dev/null 2>&1; then
    log "ERROR: Required command not found: $c"
    log "Did you forget to activate the ISIS conda environment?"
    log "Try: conda activate isis-8.3.0"
    exit 1
  fi
}

count_files() {
  local dir="$1"
  local pattern="$2"
  [[ -d "$dir" ]] || { echo 0; return; }
  find "$dir" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | wc -l
}

count_files_recursive() {
  local dir="$1"
  local pattern="$2"
  [[ -d "$dir" ]] || { echo 0; return; }
  find "$dir" -type f -name "$pattern" 2>/dev/null | wc -l
}

has_zero_byte_files() {
  local dir="$1"
  [[ -d "$dir" ]] && find "$dir" -type f -size 0 | grep -q .
}

stage_valid_exact_count_no_zero() {
  local dir="$1"
  local pattern="$2"
  local expected="$3"

  [[ -d "$dir" ]] || return 1

  local n
  n=$(count_files "$dir" "$pattern")

  if [[ "$n" -ne "$expected" ]]; then
    return 1
  fi

  if has_zero_byte_files "$dir"; then
    return 1
  fi

  return 0
}

stage_valid_exact_count_recursive_no_zero() {
  local dir="$1"
  local pattern="$2"
  local expected="$3"

  [[ -d "$dir" ]] || return 1

  local n
  n=$(count_files_recursive "$dir" "$pattern")

  if [[ "$n" -ne "$expected" ]]; then
    return 1
  fi

  if has_zero_byte_files "$dir"; then
    return 1
  fi

  return 0
}

reset_from_stage() {
  local start_stage="$1"

  case "$start_stage" in
    03)
      log "Cleaning partial/incomplete outputs from Stage 03 onward"
      rm -rf "$stage3_dir" "$stage4_dir" "$stage4b_dir" "$stage5_dir" "$stage6_dir" "$stage7_dir"
      ;;
    04)
      log "Cleaning partial/incomplete outputs from Stage 04 onward"
      rm -rf "$stage4_dir" "$stage4b_dir" "$stage5_dir" "$stage6_dir" "$stage7_dir"
      ;;
    04b)
      log "Cleaning partial/incomplete outputs from Stage 04b onward"
      rm -rf "$stage4b_dir" "$stage5_dir" "$stage6_dir" "$stage7_dir"
      ;;
    05)
      log "Cleaning partial/incomplete outputs from Stage 05 onward"
      rm -rf "$stage5_dir" "$stage6_dir" "$stage7_dir"
      ;;
    06)
      log "Cleaning partial/incomplete outputs from Stage 06 onward"
      rm -rf "$stage6_dir" "$stage7_dir"
      ;;
    07)
      log "Cleaning partial/incomplete outputs from Stage 07 onward"
      rm -rf "$stage7_dir"
      ;;
    *)
      log "ERROR: Unknown reset stage: $start_stage"
      exit 1
      ;;
  esac

  mkdir -p "$stage3_dir" "$stage4_dir" "$stage4b_dir" "$stage5_dir" "$stage6_dir" "$stage7_dir"
}

run_stage() {
  local stage_num="$1"
  local stage_name="$2"
  local img_label="$3"
  local log_file="$4"
  shift 4

  banner "Stage ${stage_num} Beginning...   [${img_label}]"
  log "Stage name : $stage_name"
  log "Logging to : $log_file"

  "$@" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}

  if [[ $status -ne 0 ]]; then
    banner "Stage ${stage_num} FAILED   [${img_label}]"
    log "See log: $log_file"
    return $status
  fi

  banner "Stage ${stage_num} Completed Successfully   [${img_label}]"
  return 0
}

mkdir -p "$ANALYSIS_ROOT" "$LOG_ROOT"

require_dir "$PJDIR"
require_dir "$RAW_DIR"
require_dir "$SRC_DIR"
require_dir "$CUB_ROOT"

require_file "$STAGE3_PY"
require_file "$STAGE4_PY"
require_file "$STAGE4B_PY"
require_file "$STAGE5_PY"
require_file "$STAGE6_PY"
require_file "$STAGE7_PY"

require_command python
require_command getkey
require_command campt

shopt -s nullglob
imgs=( "$RAW_DIR"/*.IMG )

if [[ ${#imgs[@]} -eq 0 ]]; then
  log "ERROR: No .IMG files found in $RAW_DIR"
  exit 1
fi

total_imgs=${#imgs[@]}
current_img=0
success_count=0
skip_count=0
resume_count=0
fail_count=0

banner "${PJ} Batch Run: Stage 03 -> Stage 07"
log "Total IMGs found : $total_imgs"
log "JUNO root        : $JUNO_ROOT"
log "PJ directory     : $PJDIR"
log "Source directory : $SRC_DIR"
log "Logs directory   : $LOG_ROOT"
log "ISIS campt       : $(command -v campt)"
log "ISIS getkey      : $(command -v getkey)"

for img in "${imgs[@]}"; do
  ((current_img+=1))
  base="$(basename "$img" .IMG)"

  img_cub_root="$CUB_ROOT/$base"
  img_analysis_root="$ANALYSIS_ROOT/$base"

  stage1_dir="$img_cub_root/stage_01_framelets"
  stage2_dir="$img_cub_root/stage_02_trace_polyline"
  stage3_dir="$img_cub_root/stage_03_rectified"
  stage4_dir="$img_cub_root/stage_04_perp_samples"
  stage5_dir="$img_cub_root/stage_05_graphs"
  stage6_dir="$img_cub_root/stage_06_peaks"

  stage4b_dir="$img_analysis_root/stage_04b_detectability"
  stage7_dir="$img_analysis_root/stage_07"

  stage3_log="$LOG_ROOT/${base}_03.log"
  stage4_log="$LOG_ROOT/${base}_04.log"
  stage4b_log="$LOG_ROOT/${base}_04b.log"
  stage5_log="$LOG_ROOT/${base}_05.log"
  stage6_log="$LOG_ROOT/${base}_06.log"
  stage7_log="$LOG_ROOT/${base}_07.log"

  banner "IMG ${current_img} of ${total_imgs} : ${base}"

  require_dir "$img_cub_root"
  require_dir "$stage1_dir"
  require_dir "$stage2_dir"

  stage2_csv_count=$(count_files "$stage2_dir" "*.csv")
  stage2_png_count=$(count_files "$stage2_dir" "*.png")
  stage2_limb_count=$(count_files "$stage2_dir" "*_LIMBENDPOINTS.csv")
  stage2_overlay_count=$(count_files "$stage2_dir" "*_OVERLAY.png")

  log "Stage 02 QC state:"
  log "  CSV total          : $stage2_csv_count"
  log "  PNG total          : $stage2_png_count"
  log "  LIMBENDPOINTS CSV  : $stage2_limb_count"
  log "  OVERLAY PNG        : $stage2_overlay_count"

  if [[ "$stage2_csv_count" -ne "$stage2_png_count" ]]; then
    log "WARNING: Stage 02 total CSV/PNG counts differ: CSV=$stage2_csv_count PNG=$stage2_png_count"
  fi

  orphan_count=0

  for csv in "$stage2_dir"/*_LIMBENDPOINTS.csv; do
    [[ -e "$csv" ]] || continue

    png="${csv%_LIMBENDPOINTS.csv}_OVERLAY.png"

    if [[ ! -f "$png" ]]; then
      log "ERROR: Orphan Stage 02 CSV has no matching overlay PNG:"
      log "       CSV: $csv"
      log "       PNG: $png"
      ((orphan_count+=1))
    fi
  done

  if [[ "$orphan_count" -gt 0 ]]; then
    banner "Stage 02 QC FAILED   [${base}]"
    log "Found $orphan_count orphan LIMBENDPOINTS CSV file(s)."
    log "Fix Stage 02.5 QC before running Stage 03 -> 07."
    ((fail_count+=1))
    continue
  fi

  if [[ "$stage2_csv_count" -eq 0 ]]; then
    subbanner "Skipping ${base}"
    log "Reason: no Stage 02 CSV files found"
    ((skip_count+=1))
    continue
  fi

  if has_zero_byte_files "$stage2_dir"; then
    subbanner "Skipping ${base}"
    log "Reason: zero-byte files detected in Stage 02"
    ((skip_count+=1))
    continue
  fi

  if [[ "$stage2_limb_count" -eq 0 ]]; then
    subbanner "Skipping ${base}"
    log "Reason: no valid Stage 02 LIMBENDPOINTS CSV outputs remain after QC"
    ((skip_count+=1))
    continue
  fi

  spice_fail_count=0

  log "Stage 02 SPICE preflight:"
  log "  Checking accepted LIMBENDPOINTS survivors for matching Stage 01 cube + TargetPosition..."

  for csv in "$stage2_dir"/*_LIMBENDPOINTS.csv; do
    [[ -e "$csv" ]] || continue

    stem="$(basename "$csv" _LIMBENDPOINTS.csv)"
    cub="$stage1_dir/${stem}.cub"

    if [[ ! -f "$cub" ]]; then
      log "ERROR: Missing Stage 01 cube for accepted Stage 02 survivor:"
      log "       CSV: $csv"
      log "       CUB: $cub"
      ((spice_fail_count+=1))
      continue
    fi

    if ! getkey from="$cub" grpname=Kernels keyword=TargetPosition >/dev/null 2>&1; then
      log "ERROR: Accepted Stage 02 survivor has incomplete SPICE kernels:"
      log "       CSV: $csv"
      log "       CUB: $cub"
      log "       Missing Kernels/TargetPosition. Run/re-run spiceinit or reject this survivor."
      ((spice_fail_count+=1))
      continue
    fi
  done

  if [[ "$spice_fail_count" -gt 0 ]]; then
    banner "Stage 02 SPICE PREFLIGHT FAILED   [${base}]"
    log "Found $spice_fail_count accepted Stage 02 survivor(s) whose Stage 01 cubes lack TargetPosition."
    log "Do not run Stage 03 -> 07 for this image until Stage 01/SPICE is fixed or those survivors are rejected."
    ((fail_count+=1))
    continue
  fi

  log "Stage 02 SPICE preflight passed."

  mkdir -p "$stage3_dir" "$stage4_dir" "$stage4b_dir" "$stage5_dir" "$stage6_dir" "$stage7_dir"

  rm -f "$stage3_dir"/*.tmp 2>/dev/null || true

  stage3_count=$(count_files "$stage3_dir" "*_RECTIFIED.tif")
  stage4_count=$(count_files "$stage4_dir" "*_RECTIFIED_PERP_SAMPLES.csv")
  stage5_file_count=$(find "$stage5_dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
  stage6_count=$(count_files_recursive "$stage6_dir" "*_STAGE6.csv")

  detectability_csv="$stage4b_dir/stage_04b_fragment_detectability.csv"
  stage7_csv="$stage7_dir/stage7_fragment_table.csv"

  log "Resume audit:"
  log "  Expected from Stage 02 LIMBENDPOINTS : $stage2_limb_count"
  log "  Stage 03 rectified TIFs             : $stage3_count"
  log "  Stage 04 perp sample CSVs           : $stage4_count"
  log "  Stage 04b detectability CSV         : $([[ -s "$detectability_csv" ]] && echo present || echo missing_or_empty)"
  log "  Stage 05 file count                 : $stage5_file_count"
  log "  Stage 06 peak CSVs                  : $stage6_count"
  log "  Stage 07 fragment table             : $([[ -s "$stage7_csv" ]] && echo present || echo missing_or_empty)"

  if [[ -s "$stage7_csv" ]] && ! has_zero_byte_files "$stage7_dir"; then
    subbanner "Skipping ${base}"
    log "Reason: Stage 07 already completed successfully"
    ((skip_count+=1))
    continue
  fi

  resume_from=""

  if ! stage_valid_exact_count_no_zero "$stage3_dir" "*_RECTIFIED.tif" "$stage2_limb_count"; then
    resume_from="03"
  elif ! stage_valid_exact_count_no_zero "$stage4_dir" "*_RECTIFIED_PERP_SAMPLES.csv" "$stage3_count"; then
    resume_from="04"
  elif [[ ! -s "$detectability_csv" ]] || has_zero_byte_files "$stage4b_dir"; then
    resume_from="04b"
  elif has_zero_byte_files "$stage5_dir"; then
    resume_from="05"
  elif ! stage_valid_exact_count_recursive_no_zero "$stage6_dir" "*_STAGE6.csv" "$stage4_count"; then
    resume_from="06"
  elif [[ ! -s "$stage7_csv" ]] || has_zero_byte_files "$stage7_dir"; then
    resume_from="07"
  fi

  if [[ -n "$resume_from" ]]; then
    subbanner "Resuming ${base} from Stage ${resume_from}"
    log "Reason: first incomplete/invalid stage detected by output validation"
    ((resume_count+=1))
    reset_from_stage "$resume_from"
  fi

  log "Output directories:"
  log "  Stage 03 : $stage3_dir"
  log "  Stage 04 : $stage4_dir"
  log "  Stage 04b: $stage4b_dir"
  log "  Stage 05 : $stage5_dir"
  log "  Stage 06 : $stage6_dir"
  log "  Stage 07 : $stage7_dir"

  if [[ "$resume_from" == "03" || -z "$resume_from" ]]; then
    if ! stage_valid_exact_count_no_zero "$stage3_dir" "*_RECTIFIED.tif" "$stage2_limb_count"; then
      subbanner "Here comes Stage 03..."

      if ! run_stage "03" "Rectify Limb" "$base" "$stage3_log" \
        python "$STAGE3_PY" \
          --indir "$stage2_dir" \
          --imagedir "$stage1_dir" \
          --outdir "$stage3_dir"
      then
        ((fail_count+=1))
        continue
      fi
    else
      log "Stage 03 already valid; skipping"
    fi
  else
    log "Stage 03 already valid; skipping"
  fi

  stage3_count=$(count_files "$stage3_dir" "*_RECTIFIED.tif")
  if ! stage_valid_exact_count_no_zero "$stage3_dir" "*_RECTIFIED.tif" "$stage2_limb_count"; then
    banner "Stage 03 FAILED VALIDATION   [${base}]"
    log "FAIL: expected $stage2_limb_count *_RECTIFIED.tif but found $stage3_count, or zero-byte files exist"
    ((fail_count+=1))
    continue
  fi

  if [[ "$resume_from" == "03" || "$resume_from" == "04" || -z "$resume_from" ]]; then
    if ! stage_valid_exact_count_no_zero "$stage4_dir" "*_RECTIFIED_PERP_SAMPLES.csv" "$stage3_count"; then
      subbanner "Here comes Stage 04..."

      if ! run_stage "04" "Perpendicular Sampling" "$base" "$stage4_log" \
        python "$STAGE4_PY" \
          --indir "$stage3_dir" \
          --framelet-dir "$stage1_dir" \
          --outdir "$stage4_dir" \
          --batch
      then
        ((fail_count+=1))
        continue
      fi
    else
      log "Stage 04 already valid; skipping"
    fi
  else
    log "Stage 04 already valid; skipping"
  fi

  stage4_count=$(count_files "$stage4_dir" "*_RECTIFIED_PERP_SAMPLES.csv")
  if ! stage_valid_exact_count_no_zero "$stage4_dir" "*_RECTIFIED_PERP_SAMPLES.csv" "$stage3_count"; then
    banner "Stage 04 FAILED VALIDATION   [${base}]"
    log "FAIL: expected $stage3_count *_RECTIFIED_PERP_SAMPLES.csv but found $stage4_count, or zero-byte files exist"
    ((fail_count+=1))
    continue
  fi

  if [[ "$resume_from" == "03" || "$resume_from" == "04" || "$resume_from" == "04b" || -z "$resume_from" ]]; then
    if [[ ! -s "$detectability_csv" ]] || has_zero_byte_files "$stage4b_dir"; then
      subbanner "Here comes Stage 04b..."

      if ! run_stage "04b" "Detectability Analysis" "$base" "$stage4b_log" \
        python "$STAGE4B_PY" \
          --indir "$stage4_dir" \
          --outdir "$stage4b_dir"
      then
        ((fail_count+=1))
        continue
      fi
    else
      log "Stage 04b already valid; skipping"
    fi
  else
    log "Stage 04b already valid; skipping"
  fi

  if [[ ! -s "$detectability_csv" ]] || has_zero_byte_files "$stage4b_dir"; then
    banner "Stage 04b FAILED VALIDATION   [${base}]"
    log "FAIL: missing/empty detectability CSV or zero-byte Stage 04b files detected"
    ((fail_count+=1))
    continue
  fi

  if [[ "$resume_from" == "03" || "$resume_from" == "04" || "$resume_from" == "04b" || "$resume_from" == "05" || -z "$resume_from" ]]; then
    if has_zero_byte_files "$stage5_dir" || [[ "$resume_from" == "05" || "$resume_from" == "03" || "$resume_from" == "04" || "$resume_from" == "04b" ]]; then
      subbanner "Here comes Stage 05..."

      if ! run_stage "05" "Graph Profiles" "$base" "$stage5_log" \
        python "$STAGE5_PY" \
          --indir "$stage4_dir" \
          --outdir "$stage5_dir" \
          --batch \
          --smooth "$SMOOTH_WINDOW"
      then
        ((fail_count+=1))
        continue
      fi
    else
      log "Stage 05 has no zero-byte files; skipping"
    fi
  else
    log "Stage 05 has no zero-byte files; skipping"
  fi

  if has_zero_byte_files "$stage5_dir"; then
    banner "Stage 05 FAILED VALIDATION   [${base}]"
    log "FAIL: zero-byte Stage 05 outputs detected"
    ((fail_count+=1))
    continue
  fi

  if [[ "$resume_from" == "03" || "$resume_from" == "04" || "$resume_from" == "04b" || "$resume_from" == "05" || "$resume_from" == "06" || -z "$resume_from" ]]; then
    if ! stage_valid_exact_count_recursive_no_zero "$stage6_dir" "*_STAGE6.csv" "$stage4_count"; then
      subbanner "Here comes Stage 06..."

      if ! run_stage "06" "Peak Finding" "$base" "$stage6_log" \
        python "$STAGE6_PY" \
          --indir "$stage4_dir" \
          --outdir "$stage6_dir" \
          --detectability-csv "$detectability_csv" \
          --smooth "$SMOOTH_WINDOW"
      then
        ((fail_count+=1))
        continue
      fi
    else
      log "Stage 06 already valid; skipping"
    fi
  else
    log "Stage 06 already valid; skipping"
  fi

  stage6_count=$(count_files_recursive "$stage6_dir" "*_STAGE6.csv")
  if ! stage_valid_exact_count_recursive_no_zero "$stage6_dir" "*_STAGE6.csv" "$stage4_count"; then
    banner "Stage 06 FAILED VALIDATION   [${base}]"
    log "FAIL: expected $stage4_count *_STAGE6.csv but found $stage6_count, or zero-byte files exist"
    ((fail_count+=1))
    continue
  fi

  log "Stage 06 validation found ${stage6_count} *_STAGE6.csv file(s)"

  if [[ "$resume_from" == "03" || "$resume_from" == "04" || "$resume_from" == "04b" || "$resume_from" == "05" || "$resume_from" == "06" || "$resume_from" == "07" || -z "$resume_from" ]]; then
    if [[ ! -s "$stage7_csv" ]] || has_zero_byte_files "$stage7_dir"; then
      subbanner "Here comes Stage 07..."

      if ! run_stage "07" "Haze Aggregation" "$base" "$stage7_log" \
        python "$STAGE7_PY" \
          --imgdir "$img_cub_root" \
          --stage6-root "$stage6_dir" \
          --outdir "$stage7_dir" \
          --bin-width-deg "$BIN_WIDTH_DEG" \
          --min-fragments-per-bin "$MIN_FRAGMENTS_PER_BIN" \
          --img-label "$base"
      then
        ((fail_count+=1))
        continue
      fi
    else
      log "Stage 07 already valid; skipping"
    fi
  fi

  if [[ ! -s "$stage7_csv" ]] || has_zero_byte_files "$stage7_dir"; then
    banner "Stage 07 FAILED VALIDATION   [${base}]"
    log "FAIL: missing/empty stage7_fragment_table.csv or zero-byte Stage 07 outputs detected"
    ((fail_count+=1))
    continue
  fi

  banner "${base} completed Stage 03 -> Stage 07 successfully"
  ((success_count+=1))
done

banner "${PJ} Batch Complete"
log "Succeeded : $success_count"
log "Skipped   : $skip_count"
log "Resumed   : $resume_count"
log "Failed    : $fail_count"

if [[ "$fail_count" -eq 0 ]]; then
  banner "ALL CLEAR"
  log "Stage 03 -> Stage 07 complete for ${PJ}"
  log "Next step: Stage 08 perijove aggregation"
else
  banner "RUN FINISHED WITH SOME FAILURES"
  log "Review logs in: $LOG_ROOT"
fi
