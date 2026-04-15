#!/usr/bin/env bash
set -u
set -o pipefail

PJDIR="/media/user/4TB/JUNO/data/PJ14"
RAW_DIR="$PJDIR/raw"
SRC_DIR="$PJDIR/src_snapshot"
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

run_stage() {
  local stage_num="$1"
  local stage_name="$2"
  local img_label="$3"
  local log_file="$4"
  shift 4

  banner "Stage ${stage_num} Beginning...   [${img_label}]"
  log "Logging to: $log_file"

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
fail_count=0

banner "PJ14 Batch Run: Stage 03 -> Stage 07"
log "Total IMGs found : $total_imgs"
log "Source snapshot  : $SRC_DIR"
log "Logs directory   : $LOG_ROOT"

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

  if ! compgen -G "$stage2_dir/*_LIMBENDPOINTS.csv" > /dev/null; then
    subbanner "Skipping ${base}"
    log "Reason: no valid Stage 02 outputs remain after QC"
    ((skip_count+=1))
    continue
  fi

  mkdir -p "$stage3_dir" "$stage4_dir" "$stage5_dir" "$stage6_dir"
  mkdir -p "$stage4b_dir" "$stage7_dir"

  log "Output directories:"
  log "  Stage 03 : $stage3_dir"
  log "  Stage 04 : $stage4_dir"
  log "  Stage 04b: $stage4b_dir"
  log "  Stage 05 : $stage5_dir"
  log "  Stage 06 : $stage6_dir"
  log "  Stage 07 : $stage7_dir"

  subbanner "Stage 03 input check"
  log "Image ${base} is beginning Stage 03 processing..."

  if ! run_stage "03" "Rectify Limb" "$base" "$stage3_log" \
    python "$STAGE3_PY" \
      --indir "$stage2_dir" \
      --imagedir "$stage1_dir" \
      --outdir "$stage3_dir"
  then
    ((fail_count+=1))
    continue
  fi

  if ! compgen -G "$stage3_dir/*_RECTIFIED.tif" > /dev/null; then
    banner "Stage 03 FAILED VALIDATION   [${base}]"
    log "FAIL: Stage 03 produced no *_RECTIFIED.tif"
    ((fail_count+=1))
    continue
  fi

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

  if ! compgen -G "$stage4_dir/*_RECTIFIED_PERP_SAMPLES.csv" > /dev/null; then
    banner "Stage 04 FAILED VALIDATION   [${base}]"
    log "FAIL: Stage 04 produced no *_RECTIFIED_PERP_SAMPLES.csv"
    ((fail_count+=1))
    continue
  fi

  subbanner "Here comes Stage 04b..."

  if ! run_stage "04b" "Detectability Analysis" "$base" "$stage4b_log" \
    python "$STAGE4B_PY" \
      --indir "$stage4_dir" \
      --outdir "$stage4b_dir"
  then
    ((fail_count+=1))
    continue
  fi

  detectability_csv="$stage4b_dir/stage_04b_fragment_detectability.csv"
  if [[ ! -f "$detectability_csv" ]]; then
    banner "Stage 04b FAILED VALIDATION   [${base}]"
    log "FAIL: Stage 04b missing detectability CSV"
    ((fail_count+=1))
    continue
  fi

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

  mapfile -t stage6_csvs < <(find "$stage6_dir" -type f -name '*_STAGE6.csv' | sort)

  if [[ ${#stage6_csvs[@]} -eq 0 ]]; then
    banner "Stage 06 FAILED VALIDATION   [${base}]"
    log "FAIL: Stage 06 produced no *_STAGE6.csv"
    ((fail_count+=1))
    continue
  fi

  log "Stage 06 validation found ${#stage6_csvs[@]} *_STAGE6.csv file(s)"

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

  if [[ ! -f "$stage7_dir/stage7_fragment_table.csv" ]]; then
    banner "Stage 07 FAILED VALIDATION   [${base}]"
    log "FAIL: Stage 07 missing stage7_fragment_table.csv"
    ((fail_count+=1))
    continue
  fi

  banner "${base} completed Stage 03 -> Stage 07 successfully"
  ((success_count+=1))
done

banner "PJ14 Batch Complete"
log "Succeeded : $success_count"
log "Skipped   : $skip_count"
log "Failed    : $fail_count"

if [[ "$fail_count" -eq 0 ]]; then
  banner "ALL CLEAR"
  log "Stage 03 -> Stage 07 complete for PJ14"
  log "Next step: Stage 08 perijove aggregation"
else
  banner "RUN FINISHED WITH SOME FAILURES"
  log "Review logs in: $LOG_ROOT"
fi