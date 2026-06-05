#!/usr/bin/env bash
set -euo pipefail

# Perijove batch runner: Stage 01 -> Stage 02
# Designed to stop AFTER Stage 02 so manual Stage 2.5 QC can happen before Stage 03.
#
# Assumptions:
# - Run from anywhere inside/outside the repo
# - Active conda env already has ISIS/Juno dependencies
# - Script lives in: <repo>/src/
# - Raw IMG/LBL files live in: <repo>/data/<PJ>/raw
# - Outputs go to: <repo>/data/<PJ>/cub and <repo>/data/<PJ>/logs

PJ="${1:?Usage: ./run_perijove_batch_stage1-2.sh PJ05}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUNO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PJDIR="$JUNO_ROOT/data/$PJ"
RAW_DIR="$PJDIR/raw"
SRC_DIR="$JUNO_ROOT/src"
CUB_ROOT="$PJDIR/cub"
LOG_ROOT="$PJDIR/logs"

STAGE1_PY="$SRC_DIR/01.framelets_and_spice.py"
STAGE2_PY="$SRC_DIR/02.trace_limb_polyline.py"

mkdir -p "$CUB_ROOT" "$LOG_ROOT"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

fail() {
  echo "[$(timestamp)] ERROR: $*" >&2
  exit 1
}

require_file() {
  local f="$1"
  [[ -f "$f" ]] || fail "Missing required file: $f"
}

require_dir() {
  local d="$1"
  [[ -d "$d" ]] || fail "Missing required directory: $d"
}

require_command() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || fail "Required command not found: $c"
}

count_files() {
  local dir="$1"
  local pattern="$2"
  [[ -d "$dir" ]] || { echo 0; return; }
  find "$dir" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | wc -l
}

has_zero_byte_files() {
  local dir="$1"
  [[ -d "$dir" ]] && find "$dir" -type f -size 0 | grep -q .
}

require_dir "$RAW_DIR"
require_dir "$SRC_DIR"
require_file "$STAGE1_PY"
require_file "$STAGE2_PY"

require_command python
require_command junocam2isis
require_command spiceinit
require_command isis2std

shopt -s nullglob
imgs=( "$RAW_DIR"/*.IMG )

if [[ ${#imgs[@]} -eq 0 ]]; then
  fail "No .IMG files found in $RAW_DIR"
fi

log "Found ${#imgs[@]} IMG files in $RAW_DIR"
log "JUNO root: $JUNO_ROOT"
log "Using source directory: $SRC_DIR"
log "Starting Stage 01 -> Stage 02 batch run"

success_count=0
skip_count=0
fail_count=0

for img in "${imgs[@]}"; do
  base="$(basename "$img" .IMG)"
  lbl="$RAW_DIR/${base}.LBL"

  img_dir="$CUB_ROOT/$base"
  stage1_out="$img_dir/stage_01_framelets"
  stage2_out="$img_dir/stage_02_trace_polyline"

  stage1_log="$LOG_ROOT/${base}_stage01.log"
  stage2_log="$LOG_ROOT/${base}_stage02.log"

  require_file "$lbl"
  mkdir -p "$img_dir"

  log "------------------------------------------------------------"
  log "IMG: $base"

  if [[ -f "$stage2_out/.manual_qc_done" ]]; then
    log "Skipping $base (Stage 02 manually QC'd)"
    ((skip_count+=1))
    continue
  fi

  # Stage 01
  stage1_cub_count=$(count_files "$stage1_out" "*.cub")

  if [[ "$stage1_cub_count" -eq 0 ]] || has_zero_byte_files "$stage1_out"; then
    log "Running Stage 01 for $base"

    rm -rf "$stage1_out"

    {
      echo "[$(timestamp)] START Stage 01: $base"
      python "$STAGE1_PY" \
        --img "$img" \
        --outdir "$stage1_out"
      echo "[$(timestamp)] END Stage 01: $base"
    } >"$stage1_log" 2>&1 || {
      log "Stage 01 FAILED for $base (see $stage1_log)"
      ((fail_count+=1))
      continue
    }
  else
    log "Stage 01 already exists for $base; skipping Stage 01"
  fi

  # Stage 01 validation
  stage1_cub_count=$(count_files "$stage1_out" "*.cub")
  if [[ "$stage1_cub_count" -eq 0 ]] || has_zero_byte_files "$stage1_out"; then
    log "Stage 01 validation FAILED for $base"
    ((fail_count+=1))
    continue
  fi

  # Stage 02
  stage2_limb_count=$(count_files "$stage2_out" "*_LIMBENDPOINTS.csv")
  stage2_overlay_count=$(count_files "$stage2_out" "*_OVERLAY.png")

  if [[ "$stage2_limb_count" -eq "$stage1_cub_count" ]] && \
     [[ "$stage2_overlay_count" -eq "$stage1_cub_count" ]] && \
     ! has_zero_byte_files "$stage2_out"; then
    log "Skipping Stage 02 for $base (complete outputs already present)"
    ((skip_count+=1))
    continue
  fi

  log "Running Stage 02 for $base"

  rm -rf "$stage2_out"

  {
    echo "[$(timestamp)] START Stage 02: $base"
    python "$STAGE2_PY" \
      --cubdir "$stage1_out" \
      --outdir "$stage2_out" \
      --noshow
    echo "[$(timestamp)] END Stage 02: $base"
  } >"$stage2_log" 2>&1 || {
    log "Stage 02 FAILED for $base (see $stage2_log)"
    ((fail_count+=1))
    continue
  }

  # Stage 02 validation
  stage2_limb_count=$(count_files "$stage2_out" "*_LIMBENDPOINTS.csv")
  stage2_overlay_count=$(count_files "$stage2_out" "*_OVERLAY.png")

  if [[ "$stage2_limb_count" -eq 0 ]] || \
     [[ "$stage2_overlay_count" -eq 0 ]] || \
     [[ "$stage2_limb_count" -ne "$stage2_overlay_count" ]] || \
     has_zero_byte_files "$stage2_out"; then
    log "Stage 02 validation FAILED for $base"
    log "  LIMBENDPOINTS CSV: $stage2_limb_count"
    log "  OVERLAY PNG      : $stage2_overlay_count"
    ((fail_count+=1))
    continue
  fi

  log "Completed Stage 01-02 for $base"
  log "  Stage 01 cubes       : $stage1_cub_count"
  log "  Stage 02 limb CSVs   : $stage2_limb_count"
  log "  Stage 02 overlay PNGs: $stage2_overlay_count"

  ((success_count+=1))
done

log "============================================================"
log "Batch complete"
log "Succeeded: $success_count"
log "Skipped:   $skip_count"
log "Failed:    $fail_count"
log "Manual QC is next:"
log "  - inspect stage_02_trace_polyline outputs"
log "  - delete bad *_LIMBENDPOINTS.csv and matching *_OVERLAY.png"
log "  - remove framelets where Jupiter fills the frame"
log "  - only then proceed to Stage 03-07"