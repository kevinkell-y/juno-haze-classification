#!/usr/bin/env bash
set -euo pipefail

# PJ14 batch runner: Stage 01 -> Stage 02
# Designed to stop AFTER Stage 02 so you can do manual QC before Stage 03.
#
# Assumptions:
# - Run from anywhere
# - Active conda env already has ISIS/Juno dependencies
# - Frozen code lives in: data/PJ14/src_snapshot
# - Raw IMG/LBL files live in: data/PJ14/raw

PJDIR="/media/user/4TB/JUNO/data/PJ14"
RAW_DIR="$PJDIR/raw"
SRC_DIR="$PJDIR/src_snapshot"
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

require_dir "$PJDIR"
require_dir "$RAW_DIR"
require_dir "$SRC_DIR"
require_file "$STAGE1_PY"
require_file "$STAGE2_PY"

shopt -s nullglob
imgs=( "$RAW_DIR"/*.IMG )

if [[ ${#imgs[@]} -eq 0 ]]; then
  fail "No .IMG files found in $RAW_DIR"
fi

log "Found ${#imgs[@]} IMG files in $RAW_DIR"
log "Using source snapshot: $SRC_DIR"
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

  # Skip logic: if Stage 02 already has outputs, assume done.
  if [[ -d "$stage2_out" ]] && compgen -G "$stage2_out/*_LIMBENDPOINTS.csv" > /dev/null; then
    log "Skipping $base (Stage 02 outputs already present)"
    ((skip_count+=1))
    continue
  fi

  # Stage 01
  if [[ ! -d "$stage1_out" ]] || ! compgen -G "$stage1_out/*.cub" > /dev/null; then
    log "Running Stage 01 for $base"
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

  # Stage 02
  log "Running Stage 02 for $base"
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

  log "Completed Stage 01-02 for $base"
  ((success_count+=1))
done

log "============================================================"
log "Batch complete"
log "Succeeded: $success_count"
log "Skipped:   $skip_count"
log "Failed:    $fail_count"
log "Manual QC is next:"
log "  - inspect stage_02_trace_polyline outputs"
log "  - delete bad limb traces"
log "  - remove framelets where Jupiter fills the frame"
log "  - only then proceed to Stage 03-07"
