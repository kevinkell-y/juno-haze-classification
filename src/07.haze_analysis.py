#!/usr/bin/env python3
"""
Stage 7 — Aggregate Stage 6 peak classifications across fragments/framelets,
and evaluate detached haze occurrence as a function of latitude using the
SPICE latitude at the detected limb crossing (PRIMARY peak).

Inputs:
  - *_STAGE6.csv files produced by src/06.peakfinder.py

Outputs (in --outdir):
  - stage7_fragment_table.csv
  - stage7_latitude_bins.csv
  - stage7_occurrence_by_lat.png

Scientific role of Stage 7:
  - Build one fragment-level record per accepted Stage 6 fragment
  - Use the PRIMARY limb-crossing latitude (or nearest finite SPICE fallback)
  - Count whether each fragment has a SECONDARY peak
  - IMPORTANT: include only fragments with detectability_flag == True,
    so occurrence statistics are computed only from geometrically valid,
    detectable observations
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--imgdir",
        type=str,
        required=True,
        help="IMG-scoped root directory, e.g. data/JNCR_2018197_14C00024_V01",
    )
    ap.add_argument(
        "--stage6-root",
        type=str,
        default=None,
        help="Optional override. Default: <imgdir>/cub/stage_06_peaks",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="**/*_STAGE6.csv",
        help="Glob pattern under --stage6-root to find Stage 6 CSVs.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional override. Default: <imgdir>/analysis/stage_07",
    )
    ap.add_argument(
        "--bin-width-deg",
        type=float,
        default=1.0,
        help="Latitude bin width in degrees.",
    )
    ap.add_argument(
        "--min-fragments-per-bin",
        type=int,
        default=15,
        help="Bins with fewer fragments than this are retained but flagged as low-N.",
    )
    ap.add_argument(
        "--lat-col",
        type=str,
        default="PlanetocentricLatitude",
        help="Column name for planetocentric latitude in the Stage 6 CSV.",
    )
    ap.add_argument(
        "--peak-col",
        type=str,
        default="stage6_peak_type",
        help="Column name for peak labels in Stage 6 CSV.",
    )
    ap.add_argument(
        "--img-label",
        type=str,
        default=None,
        help="Optional label to stamp on the Stage 7 plot (top-left). If omitted, inferred from Stage 6 filenames.",
    )
    return ap.parse_args()


# ============================================================
# Fragment-level record
# ============================================================

@dataclass
class FragmentRecord:
    framelet: str
    fragment_id: int
    n_samples: int

    # Limb-crossing latitude used for Stage 7 x-axis placement
    limb_lat: float | None
    lat_source: str  # "primary", "nearest_spice", "none"

    # Detached haze occurrence information
    has_secondary: bool
    secondary_count: int

    # Retained for auditability / comparison with older logic
    first_spice_lat: float | None

    # Detectability provenance carried from Stage 4b -> Stage 6 -> Stage 7
    detectability_flag: bool | None
    detectability_reason: str | None
    first_valid_slant_distance_km: float | None
    km_per_pixel_at_limb: float | None
    pixels_per_detachment: float | None


# ============================================================
# Helpers
# ============================================================

def infer_framelet_name(stage6_csv: Path) -> str:
    """
    Preferred: Stage 6 CSV filename is "<FRAMELET>_STAGE6.csv"
    Fallback: parent folder name
    """
    m = re.match(r"(.+)_STAGE6\.csv$", stage6_csv.name)
    if m:
        return m.group(1)
    return stage6_csv.parent.name


def infer_img_label_from_stage6(stage6_csvs: list[Path]) -> str:
    """
    Infer IMG ID from Stage 6 CSVs by stripping color and framelet suffixes.
    Example:
      JNCR_2018197_14C00024_V01_BLUE_0003 -> JNCR_2018197_14C00024_V01
    """
    if not stage6_csvs:
        return "UNKNOWN_IMG"

    framelet = infer_framelet_name(stage6_csvs[0])

    m = re.match(r"(JNCR_[^_]+_[^_]+_V\d+)", framelet)
    if m:
        return m.group(1)

    return framelet


def first_finite_index(x: np.ndarray) -> int | None:
    finite = np.isfinite(x)
    if not finite.any():
        return None
    return int(np.argmax(finite))  # first True


def normalize_bool_series(s: pd.Series) -> pd.Series:
    """
    Robustly normalize a pandas Series into True / False / NaN.

    Handles:
      - actual booleans
      - 0 / 1
      - strings like "true", "false", "True", "False"
    """
    if s.dtype == bool:
        return s

    if pd.api.types.is_numeric_dtype(s):
        out = s.copy()
        out = out.where(out.isna(), out.astype(float) != 0.0)
        return out

    lowered = s.astype(str).str.strip().str.lower()
    mapped = lowered.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "nan": np.nan,
        "none": np.nan,
        "": np.nan,
    })
    return mapped


def require_columns(df: pd.DataFrame, csv_path: Path, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} missing required columns: {missing}")


def summarize_detectability_for_fragment(frag: pd.DataFrame, csv_path: Path) -> tuple[bool | None, str | None, float | None, float | None, float | None]:
    """
    Extract fragment-level detectability provenance from a Stage 6 fragment.

    Since Stage 6 duplicates fragment-level provenance across all rows,
    we expect detectability fields to be constant within a fragment.
    This helper checks that and returns the representative values.
    """
    required = [
        "detectability_flag",
        "detectability_reason",
        "first_valid_slant_distance_km",
        "km_per_pixel_at_limb",
        "pixels_per_detachment",
    ]
    require_columns(frag, csv_path, required)

    det_flag = normalize_bool_series(frag["detectability_flag"])
    det_reason = frag["detectability_reason"].astype(str)

    # Consistency checks within the fragment
    unique_flag = pd.unique(det_flag.dropna())
    if len(unique_flag) > 1:
        raise RuntimeError(
            f"{csv_path.name} fragment {int(frag['fragment_id'].iloc[0])} has inconsistent detectability_flag values"
        )

    unique_reason = pd.unique(det_reason.fillna(""))
    if len(unique_reason) > 1:
        raise RuntimeError(
            f"{csv_path.name} fragment {int(frag['fragment_id'].iloc[0])} has inconsistent detectability_reason values"
        )

    def first_finite(colname: str) -> float | None:
        vals = pd.to_numeric(frag[colname], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            return None
        return float(vals.iloc[0])

    if len(unique_flag) == 0:
        flag_val = None
    else:
        flag_val = bool(unique_flag[0])

    reason_val = None if len(unique_reason) == 0 else str(unique_reason[0])

    return (
        flag_val,
        reason_val,
        first_finite("first_valid_slant_distance_km"),
        first_finite("km_per_pixel_at_limb"),
        first_finite("pixels_per_detachment"),
    )


# ============================================================
# Core Stage 7 aggregation
# ============================================================

def build_fragment_table(stage6_csvs: list[Path], lat_col: str, peak_col: str) -> pd.DataFrame:
    """
    Build one fragment-level record per usable Stage 6 fragment.

    The filtering logic is the key scientific change:
      1) fragment must be accepted
      2) fragment must not carry a rejection reason
      3) fragment must have detectability_flag == True

    This is what makes Stage 7 occurrence statistics 'quantifiably defensible':
    non-detections are only counted if the fragment was geometrically capable
    of resolving detached haze in the first place.
    """
    records: list[FragmentRecord] = []

    for csv_path in stage6_csvs:
        df = pd.read_csv(csv_path)

        require_columns(df, csv_path, ["fragment_id", "accepted", peak_col, lat_col])

        # Stage 6 fragment acceptance / rejection filter
        rej = df["reject_reason"].fillna("").astype(str)
        accepted = normalize_bool_series(df["accepted"])

        df = df[
            (accepted == True) &
            (rej == "")
        ].reset_index(drop=True)

        if df.empty:
            print(f"[warn] {csv_path.name}: no accepted fragments after Stage 6 filtering")
            continue

        framelet = infer_framelet_name(csv_path)

        for frag_id in sorted(df["fragment_id"].unique()):
            frag = df[df["fragment_id"] == frag_id].reset_index(drop=True)

            # --------------------------------------------------
            # Stage 4b/6 detectability filter
            # --------------------------------------------------
            (
                detectability_flag,
                detectability_reason,
                first_valid_slant_distance_km,
                km_per_pixel_at_limb,
                pixels_per_detachment,
            ) = summarize_detectability_for_fragment(frag, csv_path)

            # Scientific rule:
            # only geometrically detectable fragments contribute to Stage 7
            if detectability_flag is not True:
                continue

            # --------------------------------------------------
            # Determine limb-associated SPICE latitude
            # Priority:
            #   1) PRIMARY-row SPICE latitude
            #   2) Nearest SPICE-valid row to PRIMARY index
            # --------------------------------------------------
            primary_rows = frag[frag[peak_col] == "PRIMARY"]

            limb_lat = None
            lat_source = "none"

            if len(primary_rows) == 1:
                primary_idx = primary_rows.index[0]
                primary_lat = pd.to_numeric(primary_rows[lat_col], errors="coerce").iloc[0]

                if np.isfinite(primary_lat):
                    limb_lat = float(primary_lat)
                    lat_source = "primary"
                else:
                    # Fallback to nearest SPICE-valid row
                    lat_vals = pd.to_numeric(frag[lat_col], errors="coerce").to_numpy(dtype=float)
                    finite_idx = np.where(np.isfinite(lat_vals))[0]

                    if len(finite_idx) > 0:
                        nearest = finite_idx[np.argmin(np.abs(finite_idx - primary_idx))]
                        limb_lat = float(lat_vals[nearest])
                        lat_source = "nearest_spice"

            # Retain first finite SPICE latitude for auditability / comparison
            lat = pd.to_numeric(frag[lat_col], errors="coerce").to_numpy(dtype=float)
            first_i = first_finite_index(lat)

            if first_i is None:
                first_lat = None
            else:
                first_lat = float(lat[first_i])

            # Count SECONDARY peaks
            peak_types = frag[peak_col].astype(str).to_numpy()
            sec_mask = peak_types == "SECONDARY"
            sec_count = int(sec_mask.sum())

            rec = FragmentRecord(
                framelet=framelet,
                fragment_id=int(frag_id),
                n_samples=int(len(frag)),
                limb_lat=limb_lat,
                lat_source=lat_source,
                has_secondary=bool(sec_count > 0),
                secondary_count=sec_count,
                first_spice_lat=first_lat,
                detectability_flag=detectability_flag,
                detectability_reason=detectability_reason,
                first_valid_slant_distance_km=first_valid_slant_distance_km,
                km_per_pixel_at_limb=km_per_pixel_at_limb,
                pixels_per_detachment=pixels_per_detachment,
            )
            records.append(rec)

    out = pd.DataFrame([r.__dict__ for r in records])
    return out


def bin_latitudes(df_frag: pd.DataFrame, bin_width: float, min_n: int) -> pd.DataFrame:
    """
    Bin fragment-level detached-haze occurrence by limb-crossing latitude.

    Only fragments with a valid Stage 7 limb latitude are binned.
    """
    work = df_frag.copy()

    work = work[
        (work["lat_source"] != "none") &
        np.isfinite(pd.to_numeric(work["limb_lat"], errors="coerce"))
    ].copy()

    work["limb_lat"] = pd.to_numeric(work["limb_lat"], errors="coerce")

    if len(work) == 0:
        raise RuntimeError("No fragments had a finite limb_lat; cannot bin.")

    lat_min = math.floor(work["limb_lat"].min() / bin_width) * bin_width
    lat_max = math.ceil(work["limb_lat"].max() / bin_width) * bin_width
    edges = np.arange(lat_min, lat_max + bin_width, bin_width)

    work["lat_bin"] = pd.cut(work["limb_lat"], bins=edges, include_lowest=True)

    grouped = work.groupby("lat_bin", observed=True)
    summary = grouped.agg(
        n_fragments=("has_secondary", "size"),
        n_with_secondary=("has_secondary", "sum"),
        frac_with_secondary=("has_secondary", "mean"),
        lat_center=("limb_lat", "mean"),
        lat_min=("limb_lat", "min"),
        lat_max=("limb_lat", "max"),
    ).reset_index()

    summary["low_n_flag"] = summary["n_fragments"] < int(min_n)

    # Approximate 95% CI for a proportion (normal approximation)
    p = summary["frac_with_secondary"].to_numpy(float)
    n = summary["n_fragments"].to_numpy(float)
    se = np.sqrt(np.clip(p * (1 - p) / np.maximum(n, 1.0), 0, np.inf))
    summary["frac_ci95_lo"] = np.clip(p - 1.96 * se, 0, 1)
    summary["frac_ci95_hi"] = np.clip(p + 1.96 * se, 0, 1)

    return summary


def plot_occurrence(summary: pd.DataFrame, out_png: Path, img_label: str | None = None) -> None:
    """
    Plot detached haze occurrence fraction vs limb-crossing latitude.
    """
    x = summary["lat_center"].to_numpy(float)
    y = summary["frac_with_secondary"].to_numpy(float)
    ylo = summary["frac_ci95_lo"].to_numpy(float)
    yhi = summary["frac_ci95_hi"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o")
    ax.vlines(x, ylo, yhi)

    ax.set_xlabel("Planetocentric Latitude at Limb Crossing (deg)")
    ax.set_ylabel("Fraction of fragments with SECONDARY peak")
    ax.set_title("Detached haze occurrence binned by limb-crossing latitude")

    if img_label:
        ax.text(
            0.02, 0.98,
            f"JunoCam IMG: {img_label}",
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="top",
            horizontalalignment="left",
            color="#071ecd",
            alpha=1
        )

    # Mark low-N bins
    low_n = summary["low_n_flag"].to_numpy(bool)
    if low_n.any():
        ax.scatter(x[low_n], y[low_n], marker="x", s=120)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    imgdir = Path(args.imgdir).resolve()

    stage6_root = Path(args.stage6_root).resolve() if args.stage6_root else (imgdir / "cub" / "stage_06_peaks")
    outdir = Path(args.outdir).resolve() if args.outdir else (imgdir / "analysis" / "stage_07")

    outdir.mkdir(parents=True, exist_ok=True)

    stage6_csvs = sorted(stage6_root.glob(args.glob))
    if len(stage6_csvs) == 0:
        raise FileNotFoundError(f"No Stage 6 CSVs found under {stage6_root} with glob '{args.glob}'")

    df_frag = build_fragment_table(stage6_csvs, lat_col=args.lat_col, peak_col=args.peak_col)

    frag_out = outdir / "stage7_fragment_table.csv"
    df_frag.to_csv(frag_out, index=False)

    df_bins = bin_latitudes(
        df_frag,
        bin_width=float(args.bin_width_deg),
        min_n=int(args.min_fragments_per_bin)
    )
    bins_out = outdir / "stage7_latitude_bins.csv"
    df_bins.to_csv(bins_out, index=False)

    img_label = args.img_label or infer_img_label_from_stage6(stage6_csvs)

    plot_path = outdir / "stage7_occurrence_by_lat.png"
    plot_occurrence(df_bins, plot_path, img_label=img_label)

    print("[Stage 7] Complete")
    print(f"  Fragment table → {frag_out}")
    print(f"  Bin summary    → {bins_out}")
    print(f"  Plot           → {plot_path}")


if __name__ == "__main__":
    main()