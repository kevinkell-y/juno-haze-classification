#!/usr/bin/env python3
"""
Stage 5 — Aggregate Stage 6 peak classifications across fragments/framelets,
and evaluate detached haze occurrence as a function of latitude using the
first-available SPICE latitude per fragment.

Inputs:
  - One or more *_STAGE6.csv files produced by src/06.peakfinder.py

Outputs (in --outdir):
  - stage5_fragment_table.csv  (one row per fragment)
  - stage5_latitude_bins.csv   (binned summary)
  - stage5_occurrence_by_lat.png
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage6-root",
        type=str,
        default="data/cub/stage_06_peaks",
        help="Root directory containing Stage 6 outputs (framelet folders).",
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
        default="data/analysis/stage_05",
        help="Output directory for Stage 5 results.",
    )
    ap.add_argument(
        "--bin-width-deg",
        type=float,
        default=1.0,
        help="Latitude bin width in degrees (using first-available SPICE latitude).",
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
        help="Optional label to stamp on the Stage 5 plot (top-left). If omitted, inferred from Stage 6 filenames.",
    )
    return ap.parse_args()


@dataclass
class FragmentRecord:
    framelet: str
    fragment_id: int
    n_samples: int
    first_spice_i: int | None
    first_spice_lat: float | None
    has_secondary: bool
    secondary_count: int


def infer_framelet_name(stage6_csv: Path) -> str:
    # Preferred: Stage 6 CSV filename is "<FRAMELET>_STAGE6.csv"
    m = re.match(r"(.+)_STAGE6\.csv$", stage6_csv.name)
    if m:
        return m.group(1)
    # Fallback: parent folder name
    return stage6_csv.parent.name

def infer_img_label_from_stage6(stage6_csvs: list[Path]) -> str:
    """
    Infer IMG ID from Stage 6 CSVs by stripping color and framelet suffixes.
    Example:
      JNCR_2018197_14C00024_V01_BLUE_0003 → JNCR_2018197_14C00024_V01
    """
    if not stage6_csvs:
        return "UNKNOWN_IMG"

    framelet = infer_framelet_name(stage6_csvs[0])

    # Remove color + framelet suffix
    m = re.match(r"(JNCR_[^_]+_[^_]+_V\d+)", framelet)
    if m:
        return m.group(1)

    return framelet


def first_finite_index(x: np.ndarray) -> int | None:
    finite = np.isfinite(x)
    if not finite.any():
        return None
    return int(np.argmax(finite))  # first True


def build_fragment_table(stage6_csvs: list[Path], lat_col: str, peak_col: str) -> pd.DataFrame:
    records: list[FragmentRecord] = []

    for csv_path in stage6_csvs:
        df = pd.read_csv(csv_path)

        if "fragment_id" not in df.columns:
            raise KeyError(f"{csv_path} missing required column 'fragment_id'")

        if lat_col not in df.columns:
            raise KeyError(f"{csv_path} missing latitude column '{lat_col}'")

        if peak_col not in df.columns:
            raise KeyError(f"{csv_path} missing peak label column '{peak_col}'")

        framelet = infer_framelet_name(csv_path)

        for frag_id in sorted(df["fragment_id"].unique()):
            frag = df[df["fragment_id"] == frag_id].reset_index(drop=True)

            lat = frag[lat_col].to_numpy(dtype=float)
            first_i = first_finite_index(lat)

            if first_i is None:
                first_lat = None
            else:
                first_lat = float(lat[first_i])

            peak_types = frag[peak_col].astype(str).to_numpy()
            sec_mask = peak_types == "SECONDARY"
            sec_count = int(sec_mask.sum())

            rec = FragmentRecord(
                framelet=framelet,
                fragment_id=int(frag_id),
                n_samples=int(len(frag)),
                first_spice_i=first_i,
                first_spice_lat=first_lat,
                has_secondary=bool(sec_count > 0),
                secondary_count=sec_count,
            )
            records.append(rec)

    out = pd.DataFrame([r.__dict__ for r in records])
    return out


def bin_latitudes(df_frag: pd.DataFrame, bin_width: float, min_n: int) -> pd.DataFrame:
    work = df_frag.copy()

    # Drop fragments without any SPICE latitude at all (can’t place on x-axis)
    work = work[np.isfinite(work["first_spice_lat"].astype(float, errors="ignore"))].copy()
    work["first_spice_lat"] = work["first_spice_lat"].astype(float)

    if len(work) == 0:
        raise RuntimeError("No fragments had a finite first_spice_lat; cannot bin.")

    # Bin edges
    lat_min = math.floor(work["first_spice_lat"].min() / bin_width) * bin_width
    lat_max = math.ceil(work["first_spice_lat"].max() / bin_width) * bin_width
    edges = np.arange(lat_min, lat_max + bin_width, bin_width)

    work["lat_bin"] = pd.cut(work["first_spice_lat"], bins=edges, include_lowest=True)

    grouped = work.groupby("lat_bin", observed=True)
    summary = grouped.agg(
        n_fragments=("has_secondary", "size"),
        n_with_secondary=("has_secondary", "sum"),
        frac_with_secondary=("has_secondary", "mean"),
        lat_center=("first_spice_lat", "mean"),
        lat_min=("first_spice_lat", "min"),
        lat_max=("first_spice_lat", "max"),
    ).reset_index()

    summary["low_n_flag"] = summary["n_fragments"] < int(min_n)

    # Approx 95% CI for a proportion (normal approx); OK as a quick diagnostic
    p = summary["frac_with_secondary"].to_numpy(float)
    n = summary["n_fragments"].to_numpy(float)
    se = np.sqrt(np.clip(p * (1 - p) / np.maximum(n, 1.0), 0, np.inf))
    summary["frac_ci95_lo"] = np.clip(p - 1.96 * se, 0, 1)
    summary["frac_ci95_hi"] = np.clip(p + 1.96 * se, 0, 1)

    return summary


def plot_occurrence(summary: pd.DataFrame, out_png: Path, img_label: str | None = None) -> None:
    # Simple plot: fraction vs mean latitude per bin, plus CI whiskers
    x = summary["lat_center"].to_numpy(float)
    y = summary["frac_with_secondary"].to_numpy(float)
    ylo = summary["frac_ci95_lo"].to_numpy(float)
    yhi = summary["frac_ci95_hi"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o")
    ax.vlines(x, ylo, yhi)

    ax.set_xlabel("First-available SPICE Planetocentric Latitude (deg)")
    ax.set_ylabel("Fraction of fragments with SECONDARY peak")
    ax.set_title("Detached haze occurrence (proxy-binned by first SPICE latitude)")
    
    if img_label:
        ax.text(
            0.02, 0.98,
            f"JunoCam IMG: {img_label}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="left",
            alpha=0.7
        )

    # Mark low-N bins
    low_n = summary["low_n_flag"].to_numpy(bool)
    if low_n.any():
        ax.scatter(x[low_n], y[low_n], marker="x", s=120)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    stage6_root = Path(args.stage6_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stage6_csvs = sorted(stage6_root.glob(args.glob))
    if len(stage6_csvs) == 0:
        raise FileNotFoundError(f"No Stage 6 CSVs found under {stage6_root} with glob '{args.glob}'")

    df_frag = build_fragment_table(stage6_csvs, lat_col=args.lat_col, peak_col=args.peak_col)
    frag_out = outdir / "stage5_fragment_table.csv"
    df_frag.to_csv(frag_out, index=False)

    df_bins = bin_latitudes(df_frag, bin_width=float(args.bin_width_deg), min_n=int(args.min_fragments_per_bin))
    bins_out = outdir / "stage5_latitude_bins.csv"
    df_bins.to_csv(bins_out, index=False)

    img_label = args.img_label or infer_img_label_from_stage6(stage6_csvs)

    plot_path = outdir / "stage5_occurrence_by_lat.png"
    plot_occurrence(df_bins, plot_path, img_label=img_label)


    print("[Stage 5] Complete")
    print(f"  Fragment table → {frag_out}")
    print(f"  Bin summary    → {bins_out}")
    print(f"  Plot           → {plot_path}")


if __name__ == "__main__":
    main()
