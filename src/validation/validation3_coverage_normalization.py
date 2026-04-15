#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pjdir", required=True)
    ap.add_argument("--outdir", default=None)
    return ap.parse_args()


def infer_bin_width(lat_centers: pd.Series) -> float:
    vals = np.sort(pd.to_numeric(lat_centers, errors="coerce").dropna().unique())
    if len(vals) < 2:
        return 1.0
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


def main():
    args = parse_args()
    pjdir = Path(args.pjdir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else (
        pjdir / "analysis" / "PJ14_stage_08_perijove_analysis" / "validation"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    v2_path = outdir / "validation2_detectability_by_latitude.csv"
    s8_path = pjdir / "analysis" / "PJ14_stage_08_perijove_analysis" / "stage8_latitude_bins.csv"

    v2 = pd.read_csv(v2_path, low_memory=False)
    s8 = pd.read_csv(s8_path, low_memory=False)

    # Normalize Stage 8 column names
    rename_map = {}
    if "n_fragments" in s8.columns:
        rename_map["n_fragments"] = "n_stage8_fragments"
    if "n_with_secondary" in s8.columns:
        rename_map["n_with_secondary"] = "n_stage8_secondary"
    if "frac_with_secondary" in s8.columns:
        rename_map["frac_with_secondary"] = "stage8_occurrence"
    s8 = s8.rename(columns=rename_map)

    required_v2 = {"lat_center", "n_total", "n_detectable", "frac_detectable"}
    required_s8 = {"lat_center", "n_stage8_fragments", "n_stage8_secondary", "stage8_occurrence"}

    missing_v2 = required_v2 - set(v2.columns)
    missing_s8 = required_s8 - set(s8.columns)

    if missing_v2:
        raise ValueError(f"Validation 2 file missing columns: {sorted(missing_v2)}")
    if missing_s8:
        raise ValueError(f"Stage 8 latitude file missing columns: {sorted(missing_s8)}")

    v2 = v2.copy()
    s8 = s8.copy()

    v2["lat_center"] = pd.to_numeric(v2["lat_center"], errors="coerce")
    s8["lat_center"] = pd.to_numeric(s8["lat_center"], errors="coerce")

    v2 = v2.sort_values("lat_center").reset_index(drop=True)
    s8 = s8.sort_values("lat_center").reset_index(drop=True)

    bin_width = infer_bin_width(s8["lat_center"])
    tol = bin_width / 2.0 + 1e-6

    merged = pd.merge_asof(
        v2,
        s8[["lat_center", "n_stage8_fragments", "n_stage8_secondary", "stage8_occurrence"]],
        on="lat_center",
        direction="nearest",
        tolerance=tol,
    )

    # Clean, physically valid fractions
    merged["retained_detectable_coverage"] = (
        merged["n_stage8_fragments"] / merged["n_detectable"].replace(0, np.nan)
    ).clip(lower=0, upper=1)

    merged["secondary_among_detectable"] = (
        merged["n_stage8_secondary"] / merged["n_detectable"].replace(0, np.nan)
    ).clip(lower=0, upper=1)

    merged["detectable_among_total"] = merged["frac_detectable"].clip(lower=0, upper=1)

    merged.to_csv(outdir / "validation3_coverage_normalization.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        merged["lat_center"],
        merged["stage8_occurrence"],
        marker="o",
        label="Stage 8 occurrence",
    )
    ax.plot(
        merged["lat_center"],
        merged["retained_detectable_coverage"],
        marker="o",
        label="Retained / detectable coverage",
    )
    ax.plot(
        merged["lat_center"],
        merged["detectable_among_total"],
        marker="o",
        label="Detectable / total fragments",
    )
    ax.set_xlabel("Planetocentric latitude bin center (deg)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation 3 — Occurrence vs retained coverage by latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "validation3_coverage_normalization.png", dpi=200)
    plt.close(fig)

    print(f"Wrote: {outdir / 'validation3_coverage_normalization.csv'}")
    print(f"Wrote: {outdir / 'validation3_coverage_normalization.png'}")


if __name__ == "__main__":
    main()