#!/usr/bin/env python3
"""
05.graph_profiles.py

Stage 5 — Annotated profile visualization (dy-ordered, latitude-labeled)

CORE REQUIREMENT:
- dy is ALWAYS used internally for ordering and ΔBrightness computation.
- The curve is plotted in dy-space (so peak geometry is identical to dy plots),
  BUT the x-axis is labeled in physical planetocentric latitude (°) if available.
- dy is never shown to the user unless latitude data is missing/insufficient.
- No peak detection or classification happens here.

Why this works:
- Peak finding wants a consistent 1D profile coordinate (dy).
- Reviewers want physical meaning on the axis (latitude).
- We keep dy as the coordinate and relabel the axis using the dy→lat mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter



REQUIRED_COLS = {"fragment_id", "dy", "pixel_value"}

LAT_COL_ALIASES = [
    "PlanetocentricLatitude",
    "planetocentric_lat",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ΔBrightness profiles (dy-space) with latitude-labeled x-axis."
    )
    p.add_argument(
        "--csv",
        default="",
        help="Optional: process a single *_RECTIFIED_PERP_SAMPLES.csv",
    )
    p.add_argument(
        "--indir",
        default="",
        help="Stage 4 directory containing *_RECTIFIED_PERP_SAMPLES.csv files",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Stage 5 output directory (stage_05_graphs)",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Process all *_RECTIFIED_PERP_SAMPLES.csv files in --indir",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Savitzky–Golay smoothing window (odd, >=5). Default preserves detached haze features.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI for saved figures.",
    )
    p.add_argument(
        "--no-allfrags",
        action="store_true",
        help="Disable the ALL-FRAGMENTS overlay plot (per-fragment plots still generated).",
    )
    return p.parse_args()



def smooth_profile(y: np.ndarray, window: int) -> np.ndarray:
    """
    Savitzky–Golay smoothing that preserves secondary extrema.
    window must be odd and >= 5 to be meaningful.
    """
    if window < 5 or window % 2 == 0 or window >= y.size:
        return y

    return savgol_filter(y, window_length=window, polyorder=2)



def find_latitude_column(df: pd.DataFrame) -> str | None:
    for c in LAT_COL_ALIASES:
        if c in df.columns:
            return c
    return None


def build_lat_labeler(dy: np.ndarray, lat: np.ndarray):
    """
    Returns a function f(dy_tick_positions)->lat_values for tick relabeling.
    Requires at least 3 valid latitude points to be considered safe.
    """
    lat = lat.astype(float)
    dy = dy.astype(float)

    good = np.isfinite(lat) & np.isfinite(dy)
    if good.sum() < 3:
        return None

    dy_g = dy[good]
    lat_g = lat[good]

    # dy should already be sorted; ensure monotonic for interpolation safety
    order = np.argsort(dy_g)
    dy_g = dy_g[order]
    lat_g = lat_g[order]

    def f(dy_ticks: np.ndarray) -> np.ndarray:
        return np.interp(dy_ticks, dy_g, lat_g)

    return f


def apply_latitude_ticks(ax, dy_for_axis: np.ndarray, lat_for_axis: np.ndarray):
    """
    Label x-axis to explicitly show:
    - NaN (no SPICE) region
    - First valid SPICE latitude
    - Subsequent latitude values sparsely
    """

    lat = lat_for_axis.astype(float)
    dy = dy_for_axis.astype(float)

    valid = np.isfinite(lat)
    if valid.sum() < 3:
        return False

    # --- Identify first SPICE-valid sample ---
    first_idx = np.argmax(valid)
    first_dy = dy[first_idx]
    first_lat = lat[first_idx]

    # --- Draw vertical line where SPICE begins ---
    ax.axvline(
        first_dy,
        linestyle="--",
        color="gray",
        alpha=0.7,
        label="First valid SPICE intercept"
    )

    # --- Build sparse tick set ---
    ticks = []
    labels = []

    # Left-side: show NaN region explicitly
    left_dy = dy[:first_idx]
    if left_dy.size > 0:
        ticks.append(left_dy[len(left_dy) // 2])
        labels.append("NaN")

    # First SPICE point
    ticks.append(first_dy)
    labels.append(f"{first_lat:.1f}")

    # Subsequent SPICE ticks (sparse)
    post_dy = dy[first_idx:][valid[first_idx:]]
    post_lat = lat[first_idx:][valid[first_idx:]]

    step = max(1, len(post_dy) // 4)
    for d, l in zip(post_dy[::step], post_lat[::step]):
        ticks.append(d)
        labels.append(f"{l:.1f}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel(
        "Planetocentric Latitude (°) — NaN indicates no SPICE geometry"
    )

    ax.legend(loc="best", fontsize=9)


    return True



def iter_perp_csvs(indir: Path):
    return sorted(indir.glob("*_RECTIFIED_PERP_SAMPLES.csv"))


def main():
    args = parse_args()
    outroot = Path(args.outdir).expanduser().resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    csv_paths = []

    if args.csv:
        csv_paths = [Path(args.csv).expanduser().resolve()]
    else:
        if not args.indir:
            raise RuntimeError("Provide either --csv <file> or --indir <dir> (use --batch for all).")
        indir = Path(args.indir).expanduser().resolve()
        if not indir.exists():
            raise FileNotFoundError(f"indir not found: {indir}")

        cands = iter_perp_csvs(indir)
        if not cands:
            raise FileNotFoundError(f"No *_RECTIFIED_PERP_SAMPLES.csv found in {indir}")

        if args.batch:
            csv_paths = cands
        else:
            csv_paths = [cands[0]]  # quick smoke test

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing required columns: {sorted(missing)}")

        # numeric coercion
        df["pixel_value"] = pd.to_numeric(df["pixel_value"], errors="coerce")
        df["dy"] = pd.to_numeric(df["dy"], errors="coerce")
        df = df.dropna(subset=["pixel_value", "dy"])

        lat_col = find_latitude_column(df)
        have_lat_col = lat_col is not None

        stem = csv_path.name.replace("_RECTIFIED_PERP_SAMPLES.csv", "")
        outdir = outroot / stem
        outdir.mkdir(parents=True, exist_ok=True)

        # For ALL-FRAGS overlay (dy-space only; latitude labeling is not globally meaningful)
        allfrags_raw_lines = []
        allfrags_smooth_lines = []

        for frag_id in sorted(df["fragment_id"].unique()):
            frag = df[df["fragment_id"] == frag_id].copy()

            # CRITICAL: dy ordering defines the 1D profile coordinate
            frag = frag.sort_values("dy")

            dy = frag["dy"].to_numpy(float)
            brightness = frag["pixel_value"].to_numpy(float)

            # latitude exists per sample point (same rows as dy/brightness)
            lat = None
            if have_lat_col:
                frag[lat_col] = pd.to_numeric(frag[lat_col], errors="coerce")
                lat = frag[lat_col].to_numpy(float)

            # ΔBrightness (RAW) in dy-space
            d_raw = np.diff(brightness)
            dy_raw = dy[1:]

            # ΔBrightness (SMOOTH) in dy-space
            b_sm = smooth_profile(brightness, args.smooth)
            d_sm = np.diff(b_sm)
            dy_sm = dy[1:]

            # Collect for combined overlays
            allfrags_raw_lines.append((dy_raw, d_raw))
            allfrags_smooth_lines.append((dy_sm, d_sm))

            # -------- RAW plot (dy-space curve; latitude-labeled axis if possible) --------
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dy_raw, d_raw, marker="o")
            ax.set_title(f"{stem} — fragment {frag_id:04d} — RAW")
            ax.set_ylabel("ΔBrightness")

            used_lat_labels = False
            if lat is not None:
                # align latitude to dy_raw (drop first to match diff)
                used_lat_labels = apply_latitude_ticks(ax, dy_raw, lat[1:])

            if not used_lat_labels:
                ax.set_xlabel("dy (pixels)")
                print(
                    f"[WARN] fragment {frag_id:04d}: latitude labels unavailable/insufficient; "
                    f"falling back to dy axis.",
                    file=sys.stderr,
                )

            fig.tight_layout()
            fig.savefig(outdir / f"{stem}_frag{frag_id:04d}_RAW_delta.png", dpi=args.dpi)
            plt.close(fig)

            # -------- SMOOTH plot --------
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dy_sm, d_sm, marker="o")
            ax.set_title(f"{stem} — fragment {frag_id:04d} — SMOOTH")
            ax.set_ylabel("ΔBrightness")

            used_lat_labels = False
            if lat is not None:
                used_lat_labels = apply_latitude_ticks(ax, dy_sm, lat[1:])

            if not used_lat_labels:
                ax.set_xlabel("dy (pixels)")
                print(
                    f"[WARN] fragment {frag_id:04d}: latitude labels unavailable/insufficient; "
                    f"falling back to dy axis.",
                    file=sys.stderr,
                )

            fig.tight_layout()
            fig.savefig(outdir / f"{stem}_frag{frag_id:04d}_SMOOTH_delta.png", dpi=args.dpi)
            plt.close(fig)

        # -------- ALL FRAGMENTS overlay (kept in dy-space on purpose) --------
        if not args.no_allfrags:
            fig, ax = plt.subplots(figsize=(12, 7))
            for (x, y) in allfrags_raw_lines:
                ax.plot(x, y)
            ax.set_title(f"{stem} — ALL FRAGMENTS — RAW")
            ax.set_xlabel("dy (pixels)")
            ax.set_ylabel("ΔBrightness")
            fig.tight_layout()
            fig.savefig(outdir / f"{stem}_ALLFRAGS_RAW_delta_vs_dy.png", dpi=args.dpi)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 7))
            for (x, y) in allfrags_smooth_lines:
                ax.plot(x, y)
            ax.set_title(f"{stem} — ALL FRAGMENTS — SMOOTH")
            ax.set_xlabel("dy (pixels)")
            ax.set_ylabel("ΔBrightness")
            fig.tight_layout()
            fig.savefig(outdir / f"{stem}_ALLFRAGS_SMOOTH_delta_vs_dy.png", dpi=args.dpi)
            plt.close(fig)

            print(
                "[INFO] ALL-FRAGMENTS plots stay in dy-space because a single latitude axis is not "
                "well-defined across multiple fragments (different lat ranges).",
                file=sys.stderr,
            )

        print(f"[OK] Stage 05 plots written to: {outdir}")
        if have_lat_col:
            print("[INFO] Per-fragment x-axis is dy-space with latitude-labeled ticks (°).")
        else:
            print("[WARN] PlanetocentricLatitude column not found. Using dy for x-axis.")


if __name__ == "__main__":
    main()
