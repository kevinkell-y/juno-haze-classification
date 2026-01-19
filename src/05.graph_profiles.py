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
        required=True,
        help="*_RECTIFIED_PERP_SAMPLES.csv (should contain SPICE columns from Stage 04)",
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
    Plot is in dy-space, but the axis is labeled in latitude.
    """
    lat_labeler = build_lat_labeler(dy_for_axis, lat_for_axis)
    if lat_labeler is None:
        return False  # not enough valid lat to label safely

    # Choose ~6 ticks in dy-space across the visible span
    lo = float(np.nanmin(dy_for_axis))
    hi = float(np.nanmax(dy_for_axis))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return False

    ticks = np.linspace(lo, hi, 6)
    lat_vals = lat_labeler(ticks)

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.1f}" if np.isfinite(v) else "" for v in lat_vals])
    ax.set_xlabel("Planetocentric Latitude (°)")
    return True


def main():
    args = parse_args()
    csv_path = Path(args.csv).resolve()

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
    outdir = csv_path.parent / "stage_05_graphs" / stem
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
