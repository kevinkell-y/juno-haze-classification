#!/usr/bin/env python3
"""
05.graph_perp_profiles.py

Stage 4 (part 1): For each limb fragment (perpendicular red-dot sample line),
generate Wen-style delta-brightness curves and save plots.

KEY CHANGE (per your request):
- For every fragment, output TWO plots:
    (A) RAW:        ΔB computed from raw pixel_value
    (B) SMOOTHED:   smooth pixel_value FIRST, then compute ΔB

This makes peaks more pronounced and reduces false "haze" flags from pixel noise.

Inputs:
  *_RECTIFIED_PERP_SAMPLES.csv
    expected columns from Stage 3b:
      fragment_id, sample_id, rx, ry, dy, orig_x, orig_y, pixel_value

Outputs (per fragment):
  <stem>_frag####_RAW_delta_vs_<xaxis>.png
  <stem>_frag####_SMOOTH_delta_vs_<xaxis>.png

Outputs (combined overlays, optional):
  <stem>_ALLFRAGS_RAW_delta_vs_<xaxis>.png
  <stem>_ALLFRAGS_SMOOTH_delta_vs_<xaxis>.png

X-axis options:
  - dy        (default) signed offset along perpendicular centered at limb midpoint
  - ry        rectified y pixel coordinate
  - latitude  requires a column named 'planetocentric_lat' in the input CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLS = {
    "fragment_id", "sample_id", "rx", "ry", "dy", "orig_x", "orig_y", "pixel_value"
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate RAW and SMOOTHED Wen-style ΔBrightness plots per limb fragment."
    )
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to *_RECTIFIED_PERP_SAMPLES.csv",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory. Default: <csv_dir>/stage_05_graphs/<stem>/",
    )
    p.add_argument(
        "--xaxis",
        type=str,
        choices=["dy", "ry", "latitude"],
        default="dy",
        help="What to plot on the X axis. Default: dy.",
    )
    p.add_argument(
        "--abs",
        action="store_true",
        help="Plot absolute delta |ΔB| instead of signed ΔB (applies to both RAW and SMOOTH).",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=7,
        help="Moving-average window applied to BRIGHTNESS before differencing (SMOOTH plots only). "
             "Odd int recommended. 0 disables SMOOTH output smoothing (still outputs both plots). Default: 7.",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=15,
        help="Skip fragments with fewer than this many sample points after cleaning.",
    )
    p.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip combined overlay plots (all fragments).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG DPI. Default: 200",
    )
    p.add_argument(
        "--dotsize",
        type=float,
        default=12,
        help="Scatter dot size. Default: 12",
    )
    return p.parse_args()


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y
    window = int(window)
    if window > y.size:
        return y

    # Pad with edge values to avoid zero-padding artifacts at the boundaries
    k = window // 2
    y_pad = np.pad(y, (k, k), mode="edge")

    kernel = np.ones(window, dtype=float) / float(window)

    # 'valid' returns length exactly len(y) after padding
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    return y_smooth


def choose_x(df: pd.DataFrame, xaxis: str) -> np.ndarray:
    if xaxis == "dy":
        return df["dy"].to_numpy(dtype=float)
    if xaxis == "ry":
        return df["ry"].to_numpy(dtype=float)
    if xaxis == "latitude":
        if "planetocentric_lat" not in df.columns:
            raise ValueError(
                "xaxis=latitude requested, but input CSV has no 'planetocentric_lat' column.\n"
                "Add that column in the SPICE-mapping stage, then rerun with --xaxis latitude."
            )
        return df["planetocentric_lat"].to_numpy(dtype=float)
    raise ValueError(f"Unknown xaxis: {xaxis}")


def stem_from_samples_csv(p: Path) -> str:
    name = p.name
    if name.endswith("_RECTIFIED_PERP_SAMPLES.csv"):
        return name.replace("_RECTIFIED_PERP_SAMPLES.csv", "")
    return p.stem


def delta_profile(x: np.ndarray, pv: np.ndarray, *, abs_delta: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ΔB = B[i+1] - B[i].

    IMPORTANT: For plotting, align ΔB to the "right" sample (x[i+1]).
    This avoids the 0.5-step visual shift when x is integer dy.
    Returns (x_aligned, delta) where len(x_aligned) == len(delta) == N-1.
    """
    delta = np.diff(pv)
    x_aligned = x[1:]  # right-aligned; alternative would be x[:-1] (left-aligned)
    if abs_delta:
        delta = np.abs(delta)
    return x_aligned, delta

def peak_align(x: np.ndarray, y: np.ndarray, *, search_window: tuple[int, int] | None = None):
    """
    Shift x so the primary peak of y is centered at x = 0.

    search_window: optional (xmin, xmax) to restrict peak finding
    """
    if search_window is not None:
        xmin, xmax = search_window
        mask = (x >= xmin) & (x <= xmax)
        if not np.any(mask):
            return x
        xw = x[mask]
        yw = y[mask]
        x0 = xw[np.argmax(yw)]
    else:
        x0 = x[np.argmax(y)]

    return x - x0


def save_fragment_plot(
    out_png: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_mid: np.ndarray,
    delta: np.ndarray,
    *,
    dotsize: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.scatter(x_mid, delta, s=dotsize)
    ax.plot(x_mid, delta)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    # Type cleanup
    df["fragment_id"] = df["fragment_id"].astype(int)
    df["sample_id"] = df["sample_id"].astype(int)
    df["dy"] = df["dy"].astype(int)
    df["ry"] = df["ry"].astype(int)
    df["pixel_value"] = pd.to_numeric(df["pixel_value"], errors="coerce")

    # Output directory
    stem = stem_from_samples_csv(csv_path)
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = csv_path.parent / "stage_05_graphs" / stem
    outdir.mkdir(parents=True, exist_ok=True)

    # Sort along the perpendicular (dy)
    df = df.sort_values(["fragment_id", "dy"]).reset_index(drop=True)

    frag_ids = sorted(df["fragment_id"].unique().tolist())

    ylab = "|ΔBrightness|" if args.abs else "ΔBrightness"

    # Combined overlay figures
    comb_raw_fig = comb_raw_ax = None
    comb_smooth_fig = comb_smooth_ax = None
    if not args.no_combined:
        comb_raw_fig = plt.figure(figsize=(10, 6))
        comb_raw_ax = comb_raw_fig.add_subplot(1, 1, 1)
        comb_raw_ax.set_title(f"{stem} — ΔBrightness profiles (all fragments) — RAW")
        comb_raw_ax.set_xlabel(args.xaxis)
        comb_raw_ax.set_ylabel(ylab)

        comb_smooth_fig = plt.figure(figsize=(10, 6))
        comb_smooth_ax = comb_smooth_fig.add_subplot(1, 1, 1)
        comb_smooth_ax.set_title(f"{stem} — ΔBrightness profiles (all fragments) — SMOOTH(B→Δ), PEAK-ALIGNED")
        comb_smooth_ax.set_xlabel(args.xaxis)
        comb_smooth_ax.set_ylabel(ylab)

    for frag_id in frag_ids:
        frag = df[df["fragment_id"] == frag_id].copy()
        frag = frag.dropna(subset=["pixel_value"])
        if len(frag) < args.min_points:
            continue

        x = choose_x(frag, args.xaxis)
        pv_raw = frag["pixel_value"].to_numpy(dtype=float)

        # --- RAW ΔB ---
        x_mid_raw, d_raw = delta_profile(x, pv_raw, abs_delta=args.abs)

        out_png_raw = outdir / f"{stem}_frag{frag_id:04d}_RAW_delta_vs_{args.xaxis}.png"
        save_fragment_plot(
            out_png_raw,
            title=f"{stem} — fragment {frag_id:04d} — ΔBrightness (RAW)",
            x_label=args.xaxis,
            y_label=ylab,
            x_mid=x_mid_raw,
            delta=d_raw,
            dotsize=args.dotsize,
            dpi=args.dpi,
        )

        if comb_raw_ax is not None:
            comb_raw_ax.plot(x_mid_raw, d_raw, linewidth=1)

        # --- SMOOTHED: smooth BRIGHTNESS first, then ΔB ---
        pv_smooth = pv_raw.copy()
        if args.smooth and args.smooth > 1:
            pv_smooth = moving_average(pv_smooth, args.smooth)

        x_mid_sm, d_sm = delta_profile(x, pv_smooth, abs_delta=args.abs)

        # PEAK-ALIGN for combined plot ONLY
        # NOTE:
        # Peak alignment is applied ONLY to smoothed profiles.
        # Raw profiles are intentionally NOT aligned because
        # noise dominates peak location pre-smoothing.
        x_mid_sm_aligned = peak_align(
            x_mid_sm,
            d_sm,
            search_window=(-30, 30),  # adjust if needed
        )


        out_png_sm = outdir / f"{stem}_frag{frag_id:04d}_SMOOTH_delta_vs_{args.xaxis}.png"
        save_fragment_plot(
            out_png_sm,
            title=f"{stem} — fragment {frag_id:04d} — ΔBrightness (SMOOTH B→Δ, window={args.smooth})",
            x_label=args.xaxis,
            y_label=ylab,
            x_mid=x_mid_sm,
            delta=d_sm,
            dotsize=args.dotsize,
            dpi=args.dpi,
        )

        if comb_smooth_ax is not None:
            comb_smooth_ax.plot(x_mid_sm_aligned, d_sm, linewidth=1)

    # Save combined overlays
    if comb_raw_fig is not None and comb_raw_ax is not None:
        comb_raw_fig.tight_layout()
        comb_raw_fig.savefig(outdir / f"{stem}_ALLFRAGS_RAW_delta_vs_{args.xaxis}.png", dpi=args.dpi)
        plt.close(comb_raw_fig)

    if comb_smooth_fig is not None and comb_smooth_ax is not None:
        comb_smooth_fig.tight_layout()
        comb_smooth_fig.savefig(outdir / f"{stem}_ALLFRAGS_SMOOTH_delta_vs_{args.xaxis}.png", dpi=args.dpi)
        plt.close(comb_smooth_fig)

    print(f"[OK] Wrote plots to: {outdir}")
    print("     Per-fragment outputs: ..._RAW_*.png and ..._SMOOTH_*.png")
    if not args.no_combined:
        print("     Combined overlays: ..._ALLFRAGS_RAW_*.png and ..._ALLFRAGS_SMOOTH_*.png")


if __name__ == "__main__":
    main()
