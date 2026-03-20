#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stage 6 Peak Finder (Juno haze)")
    p.add_argument("--indir", required=True, help="Directory containing *_RECTIFIED_PERP_SAMPLES.csv files")
    p.add_argument("--outdir", required=True, help="IMG-scoped output directory for Stage 6")
    p.add_argument("--detectability-csv", default=None,
                   help="Optional Stage 4b fragment detectability CSV to merge into Stage 6 outputs")
    p.add_argument("--smooth", type=int, default=5, help="Savitzky-Golay window for ΔBrightness (odd, >=3)")
    p.add_argument("--savgol-poly", type=int, default=2)
    p.add_argument("--max-secondary-sep-px", type=int, default=40,
                   help="Max index (pixel) separation between PRIMARY and SECONDARY")
    p.add_argument("--min-secondary-sep-px", type=int, default=3,
                   help="Minimum index separation between PRIMARY and SECONDARY")
    p.add_argument("--min-secondary-frac", type=float, default=0.12,
                   help="Minimum secondary peak strength as fraction of PRIMARY ΔBrightness")
    p.add_argument("--min-secondary-abs", type=float, default=400,
                   help="Absolute minimum ΔBrightness for secondary peak")
    p.add_argument("--limb-window-px", type=int, default=25,
                   help="Half-width window (px) around SPICE intercept to search for limb peak")
    p.add_argument("--min-primary-frac", type=float, default=0.3,
                   help="Minimum PRIMARY peak height as fraction of global ΔBrightness max")

    return p.parse_args()


# ============================================================
# Stage 4b detectability provenance helpers
# ============================================================

def load_detectability_table(path):
    """
    Load Stage 4b fragment-level detectability table once.

    Expected keys:
      - framelet_name
      - fragment_id

    This lets Stage 6 attach the exact fragment-level detectability
    metadata produced by Stage 4b, rather than recomputing anything.
    """
    if path is None:
        return None

    df = pd.read_csv(path)

    required = {"framelet_name", "fragment_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Detectability CSV missing required columns: {sorted(missing)}"
        )

    return df


def attach_detectability_metadata(frag_out, detectability_df, framelet_name, fragment_id):
    """
    Attach Stage 4b fragment-level detectability metadata onto the full
    Stage 6 fragment output DataFrame.

    The values are duplicated across all rows of the fragment on purpose,
    so each Stage 6 row is self-describing for provenance/audit purposes.
    """
    if detectability_df is None:
        # Preserve downstream schema even when no 4b file is supplied.
        frag_out["detectability_flag"] = np.nan
        frag_out["detectability_reason"] = "no_stage4b_supplied"
        return frag_out

    match = detectability_df[
        (detectability_df["framelet_name"] == framelet_name) &
        (detectability_df["fragment_id"] == fragment_id)
    ]

    if len(match) == 0:
        frag_out["detectability_flag"] = np.nan
        frag_out["detectability_reason"] = "missing_stage4b_record"
        return frag_out

    if len(match) > 1:
        raise RuntimeError(
            f"Multiple Stage 4b records found for {framelet_name} fragment {fragment_id}"
        )

    row = match.iloc[0]

    cols_to_copy = [
        "first_valid_slant_distance_km",
        "median_slant_distance_km",
        "first_valid_spacecraft_altitude_km",
        "median_spacecraft_altitude_km",
        "geom_slant_from_altitude_km",
        "slant_geometry_delta_km",
        "pixel_scale_rad_per_px",
        "km_per_pixel_at_limb",
        "nominal_detachment_km",
        "pixels_per_detachment",
        "min_resolvable_double_peak_px",
        "detectability_flag",
        "detectability_reason",
        "first_valid_planetocentric_latitude_deg",
        "median_planetocentric_latitude_deg",
    ]

    for col in cols_to_copy:
        if col in row.index:
            frag_out[col] = row[col]

    return frag_out


# ============================================================
# Fragment analysis
# ============================================================

def analyze_fragment(frag_df, args, framelet_name):
    """
    Analyze one limb fragment:
    - Compute ΔBrightness from pixel_value
    - Find PRIMARY limb peak (pre-SPICE)
    - Optionally find SECONDARY detached haze peak
    - Plot and annotate
    """

    # --------------------------------------------------------
    # Resolve brightness column
    # --------------------------------------------------------
    if "pixel_value" in frag_df.columns:
        pixel = frag_df["pixel_value"].to_numpy(dtype=float)
    elif "PixelValue" in frag_df.columns:
        pixel = frag_df["PixelValue"].to_numpy(dtype=float)
    else:
        raise RuntimeError(
            f"No pixel value column found in fragment. Columns:\n{frag_df.columns}"
        )

    dy = frag_df["dy"].to_numpy(dtype=float)
    lat = frag_df["PlanetocentricLatitude"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Stage 6 acceptance / rejection bookkeeping
    # --------------------------------------------------------
    accepted = True
    reject_stage = ""
    reject_reason = ""
    reject_detail = ""

    # --------------------------------------------------------
    # ΔBrightness (derivative along dy)
    # --------------------------------------------------------
    delta = np.diff(pixel)

    # Optional smoothing (ONCE)
    if args.smooth is not None and args.smooth >= 3:
        if args.smooth % 2 == 0:
            raise ValueError("--smooth must be odd")
        delta = savgol_filter(
            delta,
            window_length=args.smooth,
            polyorder=args.savgol_poly,
            mode="interp"
        )

    # Align arrays to delta index space
    dy_d = dy[1:]
    lat_d = lat[1:]

    # --------------------------------------------------------
    # First SPICE intercept (index in delta-space)
    # --------------------------------------------------------
    finite_lat_idx = np.where(np.isfinite(lat_d))[0]
    if len(finite_lat_idx) > 0:
        first_spice_i = int(finite_lat_idx[0])
    else:
        first_spice_i = len(delta)  # no SPICE at all

    # --------------------------------------------------------
    # Find all ΔBrightness peaks
    # --------------------------------------------------------
    peaks, _ = find_peaks(delta)

    # --------------------------------------------------------
    # PRIMARY peak (limb-dominant rule)
    #   - Search within ±N pixels of first SPICE intercept
    #   - Choose strongest ΔBrightness peak in that window
    #   - Must dominate fragment signal
    # --------------------------------------------------------
    primary_i = None

    if len(peaks) > 0:
        # Define limb window
        lo = max(0, first_spice_i - args.limb_window_px)
        hi = min(len(delta) - 1, first_spice_i + args.limb_window_px)

        limb_candidates = [p for p in peaks if lo <= p <= hi]

        if limb_candidates:
            # Pick strongest peak in limb window
            primary_i = int(
                limb_candidates[
                    np.argmax(delta[limb_candidates])
                ]
            )

            # Dominance check vs global signal
            global_max = np.max(delta)
            if delta[primary_i] < args.min_primary_frac * global_max:
                primary_i = None

            # -------- SINGLE EXPLICIT GUARD --------
            if primary_i is not None and first_spice_i < len(delta):
                if abs(primary_i - first_spice_i) > args.limb_window_px:
                    primary_i = None

    # --------------------------------------------------------
    # SECONDARY peak (detached haze)
    #   Rules:
    #     - Must be pre-PRIMARY
    #     - Must satisfy brightness floor
    #     - Closest to limb (dy) wins
    # --------------------------------------------------------
    secondary_i = None
    best_score = None

    if primary_i is not None:
        for p in peaks:
            p = int(p)

            if p == primary_i:
                continue

            # Secondary must be inward of the limb
            if p > primary_i:
                continue

            # Min separation
            if abs(p - primary_i) < args.min_secondary_sep_px:
                continue

            # Max separation
            if abs(p - primary_i) > args.max_secondary_sep_px:
                continue

            # Must be pre-SPICE
            if p >= first_spice_i:
                continue

            # Secondary must be reasonably close to limb
            if (primary_i - p) > 0.6 * primary_i:
                continue

            # Absolute brightness floor (kills low-left noise)
            if delta[p] < args.min_secondary_abs:
                continue

            # Valley (no shoulders)
            lo = min(primary_i, p)
            hi = max(primary_i, p)

            valley = np.min(delta[lo:hi])
            if delta[p] - valley < 0.06 * delta[primary_i]:
                continue

            # Relative amplitude threshold
            if delta[p] < args.min_secondary_frac * delta[primary_i]:
                continue

            # Score strongest remaining candidate
            score = delta[p]
            if best_score is None or score > best_score:
                best_score = score
                secondary_i = p

    if accepted and secondary_i is None:
        accepted = True
        # NOTE: no_secondary_peak is NOT a rejection
        # This is a valid "no haze detected" outcome

    # --------------------------------------------------------
    # Annotate fragment CSV
    # --------------------------------------------------------
    frag_out = frag_df.copy()
    frag_out["stage6_peak_type"] = ""
    frag_out["accepted"] = accepted
    frag_out["reject_stage"] = reject_stage
    frag_out["reject_reason"] = reject_reason
    frag_out["reject_detail"] = reject_detail
    # NOTE:
    # Rejection fields apply at the fragment level and are
    # intentionally duplicated across all rows for provenance.

    if primary_i is not None:
        frag_out.iloc[
            primary_i + 1,
            frag_out.columns.get_loc("stage6_peak_type")
        ] = "PRIMARY"

    if secondary_i is not None:
        frag_out.iloc[
            secondary_i + 1,
            frag_out.columns.get_loc("stage6_peak_type")
        ] = "SECONDARY"

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(dy_d, delta, marker="o", lw=2)
    ax.set_ylabel("ΔBrightness")
    ax.set_title(
        f"{framelet_name} — fragment {int(frag_df['fragment_id'].iloc[0]):04d}"
    )

    if primary_i is not None:
        ax.scatter(
            dy_d[primary_i], delta[primary_i],
            marker="x", s=200, c="orange", label="PRIMARY"
        )

    if secondary_i is not None:
        ax.scatter(
            dy_d[secondary_i], delta[secondary_i],
            marker="x", s=200, c="green", label="SECONDARY (Detached Haze)"
        )

    if first_spice_i < len(delta):
        ax.axvline(
            dy_d[first_spice_i],
            ls="--", lw=2, c="gray",
            label="First SPICE Intercept"
        )

    # X-axis labeling with latitude
    nticks = 14
    ticks = np.linspace(0, len(dy_d) - 1, nticks, dtype=int)
    ax.set_xticks(dy_d[ticks])

    labels = []
    for i in ticks:
        if not np.isfinite(lat_d[i]):
            labels.append("NaN")
        else:
            labels.append(f"{lat_d[i]:.6f}")

    ax.set_xticklabels(labels, rotation=30)
    ax.set_xlabel("Planetocentric Latitude (deg) — NaN = no SPICE")

    ax.legend()
    fig.tight_layout()

    return frag_out, fig


# ============================================================
# CSV-level processing
# ============================================================

def process_directory(indir: Path, outdir: Path, args):
    csv_files = sorted(indir.glob("*_RECTIFIED_PERP_SAMPLES.csv"))

    if not csv_files:
        raise RuntimeError(f"No *_RECTIFIED_PERP_SAMPLES.csv files found in {indir}")

    detectability_df = load_detectability_table(args.detectability_csv)

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        framelet = csv_path.stem.replace("_RECTIFIED_PERP_SAMPLES", "")
        framelet_dir = outdir / framelet
        plot_dir = framelet_dir / "plots"

        plot_dir.mkdir(parents=True, exist_ok=True)

        augmented = []

        for frag_id in sorted(df["fragment_id"].unique()):
            frag_df = df[df["fragment_id"] == frag_id].reset_index(drop=True)

            frag_out, fig = analyze_fragment(frag_df, args, framelet)

            frag_out = attach_detectability_metadata(
                frag_out=frag_out,
                detectability_df=detectability_df,
                framelet_name=framelet,
                fragment_id=int(frag_id),
            )

            plot_path = plot_dir / f"{framelet}_fragment_{frag_id:04d}_peaks.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)

            augmented.append(frag_out)

        out_df = pd.concat(augmented, ignore_index=True)
        out_csv = framelet_dir / f"{framelet}_STAGE6.csv"
        out_df.to_csv(out_csv, index=False)

        print(f"[Stage 6] Completed {framelet}")
        print(f"  Plots → {plot_dir}")
        print(f"  CSV   → {out_csv}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()

    if not indir.exists():
        raise FileNotFoundError(f"Input directory not found: {indir}")

    outdir.mkdir(parents=True, exist_ok=True)

    process_directory(indir, outdir, args)


if __name__ == "__main__":
    main()