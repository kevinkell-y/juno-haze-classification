"""
Stage 04b: Slant Distance Detectability

Reads Stage 4 *_RECTIFIED_PERP_SAMPLES.csv outputs, computes fragment-level
slant-distance / altitude detectability metrics, and writes:

1) stage_04b_fragment_detectability.csv
2) stage_04b_img_summary.csv
3) diagnostic figures (optional)

This stage does NOT alter detection itself.
It creates a modular geometry-based interpretability layer for downstream stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Constants locked to Gerald's geometry chain
# ---------------------------------------------------------------------

JUPITER_RADIUS_KM = 70000.0
JUNOCAM_FOV_DEG = 58.0
JUNOCAM_PIXELS_X = 1600.0
PIXEL_SCALE_RAD_PER_PX = np.deg2rad(JUNOCAM_FOV_DEG / JUNOCAM_PIXELS_X)

GOOD_CASE_SLANT_KM = 24000.0
GOOD_CASE_DOUBLE_PEAK_PX = 5.0
NOMINAL_DETACHMENT_KM = 76.0

MIN_RESOLVABLE_DOUBLE_PEAK_PX = 2.5

DEFAULT_SLANT_COL = "SlantDistance"
DEFAULT_ALT_COL = "SpacecraftAltitude"
DEFAULT_LAT_COL = "PlanetocentricLatitude"


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def finite_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def geometry_slant_from_altitude(
    alt_km: float,
    radius_km: float = JUPITER_RADIUS_KM
) -> float:
    """
    Spherical approximation from Gerald's sketch:
        d^2 = 2ra + a^2
    """
    if not np.isfinite(alt_km) or alt_km < 0:
        return np.nan
    return float(np.sqrt(2.0 * radius_km * alt_km + alt_km ** 2))


def km_per_pixel_at_limb(
    slant_km: float,
    pixel_scale_rad_per_px: float = PIXEL_SCALE_RAD_PER_PX
) -> float:
    if not np.isfinite(slant_km) or slant_km <= 0:
        return np.nan
    return float(slant_km * pixel_scale_rad_per_px)


def pixels_per_nominal_detachment(
    slant_km: float,
    nominal_detachment_km: float = NOMINAL_DETACHMENT_KM,
    pixel_scale_rad_per_px: float = PIXEL_SCALE_RAD_PER_PX,
) -> float:
    km_px = km_per_pixel_at_limb(slant_km, pixel_scale_rad_per_px)
    if not np.isfinite(km_px) or km_px <= 0:
        return np.nan
    return float(nominal_detachment_km / km_px)


def detectability_decision(
    pixels_per_detachment: float,
    threshold_px: float = MIN_RESOLVABLE_DOUBLE_PEAK_PX
) -> tuple[bool, str]:
    if not np.isfinite(pixels_per_detachment):
        return False, "nonfinite_geometry"
    if pixels_per_detachment < threshold_px:
        return False, "insufficient_resolving_power"
    return True, "detectable"


# ---------------------------------------------------------------------
# Fragment summarization
# ---------------------------------------------------------------------

def first_finite(values: pd.Series) -> float:
    vals = finite_series(values)
    if len(vals) == 0:
        return np.nan
    return float(vals.iloc[0])


def median_finite(values: pd.Series) -> float:
    vals = finite_series(values)
    if len(vals) == 0:
        return np.nan
    return float(vals.median())


def infer_img_name(path: Path) -> str:
    name = path.stem
    suffix = "_RECTIFIED_PERP_SAMPLES"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


def summarize_fragment(
    frag_df: pd.DataFrame,
    img_name: str,
    slant_col: str,
    alt_col: str,
    lat_col: Optional[str],
) -> Dict:
    first_slant = first_finite(frag_df[slant_col]) if slant_col in frag_df.columns else np.nan
    median_slant = median_finite(frag_df[slant_col]) if slant_col in frag_df.columns else np.nan

    first_alt = first_finite(frag_df[alt_col]) if alt_col in frag_df.columns else np.nan
    median_alt = median_finite(frag_df[alt_col]) if alt_col in frag_df.columns else np.nan

    geom_slant = geometry_slant_from_altitude(first_alt) if np.isfinite(first_alt) else np.nan
    slant_delta = abs(first_slant - geom_slant) if np.isfinite(first_slant) and np.isfinite(geom_slant) else np.nan

    # Representative slant: first finite SPICE slant, else geometric estimate, else median slant
    if np.isfinite(first_slant):
        rep_slant = first_slant
    elif np.isfinite(geom_slant):
        rep_slant = geom_slant
    else:
        rep_slant = median_slant

    km_px = km_per_pixel_at_limb(rep_slant)
    px_per_det = pixels_per_nominal_detachment(rep_slant)
    detectable, reason = detectability_decision(px_per_det)

    row = {
        "img_name": img_name,
        "framelet_name": img_name,
        "fragment_id": int(frag_df["fragment_id"].iloc[0]),
        "n_samples": int(len(frag_df)),

        "first_valid_slant_distance_km": first_slant,
        "median_slant_distance_km": median_slant,
        "first_valid_spacecraft_altitude_km": first_alt,
        "median_spacecraft_altitude_km": median_alt,

        "geom_slant_from_altitude_km": geom_slant,
        "slant_geometry_delta_km": slant_delta,

        "pixel_scale_rad_per_px": PIXEL_SCALE_RAD_PER_PX,
        "km_per_pixel_at_limb": km_px,

        "nominal_detachment_km": NOMINAL_DETACHMENT_KM,
        "pixels_per_detachment": px_per_det,
        "min_resolvable_double_peak_px": MIN_RESOLVABLE_DOUBLE_PEAK_PX,

        "detectability_flag": bool(detectable),
        "detectability_reason": reason,
    }

    if lat_col and lat_col in frag_df.columns:
        row["first_valid_planetocentric_latitude_deg"] = first_finite(frag_df[lat_col])
        row["median_planetocentric_latitude_deg"] = median_finite(frag_df[lat_col])

    return row


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def compute_binned_median(
    x: pd.Series,
    y: pd.Series,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute median y within fixed-width x bins.
    Returns bin centers and median values for non-empty bins.
    """
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()

    if len(x) == 0:
        return np.array([]), np.array([])

    x_min = np.floor(x.min() / bin_width) * bin_width
    x_max = np.ceil(x.max() / bin_width) * bin_width
    edges = np.arange(x_min, x_max + bin_width, bin_width)

    if len(edges) < 2:
        return np.array([]), np.array([])

    centers = []
    medians = []

    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (x >= left) & (x < right)
        if np.any(in_bin):
            centers.append((left + right) / 2.0)
            medians.append(np.median(y[in_bin]))

    return np.array(centers), np.array(medians)

def plot_detectability_vs_lat(df: pd.DataFrame, out_png: Path) -> None:
    if "median_planetocentric_latitude_deg" not in df.columns:
        return

    plot_df = df.dropna(
        subset=["median_planetocentric_latitude_deg", "pixels_per_detachment"]
    ).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    detectable = plot_df["detectability_flag"] == True

    plt.scatter(
        plot_df.loc[~detectable, "median_planetocentric_latitude_deg"],
        plot_df.loc[~detectable, "pixels_per_detachment"],
        s=18,
        alpha=0.5,
        label="Non-detectable fragments",
    )
    plt.scatter(
        plot_df.loc[detectable, "median_planetocentric_latitude_deg"],
        plot_df.loc[detectable, "pixels_per_detachment"],
        s=18,
        alpha=0.5,
        label="Detectable fragments",
    )

    # Binned median trend
    bin_centers, bin_medians = compute_binned_median(
        plot_df["median_planetocentric_latitude_deg"],
        plot_df["pixels_per_detachment"],
        bin_width=2.0,
    )
    if len(bin_centers) > 0:
        plt.plot(
            bin_centers,
            bin_medians,
            linewidth=2.0,
            label="Binned median (2° bins)",
        )

    plt.axhline(
        MIN_RESOLVABLE_DOUBLE_PEAK_PX,
        linestyle="--",
        linewidth=1.5,
        label="2.5 px threshold",
    )
    plt.xlabel("Planetocentric latitude (deg)")
    plt.ylabel("Pixels per nominal detached-haze separation")
    plt.title("Stage 4b Detectability vs Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_km_per_pixel_vs_slant(df: pd.DataFrame, out_png: Path) -> None:
    plot_df = df.dropna(
        subset=["first_valid_slant_distance_km", "km_per_pixel_at_limb"]
    ).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    detectable = plot_df["detectability_flag"] == True

    plt.scatter(
        plot_df.loc[~detectable, "first_valid_slant_distance_km"],
        plot_df.loc[~detectable, "km_per_pixel_at_limb"],
        s=18,
        alpha=0.5,
        label="Non-detectable fragments",
    )
    plt.scatter(
        plot_df.loc[detectable, "first_valid_slant_distance_km"],
        plot_df.loc[detectable, "km_per_pixel_at_limb"],
        s=18,
        alpha=0.5,
        label="Detectable fragments",
    )
    
    # Binned median trend
    bin_centers, bin_medians = compute_binned_median(
        plot_df["first_valid_slant_distance_km"],
        plot_df["km_per_pixel_at_limb"],
        bin_width=1000.0,
    )
    if len(bin_centers) > 0:
        plt.plot(
            bin_centers,
            bin_medians,
            linewidth=2.0,
            label="Binned median (1000 km bins)",
        )

    plt.xlabel("Representative slant distance to limb (km)")
    plt.ylabel("Limb resolution (km/pixel)")
    plt.title("Stage 4b Limb Resolution vs Slant Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------

def process_stage4_csv(csv_path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "fragment_id" not in df.columns:
        raise ValueError(f"{csv_path} missing required column: fragment_id")

    img_name = infer_img_name(csv_path)

    rows: List[Dict] = []
    for _, frag_df in df.groupby("fragment_id", sort=True):
        rows.append(
            summarize_fragment(
                frag_df=frag_df,
                img_name=img_name,
                slant_col=args.slant_col,
                alt_col=args.alt_col,
                lat_col=args.lat_col,
            )
        )

    frag_out = pd.DataFrame(rows)

    img_summary = pd.DataFrame([{
        "img_name": img_name,
        "framelet_name": img_name,
        "n_fragments": int(len(frag_out)),
        "n_detectable_fragments": int(frag_out["detectability_flag"].sum()),
        "fraction_detectable": float(frag_out["detectability_flag"].mean()) if len(frag_out) else np.nan,
        "median_slant_distance_km": float(frag_out["median_slant_distance_km"].median()) if len(frag_out) else np.nan,
        "median_spacecraft_altitude_km": float(frag_out["median_spacecraft_altitude_km"].median()) if len(frag_out) else np.nan,
        "median_km_per_pixel_at_limb": float(frag_out["km_per_pixel_at_limb"].median()) if len(frag_out) else np.nan,
        "median_pixels_per_detachment": float(frag_out["pixels_per_detachment"].median()) if len(frag_out) else np.nan,
    }])

    return frag_out, img_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4b: Slant-distance detectability screening.")
    p.add_argument("--indir", required=True, help="Directory containing Stage 4 *_RECTIFIED_PERP_SAMPLES.csv files")
    p.add_argument("--outdir", required=True, help="Output directory for Stage 4b products")
    p.add_argument("--slant-col", default=DEFAULT_SLANT_COL)
    p.add_argument("--alt-col", default=DEFAULT_ALT_COL)
    p.add_argument("--lat-col", default=DEFAULT_LAT_COL)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(indir.glob("*_RECTIFIED_PERP_SAMPLES.csv"))
    if not csvs:
        raise FileNotFoundError(f"No Stage 4 CSVs found in {indir}")

    all_fragments = []
    all_imgs = []

    for csv_path in csvs:
        frag_df, img_df = process_stage4_csv(csv_path, args)
        all_fragments.append(frag_df)
        all_imgs.append(img_df)
        print(f"[Stage 4b] Processed {csv_path.name}: {len(frag_df)} fragments")

    frag_all = pd.concat(all_fragments, ignore_index=True)
    img_all = pd.concat(all_imgs, ignore_index=True)

    frag_csv = outdir / "stage_04b_fragment_detectability.csv"
    img_csv = outdir / "stage_04b_img_summary.csv"

    frag_all.to_csv(frag_csv, index=False)
    img_all.to_csv(img_csv, index=False)

    print(f"[Stage 4b] Wrote {frag_csv}")
    print(f"[Stage 4b] Wrote {img_csv}")

    plot_detectability_vs_lat(frag_all, outdir / "stage_04b_detectability_vs_lat.png")
    plot_km_per_pixel_vs_slant(frag_all, outdir / "stage_04b_km_per_pixel_vs_slant.png")
    print(f"[Stage 4b] Wrote diagnostic figures to {outdir}")

if __name__ == "__main__":
    main()