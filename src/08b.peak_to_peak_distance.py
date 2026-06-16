#!/usr/bin/env python3
"""
08b.peak_to_peak_distance.py

Post-Stage-8 physical characterization for any completed Perijove.

Answers:
    Across all detached-haze detections retained in Stage 8, what is the
    average PRIMARY–SECONDARY peak separation?

Headline measurement:
    peak_to_peak_rectified_dy_px = abs(dy_secondary - dy_primary)

    rectified_to_detector_scale = (2 * Stage3_radius) / Stage3_y_step

    peak_to_peak_detector_equiv_px =
        peak_to_peak_rectified_dy_px * rectified_to_detector_scale

    peak_to_peak_km =
        peak_to_peak_detector_equiv_px * km_per_pixel_at_limb

Flagship figure:
    stage08b_peak_separation_summary.png

This figure uses a real representative Stage-6 profile nearest the median
separation and pairs it with the full distribution across all retained fragments.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute measured PRIMARY–SECONDARY detached-haze peak separation after Stage 8."
    )

    p.add_argument("--pjdir", required=True, help="Perijove root directory, e.g. data/PJ14")
    p.add_argument("--stage6-glob", default="cub/*/stage_06_peaks/**/*_STAGE6.csv")
    p.add_argument("--stage8-csv", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--pj-label", default=None)

    p.add_argument(
        "--radius",
        type=float,
        default=12.0,
        help="Stage 3 rectification radius in original detector pixels.",
    )
    p.add_argument(
        "--y-step",
        type=float,
        default=96.0,
        help="Stage 3 rectified strip height in pixels.",
    )

    p.add_argument("--lat-bin-width-deg", type=float, default=2.0)
    p.add_argument("--min-fragments-per-bin", type=int, default=10)

    p.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Savitzky-Golay smoothing window for representative ΔBrightness profile.",
    )
    p.add_argument("--savgol-poly", type=int, default=2)

    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def scalar_float(value) -> float:
    try:
        x = pd.to_numeric(value, errors="coerce")
        return float(x) if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def spice_display(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "SPICE unavailable"


def infer_img_id_from_stage6_path(path: Path) -> str:
    parts = path.parts
    if "cub" in parts:
        i = parts.index("cub")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "UNKNOWN_IMG"


def infer_framelet_name(path: Path) -> str:
    m = re.match(r"(.+)_STAGE6\.csv$", path.name)
    return m.group(1) if m else path.stem


def resolve_stage8_csv(pjdir: Path, user_stage8_csv: str | None) -> Path:
    if user_stage8_csv:
        p = Path(user_stage8_csv).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    standard = pjdir / "analysis" / "stage_08_perijove_analysis" / "stage8_fragments_all.csv"
    if standard.exists():
        return standard

    candidates = sorted((pjdir / "analysis").glob("**/stage8_fragments_all.csv"))
    if candidates:
        return candidates[0]

    raw_candidates = sorted((pjdir / "analysis").glob("**/stage8_fragments_all_raw.csv"))
    if raw_candidates:
        print("[warn] stage8_fragments_all.csv not found; using raw fallback.")
        return raw_candidates[0]

    raise FileNotFoundError(f"Could not find stage8_fragments_all.csv under {pjdir / 'analysis'}")


def load_stage8_valid_keys(stage8_csv: Path) -> set[tuple[str, str, int]]:
    df = pd.read_csv(stage8_csv, low_memory=False)

    required = {"img_id", "framelet", "fragment_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{stage8_csv} missing required columns: {sorted(missing)}")

    keys: set[tuple[str, str, int]] = set()

    for _, row in df.iterrows():
        frag_id = scalar_float(row["fragment_id"])
        if not np.isfinite(frag_id):
            continue

        keys.add((str(row["img_id"]), str(row["framelet"]), int(frag_id)))

    return keys


# ============================================================
# Measurement
# ============================================================

def extract_fragment_peak_separation(
    frag: pd.DataFrame,
    *,
    img_id: str,
    framelet: str,
    fragment_id: int,
    stage6_path: Path,
    rectified_to_detector_scale: float,
) -> dict | None:
    required = {"stage6_peak_type", "dy", "km_per_pixel_at_limb"}
    if not required.issubset(frag.columns):
        return None

    peak_type = frag["stage6_peak_type"].astype(str)

    primary_rows = frag[peak_type == "PRIMARY"]
    secondary_rows = frag[peak_type == "SECONDARY"]

    if len(primary_rows) != 1 or len(secondary_rows) < 1:
        return None

    primary = primary_rows.iloc[0]
    secondary = secondary_rows.iloc[0]

    dy_primary = scalar_float(primary["dy"])
    dy_secondary = scalar_float(secondary["dy"])

    if not np.isfinite(dy_primary) or not np.isfinite(dy_secondary):
        return None

    km_px_vals = numeric(frag["km_per_pixel_at_limb"]).dropna()
    if len(km_px_vals) == 0:
        return None

    km_per_pixel_at_limb = float(km_px_vals.iloc[0])

    peak_to_peak_rectified_dy_px = abs(dy_secondary - dy_primary)
    peak_to_peak_detector_equiv_px = peak_to_peak_rectified_dy_px * rectified_to_detector_scale
    peak_to_peak_km = peak_to_peak_detector_equiv_px * km_per_pixel_at_limb

    orig_xy_px = np.nan
    dx_orig = np.nan
    dy_orig = np.nan

    if {"orig_x", "orig_y"}.issubset(frag.columns):
        ox_p = scalar_float(primary.get("orig_x"))
        oy_p = scalar_float(primary.get("orig_y"))
        ox_s = scalar_float(secondary.get("orig_x"))
        oy_s = scalar_float(secondary.get("orig_y"))

        if all(np.isfinite(v) for v in [ox_p, oy_p, ox_s, oy_s]):
            dx_orig = ox_s - ox_p
            dy_orig = oy_s - oy_p
            orig_xy_px = float(np.hypot(dx_orig, dy_orig))

    spice_radial_separation_km = np.nan
    if "LocalRadius" in frag.columns:
        r_primary = scalar_float(primary.get("LocalRadius"))
        r_secondary = scalar_float(secondary.get("LocalRadius"))
        if np.isfinite(r_primary) and np.isfinite(r_secondary):
            spice_radial_separation_km = r_secondary - r_primary

    primary_lat = scalar_float(primary.get("PlanetocentricLatitude"))
    secondary_lat = scalar_float(secondary.get("PlanetocentricLatitude"))

    primary_lon = scalar_float(primary.get("PositiveWest360Longitude"))
    if not np.isfinite(primary_lon):
        primary_lon = scalar_float(primary.get("PlanetocentricLongitude"))

    secondary_lon = scalar_float(secondary.get("PositiveWest360Longitude"))
    if not np.isfinite(secondary_lon):
        secondary_lon = scalar_float(secondary.get("PlanetocentricLongitude"))

    return {
        "img_id": img_id,
        "framelet": framelet,
        "fragment_id": int(fragment_id),
        "stage6_path": str(stage6_path),

        "primary_lat": primary_lat,
        "secondary_lat": secondary_lat,
        "primary_lon": primary_lon,
        "secondary_lon": secondary_lon,

        "dy_primary_rectified": dy_primary,
        "dy_secondary_rectified": dy_secondary,
        "peak_to_peak_rectified_dy_px": peak_to_peak_rectified_dy_px,

        "rectified_to_detector_scale": rectified_to_detector_scale,
        "peak_to_peak_detector_equiv_px": peak_to_peak_detector_equiv_px,

        "km_per_pixel_at_limb": km_per_pixel_at_limb,
        "peak_to_peak_km": peak_to_peak_km,

        "orig_xy_audit_px": orig_xy_px,
        "delta_orig_x_px": dx_orig,
        "delta_orig_y_px": dy_orig,

        "spice_radial_separation_km": spice_radial_separation_km,
        "spice_radial_separation_display": spice_display(spice_radial_separation_km),
        "spice_available_for_both_peaks": bool(np.isfinite(spice_radial_separation_km)),
    }


def collect_peak_separations(
    *,
    pjdir: Path,
    stage6_glob: str,
    valid_stage8_keys: set[tuple[str, str, int]],
    rectified_to_detector_scale: float,
) -> pd.DataFrame:
    records: list[dict] = []

    stage6_paths = sorted(pjdir.glob(stage6_glob))
    if not stage6_paths:
        raise FileNotFoundError(f"No Stage 6 CSVs found under {pjdir} using glob: {stage6_glob}")

    print(f"[08b] Stage 6 CSVs found: {len(stage6_paths)}")

    for path in stage6_paths:
        img_id = infer_img_id_from_stage6_path(path)
        framelet = infer_framelet_name(path)

        df = pd.read_csv(path, low_memory=False)

        if "fragment_id" not in df.columns:
            continue

        for fragment_id, frag in df.groupby("fragment_id", sort=True):
            fragment_id = int(fragment_id)
            key = (img_id, framelet, fragment_id)

            if key not in valid_stage8_keys:
                continue

            rec = extract_fragment_peak_separation(
                frag.reset_index(drop=True),
                img_id=img_id,
                framelet=framelet,
                fragment_id=fragment_id,
                stage6_path=path,
                rectified_to_detector_scale=rectified_to_detector_scale,
            )

            if rec is not None:
                records.append(rec)

    return pd.DataFrame(records)


# ============================================================
# Statistics
# ============================================================

def summarize_peak_separations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{
            "n_fragments": 0,
            "mean_peak_to_peak_km": np.nan,
            "median_peak_to_peak_km": np.nan,
            "std_peak_to_peak_km": np.nan,
            "p16_peak_to_peak_km": np.nan,
            "p84_peak_to_peak_km": np.nan,
            "min_peak_to_peak_km": np.nan,
            "max_peak_to_peak_km": np.nan,
            "mean_detector_equiv_px": np.nan,
            "median_detector_equiv_px": np.nan,
            "mean_rectified_dy_px": np.nan,
            "median_rectified_dy_px": np.nan,
            "n_spice_available": 0,
            "mean_spice_radial_separation_km": np.nan,
            "median_spice_radial_separation_km": np.nan,
        }])

    km = df["peak_to_peak_km"].to_numpy(float)
    det = df["peak_to_peak_detector_equiv_px"].to_numpy(float)
    rect = df["peak_to_peak_rectified_dy_px"].to_numpy(float)
    spice = numeric(df["spice_radial_separation_km"]).dropna()

    return pd.DataFrame([{
        "n_fragments": int(len(df)),

        "mean_peak_to_peak_km": float(np.nanmean(km)),
        "median_peak_to_peak_km": float(np.nanmedian(km)),
        "std_peak_to_peak_km": float(np.nanstd(km, ddof=1)) if len(df) > 1 else np.nan,
        "p16_peak_to_peak_km": float(np.nanpercentile(km, 16)),
        "p84_peak_to_peak_km": float(np.nanpercentile(km, 84)),
        "min_peak_to_peak_km": float(np.nanmin(km)),
        "max_peak_to_peak_km": float(np.nanmax(km)),

        "mean_detector_equiv_px": float(np.nanmean(det)),
        "median_detector_equiv_px": float(np.nanmedian(det)),

        "mean_rectified_dy_px": float(np.nanmean(rect)),
        "median_rectified_dy_px": float(np.nanmedian(rect)),

        "n_spice_available": int(len(spice)),
        "fraction_spice_available": float(len(spice) / len(df)) if len(df) else np.nan,
        "mean_spice_radial_separation_km": float(spice.mean()) if len(spice) else np.nan,
        "median_spice_radial_separation_km": float(spice.median()) if len(spice) else np.nan,
        "std_spice_radial_separation_km": float(spice.std(ddof=1)) if len(spice) > 1 else np.nan,
    }])


def summarize_by_latitude(df: pd.DataFrame, *, bin_width: float, min_fragments: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["primary_lat"] = numeric(work["primary_lat"])
    work["peak_to_peak_km"] = numeric(work["peak_to_peak_km"])
    work = work.dropna(subset=["primary_lat", "peak_to_peak_km"])

    if work.empty:
        return pd.DataFrame()

    lat_min = np.floor(work["primary_lat"].min() / bin_width) * bin_width
    lat_max = np.ceil(work["primary_lat"].max() / bin_width) * bin_width
    if lat_min == lat_max:
        lat_max = lat_min + bin_width

    edges = np.arange(lat_min, lat_max + bin_width, bin_width)
    work["lat_bin"] = pd.cut(work["primary_lat"], bins=edges, include_lowest=True)

    out = work.groupby("lat_bin", observed=True).agg(
        n_fragments=("peak_to_peak_km", "size"),
        lat_center=("primary_lat", "mean"),
        lat_min=("primary_lat", "min"),
        lat_max=("primary_lat", "max"),
        mean_peak_to_peak_km=("peak_to_peak_km", "mean"),
        median_peak_to_peak_km=("peak_to_peak_km", "median"),
        std_peak_to_peak_km=("peak_to_peak_km", "std"),
        p16_peak_to_peak_km=("peak_to_peak_km", lambda x: np.nanpercentile(x, 16)),
        p84_peak_to_peak_km=("peak_to_peak_km", lambda x: np.nanpercentile(x, 84)),
        n_spice_available=("spice_available_for_both_peaks", "sum"),
    ).reset_index()

    out["low_n_flag"] = out["n_fragments"] < int(min_fragments)
    return out


# ============================================================
# Plot helpers
# ============================================================

def smooth_delta(delta: np.ndarray, window: int, poly: int) -> np.ndarray:
    if window < 3 or window % 2 == 0 or window >= len(delta):
        return delta
    return savgol_filter(delta, window_length=window, polyorder=poly, mode="interp")


def load_representative_profile(
    df: pd.DataFrame,
    *,
    smooth: int,
    savgol_poly: int,
) -> dict | None:
    """
    Pick the fragment closest to the median separation and reconstruct the real
    Stage-6 ΔBrightness profile.
    """
    if df.empty:
        return None

    median_km = float(df["peak_to_peak_km"].median())
    rep_row = df.iloc[(df["peak_to_peak_km"] - median_km).abs().argsort().iloc[0]]

    stage6_path = Path(rep_row["stage6_path"])
    frag_id = int(rep_row["fragment_id"])

    if not stage6_path.exists():
        return None

    full = pd.read_csv(stage6_path, low_memory=False)

    frag = full[full["fragment_id"] == frag_id].copy()
    if frag.empty:
        return None

    frag["dy"] = numeric(frag["dy"])
    frag["pixel_value"] = numeric(frag["pixel_value"])
    frag = frag.dropna(subset=["dy", "pixel_value"]).sort_values("dy").reset_index(drop=True)

    if len(frag) < 4:
        return None

    dy = frag["dy"].to_numpy(float)
    brightness = frag["pixel_value"].to_numpy(float)

    delta = np.diff(brightness)
    delta = smooth_delta(delta, smooth, savgol_poly)
    dy_delta = dy[1:]

    primary_rows = frag[frag["stage6_peak_type"].astype(str) == "PRIMARY"]
    secondary_rows = frag[frag["stage6_peak_type"].astype(str) == "SECONDARY"]

    if len(primary_rows) != 1 or len(secondary_rows) < 1:
        return None

    primary_dy = float(primary_rows.iloc[0]["dy"])
    secondary_dy = float(secondary_rows.iloc[0]["dy"])

    scale = float(rep_row["rectified_to_detector_scale"])
    km_px = float(rep_row["km_per_pixel_at_limb"])

    x_km = (dy_delta - primary_dy) * scale * km_px
    primary_x = 0.0
    secondary_x = (secondary_dy - primary_dy) * scale * km_px

    primary_idx = int(np.argmin(np.abs(dy_delta - primary_dy)))
    secondary_idx = int(np.argmin(np.abs(dy_delta - secondary_dy)))

    return {
        "row": rep_row,
        "x_km": x_km,
        "delta": delta,
        "primary_x": primary_x,
        "secondary_x": secondary_x,
        "primary_y": float(delta[primary_idx]),
        "secondary_y": float(delta[secondary_idx]),
    }


def add_label(ax, label: str | None) -> None:
    if label:
        ax.text(0.02, 0.96, label, transform=ax.transAxes, va="top", ha="left", fontsize=11)


# ============================================================
# Plots
# ============================================================

def plot_histogram(df: pd.DataFrame, out_png: Path, label: str | None) -> None:
    if df.empty:
        return

    vals = df["peak_to_peak_km"].dropna()
    mean = vals.mean()
    median = vals.median()
    p16 = np.nanpercentile(vals, 16)
    p84 = np.nanpercentile(vals, 84)
    std = vals.std(ddof=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=30, alpha=0.9)

    ax.axvspan(p16, p84, alpha=0.18, label="16–84% range")
    ax.axvline(mean, linestyle="--", linewidth=2, label=f"Mean = {mean:.1f} km")
    ax.axvline(median, linestyle=":", linewidth=2, label=f"Median = {median:.1f} km")

    ax.text(
        0.68, 0.70,
        f"N = {len(vals)}\nMean = {mean:.1f} km\nMedian = {median:.1f} km\nσ = {std:.1f} km",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.12),
    )

    ax.set_xlabel("PRIMARY–SECONDARY peak separation (km)")
    ax.set_ylabel("Number of fragments")
    ax.set_title("Measured detached-haze peak-to-peak separation")
    add_label(ax, label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_vs_latitude(df: pd.DataFrame, out_png: Path, label: str | None) -> None:
    if df.empty:
        return

    work = df.copy()
    work["primary_lat"] = numeric(work["primary_lat"])
    work["peak_to_peak_km"] = numeric(work["peak_to_peak_km"])
    work = work.dropna(subset=["primary_lat", "peak_to_peak_km"])

    if work.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(work["primary_lat"], work["peak_to_peak_km"], s=20, alpha=0.55)

    ax.set_xlabel("PRIMARY peak planetocentric latitude (deg)")
    ax.set_ylabel("PRIMARY–SECONDARY peak separation (km)")
    ax.set_title("Detached-haze peak separation vs latitude")
    add_label(ax, label)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_binned_by_latitude(lat_df: pd.DataFrame, out_png: Path, label: str | None) -> None:
    if lat_df.empty:
        return

    x = lat_df["lat_center"].to_numpy(float)
    y = lat_df["median_peak_to_peak_km"].to_numpy(float)
    lo = lat_df["p16_peak_to_peak_km"].to_numpy(float)
    hi = lat_df["p84_peak_to_peak_km"].to_numpy(float)
    yerr = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, label="Median with 16–84% range")

    low = lat_df["low_n_flag"].to_numpy(bool)
    if low.any():
        ax.scatter(x[low], y[low], marker="x", s=120, label="Low-N bin")

    ax.set_xlabel("PRIMARY peak planetocentric latitude (deg)")
    ax.set_ylabel("PRIMARY–SECONDARY peak separation (km)")
    ax.set_title("Latitude-binned detached-haze peak separation")
    add_label(ax, label)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_spice_validation(df: pd.DataFrame, out_png: Path, label: str | None) -> None:
    work = df.copy()
    work["peak_to_peak_km"] = numeric(work["peak_to_peak_km"])
    work["spice_radial_separation_km"] = numeric(work["spice_radial_separation_km"])
    work = work.dropna(subset=["peak_to_peak_km", "spice_radial_separation_km"])

    if work.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "SPICE validation unavailable\nNo fragments had LocalRadius for both peaks.",
            ha="center",
            va="center",
            fontsize=14,
        )
        add_label(ax, label)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        return

    x = work["peak_to_peak_km"].to_numpy(float)
    y = work["spice_radial_separation_km"].to_numpy(float)

    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=24, alpha=0.65)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, label="1:1 line")

    ax.set_xlabel("Detector-geometry peak separation (km)")
    ax.set_ylabel("SPICE LocalRadius separation (km)")
    ax.set_title("SPICE validation of detached-haze peak separation")
    add_label(ax, label)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_summary_figure(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    out_png: Path,
    *,
    label: str | None,
    smooth: int,
    savgol_poly: int,
) -> None:
    """
    Publication flagship figure:
      A) Real representative Stage-6 ΔBrightness profile nearest the median.
      B) Full distribution of measured peak separations.
    """
    if df.empty:
        return

    rep = load_representative_profile(df, smooth=smooth, savgol_poly=savgol_poly)
    vals = df["peak_to_peak_km"].dropna()

    row = summary.iloc[0]
    mean = float(row["mean_peak_to_peak_km"])
    median = float(row["median_peak_to_peak_km"])
    std = float(row["std_peak_to_peak_km"])
    p16 = float(row["p16_peak_to_peak_km"])
    p84 = float(row["p84_peak_to_peak_km"])
    n = int(row["n_fragments"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(11, 10),
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

    # Panel A: representative real profile
    if rep is not None:
        x = rep["x_km"]
        y = rep["delta"]
        primary_x = rep["primary_x"]
        secondary_x = rep["secondary_x"]
        primary_y = rep["primary_y"]
        secondary_y = rep["secondary_y"]
        measured_km = float(abs(secondary_x - primary_x))

        ax1.plot(x, y, linewidth=2)
        ax1.scatter([primary_x], [primary_y], s=95, zorder=5, label="PRIMARY (visible limb)")
        ax1.scatter([secondary_x], [secondary_y], s=95, zorder=5, label="SECONDARY (detached haze)")
        
        y_arrow = min(primary_y, secondary_y) - 0.12 * (np.nanmax(y) - np.nanmin(y))
        ax1.annotate(
            "",
            xy=(secondary_x, y_arrow),
            xytext=(primary_x, y_arrow),
            arrowprops=dict(arrowstyle="<->", linewidth=2),
        )
        ax1.text(
            (secondary_x + primary_x) / 2,
            y_arrow,
            f"Δ = {measured_km:.1f} km",
            ha="center",
            va="bottom",
            fontsize=12,
        )

        ax1.axvline(0, linestyle=":", linewidth=1)
        ax1.set_xlabel("Distance along limb normal (km)")
        ax1.set_ylabel("ΔBrightness")
        ax1.set_title("(A) Representative Stage-6 brightness-gradient profile (nearest median separation)")
        ax1.legend(loc="best")
    else:
        ax1.axis("off")
        ax1.text(
            0.5, 0.5,
            "Representative Stage-6 profile unavailable",
            ha="center",
            va="center",
            fontsize=14,
        )

    add_label(ax1, label)

    # Panel B: distribution
    ax2.hist(vals, bins=30, alpha=0.9)
    ax2.axvspan(p16, p84, alpha=0.18, label="16–84% range")
    ax2.axvline(mean, linestyle="--", linewidth=2, label=f"Mean = {mean:.1f} km")
    ax2.axvline(median, linestyle=":", linewidth=2, label=f"Median = {median:.1f} km")

    ax2.text(
        0.68, 0.68,
        f"N = {n}\nMean = {mean:.1f} km\nMedian = {median:.1f} km\nσ = {std:.1f} km\n16–84% = {p16:.1f}–{p84:.1f} km",
        transform=ax2.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.12),
    )

    ax2.set_xlabel("PRIMARY–SECONDARY peak separation (km)")
    ax2.set_ylabel("Number of fragments")
    ax2.set_title("(B) Distribution across all retained detached-haze detections")
    ax2.legend(loc="best")

    fig.suptitle("Measured detached-haze peak separation", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    pjdir = Path(args.pjdir).expanduser().resolve()
    stage8_csv = resolve_stage8_csv(pjdir, args.stage8_csv)

    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else pjdir / "analysis" / "stage_08b_peak_to_peak"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    if args.y_step <= 0:
        raise ValueError("--y-step must be > 0")
    if args.radius <= 0:
        raise ValueError("--radius must be > 0")

    rectified_to_detector_scale = (2.0 * float(args.radius)) / float(args.y_step)

    print(f"[08b] PJ directory: {pjdir}")
    print(f"[08b] Stage 8 CSV : {stage8_csv}")
    print(f"[08b] Output dir  : {outdir}")
    print(f"[08b] Stage 3 radius: {args.radius}")
    print(f"[08b] Stage 3 y_step: {args.y_step}")
    print(f"[08b] rectified_to_detector_scale = {rectified_to_detector_scale:.6f}")

    print("[08b] Loading Stage 8 retained fragment keys...")
    valid_keys = load_stage8_valid_keys(stage8_csv)
    print(f"[08b] Retained Stage 8 keys: {len(valid_keys)}")

    print("[08b] Reading Stage 6 outputs and computing peak separations...")
    df = collect_peak_separations(
        pjdir=pjdir,
        stage6_glob=args.stage6_glob,
        valid_stage8_keys=valid_keys,
        rectified_to_detector_scale=rectified_to_detector_scale,
    )

    out_frag = outdir / "stage08b_peak_to_peak_fragments.csv"
    df.to_csv(out_frag, index=False)
    print(f"[08b] Wrote fragment-level table: {out_frag}")
    print(f"[08b] Fragments with measured PRIMARY–SECONDARY separation: {len(df)}")

    summary = summarize_peak_separations(df)
    out_summary = outdir / "stage08b_peak_to_peak_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[08b] Wrote summary: {out_summary}")

    lat_summary = summarize_by_latitude(
        df,
        bin_width=args.lat_bin_width_deg,
        min_fragments=args.min_fragments_per_bin,
    )
    out_lat = outdir / "stage08b_peak_to_peak_by_latitude.csv"
    lat_summary.to_csv(out_lat, index=False)
    print(f"[08b] Wrote latitude summary: {out_lat}")

    print("[08b] Writing publication figures...")

    plot_summary_figure(
        df,
        summary,
        outdir / "stage08b_peak_separation_summary.png",
        label=args.pj_label,
        smooth=args.smooth,
        savgol_poly=args.savgol_poly,
    )

    plot_histogram(
        df,
        outdir / "stage08b_peak_to_peak_histogram.png",
        args.pj_label,
    )

    plot_vs_latitude(
        df,
        outdir / "stage08b_peak_to_peak_vs_latitude.png",
        args.pj_label,
    )

    plot_binned_by_latitude(
        lat_summary,
        outdir / "stage08b_peak_to_peak_by_latitude.png",
        args.pj_label,
    )

    plot_spice_validation(
        df,
        outdir / "stage08b_spice_vs_detector_validation.png",
        args.pj_label,
    )

    print("[08b] Done.")

    row = summary.iloc[0]
    print()
    print("Headline result:")
    print(f"  N fragments: {int(row['n_fragments'])}")
    print(f"  Mean PRIMARY–SECONDARY separation:   {row['mean_peak_to_peak_km']:.2f} km")
    print(f"  Median PRIMARY–SECONDARY separation: {row['median_peak_to_peak_km']:.2f} km")
    print(f"  Std dev:                             {row['std_peak_to_peak_km']:.2f} km")
    print(f"  16–84% range:                        {row['p16_peak_to_peak_km']:.2f}–{row['p84_peak_to_peak_km']:.2f} km")
    print(f"  Min / Max:                           {row['min_peak_to_peak_km']:.2f} / {row['max_peak_to_peak_km']:.2f} km")
    print(f"  Mean detector-equivalent separation: {row['mean_detector_equiv_px']:.2f} px")
    print(f"  Mean rectified-dy separation:        {row['mean_rectified_dy_px']:.2f} px")
    print(f"  SPICE radial separation available:   {int(row['n_spice_available'])}/{int(row['n_fragments'])}")

    if np.isfinite(row["mean_spice_radial_separation_km"]):
        print(f"  Mean SPICE radial separation:        {row['mean_spice_radial_separation_km']:.2f} km")
    else:
        print("  Mean SPICE radial separation:        SPICE unavailable")


if __name__ == "__main__":
    main()