#!/usr/bin/env python3
"""
08.perijove_analysis.py — Perijove Analysis (Stage 08)

What this does (Stage 08):
  1) Loads all Stage-7 fragment tables across a Perijove (stage7_fragment_table.csv)
  2) Re-opens Stage-6 per-sample CSVs to pull *limb-crossing geometry* from the
     PRIMARY peak row, explicitly:
        - PlanetocentricLatitude
        - PositiveWest360Longitude   <-- IMPORTANT (your required longitude)
        - SlantDistance             (stored + available for future use)
     using an explicit rule:
        a) Use the PRIMARY row’s values (lat/lon/slant) if all are finite
        b) Else fallback to the *nearest* row (by sample index) where ALL THREE
           (lat/lon/slant) are finite simultaneously
  3) Produces Perijove-wide (fragment-weighted) occurrence vs latitude + CI
  4) Produces IMG×latitude heatmap (fraction with SECONDARY per IMG per lat bin)
  5) Produces a LAT×LON “map” heatmap (fraction with SECONDARY per lat/lon cell)

Inputs (expected under --pjdir):
  - **/analysis/stage_07/stage7_fragment_table.csv
  - **/cub/stage_06_peaks/**/*_STAGE6.csv

Outputs (in --outdir, default: <pjdir>/analysis/stage_08):
  - stage8_fragments_all.csv
  - stage8_img_summary.csv
  - stage8_latitude_bins.csv
  - stage8_occurrence_by_lat.png
  - stage8_heatmap_img_vs_lat.png
  - stage8_lonlat_grid.csv
  - stage8_lonlat_map.png
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pjdir", type=str, required=True,
                    help="Perijove-scoped root directory containing IMG dirs.")

    ap.add_argument("--stage7-glob", type=str,
                    default="**/analysis/stage_07/stage7_fragment_table.csv",
                    help="Glob under --pjdir to find Stage 7 fragment tables.")

    ap.add_argument("--stage6-glob", type=str,
                    default="**/cub/stage_06_peaks/**/*_STAGE6.csv",
                    help="Glob under --pjdir to find Stage 6 per-sample CSVs.")

    ap.add_argument("--outdir", type=str, default=None,
                    help="Output dir. Default: <pjdir>/analysis/stage_08")

    ap.add_argument("--pj-label", type=str, default=None,
                    help="Label stamped on plots, e.g. 'Perijove 14'.")

    ap.add_argument("--img-id-col", type=str, default="img_id",
                    help="Column name added for IMG identifier.")

    # Latitude aggregation
    ap.add_argument("--lat-bin-width-deg", type=float, default=1.0,
                    help="Latitude bin width (degrees).")
    ap.add_argument("--min-fragments-per-lat-bin", type=int, default=30,
                    help="Low-N flag threshold for Perijove latitude bins.")

    # IMG×lat heatmap
    ap.add_argument("--min-fragments-per-img-lat-bin", type=int, default=8,
                    help="Mask IMG×lat cells with <N fragments (set NaN).")

    # Lon/Lat map
    ap.add_argument("--lon-bin-width-deg", type=float, default=5.0,
                    help="Longitude bin width (degrees) for lon/lat map (0–360).")
    ap.add_argument("--min-fragments-per-lonlat-cell", type=int, default=10,
                    help="Mask lon/lat cells with <N fragments (set NaN).")
    ap.add_argument("--map-lat-min", type=float, default=None,
                    help="Optional: restrict lon/lat map to lat >= this.")
    ap.add_argument("--map-lat-max", type=float, default=None,
                    help="Optional: restrict lon/lat map to lat <= this.")

    return ap.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def infer_img_id_from_path(stage7_path: Path) -> str:
    """
    Infer IMG ID from:
      <IMGDIR>/analysis/stage_07/stage7_fragment_table.csv
    """
    parts = stage7_path.parts  # these are STRINGS already

    try:
        idx = parts.index("analysis")
        if idx > 0:
            return parts[idx - 1]
    except ValueError:
        pass

    # Fallback (robust for minor layout differences)
    if stage7_path.parent.name == "stage_07":
        return stage7_path.parent.parent.parent.name  # IMGDIR
    return stage7_path.parent.name


def infer_framelet_name(stage6_csv: Path) -> str:
    # Stage 6 naming: "<FRAMELET>_STAGE6.csv"
    m = re.match(r"(.+)_STAGE6\.csv$", stage6_csv.name)
    if m:
        return m.group(1)
    return stage6_csv.parent.name


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        denom = 1.0 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / denom
        half = (z * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denom
        lo = np.clip(center - half, 0, 1)
        hi = np.clip(center + half, 0, 1)
    lo = np.where(n > 0, lo, np.nan)
    hi = np.where(n > 0, hi, np.nan)
    return lo, hi


def make_edges(x: np.ndarray, bin_width: float, *, pad_to: tuple[float, float] | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if pad_to is not None:
        lo, hi = pad_to
        x_min, x_max = lo, hi
    else:
        x_min, x_max = np.nanmin(x), np.nanmax(x)

    lo_edge = math.floor(x_min / bin_width) * bin_width
    hi_edge = math.ceil(x_max / bin_width) * bin_width
    return np.arange(lo_edge, hi_edge + bin_width, bin_width)


# -----------------------------
# Load Stage 7
# -----------------------------
def load_stage7_all(pjdir: Path, stage7_glob: str, img_id_col: str) -> pd.DataFrame:
    paths = sorted(pjdir.glob(stage7_glob))
    if not paths:
        raise FileNotFoundError(f"No Stage-7 fragment tables under {pjdir} with glob '{stage7_glob}'")

    frames = []
    for p in paths:
        df = pd.read_csv(p)

        required = ["framelet", "fragment_id", "has_secondary"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{p} missing required columns: {missing}")

        df = df.copy()
        df[img_id_col] = infer_img_id_from_path(p)
        df["_stage7_path"] = str(p)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Normalize types
    out["fragment_id"] = pd.to_numeric(out["fragment_id"], errors="coerce").astype("Int64")
    out["has_secondary"] = out["has_secondary"].astype(bool)
    return out


# -----------------------------
# Pull limb-crossing geometry from Stage 6
# -----------------------------
@dataclass
class LimbGeom:
    limb_lat: float | None
    limb_lon: float | None
    limb_slant_km: float | None
    geom_source: str   # "primary" | "nearest_spice" | "none"


def compute_limb_geom_from_stage6_fragment(
    frag: pd.DataFrame,
    lat_col: str = "PlanetocentricLatitude",
    lon_col: str = "PositiveWest360Longitude",
    slant_col: str = "SlantDistance",
    peak_col: str = "stage6_peak_type",
) -> LimbGeom:
    """
    Explicit rule:
      1) Find PRIMARY row (must be exactly 1)
      2) If PRIMARY has finite lat/lon/slant -> use it ("primary")
      3) Else choose nearest row (by row index distance to PRIMARY index)
         where lat/lon/slant are ALL finite simultaneously ("nearest_spice")
      4) Else "none"
    """
    if peak_col not in frag.columns:
        return LimbGeom(None, None, None, "none")

    primary_rows = frag[frag[peak_col].astype(str) == "PRIMARY"]
    if len(primary_rows) != 1:
        return LimbGeom(None, None, None, "none")

    primary_idx = primary_rows.index[0]

    lat = pd.to_numeric(primary_rows[lat_col], errors="coerce").iloc[0] if lat_col in frag.columns else np.nan
    lon = pd.to_numeric(primary_rows[lon_col], errors="coerce").iloc[0] if lon_col in frag.columns else np.nan
    slt = pd.to_numeric(primary_rows[slant_col], errors="coerce").iloc[0] if slant_col in frag.columns else np.nan

    if np.isfinite(lat) and np.isfinite(lon) and np.isfinite(slt):
        return LimbGeom(float(lat), float(lon), float(slt), "primary")

    # Fallback: nearest row with ALL finite simultaneously
    lat_vals = pd.to_numeric(frag[lat_col], errors="coerce").to_numpy(float) if lat_col in frag.columns else np.full(len(frag), np.nan)
    lon_vals = pd.to_numeric(frag[lon_col], errors="coerce").to_numpy(float) if lon_col in frag.columns else np.full(len(frag), np.nan)
    slt_vals = pd.to_numeric(frag[slant_col], errors="coerce").to_numpy(float) if slant_col in frag.columns else np.full(len(frag), np.nan)

    finite_all = np.isfinite(lat_vals) & np.isfinite(lon_vals) & np.isfinite(slt_vals)
    idxs = np.where(finite_all)[0]
    if len(idxs) == 0:
        return LimbGeom(None, None, None, "none")

    nearest = idxs[np.argmin(np.abs(idxs - primary_idx))]
    return LimbGeom(float(lat_vals[nearest]), float(lon_vals[nearest]), float(slt_vals[nearest]), "nearest_spice")


def load_stage6_geom_index(
    pjdir: Path,
    stage6_glob: str,
) -> pd.DataFrame:
    """
    Builds a lookup table of limb-crossing geometry for each (framelet, fragment_id)
    across the entire Perijove by re-reading Stage 6 CSVs once each.

    NOTE:
    Fragment acceptance is finalized in Stage 7. Stage 08 does not filter on
    'accepted' or 'reject_reason'; it only recovers SPICE geometry and relies on
    the Stage 7 merge to select valid fragments.
    """
    paths = sorted(pjdir.glob(stage6_glob))
    if not paths:
        raise FileNotFoundError(f"No Stage-6 CSVs under {pjdir} with glob '{stage6_glob}'")

    records = []
    for p in paths:
        df = pd.read_csv(p)

        # NOTE:
        # Stage 08 does NOT filter on 'accepted' or 'reject_reason'.
        # Stage 7 has already finalized fragment acceptance.
        # Here we only recover limb-crossing SPICE geometry.
        df = df.copy()

        if "fragment_id" not in df.columns:
            raise KeyError(f"{p} missing 'fragment_id' column")
        framelet = infer_framelet_name(p)

        # For stable indexing when choosing "nearest row", keep original row order
        # (the CSV is already in sample order)
        for frag_id in sorted(df["fragment_id"].unique()):
            frag = df[df["fragment_id"] == frag_id]
            geom = compute_limb_geom_from_stage6_fragment(frag)

            records.append({
                "framelet": framelet,
                "fragment_id": int(frag_id),
                "limb_lat": geom.limb_lat,
                "limb_lon": geom.limb_lon,
                "limb_slant_km": geom.limb_slant_km,
                "geom_source": geom.geom_source,
                "_stage6_path": str(p),
            })

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError("Stage-6 geom index ended up empty after filtering. (No accepted fragments?)")
    return out


# -----------------------------
# Aggregations
# -----------------------------
def perijove_lat_bins(df: pd.DataFrame, lat_bin_w: float, min_n: int) -> pd.DataFrame:
    edges = make_edges(df["limb_lat"].to_numpy(float), lat_bin_w)
    work = df.copy()
    work["lat_bin"] = pd.cut(work["limb_lat"], bins=edges, include_lowest=True)

    g = work.groupby("lat_bin", observed=True)
    out = g.agg(
        n_fragments=("has_secondary", "size"),
        n_with_secondary=("has_secondary", "sum"),
        frac_with_secondary=("has_secondary", "mean"),
        lat_center=("limb_lat", "mean"),
        lat_min=("limb_lat", "min"),
        lat_max=("limb_lat", "max"),
        n_imgs=("img_id", "nunique"),
    ).reset_index()

    out["low_n_flag"] = out["n_fragments"] < int(min_n)
    lo, hi = wilson_ci(out["n_with_secondary"].to_numpy(), out["n_fragments"].to_numpy())
    out["frac_ci95_lo"] = lo
    out["frac_ci95_hi"] = hi
    return out


def per_img_summary(df: pd.DataFrame, img_id_col: str) -> pd.DataFrame:
    g = df.groupby(img_id_col)
    out = g.agg(
        n_fragments=("has_secondary", "size"),
        n_with_secondary=("has_secondary", "sum"),
        frac_with_secondary=("has_secondary", "mean"),
        lat_min=("limb_lat", "min"),
        lat_max=("limb_lat", "max"),
        lon_min=("limb_lon", "min"),
        lon_max=("limb_lon", "max"),
        slant_min=("limb_slant_km", "min"),
        slant_max=("limb_slant_km", "max"),
        n_framelets=("framelet", "nunique"),
    ).reset_index()
    return out


def build_img_lat_heatmap(
    df: pd.DataFrame,
    img_id_col: str,
    lat_bin_w: float,
    min_cell_n: int,
):
    lat_edges = make_edges(df["limb_lat"].to_numpy(float), lat_bin_w)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0

    work = df.copy()
    work["lat_bin_i"] = np.digitize(work["limb_lat"].to_numpy(float), lat_edges, right=False) - 1
    work = work[(work["lat_bin_i"] >= 0) & (work["lat_bin_i"] < len(lat_centers))].copy()

    imgs = sorted(work[img_id_col].astype(str).unique().tolist())
    mat = np.full((len(imgs), len(lat_centers)), np.nan, dtype=float)

    for r, img in enumerate(imgs):
        d = work[work[img_id_col].astype(str) == img]
        gg = d.groupby("lat_bin_i")
        for i, sub in gg:
            if len(sub) < int(min_cell_n):
                mat[r, i] = np.nan
            else:
                mat[r, i] = float(sub["has_secondary"].mean())

    return imgs, lat_centers, mat


def build_lonlat_grid(
    df: pd.DataFrame,
    lat_bin_w: float,
    lon_bin_w: float,
    min_cell_n: int,
    map_lat_min: float | None,
    map_lat_max: float | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      grid_df (long form)
      lat_edges, lon_edges
      mat (2D array) with NaNs masked for low-N
    """
    work = df.copy()

    if map_lat_min is not None:
        work = work[work["limb_lat"] >= float(map_lat_min)]
    if map_lat_max is not None:
        work = work[work["limb_lat"] <= float(map_lat_max)]

    # Longitude: PositiveWest360Longitude should be [0,360). We'll bin 0..360.
    lon_edges = np.arange(0.0, 360.0 + lon_bin_w, lon_bin_w)
    lat_edges = make_edges(work["limb_lat"].to_numpy(float), lat_bin_w)

    work["lat_bin"] = pd.cut(work["limb_lat"], bins=lat_edges, include_lowest=True)
    work["lon_bin"] = pd.cut(work["limb_lon"], bins=lon_edges, include_lowest=True, right=False)

    g = work.groupby(["lat_bin", "lon_bin"], observed=True)
    grid = g.agg(
        n_fragments=("has_secondary", "size"),
        frac_with_secondary=("has_secondary", "mean"),
        lat_center=("limb_lat", "mean"),
        lon_center=("limb_lon", "mean"),
    ).reset_index()

    grid["masked_low_n"] = grid["n_fragments"] < int(min_cell_n)

    # Build matrix for plotting
    lat_bins = grid["lat_bin"].cat.categories if hasattr(grid["lat_bin"], "cat") else sorted(work["lat_bin"].unique())
    lon_bins = grid["lon_bin"].cat.categories if hasattr(grid["lon_bin"], "cat") else sorted(work["lon_bin"].unique())

    lat_bin_list = list(lat_bins)
    lon_bin_list = list(lon_bins)

    lat_index = {b: i for i, b in enumerate(lat_bin_list)}
    lon_index = {b: j for j, b in enumerate(lon_bin_list)}

    mat = np.full((len(lat_bin_list), len(lon_bin_list)), np.nan, dtype=float)

    for _, row in grid.iterrows():
        i = lat_index.get(row["lat_bin"])
        j = lon_index.get(row["lon_bin"])
        if i is None or j is None:
            continue
        if bool(row["masked_low_n"]):
            mat[i, j] = np.nan
        else:
            mat[i, j] = float(row["frac_with_secondary"])

    return grid, lat_edges, lon_edges, mat


# -----------------------------
# Plotting
# -----------------------------
def plot_occurrence_by_lat(df_bins: pd.DataFrame, out_png: Path, label: str | None) -> None:
    x = df_bins["lat_center"].to_numpy(float)
    y = df_bins["frac_with_secondary"].to_numpy(float)
    ylo = df_bins["frac_ci95_lo"].to_numpy(float)
    yhi = df_bins["frac_ci95_hi"].to_numpy(float)
    low_n = df_bins["low_n_flag"].to_numpy(bool)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o")
    ax.vlines(x, ylo, yhi)

    ax.set_xlabel("Planetocentric Latitude at Limb Crossing (deg)")
    ax.set_ylabel("Fraction of fragments with SECONDARY peak")
    ax.set_title("Detached haze occurrence — Perijove aggregate (fragment-weighted)")

    if label:
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=13, verticalalignment="top", horizontalalignment="left")

    if low_n.any():
        ax.scatter(x[low_n], y[low_n], marker="x", s=120)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_img_lat_heatmap(imgs: list[str], lat_centers: np.ndarray, mat: np.ndarray, out_png: Path, label: str | None) -> None:
    fig, ax = plt.subplots(figsize=(12, max(4, 0.25 * len(imgs))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", origin="lower")

    ax.set_xlabel("Latitude bin center (deg)")
    ax.set_ylabel("IMG")
    ax.set_title("Detached haze occurrence heatmap (per IMG × latitude)")

    # sparse x ticks
    if len(lat_centers) <= 25:
        xticks = np.arange(len(lat_centers))
    else:
        step = max(1, len(lat_centers) // 20)
        xticks = np.arange(0, len(lat_centers), step)

    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{lat_centers[i]:.0f}" for i in xticks])

    ax.set_yticks(np.arange(len(imgs)))
    ax.set_yticklabels(imgs)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction with SECONDARY peak (masked if low-N)")

    if label:
        ax.text(0.02, 1.02, label, transform=ax.transAxes,
                fontsize=12, verticalalignment="bottom", horizontalalignment="left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_lonlat_map(
    mat: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    out_png: Path,
    label: str | None,
    map_lat_min: float | None,
    map_lat_max: float | None,
) -> None:
    # imshow expects [rows, cols] => [lat_bins, lon_bins]
    fig, ax = plt.subplots(figsize=(12, 6))

    extent = [lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]]
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", origin="lower", extent=extent)

    ax.set_xlabel("PositiveWest360Longitude (deg)")
    ax.set_ylabel("PlanetocentricLatitude (deg)")
    ax.set_title("Detached haze occurrence map (lat × lon) — Perijove aggregate")

    if map_lat_min is not None or map_lat_max is not None:
        ax.set_ylim(
            map_lat_min if map_lat_min is not None else lat_edges[0],
            map_lat_max if map_lat_max is not None else lat_edges[-1],
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction with SECONDARY peak (masked if low-N)")

    if label:
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=12, verticalalignment="top", horizontalalignment="left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    pjdir = Path(args.pjdir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else (pjdir / "analysis" / "stage_08")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load Stage 7
    df7 = load_stage7_all(pjdir, args.stage7_glob, args.img_id_col)
    print(f"[Stage 08] A) Loaded Stage-7 fragments: {len(df7):,} rows")

    # Load geometry index from Stage 6 (lat + PositiveWest360Longitude + SlantDistance at PRIMARY)
    geom = load_stage6_geom_index(pjdir, args.stage6_glob)
    print(f"[Stage 08] B) Built Stage-6 geometry index: {len(geom):,} fragment keys")

    # Merge geometry onto Stage 7 fragment rows
    # Key: (framelet, fragment_id)
    df = df7.merge(
        geom[["framelet", "fragment_id", "limb_lat", "limb_lon", "limb_slant_km", "geom_source"]],
        on=["framelet", "fragment_id"],
        how="left",
        validate="many_to_one",
    )

    # ---- Normalize geometry columns after merge ----
    # Stage 6 SPICE geometry is authoritative for Stage 08

    # If pandas suffixed latitude due to overlap, prefer Stage-6 (_y)
    if "limb_lat_y" in df.columns:
        df["limb_lat"] = df["limb_lat_y"]
    elif "limb_lat" not in df.columns:
        raise KeyError(
            "No Stage-6 limb latitude column found after merge "
            "(expected limb_lat_y from Stage 6)"
        )

    # Longitude & slant distance should already be Stage-6-native
    if "limb_lon" not in df.columns:
        raise KeyError("Missing limb_lon after merge (Stage-6 longitude expected)")

    if "limb_slant_km" not in df.columns:
        raise KeyError("Missing limb_slant_km after merge (Stage-6 slant distance expected)")

    # Optional cleanup: drop Stage-7 latitude proxy
    if "limb_lat_x" in df.columns:
        df = df.drop(columns=["limb_lat_x"])
    
    required = ["limb_lat", "limb_lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required SPICE geometry columns after normalization: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    n_missing = df["limb_lat"].isna().sum()
    print(f"[Stage 08] C) Geometry merged (missing limb_lat: {n_missing:,})")

    # Filter fragments to those with valid limb-crossing geometry
    df["limb_lat"] = pd.to_numeric(df["limb_lat"], errors="coerce")
    df["limb_lon"] = pd.to_numeric(df["limb_lon"], errors="coerce")
    df["limb_slant_km"] = pd.to_numeric(df["limb_slant_km"], errors="coerce")

    df_valid = df[np.isfinite(df["limb_lat"]) & np.isfinite(df["limb_lon"])].copy()
    if df_valid.empty:
        raise RuntimeError("No fragments have finite limb_lat AND limb_lon after Stage-6 merge.")

    # Save concatenated tables
    df.to_csv(outdir / "stage8_fragments_all_raw.csv", index=False)
    df_valid.to_csv(outdir / "stage8_fragments_all.csv", index=False)

    # Per-IMG summary
    df_img = per_img_summary(df_valid, args.img_id_col)
    df_img.to_csv(outdir / "stage8_img_summary.csv", index=False)

    # Perijove latitude bins
    df_bins = perijove_lat_bins(df_valid, float(args.lat_bin_width_deg), int(args.min_fragments_per_lat_bin))
    df_bins.to_csv(outdir / "stage8_latitude_bins.csv", index=False)

    # Plot: occurrence vs latitude
    plot_occurrence_by_lat(df_bins, outdir / "stage8_occurrence_by_lat.png", args.pj_label)

    # IMG×lat heatmap
    imgs, lat_centers, img_lat_mat = build_img_lat_heatmap(
        df_valid,
        args.img_id_col,
        float(args.lat_bin_width_deg),
        int(args.min_fragments_per_img_lat_bin),
    )
    plot_img_lat_heatmap(imgs, lat_centers, img_lat_mat, outdir / "stage8_heatmap_img_vs_lat.png", args.pj_label)

    # Lon/Lat map grid
    grid, lat_edges, lon_edges, lonlat_mat = build_lonlat_grid(
        df_valid,
        float(args.lat_bin_width_deg),
        float(args.lon_bin_width_deg),
        int(args.min_fragments_per_lonlat_cell),
        args.map_lat_min,
        args.map_lat_max,
    )
    grid.to_csv(outdir / "stage8_lonlat_grid.csv", index=False)
    plot_lonlat_map(
        lonlat_mat,
        lat_edges,
        lon_edges,
        outdir / "stage8_lonlat_map.png",
        args.pj_label,
        args.map_lat_min,
        args.map_lat_max,
    )

    print("[Stage 08] Complete")
    print(f"  PJ dir                 → {pjdir}")
    print(f"  Output dir             → {outdir}")
    print(f"  Fragments (raw)        → {outdir / 'stage8_fragments_all_raw.csv'}")
    print(f"  Fragments (geom-valid) → {outdir / 'stage8_fragments_all.csv'}")
    print(f"  IMG summary            → {outdir / 'stage8_img_summary.csv'}")
    print(f"  Lat bins               → {outdir / 'stage8_latitude_bins.csv'}")
    print(f"  Plot (lat curve)       → {outdir / 'stage8_occurrence_by_lat.png'}")
    print(f"  Plot (IMG×lat)         → {outdir / 'stage8_heatmap_img_vs_lat.png'}")
    print(f"  Lon/Lat grid           → {outdir / 'stage8_lonlat_grid.csv'}")
    print(f"  Plot (lon/lat map)     → {outdir / 'stage8_lonlat_map.png'}")
    print()
    print("Geometry rule used for limb-crossing lat/lon/slant:")
    print("  - PRIMARY row values if finite; else nearest row where (lat, lon, slant) are all finite.")


if __name__ == "__main__":
    main()
