#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pjdir", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--lat-bin-width-deg", type=float, default=1.0)
    return ap.parse_args()


def make_edges(x: np.ndarray, bin_width: float) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError("No finite latitude values available for binning.")
    lo = math.floor(np.nanmin(finite) / bin_width) * bin_width
    hi = math.ceil(np.nanmax(finite) / bin_width) * bin_width
    return np.arange(lo, hi + bin_width, bin_width)


def infer_framelet_name(stage6_csv: Path) -> str:
    m = re.match(r"(.+)_STAGE6\.csv$", stage6_csv.name)
    return m.group(1) if m else stage6_csv.parent.name


def compute_rep_lat(frag: pd.DataFrame) -> float | None:
    if "stage6_peak_type" not in frag.columns or "PlanetocentricLatitude" not in frag.columns:
        return None

    primary_rows = frag[frag["stage6_peak_type"].astype(str) == "PRIMARY"]
    if len(primary_rows) != 1:
        return None

    primary_pos = frag.index.get_loc(primary_rows.index[0])

    lat_vals = pd.to_numeric(frag["PlanetocentricLatitude"], errors="coerce").to_numpy(float)

    primary_lat = lat_vals[primary_pos]
    if np.isfinite(primary_lat):
        return float(primary_lat)

    finite_idxs = np.where(np.isfinite(lat_vals))[0]
    if len(finite_idxs) == 0:
        return None

    nearest = finite_idxs[np.argmin(np.abs(finite_idxs - primary_pos))]
    return float(lat_vals[nearest])


def interval_mid(iv) -> float:
    return float((iv.left + iv.right) / 2.0)


def main():
    args = parse_args()
    pjdir = Path(args.pjdir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else (
        pjdir / "analysis" / "PJ14_stage_08_perijove_analysis" / "validation"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    stage6_paths = sorted((pjdir / "cub").glob("*/stage_06_peaks/**/*_STAGE6.csv"))

    for p in stage6_paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue

        if "fragment_id" not in df.columns:
            continue

        img_id = p.parts[p.parts.index("cub") + 1] if "cub" in p.parts else p.parent.parent.name
        framelet = infer_framelet_name(p)

        for frag_id, frag in df.groupby("fragment_id"):
            lat = compute_rep_lat(frag)
            if lat is None or not np.isfinite(lat):
                continue

            det = False
            if "detectability_flag" in frag.columns:
                val = frag["detectability_flag"].iloc[0]
                det = str(val).strip().lower() in {"true", "1", "yes"}

            has_secondary = False
            if "stage6_peak_type" in frag.columns:
                has_secondary = (frag["stage6_peak_type"].astype(str) == "SECONDARY").any()

            records.append({
                "img_id": img_id,
                "framelet": framelet,
                "fragment_id": int(frag_id),
                "limb_lat": float(lat),
                "detectable": bool(det),
                "has_secondary": bool(has_secondary),
            })

    all_frags = pd.DataFrame(records)
    all_frags.to_csv(outdir / "validation2_fragment_detectability_all.csv", index=False)

    edges = make_edges(all_frags["limb_lat"].to_numpy(float), args.lat_bin_width_deg)
    all_frags["lat_bin"] = pd.cut(all_frags["limb_lat"], bins=edges, include_lowest=True)

    g = all_frags.groupby("lat_bin", observed=True).agg(
        n_total=("fragment_id", "size"),
        n_detectable=("detectable", "sum"),
        n_nondetectable=("detectable", lambda s: (~s).sum()),
        n_secondary_detectable=("has_secondary", lambda s: s[all_frags.loc[s.index, "detectable"]].sum()),
    ).reset_index()

    g["lat_left"] = g["lat_bin"].map(lambda iv: float(iv.left))
    g["lat_right"] = g["lat_bin"].map(lambda iv: float(iv.right))
    g["lat_center"] = g["lat_bin"].map(interval_mid)
    g["frac_detectable"] = g["n_detectable"] / g["n_total"].replace(0, np.nan)

    out = g[[
        "lat_bin",
        "lat_left",
        "lat_right",
        "lat_center",
        "n_total",
        "n_detectable",
        "n_nondetectable",
        "n_secondary_detectable",
        "frac_detectable",
    ]].copy()

    out.to_csv(outdir / "validation2_detectability_by_latitude.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(out["lat_center"], out["n_total"], marker="o", label="Total fragments")
    ax.plot(out["lat_center"], out["n_detectable"], marker="o", label="Detectable fragments")
    ax.plot(out["lat_center"], out["n_nondetectable"], marker="o", label="Non-detectable fragments")
    ax.set_xlabel("Planetocentric latitude bin center (deg)")
    ax.set_ylabel("Fragment count")
    ax.set_title("Validation 2 — Detectable vs non-detectable fragments by latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "validation2_detectability_by_latitude.png", dpi=200)
    plt.close(fig)

    print(f"Wrote: {outdir / 'validation2_fragment_detectability_all.csv'}")
    print(f"Wrote: {outdir / 'validation2_detectability_by_latitude.csv'}")
    print(f"Wrote: {outdir / 'validation2_detectability_by_latitude.png'}")


if __name__ == "__main__":
    main()