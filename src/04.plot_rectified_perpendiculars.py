#!/usr/bin/env python3
"""
04.plot_rectified_perpendiculars.py

Stage 4: In rectified space, detect the two limb edges (paired fragments) per column,
compute their midpoint, generate perpendicular sample points, associate each point with
original pixel coordinates + pixel_value via the rectified mapping, and export:

Outputs (one framelet):
  - *_RECTIFIED_PERP_SAMPLES.csv
      fragment_id, sample_id, rx, ry, orig_x, orig_y, pixel_value
  - *_RECTIFIED_PERP_OVERLAY.png
      rectified image + blue edge dots (two edges) + red perpendicular sample dots

Inputs (one framelet):
  - *_RECTIFIED_MAPPING.csv (must contain: rect_x, rect_y, orig_x, orig_y, pixel_value)
  - *_RECTIFIED.tif or *_RECTIFIED.png next to mapping CSV
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import tempfile
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (rect_x, rect_y) -> (orig_x, orig_y, pixel_value)
Mapping = Dict[Tuple[int, int], Tuple[float, float, float]]


# ----------------------------
# Mapping loader (fast lookup)
# ----------------------------

def load_rectified_mapping(mapping_csv: Path) -> Mapping:
    """
    Load *_RECTIFIED_MAPPING.csv into memory once.

    Expected columns:
      rect_x, rect_y, orig_x, orig_y, pixel_value
    """
    mapping: Mapping = {}
    with mapping_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"rect_x", "rect_y", "orig_x", "orig_y", "pixel_value"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Mapping CSV missing required columns. "
                f"Expected {sorted(required)}, found {reader.fieldnames}"
            )

        for row in reader:
            rx = int(float(row["rect_x"]))
            ry = int(float(row["rect_y"]))
            ox = float(row["orig_x"])
            oy = float(row["orig_y"])
            pv = float(row["pixel_value"])
            mapping[(rx, ry)] = (ox, oy, pv)

    return mapping


# ----------------------------
# Image helpers
# ----------------------------

def read_image_grayscale(path: Path) -> np.ndarray:
    """
    Read an image with matplotlib and return a 2D grayscale array (float32).
    """
    img = mpimg.imread(str(path))
    if img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        gray = img[..., :3].mean(axis=2).astype(np.float32)
    return gray


def find_rectified_image(mapping_csv: Path) -> Path:
    """
    Infer rectified image path from mapping csv stem.
    Prefer .tif, else .png.
    Example:
      *_RECTIFIED_MAPPING.csv -> *_RECTIFIED.tif or *_RECTIFIED.png
    """
    base = mapping_csv.name.replace("_RECTIFIED_MAPPING.csv", "_RECTIFIED")
    tif = mapping_csv.with_name(base + ".tif")
    png = mapping_csv.with_name(base + ".png")
    if tif.exists():
        return tif
    if png.exists():
        return png
    raise FileNotFoundError(
        f"Could not find rectified image next to mapping CSV. Looked for:\n  {tif}\n  {png}"
    )

def find_cub_for_mapping(mapping_csv: Path, framelet_dir: Path) -> Path:
    stem = mapping_csv.name.replace("_RECTIFIED_MAPPING.csv", "")
    exact = framelet_dir / f"{stem}.cub"

    if exact.exists():
        return exact

    matches = sorted(framelet_dir.glob(stem + "*.cub"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No framelet .cub found for {stem} in {framelet_dir}"
    )


# ----------------------------
# Edge detection in rectified space
# ----------------------------

@dataclass
class EdgePair:
    x: int
    y_top: int
    y_bottom: int
    y_mid: int


def find_two_edges_in_column(
    col: np.ndarray,
    y_offset: int,
    min_sep: int,
) -> Optional[Tuple[int, int]]:
    """
    Given a 1D column slice of intensities col (y increasing),
    find the two strongest gradient magnitudes with a minimum separation.

    Returns (y1, y2) in absolute image coordinates (including y_offset), ordered y1 < y2.
    """
    # gradient magnitude along y
    g = np.abs(np.diff(col))
    if g.size < 2:
        return None

    i1 = int(np.argmax(g))
    # mask out neighborhood around i1 to enforce separation
    g2 = g.copy()
    lo = max(0, i1 - min_sep)
    hi = min(g2.size, i1 + min_sep + 1)
    g2[lo:hi] = -1.0

    if np.all(g2 < 0):
        return None

    i2 = int(np.argmax(g2))
    y1 = i1 + y_offset
    y2 = i2 + y_offset

    if y1 == y2:
        return None

    if y1 > y2:
        y1, y2 = y2, y1

    return y1, y2


def detect_edge_pairs(
    gray: np.ndarray,
    x_step: int = 20,
    y_margin: int = 5,
    edge_min_sep: int = 12,
) -> List[EdgePair]:
    """
    For sampled x columns, detect two limb edges (paired fragments) via gradient peaks,
    then compute the midpoint between edges.

    Returns a list of EdgePair objects.
    """
    h, w = gray.shape
    pairs: List[EdgePair] = []

    y0 = y_margin
    y1 = h - y_margin - 1
    if y1 <= y0 + 5:
        raise ValueError("Rectified image too small in Y to detect edges robustly.")

    for x in range(0, w, x_step):
        col = gray[y0:y1, x]
        edges = find_two_edges_in_column(col, y_offset=y0, min_sep=edge_min_sep)
        if edges is None:
            continue
        yt, yb = edges
        ymid = int(round((yt + yb) / 2))
        pairs.append(EdgePair(x=x, y_top=yt, y_bottom=yb, y_mid=ymid))

    return pairs


# ----------------------------
# Perpendicular sampling + mapping association
# ----------------------------

@dataclass
class SamplePoint:
    fragment_id: int
    sample_id: int
    rx: int
    ry: int
    dy: int              # signed offset from limb midpoint in rectified space
    orig_x: float
    orig_y: float
    pixel_value: float
    spice: dict          # NEW: holds CAMPT output columns for this point


def generate_perpendicular_samples(
    edge_pairs: List[EdgePair],
    mapping: Mapping,
    half_len: int = 80,
    dy_step: int = 2,
) -> List[SamplePoint]:
    """
    For each EdgePair, generate a vertical perpendicular centered at y_mid.
    Each sample point is mapped to original coords + pixel_value using mapping dict.

    Returns a flat list of SamplePoint rows (one row per red dot).
    """
    samples: List[SamplePoint] = []

    for frag_id, ep in enumerate(edge_pairs):
        sample_idx = 0
        rx = int(ep.x)
        for dy in range(-half_len, half_len + 1, dy_step):
            ry = int(ep.y_mid + dy)
            mapped = mapping.get((rx, ry))
            if mapped is None:
                continue
            ox, oy, pv = mapped
            samples.append(
                SamplePoint(
                    fragment_id=frag_id,
                    sample_id=sample_idx,
                    rx=rx,
                    ry=ry,
                    dy=int(dy),
                    orig_x=ox,
                    orig_y=oy,
                    pixel_value=pv,
                    spice={},
                )
            )
            sample_idx += 1


    return samples


# ----------------------------
# Outputs
# ----------------------------

def write_samples_csv(out_csv: Path, samples: List[SamplePoint]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    base_fields = ["fragment_id", "sample_id", "rx", "ry", "dy", "orig_x", "orig_y", "pixel_value"]

    # Union of CAMPT keys across points
    spice_keys = set()
    for s in samples:
        if isinstance(s.spice, dict):
            spice_keys.update(s.spice.keys())

    # Prefer common geometry columns first (if present)
    preferred = [
        "PlanetocentricLatitude",
        "PlanetocentricLongitude",
        "SlantDistance",
        "LocalRadius",
        "SpacecraftAltitude",
        "Phase",
        "Emission",
        "Incidence",
        "Sample",
        "Line",
    ]
    ordered_spice = [k for k in preferred if k in spice_keys] + sorted([k for k in spice_keys if k not in preferred])

    fieldnames = base_fields + ordered_spice

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for s in samples:
            row = {
                "fragment_id": s.fragment_id,
                "sample_id": s.sample_id,
                "rx": s.rx,
                "ry": s.ry,
                "dy": s.dy,
                "orig_x": s.orig_x,
                "orig_y": s.orig_y,
                "pixel_value": s.pixel_value,
            }
            if isinstance(s.spice, dict):
                for k in ordered_spice:
                    if k in s.spice:
                        row[k] = s.spice[k]
            w.writerow(row)


def save_overlay_png(
    out_png: Path,
    rect_gray: np.ndarray,
    edge_pairs: List[EdgePair],
    samples: List[SamplePoint],
    title: str = "",
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.imshow(rect_gray, cmap="gray", origin="upper")

    # Blue dots: two edge traces (paired limb fragments)
    xs = [ep.x for ep in edge_pairs]
    y_top = [ep.y_top for ep in edge_pairs]
    y_bot = [ep.y_bottom for ep in edge_pairs]
    plt.scatter(xs, y_top, s=10, c="deepskyblue", marker="o", label="Limb edge (fragment A)")
    plt.scatter(xs, y_bot, s=10, c="deepskyblue", marker="o", label="Limb edge (fragment B)")

    # Red dots: perpendicular samples
    rx = [s.rx for s in samples]
    ry = [s.ry for s in samples]
    plt.scatter(rx, ry, s=3, c="red", marker="o", label="Perpendicular samples")

    if title:
        plt.title(title)

    plt.legend(loc="upper right", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 4: Detect rectified limb edges, build perpendicular samples, export overlay + CSV."
    )
   
    p.add_argument("--x-step", type=int, default=20, help="Sample stride in x for edge detection.")
    p.add_argument("--y-margin", type=int, default=5, help="Exclude y edges when detecting gradients.")
    p.add_argument("--edge-min-sep", type=int, default=12, help="Min separation (pixels) between detected edges.")
    p.add_argument("--half-len", type=int, default=80, help="Half-length (pixels) of perpendicular sampling line.")
    p.add_argument("--dy-step", type=int, default=2, help="Pixel step between red dots along the perpendicular.")
    p.add_argument("--no-overlay", action="store_true", help="Skip saving overlay image.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present.")
    p.add_argument("--indir", required=True, help="Stage 3 directory containing *_RECTIFIED_MAPPING.csv files")

    p.add_argument("--framelet-dir", required=True, help="Stage 1 directory containing framelet .cub files")
    p.add_argument("--outdir", required=True, help="Stage 4 output directory (stage_04_perp_samples)")
    p.add_argument("--mapping", default="", help="Optional: process a single *_RECTIFIED_MAPPING.csv")
    p.add_argument("--batch", action="store_true", help="Process all *_RECTIFIED_MAPPING.csv files in --indir")

    return p.parse_args()

def iter_mapping_csvs(indir: Path):
    return sorted(indir.glob("*_RECTIFIED_MAPPING.csv"))

def run_campt_cli(cub_path: Path, samples: List[SamplePoint]) -> pd.DataFrame:
    """
    Run ISIS campt (CLI) on red-dot sample points and return flat CSV as DataFrame.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        coordlist = td / "coordlist.csv"
        out_csv = td / "campt_out.csv"

        # coordlist format: sample,line  (NO HEADER, EXACTLY TWO COLUMNS)
        with coordlist.open("w", newline="") as f:
            w = csv.writer(f)
            for s in samples:
                w.writerow([f"{s.orig_x:.10f}", f"{s.orig_y:.10f}"])


        cmd = [
            "campt",
            f"from={cub_path}",
            f"to={out_csv}",
            "usecoordlist=true",
            "coordtype=image",
            f"coordlist={coordlist}",
            "format=flat",
        ]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"campt failed\nCMD: {' '.join(cmd)}\nSTDERR:\n{r.stderr}"
            )

        # First: read header only to get the true column count
        with out_csv.open("r") as f:
            header_line = f.readline().rstrip("\n")
            ncols = header_line.count(",") + 1

        # Now read exactly that many columns from every row
        return pd.read_csv(
            out_csv,
            sep=",",
            engine="c",
            header=0,
            usecols=list(range(ncols)),
        )

def attach_campt_df(samples: List[SamplePoint], campt_df: pd.DataFrame) -> None:
    """
    Attach CAMPT rows to SamplePoint.spice by strict positional order.
    Requires 1:1 row correspondence — enforced upstream.
    """
    for i, s in enumerate(samples):
        s.spice = campt_df.iloc[i].to_dict()


def process_one(mapping_csv, args, outdir: Path, framelet_dir: Path):
    rect_img_path = find_rectified_image(mapping_csv)

    stem = mapping_csv.name.replace("_RECTIFIED_MAPPING.csv", "")
    out_csv = outdir / f"{stem}_RECTIFIED_PERP_SAMPLES.csv"
    out_png = outdir / f"{stem}_RECTIFIED_PERP_OVERLAY.png"


    if (out_csv.exists() or out_png.exists()) and (not args.overwrite):
        print(f"[SKIP] {stem} (outputs exist)")
        return

    print(f"\n[Stage 4] Mapping CSV:   {mapping_csv.name}")
    print(f"[Stage 4] Rectified img: {rect_img_path.name}")

    rect_gray = read_image_grayscale(rect_img_path)

    print("[Stage 4] Loading mapping CSV...")
    mapping = load_rectified_mapping(mapping_csv)
    print(f"[Stage 4] Mapping loaded: {len(mapping):,} entries")

    print("[Stage 4] Detecting paired limb edges in rectified space...")
    edge_pairs = detect_edge_pairs(
        rect_gray,
        x_step=args.x_step,
        y_margin=args.y_margin,
        edge_min_sep=args.edge_min_sep,
    )
    print(f"[Stage 4] Edge pairs found: {len(edge_pairs)}")

    if not edge_pairs:
        print(f"[WARN] No edge pairs detected for {stem}. Try tuning --edge-min-sep / --x-step.")
        return

    print("[Stage 4] Generating perpendicular samples + mapping association...")
    samples = generate_perpendicular_samples(
        edge_pairs=edge_pairs,
        mapping=mapping,
        half_len=args.half_len,
        dy_step=args.dy_step,
    )
    
    print(f"[Stage 4] Sample points: {len(samples)}")

    print("[Stage 4] Finding .cub and running CAMPT for red-dot pixels...")
    cub_path = find_cub_for_mapping(mapping_csv, framelet_dir)

    campt_df = run_campt_cli(cub_path, samples)

    # HARD SAFETY CHECK — NO ALIGNMENT IF COUNTS DIFFER
    if len(campt_df) != len(samples):
        raise RuntimeError(
            f"CAMPT returned {len(campt_df)} rows for {len(samples)} samples — alignment unsafe"
        )

    attach_campt_df(samples, campt_df)

    print(f"[Stage4] CAMPT attached: {len(samples)} points")
    
    write_samples_csv(out_csv, samples)
    print(f"[Stage 4] Wrote samples CSV: {out_csv.name}")



    if not args.no_overlay:
        save_overlay_png(
            out_png,
            rect_gray,
            edge_pairs,
            samples,
            title=stem,
        )
        print(f"[Stage 4] Wrote overlay PNG: {out_png.name}")

def main() -> None:
    args = parse_args()

    indir = Path(args.indir).expanduser().resolve()
    framelet_dir = Path(args.framelet_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not indir.exists():
        raise FileNotFoundError(f"indir not found: {indir}")

    if not framelet_dir.exists():
        raise FileNotFoundError(f"framelet-dir not found: {framelet_dir}")

    outdir.mkdir(parents=True, exist_ok=True)


    # Choose which mapping CSVs to process
    if args.mapping:
        mapping_csvs = [Path(args.mapping).expanduser().resolve()]
    else:
        mapping_csvs = iter_mapping_csvs(indir)
        if not mapping_csvs:
            raise FileNotFoundError(f"No *_RECTIFIED_MAPPING.csv found in {indir}")

        # If not batch, just take the first one as a quick smoke test
        if not args.batch:
            mapping_csvs = [mapping_csvs[0]]

    # Process
    for mapping_csv in mapping_csvs:
        try:
            process_one(mapping_csv, args, outdir, framelet_dir)
        except Exception as e:
            print(f"[error] {mapping_csv.name}: {e}")




if __name__ == "__main__":
    main()
