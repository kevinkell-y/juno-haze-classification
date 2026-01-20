#!/usr/bin/env python3
# 03.rectify_limb.py — rectifies Stage-2 limb polylines into a tangent-plane strip.
# Inputs:
#   Stage-2 limb polylines (*_LIMBENDPOINTS.csv)
#   Stage-1 framelet radiance images (*.tif / *.png)
# Outputs:
#   Stage-3 rectified limb strips and mapping tables

import sys, glob, csv, math
from pathlib import Path
import numpy as np
from PIL import Image
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------- I/O ----------------

def load_limb_points(csv_file: Path):
    pts = []
    with open(csv_file, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)  # skip header if present
        for row in r:
            if len(row) < 2: 
                continue
            try:
                x, y = float(row[0]), float(row[1])
                pts.append((x, y))
            except ValueError:
                pass
    # drop duplicates
    dedup = []
    for p in pts:
        if not dedup or (abs(p[0]-dedup[-1][0]) > 1e-6 or abs(p[1]-dedup[-1][1]) > 1e-6):
            dedup.append(p)
    print(f"✅ Loaded {len(dedup)} limb points from {csv_file.name}")
    return dedup

def load_image_16bit(path: Path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:  # take first channel if RGB
        arr = arr[:, :, 0]
    if arr.dtype == np.uint8:
        arr = (arr.astype(np.uint16) << 8)  # promote 8→16
    elif arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr

def save_16bit_tif(arr16: np.ndarray, path: Path):
    Image.fromarray(arr16, mode="I;16").save(path, "TIFF")

def save_8bit_png(arr16: np.ndarray, path: Path):
    arr8 = (arr16 >> 8).astype(np.uint8)
    Image.fromarray(arr8).save(path)

# --------------- Geometry helpers ---------------

def bilinear(arr, x, y):
    H, W = arr.shape
    if x < 0 or y < 0 or x >= W-1 or y >= H-1:
        return 0
    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0
    v00 = arr[y0,   x0  ]
    v10 = arr[y0,   x0+1]
    v01 = arr[y0+1, x0  ]
    v11 = arr[y0+1, x0+1]
    return (v00*(1-dx)*(1-dy) + v10*dx*(1-dy) + v01*(1-dx)*dy + v11*dx*dy)

def cumulative_arclen(points):
    s = [0.0]
    for i in range(1, len(points)):
        x0,y0 = points[i-1]; x1,y1 = points[i]
        s.append(s[-1] + math.hypot(x1-x0, y1-y0))
    return s

def resample_polyline(points, spacing=8.0, trim_end=0.0):
    """
    Resample polyline at ~uniform spacing along arc length.
    trim_end: trim this much arc length from each end (px) after resampling.
    """
    if len(points) < 2:
        return points[:]
    s = cumulative_arclen(points)
    total = s[-1]
    if total < 1e-6:
        return points[:]
    # targets
    t0 = trim_end
    t1 = max(0.0, total - trim_end)
    if t1 - t0 < spacing:
        # nothing to do; return original
        return points[:]
    targets = np.arange(t0, t1 + 0.5*spacing, spacing)

    # piecewise linear interpolation
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    s_arr = np.array(s, dtype=float)

    out = []
    j = 0
    for st in targets:
        while j+1 < len(points) and s_arr[j+1] < st:
            j += 1
        if j+1 == len(points):
            out.append(points[-1]); break
        # interpolate between j and j+1
        t = 0.0 if s_arr[j+1] == s_arr[j] else (st - s_arr[j])/(s_arr[j+1]-s_arr[j])
        x = xs[j] + t*(xs[j+1]-xs[j])
        y = ys[j] + t*(ys[j+1]-ys[j])
        out.append((float(x), float(y)))
    return out

# --------------- Rectification ---------------

def rectify_limb(arr, limb_points, x_step=20, y_step=96, spacing=8.0, trim_px=8.0, radius=12.0, mapping_csv: Path|None=None):
    """
    Rectify into a horizontal strip.
    spacing : target spacing (px) between consecutive resampled limb points
    trim_px : clip ends after resampling (arc-length px) to avoid edge artifacts
    radius  : half-width across the normal (px) for strip thickness geometry
    """
    print(f"Initial limb points: {len(limb_points)}")
    if len(limb_points) < 2:
        print("❌ Not enough points.")
        return None

    # Resample to uniform spacing so segments aren't ~2 px apart
    rs = resample_polyline(limb_points, spacing=spacing, trim_end=trim_px)
    print(f"Resampled limb points: {len(rs)} (spacing≈{spacing}px, trim={trim_px}px)")
    if len(rs) < 2:
        print("❌ Not enough resampled points.")
        return None

    H, W = arr.shape
    width  = x_step * (len(rs)-1)
    height = int(y_step)
    rect = np.zeros((height, width), dtype=np.uint16)

    writer = None
    if mapping_csv is not None:
        mf = open(mapping_csv, "w", newline="")
        writer = csv.writer(mf)
        writer.writerow(["rect_x","rect_y","orig_x","orig_y","pixel_value"])

    used = 0
    for i in range(1, len(rs)):
        x1,y1 = rs[i-1]; x2,y2 = rs[i]
        dx, dy = x2-x1, y2-y1
        seglen = math.hypot(dx, dy)
        if seglen < 1.0:  # ultra-short after resample? skip
            continue

        # tangent and normal
        angle = math.atan2(dy, dx)
        nx, ny = -math.sin(angle), math.cos(angle)

        # normal band (p1↔p2) at first endpoint; along-limb (p1→p3) at second endpoint
        p1x, p1y = x1 - radius*nx, y1 - radius*ny
        p2x, p2y = x1 + radius*nx, y1 + radius*ny
        p3x, p3y = x2 - radius*nx, y2 - radius*ny

        # Bounds check
        if not (0 <= p1x < W and 0 <= p2x < W and 0 <= p3x < W): 
            continue
        if not (0 <= p1y < H and 0 <= p2y < H and 0 <= p3y < H): 
            continue

        col_dx, col_dy = p2x - p1x, p2y - p1y
        row_dx, row_dy = p3x - p1x, p3y - p1y

        for xi in range(x_step):
            sx = p1x + (xi / x_step) * row_dx
            sy = p1y + (xi / x_step) * row_dy
            for yi in range(height):
                px = sx + (yi / height) * col_dx
                py = sy + (yi / height) * col_dy
                val = int(bilinear(arr, px, py))

                rx = used * x_step + xi
                ry = height - 1 - yi  # flip so "up" indexes outward if desired
                rect[ry, rx] = val
                if writer:
                    writer.writerow([rx, ry, px, py, val])

        used += 1

    if writer:
        mf.close()

    print(f"✅ Segments used: {used}")
    rect = rect[:, :used * x_step]
    if rect.size == 0:
        return None
    print("Final output shape:", rect.shape)
    return rect

# --------------- Per-file workflow ---------------

def process_csv(csv_path: Path, image_dir: Path, outdir: Path,
                x_step=20, y_step=96, spacing=8.0, trim_px=8.0, radius=12.0):
    stem = csv_path.name.replace("_LIMBENDPOINTS.csv", "")

    candidates = [
        image_dir / f"{stem}.tif",
        image_dir / f"{stem}.png",
    ]

    img_path = next((p for p in candidates if p.exists()), None)
    if img_path is None:
        raise FileNotFoundError(
            f"No image found for {csv_path.name} in {image_dir}"
        )
    
    arr = load_image_16bit(img_path)
    print(f"TIFF/PNG: {img_path.name}  shape={arr.shape}  dtype={arr.dtype}  range={arr.min()}–{arr.max()}")

    limb_pts = load_limb_points(csv_path)

    stem = csv_path.name.replace("_LIMBENDPOINTS.csv", "")
    rect_tif = outdir / f"{stem}_RECTIFIED.tif"
    rect_png = outdir / f"{stem}_RECTIFIED.png"
    map_csv  = outdir / f"{stem}_RECTIFIED_MAPPING.csv"

    rect = rectify_limb(arr, limb_pts,
                        x_step=x_step, y_step=y_step,
                        spacing=spacing, trim_px=trim_px, radius=radius,
                        mapping_csv=map_csv)
    if rect is None or rect.size == 0:
        print(f"❌ No rectified output for {csv_path.name}")
        return

    save_16bit_tif(rect, rect_tif)
    save_8bit_png(rect, rect_png)
    print(f"✅ Wrote {rect_tif.name} and {rect_png.name}")

# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 03 — rectify limb polylines into tangent-plane strips"
    )
    parser.add_argument(
        "--indir",
        required=True,
        help="Input directory containing *_LIMBENDPOINTS.csv from Stage 2"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for rectified limb products (stage_03_rectified)"
    )
    parser.add_argument(
        "--x-step", type=int, default=20
    )
    parser.add_argument(
        "--y-step", type=int, default=96
    )
    parser.add_argument(
        "--spacing", type=float, default=8.0
    )
    parser.add_argument(
        "--trim-px", type=float, default=8.0
    )
    parser.add_argument(
        "--radius", type=float, default=12.0
    )
    parser.add_argument(
        "--imagedir", required=True, help="Directory containing original framelet images from Stage 1"
    )

    args = parser.parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()

    if not indir.exists():
        raise FileNotFoundError(f"Input directory not found: {indir}")

    imagedir = Path(args.imagedir).resolve()
    if not imagedir.exists():
        raise FileNotFoundError(f"Image directory not found: {imagedir}")

    outdir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(indir.glob("*_LIMBENDPOINTS.csv"))
    if not csvs:
        raise RuntimeError(f"No *_LIMBENDPOINTS.csv found in {indir}")

    for csv_path in csvs:
        try:
            process_csv(
                csv_path,
                image_dir=imagedir,
                outdir=outdir,
                x_step=args.x_step,
                y_step=args.y_step,
                spacing=args.spacing,
                trim_px=args.trim_px,
                radius=args.radius
            )
        except Exception as e:
            print(f"[error] {csv_path.name}: {e}")


if __name__ == "__main__":
    main()
