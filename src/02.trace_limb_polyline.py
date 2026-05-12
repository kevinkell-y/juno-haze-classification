#!/usr/bin/env python3
# Stage 02 — Robust limb tracing (top/bottom/left/right)
# - Seeds on interior limb (dark→bright), rejects sensor borders
# - Follows edge tangent in both directions (fast, no hangs)
# - Saves beside each .cub:
#     <stem>.tif, <stem>.png, <stem>_OVERLAY.png, <stem>_LIMBENDPOINTS.csv
# - Shows matplotlib preview unless --noshow is passed

from pathlib import Path
import sys, glob, csv, math, subprocess
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import argparse

# Project root (…/juno)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------------------------- ISIS export -------------------------

def cub_to_images(cub_path: Path):
    """
    Export next to the CUB:
      - 16-bit .tif   (science)
      - 8-bit  .png   (quicklook)
    If input is already .png, reuse it (no ISIS conversion).
    """
    if cub_path.suffix.lower() == ".png":
        tif_path = cub_path.with_suffix(".tif")
        png_path = cub_path
        return tif_path, png_path

    tif_path = cub_path.with_suffix(".tif")
    png_path = cub_path.with_suffix(".png")

    subprocess.run([
        "isis2std",
        f"from={cub_path}",
        f"to={tif_path}",
        "format=tiff",
        "bittype=U16BIT",
    ], check=True)

    subprocess.run([
        "isis2std",
        f"from={cub_path}",
        f"to={png_path}",
        "format=png",
    ], check=True)

    return tif_path, png_path

# ------------------------- Gradient field ----------------------

def edge_fields(img: Image.Image):
    """
    Return (g, gx, gy, mag) where:
      g   = grayscale float32 normalized to [0,1]
      gx, gy = central-diff gradients
      mag = sqrt(gx^2 + gy^2)
    """
    g = np.asarray(img.convert("L"), dtype=np.float32)
    maxv = g.max()
    if maxv > 0:
        g /= maxv

    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    return g, gx, gy, mag

def _bilinear(g, x, y):
    """Bilinear sample of float32 grayscale array g at (x,y)."""
    h, w = g.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = int(np.floor(x)); x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y)); y1 = min(y0 + 1, h - 1)
    dx = x - x0; dy = y - y0
    v00 = g[y0, x0]; v10 = g[y0, x1]; v01 = g[y1, x0]; v11 = g[y1, x1]
    return (v00 * (1 - dx) + v10 * dx) * (1 - dy) + (v01 * (1 - dx) + v11 * dx) * dy

# ------------------------- Seeding ------------------------------

def seed_interior_limb(g, gx, gy, mag, margin_px: int = 40,
                       step: int = 3, contrast_min: float = 0.05):
    """
    Interior seeding: scan inside the frame for a strong 'dark→bright' edge
    (space darker than planet). Reject near-purely vertical/horizontal seams.
    """
    h, w = g.shape
    probe = 3.0
    best_score = -1.0
    best_xy = (w / 2.0, h / 2.0)

    for y in range(margin_px, h - margin_px, step):
        row_gx = gx[y, :]
        row_gy = gy[y, :]
        row_mag = mag[y, :]
        for x in range(margin_px, w - margin_px, step):
            gxv = float(row_gx[x]); gyv = float(row_gy[x])
            mval = math.hypot(gxv, gyv)
            if mval < 1e-3:
                continue

            nx, ny = gxv / mval, gyv / mval  # gradient points toward brighter side
            i_plus  = _bilinear(g, x + probe * nx, y + probe * ny)
            i_minus = _bilinear(g, x - probe * nx, y - probe * ny)
            contrast = i_plus - i_minus
            if contrast < contrast_min:
                continue

            # penalize near-vertical / near-horizontal seams (sensor artifacts)
            ratio = abs(gxv) / (abs(gyv) + 1e-8)
            if ratio < 0.1 or ratio > 10.0:
                continue

            score = mval * contrast
            if score > best_score:
                best_score = score
                best_xy = (float(x), float(y))

    return best_xy

# ------------------------- Tangent tracing ----------------------

def trace_one_side(gx, gy, mag, seed_xy, sign=+1, step_px=2.0, normal_snap=2,
                   max_steps=4000, min_grad=1e-3, stall_tol=0.3):
    """
    Follow the limb along the edge *tangent* starting from seed in one direction.
    sign: +1 (forward) or -1 (reverse).
    """
    h, w = mag.shape
    pts = []
    x, y = seed_xy
    stalls = 0

    for _ in range(max_steps):
        xi, yi = int(round(x)), int(round(y))
        if xi < 1 or yi < 1 or xi >= w - 1 or yi >= h - 1:
            break

        gxv = float(gx[yi, xi])
        gyv = float(gy[yi, xi])
        m = math.hypot(gxv, gyv)
        if m < min_grad:
            stalls += 1
            if stalls >= 8:
                break
            # gentle re-seed to the strongest neighbor
            nb = mag[max(yi - 2, 0):min(yi + 3, h), max(xi - 2, 0):min(xi + 3, w)]
            if nb.size and nb.max() > m:
                oy, ox = np.unravel_index(np.argmax(nb), nb.shape)
                x = max(xi - 2, 0) + ox
                y = max(yi - 2, 0) + oy
            continue

        stalls = 0
        # tangent = gradient rotated by ±90°
        tx, ty = -gyv / m, gxv / m
        if sign < 0:
            tx, ty = -tx, -ty

        # propose step
        px, py = x + step_px * tx, y + step_px * ty

        # snap along local normal (gradient direction) for better adherence
        nx, ny = gxv / (m + 1e-8), gyv / (m + 1e-8)
        best_x, best_y, best_mag = px, py, -1.0
        for k in range(-normal_snap, normal_snap + 1):
            cx, cy = px + k * nx, py + k * ny
            cxi, cyi = int(round(cx)), int(round(cy))
            if cxi < 0 or cyi < 0 or cxi >= w or cyi >= h:
                continue
            val = mag[cyi, cxi]
            if val > best_mag:
                best_mag = val
                best_x, best_y = cx, cy

        if abs(best_x - x) < stall_tol and abs(best_y - y) < stall_tol:
            break

        x, y = best_x, best_y
        pts.append((x, y))

        if x <= 0 or y <= 0 or x >= (w - 1) or y >= (h - 1):
            break

    return pts

def trace_limb_polyline(img: Image.Image):
    """
    1) Build gradient field.
    2) Seed inside the frame on a true dark→bright limb.
    3) Trace tangent both directions and merge.
    """
    g, gx, gy, mag = edge_fields(img)
    seed = seed_interior_limb(g, gx, gy, mag)  # robust interior seeding

    fwd = trace_one_side(gx, gy, mag, seed, sign=+1)
    bwd = trace_one_side(gx, gy, mag, seed, sign=-1)

    pts = list(reversed(bwd)) + [seed] + fwd
    return pts

# ------------------------- Overlay & CSV -----------------------

def draw_dots(image: Image.Image, pts, color=(38, 247, 253), r=1):
    draw = ImageDraw.Draw(image)
    for x, y in pts:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

def save_outputs(cub_path: Path, png_path: Path, pts, outdir: Path):
    stem = cub_path.stem

    color = Image.open(png_path).convert("RGB")
    draw_dots(color, pts, color=(38, 247, 253), r=1)
    overlay_path = outdir / f"{stem}_OVERLAY.png"
    color.save(overlay_path, "PNG")

    csv_path = outdir / f"{stem}_LIMBENDPOINTS.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for x, y in pts:
            w.writerow([float(x), float(y)])

    print(f"✓ {cub_path.name} → {overlay_path.name}, {csv_path.name} (points={len(pts)})")

# ------------------------- Per-file & Main ---------------------

def process_cub(cub_path: Path, outdir: Path, show=True):
    tif_path, png_path = cub_to_images(cub_path)
    img = Image.open(png_path)
    pts = trace_limb_polyline(img)
    save_outputs(cub_path, png_path, pts, outdir)

    if show:
        # matplotlib sanity check; close window to continue
        plt.figure(figsize=(12, 2))
        plt.imshow(Image.open(png_path), cmap="gray")
        xs, ys = zip(*pts) if pts else ([], [])
        if xs:
            plt.scatter(xs, ys, s=4, c="cyan")
        plt.title(cub_path.stem)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Stage 02 — trace planetary limb polylines from framelet CUBs"
    )
    parser.add_argument(
        "--cubdir",
        required=True,
        help="Input directory of framelet CUBs (stage_01_framelets)"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for limb polylines (stage_02_trace_polyline)"
    )
    parser.add_argument(
        "--noshow",
        action="store_true",
        help="Disable matplotlib preview windows"
    )

    args = parser.parse_args()
    cubdir = Path(args.cubdir)
    outdir = Path(args.outdir)
    show = not args.noshow

    if not cubdir.exists():
        raise FileNotFoundError(f"Input cubdir not found: {cubdir}")

    outdir.mkdir(parents=True, exist_ok=True)

    cubs = sorted(cubdir.glob("*.cub"))
    if not cubs:
        raise RuntimeError(f"No .cub files found in {cubdir}")

    for cub in cubs:
        try:
            process_cub(cub, outdir=outdir, show=show)
        except Exception as e:
            print(f"[error] {cub.name}: {e}")


if __name__ == "__main__":
    main()
