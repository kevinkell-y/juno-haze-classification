#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pjdir", required=True, help="Perijove root, e.g. data/PJ05")
    ap.add_argument("--outdir", default=None, help="Default: <pjdir>/analysis/stage_08_perijove_analysis/validation")
    return ap.parse_args()


def count_stage6_fragments(stage6_root: Path):
    csvs = sorted(stage6_root.rglob("*_STAGE6.csv"))
    n_stage6_csvs = len(csvs)

    n_stage6_fragments = 0
    n_detectable_fragments = 0

    for p in csvs:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue

        if "fragment_id" not in df.columns:
            continue

        for frag_id, frag in df.groupby("fragment_id"):
            n_stage6_fragments += 1

            if "detectability_flag" in frag.columns:
                val = frag["detectability_flag"].iloc[0]
                is_det = str(val).strip().lower() in {"true", "1", "yes"}
                if is_det:
                    n_detectable_fragments += 1

    return n_stage6_csvs, n_stage6_fragments, n_detectable_fragments


def main():
    args = parse_args()
    pjdir = Path(args.pjdir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else (pjdir / "analysis" / "stage_08_perijove_analysis" / "validation")
    outdir.mkdir(parents=True, exist_ok=True)

    analysis_dir = pjdir / "analysis"
    cub_dir = pjdir / "cub"

    stage8_dir = pjdir / "analysis" / "stage_08_perijove_analysis"
    stage8_raw = stage8_dir / "stage8_fragments_all_raw.csv"
    stage8_valid = stage8_dir / "stage8_fragments_all.csv"

    df_raw = pd.read_csv(stage8_raw, low_memory=False) if stage8_raw.exists() else pd.DataFrame()
    df_valid = pd.read_csv(stage8_valid, low_memory=False) if stage8_valid.exists() else pd.DataFrame()

    records = []

    img_dirs = sorted([p for p in cub_dir.iterdir() if p.is_dir() and p.name.startswith("JNCR_")])
    for imgdir in img_dirs:
        img_id = imgdir.name

        stage6_root = imgdir / "stage_06_peaks"
        stage7_csv = analysis_dir / img_id / "stage_07" / "stage7_fragment_table.csv"

        n_stage6_csvs, n_stage6_fragments, n_detectable_fragments = count_stage6_fragments(stage6_root) if stage6_root.exists() else (0, 0, 0)

        n_stage7_rows = 0
        if stage7_csv.exists():
            try:
                n_stage7_rows = max(sum(1 for _ in open(stage7_csv, "r", encoding="utf-8")) - 1, 0)
            except Exception:
                n_stage7_rows = 0

        n_raw_rows = int((df_raw["img_id"] == img_id).sum()) if "img_id" in df_raw.columns else 0
        n_valid_rows = int((df_valid["img_id"] == img_id).sum()) if "img_id" in df_valid.columns else 0

        contributes_to_stage8 = n_valid_rows > 0

        records.append({
            "img_id": img_id,
            "n_stage6_csvs": n_stage6_csvs,
            "n_stage6_fragments": n_stage6_fragments,
            "n_detectable_fragments": n_detectable_fragments,
            "n_stage7_rows": n_stage7_rows,
            "n_stage8_raw_rows": n_raw_rows,
            "n_stage8_valid_rows": n_valid_rows,
            "contributes_to_stage8": contributes_to_stage8,
        })

    out = pd.DataFrame(records).sort_values("img_id")
    out.to_csv(outdir / "validation1_img_inventory.csv", index=False)

    # Improved publication-style validation plot
    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(out))

    # Bars = Stage 6 fragment inventory
    ax.bar(
        x,
        out["n_stage6_fragments"],
        width=0.85,
        alpha=0.85,
        label="Stage 6 fragments",
        zorder=1,
    )

    # Line = retained Stage 7 rows
    ax.plot(
        x,
        out["n_stage7_rows"],
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="Stage 7 retained fragments",
        zorder=3,
    )

    # Highlight IMG entries contributing to Stage 8
    contributes = out["contributes_to_stage8"].astype(bool).to_numpy()

    ax.scatter(
        x[contributes],
        out.loc[contributes, "n_stage7_rows"],
        s=45,
        zorder=4,
        label="Contributes to Stage 8",
    )

    # Cleaner x-axis labels
    labels = out["img_id"].tolist()

    step = 4  # show every 4th label
    tick_idx = np.arange(0, len(labels), step)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [labels[i] for i in tick_idx],
        rotation=90,
        fontsize=9,
    )

    ax.set_ylabel("Fragment count")
    ax.set_xlabel("JunoCam IMG product")
    ax.set_title(
        "Validation 1 — Per-image retained fragment accounting",
        fontsize=18,
        pad=12,
    )

    # Gridlines improve readability enormously
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
        zorder=0,
    )

    # Better legend placement
    ax.legend(
        loc="upper left",
        frameon=True,
    )

    # Slight margins
    ax.margins(x=0.01)

    ax.set_axisbelow(True)
    
    fig.tight_layout()

    fig.savefig(
        outdir / "validation1_img_inventory.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Wrote: {outdir / 'validation1_img_inventory.csv'}")
    print(f"Wrote: {outdir / 'validation1_img_inventory.png'}")


if __name__ == "__main__":
    main()
