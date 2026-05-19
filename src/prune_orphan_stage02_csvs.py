#!/usr/bin/env python3
from pathlib import Path
import argparse

def expected_png_for_csv(csv_path: Path) -> Path:
    name = csv_path.name

    # Your current Stage 02 naming pattern:
    # ..._LIMBENDPOINTS.csv  ->  ..._OVERLAY.png
    if name.endswith("_LIMBENDPOINTS.csv"):
        return csv_path.with_name(name.replace("_LIMBENDPOINTS.csv", "_OVERLAY.png"))

    # Fallback: same stem .png
    return csv_path.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pjdir", default="data/PJ14", help="Perijove directory")
    parser.add_argument("--delete", action="store_true", help="Actually delete orphan CSVs")
    args = parser.parse_args()

    pjdir = Path(args.pjdir)

    csvs = sorted(pjdir.glob("cub/*/stage_02_trace_polyline/*.csv"))

    orphaned = []

    for csv_path in csvs:
        png_path = expected_png_for_csv(csv_path)

        if not png_path.exists():
            orphaned.append(csv_path)
            print(f"ORPHAN CSV: {csv_path}")
            print(f"  missing PNG: {png_path.name}")

            if args.delete:
                csv_path.unlink()
                print("  deleted")

    print()
    print(f"Checked CSVs: {len(csvs)}")
    print(f"Orphan CSVs: {len(orphaned)}")

    if not args.delete:
        print()
        print("Dry run only. To actually delete:")
        print(f"  python src/prune_orphan_stage02_csvs.py --pjdir {pjdir} --delete")


if __name__ == "__main__":
    main()
