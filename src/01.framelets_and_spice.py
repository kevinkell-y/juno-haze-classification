#!/usr/bin/env python3
"""
01.framelets-and-spice.py — Stage 1 (offline SPICE + manifests)
------------------------------------------------------------
Programmer: Kevin Kelly
Date: November 8, 2025
Organization: NASA Jet Propulsion Laboratory
Mission: Juno
Group: Planetary Science 322

- IMG/LBL -> framelet CUBs
- SPICE attach (offline)
- Per-cube manifest with expanded absolute kernel paths
"""

from pathlib import Path
import os, sys, glob, shutil, subprocess, re
import argparse

# Project layout: data/raw/*.{IMG,LBL}  ->  data/cub/*.cub
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT]

# Ensure ISIS in PATH and ISIS envs exist (don’t overwrite if already set)
ENV = os.environ.copy()

if "ISISROOT" not in ENV or "ISISDATA" not in ENV:
    sys.exit(
        "ISIS environment not detected.\n"
        "Activate your ISIS conda environment before running:\n"
        "  conda activate isis-8.3.0"
    )

isis_bin = Path(ENV["ISISROOT"]) / "bin"
if isis_bin.as_posix() not in ENV.get("PATH", ""):
    ENV["PATH"] = f"{isis_bin}:{ENV.get('PATH','')}"

def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    print(">>", " ".join(cmd))
    return subprocess.run(cmd, env=ENV, text=True, capture_output=True, check=check)

def die(msg, cp: subprocess.CalledProcessError | None = None):
    if cp is not None:
        print("\nSTDOUT:\n" + cp.stdout, file=sys.stderr)
        print("\nSTDERR:\n" + cp.stderr, file=sys.stderr)
    sys.exit(msg if isinstance(msg, int) else 1)

def find_label(img_base: str) -> Path:
    for d in SEARCH_DIRS:
        p = Path(d) / f"{img_base}.LBL"
        if p.exists():
            return p
    print(f"Could not find {img_base}.LBL in {', '.join(map(str, SEARCH_DIRS))}", file=sys.stderr)
    sys.exit(1)

def junocam2isis(lbl_path: Path, out_dir: Path, img_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cub = out_dir / f"{img_name}.cub"
    try:
        run(["junocam2isis", f"from={lbl_path}", f"to={base_cub}"])
    except subprocess.CalledProcessError as e:
        die("junocam2isis failed", e)

def list_framelets(out_dir: Path, img_name: str) -> list[Path]:
    pattern = out_dir / f"{img_name}_*.cub"
    cubs = sorted(Path(p) for p in glob.glob(str(pattern)))
    if not cubs:
        print(f"WARNING: no framelet cubes found with {pattern}")
    return cubs

def spiceinit_offline(cub: Path) -> None:
    # attach=true web=false ckpredicted=true (offline; uses local kernels DBs)
    try:
        run(["spiceinit", f"from={cub}", "attach=true", "web=false", "ckpredicted=true"])
    except subprocess.CalledProcessError as e:
        die(f"spiceinit failed on {cub.name}", e)

def extract_kernel_block(cub: Path) -> str:
    # Use catlab to print PVL, then slice Kernels group
    try:
        cp = run(["catlab", f"from={cub}"])
    except subprocess.CalledProcessError as e:
        die(f"catlab failed on {cub.name}", e)
    txt = cp.stdout
    start = txt.find("Group = Kernels")
    end   = txt.find("End_Group", start)
    return "" if start < 0 or end < 0 else txt[start:end+len("End_Group")] + "\n"

def expand_env_vars(pvl: str) -> str:
    # Replace $base, $juno, $ISISDATA etc. with absolute paths from ENV
    def repl(m):
        name = m.group(1)
        return os.path.abspath(ENV.get(name, m.group(0)))
    return re.sub(r"\$([A-Za-z_]+)", repl, pvl)

def write_manifest(cub: Path, pvl: str) -> Path:
    out = cub.with_name(cub.stem + "_kernels.txt")
    out.write_text(pvl)
    return out

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1 — JunoCam framelet ingestion + SPICE"
    )
    p.add_argument(
        "--img",
        required=True,
        help="Path to JunoCam .IMG file (with .LBL alongside)",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory for stage-01 framelet cubes (default: "
            "data/<IMG_NAME>/cub/stage_01_framelets)"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    img_path = Path(args.img).resolve()

    if not img_path.exists():
        raise FileNotFoundError(img_path)

    IMG_NAME = img_path.stem
    
    lbl = img_path.with_suffix(".LBL")
    if not lbl.exists():
        raise FileNotFoundError(
            f"Expected label file next to IMG but did not find:\n{lbl}"
        )
    print("ISISROOT:", ENV.get("ISISROOT"))
    print("ISISDATA:", ENV.get("ISISDATA"))
    print("which junocam2isis:", shutil.which("junocam2isis", path=ENV["PATH"]))
    print("which spiceinit   :", shutil.which("spiceinit",   path=ENV["PATH"]))
    print("which catlab      :", shutil.which("catlab",      path=ENV["PATH"]))

    print(f"Found label: {lbl}")

        # Default: data/<IMG_NAME>/cub/stage_01_framelets
    if args.outdir is None:
        out_dir = PROJECT_ROOT / "data" / IMG_NAME / "cub" / "stage_01_framelets"
    else:
        out_dir = Path(args.outdir).resolve()

    print(f"Converting {IMG_NAME}.LBL/.IMG -> framelet CUBs in {out_dir} ...")
    junocam2isis(lbl, out_dir, IMG_NAME)

    cubs = list_framelets(out_dir, IMG_NAME)
    if not cubs:
        die("No framelet .cub files detected. Update out_dir/pattern if needed.")

    print(f"Found {len(cubs)} framelet cubes; running offline spiceinit + manifests ...")
    for cub in cubs:
        spiceinit_offline(cub)
        kernels = extract_kernel_block(cub)
        if kernels:
            manifest = write_manifest(cub, expand_env_vars(kernels))
            print(f"  ✓ {cub.name}  →  {manifest.name}")
        else:
            print(f"  ⚠ {cub.name}: Kernels group not found (skipped manifest)")

    print("\n✅ Stage 1 complete.")
    print("Outputs:")
    print(" - Framelet cubes:", out_dir)
    print(" - Kernel manifests: *_kernels.txt next to each cube")

if __name__ == "__main__":
    main()