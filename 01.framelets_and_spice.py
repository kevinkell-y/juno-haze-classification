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

# ---- config (edit IMG_NAME if processing a different image)
IMG_NAME = "JNCR_2018197_14C00024_V01"

# Project layout: data/raw/*.{IMG,LBL}  ->  data/cub/*.cub
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT]
CUB_DIR = PROJECT_ROOT / "data" / "cub"

# Ensure ISIS in PATH and ISIS envs exist (don’t overwrite if already set)
ENV = os.environ.copy()
ENV.setdefault("ISISROOT", "/home/user/miniconda/envs/juno-isis")
ENV.setdefault("ISISDATA", "/home/user/miniconda/envs/juno-isis/isisdata")
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

def junocam2isis(lbl_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cub = out_dir / f"{IMG_NAME}.cub"
    try:
        run(["junocam2isis", f"from={lbl_path}", f"to={base_cub}"])
    except subprocess.CalledProcessError as e:
        die("junocam2isis failed", e)

def list_framelets(out_dir: Path) -> list[Path]:
    pattern = out_dir / f"{IMG_NAME}_*.cub"
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

def main():
    print("ISISROOT:", ENV.get("ISISROOT"))
    print("ISISDATA:", ENV.get("ISISDATA"))
    print("which junocam2isis:", shutil.which("junocam2isis", path=ENV["PATH"]))
    print("which spiceinit   :", shutil.which("spiceinit",   path=ENV["PATH"]))
    print("which catlab      :", shutil.which("catlab",      path=ENV["PATH"]))

    lbl = find_label(IMG_NAME)
    print(f"Found label: {lbl}")

    print(f"Converting {IMG_NAME}.LBL/.IMG -> framelet CUBs in {CUB_DIR} ...")
    junocam2isis(lbl, CUB_DIR)

    cubs = list_framelets(CUB_DIR)
    if not cubs:
        die("No framelet .cub files detected. Update CUB_DIR/pattern if needed.")

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
    print(" - Framelet cubes:", CUB_DIR)
    print(" - Kernel manifests: *_kernels.txt next to each cube")

if __name__ == "__main__":
    main()