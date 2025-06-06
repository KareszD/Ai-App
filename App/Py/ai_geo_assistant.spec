# ai_geo_assistant.spec
# -*- mode: python -*-

import os
import sys
from pathlib import Path
import sysconfig

# Import PyInstaller hooks
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# ─── 1) Locate project_root and requirements.txt ─────────────────────────────
# We assume you run `pyinstaller ai_geo_assistant.spec` from “Ai-App\App\Py”
#
# Therefore os.getcwd() == "…\Ai-App\App\Py"
project_root = Path(os.getcwd())
reqs_path    = project_root / "requirements.txt"

if not reqs_path.exists():
    raise FileNotFoundError(f"requirements.txt not found at {reqs_path!r}")

# ─── 2) Parse requirements.txt into top-level package names ──────────────────
pkgs = []
for line in reqs_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    # Remove version “pins” (==, >=, >)
    name = line.split("==")[0].split(">=")[0].split(">")[0]
    pkgs.append(name)

# ─── 3) hidden_imports: collect every submodule for each package ──────────────
hidden_imports = []
for pkg in pkgs:
    hidden_imports += collect_submodules(pkg)

# If there are any other dynamic imports you know (e.g. setproctitle),
# you can append them here:
# hidden_imports.append("setproctitle")

# ─── 4) Bundle Data/ and JB/ folders into datas ───────────────────────────────
def walk_folder(folder_path: Path):
    """
    Recursively collect (src_abs_path, dest_relative_path) for all files under
    folder_path, so PyInstaller can copy them into the frozen app under <app>\<dest>.
    """
    out = []
    base = Path(folder_path)
    for f in base.rglob("*"):
        if f.is_file():
            src = str(f.resolve())
            # If f = “Ai-App\App\Data\labels.json”, then
            # f.relative_to(base.parent) == “Data\labels.json”
            dst = str(f.relative_to(base.parent))
            out.append((src, dst))
    return out

datas = []
datas += walk_folder(project_root / "Data")
datas += walk_folder(project_root / "JB")

# ─── 5) Bundle the entire stdlib “encodings” folder from THE ACTIVE PYTHON ────
# By using sysconfig.get_paths()["stdlib"], we locate the “Lib” directory of
# whichever interpreter is currently running—so if you activated .venv, it points
# to “Ai-App\App\.venv\Lib”. Then “encodings” lives under there.
# collect_data_files("encodings") returns a list of (src, dst) pairs automatically.
stdlib_path    = Path(sysconfig.get_paths()["stdlib"])   # e.g. “…\.venv\Lib”
encodings_data = collect_data_files("encodings")

# Append those pairs to datas
datas += encodings_data

# ─── 6) Build Analysis / PYZ / EXE / COLLECT ──────────────────────────────────
block_cipher = None

a = Analysis(
    scripts       = ["api.py"],            # your Flask entry‐point (flips open later)
    pathex        = [str(project_root)],   # → Ai-App\App\Py
    binaries      = [],                    # binaries get collected by COLLECT
    datas         = datas,                 # Data/, JB/, and encodings/…
    hiddenimports = hidden_imports,        # all submodules of each requirements.txt pkg
    hookspath     = [],
    runtime_hooks = [],
    excludes      = [],                    # if you want to explicitly exclude a package
    cipher        = block_cipher,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=False,  # keep default; COLLECT will gather binaries
    noarchive=True,          # <— leave stdlib unpacked on disk, not zipped
    name      = "ai_geo_assistant",
    debug     = False,
    strip     = False,
    upx       = True,
    console   = True,        # show console for errors
)

coll = COLLECT(
    exe,
    a.binaries,   # all .pyd/.dll files end up here next to the EXE
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="ai_geo_assistant"
)
