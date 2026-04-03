import sys
import json
import subprocess
import os
import shutil
from pathlib import Path

def pip(venv: Path, *args: str) -> None:
    exe = venv / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    subprocess.check_call([str(exe), *args])

def main(args):
    python_exe = args["python_exe"]   # Python fourni par Modly (on ne vérifie plus la version)
    ext_dir    = Path(args["ext_dir"])
    venv       = ext_dir / "venv"

    print("[setup] Using Python:", python_exe)
    print("[setup] Extension dir:", ext_dir)

    if venv.exists():
        shutil.rmtree(venv)

    # Crée un venv à partir du Python fourni
    subprocess.check_call([python_exe, "-m", "venv", str(venv)])

    # Upgrade pip de base
    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")

    # PyTorch GPU — CUDA 12.4
    print("[setup] Installing PyTorch (CUDA 12.4)…")
    pip(
        venv,
        "install",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    )

    # Dépendances générales
    print("[setup] Installing core deps…")
    pip(
        venv,
        "install",
        "numpy",
        "trimesh",
        "pymeshlab",
        "huggingface_hub",
        "safetensors",
        "einops",
        "transformers>=4.46.0",
        "accelerate",
        "mesh2sdf",
        "scikit-image",
    )

    # MeshAnythingV2 depuis GitHub
    print("[setup] Installing MeshAnythingV2…")
    pip(venv, "install", "git+https://github.com/buaacyw/MeshAnythingV2.git")

    print("[setup] Done")

if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        raise RuntimeError("No JSON args on stdin")
    main(json.loads(raw))