import sys
import json
import subprocess
import os
import shutil
from pathlib import Path

def run(cmd):
    print("[setup] Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def pip(venv: Path, *args: str) -> None:
    exe = venv / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    run([str(exe), *args])

def main(args):
    python_exe = args["python_exe"]   # Python fourni par Modly
    ext_dir    = Path(args["ext_dir"])
    venv       = ext_dir / "venv"

    print("[setup] Using Python:", python_exe)
    print("[setup] Extension dir:", ext_dir)

    # Créer le venv si absent
    if not venv.exists():
        print("[setup] Creating virtual environment…")
        run([python_exe, "-m", "venv", str(venv)])
    else:
        print("[setup] Virtual environment already exists.")

    # IMPORTANT : ne pas mettre à jour pip → Windows bloque
    print("[setup] Installing PyTorch (CUDA 12.4)…")
    pip(
        venv,
        "install",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    )

    print("[setup] Installing core dependencies…")
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

    print("[setup] Installing MeshAnythingV2 from GitHub…")
    pip(venv, "install", "git+https://github.com/buaacyw/MeshAnythingV2.git")

    print("[setup] Done.")

if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        raise RuntimeError("No JSON args on stdin")
    main(json.loads(raw))