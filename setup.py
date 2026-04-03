import sys
import json
import subprocess
import os
from pathlib import Path

def run(cmd):
    print("[setup] Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def pip(venv: Path, *args: str) -> None:
    exe = venv / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    run([str(exe), *args])

def main(args):
    python_exe = args["python_exe"]   # Python fourni par Modly (3.14)
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

    # Installer PyTorch CPU (compatible Python 3.14)
    print("[setup] Installing PyTorch CPU (Python 3.14 compatible)…")
    pip(
        venv,
        "install",
        "torch==2.11.0+cpu",
        "--index-url",
        "https://download.pytorch.org/whl/cpu"
    )

    # Dépendances MeshAnythingV2
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

    # MeshAnythingV2 depuis GitHub
    print("[setup] Installing MeshAnythingV2 from GitHub…")
    pip(venv, "install", "git+https://github.com/buaacyw/MeshAnythingV2.git")

    print("[setup] Done.")

if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        raise RuntimeError("No JSON args on stdin")
    main(json.loads(raw))