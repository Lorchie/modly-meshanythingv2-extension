import sys
import json
import subprocess
import os
import shutil
from pathlib import Path

def pip(venv, *args):
    exe = venv / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    subprocess.check_call([str(exe), *args])

def install_meshanything(venv):
    """Installe MeshAnythingV2 depuis HuggingFace."""
    pip(venv, "install", "git+https://github.com/buaacyw/MeshAnythingV2.git")

def main(args):
    python_exe = args["python_exe"]
    ext_dir    = Path(args["ext_dir"])
    venv       = ext_dir / "venv"

    print("[setup] Using Python:", python_exe)
    print("[setup] Extension directory:", ext_dir)

    if venv.exists():
        shutil.rmtree(venv)

    subprocess.check_call([python_exe, "-m", "venv", str(venv)])

    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")

    # Torch CPU = compatible partout
    pip(
        venv,
        "install",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "--index-url",
        "https://download.pytorch.org/whl/cpu"
    )

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
        "scikit-image"
    )

    install_meshanything(venv)

    print("[setup] Done")

if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        raise RuntimeError("No JSON received on stdin")
    main(json.loads(raw))