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
    python_exe = args["python_exe"]
    ext_dir    = Path(args["ext_dir"])
    venv       = ext_dir / "venv"

    print("[setup] Using Python:", python_exe)
    print("[setup] Extension dir:", ext_dir)

    # Create venv
    if not venv.exists():
        print("[setup] Creating virtual environment…")
        run([python_exe, "-m", "venv", str(venv)])
    else:
        print("[setup] Virtual environment already exists.")

    # Install Torch CPU
    pip(
        venv,
        "install",
        "torch==2.11.0+cpu",
        "--index-url",
        "https://download.pytorch.org/whl/cpu"
    )

    # Core dependencies
    pip(
        venv,
        "install",
        "numpy",
        "scipy",
        "scikit-image",
        "einops",
        "trimesh",
        "mesh2sdf",
        "huggingface_hub",
        "safetensors",
        "accelerate",
        "transformers>=4.46.0"
    )

    # Clone MeshAnythingV2 repo
    repo_dir = ext_dir / "MeshAnythingV2"
    if not repo_dir.exists():
        run(["git", "clone", "https://github.com/buaacyw/MeshAnythingV2.git", str(repo_dir)])

    print("[setup] Done.")

if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        raise RuntimeError("No JSON args on stdin")
    main(json.loads(raw))