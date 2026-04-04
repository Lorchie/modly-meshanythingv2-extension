import os
import subprocess
import sys
from pathlib import Path
import json

def run(cmd, cwd=None):
    print("[vendor] Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)

def pip(venv, *args):
    exe = venv / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
    run([str(exe), *args])

def main():
    # Modly envoie les arguments JSON via stdin
    raw = sys.stdin.read().strip()
    if not raw:
        raise RuntimeError("No JSON args received from Modly")

    args = json.loads(raw)
    ext_dir = Path(args["ext_dir"])
    python_exe = args["python_exe"]

    print("[vendor] Extension directory:", ext_dir)
    print("[vendor] Python executable:", python_exe)

    # Création du venv
    venv = ext_dir / "venv"
    if not venv.exists():
        print("[vendor] Creating virtual environment…")
        run([python_exe, "-m", "venv", str(venv)])
    else:
        print("[vendor] Virtual environment already exists.")

    # Installation PyTorch CPU
    pip(
        venv,
        "install",
        "torch==2.1.1+cpu",
        "--index-url",
        "https://download.pytorch.org/whl/cpu"
    )

    # Dépendances MeshAnythingV2
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

    # Clone du repo MeshAnythingV2
    repo_dir = ext_dir / "MeshAnythingV2"
    if not repo_dir.exists():
        print("[vendor] Cloning MeshAnythingV2 repository…")
        run(["git", "clone", "https://github.com/buaacyw/MeshAnythingV2.git", str(repo_dir)])
    else:
        print("[vendor] MeshAnythingV2 repo already exists.")

    print("[vendor] Build complete.")

if __name__ == "__main__":
    main()