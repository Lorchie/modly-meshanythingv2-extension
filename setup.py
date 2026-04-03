import io
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://github.com/buaacyw/MeshAnythingV2/archive/refs/heads/main.zip"


def pip(venv: Path, *args):
    exe = venv / ("Scripts/pip.exe" if platform.system() == "Windows" else "bin/pip")
    subprocess.run([str(exe), *args], check=True)


def install_torch(venv: Path):
    # Version robuste, CPU-only (fonctionne partout)
    pkgs = ["torch==2.5.1", "torchvision==0.20.1"]
    index = "https://download.pytorch.org/whl/cpu"

    try:
        pip(venv, "install", *pkgs, "--index-url", index)
    except Exception:
        # fallback sans version exacte si jamais
        pip(venv, "install", "torch", "torchvision")


def install_meshanything(venv: Path):
    py = venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")

    site = subprocess.check_output(
        [str(py), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()

    print("[setup] Download MeshAnything...")
    data = urllib.request.urlopen(ZIP_URL).read()

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for f in z.namelist():
            if "MeshAnythingV2-main/MeshAnything/" in f:
                rel = f.split("MeshAnythingV2-main/")[1]
                p = Path(site) / rel
                if f.endswith("/"):
                    p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(z.read(f))

        Path(site, "mesh_to_pc.py").write_bytes(
            z.read("MeshAnythingV2-main/mesh_to_pc.py")
        )


def setup(python_exe: str, ext_dir: str):
    ext_dir = Path(ext_dir)
    venv = ext_dir / "venv"

    print("[setup] Using system Python:", python_exe)
    print("[setup] Extension dir:", ext_dir)

    if venv.exists():
        shutil.rmtree(venv)

    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")

    print("[setup] Installing torch...")
    install_torch(venv)

    print("[setup] Installing Python deps...")
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

    print("[setup] Installing MeshAnythingV2 code...")
    install_meshanything(venv)

    print("[setup] Done")


if __name__ == "__main__":
    # On lit le JSON depuis stdin (conforme à index.js)
    raw = sys.stdin.read().strip()
    if not raw:
        raise RuntimeError("No JSON arguments received on stdin")

    args = json.loads(raw)
    setup(
        args["python_exe"],
        args["ext_dir"],
    )