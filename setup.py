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


def install_torch(venv, pkgs, index):
    try:
        pip(venv, "install", *pkgs, "--index-url", index)
    except:
        pip(venv, "install", *(p.split("==")[0] for p in pkgs))


def install_meshanything(venv):
    py = venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")

    site = subprocess.check_output(
        [str(py), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True
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


def setup(python_exe, ext_dir, gpu_sm, cuda_version=0):
    ext_dir = Path(ext_dir)
    venv = ext_dir / "venv"

    print("[setup] Using Modly Python:", python_exe)

    if venv.exists():
        shutil.rmtree(venv)

    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    pip(venv, "install", "--upgrade", "pip", "setuptools", "wheel")

    # torch
    if gpu_sm >= 100 or cuda_version >= 128:
        pkgs = ["torch==2.7.0", "torchvision==0.22.0"]
        idx = "https://download.pytorch.org/whl/cu128"
    elif gpu_sm >= 70:
        pkgs = ["torch==2.6.0", "torchvision==0.21.0"]
        idx = "https://download.pytorch.org/whl/cu124"
    else:
        pkgs = ["torch==2.5.1", "torchvision==0.20.1"]
        idx = "https://download.pytorch.org/whl/cu118"

    install_torch(venv, pkgs, idx)

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

    install_meshanything(venv)

    print("[setup] Done")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            args["python_exe"],
            args["ext_dir"],
            args.get("gpu_sm", 86),
            args.get("cuda_version", 0),
        )