"""
MeshAnythingV2 — extension setup script (robust version)

Fixes:
- Handles Python 3.13+ incompatibility (auto fallback to 3.10–3.12)
- Fixes GPU SM=0 case (no forced CUDA install)
- Adds resilient PyTorch install (fallbacks)
"""

import io
import json
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

_MESHANYTHINGV2_ZIP = "https://github.com/buaacyw/MeshAnythingV2/archive/refs/heads/main.zip"


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def find_compatible_python():
    candidates = ["py -3.12", "py -3.11", "py -3.10"]
    for cmd in candidates:
        try:
            subprocess.run(
                cmd.split() + ["-c", "import sys"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return cmd.split()
        except:
            continue
    raise RuntimeError("No compatible Python (3.10–3.12) found.")


# ------------------------------------------------------------
# PyTorch install (robust)
# ------------------------------------------------------------
def install_torch(venv: Path, torch_pkgs: list, torch_index: str | None) -> None:
    # Step 1 — pinned CUDA
    if torch_index:
        try:
            pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
            return
        except subprocess.CalledProcessError:
            print("[setup] Pinned CUDA install failed, trying unpinned...")

        # Step 2 — unpinned CUDA
        try:
            unpinned = [p.split("==")[0] for p in torch_pkgs]
            pip(venv, "install", *unpinned, "--index-url", torch_index)
            return
        except subprocess.CalledProcessError:
            print("[setup] CUDA index failed, fallback to PyPI...")

    # Step 3 — PyPI fallback
    unpinned = [p.split("==")[0] for p in torch_pkgs]
    pip(venv, "install", *unpinned)


# ------------------------------------------------------------
# Optional flash-attn
# ------------------------------------------------------------
def install_flash_attn(venv: Path) -> None:
    try:
        pip(venv, "install", "flash-attn", "--no-build-isolation")
        print("[setup] flash-attn installed.")
    except Exception as e:
        print(f"[setup] flash-attn skipped: {e}")


# ------------------------------------------------------------
# MeshAnything install
# ------------------------------------------------------------
def install_meshanything(venv: Path) -> None:
    is_win = platform.system() == "Windows"
    exe = venv / ("Scripts/python.exe" if is_win else "bin/python")

    site_packages = subprocess.check_output(
        [str(exe), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()

    dest_pkg = Path(site_packages) / "MeshAnything"
    dest_mesh2pc = Path(site_packages) / "mesh_to_pc.py"

    if dest_pkg.exists() and dest_mesh2pc.exists():
        print("[setup] MeshAnything already installed.")
        return

    print("[setup] Downloading MeshAnythingV2...")
    with urllib.request.urlopen(_MESHANYTHINGV2_ZIP, timeout=180) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if "MeshAnythingV2-main/MeshAnything/" in member:
                rel = member.split("MeshAnythingV2-main/")[1]
                target = Path(site_packages) / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        if "MeshAnythingV2-main/mesh_to_pc.py" in zf.namelist():
            dest_mesh2pc.write_bytes(zf.read("MeshAnythingV2-main/mesh_to_pc.py"))

    print("[setup] MeshAnything installed.")


# ------------------------------------------------------------
# Main setup
# ------------------------------------------------------------
def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"

    # Python compatibility
    if sys.version_info >= (3, 13):
        print("[setup] Python incompatible, searching fallback...")
        py_cmd = find_compatible_python()
    else:
        py_cmd = [python_exe]

    print(f"[setup] Creating venv at {venv} ...")
    subprocess.run(py_cmd + ["-m", "venv", str(venv)], check=True)

    # --------------------------------------------------------
    # Torch selection
    # --------------------------------------------------------
    if gpu_sm == 0:
        print("[setup] No GPU detected -> CPU PyTorch")
        torch_pkgs = ["torch", "torchvision"]
        torch_index = None

    elif gpu_sm >= 100 or cuda_version >= 128:
        torch_pkgs = ["torch==2.7.0", "torchvision==0.22.0"]
        torch_index = "https://download.pytorch.org/whl/cu128"
        print("[setup] Blackwell GPU -> CUDA 12.8")

    elif gpu_sm >= 70:
        torch_pkgs = ["torch==2.6.0", "torchvision==0.21.0"]
        torch_index = "https://download.pytorch.org/whl/cu124"
        print("[setup] Modern GPU -> CUDA 12.4")

    else:
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1"]
        torch_index = "https://download.pytorch.org/whl/cu118"
        print("[setup] Legacy GPU -> CUDA 11.8")

    print("[setup] Installing PyTorch...")
    install_torch(venv, torch_pkgs, torch_index)

    # --------------------------------------------------------
    # Dependencies
    # --------------------------------------------------------
    print("[setup] Installing dependencies...")
    pip(
        venv,
        "install",
        "Pillow",
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

    install_flash_attn(venv)
    install_meshanything(venv)

    print("[setup] Done:", venv)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            args["python_exe"],
            Path(args["ext_dir"]),
            int(args.get("gpu_sm", 86)),
            int(args.get("cuda_version", 0)),
        )
    else:
        raise RuntimeError("Invalid arguments")