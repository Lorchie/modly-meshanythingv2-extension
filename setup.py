"""
MeshAnythingV2 — extension setup script.

Creates an isolated venv and installs all required dependencies.
MeshAnythingV2 source (MeshAnything/ package + mesh_to_pc.py) is installed
directly from GitHub into the venv's site-packages.

Called by Modly at extension install time with:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}'
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


def pip(venv: Path, *args: str) -> None:
    is_win  = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def install_flash_attn(venv: Path) -> None:
    """
    Attempt to install flash-attn for faster attention.
    ALL failures are caught silently — this is strictly optional.
    MeshAnythingV2 runs fine without it.
    """
    try:
        pip(venv, "install", "flash-attn", "--no-build-isolation")
        print("[setup] flash-attn installed.")
    except Exception as e:
        print(f"[setup] flash-attn not available ({e}), skipping (optional).")


def install_meshanything(venv: Path) -> None:
    """
    Downloads MeshAnythingV2 source from GitHub and installs:
      - MeshAnything/   (the model package)
      - mesh_to_pc.py   (top-level point cloud helper)
    directly into the venv's site-packages.
    """
    is_win = platform.system() == "Windows"
    exe    = venv / ("Scripts/python.exe" if is_win else "bin/python")

    site_packages = subprocess.check_output(
        [str(exe), "-c",
         "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"],
        text=True,
    ).strip()

    dest_pkg       = Path(site_packages) / "MeshAnything"
    dest_mesh2pc   = Path(site_packages) / "mesh_to_pc.py"

    if dest_pkg.exists() and dest_mesh2pc.exists():
        print("[setup] MeshAnything already installed, skipping.")
        return

    print("[setup] Downloading MeshAnythingV2 source from GitHub ...")
    with urllib.request.urlopen(_MESHANYTHINGV2_ZIP, timeout=180) as resp:
        data = resp.read()

    pkg_prefix      = "MeshAnythingV2-main/MeshAnything/"
    mesh2pc_member  = "MeshAnythingV2-main/mesh_to_pc.py"
    strip           = "MeshAnythingV2-main/"

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Install MeshAnything/ package
        for member in zf.namelist():
            if not member.startswith(pkg_prefix):
                continue
            rel    = member[len(strip):]
            target = Path(site_packages) / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

        # Install mesh_to_pc.py as a top-level module
        if mesh2pc_member in zf.namelist():
            dest_mesh2pc.write_bytes(zf.read(mesh2pc_member))
            print(f"[setup] mesh_to_pc.py installed to {site_packages}.")
        else:
            raise RuntimeError(
                f"mesh_to_pc.py not found in archive. "
                f"Expected: {mesh2pc_member}"
            )

    print(f"[setup] MeshAnything installed to {site_packages}.")


def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"

    print(f"[setup] Creating venv at {venv} ...")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch — select build based on GPU architecture + CUDA driver
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100 or cuda_version >= 128:
        # Blackwell (RTX 50xx, B100 ...) — requires cu128
        torch_pkgs  = ["torch==2.7.0", "torchvision==0.22.0"]
        torch_index = "https://download.pytorch.org/whl/cu128"
        print(f"[setup] GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8 (Blackwell)")
    elif gpu_sm == 0 or gpu_sm >= 70:
        # Ampere / Ada Lovelace / Hopper
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        torch_index = "https://download.pytorch.org/whl/cu124"
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
    else:
        # Pascal / Volta / Turing (sm_60–sm_75)
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        torch_index = "https://download.pytorch.org/whl/cu118"
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")

    print("[setup] Installing PyTorch ...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies ...")
    pip(venv, "install",
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

    # ------------------------------------------------------------------ #
    # flash-attn (optional — ALL failures caught silently)
    # ------------------------------------------------------------------ #
    install_flash_attn(venv)

    # ------------------------------------------------------------------ #
    # MeshAnythingV2 source (MeshAnything/ + mesh_to_pc.py)
    # ------------------------------------------------------------------ #
    install_meshanything(venv)

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            args["python_exe"],
            Path(args["ext_dir"]),
            int(args.get("gpu_sm", 86)),
            int(args.get("cuda_version", 0)),
        )
    else:
        # Read JSON from stdin (avoids CLI quoting issues on Windows)
        raw = sys.stdin.read().strip()
        if not raw:
            print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
            print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}\'')
            sys.exit(1)
        args = json.loads(raw)
        setup(
            args["python_exe"],
            Path(args["ext_dir"]),
            int(args.get("gpu_sm", 86)),
            int(args.get("cuda_version", 0)),
        )
