"""
Build the vendor/ directory for the MeshAnythingV2 extension.

Run this script once (with the app's venv active) to populate vendor/.
The resulting vendor/ folder is committed to the extension repository
so end users never need to install anything manually at runtime.

Usage:
    python build_vendor.py

Requirements (must be run from the app's venv which has PyTorch installed):
    - pip (always available)
    - CUDA toolchain is NOT required — MeshAnything is pure Python

What gets vendored:
    vendor/MeshAnything/    — the model package
    vendor/mesh_to_pc.py    — top-level point cloud helper

Note: mesh2sdf and scikit-image contain compiled C extensions and cannot
be vendored as pure Python. They are installed into the venv by setup.py.
Do NOT attempt to vendor them here.
"""
import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

VENDOR              = Path(__file__).parent / "vendor"
_MESHANYTHINGV2_ZIP = "https://github.com/buaacyw/MeshAnythingV2/archive/refs/heads/main.zip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def vendor_meshanything(dest: Path) -> None:
    """
    Download MeshAnythingV2 source from GitHub and extract:
      - MeshAnything/  (the model package) → vendor/MeshAnything/
      - mesh_to_pc.py  (top-level module)  → vendor/mesh_to_pc.py
    """
    dest_pkg     = dest / "MeshAnything"
    dest_mesh2pc = dest / "mesh_to_pc.py"

    if dest_pkg.exists() and dest_mesh2pc.exists():
        print("  MeshAnything/ and mesh_to_pc.py already present, skipping.")
        return

    print("  Downloading MeshAnythingV2 source from GitHub ...")
    with urllib.request.urlopen(_MESHANYTHINGV2_ZIP, timeout=180) as resp:
        data = resp.read()

    pkg_prefix     = "MeshAnythingV2-main/MeshAnything/"
    mesh2pc_member = "MeshAnythingV2-main/mesh_to_pc.py"
    strip          = "MeshAnythingV2-main/"

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()

        # Extract MeshAnything/ package
        pkg_members = [m for m in members if m.startswith(pkg_prefix)]
        if not pkg_members:
            raise RuntimeError(
                f"MeshAnything/ package not found in archive. "
                f"Expected prefix: {pkg_prefix}"
            )
        for member in pkg_members:
            rel    = member[len(strip):]
            target = dest / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

        print(f"  MeshAnything/ extracted to {dest}.")

        # Extract mesh_to_pc.py as a top-level module
        if mesh2pc_member not in members:
            raise RuntimeError(
                f"mesh_to_pc.py not found in archive. "
                f"Expected: {mesh2pc_member}"
            )
        dest_mesh2pc.write_bytes(zf.read(mesh2pc_member))
        print(f"  mesh_to_pc.py extracted to {dest}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Building vendor/ in {VENDOR}")
    VENDOR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Vendoring MeshAnything source (MeshAnything/ + mesh_to_pc.py) ...")
    try:
        vendor_meshanything(VENDOR)
    except Exception as exc:
        print(f"  ERROR: MeshAnything vendor failed: {exc}")
        raise SystemExit(1)

    print("\nDone! vendor/ is ready.")
    print("Commit the vendor/ directory to the extension repository.")


if __name__ == "__main__":
    main()
