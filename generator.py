"""
MeshAnythingV2 extension for Modly.

Reference : https://huggingface.co/Yiwen-ntu/MeshAnythingV2
GitHub    : https://github.com/buaacyw/MeshAnythingV2

Pure-Python dependencies (einops, transformers, accelerate, etc.) are
installed into the extension venv by setup.py.

MeshAnything source (MeshAnything/ package + mesh_to_pc.py) is bundled in
vendor/ by build_vendor.py.

To rebuild vendor/:
    python build_vendor.py   (run once with the app's venv active)
"""
import sys
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_EXTENSION_DIR = Path(__file__).parent


class MeshAnythingV2Generator(BaseGenerator):
    MODEL_ID     = "meshanythingv2"
    DISPLAY_NAME = "MeshAnythingV2"
    VRAM_GB      = 8

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        return (self.model_dir / "config.json").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._auto_download()

        self._setup_vendor()

        import torch
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs
        from MeshAnything.models.meshanything_v2 import MeshAnythingV2

        kwargs      = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision="fp16",
            kwargs_handlers=[kwargs],
        )

        print(f"[MeshAnythingV2Generator] Loading model from {self.model_dir} ...")
        model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")
        model = accelerator.prepare(model)

        self._model       = model
        self._accelerator = accelerator
        self._device      = str(accelerator.device)
        print(f"[MeshAnythingV2Generator] Loaded on {self._device}.")

    def unload(self) -> None:
        self._accelerator = None
        self._device      = None
        super().unload()

    # ------------------------------------------------------------------ #
    # Node dispatch
    # ------------------------------------------------------------------ #

    def generate(
        self,
        mesh_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Entry point for the default 'generate' node (called by Modly framework)."""
        return self._run_generate(mesh_bytes, params, progress_cb, cancel_event)

    def run_node(
        self,
        node_id: str,
        input_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        if node_id == "preprocess":
            return self._run_preprocess(input_bytes, params, progress_cb, cancel_event)
        return self._run_generate(input_bytes, params, progress_cb, cancel_event)

    # ------------------------------------------------------------------ #
    # Node: generate
    # ------------------------------------------------------------------ #

    def _run_generate(
        self,
        mesh_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import io
        import torch  # noqa: F811
        import trimesh
        from accelerate.utils import set_seed
        from mesh_to_pc import process_mesh_to_pc

        face_number    = int(params.get("face_number", 800))
        mc             = bool(params.get("mc", True))
        mc_level       = int(params.get("mc_level", 7))
        no_pc_vertices = int(params.get("no_pc_vertices", 8192))
        sampling       = bool(params.get("sampling", False))
        seed           = int(params.get("seed", 0))

        # Clamp to hard model limit
        face_number = min(face_number, 1600)

        set_seed(seed)

        # ── Load input mesh ──────────────────────────────────────────────
        self._report(progress_cb, 5, "Loading mesh...")
        mesh = trimesh.load(io.BytesIO(mesh_bytes), force="mesh")
        self._check_cancelled(cancel_event)

        # ── Point cloud ──────────────────────────────────────────────────
        self._report(progress_cb, 10, "Sampling point cloud...")
        if mc:
            self._report(progress_cb, 10, "Running Marching Cubes...")
        pc_list, _ = process_mesh_to_pc(
            [mesh],
            marching_cubes=mc,
            sample_num=no_pc_vertices,
            mc_level=mc_level,
        )
        pc_normal = pc_list[0]  # shape (N, 6), float16
        self._check_cancelled(cancel_event)

        # ── Normalise ────────────────────────────────────────────────────
        self._report(progress_cb, 20, "Normalising point cloud...")
        pc_coor = pc_normal[:, :3].astype(np.float32)
        normals = pc_normal[:, 3:].astype(np.float32)

        bounds  = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
        pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
        pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995

        assert (np.linalg.norm(normals, axis=-1) > 0.99).all(), \
            "Normals are not unit vectors — point cloud may be invalid."

        pc_normal_norm = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
        self._check_cancelled(cancel_event)

        # ── Inference ────────────────────────────────────────────────────
        self._report(progress_cb, 25, "Running MeshAnythingV2 inference...")

        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 25, 85, "Generating mesh...", stop_evt, 5.0),
                daemon=True,
            )
            t.start()

        try:
            batch = torch.tensor(pc_normal_norm, dtype=torch.float16).unsqueeze(0)
            batch = batch.to(self._accelerator.device)

            with self._accelerator.autocast():
                with torch.no_grad():
                    outputs = self._model(batch, sampling=sampling)
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # ── Post-process ─────────────────────────────────────────────────
        self._report(progress_cb, 87, "Post-processing mesh...")

        recon_mesh = outputs[0]
        valid_mask = torch.all(~torch.isnan(recon_mesh.reshape((-1, 9))), dim=1)
        recon_mesh = recon_mesh[valid_mask]  # n_valid_face x 3 x 3

        vertices       = recon_mesh.reshape(-1, 3).cpu().numpy()
        vertices_index = np.arange(len(vertices))
        triangles      = vertices_index.reshape(-1, 3)

        scene_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            force="mesh",
            merge_primitives=True,
        )
        scene_mesh.merge_vertices()
        scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
        scene_mesh.update_faces(scene_mesh.unique_faces())
        scene_mesh.remove_unreferenced_vertices()
        scene_mesh.fix_normals()
        self._check_cancelled(cancel_event)

        # ── Optional simplification ──────────────────────────────────────
        if face_number > 0 and len(scene_mesh.faces) > face_number:
            self._report(progress_cb, 93, "Simplifying mesh...")
            scene_mesh = self._simplify(scene_mesh, face_number)

        # ── Export ───────────────────────────────────────────────────────
        self._report(progress_cb, 96, "Exporting GLB...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        scene_mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Node: preprocess
    # ------------------------------------------------------------------ #

    def _run_preprocess(
        self,
        mesh_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import io
        import trimesh
        from mesh_to_pc import process_mesh_to_pc

        mc_level = int(params.get("mc_level", 7))

        self._report(progress_cb, 5, "Loading mesh...")
        mesh = trimesh.load(io.BytesIO(mesh_bytes), force="mesh")
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 15, "Running Marching Cubes...")
        _, mesh_list = process_mesh_to_pc(
            [mesh],
            marching_cubes=True,
            sample_num=8192,
            mc_level=mc_level,
        )
        watertight_mesh = mesh_list[0]
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 90, "Exporting GLB...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_preprocess.glb"
        path = self.outputs_dir / name
        watertight_mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Vendor setup
    # ------------------------------------------------------------------ #

    def _setup_vendor(self) -> None:
        # Add vendor/ to sys.path so MeshAnything and mesh_to_pc are importable.
        # vendor/ is populated once by running build_vendor.py.
        vendor_dir = _EXTENSION_DIR / "vendor"
        if vendor_dir.exists() and str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))

        try:
            from MeshAnything.models.meshanything_v2 import MeshAnythingV2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "[MeshAnythingV2Generator] MeshAnything not found in vendor/. "
                "Run build_vendor.py with the app's Python to build vendor/, "
                "or click Repair on the Models page to run setup.py."
            ) from exc

        try:
            from mesh_to_pc import process_mesh_to_pc  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "[MeshAnythingV2Generator] mesh_to_pc not found in vendor/. "
                "Run build_vendor.py with the app's Python to rebuild vendor/."
            ) from exc

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _simplify(self, mesh, target_faces: int):
        try:
            import pymeshlab
            import trimesh as _trimesh

            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=mesh.vertices,
                face_matrix=mesh.faces,
            ))
            ms.meshing_merge_close_vertices()
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            m = ms.current_mesh()
            return _trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
        except Exception as exc:
            print(f"[MeshAnythingV2Generator] Simplification skipped: {exc}")
            return mesh

    def _report(
        self,
        progress_cb: Optional[Callable[[int, str], None]],
        pct: int,
        msg: str,
    ) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    def _check_cancelled(self, cancel_event: Optional[threading.Event]) -> None:
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()
