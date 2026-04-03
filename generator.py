"""
MeshAnythingV2 — Python subprocess entry for Modly process extension.

Modly (via processor.js) spawns this script and communicates via stdin/stdout
using a JSON-lines protocol:

  stdin  → one JSON line: { input, params, nodeId, workspaceDir, tempDir, extDir }
  stdout → JSON lines:    { type: "progress"|"log"|"done"|"error", ... }
"""
import io
import json
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np


# ── Protocol helpers ────────────────────────────────────────────────────────

def _progress(percent: int, label: str = '') -> None:
    print(json.dumps({'type': 'progress', 'percent': percent, 'label': label}), flush=True)

def _log(message: str) -> None:
    print(json.dumps({'type': 'log', 'message': str(message)}), flush=True)

def _done(result: dict) -> None:
    print(json.dumps({'type': 'done', 'result': result}), flush=True)

def _error(message: str) -> None:
    print(json.dumps({'type': 'error', 'message': str(message)}), flush=True)


# ── Node: generate ───────────────────────────────────────────────────────────

def run_generate(input_path: str, params: dict, workspace_dir: str) -> str:
    import torch
    import trimesh
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from MeshAnything.models.meshanything_v2 import MeshAnythingV2
    from mesh_to_pc import process_mesh_to_pc

    face_number    = min(int(params.get('face_number', 800)), 1600)
    mc             = bool(params.get('mc', True))
    mc_level       = int(params.get('mc_level', 7))
    no_pc_vertices = int(params.get('no_pc_vertices', 8192))
    sampling       = bool(params.get('sampling', False))
    seed           = int(params.get('seed', 0))

    set_seed(seed)

    # Load model
    _progress(5, 'Loading model...')
    kwargs      = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='fp16', kwargs_handlers=[kwargs])
    model = MeshAnythingV2.from_pretrained('Yiwen-ntu/MeshAnythingV2')
    model = accelerator.prepare(model)
    _log(f'Model loaded on {accelerator.device}')

    # Load mesh
    _progress(10, 'Loading mesh...')
    with open(input_path, 'rb') as f:
        mesh = trimesh.load(io.BytesIO(f.read()), force='mesh')

    # Point cloud
    _progress(15, 'Running Marching Cubes...' if mc else 'Sampling point cloud...')
    pc_list, _ = process_mesh_to_pc(
        [mesh],
        marching_cubes=mc,
        sample_num=no_pc_vertices,
        mc_level=mc_level,
    )
    pc_normal = pc_list[0]  # (N, 6) float16

    # Normalise
    _progress(20, 'Normalising point cloud...')
    pc_coor = pc_normal[:, :3].astype(np.float32)
    normals = pc_normal[:, 3:].astype(np.float32)
    bounds  = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
    pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
    pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995
    if not (np.linalg.norm(normals, axis=-1) > 0.99).all():
        raise ValueError('Normals are not unit vectors — point cloud may be invalid.')
    pc_normal_norm = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)

    # Inference with smooth progress ticker
    _progress(25, 'Running MeshAnythingV2 inference...')
    stop_evt = threading.Event()

    def _ticker():
        pct = 25
        while not stop_evt.is_set():
            time.sleep(5)
            pct = min(pct + 3, 84)
            _progress(pct, 'Generating mesh...')

    ticker = threading.Thread(target=_ticker, daemon=True)
    ticker.start()

    try:
        batch = torch.tensor(pc_normal_norm, dtype=torch.float16).unsqueeze(0)
        batch = batch.to(accelerator.device)
        with accelerator.autocast():
            with torch.no_grad():
                outputs = model(batch, sampling=sampling)
    finally:
        stop_evt.set()

    # Post-process
    _progress(87, 'Post-processing mesh...')
    recon_mesh = outputs[0]
    valid_mask = torch.all(~torch.isnan(recon_mesh.reshape((-1, 9))), dim=1)
    recon_mesh = recon_mesh[valid_mask]

    vertices       = recon_mesh.reshape(-1, 3).cpu().numpy()
    vertices_index = np.arange(len(vertices))
    triangles      = vertices_index.reshape(-1, 3)

    scene_mesh = trimesh.Trimesh(
        vertices=vertices, faces=triangles,
        force='mesh', merge_primitives=True,
    )
    scene_mesh.merge_vertices()
    scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
    scene_mesh.update_faces(scene_mesh.unique_faces())
    scene_mesh.remove_unreferenced_vertices()
    scene_mesh.fix_normals()

    # Optional simplification
    if face_number > 0 and len(scene_mesh.faces) > face_number:
        _progress(93, 'Simplifying mesh...')
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=scene_mesh.vertices,
                face_matrix=scene_mesh.faces,
            ))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=face_number)
            m = ms.current_mesh()
            scene_mesh = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
        except Exception as exc:
            _log(f'Simplification skipped: {exc}')

    # Export
    _progress(96, 'Exporting GLB...')
    out_path = Path(workspace_dir) / f'{int(time.time())}_{uuid.uuid4().hex[:8]}.glb'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene_mesh.export(str(out_path))

    _progress(100, 'Done')
    return str(out_path)


# ── Node: preprocess ─────────────────────────────────────────────────────────

def run_preprocess(input_path: str, params: dict, workspace_dir: str) -> str:
    import trimesh
    from mesh_to_pc import process_mesh_to_pc

    mc_level = int(params.get('mc_level', 7))

    _progress(5, 'Loading mesh...')
    with open(input_path, 'rb') as f:
        mesh = trimesh.load(io.BytesIO(f.read()), force='mesh')

    _progress(15, 'Running Marching Cubes...')
    _, mesh_list = process_mesh_to_pc(
        [mesh],
        marching_cubes=True,
        sample_num=8192,
        mc_level=mc_level,
    )
    watertight = mesh_list[0]

    _progress(90, 'Exporting GLB...')
    out_path = Path(workspace_dir) / f'{int(time.time())}_{uuid.uuid4().hex[:8]}_preprocess.glb'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    watertight.export(str(out_path))

    _progress(100, 'Done')
    return str(out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    data         = json.loads(sys.stdin.readline())
    input_data   = data['input']
    params       = data['params']
    node_id      = data['nodeId']
    workspace    = data['workspaceDir']

    input_path = input_data.get('filePath')

    try:
        if node_id == 'preprocess':
            out = run_preprocess(input_path, params, workspace)
        else:
            out = run_generate(input_path, params, workspace)
        _done({'filePath': out})
    except Exception:
        _error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
