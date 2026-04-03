"""
MeshAnythingV2 — Python subprocess entry for Modly process extension.

Modly (via processor.js) spawns this script and communicates via stdin/stdout
using a JSON-lines protocol:

  stdin  → one JSON line: { input, params, nodeId, workspaceDir, tempDir, extDir }
  stdout → JSON lines:    { type: "progress"|"log"|"done"|"error", ... }
"""
import io
import json
import platform
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

# ── File logger (backup — written even if Modly drops JSON logs) ─────────────

_LOG_FILE = Path(__file__).parent / 'generator.log'

def _flog(message: str) -> None:
    try:
        ts = time.strftime('%H:%M:%S')
        with open(_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f'[{ts}] {message}\n')
    except Exception:
        pass


# ── Protocol helpers ────────────────────────────────────────────────────────

def _progress(percent: int, label: str = '') -> None:
    print(json.dumps({'type': 'progress', 'percent': percent, 'label': label}), flush=True)

def _log(message: str) -> None:
    _flog(message)
    print(json.dumps({'type': 'log', 'message': str(message)}), flush=True)

def _done(result: dict) -> None:
    _flog(f'DONE → {result}')
    print(json.dumps({'type': 'done', 'result': result}), flush=True)

def _error(message: str) -> None:
    _flog(f'ERROR → {message}')
    print(json.dumps({'type': 'error', 'message': str(message)}), flush=True)


# ── Debug helpers ────────────────────────────────────────────────────────────

def _log_env() -> None:
    """Log environment info useful for debugging."""
    _log(f'[env] Python: {sys.version}')
    _log(f'[env] Platform: {platform.platform()}')
    _log(f'[env] Executable: {sys.executable}')
    _log(f'[env] sys.path: {sys.path}')

def _log_import(name: str) -> None:
    _log(f'[import] Loading {name}...')

def _log_import_ok(name: str, version: str = '') -> None:
    v = f' v{version}' if version else ''
    _log(f'[import] {name} OK{v}')

def _log_step(label: str) -> None:
    _log(f'[step] {label}')


# ── Node: generate ───────────────────────────────────────────────────────────

def run_generate(input_path: str, params: dict, workspace_dir: str) -> str:
    _log_step('Importing numpy...')
    _log_import('numpy')
    import numpy as np
    _log_import_ok('numpy', np.__version__)

    _log_import('torch')
    import torch
    _log_import_ok('torch', torch.__version__)
    _log(f'[torch] CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        _log(f'[torch] CUDA device: {torch.cuda.get_device_name(0)}')
        _log(f'[torch] CUDA version: {torch.version.cuda}')
        _log(f'[torch] VRAM total: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB')

    _log_import('trimesh')
    import trimesh
    _log_import_ok('trimesh', trimesh.__version__)

    _log_import('accelerate')
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    import accelerate
    _log_import_ok('accelerate', accelerate.__version__)

    _log_import('MeshAnything')
    from MeshAnything.models.meshanything_v2 import MeshAnythingV2
    _log_import_ok('MeshAnything')

    _log_import('mesh_to_pc')
    from mesh_to_pc import process_mesh_to_pc
    _log_import_ok('mesh_to_pc')

    face_number    = min(int(params.get('face_number', 800)), 1600)
    mc             = bool(params.get('mc', True))
    mc_level       = int(params.get('mc_level', 7))
    no_pc_vertices = int(params.get('no_pc_vertices', 8192))
    sampling       = bool(params.get('sampling', False))
    seed           = int(params.get('seed', 0))

    _log(f'[params] face_number={face_number} mc={mc} mc_level={mc_level} '
         f'no_pc_vertices={no_pc_vertices} sampling={sampling} seed={seed}')
    _log(f'[input] file: {input_path}')

    set_seed(seed)

    # Load model
    _progress(5, 'Loading model...')
    _log_step('Initialising Accelerator...')
    kwargs      = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='fp16', kwargs_handlers=[kwargs])
    _log(f'[accelerator] device: {accelerator.device}')

    _log_step('Loading MeshAnythingV2 from HuggingFace (Yiwen-ntu/MeshAnythingV2)...')
    model = MeshAnythingV2.from_pretrained('Yiwen-ntu/MeshAnythingV2')
    _log_step('Preparing model with accelerator...')
    model = accelerator.prepare(model)
    _log(f'[model] Loaded on {accelerator.device}')
    if torch.cuda.is_available():
        _log(f'[torch] VRAM used after model load: {torch.cuda.memory_allocated(0) // 1024**2} MB')

    # Load mesh
    _progress(10, 'Loading mesh...')
    _log_step(f'Reading mesh from: {input_path}')
    with open(input_path, 'rb') as f:
        raw = f.read()
    _log(f'[mesh] File size: {len(raw)} bytes')
    mesh = trimesh.load(io.BytesIO(raw), force='mesh')
    _log(f'[mesh] Vertices: {len(mesh.vertices)}  Faces: {len(mesh.faces)}')
    _log(f'[mesh] Bounds: {mesh.bounds.tolist()}')

    # Point cloud
    _progress(15, 'Running Marching Cubes...' if mc else 'Sampling point cloud...')
    _log_step('process_mesh_to_pc...')
    pc_list, _ = process_mesh_to_pc(
        [mesh],
        marching_cubes=mc,
        sample_num=no_pc_vertices,
        mc_level=mc_level,
    )
    pc_normal = pc_list[0]  # (N, 6) float16
    _log(f'[pc] shape: {pc_normal.shape}  dtype: {pc_normal.dtype}')

    # Normalise
    _progress(20, 'Normalising point cloud...')
    _log_step('Normalising...')
    pc_coor = pc_normal[:, :3].astype(np.float32)
    normals = pc_normal[:, 3:].astype(np.float32)
    bounds  = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
    pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
    pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995
    norm_check = (np.linalg.norm(normals, axis=-1) > 0.99).all()
    _log(f'[pc] normals unit-vector check: {norm_check}')
    if not norm_check:
        raise ValueError('Normals are not unit vectors — point cloud may be invalid.')
    pc_normal_norm = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
    _log(f'[pc] normalised shape: {pc_normal_norm.shape}')

    # Inference with smooth progress ticker
    _progress(25, 'Running MeshAnythingV2 inference...')
    _log_step('Starting inference...')
    stop_evt = threading.Event()

    def _ticker():
        pct = 25
        while not stop_evt.is_set():
            time.sleep(5)
            pct = min(pct + 3, 84)
            _progress(pct, 'Generating mesh...')

    ticker = threading.Thread(target=_ticker, daemon=True)
    ticker.start()

    t0 = time.time()
    try:
        batch = torch.tensor(pc_normal_norm, dtype=torch.float16).unsqueeze(0)
        _log(f'[inference] batch shape: {list(batch.shape)}  device: {accelerator.device}')
        batch = batch.to(accelerator.device)
        with accelerator.autocast():
            with torch.no_grad():
                outputs = model(batch, sampling=sampling)
    finally:
        stop_evt.set()

    _log(f'[inference] Done in {time.time() - t0:.1f}s')

    # Post-process
    _progress(87, 'Post-processing mesh...')
    _log_step('Post-processing output...')
    recon_mesh = outputs[0]
    _log(f'[output] raw shape: {list(recon_mesh.shape)}')
    valid_mask = torch.all(~torch.isnan(recon_mesh.reshape((-1, 9))), dim=1)
    _log(f'[output] valid faces: {valid_mask.sum().item()} / {valid_mask.shape[0]}')
    recon_mesh = recon_mesh[valid_mask]

    vertices       = recon_mesh.reshape(-1, 3).cpu().numpy()
    vertices_index = np.arange(len(vertices))
    triangles      = vertices_index.reshape(-1, 3)
    _log(f'[mesh] vertices: {len(vertices)}  faces: {len(triangles)}')

    scene_mesh = trimesh.Trimesh(
        vertices=vertices, faces=triangles,
        force='mesh', merge_primitives=True,
    )
    scene_mesh.merge_vertices()
    scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
    scene_mesh.update_faces(scene_mesh.unique_faces())
    scene_mesh.remove_unreferenced_vertices()
    scene_mesh.fix_normals()
    _log(f'[mesh] after cleanup — vertices: {len(scene_mesh.vertices)}  faces: {len(scene_mesh.faces)}')

    # Optional simplification
    if face_number > 0 and len(scene_mesh.faces) > face_number:
        _progress(93, 'Simplifying mesh...')
        _log_step(f'Simplifying to {face_number} faces...')
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
            _log(f'[simplify] result — vertices: {len(scene_mesh.vertices)}  faces: {len(scene_mesh.faces)}')
        except Exception as exc:
            _log(f'[simplify] skipped: {exc}')
    else:
        _log(f'[simplify] skipped (faces {len(scene_mesh.faces)} <= target {face_number})')

    # Export
    _progress(96, 'Exporting GLB...')
    out_path = Path(workspace_dir) / f'{int(time.time())}_{uuid.uuid4().hex[:8]}.glb'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log_step(f'Exporting to: {out_path}')
    scene_mesh.export(str(out_path))
    _log(f'[export] File size: {out_path.stat().st_size} bytes')

    _progress(100, 'Done')
    return str(out_path)


# ── Node: preprocess ─────────────────────────────────────────────────────────

def run_preprocess(input_path: str, params: dict, workspace_dir: str) -> str:
    _log_import('trimesh')
    import trimesh
    _log_import_ok('trimesh', trimesh.__version__)

    _log_import('mesh_to_pc')
    from mesh_to_pc import process_mesh_to_pc
    _log_import_ok('mesh_to_pc')

    mc_level = int(params.get('mc_level', 7))
    _log(f'[params] mc_level={mc_level}')
    _log(f'[input] file: {input_path}')

    _progress(5, 'Loading mesh...')
    with open(input_path, 'rb') as f:
        raw = f.read()
    _log(f'[mesh] File size: {len(raw)} bytes')
    mesh = trimesh.load(io.BytesIO(raw), force='mesh')
    _log(f'[mesh] Vertices: {len(mesh.vertices)}  Faces: {len(mesh.faces)}')

    _progress(15, 'Running Marching Cubes...')
    _log_step('process_mesh_to_pc (marching cubes only)...')
    _, mesh_list = process_mesh_to_pc(
        [mesh],
        marching_cubes=True,
        sample_num=8192,
        mc_level=mc_level,
    )
    watertight = mesh_list[0]
    _log(f'[mc] Vertices: {len(watertight.vertices)}  Faces: {len(watertight.faces)}')

    _progress(90, 'Exporting GLB...')
    out_path = Path(workspace_dir) / f'{int(time.time())}_{uuid.uuid4().hex[:8]}_preprocess.glb'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log_step(f'Exporting to: {out_path}')
    watertight.export(str(out_path))
    _log(f'[export] File size: {out_path.stat().st_size} bytes')

    _progress(100, 'Done')
    return str(out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _log_env()
    _log_step('Reading stdin...')

    raw_line = sys.stdin.readline()
    _log(f'[stdin] received {len(raw_line)} bytes')

    data         = json.loads(raw_line)
    input_data   = data['input']
    params       = data['params']
    node_id      = data['nodeId']
    workspace    = data['workspaceDir']

    _log(f'[main] nodeId: {node_id}')
    _log(f'[main] workspaceDir: {workspace}')
    _log(f'[main] params: {json.dumps(params)}')

    input_path = input_data.get('filePath')
    _log(f'[main] inputFilePath: {input_path}')

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
    try:
        main()
    except Exception:
        print(json.dumps({'type': 'error', 'message': traceback.format_exc()}), flush=True)
        sys.exit(1)
