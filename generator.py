import sys
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

EXT_DIR = Path(__file__).parent
REPO_DIR = EXT_DIR / "MeshAnythingV2"

# Add repo to path
sys.path.append(str(REPO_DIR))

# Import model + preprocess
from MeshAnything.models.meshanything_v2 import MeshAnythingV2
from mesh_to_pc import process_mesh_to_pc


_device = torch.device("cpu")
_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    print("[MeshAnythingV2] Loading model from HuggingFace…")
    model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")
    model.to(_device)
    model.eval()

    _model = model
    return model


# -------------------------
# PREPROCESS
# -------------------------
def preprocess_mesh(mesh_path, params, out_path):
    mc_level = params.get("mc_level", 7)

    mesh = trimesh.load(mesh_path)
    pc_list, _ = process_mesh_to_pc([mesh], marching_cubes=True, mc_level=mc_level)

    pts = pc_list[0][:, :3]
    cloud = trimesh.points.PointCloud(pts)
    cloud.export(out_path)

    return out_path


# -------------------------
# GENERATE (REMESH)
# -------------------------
def generate_mesh(mesh_path, params, out_path):
    model = load_model()

    sampling = params.get("sampling", False)
    seed = params.get("seed", 0)
    mc = params.get("mc", False)
    mc_level = params.get("mc_level", 7)
    no_pc_vertices = params.get("no_pc_vertices", 8192)

    torch.manual_seed(seed)
    np.random.seed(seed)

    mesh = trimesh.load(mesh_path)

    pc_list, _ = process_mesh_to_pc(
        [mesh],
        marching_cubes=mc,
        mc_level=mc_level,
        sample_num=no_pc_vertices
    )
    pc = pc_list[0]

    pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(_device)

    with torch.no_grad():
        gen_mesh = model(pc_tensor, sampling=sampling)[0]  # (nf,3,3)

    vertices = gen_mesh.reshape(-1, 3).cpu().numpy()
    faces = np.arange(len(vertices)).reshape(-1, 3)

    scene_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    scene_mesh.merge_vertices()
    scene_mesh.remove_unreferenced_vertices()
    scene_mesh.fix_normals()

    scene_mesh.export(out_path)
    return out_path


# -------------------------
# CLI ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit("Usage: generator.py <node_id> <input_mesh> <output_mesh> <params_json>")

    node_id = sys.argv[1]
    input_mesh = sys.argv[2]
    output_mesh = sys.argv[3]
    params = json.loads(sys.argv[4])

    if node_id == "preprocess":
        preprocess_mesh(input_mesh, params, output_mesh)
    elif node_id == "generate":
        generate_mesh(input_mesh, params, output_mesh)
    else:
        raise RuntimeError(f"Unknown node id: {node_id}")