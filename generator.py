import sys
import json
import traceback
from pathlib import Path

def send(type_, data):
    print(json.dumps({"type": type_, **data}), flush=True)

def main():
    try:
        data = json.loads(sys.stdin.readline())

        input_path = data["input"]["filePath"]
        workspace  = Path(data["workspaceDir"])

        send("log", {"message": f"Input: {input_path}"})

        import torch
        import trimesh
        from MeshAnything.models.meshanything_v2 import MeshAnythingV2
        from mesh_to_pc import process_mesh_to_pc

        send("log", {"message": f"torch={torch.__version__}"})

        mesh = trimesh.load(input_path)

        send("progress", {"percent": 20, "label": "Converting mesh to point cloud"})
        pc, _ = process_mesh_to_pc([mesh])
        pc = pc[0]

        send("progress", {"percent": 40, "label": "Loading model"})
        model = MeshAnythingV2.from_pretrained("Yiwen-ntu/MeshAnythingV2")

        batch = torch.tensor(pc).unsqueeze(0)

        send("progress", {"percent": 60, "label": "Running inference"})
        with torch.no_grad():
            out = model(batch)

        verts = out[0].reshape(-1, 3).cpu().numpy()
        faces = [list(range(i, i+3)) for i in range(0, len(verts), 3)]

        out_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        out_path = workspace / "result.glb"

        send("progress", {"percent": 90, "label": "Exporting GLB"})
        out_mesh.export(out_path)

        send("done", {"result": {"filePath": str(out_path)}})

    except Exception:
        send("error", {"message": traceback.format_exc()})
        sys.exit(1)

if __name__ == "__main__":
    main()