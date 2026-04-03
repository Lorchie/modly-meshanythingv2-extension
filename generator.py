import json
import sys
import traceback
from pathlib import Path


def send(type_, data):
    print(json.dumps({"type": type_, **data}), flush=True)


def main():
    try:
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("No input received on stdin")

        data = json.loads(line)

        input_path = data["input"]["filePath"]
        params = data["params"]
        node = data["nodeId"]
        out_dir = Path(data["workspaceDir"])

        import torch
        import trimesh
        from MeshAnything.models.meshanything_v2 import MeshAnythingV2
        from mesh_to_pc import process_mesh_to_pc

        send("log", {"message": f"torch={torch.__version__}"})
        send("log", {"message": f"node={node}"})

        mesh = trimesh.load(input_path)

        pc, _ = process_mesh_to_pc([mesh])
        pc = pc[0]

        model = MeshAnythingV2.from_pretrained("Yiwen-ntu/MeshAnythingV2")

        batch = torch.tensor(pc).unsqueeze(0)

        with torch.no_grad():
            out = model(batch)

        verts = out[0].reshape(-1, 3).cpu().numpy()
        faces = list(range(len(verts)))
        faces = [faces[i:i + 3] for i in range(0, len(faces), 3)]

        out_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        out_path = out_dir / "result.glb"
        out_mesh.export(out_path)

        send("done", {"result": {"filePath": str(out_path)}})

    except Exception:
        send("error", {"message": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()