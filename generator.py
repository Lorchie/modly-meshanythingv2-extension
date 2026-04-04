import os
import shutil
from typing import Dict, Any

# Import du pipeline MeshAnythingV2 (repo officiel buaacyw)
from meshanythingv2.pipeline import MeshAnythingV2Pipeline


class MeshAnythingV2Generator:
    def __init__(self):
        """
        Chargement du pipeline MeshAnythingV2.
        Modly utilise le venv global : Documents/Modly/dependencies/venv
        Donc toutes les dépendances doivent être installées dedans.
        """
        self.pipeline = MeshAnythingV2Pipeline(
            device="cpu"  # tu peux mettre "cuda" si tu veux
        )

    def _preprocess_mesh(self, input_mesh: str, mc_level: int, work_dir: str) -> str:
        output_mesh = os.path.join(work_dir, "preprocessed.obj")

        self.pipeline.preprocess(
            input_mesh=input_mesh,
            output_mesh=output_mesh,
            mc_level=mc_level
        )

        return output_mesh

    def _remesh_mesh(
        self,
        input_mesh: str,
        face_number: int,
        mc: bool,
        mc_level: int,
        no_pc_vertices: int,
        sampling: bool,
        seed: int,
        work_dir: str,
    ) -> str:

        output_mesh = os.path.join(work_dir, "remeshed.obj")

        self.pipeline.remesh(
            input_mesh=input_mesh,
            output_mesh=output_mesh,
            face_number=face_number,
            mc=mc,
            mc_level=mc_level,
            no_pc_vertices=no_pc_vertices,
            sampling=sampling,
            seed=seed
        )

        return output_mesh

    def generate(
        self,
        node_id: str,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        work_dir: str,
    ) -> Dict[str, Any]:

        os.makedirs(work_dir, exist_ok=True)

        input_mesh = inputs.get("mesh")
        if not input_mesh or not os.path.exists(input_mesh):
            raise ValueError(f"Input mesh not found: {input_mesh}")

        if node_id == "preprocess":
            output_mesh = self._preprocess_mesh(
                input_mesh=input_mesh,
                mc_level=int(params.get("mc_level", 7)),
                work_dir=work_dir
            )

        elif node_id == "generate":
            output_mesh = self._remesh_mesh(
                input_mesh=input_mesh,
                face_number=int(params.get("face_number", 800)),
                mc=bool(params.get("mc", True)),
                mc_level=int(params.get("mc_level", 7)),
                no_pc_vertices=int(params.get("no_pc_vertices", 8192)),
                sampling=bool(params.get("sampling", False)),
                seed=int(params.get("seed", 0)),
                work_dir=work_dir
            )

        else:
            raise ValueError(f"Unknown node_id: {node_id}")

        return {"mesh": output_mesh}