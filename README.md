# MeshAnythingV2 — Modly Extension

Turn any raw 3D mesh into a clean, structured mesh ready for CAD, 3D printing, or game engines — automatically.

---

## What it does

Image-to-3D models like Hunyuan3D, TripoSG or TRELLIS are great at capturing shape, but their output is messy: hundreds of thousands of tiny, unstructured triangles with no edge flow, holes, and self-intersections.

**MeshAnythingV2 fixes that.** It reads your mesh, samples a point cloud from its surface, and generates a brand-new clean mesh with proper topology — the way a 3D artist would model it by hand. The output is capped at **1600 faces**, lightweight and export-ready.

---

## Where it fits in Modly

**Image → [Hunyuan3D / TripoSG / TRELLIS] → raw mesh → [MeshAnythingV2] → clean mesh → CAD / 3D Print**

---

## Requirements

| | |
|---|---|
| GPU | NVIDIA CUDA, 8 GB VRAM minimum |
| AMD / Apple Silicon / CPU | Not supported |

---

## Nodes

### MeshAnythingV2
Full pipeline: takes your mesh, runs the model, returns a clean mesh.

| Parameter | Default | Range | Description |
|---|---|---|---|
| Max Faces | 800 | 100 – 1600 | Target face count. The model hard-caps at 1600. |
| Marching Cubes | On | — | Makes the input watertight before processing. Recommended for most meshes. |
| MC Octree Depth | 7 | 6 – 8 | Detail level for the watertight conversion. Higher = slower but finer. |
| Point Cloud Size | 8192 | 8192 – 100000 | Points sampled from the surface. More points = better coverage of fine details. |
| Sampling | Off | — | Adds randomness to the generation. Turn on to get variation across runs. |
| Seed | 0 | 0 – 10000000 | Fixes the output for reproducibility. |

### MeshAnythingV2 Preprocess
Marching Cubes only — no model inference. Converts a non-watertight mesh into a clean watertight one. Useful to inspect or use the preprocessed geometry on its own.

| Parameter | Default | Range | Description |
|---|---|---|---|
| MC Octree Depth | 7 | 6 – 8 | Detail level for the watertight conversion. |

---

## Model

- **HuggingFace:** [Yiwen-ntu/MeshAnythingV2](https://huggingface.co/Yiwen-ntu/MeshAnythingV2)
- **Paper & source:** [github.com/buaacyw/MeshAnythingV2](https://github.com/buaacyw/MeshAnythingV2)
