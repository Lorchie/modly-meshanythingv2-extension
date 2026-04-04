# MeshAnythingV2 – Modly Extension

High‑quality **mesh reconstruction** and **remeshing** using the official **MeshAnythingV2** model.

HuggingFace model:  
👉 https://huggingface.co/Yiwen-ntu/meshanythingv2

This extension converts noisy or irregular meshes into clean, CAD‑ready geometry using a transformer‑based 3D generator.

---

## 🚀 Features

- Mesh → clean reconstructed mesh  
- Optional watertight reconstruction (marching cubes)  
- Point‑cloud normalization  
- Deterministic or sampling‑based generation  
- CPU‑only compatible  
- Two‑node Modly pipeline: `preprocess` → `generate`

---

## 🧩 Nodes & Parameters

### **1. `preprocess`**
Converts an input mesh into a normalized point cloud.  
Useful before running the MeshAnythingV2 generator.

mc_level
- Resolution used for marching cubes
- Higher = more accurate but slower
- Used only when mc = true in the generate node


### **2. `generate`**
Runs the MeshAnythingV2 model to reconstruct a clean mesh.

face_number
- Approximate target number of faces
- MeshAnythingV2 does not strictly enforce this
mc
- Apply marching cubes before sampling the point cloud
- Helps fix holes and non‑manifold geometry
mc_level
- Resolution for marching cubes (same as preprocess)
no_pc_vertices
- Number of sampled points from the mesh
- 8192 recommended for stable results
sampling
- false → deterministic output
- true → stochastic decoding (variations possible)
seed
- Controls randomness when sampling is enabled



📝 License
MeshAnythingV2 is released under the MIT License.
This extension integrates the official implementation

---

🤝 Credits
- MeshAnythingV2 by [Buaacyw](https://github.com/buaacyw)
- Modly integration by [Lightning Pixel](https://github.com/lightningpixel)

