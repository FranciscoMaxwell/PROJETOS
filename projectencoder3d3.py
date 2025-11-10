"""
train_deepsdf_multi_safe.py
Treina DeepSDF em múltiplos OBJ e gera malhas contínuas.
Projetado para lidar com múltiplos objetos diferentes.
Requisitos:
pip install numpy torch trimesh open3d scikit-image
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import trimesh
from skimage import measure
from tkinter import Tk, filedialog

# ================= CONFIG =================
LATENT_DIM = 1024       # vetor latente por objeto
LR = 1e-4
EPOCHS = 500            # reduzido para teste rápido
BATCH_SIZE = 8
NUM_SAMPLES = 5000      # pontos SDF por mesh
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ================= DIALOGO PASTA =================
Tk().withdraw()
PASTA_MODELOS = filedialog.askdirectory(title="Selecione a pasta com OBJ")
if not PASTA_MODELOS:
    raise SystemExit("Nenhuma pasta selecionada.")
print(f"Usando pasta: {PASTA_MODELOS}")

# ================= UTIL =================
def sample_sdf_from_mesh(mesh_path, num_samples=NUM_SAMPLES):
    mesh = trimesh.load(mesh_path, force='mesh')
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"Mesh inválido: {mesh_path}")

    mesh.apply_translation(-mesh.centroid)
    scale = max(mesh.extents)
    if scale > 0:
        mesh.apply_scale(1.0/scale)

    bbox_min = mesh.bounds[0] - 0.1
    bbox_max = mesh.bounds[1] + 0.1
    points = np.random.uniform(bbox_min, bbox_max, size=(num_samples, 3))
    sdf = trimesh.proximity.signed_distance(mesh, points)
    return points.astype(np.float32), sdf.astype(np.float32)

def build_dataset():
    arquivos = [f for f in os.listdir(PASTA_MODELOS) if f.lower().endswith(".obj")]
    if len(arquivos) == 0:
        raise RuntimeError("Nenhum OBJ encontrado.")

    data_pts, data_sdf, obj_ids = [], [], []
    for idx, f in enumerate(arquivos):
        try:
            pts, sdf = sample_sdf_from_mesh(os.path.join(PASTA_MODELOS, f))
            data_pts.append(pts)
            data_sdf.append(sdf)
            obj_ids.extend([idx]*pts.shape[0])
            print(f"[{idx+1}/{len(arquivos)}] Processado: {f}")
        except Exception as e:
            print(f"Erro em {f}: {e}")
    return (
        torch.tensor(np.concatenate(data_pts)),
        torch.tensor(np.concatenate(data_sdf)),
        torch.tensor(obj_ids)
    ), len(arquivos)

# ================= MODELO =================
class DeepSDF(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(3 + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, xyz, z):
        x = torch.cat([xyz, z], dim=-1)
        return self.mlp(x).squeeze(-1)

# ================= TREINO =================
def train(model, data_pts, data_sdf, obj_ids, num_objects):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Latentes individuais por objeto
    z_latent = nn.Parameter(torch.randn(num_objects, LATENT_DIM).to(DEVICE))
    optimizer.add_param_group({'params': z_latent})

    N = data_pts.shape[0]
    indices = np.arange(N)

    for epoch in range(1, EPOCHS+1):
        np.random.shuffle(indices)
        total_loss = 0.0
        for i in range(0, N, BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            pts_batch = data_pts[batch_idx].to(DEVICE)
            sdf_batch = data_sdf[batch_idx].to(DEVICE)
            ids_batch = obj_ids[batch_idx].to(DEVICE)
            z_batch = z_latent[ids_batch]

            pred_sdf = model(pts_batch, z_batch)
            loss = ((pred_sdf - sdf_batch)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * pts_batch.shape[0]

        print(f"[Ep {epoch}] Loss={total_loss/N:.6f}")

    return model, z_latent

# ================= GERAÇÃO =================
def generate_mesh(model, z_vec, resolution=64):
    model.eval()
    x = np.linspace(-1.2, 1.2, resolution)
    y = np.linspace(-1.2, 1.2, resolution)
    z = np.linspace(-1.2, 1.2, resolution)
    grid = np.stack(np.meshgrid(x, y, z), -1).reshape(-1,3)
    grid_t = torch.tensor(grid, dtype=torch.float32).to(DEVICE)
    z_batch = z_vec.repeat(grid_t.shape[0], 1)
    with torch.no_grad():
        sdf_vals = model(grid_t, z_batch).cpu().numpy()
    sdf_vals = sdf_vals.reshape(resolution, resolution, resolution)
    verts, faces, _, _ = measure.marching_cubes(sdf_vals, level=0)
    mesh = trimesh.Trimesh(vertices=verts/(resolution/2)-1, faces=faces)
    return mesh

# ================= MAIN =================
def main():
    t0 = time.time()
    (data_pts, data_sdf, obj_ids), num_objects = build_dataset()
    model = DeepSDF()
    model, z_latent = train(model, data_pts, data_sdf, obj_ids, num_objects)

    # Gera mesh para cada objeto individualmente
    for idx in range(num_objects):
        try:
            mesh = generate_mesh(model, z_latent[idx].unsqueeze(0))
            out_path = os.path.join(PASTA_MODELOS, f"deepsdf_mesh_obj{idx}_{int(time.time())}.obj")
            mesh.export(out_path)
            print(f"✅ Mesh gerada: {out_path}")
        except Exception as e:
            print(f"Erro gerando mesh do objeto {idx}: {e}")

    print(f"\nProcesso concluído em {(time.time()-t0)/60:.2f} minutos.")

if __name__ == "__main__":
    main()
