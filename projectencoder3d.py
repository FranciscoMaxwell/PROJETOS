import os
import math
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tkinter import Tk, filedialog
import time

# ============================================================
# Configurações principais
# ============================================================
Tk().withdraw()
PASTA_MODELOS = filedialog.askdirectory(title="Selecione a pasta com os modelos .OBJ")
if not PASTA_MODELOS:
    raise ValueError("❌ Nenhuma pasta selecionada. O script será encerrado.")
print(f"📁 Pasta selecionada: {PASTA_MODELOS}")

RESOLUCAO = 32        # 32 rápido, 48/64 melhor qualidade (precisa de mais VRAM)
EPOCHS = 250         # ajuste conforme seu tempo/GPU
BATCH_SIZE = 8
LR = 1e-3
LATENT_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "modelos_treinados"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"▶ Usando dispositivo: {DEVICE}")

# ============================================================
# Função: carrega e voxeliza cada .obj, retorna grid (RESOLUCAO^3)
# ============================================================
def carregar_modelo_voxel(path, resolucao=RESOLUCAO, debug=False):
    try:
        mesh = trimesh.load(path, force='mesh')
        if mesh.is_empty:
            if debug: print("mesh vazio:", path)
            return None

        # centraliza
        mesh.apply_translation(-mesh.centroid)
        # escala para ocupar ~90% da caixa
        max_extent = np.max(mesh.extents)
        if max_extent <= 0:
            return None
        mesh.apply_scale(0.9 / max_extent)

        # voxeliza
        # pitch = tamanho do voxel; usando pitch = 1/resolução para obter grid aproximadamente resolucao^3
        pitch = 1.0 / resolucao
        voxel = mesh.voxelized(pitch=pitch)
        grid = voxel.matrix.astype(np.float32)

        # se o grid for maior que resolucao, centralizar e cortar/recortar
        fixed = np.zeros((resolucao, resolucao, resolucao), dtype=np.float32)
        # center the voxel grid inside fixed
        src_shape = grid.shape
        # compute start indices to center smaller grid into fixed
        offset = ((resolucao - src_shape[0]) // 2,
                  (resolucao - src_shape[1]) // 2,
                  (resolucao - src_shape[2]) // 2)
        # if source > target on any axis, we crop source centrally
        src_slices = []
        tgt_slices = []
        for i in range(3):
            if src_shape[i] <= resolucao:
                src_slices.append(slice(0, src_shape[i]))
                tgt_slices.append(slice(offset[i], offset[i] + src_shape[i]))
            else:
                # crop source
                start = (src_shape[i] - resolucao) // 2
                src_slices.append(slice(start, start + resolucao))
                tgt_slices.append(slice(0, resolucao))
        cropped = grid[src_slices[0], src_slices[1], src_slices[2]]
        fixed[tgt_slices[0], tgt_slices[1], tgt_slices[2]] = cropped
        return fixed
    except Exception as e:
        print(f"Erro ao carregar {os.path.basename(path)}: {e}")
        return None

# ============================================================
# Dataset PyTorch
# ============================================================
class VoxelDataset(Dataset):
    def __init__(self, folder):
        arquivos = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".obj")]
        data = []
        print("Carregando modelos:", len(arquivos))
        for i, a in enumerate(arquivos):
            g = carregar_modelo_voxel(a)
            if g is not None:
                data.append(g)
            else:
                print("pulado:", os.path.basename(a))
        if len(data) == 0:
            raise RuntimeError("Nenhum voxel válido carregado.")
        self.data = np.stack(data).astype(np.float32)
        # normaliza para 0-1 (já está binário, mas garantimos float)
        self.data = np.clip(self.data, 0.0, 1.0)
        print(f"Dataset compilado: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = np.expand_dims(x, axis=0)  # (1, D, D, D)
        return torch.tensor(x, dtype=torch.float32)

# ============================================================
# VAE 3D (Conv3D encoder / deconv decoder)
# ============================================================
class VAE3D(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1), # -> D/2
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1), # -> D/4
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), # -> D/8
            nn.BatchNorm3d(128), nn.ReLU(),
        )
        # compute flatten size
        # for RESOLUCAO divisible by 8, final spatial dims = RESOLUCAO/8
        final_dim = RESOLUCAO // 8
        self.flatten_size = 128 * final_dim * final_dim * final_dim
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # decoder
        self.fc_dec = nn.Linear(latent_dim, self.flatten_size)
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), # -> D/4
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), # -> D/2
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1), # -> D
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), 128, RESOLUCAO // 8, RESOLUCAO // 8, RESOLUCAO // 8)
        x_recon = self.dec(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# ============================================================
# Funções utilitárias (loss VAE)
# ============================================================
def loss_function(recon_x, x, mu, logvar):
    # BCE per voxel
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld

# ============================================================
# Carrega dataset e dataloader
# ============================================================
dataset = VoxelDataset(PASTA_MODELOS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = VAE3D(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# Treinamento
# ============================================================
print("\n🚀 Iniciando treinamento VAE 3D...\n")
best_loss = float('inf')
start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_kld = 0.0
    for batch in dataloader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss, bce, kld = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()

    avg_loss = total_loss / len(dataset)
    avg_bce = total_bce / len(dataset)
    avg_kld = total_kld / len(dataset)

    print(f"Ep {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | KLD: {avg_kld:.4f}")

    # salvar checkpoint ocasionalmente e sample
    if epoch % 25 == 0 or epoch == EPOCHS:
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"vae_epoch_{epoch}.pth"))
        # gerar amostra rápida
        model.eval()
        with torch.no_grad():
            # sample a partir da normal padrão -> decode
            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            sample = model.decode(z).cpu().numpy()[0, 0]  # (RESOLUCAO^3)
            np.save(os.path.join(OUT_DIR, f"sample_epoch_{epoch}.npy"), sample)
        model.train()

elapsed = time.time() - start_time
print(f"\n✅ Treinamento finalizado em {elapsed/60:.2f} min. Modelos e amostras salvos em: {OUT_DIR}")

# ============================================================
# Geração final: amostrar vários z, converter para mesh e salvar .obj
# ============================================================
model.eval()
NUM_SAMPLES = 3
THRESHOLD = 0.4  # ajuste para densidade do voxel antes do marching cubes
for i in range(NUM_SAMPLES):
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM).to(DEVICE)  # amostra padrão normal
        voxel = model.decode(z).cpu().numpy()[0, 0]  # shape (D,D,D)
    # binariza
    vox_bin = (voxel >= THRESHOLD).astype(np.uint8)

    # tenta converter para mesh (marching cubes). Usamos funções do trimesh se disponíveis.
    mesh = None
    try:
        # primeira tentativa: trimesh.voxel.ops.matrix_to_marching_cubes (algumas versões)
        try:
            mesh = trimesh.voxel.ops.matrix_to_marching_cubes(vox_bin, pitch=1.0)
        except Exception:
            # outra forma: criar VoxelGrid e usar marching_cubes ou as_boxes
            try:
                vg = trimesh.voxel.VoxelGrid(vox_bin)
                # some versions expose .marching_cubes attribute:
                if hasattr(vg, "marching_cubes") and vg.marching_cubes is not None:
                    mesh = vg.marching_cubes
                else:
                    # fallback: as_boxes (gera cubo por voxel -> pode ficar "lego")
                    mesh = vg.as_boxes()
            except Exception as e:
                print("falha ao criar mesh via VoxelGrid:", e)
                mesh = None

        if mesh is None:
            raise RuntimeError("Mesh não criado.")
        # salvar mesh
        nome = os.path.join(OUT_DIR, f"gerado_sample_{i+1}.obj")
        mesh.export(nome)
        print(f"🎉 Exportado: {nome}")

    except Exception as e:
        # em caso de falha ao exportar mesh, salvamos o ponto como npy
        print(f"⚠️ Erro ao transformar voxels em mesh: {e}")
        np.save(os.path.join(OUT_DIR, f"gerado_sample_{i+1}.npy"), voxel)
        print(f"Apenas .npy salvo: gerado_sample_{i+1}.npy")

print("\n🏁 Processo de geração concluído. Verifique os arquivos em:", OUT_DIR)
print("Dicas: abra o .obj no Blender e aplique Smooth / Remesh para melhorar a superfície.")
