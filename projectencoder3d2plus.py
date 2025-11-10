#!/usr/bin/env python3
# train_and_generate_3d_turbo_seguro.py
"""
Versão TurboSeguro — mudanças NÃO invasivas na arquitetura:
- scheduler de LR (ExponentialLR)
- peso KLD ajustável (KLD_WEIGHT) para forçar reconstrução melhor inicialmente
- salva estado do scheduler no checkpoint
- gera amostras .obj toda vez que salvamos checkpoint (gera com nome incluindo época)
- threshold adaptável na exportação (configurável)
Mantém a mesma arquitetura do modelo — compatível com teu .pth.
"""

import os
import time
import math
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import measure
from tkinter import Tk, filedialog

# ========== DIALOGO PASTA ==========
Tk().withdraw()
PASTA_MODELOS = filedialog.askdirectory(title="Selecione a pasta com os arquivos .OBJ")
if not PASTA_MODELOS:
    raise SystemExit("Nenhuma pasta selecionada. Saindo.")
print(f"Usando pasta: {PASTA_MODELOS}")

# ========== CONFIGURÁVEIS ==========
RESOLUCAO = 80         # 32 rápido / 48 melhor / 64+ muito custoso (RAM/GPU)
LATENT_DIM = 1024       # manter para compatibilidade
BATCH_SIZE = 12
LR = 1e-4
EPOCHS_PER_RUN = 4502    # quantas épocas treina a cada execução
SAVE_EVERY = 4000         # salva checkpoint a cada N épocas
GENERATE_ON_SAVE = True   # gerar amostras quando salvar checkpoint
NUM_OUTPUTS = 1         # quantos .obj gerar ao final / por checkpoint
THRESHOLD = 0.45        # threshold default para binarizar voxel antes de marching cubes
CHECKPOINT_NAME = os.path.join(PASTA_MODELOS, "vae3d_checkpoint.pth")
OUT_PREFIX = os.path.join(PASTA_MODELOS, "gerado")
VERBOSE = True
SEED = 42

# ===== TurboSeguro-specific =====
KLD_WEIGHT = 0.10            # reduz o peso do KLD para priorizar reconstrução inicialmente (compatível com .pth)
RECON_WEIGHT = 1.0
SCHED_GAMMA = 0.9995        # fator do ExponentialLR
MIN_LR = 1e-6               # limite inferior de LR
ADAPTIVE_THRESHOLD_MIN = 0.35  # ao gerar amostras intermediárias, permitir thresholds mais baixos
# =================================

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# ========== UTIL: carregar e voxelizar um .obj em memória ==========
def obj_to_voxel(path, resolucao=RESOLUCAO, debug=False):
    try:
        mesh = trimesh.load(path, force='mesh')
        if mesh.is_empty:
            if debug: print("Mesh vazio:", path)
            return None

        # centraliza e escala para 90% da caixa
        mesh.apply_translation(-mesh.centroid)
        max_extent = float(np.max(mesh.extents))
        if max_extent <= 0:
            return None
        mesh.apply_scale(0.9 / max_extent)

        # voxeliza com pitch ~ 1/resolucao
        pitch = 1.0 / resolucao
        voxel = mesh.voxelized(pitch=pitch)
        grid = voxel.matrix.astype(np.float32)  # (nx, ny, nz) binário 0/1

        # centro e encaixa no tamanho alvo RESOLUCAO^3
        fixed = np.zeros((resolucao, resolucao, resolucao), dtype=np.float32)
        src = grid
        src_shape = src.shape
        # calcular cortes e offsets centralizados
        src_slices = []
        tgt_slices = []
        for i in range(3):
            if src_shape[i] <= resolucao:
                start_src = 0
                start_tgt = (resolucao - src_shape[i]) // 2
                length = src_shape[i]
            else:
                start_src = (src_shape[i] - resolucao) // 2
                start_tgt = 0
                length = resolucao
            src_slices.append(slice(start_src, start_src + length))
            tgt_slices.append(slice(start_tgt, start_tgt + length))
        fixed[tgt_slices[0], tgt_slices[1], tgt_slices[2]] = src[src_slices[0], src_slices[1], src_slices[2]]
        return fixed
    except Exception as e:
        print(f"Erro voxelizar {os.path.basename(path)}: {e}")
        return None

# ========== Monta dataset em memória (usa a pasta escolhida) ==========
def build_dataset(resolucao=RESOLUCAO, verbose=VERBOSE):
    arquivos = [f for f in os.listdir(PASTA_MODELOS) if f.lower().endswith(".obj")]
    if len(arquivos) == 0:
        raise RuntimeError("Nenhum arquivo .obj encontrado na pasta selecionada.")
    voxels = []
    print(f"Encontrados {len(arquivos)} .obj — convertendo para voxels (res={resolucao})...")
    for i, f in enumerate(arquivos, 1):
        path = os.path.join(PASTA_MODELOS, f)
        v = obj_to_voxel(path, resolucao=resolucao)
        if v is None:
            print(f"Pulando {f}")
            continue
        voxels.append(v)
        if verbose:
            print(f"[{i}/{len(arquivos)}] {f} -> voxel shape {v.shape} | filled: {v.sum():.0f}")
    if len(voxels) == 0:
        raise RuntimeError("Nenhum voxel válido gerado a partir dos .obj.")
    data = np.stack(voxels).astype(np.float32)  # (N, D, D, D)
    data = np.clip(data, 0.0, 1.0)
    print(f"Dataset construído: {data.shape} (N modelos)")
    return torch.tensor(data).unsqueeze(1)  # (N, 1, D, D, D)

# ========== VAE 3D (MESMA ARQUITETURA) ==========
class VAE3D(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, resolucao=RESOLUCAO):
        super().__init__()
        # encoder conv layers (reduz espaço por 2 a cada camada)
        self.enc = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1), nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, 4, 2, 1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, 4, 2, 1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2),
        )
        final_spatial = resolucao // 16 if resolucao >= 32 else max(1, resolucao // 8)
        self.final_spatial = final_spatial
        self.flat_size = 256 * final_spatial * final_spatial * final_spatial
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

        # decoder
        self.fc_dec = nn.Linear(latent_dim, self.flat_size)
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 4, 2, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 4, 2, 1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 4, 2, 1), nn.Sigmoid(),
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
        h = h.view(-1, 256, self.final_spatial, self.final_spatial, self.final_spatial)
        x = self.dec(h)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ========== LOSS VAE (com peso customizável) ==========
def vae_loss(recon_x, x, mu, logvar, recon_weight=RECON_WEIGHT, kld_weight=KLD_WEIGHT):
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_weight * bce + kld_weight * kld, bce, kld

# ========== Geração: samplear z ~ N(0,1) e exportar ==========
def generate_and_export(model, out_prefix=OUT_PREFIX, num_samples=NUM_OUTPUTS, device=DEVICE, epoch=None, threshold=THRESHOLD):
    model = model.to(device)
    model.eval()
    for i in range(num_samples):
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(device)
            voxel = model.decode(z).cpu().numpy()[0, 0]  # (D,D,D)
        vox_bin = (voxel >= threshold).astype(np.float32)
        ts = int(time.time())
        ep_tag = f"_ep{epoch}" if epoch is not None else ""
        try:
            verts, faces, normals, values = measure.marching_cubes(vox_bin, level=0.5)
            nome = f"{out_prefix}{ep_tag}_{ts}_{i+1}.obj"
            with open(nome, "w") as f:
                for v in verts:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces + 1:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            print(f"Exportado (marching_cubes): {nome} | voxels filled: {vox_bin.sum():.0f}")
        except Exception as e:
            print("marching_cubes falhou:", e, "tentando fallback as_boxes()")
            try:
                vg = trimesh.voxel.VoxelGrid(vox_bin)
                mesh = vg.as_boxes()
                nome = f"{out_prefix}{ep_tag}_boxes_{ts}_{i+1}.obj"
                mesh.export(nome)
                print(f"Exportado (as_boxes fallback): {nome} | voxels filled: {vox_bin.sum():.0f}")
            except Exception as e2:
                print("Fallback também falhou:", e2)
                np.save(f"{out_prefix}_failed_voxel_{i+1}.npy", voxel)
                print(f"Salvo voxel .npy: {out_prefix}_failed_voxel_{i+1}.npy")

# ========== Treina / continua a partir do checkpoint (com scheduler) ==========
def train_and_checkpoint(model, data_tensor, device=DEVICE):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHED_GAMMA)
    N = data_tensor.shape[0]
    data_tensor = data_tensor.to(device)
    start_epoch = 1

    # tenta carregar checkpoint (mantendo compatibilidade)
    if os.path.exists(CHECKPOINT_NAME):
        ckpt = torch.load(CHECKPOINT_NAME, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        opt_state = ckpt.get("optim_state", None)
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception as e:
                print("Falha ao carregar estado do otimizador (compatibilidade) — continuará com novo otimizador:", e)
        # carregar scheduler state se existir (compatível)
        sched_state = ckpt.get("sched_state", None)
        if sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
            except Exception as e:
                print("Falha ao carregar estado do scheduler (compatibilidade) — continuará com novo scheduler:", e)
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"Checkpoint carregado. Continuando a partir da época {start_epoch}")

    dataset_size = N
    indices = np.arange(N)

    for epoch in range(start_epoch, start_epoch + EPOCHS_PER_RUN):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        total_bce = 0.0
        total_kld = 0.0
        batches = math.ceil(N / BATCH_SIZE)
        for b in range(batches):
            batch_idx = indices[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            batch = data_tensor[batch_idx].to(device)
            recon, mu, logvar = model(batch)
            loss, bce, kld = vae_loss(recon, batch, mu, logvar, recon_weight=RECON_WEIGHT, kld_weight=KLD_WEIGHT)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
        # step scheduler and clamp lr
        scheduler.step()
        for param_group in optimizer.param_groups:
            if param_group['lr'] < MIN_LR:
                param_group['lr'] = MIN_LR
        avg_loss = total_loss / dataset_size
        avg_bce = total_bce / dataset_size
        avg_kld = total_kld / dataset_size
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Ep {epoch}] loss={avg_loss:.4f} BCE={avg_bce:.4f} KLD={avg_kld:.4f} LR={current_lr:.6e}")

        # salva checkpoint periodicamente
        if (epoch % SAVE_EVERY == 0) or (epoch == start_epoch + EPOCHS_PER_RUN - 1):
            try:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict(),
                }, CHECKPOINT_NAME)
                print(f"Checkpoint salvo em {CHECKPOINT_NAME} (época {epoch})")
            except Exception as e:
                print("Falha ao salvar checkpoint:", e)

            # se configurado, gera amostras com threshold adaptativo (mais permissivo para épocas intermediárias)
            if GENERATE_ON_SAVE:
                adapt_thresh = max(ADAPTIVE_THRESHOLD_MIN, THRESHOLD * (0.98 ** (epoch // SAVE_EVERY)))
                print(f"Gerando amostras de checkpoint com threshold={adapt_thresh:.3f} (época {epoch})")
                generate_and_export(model, out_prefix=OUT_PREFIX, num_samples=NUM_OUTPUTS, device=device, epoch=epoch, threshold=adapt_thresh)

    return model

# ========== MAIN ==========
def main():
    t0 = time.time()
    data = build_dataset(resolucao=RESOLUCAO)
    model = VAE3D(latent_dim=LATENT_DIM, resolucao=RESOLUCAO)
    print("Modelo criado. Parâmetros:", sum(p.numel() for p in model.parameters()))
    model = train_and_checkpoint(model, data, device=DEVICE)
    # geração final com threshold padrão
    generate_and_export(model, out_prefix=OUT_PREFIX, num_samples=NUM_OUTPUTS, device=DEVICE, epoch="final", threshold=THRESHOLD)
    elapsed = time.time() - t0
    print(f"\nProcesso concluído em {elapsed/60:.2f} minutos. Cheque os .obj gerados em: {PASTA_MODELOS}")

if __name__ == "__main__":
    main()
