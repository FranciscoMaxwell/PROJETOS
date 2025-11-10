# train_3d_vae_and_classifier.py
"""
Treinamento VAE3D + Classificador 3D
Uso:
    python train_3d_vae_and_classifier.py --data_dir /caminho/Projeto_3D/modelos_voxelizados --out_dir out_models
"""

import os
import sys
import argparse
import math
import time
import json
import random
from glob import glob

import numpy as np
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

# --------------------------
# Utils: device detection + OOM-safe runner
# --------------------------
def get_device(prefer_gpu=True):
    try:
        if prefer_gpu and torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")

def try_run_with_fallback(fn, *args, prefer_gpu=True, **kwargs):
    """
    Tenta rodar `fn(*args, **kwargs)` no device preferido (cuda se disponível).
    Se ocorrer OOM, roda novamente em CPU e retorna (result, device_used, oom_flag).
    fn deve aceitar um argument device=...
    """
    device = get_device(prefer_gpu=prefer_gpu)
    try:
        return fn(device=device, *args, **kwargs), device, False
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            device_cpu = torch.device("cpu")
            try:
                return fn(device=device_cpu, *args, **kwargs), device_cpu, True
            except Exception as e2:
                raise e2
        else:
            raise e

# --------------------------
# Dataset
# --------------------------
class VoxelDataset(Dataset):
    """
    Dataset que carrega arquivos .npy contendo grids 3D (uint8 0/1 or float)
    Espera arquivos no formato: /.../<style>/<name>_*.npy
    Pode usar relatorio.csv para labels (opcional).
    """
    def __init__(self, root_dir, file_list=None, transform=None, dtype=np.float32, clip_threshold=0.5):
        self.root_dir = root_dir
        if file_list is None:
            # busca recursiva por .npy (não carrega masks aqui)
            self.files = []
            for sub in os.listdir(root_dir):
                path = os.path.join(root_dir, sub)
                if os.path.isdir(path):
                    self.files += glob(os.path.join(path, "*.npy"))
            # filtrar arquivos de mask se você tiver máscaras no mesmo diretório com suffix
            self.files = [f for f in self.files if "_class_" not in os.path.basename(f) and "_comp_" not in os.path.basename(f)]
        else:
            self.files = file_list
        self.transform = transform
        self.dtype = dtype
        self.clip_threshold = clip_threshold

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        arr = np.load(p)
        # garantir shape (D,D,D) ou (1,D,D,D)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        # normalizar para [0,1] float
        if arr.dtype == np.uint8 or arr.dtype == np.int:
            arr = arr.astype(np.float32) / 1.0
        else:
            arr = arr.astype(np.float32)
        # clip / threshold (caso seja probabilístico)
        arr = np.clip(arr, 0.0, 1.0)
        # garantir dimensão (1,D,D,D)
        arr = np.expand_dims(arr, axis=0)
        if self.transform:
            arr = self.transform(arr)
        return torch.tensor(arr, dtype=torch.float32), os.path.basename(p)

# --------------------------
# Modelos: VAE 3D
# --------------------------
class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, z_dim=256):
        super().__init__()
        # conv blocks - keep params small for 6GB VRAM
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.ReLU())
        # compute flattened size for 80->?
        # with 4 stride-2 ops: 80 -> 40 -> 20 -> 10 -> 5
        self.flatten_size = 256 * 5 * 5 * 5
        self.fc_mu = nn.Linear(self.flatten_size, z_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, z_dim)

    def forward(self, x):
        # x: (B,1,80,80,80)
        x = self.conv1(x)  # -> (B,32,40,40,40)
        x = self.conv2(x)  # -> (B,64,20,20,20)
        x = self.conv3(x)  # -> (B,128,10,10,10)
        x = self.conv4(x)  # -> (B,256,5,5,5)
        x = x.view(x.shape[0], -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder3D(nn.Module):
    def __init__(self, out_channels=1, z_dim=256):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 256 * 5 * 5 * 5)
        # transpose convs to upscale back to 80
        self.deconv1 = nn.Sequential(nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1), nn.BatchNorm3d(32), nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose3d(32, out_channels, 4, stride=2, padding=1))
        # after deconv4 -> (B,1,80,80,80) approximately
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.shape[0], 256, 5, 5, 5)
        x = self.deconv1(x)  # -> (B,128,10,10,10)
        x = self.deconv2(x)  # -> (B,64,20,20,20)
        x = self.deconv3(x)  # -> (B,32,40,40,40)
        x = self.deconv4(x)  # -> (B,1,80,80,80)
        x = torch.sigmoid(x)  # voxel occupancy prob
        return x

class VAE3D(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.encoder = Encoder3D(z_dim=z_dim)
        self.decoder = Decoder3D(z_dim=z_dim)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# --------------------------
# Classificador 3D simples (usa encoder backbone)
# --------------------------
class Classifier3D(nn.Module):
    def __init__(self, z_dim=256, n_classes=4):
        super().__init__()
        self.encoder = Encoder3D(z_dim=z_dim)  # we just use encoder to get mu/logvar
        # linear classifier
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        mu, logvar = self.encoder(x)
        # use mu as feature
        logits = self.classifier(mu)
        return logits

# --------------------------
# Losses
# --------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # reconstruction loss as BCE per voxel
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)  # per-batch average
    # KLD
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kld, recon_loss, kld

# --------------------------
# Training loop
# --------------------------
def train_vae_epoch(model, loader, optim, scaler, device, beta=1.0, accumulate_steps=1):
    model.train()
    pbar = tqdm(loader, desc="train")
    total_loss = 0.0
    recon_sum = 0.0
    kld_sum = 0.0
    step = 0
    optim.zero_grad()
    for x, _name in pbar:
        x = x.to(device)
        with autocast(enabled=(device.type=='cuda')):
            recon, mu, logvar = model(x)
            loss, recon_l, kld_l = vae_loss(recon, x, mu, logvar, beta=beta)
            loss = loss / accumulate_steps
        scaler.scale(loss).backward()
        step += 1
        if step % accumulate_steps == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
        total_loss += loss.item() * accumulate_steps
        recon_sum += recon_l.item()
        kld_sum += kld_l.item()
        pbar.set_postfix({"loss": total_loss/(step if step>0 else 1), "recon": recon_sum/(step if step>0 else 1), "kld": kld_sum/(step if step>0 else 1)})
    return total_loss/len(loader), recon_sum/len(loader), kld_sum/len(loader)

def validate_vae_epoch(model, loader, device, beta=1.0):
    model.eval()
    total_loss = 0.0
    recon_sum = 0.0
    kld_sum = 0.0
    with torch.no_grad():
        for x, _name in tqdm(loader, desc="val"):
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, recon_l, kld_l = vae_loss(recon, x, mu, logvar, beta=beta)
            total_loss += loss.item()
            recon_sum += recon_l.item()
            kld_sum += kld_l.item()
    n = len(loader)
    return total_loss/n, recon_sum/n, kld_sum/n

# --------------------------
# Checkpoint utils
# --------------------------
def save_checkpoint(path, model, optim, scaler, epoch):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None
    }
    torch.save(state, path)

def load_checkpoint(path, model, optim=None, scaler=None, device=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optim is not None and "optim_state" in ckpt and ckpt["optim_state"] is not None:
        optim.load_state_dict(ckpt["optim_state"])
    if scaler is not None and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0)

# --------------------------
# Main training orchestration
# --------------------------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device selection (attempt GPU)
    device = get_device(prefer_gpu=not args.force_cpu)
    print("Device:", device)

    # dataset
    ds = VoxelDataset(args.data_dir)
    n = len(ds)
    if n == 0:
        print("Nenhum .npy encontrado em", args.data_dir)
        return
    n_val = max( int(n * args.val_frac),  max(1, int(n*0.05)) )
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    print(f"Samples total: {n}, train: {len(train_ds)}, val: {len(val_ds)}")

    # dataloaders - batch size pequeno por VRAM
    batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1,args.num_workers//2), pin_memory=True)

    # model, optimizer, scaler
    vae = VAE3D(z_dim=args.z_dim).to(device)
    optimizer = Adam(vae.parameters(), lr=args.lr, weight_decay=1e-6)
    scaler = GradScaler(enabled=(device.type=='cuda'))

    best_val = 1e9
    start_epoch = 1
    if args.resume and os.path.exists(args.ckpt):
        try:
            start_epoch = load_checkpoint(args.ckpt, vae, optimizer, scaler, device) + 1
            print("Resume from epoch", start_epoch)
        except Exception as e:
            print("Falha ao carregar checkpoint:", e)

    # training loop
    for epoch in range(start_epoch, args.epochs+1):
        try:
            train_loss, train_recon, train_kld = train_vae_epoch(vae, train_loader, optimizer, scaler, device, beta=args.beta, accumulate_steps=args.accum)
            val_loss, val_recon, val_kld = validate_vae_epoch(vae, val_loader, device, beta=args.beta)
        except RuntimeError as e:
            # fallback on OOM: try CPU
            if 'out of memory' in str(e).lower():
                print("OOM detected. Retrying epoch on CPU. Consider lowering batch size.")
                device = torch.device("cpu")
                vae = VAE3D(z_dim=args.z_dim).to(device)
                optimizer = Adam(vae.parameters(), lr=args.lr, weight_decay=1e-6)
                scaler = GradScaler(enabled=False)
                train_loss, train_recon, train_kld = train_vae_epoch(vae, train_loader, optimizer, scaler, device, beta=args.beta, accumulate_steps=args.accum)
                val_loss, val_recon, val_kld = validate_vae_epoch(vae, val_loader, device, beta=args.beta)
            else:
                raise

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} recon={val_recon:.4f} kld={val_kld:.4f}")

        # checkpoint
        ckpt_path = os.path.join(args.out_dir, f"vae_epoch_{epoch}.pt")
        save_checkpoint(ckpt_path, vae, optimizer, scaler, epoch)

        # keep best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.out_dir, "vae_best.pt")
            save_checkpoint(best_path, vae, optimizer, scaler, epoch)
            print("New best saved:", best_path)

        # optionally save reconstructions on val set
        if epoch % args.save_every == 0:
            vae.eval()
            os.makedirs(args.samples_dir, exist_ok=True)
            with torch.no_grad():
                for i, (x, name) in enumerate(val_loader):
                    x = x.to(device)
                    recon, _, _ = vae(x)
                    recon_np = recon.detach().cpu().numpy()
                    # salvar primeiro batch (pode salvar todos, dependendo do espaço)
                    for b in range(recon_np.shape[0]):
                        fname = os.path.join(args.samples_dir, f"recon_e{epoch}_{i}_{b}_{name[b]}.npy")
                        np.save(fname, recon_np[b])
                    # só salvar a primeira mini-batch do val para economia
                    break

    print("Treino finalizado.")

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Pasta contendo modelos voxelizados (subpasta por estilo).")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="onde salvar checkpoints")
    parser.add_argument("--samples_dir", type=str, default="samples", help="onde salvar reconstrucao .npy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=1.0, help="peso do KLD")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default="checkpoints/vae_best.pt")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    main(args)
