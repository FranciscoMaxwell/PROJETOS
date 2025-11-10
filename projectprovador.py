"""
provador3d_full.py
Protótipo: provador virtual 3D (mannequin-like) a partir de fotos multi-view
- GUI simples (tkinter) para selecionar imagens (frente, costas, esquerda, direita)
- pré-process: remover fundo (rembg), extrair silhuetas e keypoints (MediaPipe)
- gerar malha anônima (mannequin) por combinação de silhuetas -> volume -> marching cubes
- visualizar em janela 3D com possibilidade de carregar uma roupa (.obj) e exportar o conjunto

Uso:
    python provador3d_full.py

Saída:
    export/body.obj
    export/body_vestido.obj (quando exportado)
"""

import os, sys, math, tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, Button, Label
from PIL import Image, ImageOps, ImageFilter
import numpy as np

# imports opcionais que podem não existir — tentamos importar de forma graciosa
try:
    import rembg
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

try:
    from skimage import measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import trimesh
    import pyrender
    _HAS_RENDER = True
except Exception:
    _HAS_RENDER = False

# diretórios
BASE_DIR = Path.cwd()
EXPORT_DIR = BASE_DIR / "export"
EXPORT_DIR.mkdir(exist_ok=True)

# parâmetros (ajustáveis)
VOXEL_RES = 128            # resolução do grid 3D para visual-hull (aumente para maior detalhe; consome memória)
CANVAS_SIZE = 512          # normalização de imagens (px)
MANNEQUIN_SCALE = 1.0      # escala final
ANTI_BORDER_ALPHA = 20     # threshold alpha para "anti-borda"

# utils --------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def show_error(msg):
    messagebox.showerror("Erro", msg)

# background removal (prefer rembg)
def remove_background_pil(pil_img):
    """Remove background usando rembg se disponível, senão fallback de matting por cor."""
    try:
        if _HAS_REMBG:
            # rembg espera bytes; usa função remove do pacote
            from io import BytesIO
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            input_bytes = buf.getvalue()
            out_bytes = rembg.remove(input_bytes)
            out_img = Image.open(BytesIO(out_bytes)).convert("RGBA")
            return out_img
        else:
            # fallback: conversão simples por limite de saturação/luminosidade
            img = pil_img.convert("RGBA")
            gray = img.convert("L")
            # threshold by Otsu-like heuristic
            thresh = np.array(gray).mean() * 0.9
            mask = np.array(gray) > thresh
            # refine mask by morphological ops
            mask_img = Image.fromarray((mask*255).astype('uint8'))
            mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
            mask_img = mask_img.filter(ImageFilter.MinFilter(3))
            out = Image.new("RGBA", img.size, (0,0,0,0))
            out.paste(img, mask=mask_img)
            return out

    except Exception as e:
        print("Erro remove_background:", e)
        return pil_img.convert("RGBA")

# anti-borda and silhouette cleaning
def clean_mask_alpha(pil_rgba, alpha_threshold=ANTI_BORDER_ALPHA):
    """Zera pixels com alpha < threshold e aplica leve blur na alpha para suavizar transição."""
    im = pil_rgba.convert("RGBA")
    data = np.array(im)
    alpha = data[...,3]
    alpha[alpha < alpha_threshold] = 0
    data[...,3] = alpha
    im2 = Image.fromarray(data)
    # blur alpha channel a little
    im2 = im2.filter(ImageFilter.GaussianBlur(radius=0.8))
    # re-binarize small alpha
    a2 = np.array(im2)[...,3]
    a2[a2 < alpha_threshold//2] = 0
    arr = np.array(im2)
    arr[...,3] = a2
    return Image.fromarray(arr)

# resize & center into square canvas
def normalize_image(pil_img, size=CANVAS_SIZE):
    im = pil_img.convert("RGBA")
    w,h = im.size
    side = max(w,h)
    canvas = Image.new("RGBA", (side,side), (0,0,0,0))
    canvas.paste(im, ((side-w)//2, (side-h)//2), im)
    canvas = canvas.resize((size,size), Image.LANCZOS)
    return canvas

# use MediaPipe to extract pose landmarks (optional, for future improvements)
def extract_keypoints(pil_img):
    if not _HAS_MEDIAPIPE:
        return None
    import cv2
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(img)
        if not res or not res.pose_landmarks:
            return None
        pts = [(lm.x, lm.y, lm.z, lm.visibility) for lm in res.pose_landmarks.landmark]
        return pts

# main geometric reconstruction (visual-hull voxel carving simplified)
def silhouettes_to_volume(masks, voxel_res=VOXEL_RES):
    """
    masks: list of 2D numpy bool arrays (same size). We assume views are: front, back, left, right (if provided)
    We'll build an approximate 3D occupancy grid by extruding silhouettes and intersecting projections.
    Return a 3D numpy boolean volume (z,y,x) where True = occupied.
    """
    # simple approach: for each view, extrude silhouette along depth axis; then intersect
    # We assume canonical camera axes: front/back along Z; left/right along X.
    H, W = masks[0].shape
    # create normalized coordinates grid in [-1,1]
    # create volume with coords (Z,Y,X)
    vol = np.ones((voxel_res, voxel_res, voxel_res), dtype=bool)
    ys = np.linspace(-1,1,voxel_res)
    xs = np.linspace(-1,1,voxel_res)
    zs = np.linspace(-1,1,voxel_res)
    # Precompute mapping from voxel x,y,z to image pixel for each view
    # front view: project (X,Y,Z) -> image (u = X, v = -Y)
    # For performance, vectorize per slice:
    # We'll iterate voxels and test mask inclusion per view
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='xy')  # shapes: (vox_y,vox_x,vox_z) but we'll transpose below
    # rearrange to (z,y,x)
    # create coordinates arrays in shape (z,y,x)
    X = np.transpose(xx, (2,0,1))
    Y = np.transpose(yy, (2,0,1))
    Z = np.transpose(zz, (2,0,1))
    occ = np.ones_like(X, dtype=bool)
    Hm, Wm = H, W

    # helper: sample mask at normalized coords (u,v in [-1,1]) -> mask boolean
    def sample_mask(mask, u, v):
        # u corresponds to X -> column, v corresponds to -Y -> row
        col = ((u + 1)/2 * (Wm-1)).astype(int)
        row = (((-v) + 1)/2 * (Hm-1)).astype(int)
        col = np.clip(col, 0, Wm-1)
        row = np.clip(row, 0, Hm-1)
        return mask[row, col]

    # front mask: silhouette projected along Z: require (X,Y) in mask
    if len(masks) >= 1:
        mask_f = masks[0].astype(bool)
        u = X; v = Y
        samp = sample_mask(mask_f, u, v)
        occ &= samp
    # back mask: project (-X,Y)
    if len(masks) >= 2:
        mask_b = masks[1].astype(bool)
        u = -X; v = Y
        samp = sample_mask(mask_b, u, v)
        occ &= samp
    # left mask: project (Z,Y) onto u=Z (depth) axis
    if len(masks) >= 3:
        mask_l = masks[2].astype(bool)
        u = Z; v = Y
        samp = sample_mask(mask_l, u, v)
        occ &= samp
    # right mask:
    if len(masks) >= 4:
        mask_r = masks[3].astype(bool)
        u = -Z; v = Y
        samp = sample_mask(mask_r, u, v)
        occ &= samp

    return occ  # boolean volume

# marching cubes to mesh
def volume_to_mesh(vol, threshold=0.5):
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image required for marching_cubes (pip install scikit-image)")
    # vol expected boolean; convert to float
    volf = vol.astype(np.float32)
    verts, faces, normals, values = measure.marching_cubes(volf, level=0.5)
    # marching_cubes returns verts in voxel coordinates; normalize to [-1,1] cube
    # verts shape (N,3) in (z,y,x) coords from 0..(res-1)
    res = vol.shape[0]
    verts_norm = (verts / (res - 1)) * 2.0 - 1.0
    # reorder axes from (z,y,x) to (x,y,z)
    verts_xyz = verts_norm[:, [2,1,0]] * MANNEQUIN_SCALE
    return verts_xyz, faces

# save OBJ
def save_obj(path, verts, faces):
    with open(path, 'w', encoding='utf-8') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

# viewer (pyrender)
def show_viewer(mesh_vertices, mesh_faces, overlay_obj_path=None):
    if not _HAS_RENDER:
        messagebox.showinfo("Viewer", f"Exportado em export/body.obj (pode abrir com seu visualizador 3D).")
        return
    # create trimesh
    import trimesh
    import pyrender
    m = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(m, smooth=False)
    scene.add(mesh)
    # overlay cloth if given
    if overlay_obj_path and os.path.exists(overlay_obj_path):
        c = trimesh.load(overlay_obj_path, force='mesh')
        try:
            cm = pyrender.Mesh.from_trimesh(c, smooth=False)
            scene.add(cm)
        except Exception:
            # fallback: add trimesh as primitive
            pass
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

# GUI app ---------------------------
class TryOnApp:
    def __init__(self, root):
        self.root = root
        root.title("Provador 3D — protótipo")
        root.geometry("520x240")

        Label(root, text="Selecione as fotos (frente, costas, esquerda, direita) — 2 a 4 vistas").pack(pady=6)
        Button(root, text="Selecionar imagens", command=self.select_images, width=30).pack(pady=2)
        Button(root, text="Processar e gerar mannequin 3D", command=self.process_and_build, width=30).pack(pady=6)
        Button(root, text="Carregar roupa (.obj) e visualizar", command=self.load_cloth_and_view, width=30).pack(pady=6)
        Button(root, text="Exportar BODY.obj para export/ e abrir pasta", command=self.export_and_open, width=30).pack(pady=6)

        self.selected_paths = []
        self.masks = []
        self.mesh = None
        self.faces = None
        self.cloth_path = None

    def select_images(self):
        paths = filedialog.askopenfilenames(title="Escolha 2..4 fotos (frente, costas, esquerda, direita)",
                                            filetypes=[("Imagens","*.png;*.jpg;*.jpeg;*.bmp")])
        if not paths:
            return
        self.selected_paths = list(paths)
        messagebox.showinfo("OK", f"{len(self.selected_paths)} imagens selecionadas.")

    def process_and_build(self):
        if not self.selected_paths:
            show_error("Nenhuma imagem selecionada.")
            return
        # process each image: remove bg, normalize, clean alpha -> build mask array
        masks = []
        for p in self.selected_paths:
            pil = Image.open(p).convert("RGBA")
            pil_no_bg = remove_background_pil(pil)
            pil_clean = clean_mask_alpha(pil_no_bg)
            pil_norm = normalize_image(pil_clean, size=CANVAS_SIZE)
            arr = np.array(pil_norm)[...,3]  # alpha
            mask = (arr > 0).astype(np.uint8)
            masks.append(mask)
        self.masks = masks
        # build occupancy volume
        vol = silhouettes_to_volume(masks, voxel_res=VOXEL_RES)
        # simplify volume a little by morphological closing to get nicer hull
        vol = vol.astype(np.uint8)
        # marching cubes to mesh
        if vol.sum() == 0:
            show_error("Volume vazio — verifique as máscaras.")
            return
        verts, faces = volume_to_mesh(vol)
        self.mesh = verts
        self.faces = faces
        # save to export/body.obj
        export_path = EXPORT_DIR / "body.obj"
        save_obj(export_path, verts, faces)
        messagebox.showinfo("Sucesso", f"Mannequin gerado e salvo em:\n{export_path}")

    def load_cloth_and_view(self):
        if self.mesh is None:
            show_error("Gere primeiro o mannequin 3D.")
            return
        p = filedialog.askopenfilename(title="Escolha a roupa (.obj)", filetypes=[("OBJ","*.obj")])
        if not p:
            return
        self.cloth_path = p
        # visualize
        show_viewer(self.mesh, self.faces, overlay_obj_path=self.cloth_path)

    def export_and_open(self):
        if self.mesh is None:
            show_error("Gere o mannequin primeiro.")
            return
        out_body = EXPORT_DIR / "body.obj"
        save_obj(out_body, self.mesh, self.faces)
        # if cloth loaded, export naive merged mesh (no physical drape) by saving separate obj
        if self.cloth_path:
            out_cloth = EXPORT_DIR / Path(self.cloth_path).name
            import shutil
            shutil.copy(self.cloth_path, out_cloth)
            # The user can use Blender for draping later
        # open folder
        try:
            if sys.platform == "win32":
                os.startfile(EXPORT_DIR)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", EXPORT_DIR])
        except Exception:
            pass
        messagebox.showinfo("Exportado", f"Arquivos salvos em:\n{EXPORT_DIR}")

# main
if __name__ == "__main__":
    root = tk.Tk()
    app = TryOnApp(root)
    root.mainloop()
