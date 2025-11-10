#!/usr/bin/env python3
# organiza_voxel_treinamento_v3_part1.py
"""
Parte 1/3 - Imports, configuração, detecção Blender, script temporário do Blender e funções utilitárias.
Versão robusta: imports protegidos, carga opcional de modelo semântico (CLIP fallback),
flags de disponibilidade, config file, e run_blender_export seguro.
"""

import os
import sys
import shutil
import json
import time
import math
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from datetime import datetime
from tkinter import Tk, filedialog
import logging
import csv
import traceback


# --- configurações de logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------
# Dependências opcionais (import protegido)
# -----------------------------
HAS_TORCH = False
torch = None
DEVICE = "cpu"

try:
    import torch
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    HAS_TORCH = False
    DEVICE = "cpu"

# libs que são essenciais para pipeline (trimesh/numpy/tqdm/PIL)
try:
    import trimesh
except Exception as e:
    logging.error("trimesh não encontrado. Instale: pip install trimesh")
    raise

try:
    import numpy as np
except Exception:
    logging.error("numpy não encontrado. Instale: pip install numpy")
    raise

try:
    from tqdm import tqdm
except Exception:
    logging.error("tqdm não encontrado. Instale: pip install tqdm")
    raise

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# sklearn cosine_similarity (opcional se semantic disponível)
try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# transformers / CLIP model carregados de forma protegida
MODEL = None
TOKENIZER = None
TEXT_EMB = None
SEMANTIC_MODEL_AVAILABLE = False

try:
    # tentar carregar CLIPModel (API moderna)
    from transformers import CLIPModel, CLIPTokenizer
    try:
        TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        if HAS_TORCH and DEVICE == "cuda":
            MODEL = MODEL.to(DEVICE)
        SEMANTIC_MODEL_AVAILABLE = True
        logging.info("[Semantic] CLIPModel carregado (openai/clip-vit-base-patch16).")
    except Exception as e:
        logging.warning(f"[Semantic] Falha ao carregar CLIPModel: {e}")
        MODEL = None
        TOKENIZER = None
        SEMANTIC_MODEL_AVAILABLE = False
except Exception:
    # transformers/CLIP não disponível -> semantic disabled
    SEMANTIC_MODEL_AVAILABLE = False

# Vocabulário universal — objetos de todo tipo
LABELS = [
    # utensílios e domésticos
    "cup", "mug", "spoon", "fork", "knife", "plate", "bottle", "bowl", "glass",
    "table", "chair", "sofa", "television", "remote control", "lamp",
    "bed", "pillow", "blanket", "book", "pen", "phone", "laptop",
    # transporte
    "car", "bus", "truck", "motorcycle", "bicycle", "train", "airplane", "boat",
    "key", "steering wheel", "seat", "tire", "mirror",
    # estruturas e arquitetura
    "house", "building", "door", "window", "roof", "antenna", "stairs",
    "bridge", "tower", "wall", "floor", "fence", "garage",
    # corpo humano
    "human", "head", "hand", "foot", "leg", "arm", "face", "body",
    # natureza
    "tree", "rock", "flower", "grass", "mountain", "water", "cloud",
    # tecnológicos
    "robot", "drone", "machine", "helmet", "camera", "tool",
    # interiores
    "kitchen", "bedroom", "bathroom", "car interior", "office", "street",
    # animais
    "animal", "cat", "dog", "horse", "bird", "fish", "insect"
]

# gerar embeddings textuais se possível (numérico e normalizado)
if SEMANTIC_MODEL_AVAILABLE and HAS_SKLEARN:
    try:
        # tokenizer -> tensores (sem mover para device aqui; CLIPTokenizer returns dict of lists)
        # CLIPModel from transformers expects inputs via feature extractor or tokenizer differently;
        # para simplificar e tornar robusto, vamos usar a API de text inputs do tokenizer e do model.
        text_inputs = TOKENIZER(LABELS, padding=True, return_tensors="pt", truncation=True)
        if HAS_TORCH and DEVICE == "cuda":
            text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
        with torch.no_grad():
            if hasattr(MODEL, "get_text_features"):
                TEXT_EMB = MODEL.get_text_features(**text_inputs)
            else:
                out = MODEL.get_input_embeddings()(text_inputs["input_ids"])
                TEXT_EMB = out.mean(dim=1)
        if TEXT_EMB is not None and hasattr(TEXT_EMB, "norm"):
            TEXT_EMB = TEXT_EMB / (TEXT_EMB.norm(dim=-1, keepdim=True) + 1e-9)
        logging.info("[Semantic] Cache textual pronto.")
    except Exception as e:
        logging.warning(f"[Semantic] Falha ao gerar embeddings textuais: {e}")
        TEXT_EMB = None
        SEMANTIC_MODEL_AVAILABLE = False
else:
    if not SEMANTIC_MODEL_AVAILABLE:
        logging.info("[Semantic] Modelo semântico não disponível. Funções semânticas serão heurísticas.")
    elif not HAS_SKLEARN:
        logging.info("[Semantic] sklearn não disponível — similaridade baseada em ML desabilitada.")
    TEXT_EMB = None

# -----------------------------
# Configurações principais
# -----------------------------
DEFAULT_VOXEL_RES = 80
OUT_ROOT_DIRNAME = "Projeto_3D"
CONFIG_FILE = os.path.expanduser("~/.organiza_voxel_config_v3.json")
MAX_WORKERS = 2
BLENDER_SEARCH_PATHS = [
    r"C:\Program Files\Blender Foundation\Blender\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    "/usr/bin/blender",
    "/usr/local/bin/blender",
    "/Applications/Blender.app/Contents/MacOS/Blender"
]
SUPPORTED_INPUT_EXTS = (".obj", ".fbx", ".glb", ".gltf", ".blend", ".ply", ".stl", ".dae", ".3ds", ".abc", ".usd")
TEXTURE_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr")

# Heurísticas / mapeamentos
CATEGORY_TARGETS = {
    "child": 1.3,
    "adolescent": 1.5,
    "adult": 1.7,
    "tall_adult": 1.9,
    "animal": 1.2,
    "mount": 2.0,
    "robot": 2.2,
    "prop": 1.0,
    "scene": 8.0
}

STYLE_BUCKETS = {
    "low": (0, 2000),
    "mid": (2000, 30000),
    "high": (30000, 999999999)
}

ACCESSORY_KEYWORDS = ["hat","cap","helmet","sword","blade","knife","umbrella","gun","pistol","shield","backpack","bag","axe","spear","staff","weapon","prop","glove","boots","wing","wings","tail","armor","cape"]
MOUNT_KEYWORDS = ["horse","drag","dragon","mount","ride","steed","pony","bike","motor","cavalry","fenix","phoenix"]
HUMAN_KEYWORDS = ["human","man","woman","female","male","person","character","body","avatar","girl","boy","child","adult","teen"]
FURRY_KEYWORDS = ["fur","furry","pelt","hairy","scales","spikes","spike","quill"]
VEHICLE_KEYWORDS = ["car","truck","vehicle","automobile","sedan","coupe","van","bus","jeep","auto","motor","bike","motorbike","motorcycle"]

# -----------------------------
# utilitários
# -----------------------------
def safe_makedirs(p):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# -----------------------------
# Blender detection / config
# -----------------------------

def find_blender():
    cfg = load_json(CONFIG_FILE)
    if cfg and "blender_path" in cfg and os.path.exists(cfg["blender_path"]):
        return cfg["blender_path"]
    for p in BLENDER_SEARCH_PATHS:
        if os.path.exists(p):
            return p
    for cmd in ("blender", "blender.exe"):
        try:
            res = shutil.which(cmd)
            if res:
                return res
        except Exception:
            pass
    return None


def ask_blender_path_and_save():
    logging.info("Blender não encontrado automaticamente. Pedir ao usuário o caminho.")
    Tk().withdraw()
    path = filedialog.askopenfilename(title="Selecione o executável do Blender")
    if path and os.path.exists(path):
        save_json(CONFIG_FILE, {"blender_path": path})
        return path
    return None

# -----------------------------
# blender tmp script (melhorado: exporta objetos separados, detecta armature, partículas)
# -----------------------------
# Observação: Template usa JSON para devolver metadados. Mantive o conteúdo funcional; a parte 2
# fará integração da saída meta para máscaras semânticas/voxel masks.
BLENDER_TMP_SCRIPT_TEMPLATE = r'''
import bpy, sys, json, os, math

# argumentos passados via -- "<src>" "<out_fbx>" "<meta_json>"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

src = argv[0] if len(argv) > 0 else None
out_fbx = argv[1] if len(argv) > 1 else None
meta_out = argv[2] if len(argv) > 2 else None

meta = {"has_armature": False, "num_armatures": 0, "has_animation": False, "objects": 0, "verts": 0, "has_textures": False, "materials": [], "particle_systems": 0, "poses": []}

try:
    ext = os.path.splitext(src)[1].lower()
    # importar conforme extensão
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=src)
    else:
        if ext in (".fbx",):
            bpy.ops.import_scene.fbx(filepath=src)
        elif ext in (".obj",):
            bpy.ops.import_scene.obj(filepath=src)
        elif ext in (".gltf", ".glb"):
            bpy.ops.import_scene.gltf(filepath=src)
        else:
            try:
                bpy.ops.wm.open_mainfile(filepath=src)
            except Exception:
                pass
    bpy.context.view_layer.update()
    objs = list(bpy.data.objects)
    meta["objects"] = len(objs)
    # collect materials and detect features
    for o in objs:
        try:
            if o.type == 'ARMATURE':
                meta["has_armature"] = True
                meta["num_armatures"] += 1
            if getattr(o, "animation_data", None) and getattr(o.animation_data, "action", None):
                meta["has_animation"] = True
            if o.type == 'MESH':
                meta["verts"] += len(o.data.vertices)
                for ms in o.material_slots:
                    if not ms.material:
                        continue
                    meta.setdefault("materials", [])
                    meta["materials"].append(ms.material.name)
                    nodes = getattr(ms.material, "node_tree", None)
                    if nodes:
                        for node in nodes.nodes:
                            if node.type == 'TEX_IMAGE':
                                meta["has_textures"] = True
            if getattr(o, "particle_systems", None):
                if len(o.particle_systems) > 0:
                    meta["particle_systems"] += len(o.particle_systems)
        except Exception:
            pass

    # apply transforms safely
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception:
        pass

    # export fbx
    if out_fbx:
        try:
            bpy.ops.export_scene.fbx(filepath=out_fbx, embed_textures=False, path_mode='COPY', apply_scale_options='FBX_SCALE_ALL')
        except Exception:
            pass

    # Pose analysis using armature if available
    poses = set()
    armatures = [o for o in bpy.data.objects if o.type=='ARMATURE']
    if armatures:
        for arm in armatures:
            try:
                meta["has_armature"] = True
                bpy.context.view_layer.objects.active = arm
                bpy.ops.object.mode_set(mode='POSE')
                pbones = arm.pose.bones
                names = {b.name.lower(): b for b in pbones}
                def find_bone(keys):
                    for k in keys:
                        for n,b in names.items():
                            if k in n:
                                return b
                    return None
                head_b = find_bone(['head','neck'])
                hip_b = find_bone(['hip','hips','pelvis','root','spine'])
                l_hand = find_bone(['hand.l','hand_l','left_hand','left'])
                r_hand = find_bone(['hand.r','hand_r','right_hand','right'])
                bones = list(pbones)
                if not head_b and bones:
                    head_b = bones[0]
                if not hip_b and bones:
                    hip_b = bones[int(len(bones)/2)]
                def bone_pos(b):
                    try:
                        return (arm.matrix_world @ b.head).x, (arm.matrix_world @ b.head).y, (arm.matrix_world @ b.head).z
                    except Exception:
                        try:
                            return (b.head.x, b.head.y, b.head.z)
                        except Exception:
                            return (0,0,0)
                hip_pos = bone_pos(hip_b) if hip_b else None
                head_pos = bone_pos(head_b) if head_b else None
                lpos = bone_pos(l_hand) if l_hand else None
                rpos = bone_pos(r_hand) if r_hand else None
                if hip_pos and head_pos:
                    v = (head_pos[0]-hip_pos[0], head_pos[1]-hip_pos[1], head_pos[2]-hip_pos[2])
                    vz = abs(v[2])
                    norm = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) + 1e-9
                    vertical_ratio = vz / norm
                    if vertical_ratio > 0.7 and norm > 0.4:
                        poses.add('standing')
                    elif vertical_ratio > 0.4 and norm > 0.2:
                        poses.add('sitting_or_crouched')
                    else:
                        poses.add('lying')
                if head_pos and lpos and math.sqrt((lpos[0]-head_pos[0])**2 + (lpos[1]-head_pos[1])**2 + (lpos[2]-head_pos[2])**2) < 0.25:
                    poses.add('left_hand_on_head')
                if head_pos and rpos and math.sqrt((rpos[0]-head_pos[0])**2 + (rpos[1]-head_pos[1])**2 + (rpos[2]-head_pos[2])**2) < 0.25:
                    poses.add('right_hand_on_head')
                if getattr(arm, 'animation_data', None) and getattr(arm.animation_data, 'action', None):
                    poses.add('animated')
            except Exception:
                pass
    else:
        # fallback direct mesh heuristic
        try:
            meshes = [o for o in bpy.data.objects if o.type=='MESH']
            if meshes:
                coords = []
                for m in meshes:
                    try:
                        for v in m.data.vertices:
                            co = m.matrix_world @ v.co
                            coords.append((co.x, co.y, co.z))
                    except Exception:
                        pass
                if coords:
                    zs = [c[2] for c in coords]
                    height = max(zs)-min(zs) if zs else 0
                    if height > 1.1:
                        poses.add('standing')
                    elif height > 0.5:
                        poses.add('sitting_or_crouched')
                    else:
                        poses.add('lying')
        except Exception:
            pass

    meta['poses'] = list(poses)

except Exception:
    pass

# write meta
try:
    if meta_out:
        with open(meta_out, 'w', encoding='utf-8') as f:
            json.dump(meta, f)
except Exception:
    pass
'''
# -----------------------------
# Run Blender export (usa o template acima)
# -----------------------------

def run_blender_export(blender_bin, src_path, out_dir, timeout=600):
    """
    Executa Blender headless para exportar e coletar meta JSON.
    Retorna (out_fbx_path | None, meta_dict)
    """
    safe_makedirs(out_dir)
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_fbx = os.path.join(out_dir, f"{base}_export.fbx")
    meta_json = os.path.join(out_dir, f"{base}_meta.json")
    tmp_script = os.path.join(out_dir, f"tmp_blender_export_{int(time.time()*1000)}.py")
    try:
        # escrever o script temporário (com overwrite seguro)
        with open(tmp_script, "w", encoding="utf-8") as f:
            f.write(BLENDER_TMP_SCRIPT_TEMPLATE)
        cmd = [blender_bin, "-b", "--python", tmp_script, "--", src_path, out_fbx, meta_json]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        meta = {}
        if os.path.exists(meta_json):
            try:
                with open(meta_json, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
            except Exception:
                meta = {}
        if os.path.exists(out_fbx):
            return out_fbx, meta
        else:
            stderr = proc.stderr.decode("utf-8", errors="ignore") if proc and proc.stderr else ""
            return None, {"error": "export_failed", "stderr": stderr}
    except subprocess.TimeoutExpired:
        return None, {"error": "timeout"}
    except Exception as e:
        return None, {"error": str(e)}
    finally:
        try:
            if os.path.exists(tmp_script):
                os.remove(tmp_script)
        except Exception:
            pass

# Fim da parte 1/3

# -----------------------------
# Parte 2/3 - Heurísticas, detecção semântica, voxelização e geração de máscaras
# -----------------------------

# dependência opcional para rotulagem de componentes 3D
try:
    from scipy import ndimage as scipy_ndimage
    HAS_SCIPY = True
except Exception:
    scipy_ndimage = None
    HAS_SCIPY = False
    logging.info("[Mask] scipy.ndimage não encontrado — connected-component masking será limitado.")

# -----------------------------
# Heurísticas avançadas
# -----------------------------
def detect_category_by_name_and_mesh(name, mesh):
    """
    Retorna (category_str, target_height_meters)
    Usa heurísticas por nome e bounding-box do mesh.
    """
    lower = (name or "").lower()
    # name-based overrides
    if any(k in lower for k in MOUNT_KEYWORDS):
        return "mount", CATEGORY_TARGETS["mount"]
    if any(k in lower for k in VEHICLE_KEYWORDS):
        return "prop", CATEGORY_TARGETS["prop"]
    if any(k in lower for k in HUMAN_KEYWORDS):
        # tentar inferir altura via bounds
        try:
            dims = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
            h = float(dims[2]) if dims[2] > 0 else float(max(dims))
        except Exception:
            h = CATEGORY_TARGETS["adult"]
        if h <= 0.0:
            return "adult", CATEGORY_TARGETS["adult"]
        if h < 1.1:
            return "child", CATEGORY_TARGETS["child"]
        if h < 1.6:
            return "adolescent", CATEGORY_TARGETS["adolescent"]
        return "adult", CATEGORY_TARGETS["adult"]
    # fallback shape heuristics
    try:
        dims = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
        maxd = float(max(dims))
    except Exception:
        maxd = CATEGORY_TARGETS["adult"]
    if maxd > CATEGORY_TARGETS["scene"]:
        return "scene", CATEGORY_TARGETS["scene"]
    if dims[2] < 0.9:
        return "animal", CATEGORY_TARGETS["animal"]
    return "adult", CATEGORY_TARGETS["adult"]


def detect_style_by_vertex_count(mesh):
    """Classifica estilo por contagem de vértices."""
    try:
        v = len(mesh.vertices)
    except Exception:
        v = 0
    for style, (lo, hi) in STYLE_BUCKETS.items():
        if lo <= v < hi:
            return style
    return "mid"


def detect_furry_from_meta_and_materials(meta, base_name):
    """Heurística para detectar furry/pelts/escamas via material names ou particle systems."""
    lower = (base_name or "").lower()
    if meta.get("particle_systems", 0) > 0:
        return True
    mats = meta.get("materials", []) or []
    for m in mats:
        ml = m.lower() if isinstance(m, str) else ""
        if any(k in ml for k in FURRY_KEYWORDS):
            return True
    if any(k in lower for k in FURRY_KEYWORDS):
        return True
    return False

# -----------------------------
# Detecção semântica (robusta / fallback)
# -----------------------------
def detect_semantic_objects(mesh, max_points=2048, top_k=3):
    """
    Detecta etiquetas semânticas para o mesh.
    Usa MODEL/TEXT_EMB se disponíveis; caso contrário, aplica heurísticas baseadas em vértices/volume.
    Retorna dict com keys: tags (list), scores (list), description (str).
    """
    try:
        if not SEMANTIC_MODEL_AVAILABLE or MODEL is None or TEXT_EMB is None or not HAS_SKLEARN:
            # fallback heurístico simples
            tags = []
            try:
                vcount = len(mesh.vertices) if hasattr(mesh, "vertices") else 0
                vol = getattr(mesh, "volume", 0) if hasattr(mesh, "volume") else 0
                if vcount < 500:
                    tags = ["prop"]
                elif vcount < 5000:
                    tags = ["character"]
                else:
                    tags = ["scene"]
                desc = f"Heuristic fallback: vcount={vcount}, vol={vol}"
            except Exception:
                tags = []
                desc = "Heuristic fallback (unknown)"
            return {"tags": tags, "scores": [], "description": desc}

        # amostrar pontos do mesh
        pts = None
        try:
            if hasattr(mesh, "sample"):
                pts = mesh.sample(max_points)
            else:
                # fallback: pegar vertices
                pts = np.asarray(mesh.vertices)[:max_points]
        except Exception:
            try:
                pts = np.asarray(mesh.vertices)[:max_points]
            except Exception:
                pts = None

        if pts is None or len(pts) == 0:
            return {"tags": [], "scores": [], "description": "No points for semantic detection"}

        points = np.asarray(pts, dtype=np.float32)
        if points.ndim == 1:
            # corrupt shape
            points = points.reshape(-1, 3)[:max_points]
        points_t = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        if HAS_TORCH and DEVICE == "cuda":
            points_t = points_t.to(DEVICE)

        # usar encoder de pointcloud se houver (PointCLIP custom)
        with torch.no_grad():
            if hasattr(MODEL, "encode_pointcloud"):
                point_emb = MODEL.encode_pointcloud(points_t)
            else:
                # sem encoder, aplicar PCA-based descriptor e fallback textual similarity = None
                centroid = points.mean(axis=0).tolist()
                dims = (np.ptp(points, axis=0)).tolist()
                return {"tags": [], "scores": [], "description": f"PCA/centroid fallback: centroid={centroid}, dims={dims}"}

            point_emb = point_emb / (point_emb.norm(dim=-1, keepdim=True) + 1e-9)

        point_np = point_emb.detach().cpu().numpy()
        text_np = TEXT_EMB.detach().cpu().numpy()
        sims = cosine_similarity(point_np, text_np)[0]
        top_idx = np.argsort(sims)[-top_k:][::-1]
        tags = [LABELS[i] for i in top_idx]
        scores = [float(sims[i]) for i in top_idx]
        description = f"Object likely to be {', '.join(tags[:-1])} or {tags[-1]}." if tags else "No semantic tags."
        return {"tags": tags, "scores": scores, "description": description}
    except Exception as e:
        logging.debug(f"[Semantic] Erro na detecção: {e}")
        return {"tags": [], "scores": [], "description": "Detection failed."}

# -----------------------------
# Voxelização avançada + geração de máscaras
# -----------------------------
def voxelize_mesh(mesh, target_res=DEFAULT_VOXEL_RES, prefer_gpu=False, generate_masks=True):
    """
    Voxeliza a malha para uma matriz (D,D,D) uint8 e opcionalmente gera máscaras:
      - mask_components: máscaras por componente conectado (cada componente -> binário)
      - mask_classes: máscaras por classe heurística (human/animal/prop/mount/accessory)
    Retorna tuple: (vox_array, masks_dict) onde masks_dict pode conter keys: components, classes
    """
    try:
        # aceitar Scene convertendo para Trimesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        max_extent = float(max(mesh.extents)) if hasattr(mesh, "extents") else 0.0
        if max_extent <= 0:
            return None, {}

        # pitch para voxelização
        pitch = max_extent / float(target_res)
        try:
            vg = mesh.voxelized(pitch=pitch)
            mat = vg.matrix.astype(np.uint8)
        except Exception as e:
            logging.debug(f"voxelized falhou (tentativa alternativa): {e}")
            # fallback: rasterização simples via sample points -> grid occupancy
            try:
                pts = np.asarray(mesh.sample(min(20000, target_res**3)))
                mins = pts.min(axis=0)
                maxs = pts.max(axis=0)
                scale = (maxs - mins) / float(target_res)
                scale[scale == 0] = 1e-6
                inds = np.floor((pts - mins) / scale).astype(int)
                inds = np.clip(inds, 0, target_res-1)
                mat = np.zeros((target_res, target_res, target_res), dtype=np.uint8)
                mat[inds[:,0], inds[:,1], inds[:,2]] = 1
            except Exception as e2:
                logging.debug(f"fallback sample->grid falhou: {e2}")
                return None, {}

        # centralizar em D^3
        D = int(target_res)
        out = np.zeros((D, D, D), dtype=np.uint8)
        s0, s1, s2 = mat.shape
        start0 = max((D - s0) // 2, 0)
        start1 = max((D - s1) // 2, 0)
        start2 = max((D - s2) // 2, 0)
        end0 = min(start0 + s0, D)
        end1 = min(start1 + s1, D)
        end2 = min(start2 + s2, D)
        out[start0:end0, start1:end1, start2:end2] = mat[:(end0 - start0), :(end1 - start1), :(end2 - start2)]

        masks = {"components": [], "classes": {}}

        if generate_masks:
            # connected components (se scipy disponível)
            try:
                if HAS_SCIPY and scipy_ndimage is not None:
                    labeled, num = scipy_ndimage.label(out)
                    # extrair máscaras para componentes pequenos (acessórios) e maior (principal)
                    counts = np.bincount(labeled.flatten())
                    # zerar label 0 (background)
                    counts[0] = 0
                    comp_indices = np.argsort(counts)[::-1]  # maior -> menor
                    # criar máscara principal e máscaras de acessório
                    comp_masks = {}
                    for idx in comp_indices:
                        if idx == 0:
                            continue
                        mask = (labeled == idx).astype(np.uint8)
                        comp_masks[int(idx)] = mask
                    masks["components"] = comp_masks
                    # heurística: componente maior -> principal; menores -> accessories
                    if len(comp_indices) > 1:
                        main_idx = comp_indices[0]
                        accessory_idxs = [int(i) for i in comp_indices[1:]]
                    else:
                        main_idx = comp_indices[0] if len(comp_indices) == 1 else None
                        accessory_idxs = []
                else:
                    # scipy não disponível: usar simple heuristic (center mass as main)
                    coords = np.argwhere(out)
                    if coords.size == 0:
                        masks["components"] = {}
                        main_idx = None
                        accessory_idxs = []
                    else:
                        centroid = coords.mean(axis=0)
                        # distance from centroid -> choose voxels near centroid as main
                        dists = np.linalg.norm(coords - centroid, axis=1)
                        thr = np.percentile(dists, 60)
                        main_mask = np.zeros_like(out)
                        for i, c in enumerate(coords):
                            if dists[i] <= thr:
                                main_mask[tuple(c)] = 1
                        masks["components"] = {"main": main_mask}
                        main_idx = "main"
                        accessory_idxs = []
            except Exception as e:
                logging.debug(f"[Mask] Falha ao rotular componentes: {e}")
                masks["components"] = {}

            # classes via heurística: tentamos usar semantic tags (detect_semantic_objects)
            try:
                sem = detect_semantic_objects(mesh, max_points=1024, top_k=3)
                tags = sem.get("tags", []) if isinstance(sem, dict) else []
            except Exception:
                tags = []

            # mapear tags para classes simplificadas
            class_map = {"human": ["human", "person", "man", "woman", "girl", "boy", "child", "adult"],
                         "animal": ["animal", "cat", "dog", "horse", "bird", "fish"],
                         "vehicle": ["car","bus","truck","motorcycle","bicycle","train","airplane","boat"],
                         "prop": ["cup","table","chair","prop","sword","knife","shield"],
                         "scene": ["house","building","door","window","tower","bridge"]}
            # inicializar classes vazias
            for cname in class_map.keys():
                masks["classes"][cname] = np.zeros_like(out, dtype=np.uint8)

            # heurística simples: se 'human' tag presente -> mark central region as human
            assigned_main = False
            if tags:
                t0 = tags[0].lower()
                matched = False
                for cname, keywords in class_map.items():
                    if any(k in t0 for k in keywords):
                        # se temos componentes via scipy, atribuir componente maior ao classe
                        if HAS_SCIPY and hasattr(masks.get("components"), "keys"):
                            if isinstance(masks["components"], dict) and len(masks["components"])>0:
                                # pegar o maior componente se possível
                                if isinstance(list(masks["components"].keys())[0], int):
                                    # procurar maior label mask by sum
                                    best_label = None
                                    best_sum = 0
                                    for lab, m in masks["components"].items():
                                        s = np.sum(m)
                                        if s > best_sum:
                                            best_sum = s
                                            best_label = lab
                                    if best_label is not None:
                                        masks["classes"][cname] = (masks["components"][best_label] > 0).astype(np.uint8)
                                        assigned_main = True
                                        matched = True
                                        break
                        # fallback: mark center cube region
                        if not assigned_main:
                            Dmid = D//2
                            r = max(1, D//6)
                            masks["classes"][cname][Dmid-r:Dmid+r, Dmid-r:Dmid+r, Dmid-r:Dmid+r] = out[Dmid-r:Dmid+r, Dmid-r:Dmid+r, Dmid-r:Dmid+r]
                            matched = True
                            break
                if not matched:
                    # se não casou, marcar como 'prop'
                    masks["classes"]["prop"] = out.copy()
            else:
                # sem tags: tentar inferir por volume/height heuristics
                try:
                    dims = mesh.bounds[1] - mesh.bounds[0]
                    if dims[2] > 1.2:
                        masks["classes"]["human"] = out.copy()
                    else:
                        masks["classes"]["prop"] = out.copy()
                except Exception:
                    masks["classes"]["prop"] = out.copy()

            # accessories: componentes pequenos (se detectado)
            try:
                accessories_mask = np.zeros_like(out, dtype=np.uint8)
                if isinstance(masks.get("components"), dict) and len(masks["components"])>0:
                    # garantir que indexes existam
                    for lab, m in masks["components"].items():
                        # identificar pequenos
                        if isinstance(lab, int):
                            s = np.sum(m)
                            if s > 0 and s < (0.02 * out.size):  # <2% do volume -> accessory
                                accessories_mask = np.logical_or(accessories_mask, m)
                    masks["classes"]["accessory"] = accessories_mask.astype(np.uint8)
                else:
                    masks["classes"]["accessory"] = np.zeros_like(out, dtype=np.uint8)
            except Exception:
                masks["classes"]["accessory"] = np.zeros_like(out, dtype=np.uint8)

        return out, masks
    except MemoryError:
        logging.warning("MemoryError durante voxelização — tentando resoluções menores")
        try:
            for r in (int(target_res * 0.75), int(target_res * 0.5), int(target_res * 0.25)):
                if r < 8:
                    break
                try:
                    return voxelize_mesh(mesh, target_res=r, prefer_gpu=False, generate_masks=generate_masks)
                except Exception:
                    continue
        except Exception:
            pass
        return None, {}
    except Exception as e:
        logging.debug(f"Erro voxelização: {e}")
        return None, {}

# -----------------------------
# Texturas / Pele HSV
# -----------------------------
def collect_texture_files_near(src_path):
    """Procura por arquivos de textura na mesma pasta (recursivo)."""
    folder = os.path.dirname(src_path)
    texs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(TEXTURE_EXTS):
                texs.append(os.path.join(root, f))
    return texs


def detect_skin_tone_from_textures(tex_paths, sample_limit=5):
    """
    Analisa amostras de texturas e tenta inferir tons de pele via HLS.
    Retorna dict: {"has_skin": bool, "avg_hsv": (H,S,L) or None}
    """
    if not HAS_PIL or not tex_paths:
        return {"has_skin": False, "avg_hsv": None}
    import colorsys
    samples = []
    tried = 0
    for t in tex_paths:
        if tried >= sample_limit:
            break
        try:
            img = Image.open(t).convert('RGB')
            w, h = img.size
            # grid samples 3x3
            for sx in (0.2, 0.5, 0.8):
                for sy in (0.2, 0.5, 0.8):
                    px = int(w * sx)
                    py = int(h * sy)
                    try:
                        r, g, b = img.getpixel((px, py))
                        samples.append((r, g, b))
                    except Exception:
                        continue
            tried += 1
        except Exception:
            continue
    if not samples:
        return {"has_skin": False, "avg_hsv": None}
    hs = []
    for r, g, b in samples:
        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
        # convert h 0..1 to 0..360
        hs.append((h*360.0, s, l))
    skin_count = 0
    for h, s, l in hs:
        if (h >= 340 or h <= 60) and (s >= 0.12 and s <= 0.9) and (l >= 0.12 and l <= 0.95):
            skin_count += 1
    has_skin = skin_count >= max(1, len(hs)//4)
    avg = (sum([h for h,_,_ in hs])/len(hs), sum([s for _,s,_ in hs])/len(hs), sum([l for _,_,l in hs])/len(hs))
    return {"has_skin": bool(has_skin), "avg_hsv": avg}

# -----------------------------
# Análise de pose (a partir de meta/FBX)
# -----------------------------
def analyze_pose_from_meta_and_fbx(meta, fbx_path):
    """
    Retorna (poses_list, has_animation_bool)
    Usa meta gerado pelo Blender quando disponível; senão heurística por bounding box.
    """
    poses = []
    has_animation = bool(meta.get("has_animation", False))
    if meta.get("has_armature", False):
        poses.append("armature_present")
        if has_animation:
            poses.append("animated")
        else:
            poses.append("static_pose_detected")
    else:
        try:
            scene = trimesh.load(fbx_path, force='scene')
            mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
            bbox = mesh.bounds
            dims = bbox[1] - bbox[0]
            h = float(dims[2])
            if h < 0.6:
                poses.append('lying_or_crouched')
            elif h < 1.1:
                poses.append('sitting_or_crouched')
            else:
                poses.append('standing')
        except Exception:
            poses.append('unknown')
    return poses, has_animation

# -----------------------------
# Separar acessórios via componentes conectados (melhorado)
# -----------------------------
def split_mesh_and_save_accessories(scene_or_mesh, out_dir, base_name, timestamp, accessory_res=48):
    """
    Separa componentes conectados (heurística) e salva acessórios menores como .npy.
    Retorna (list_saved_paths, main_component_mesh)
    """
    saved = []
    try:
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh
        parts = []
        try:
            parts = mesh.split(only_watertight=False)
        except Exception:
            # fallback: considerar mesh inteiro como único part
            parts = [mesh]
        parts_sorted = sorted(parts, key=lambda m: (m.volume if hasattr(m, 'volume') else len(getattr(m, 'vertices', []))), reverse=True)
        if not parts_sorted:
            return [], mesh
        main = parts_sorted[0]
        for i, p in enumerate(parts_sorted[1:], start=1):
            fname = os.path.join(out_dir, f"{base_name}_{timestamp}_part{i}.npy")
            vox, masks = voxelize_mesh(p, target_res=accessory_res, prefer_gpu=False, generate_masks=False)
            if vox is not None:
                try:
                    np.save(fname, vox)
                    saved.append(fname)
                except Exception:
                    pass
    except Exception as e:
        logging.debug(f"Erro ao separar acessórios: {e}")
        return [], scene_or_mesh
    return saved, main

# Fim da parte 2/3

# -----------------------------
# Parte 3/3 - Integração final: processamento por arquivo, runner, fallback GPU<->CPU e CLI
# -----------------------------

# -----------------------------
# Função principal de processamento por arquivo (corrigida e robusta)
# -----------------------------
def process_single_file(blender_bin, src_path, out_root, temp_root, device_mode='auto'):
    """
    Processa um arquivo 3D completo:
      - export via Blender (FBX + meta)
      - load via trimesh
      - detecção categoria/estilo/furry/textures/anim
      - normalização (centering + scaling)
      - voxelização (tentando GPU-mode se aplicável; fallback para CPU ao OOM)
      - geração de máscaras (component & class masks)
      - salvamento .npy do voxel e máscaras (se existirem)
      - cópia FBX para pastas com/sem textura, animados/estaticos, montarias
      - gravação de linha no relatorio CSV (append)
    Retorna dicionário de resultado.
    """
    start_t = time.time()
    base_name = os.path.splitext(os.path.basename(src_path))[0]
    safe_makedirs(os.path.join(out_root, "originais_preservados"))
    try:
        shutil.copy2(src_path, os.path.join(out_root, "originais_preservados", os.path.basename(src_path)))
    except Exception:
        pass

    file_temp_dir = os.path.join(temp_root, f"tmp_{base_name}_{int(time.time()*1000)}")
    safe_makedirs(file_temp_dir)

    # Export via Blender
    fbx_path, meta = run_blender_export(blender_bin, src_path, file_temp_dir)
    if fbx_path is None:
        return {"source": src_path, "status": "export_fail", "message": str(meta), "time": round(time.time() - start_t, 3)}

    # Load scene
    try:
        scene = trimesh.load(fbx_path, force='scene')
        if scene is None:
            raise RuntimeError("trimesh.load retornou None")
    except Exception as e:
        logging.debug(traceback.format_exc())
        return {"source": src_path, "status": "load_fail", "message": str(e), "time": round(time.time() - start_t, 3)}

    try:
        main_mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    except Exception:
        main_mesh = scene if isinstance(scene, trimesh.Trimesh) else None

    # Metadata detection
    category, target_h = detect_category_by_name_and_mesh(base_name, main_mesh)
    style = detect_style_by_vertex_count(main_mesh)
    detected_textures = bool(meta.get('has_textures', False)) or len(collect_texture_files_near(src_path)) > 0
    has_anim = bool(meta.get("has_armature", False) or meta.get("has_animation", False))
    is_furry = detect_furry_from_meta_and_materials(meta, base_name)
    is_vehicle = any(k in base_name.lower() for k in VEHICLE_KEYWORDS)

    # choose voxel resolution based on category/style
    if category == "scene":
        voxel_res = 128
    elif category == "mount":
        voxel_res = 128
    elif category == "animal":
        voxel_res = 80
    elif category == "child":
        voxel_res = 80
    else:
        if style == "low":
            voxel_res = 64
        elif style == "mid":
            voxel_res = 80
        else:
            voxel_res = 96

    # normalize: center and scale to target_h (preserve proportions)
    try:
        bbox = main_mesh.bounds
        center = bbox.mean(axis=0)
        main_mesh.apply_translation(-center)
        dims = bbox[1] - bbox[0]
        curr_h = float(dims[2]) if dims[2] > 0 else float(max(dims))
        if curr_h <= 0:
            scale_factor = 1.0
        else:
            scale_factor = float(target_h) / curr_h
        main_mesh.apply_scale(scale_factor)
    except Exception:
        scale_factor = 1.0

    # Voxelize: attempt GPU mode if available and requested; on OOM fallback to CPU
    vox = None
    masks = {}
    tried_devices = []
    preferred = device_mode
    # decide order of devices to try
    if preferred == 'gpu' or (preferred == 'auto' and HAS_TORCH and torch.cuda.is_available()):
        device_order = ['gpu', 'cpu']
    else:
        device_order = ['cpu']

    for dev in device_order:
        tried_devices.append(dev)
        try:
            if dev == 'gpu':
                # keep same function but flags can be used if we later add GPU-accelerated voxelization
                vox, masks = voxelize_mesh(main_mesh, target_res=voxel_res, prefer_gpu=True, generate_masks=True)
            else:
                vox, masks = voxelize_mesh(main_mesh, target_res=voxel_res, prefer_gpu=False, generate_masks=True)
            if vox is not None:
                used_device = dev
                break
        except Exception as e:
            # catch OutOfMemory for torch specifically
            if HAS_TORCH and isinstance(e, RuntimeError) and 'out of memory' in str(e).lower():
                logging.warning(f"OOM on device {dev} for {base_name}, trying next device.")
                # if GPU OOM, try CPU next
                continue
            logging.debug(f"Erro voxelização device {dev}: {e}")
    else:
        return {"source": src_path, "status": "voxel_fail", "message": "voxelization returned None on all devices", "time": round(time.time() - start_t, 3)}

    if vox is None:
        return {"source": src_path, "status": "voxel_fail", "message": "voxelization returned None", "time": round(time.time() - start_t, 3)}

    # Prepare directories
    models_voxelized_dir = os.path.join(out_root, "modelos_voxelizados")
    com_textura_dir = os.path.join(out_root, "modelos_com_textura")
    sem_textura_dir = os.path.join(out_root, "modelos_sem_textura")
    animados_dir = os.path.join(out_root, "modelos_animados")
    estaticos_dir = os.path.join(out_root, "modelos_estaticos")
    acessorios_dir = os.path.join(out_root, "acessorios")
    montarias_dir = os.path.join(out_root, "montarias")
    estilos_dir = os.path.join(out_root, "Estilos")
    relatorios_dir = os.path.join(out_root, "relatorios")
    temporarios_dir = os.path.join(out_root, "temporarios")
    poses_dir = os.path.join(out_root, "Poses")
    anim_dir = os.path.join(out_root, "Animacoes")
    texturizacao_dir = os.path.join(out_root, "texturizacao")
    moldes_uv_dir = os.path.join(texturizacao_dir, "moldes_uv")

    for d in (models_voxelized_dir, com_textura_dir, sem_textura_dir, animados_dir, estaticos_dir,
              acessorios_dir, montarias_dir, estilos_dir, relatorios_dir, temporarios_dir, poses_dir, anim_dir,
              texturizacao_dir, moldes_uv_dir):
        safe_makedirs(d)

    # ensure style subfolder in modelos_voxelizados
    style_folder = os.path.join(models_voxelized_dir, style)
    safe_makedirs(style_folder)

    timestamp = int(time.time())
    voxel_name = f"{base_name}_{timestamp}.npy"
    voxel_out = os.path.join(style_folder, voxel_name)

    # Save voxel grid
    try:
        np.save(voxel_out, vox)
    except Exception as e:
        logging.debug(traceback.format_exc())
        return {"source": src_path, "status": "save_fail", "message": str(e), "time": round(time.time()-start_t,3)}

    # Save masks (components + class masks) if available
    mask_files = []
    try:
        if masks:
            masks_dir = os.path.join(style_folder, f"{base_name}_{timestamp}_masks")
            safe_makedirs(masks_dir)
            # components (dict of int->mask)
            comps = masks.get("components", {})
            if isinstance(comps, dict):
                for lab, m in comps.items():
                    fname = os.path.join(masks_dir, f"{base_name}_{timestamp}_comp_{lab}.npy")
                    try:
                        np.save(fname, m)
                        mask_files.append(os.path.relpath(fname, out_root))
                    except Exception:
                        pass
            # class masks
            classes = masks.get("classes", {})
            if isinstance(classes, dict):
                for cname, cm in classes.items():
                    fname = os.path.join(masks_dir, f"{base_name}_{timestamp}_class_{cname}.npy")
                    try:
                        np.save(fname, cm)
                        mask_files.append(os.path.relpath(fname, out_root))
                    except Exception:
                        pass
    except Exception as e:
        logging.debug(f"Falha ao salvar masks: {e}")

    # copy FBX to com/sem textura
    try:
        if detected_textures:
            shutil.copy2(fbx_path, os.path.join(com_textura_dir, os.path.basename(fbx_path)))
        else:
            shutil.copy2(fbx_path, os.path.join(sem_textura_dir, os.path.basename(fbx_path)))
    except Exception:
        pass

    # animado/estatico
    try:
        if has_anim:
            shutil.copy2(fbx_path, os.path.join(animados_dir, os.path.basename(fbx_path)))
        else:
            shutil.copy2(fbx_path, os.path.join(estaticos_dir, os.path.basename(fbx_path)))
    except Exception:
        pass

    # accessories: save list from split_mesh_and_save_accessories (if any)
    try:
        acc_saved, main_component = split_mesh_and_save_accessories(main_mesh, acessorios_dir, base_name, timestamp)
    except Exception:
        acc_saved = []

    # mounts
    try:
        if any(k in base_name.lower() for k in MOUNT_KEYWORDS):
            shutil.copy2(fbx_path, os.path.join(montarias_dir, os.path.basename(fbx_path)))
    except Exception:
        pass

    # poses analysis
    poses_detected, has_anim = analyze_pose_from_meta_and_fbx(meta, fbx_path)

    # skin tone analysis
    texs = collect_texture_files_near(src_path)
    skin_info = detect_skin_tone_from_textures(texs)

    # Prepare CSV/relatorio row
    rel_csv = os.path.join(relatorios_dir, "relatorio.csv")
    header = ["source","base","status","category","style","has_texture","has_animation","is_furry","is_vehicle","has_skin_texture","voxel_file","masks","time_seconds","poses","accessories","semantic_tags","semantic_desc"]
    row = {
        "source": src_path,
        "base": base_name,
        "status": "ok",
        "category": category,
        "style": style,
        "has_texture": bool(detected_textures),
        "has_animation": bool(has_anim),
        "is_furry": bool(is_furry),
        "is_vehicle": bool(is_vehicle),
        "has_skin_texture": bool(skin_info.get('has_skin', False)),
        "voxel_file": os.path.relpath(voxel_out, out_root),
        "masks": ";".join(mask_files),
        "time_seconds": round(time.time() - start_t, 3),
        "poses": ";".join(poses_detected),
        "accessories": ";".join([os.path.relpath(a, out_root) for a in acc_saved]),
        "semantic_tags": ";".join(detect_semantic_objects(main_mesh).get("tags", [])),
        "semantic_desc": detect_semantic_objects(main_mesh).get("description", "")
    }

    write_header = not os.path.exists(rel_csv)
    try:
        with open(rel_csv, "a", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception:
        logging.debug("Falha ao escrever CSV.", exc_info=True)

    # cleanup temp
    try:
        if os.path.exists(file_temp_dir):
            shutil.rmtree(file_temp_dir, ignore_errors=True)
    except Exception:
        pass

    logging.info(f"[OK] {base_name} -> {row['voxel_file']} ({row['time_seconds']}s) tried_devices={tried_devices}")
    return {"source": src_path, "status": "ok", "meta": row}

# -----------------------------
# Runner com limite de processos e semáforo
# -----------------------------
def decide_device_mode():
    """
    Decide entre 'auto','cpu' ou 'gpu' preferência.
    'auto' tenta GPU se disponível.
    """
    if HAS_TORCH:
        try:
            if torch.cuda.is_available():
                logging.info(f"CUDA disponível: {torch.cuda.device_count()} devices. Usando modo 'auto' com preferência GPU.")
                return 'auto'
        except Exception:
            pass
    return 'cpu'

def run_pipeline(root_folder, max_workers=MAX_WORKERS):
    blender_bin = find_blender()
    if not blender_bin:
        blender_bin = ask_blender_path_and_save()
    if not blender_bin or not os.path.exists(blender_bin):
        logging.error("Blender não encontrado. Instale ou informe o caminho e tente novamente.")
        return
    out_root = os.path.join(root_folder, OUT_ROOT_DIRNAME)
    safe_makedirs(out_root)
    safe_makedirs(os.path.join(out_root, "temporarios"))

    # collect files
    files = []
    for dirpath, _, filenames in os.walk(root_folder):
        # skip output folder to avoid reprocessing
        if os.path.abspath(dirpath).startswith(os.path.abspath(out_root)):
            continue
        for f in filenames:
            if f.lower().endswith(SUPPORTED_INPUT_EXTS):
                files.append(os.path.join(dirpath, f))
    logging.info(f"Arquivos 3D encontrados: {len(files)}")
    if not files:
        logging.warning("Nenhum arquivo 3D suportado encontrado na pasta selecionada.")
        return

    temp_root = os.path.join(out_root, "temporarios")
    safe_makedirs(temp_root)
    device_mode = decide_device_mode()

    sem = Semaphore(max_workers)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {}
        for f in files:
            sem.acquire()
            def make_task(src):
                try:
                    res = process_single_file(blender_bin, src, out_root, temp_root, device_mode=device_mode)
                    return res
                finally:
                    sem.release()
            fut = exe.submit(make_task, f)
            futures[fut] = f

        # use tqdm to show progress as tasks complete
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processando modelos", unit="file"):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                logging.error(f"Erro ao processar: {e}")

    # summary
    summary_path = os.path.join(out_root, "relatorios", f"summary_{int(time.time())}.json")
    try:
        save_json(summary_path, {"processed": len(results), "timestamp": time.ctime(), "results": results})
    except Exception:
        pass
    logging.info("Processamento finalizado. Saída em: %s", out_root)

# -----------------------------
# Execução principal (CLI minimal)
# -----------------------------
def main():
    Tk().withdraw()
    folder = filedialog.askdirectory(title="Selecione a pasta raiz contendo os modelos (varredura recursiva)")
    if not folder:
        logging.info("Nenhuma pasta selecionada. Saindo.")
        return
    max_w = MAX_WORKERS
    logging.info(f"Executando com até {max_w} processos simultâneos (ajustável no topo do script).")
    run_pipeline(folder, max_workers=max_w)

if __name__ == "__main__":
    main()
