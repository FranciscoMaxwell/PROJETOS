#!/usr/bin/env python3
# organiza_voxel_treinamento_v2.py
"""
Versão 2 - Pipeline completo com heurísticas avançadas e fallback GPU/CPU.
- Implementa: detecção de furries/escamas/dragões/veículos, agrupamento de personagens,
  análise de pose (a partir de armature quando disponível, heurística sem armature),
  análise HSV para tons de pele, export de acessórios separados, separação por poses/animações,
  fallback automático GPU->CPU, detecção automática do Blender, headless export, limite de processos,
  logs/tqdm/relatórios.

OBS: Algumas classificações (pose sem armature, distinção estilística avançada,
classificação de cenas complexas) usam heurísticas. Para detecção perfeita de poses e
contextos complexos recomenda-se treinar/usar modelos ML específicos (ex: modelos de pose 3D).
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

# libs externas
try:
    import trimesh
except Exception:
    print("Instale trimesh: pip install trimesh")
    raise
try:
    import numpy as np
except Exception:
    print("Instale numpy: pip install numpy")
    raise
try:
    from tqdm import tqdm
except Exception:
    print("Instale tqdm: pip install tqdm")
    raise
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# opcional: detecção de GPU via torch
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# -----------------------------
# Configurações principais
# -----------------------------
DEFAULT_VOXEL_RES = 80
OUT_ROOT_DIRNAME = "Projeto_3D"
CONFIG_FILE = os.path.expanduser("~/.organiza_voxel_config_v2.json")
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

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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
    logging.info("Blender não encontrado automaticamente. Peça ao usuário o caminho.")
    Tk().withdraw()
    path = filedialog.askopenfilename(title="Selecione o executável do Blender")
    if path and os.path.exists(path):
        save_json(CONFIG_FILE, {"blender_path": path})
        return path
    return None

# -----------------------------
# blender tmp script (melhorado: exporta objetos separados, detecta armature, partículas)
# -----------------------------
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

def safe_name(o):
    try:
        return o.name
    except Exception:
        return ""

try:
    ext = os.path.splitext(src)[1].lower()
    # abrir/ importar
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

    # apply transforms
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
    def vec(a):
        return a.x, a.y, a.z

    def vec_len(v):
        return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

    def vec_sub(a,b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

    def dot(a,b):
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

    def angle_between(a,b):
        la = vec_len(a)
        lb = vec_len(b)
        if la==0 or lb==0:
            return 0.0
        v = dot(a,b)/(la*lb)
        v = max(-1.0, min(1.0, v))
        return math.degrees(math.acos(v))

    poses = set()
    armatures = [o for o in bpy.data.objects if o.type=='ARMATURE']
    if armatures:
        for arm in armatures:
            try:
                meta["has_armature"] = True
                # ensure pose mode accessible
                bpy.context.view_layer.objects.active = arm
                bpy.ops.object.mode_set(mode='POSE')
                pbones = arm.pose.bones
                # pick common bone names
                names = {b.name.lower(): b for b in pbones}
                def find_bone(keys):
                    for k in keys:
                        for n,b in names.items():
                            if k in n:
                                return b
                    return None
                head_b = find_bone(['head','neck'])
                hip_b = find_bone(['hip','hips','pelvis','root','spine'])
                l_hand = find_bone(['hand.l','hand_l','.l hand','left_hand','hand_l','left'])
                r_hand = find_bone(['hand.r','hand_r','.r hand','right_hand','hand_r','right'])
                # fallback to any bone names
                bones = list(pbones)
                if not head_b and bones:
                    head_b = bones[0]
                if not hip_b and bones:
                    hip_b = bones[int(len(bones)/2)]
                # compute vectors in armature local
                def bone_pos(b):
                    try:
                        return vec(arm.matrix_world @ b.head)
                    except Exception:
                        try:
                            return vec(b.head)
                        except Exception:
                            return (0,0,0)
                hip_pos = bone_pos(hip_b) if hip_b else None
                head_pos = bone_pos(head_b) if head_b else None
                lpos = bone_pos(l_hand) if l_hand else None
                rpos = bone_pos(r_hand) if r_hand else None
                if hip_pos and head_pos:
                    v = vec_sub(head_pos, hip_pos)
                    # vertical component relative to total length
                    vertical_ratio = abs(v[2]) / (vec_len(v) + 1e-9)
                    if vertical_ratio > 0.7 and vec_len(v) > 0.4:
                        poses.add('standing')
                    elif vertical_ratio > 0.4 and vec_len(v) > 0.2:
                        poses.add('sitting_or_crouched')
                    else:
                        poses.add('lying')
                # hands relative to head
                if head_pos and lpos and vec_len(vec_sub(lpos, head_pos)) < 0.25:
                    poses.add('left_hand_on_head')
                if head_pos and rpos and vec_len(vec_sub(rpos, head_pos)) < 0.25:
                    poses.add('right_hand_on_head')
                # arms raised: hands higher than head.z by threshold
                if head_pos and lpos and lpos[2] > head_pos[2] + 0.05:
                    poses.add('left_arm_raised')
                if head_pos and rpos and rpos[2] > head_pos[2] + 0.05:
                    poses.add('right_arm_raised')
                # if animation present
                if getattr(arm, 'animation_data', None) and getattr(arm.animation_data, 'action', None):
                    poses.add('animated')
            except Exception:
                pass
    else:
        # no armature: fallback using mesh bounding boxes
        try:
            meshes = [o for o in bpy.data.objects if o.type=='MESH']
            if meshes:
                # compute combined bbox
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



def run_blender_export(blender_bin, src_path, out_dir, timeout=600):
    safe_makedirs(out_dir)
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_fbx = os.path.join(out_dir, f"{base}_export.fbx")
    meta_json = os.path.join(out_dir, f"{base}_meta.json")
    tmp_script = os.path.join(out_dir, f"tmp_blender_export_{int(time.time()*1000)}.py")
    try:
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
            return None, {"error": "export_failed", "stderr": proc.stderr.decode("utf-8", errors="ignore")}
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

# -----------------------------
# Heurísticas avançadas
# -----------------------------

def detect_category_by_name_and_mesh(name, mesh):
    lower = name.lower()
    # name-based overrides
    if any(k in lower for k in MOUNT_KEYWORDS):
        return "mount", CATEGORY_TARGETS["mount"]
    if any(k in lower for k in VEHICLE_KEYWORDS):
        return "prop", CATEGORY_TARGETS["prop"]
    if any(k in lower for k in HUMAN_KEYWORDS):
        dims = mesh.bounds[1] - mesh.bounds[0]
        h = float(dims[2]) if dims[2] > 0 else float(max(dims))
        if h < 1.1:
            return "child", CATEGORY_TARGETS["child"]
        if h < 1.6:
            return "adolescent", CATEGORY_TARGETS["adolescent"]
        return "adult", CATEGORY_TARGETS["adult"]
    # fallback shape heuristics
    dims = mesh.bounds[1] - mesh.bounds[0]
    maxd = float(max(dims))
    if maxd > CATEGORY_TARGETS["scene"]:
        return "scene", CATEGORY_TARGETS["scene"]
    if dims[2] < 0.9:
        return "animal", CATEGORY_TARGETS["animal"]
    return "adult", CATEGORY_TARGETS["adult"]


def detect_style_by_vertex_count(mesh):
    try:
        v = len(mesh.vertices)
    except Exception:
        v = 0
    for style, (lo, hi) in STYLE_BUCKETS.items():
        if lo <= v < hi:
            return style
    return "mid"


def detect_furry_from_meta_and_materials(meta, base_name):
    # heuristics: particle systems, material names or keywords
    lower = base_name.lower()
    if meta.get("particle_systems", 0) > 0:
        return True
    mats = meta.get("materials", [])
    for m in mats:
        ml = m.lower()
        if any(k in ml for k in FURRY_KEYWORDS):
            return True
    # name hints
    if any(k in lower for k in FURRY_KEYWORDS):
        return True
    return False

# -----------------------------
# Voxelização (com tentativa de usar GPU se disponível/necessário)
# -----------------------------

def voxelize_mesh(mesh, target_res=DEFAULT_VOXEL_RES, prefer_gpu=False):
    """Retorna array numpy (D,D,D) com 0/1 uint8. Usa trimesh.voxelized (CPU).
    Se ocorrer MemoryError ou outras exceções relacionadas a VRAM, tenta fallback.
    """
    try:
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        max_extent = float(max(mesh.extents))
        if max_extent <= 0:
            return None
        pitch = max_extent / float(target_res)
        vg = mesh.voxelized(pitch=pitch)
        mat = vg.matrix.astype(np.uint8)
        D = target_res
        out = np.zeros((D, D, D), dtype=np.uint8)
        s0, s1, s2 = mat.shape
        start0 = max((D - s0) // 2, 0)
        start1 = max((D - s1) // 2, 0)
        start2 = max((D - s2) // 2, 0)
        end0 = min(start0 + s0, D)
        end1 = min(start1 + s1, D)
        end2 = min(start2 + s2, D)
        out[start0:end0, start1:end1, start2:end2] = mat[:(end0 - start0), :(end1 - start1), :(end2 - start2)]
        return out
    except MemoryError:
        logging.warning("MemoryError durante voxelização — tentativa de fallback para resolução menor")
        try:
            # tentar resoluções menores até conseguir
            for r in (int(target_res * 0.75), int(target_res * 0.5), int(target_res * 0.25)):
                if r < 8:
                    break
                try:
                    return voxelize_mesh(mesh, target_res=r, prefer_gpu=False)
                except Exception:
                    continue
        except Exception:
            pass
        return None
    except Exception as e:
        logging.debug(f"Erro voxelização: {e}")
        return None

# -----------------------------
# Textura / Pele HSV
# -----------------------------

def collect_texture_files_near(src_path):
    folder = os.path.dirname(src_path)
    texs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(TEXTURE_EXTS):
                texs.append(os.path.join(root, f))
    return texs


def detect_skin_tone_from_textures(tex_paths, sample_limit=5):
    """Analisa texturas (amostragem) e tenta achar pixels com HSV dentro de faixa de pele.
    Retorna dicionário com média HSV encontrada e booleano se pele detectada.
    Heurística: verifica tons de H entre 0-50/340-360 (peles podem variar) e S/V dentro de faixas.
    """
    if not HAS_PIL or not tex_paths:
        return {"has_skin": False, "avg_hsv": None}
    samples = []
    tried = 0
    for t in tex_paths:
        if tried >= sample_limit:
            break
        try:
            img = Image.open(t).convert('RGB')
            w, h = img.size
            # sample grid
            for sx in (0.25, 0.5, 0.75):
                for sy in (0.25, 0.5, 0.75):
                    px = int(w * sx)
                    py = int(h * sy)
                    r, g, b = img.getpixel((px, py))
                    samples.append((r, g, b))
            tried += 1
        except Exception:
            continue
    if not samples:
        return {"has_skin": False, "avg_hsv": None}
    import colorsys
    hs = []
    for r, g, b in samples:
        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
        # convert h 0..1 to 0..360
        hs.append((h*360.0, s, l))
    # check if many samples fall into a broad "skin" region
    skin_count = 0
    for h, s, l in hs:
        # permissive ranges: hue 0-60 (reddish to yellow) and saturation moderate
        if (h >= 340 or h <= 60) and (s >= 0.15 and s <= 0.9) and (l >= 0.15 and l <= 0.9):
            skin_count += 1
    has_skin = skin_count >= max(1, len(hs)//4)
    avg = (sum([h for h,_,_ in hs])/len(hs), sum([s for _,s,_ in hs])/len(hs), sum([l for _,_,l in hs])/len(hs))
    return {"has_skin": bool(has_skin), "avg_hsv": avg}

# -----------------------------
# Análise de pose (a partir de armature ou heurística)
# -----------------------------

def analyze_pose_from_meta_and_fbx(meta, fbx_path):
    """Tenta extrair poses da FBX: se houver armature/animation flag no meta retorna análise básica.
    Quando armature existe, classificamos poses por keyframes/ângulos dos ossos (heurística).
    Se não houver armature, usamos PCA do mesh para tentar inferir orientação (deitado/sentado/em pé).
    Retorna lista de poses detectadas (strings) e se tem animação.
    """
    poses = []
    has_animation = bool(meta.get("has_animation", False))
    # heurística simples: se tem armature e animation -> marca como 'animado' e tenta extrair 'idle','walk','run'
    if meta.get("has_armature", False):
        poses.append("armature_present")
        if has_animation:
            poses.append("animated")
            # placeholder heuristics: marcar 'walking_like' se num verts > X and duration>0
            # Detalhes mais finos exigiriam parsing da FBX via blender ou biblioteca específica
        else:
            poses.append("static_pose_detected")
    else:
        # sem armature -> fallback: use trimesh to estimate orientation
        try:
            scene = trimesh.load(fbx_path, force='scene')
            if isinstance(scene, trimesh.Scene):
                mesh = scene.dump(concatenate=True)
            else:
                mesh = scene
            bbox = mesh.bounds
            dims = bbox[1] - bbox[0]
            h = float(dims[2])
            # heurística: muito baixo -> deitado ou sentado
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
# Separar acessórios via connected components
# -----------------------------

def split_mesh_and_save_accessories(scene_or_mesh, out_dir, base_name, timestamp):
    """Tenta separar componentes conectados do mesh e salvar cada um como npy separado (nuvem voxelizada).
    Retorna lista de arquivos salvos (acessórios) e o componente principal.
    """
    saved = []
    try:
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh
        parts = mesh.split(only_watertight=False)
        # heurística: o maior volume é o principal
        parts_sorted = sorted(parts, key=lambda m: m.volume if hasattr(m, 'volume') else len(m.vertices), reverse=True)
        if not parts_sorted:
            return [], mesh
        main = parts_sorted[0]
        for i, p in enumerate(parts_sorted[1:], start=1):
            fname = os.path.join(out_dir, f"{base_name}_{timestamp}_part{i}.npy")
            # voxelizar cada parte com res menor (para economia) e salvar
            vox = voxelize_mesh(p, target_res=48)
            if vox is not None:
                np.save(fname, vox)
                saved.append(fname)
    except Exception as e:
        logging.debug(f"Erro ao separar acessórios: {e}")
        return [], scene_or_mesh
    return saved, main

# -----------------------------
# processamento de arquivo
# -----------------------------

def process_single_file(blender_bin, src_path, out_root, temp_root, device_mode='auto'):
    start_t = time.time()
    base_name = os.path.splitext(os.path.basename(src_path))[0]
    orig_preserved_dir = os.path.join(out_root, "originais_preservados")
    safe_makedirs(orig_preserved_dir)
    try:
        shutil.copy2(src_path, os.path.join(orig_preserved_dir, os.path.basename(src_path)))
    except Exception:
        pass
    file_temp_dir = os.path.join(temp_root, f"tmp_{base_name}_{int(time.time()*1000)}")
    safe_makedirs(file_temp_dir)
    fbx_path, meta = run_blender_export(blender_bin, src_path, file_temp_dir)
    if fbx_path is None:
        return {"source": src_path, "status": "export_fail", "message": str(meta), "time": round(time.time() - start_t, 3)}
    try:
        scene = trimesh.load(fbx_path, force='scene')
        if scene is None:
            raise RuntimeError("trimesh.load retornou None")
    except Exception as e:
        return {"source": src_path, "status": "load_fail", "message": str(e), "time": round(time.time() - start_t, 3)}
    try:
        main_mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    except Exception:
        main_mesh = scene if isinstance(scene, trimesh.Trimesh) else None
    # metadata detection
    category, target_h = detect_category_by_name_and_mesh(base_name, main_mesh)
    style = detect_style_by_vertex_count(main_mesh)
    detected_textures = bool(meta.get('has_textures', False)) or len(collect_texture_files_near(src_path)) > 0
    has_anim = bool(meta.get("has_armature", False) or meta.get("has_animation", False))
    is_furry = detect_furry_from_meta_and_materials(meta, base_name)
    is_vehicle = any(k in base_name.lower() for k in VEHICLE_KEYWORDS)
    # category-specific voxel resolution
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
    # normalize scale/center
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
    # try voxelize with fallback
    vox = voxelize_mesh(main_mesh, target_res=voxel_res, prefer_gpu=(device_mode=='gpu' or device_mode=='auto'))
    if vox is None:
        return {"source": src_path, "status": "voxel_fail", "message": "voxelization returned None", "time": round(time.time() - start_t, 3)}
    # output dirs
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
    for d in (models_voxelized_dir, com_textura_dir, sem_textura_dir, animados_dir, estaticos_dir,
              acessorios_dir, montarias_dir, estilos_dir, relatorios_dir, temporarios_dir, poses_dir, anim_dir):
        safe_makedirs(d)
    style_folder = os.path.join(estilos_dir, style)
    safe_makedirs(style_folder)
    timestamp = int(time.time())
    voxel_name = f"{base_name}_{timestamp}.npy"
    voxel_out = os.path.join(models_voxelized_dir, style, voxel_name)
    safe_makedirs(os.path.dirname(voxel_out))
    try:
        np.save(voxel_out, vox)
    except Exception as e:
        return {"source": src_path, "status": "save_fail", "message": str(e), "time": round(time.time()-start_t,3)}
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
    # accessories: attempt split
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
    # skin tone
    texs = collect_texture_files_near(src_path)
    skin_info = detect_skin_tone_from_textures(texs)
    # write csv
    rel_csv = os.path.join(relatorios_dir, "relatorio.csv")
    header = ["source","base","status","category","style","has_texture","has_animation","is_furry","is_vehicle","has_skin_texture","voxel_file","time_seconds","poses","accessories"]
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
        "time_seconds": round(time.time() - start_t, 3),
        "poses": ";".join(poses_detected),
        "accessories": ";".join([os.path.relpath(a, out_root) for a in acc_saved])
    }
    write_header = not os.path.exists(rel_csv)
    try:
        import csv
        with open(rel_csv, "a", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception:
        pass
    # cleanup temp
    try:
        if os.path.exists(file_temp_dir):
            shutil.rmtree(file_temp_dir, ignore_errors=True)
    except Exception:
        pass
    logging.info(f"[OK] {base_name} -> {row['voxel_file']} ({row['time_seconds']}s)")
    return {"source": src_path, "status": "ok", "meta": row}

# -----------------------------
# Runner com detecção de device e fallback
# -----------------------------

def decide_device_mode():
    # se torch disponível e cuda ok, podemos marcar 'gpu'
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
    files = []
    for dirpath, _, filenames in os.walk(root_folder):
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
            futures[exe.submit(make_task, f)] = f
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processando modelos", unit="file"):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                logging.error(f"Erro ao processar: {e}")
    summary_path = os.path.join(out_root, "relatorios", f"summary_{int(time.time())}.json")
    try:
        save_json(summary_path, {"processed": len(results), "timestamp": time.ctime(), "results": results})
    except Exception:
        pass
    logging.info("Processamento finalizado. Saída em: %s", out_root)

# -----------------------------
# Execução principal
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
