"""
organizador_voxel_inteligente.py

Batch processor para datasets 3D:
- Varre recursivamente uma pasta raiz e processa arquivos .obj/.glb/.gltf/.fbx/.blend
- Converte arquivos .blend/.fbx para .glb se Blender estiver instalado (modo background)
- Normaliza e centraliza modelos (TARGET_HEIGHT)
- Separa: modelo principal, acessórios próximos, cena/plataforma (mas evita fatiar partes do corpo)
- Preserva casos com múltiplos personagens (não separa se são co-dependentes)
- Classifica e move texturas (4k/2k/1k/low), gera mixes entre resoluções
- Aplica filtro opcional de cor de pele para modelos humanos
- Voxeliza para .npy em RES=80³ (usa trimesh)
- Gera arquivos: modelo_XXX.glb, modelo_XXX.npy, acessorios_XXX.npy, cenario_XXX.npy, modelo_XXX.json
- Gera report.csv com resumo
- Comentários extensivos explicando cada parte do código
"""

import os
import sys
import shutil
import subprocess
import json
import csv
import math
from collections import defaultdict
from tkinter import Tk, filedialog

# ---- requisitos opcionais (tentativa de import). O script funciona parcialmente se faltarem.
HAS_TRIMESH = True
HAS_PIL = True
HAS_PYGLTF = True
try:
    import trimesh
    import numpy as np
except Exception:
    print("! Aviso: trimesh/numpy não disponíveis. Voxelização será pulada.")
    HAS_TRIMESH = False

try:
    from PIL import Image
except Exception:
    print("! Aviso: Pillow (PIL) não disponível. Processamento de texturas será limitado.")
    HAS_PIL = False

try:
    from pygltflib import GLTF2
except Exception:
    print("! Aviso: pygltflib não disponível. Extração direta de imagens de .glb/.gltf pode ser limitada.")
    HAS_PYGLTF = False

# -----------------------------
# CONFIGURAÇÕES (ajuste conforme necessidade)
# -----------------------------
VOXEL_RES = 80                 # resolução 80^3
TARGET_HEIGHT = 2.0            # escala alvo em metros (2.0 por padrão pra corpo + objetos)
DIST_ACCESSORY_LIMIT = 0.4     # proporção (do maior eixo) para classificar "longe" como cena
SCENE_SIZE_RATIO = 0.6         # se objeto tem volume >= 0.6 * main_vol -> é cena/plataforma
SUPPORTED_EXT = (".glb", ".gltf", ".fbx", ".obj", ".blend")

# Texturas
TEXTURE_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr")
RES_THRESHOLDS = {"4k": 3000, "2k": 1800, "1k": 900}

# palavras-chave para identificar humano ou acessórios a partir do nome
HUMAN_KEYWORDS = ["human","man","woman","male","female","person","character","body","bust","avatar"]
ACCESSORY_KEYWORDS = ["hat","cap","helmet","sword","blade","knife","umbrella","gun","pistol","shield","backpack","bag","axe","spear","staff","weapon","prop","glove","boots","wing","wings","tail"]

# estruturas das pastas de saída (será criada dentro da pasta escolhida)
OUT_ROOT_NAME = "processed_dataset_intel"
OUT_MODELS = "models"
OUT_ANIMS = "anims"
OUT_NPY = "npy"
OUT_META = "meta"
OUT_TEX = "textures"

# Se TRUE, tenta chamar blender em background para converter .blend/.fbx -> .glb
AUTO_BLENDER_CONVERT = True

# Move texturas originais (True) ou copia (False)
MOVE_TEXTURES = True

# Mix de texturas
MIX_TEXTURES = True
MIX_RATIO = 0.5

# Process only with animation? (we will process all by default; user earlier wanted only anims optional)
PROCESS_ONLY_WITH_ANIM = False

# -----------------------------
# UTIL: encontrar blender no PATH
# -----------------------------
def find_blender_executable():
    """Procura 'blender' no PATH; retorna caminho ou None."""
    candidates = ["blender", "blender.exe"]
    for c in candidates:
        path = shutil.which(c)
        if path:
            return path
    return None

BLENDER_BIN = find_blender_executable()
if BLENDER_BIN:
    print(f"[info] Blender detectado em: {BLENDER_BIN}")
else:
    print("[info] Blender não encontrado no PATH. Conversão de .blend/.fbx via Blender estará desabilitada.")

# -----------------------------
# FUNÇÕES: conversão via Blender (se necessário)
# -----------------------------
def convert_with_blender_to_glb(src_path, dest_glb_path):
    """
    Usa Blender em modo background para abrir src_path (.blend/.fbx/.obj/.gltf) e exportar para GLB.
    Retorna True se ok, False caso contrário.
    """
    if not BLENDER_BIN:
        print("  -> Blender não disponível; não convertendo:", src_path)
        return False
    # cria um script temporário que o Blender vai executar
    script = f"""
import bpy, sys
bpy.ops.wm.open_mainfile(filepath=r'{src_path}') if '{src_path.lower().endswith(".blend")}' else bpy.ops.wm.read_factory_settings(use_empty=True)
# se não for .blend, importamos com o import correto baseado na extensão:
ext = '{os.path.splitext(src_path)[1].lower()}'
if ext == '.fbx':
    bpy.ops.import_scene.fbx(filepath=r'{src_path}')
elif ext == '.obj':
    bpy.ops.import_scene.obj(filepath=r'{src_path}')
elif ext in ('.glb', '.gltf'):
    bpy.ops.import_scene.gltf(filepath=r'{src_path}')
# export glb
bpy.ops.export_scene.gltf(filepath=r'{dest_glb_path}', export_format='GLB', export_apply=True, export_animation=True)
"""
    # write temp script
    tmp = os.path.join(os.path.dirname(dest_glb_path), "tmp_blender_export.py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(script)
    # call blender
    try:
        subprocess.run([BLENDER_BIN, "-b", "--python", tmp], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(tmp)
        return os.path.exists(dest_glb_path)
    except Exception as e:
        print("  -> Erro ao chamar Blender:", e)
        try:
            os.remove(tmp)
        except:
            pass
        return False

# -----------------------------
# FUNÇÕES: leitura de glb para descobrir animações (pygltflib)
# -----------------------------
def gltf_has_animation(glb_path):
    if not HAS_PYGLTF:
        return False
    try:
        g = GLTF2().load(glb_path)
        return bool(g.animations and len(g.animations) > 0)
    except Exception:
        return False

# -----------------------------
# FUNÇÕES: carregar malha com trimesh (fallback se necessário)
# -----------------------------
def load_trimesh(path):
    """
    Tenta carregar com trimesh.load. Retorna obj trimesh.Trimesh ou trimesh.Scene.
    """
    try:
        obj = trimesh.load(path, force='mesh' if path.lower().endswith('.obj') else None)
        return obj
    except Exception as e:
        print("  -> trimesh.load falhou para", path, ":", e)
        return None

# -----------------------------
# FUNÇÕES: classificação de texturas
# -----------------------------
def classify_texture_size(path):
    """Retorna ('4k'|'2k'|'1k'|'low'|'unknown', (w,h))"""
    if not HAS_PIL:
        return "unknown", (0,0)
    try:
        with Image.open(path) as im:
            w, h = im.size
        max_side = max(w, h)
        if max_side >= RES_THRESHOLDS["4k"]:
            return "4k", (w,h)
        elif max_side >= RES_THRESHOLDS["2k"]:
            return "2k", (w,h)
        elif max_side >= RES_THRESHOLDS["1k"]:
            return "1k", (w,h)
        else:
            return "low", (w,h)
    except Exception as e:
        print("  -> Erro ler textura:", e)
        return "unknown", (0,0)

def avg_color_rgb(path):
    """Calcula média RGB da imagem (retorna tupla normalizada 0-1). Usa pillow."""
    if not HAS_PIL:
        return None
    try:
        with Image.open(path).convert("RGB") as im:
            im = im.resize((64,64))  # reduzir para acelerar
            arr = np.array(im).astype(np.float32) / 255.0
            avg = arr.mean(axis=(0,1))
            return tuple(avg.tolist())
    except Exception:
        return None

def is_skin_color(rgb):
    """
    Heurística simples para detectar tons de pele "naturais".
    Evitamos azuis/verdes fortes. Permitimos gama ampla de tons marrons/bege/rosados.
    rgb = (r,g,b) cada 0..1
    """
    if rgb is None:
        return False
    r,g,b = rgb
    # evita cores com muito azul ou verde dominando:
    if b > r + 0.15 or g > r + 0.2:
        return False
    # evita saturações muito altas em azul/green
    # check luminance - skins have moderate luminance
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    if lum < 0.03 or lum > 0.95:
        return False
    # normalized diff r - g small to positive (leans slightly red)
    if r < 0.25:
        # muito escuro para identificar
        return True  # permite tons muito escuros também
    # else permitimos uma grande faixa baseada em r value
    return True

# -----------------------------
# GEOMETRIA: volumes, centroides e decisão de grupos
# -----------------------------
def compute_approx_volume(mesh):
    """
    Aproxima volume pelo produto das dimensões da bounding box.
    mesh: trimesh.Trimesh ou scene. Se scene: combine.
    """
    try:
        bbox = mesh.bounds  # [[minx,miny,minz],[maxx,maxy,maxz]]
        dims = bbox[1] - bbox[0]
        vol = float(dims[0]*dims[1]*dims[2])
        return vol, dims
    except Exception:
        return 0.0, np.array([0.0,0.0,0.0])

def decide_groups(scene_or_mesh):
    """
    Recebe objeto carregado (trimesh.Scene ou list de Trimesh) e decide:
    - main (mesh principal)
    - accessories (lista de meshes)
    - scene/platform (lista)
    Evita fatiar corpos: se vários meshes tem volume comparável (>=0.25*main), e distancias curtas,
    considera "multi-characters" e retorna main como union (não separa).
    """
    # normalizar para lista de geometries
    geoms = []
    if isinstance(scene_or_mesh, trimesh.Scene):
        for name, g in scene_or_mesh.geometry.items():
            geoms.append((name, g))
    elif isinstance(scene_or_mesh, list):
        geoms = list(enumerate(scene_or_mesh))
    else:
        # single mesh
        return [scene_or_mesh], [], []

    # calcular volume aproximado e centroid
    info = []
    for name, g in geoms:
        vol, dims = compute_approx_volume(g)
        centroid = g.bounds.mean(axis=0)
        info.append({"name": name, "mesh": g, "vol": vol, "dims": dims, "centroid": centroid})

    if not info:
        return [], [], []

    # ordenar por volume decrescente
    info.sort(key=lambda x: x["vol"], reverse=True)
    main = info[0]
    main_vol = main["vol"]
    main_dims = main["dims"]
    main_centroid = main["centroid"]

    accessories = []
    scene_parts = []
    # define threshold for second character detection: if there is another geometry with volume >= 0.25*main and close enough,
    # we will consider it a co-character and prefer to keep them together (i.e. don't separate).
    co_character_candidates = []
    for it in info[1:]:
        dist = np.linalg.norm(it["centroid"] - main_centroid)
        # if volume comparable:
        if it["vol"] >= 0.25 * main_vol and dist < max(main_dims)*1.5:
            co_character_candidates.append(it)

    if co_character_candidates:
        # treat whole cluster (main + candidates) as single main (do not separate)
        main_group = [main] + co_character_candidates
        # everything else classify relative to centroid average
        aggregate_centroid = np.mean([m["centroid"] for m in main_group], axis=0)
        # any geometry far and large => scene, else accessory if near
        for it in info:
            if it in main_group:
                continue
            dist = np.linalg.norm(it["centroid"] - aggregate_centroid)
            if it["vol"] >= SCENE_SIZE_RATIO * main_vol or dist > DIST_ACCESSORY_LIMIT * max(main_dims):
                scene_parts.append(it)
            else:
                accessories.append(it)
        # return main as union of main_group
        return [m["mesh"] for m in main_group], [a["mesh"] for a in accessories], [s["mesh"] for s in scene_parts]
    else:
        # usual single main case
        for it in info[1:]:
            dist = np.linalg.norm(it["centroid"] - main_centroid)
            if it["vol"] >= SCENE_SIZE_RATIO * main_vol or dist > DIST_ACCESSORY_LIMIT * max(main_dims):
                scene_parts.append(it)
            else:
                accessories.append(it)
        return [main["mesh"]], [a["mesh"] for a in accessories], [s["mesh"] for s in scene_parts]

# -----------------------------
# VOXELIZAÇÃO (USANDO TRIMESH)
# -----------------------------
def voxelize_mesh_to_npy(mesh_obj, out_npy_path, res=VOXEL_RES):
    """
    Recebe: trimesh.Trimesh (ou scene concatenated) e salva voxel .npy (bool matrix)
    """
    if not HAS_TRIMESH:
        return False, "trimesh_missing"
    try:
        # ensure we have a single Trimesh
        if isinstance(mesh_obj, trimesh.Scene):
            mesh = mesh_obj.dump(concatenate=True)
        else:
            mesh = mesh_obj
        pitch = mesh.extents.max() / float(res)
        vg = mesh.voxelized(pitch=pitch)
        mat = vg.matrix.astype(np.uint8)
        np.save(out_npy_path, mat)
        return True, None
    except Exception as e:
        return False, str(e)

# -----------------------------
# TEXTURAS: localizar no diretório do modelo e mover/copy
# -----------------------------
def collect_textures_near_file(src_file):
    """
    Procura imagens no mesmo diretório do src_file ou em subpastas comuns (textures/, images/)
    e retorna lista de caminhos absolutos únicos.
    """
    folder = os.path.dirname(src_file)
    found = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(TEXTURE_EXTS):
                found.append(os.path.join(root, f))
    # unique preserve order
    seen = set()
    res = []
    for p in found:
        if p not in seen:
            res.append(p); seen.add(p)
    return res

def move_texture(src, dst_folder, prefix_name):
    os.makedirs(dst_folder, exist_ok=True)
    base = os.path.basename(src)
    dest = os.path.join(dst_folder, f"{prefix_name}_{base}")
    try:
        if MOVE_TEXTURES:
            shutil.move(src, dest)
        else:
            shutil.copy2(src, dest)
        return dest
    except Exception as e:
        print("  -> Erro mover textura:", e)
        return None

def mix_two_textures(a, b, out_path, ratio=MIX_RATIO):
    """
    Blend simples com PIL, salva PNG.
    """
    if not HAS_PIL:
        return False, "PIL_missing"
    try:
        A = Image.open(a).convert("RGBA")
        B = Image.open(b).convert("RGBA")
        # resize to same size (max)
        target = (max(A.width, B.width), max(A.height, B.height))
        A = A.resize(target, Image.LANCZOS)
        B = B.resize(target, Image.LANCZOS)
        M = Image.blend(A, B, alpha=ratio)
        M.save(out_path)
        return True, None
    except Exception as e:
        return False, str(e)

# -----------------------------
# UTIL: json/meta and csv report
# -----------------------------
def write_json_metadata(meta, outpath):
    try:
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print("  -> Erro escrever json:", e)
        return False

# -----------------------------
# MAIN: processamento de um arquivo
# -----------------------------
def process_source_file(src_path, counters, out_root):
    """
    Processa UM arquivo de entrada e gera saídas estruturadas:
    - cria pasta para modelo: out_root/models/modelo_XXX/
    - export glb (se necessário via blender), normaliza via trimesh by bounding box,
    - decide grupos, salva glb/npys, organiza texturas, aplica filtro de pele quando aplicável,
    - gera JSON meta e atualiza contador.
    """
    print("\n>>>> Processando:", src_path)
    filename = os.path.basename(src_path)
    base_noext = os.path.splitext(filename)[0]

    # decide nome e pastas de saída
    idx = counters["model"]
    model_tag = f"modelo_{idx:03d}"
    model_dir = os.path.join(out_root, OUT_MODELS, model_tag)
    os.makedirs(model_dir, exist_ok=True)
    tex_dir = os.path.join(out_root, OUT_TEX, model_tag)
    os.makedirs(tex_dir, exist_ok=True)

    # 1) Se arquivo for .blend ou .fbx - tentar converter para .glb via Blender
    ext = os.path.splitext(src_path)[1].lower()
    local_glb = None
    if ext in (".blend", ".fbx") and AUTO_BLENDER_CONVERT and BLENDER_BIN:
        glb_tmp = os.path.join(model_dir, base_noext + ".glb")
        ok = convert_with_blender_to_glb(src_path, glb_tmp)
        if ok:
            local_glb = glb_tmp
            print("  -> Convertido via Blender para:", local_glb)
        else:
            print("  -> Falha conversão Blender; tentaremos importar direto com trimesh (se possível).")

    # 2) Se já é glb/gltf ou conversão deu certo, processe com trimesh directly
    if ext in (".glb", ".gltf") and not local_glb:
        local_glb = src_path

    # 3) Tentar carregar com trimesh (obj/glb)
    mesh_obj = None
    if local_glb:
        mesh_obj = load_trimesh(local_glb)
    else:
        # for .obj or .fbx fallback (if not converted) try to load original
        mesh_obj = load_trimesh(src_path)

    if mesh_obj is None:
        print("  -> Não foi possível carregar malha com trimesh. Pulando arquivo.")
        return counters

    # 4) Detect animation: for glb use pygltflib; for fbx/blend we used Blender earlier
    has_anim = False
    if ext in (".glb", ".gltf") or local_glb:
        path_for_anim_check = local_glb if local_glb else src_path
        has_anim = gltf_has_animation(path_for_anim_check) if HAS_PYGLTF else False
    elif ext == ".fbx":
        # if converted via blender we may have checked; else assume possible
        has_anim = False

    if PROCESS_ONLY_WITH_ANIM and (not has_anim):
        print("  -> Configurado para processar apenas arquivos com animação; pulando.")
        return counters

    # 5) Decide groups (main, accessories, scene) - uses heuristics in decide_groups()
    main_list, accessory_list, scene_list = decide_groups(mesh_obj)

    # 6) If main_list contains more than one mesh (multi-character cluster) we will keep them together
    #    For convenience concatenate geometries into single Trimesh for export/voxel (if trimesh supports)
    def union_meshes(mesh_list):
        if not mesh_list:
            return None
        if len(mesh_list) == 1:
            return mesh_list[0]
        try:
            # try to concatenate into a single mesh
            return trimesh.util.concatenate(mesh_list)
        except Exception as e:
            print("  -> Falha concatenar meshes:", e)
            return mesh_list[0]

    main_union = union_meshes(main_list)
    accessory_union = union_meshes(accessory_list) if accessory_list else None
    scene_union = union_meshes(scene_list) if scene_list else None

    # 7) Normalize scale and center main_union so that height == TARGET_HEIGHT
    #    We'll compute bounding box height and scale accordingly (centered at origin)
    def normalize_and_export_trimesh(mesh, out_glb_path):
        """
        Normaliza mesh, centraliza em origem e exporta para GLB via trimesh's export.
        """
        try:
            # center to origin by subtracting centroid of bounds
            bbox = mesh.bounds
            center = bbox.mean(axis=0)
            mesh.apply_translation(-center)
            dims = bbox[1] - bbox[0]
            height = float(dims[2]) if dims[2] > 0 else max(dims)
            if height <= 0:
                scale = 1.0
            else:
                scale = TARGET_HEIGHT / height
            mesh.apply_scale(scale)
            # export to glb (trimesh exports to glb via GLTF)
            data = mesh.export(file_type='glb')
            with open(out_glb_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print("  -> Erro normalizar/exportar:", e)
            return False

    model_glb_out = os.path.join(model_dir, model_tag + ".glb")
    ok_norm = normalize_and_export_trimesh(main_union, model_glb_out)
    if not ok_norm:
        print("  -> Falha export GLB do modelo principal; pulando resto.")
        return counters

    # save original source copy inside model dir for reference
    try:
        shutil.copy2(src_path, os.path.join(model_dir, os.path.basename(src_path)))
    except Exception:
        pass

    # 8) Export accessory & scene glbs if present (we export normalized versions too)
    acc_glb_out = None
    if accessory_union is not None:
        acc_glb_out = os.path.join(model_dir, f"acessorios_{model_tag}.glb")
        normalize_and_export_trimesh(accessory_union, acc_glb_out)

    scene_glb_out = None
    if scene_union is not None:
        scene_glb_out = os.path.join(model_dir, f"cenario_{model_tag}.glb")
        normalize_and_export_trimesh(scene_union, scene_glb_out)

    # 9) Voxelizar cada parte (se possível)
    npy_model_out = os.path.join(out_root, OUT_NPY, model_tag + ".npy")
    os.makedirs(os.path.dirname(npy_model_out), exist_ok=True)
    voxel_model_status = "skipped"
    if HAS_TRIMESH:
        ok, err = voxelize_mesh_to_npy(main_union, npy_model_out, VOXEL_RES)
        voxel_model_status = "ok" if ok else f"fail:{err}"
    else:
        voxel_model_status = "trimesh_missing"

    npy_acc_out = None
    if accessory_union is not None:
        npy_acc_out = os.path.join(out_root, OUT_NPY, f"acessorios_{model_tag}.npy")
        if HAS_TRIMESH:
            ok, err = voxelize_mesh_to_npy(accessory_union, npy_acc_out, VOXEL_RES)
            # ignore result; store status if needed

    npy_scene_out = None
    if scene_union is not None:
        npy_scene_out = os.path.join(out_root, OUT_NPY, f"cenario_{model_tag}.npy")
        if HAS_TRIMESH:
            ok, err = voxelize_mesh_to_npy(scene_union, npy_scene_out, VOXEL_RES)

    # 10) TEXTURAS: localizar na pasta do source e mover/categorizar
    textures = collect_textures_near_file(src_path)
    textures_moved = []
    texture_meta = {}
    if textures:
        for t in textures:
            q, size = classify_texture_size(t)
            moved = move_texture(t, tex_dir, model_tag)
            if moved:
                textures_moved.append(moved)
                texture_meta[os.path.basename(moved)] = {"quality": q, "size": size, "orig": t}
    # mixers: mix best (4k/2k with low) to improve dataset variety
    mixes = []
    if MIX_TEXTURES and HAS_PIL and textures_moved:
        # group by quality
        byq = defaultdict(list)
        for p in textures_moved:
            q,_ = classify_texture_size(p)
            byq[q].append(p)
        # pair 4k with 1k or low, 2k with 1k, etc.
        for high, low in [("4k","1k"), ("4k","low"), ("2k","1k"), ("2k","low")]:
            if byq.get(high) and byq.get(low):
                a = byq[high][0]
                b = byq[low][0]
                outmix = os.path.join(tex_dir, f"{model_tag}_mix_{high}_{low}.png")
                ok, err = mix_two_textures(a,b,outmix,MIX_RATIO)
                if ok:
                    mixes.append(outmix)
                    texture_meta[os.path.basename(outmix)] = {"quality":"mix", "size": None, "orig":[a,b]}

    # 11) APLICAR FILTRO DE PELE (se detectarmos que é humano)
    # heurística: se o nome do arquivo (ou diretório pai) contiver palavras de HUMAN_KEYWORDS
    lower_name = filename.lower()
    human_like = any(k in lower_name for k in HUMAN_KEYWORDS)
    # additional heuristic: if there is an image with 'skin' or 'face' in name, tag as human-like
    if not human_like and textures_moved:
        for p in textures_moved:
            if "skin" in os.path.basename(p).lower() or "face" in os.path.basename(p).lower():
                human_like = True
                break

    skin_filter_results = {}
    if human_like and HAS_PIL and textures_moved:
        # tentar identificar texturas que parecem ser pele pela proximidade no nome e by color
        for p in textures_moved:
            bname = os.path.basename(p).lower()
            # guess that files with face/skin/body in name are skin textures
            is_candidate = ("skin" in bname) or ("body" in bname) or ("face" in bname) or ("diffuse" in bname and "cloth" not in bname)
            if is_candidate:
                avg = avg_color_rgb(p)
                ok_skin = is_skin_color(avg)
                skin_filter_results[p] = {"avg_rgb": avg, "is_skin": ok_skin}
                if not ok_skin:
                    # move to a "exotic_colors" subfolder for review, but do NOT delete
                    exotic_dir = os.path.join(model_dir, "textures_exotic_colors")
                    os.makedirs(exotic_dir, exist_ok=True)
                    newp = os.path.join(exotic_dir, os.path.basename(p))
                    try:
                        if MOVE_TEXTURES:
                            shutil.move(p, newp)
                        else:
                            shutil.copy2(p, newp)
                        # update metadata
                        texture_meta[os.path.basename(newp)] = texture_meta.pop(os.path.basename(p), {})
                        texture_meta[os.path.basename(newp)]["note"] = "moved_exotic_color"
                        skin_filter_results[p]["moved_to"] = newp
                    except Exception as e:
                        print("  -> falha mover textura exótica:", e)

    # 12) gerar JSON com meta
    meta = {
        "source_file": src_path,
        "model_tag": model_tag,
        "model_glb": os.path.relpath(model_glb_out, out_root) if model_glb_out else None,
        "accessory_glb": os.path.relpath(acc_glb_out, out_root) if acc_glb_out else None,
        "scene_glb": os.path.relpath(scene_glb_out, out_root) if scene_glb_out else None,
        "model_npy": os.path.relpath(npy_model_out, out_root) if os.path.exists(npy_model_out) else None,
        "accessory_npy": os.path.relpath(npy_acc_out, out_root) if npy_acc_out and os.path.exists(npy_acc_out) else None,
        "scene_npy": os.path.relpath(npy_scene_out, out_root) if npy_scene_out and os.path.exists(npy_scene_out) else None,
        "textures": texture_meta,
        "textures_moved": [os.path.relpath(p, out_root) for p in textures_moved],
        "mixes": [os.path.relpath(m, out_root) for m in mixes],
        "human_like": human_like,
        "skin_filter": skin_filter_results,
        "has_animation": has_anim
    }
    json_out = os.path.join(out_root, OUT_META, model_tag + ".json")
    write_json_metadata(meta, json_out)

    # 13) contabiliza e retorna
    counters["model"] += 1
    if acc_glb_out:
        counters["accessory"] += 1
    if scene_glb_out:
        counters["scene"] += 1
    if has_anim:
        counters["anim"] += 1

    # add row for csv (return minimal info)
    row = {
        "source": src_path,
        "model_tag": model_tag,
        "model_glb": model_glb_out,
        "model_npy": npy_model_out if os.path.exists(npy_model_out) else None,
        "accessory_glb": acc_glb_out,
        "accessory_npy": npy_acc_out if npy_acc_out and os.path.exists(npy_acc_out) else None,
        "scene_glb": scene_glb_out,
        "scene_npy": npy_scene_out if npy_scene_out and os.path.exists(npy_scene_out) else None,
        "has_animation": has_anim,
        "num_textures": len(textures_moved),
        "voxel_status": voxel_model_status
    }
    return counters, row

# -----------------------------
# ENTRYPOINT: escaneia a pasta raiz recursivamente e processa
# -----------------------------
def main():
    print("=== Organizador Voxel Inteligente ===")
    Tk().withdraw()
    root = filedialog.askdirectory(title="Selecione a pasta raiz contendo seus modelos (varredura recursiva)")
    if not root:
        print("Nenhuma pasta selecionada. Saindo.")
        return

    out_root = os.path.join(root, OUT_ROOT_NAME)
    os.makedirs(out_root, exist_ok=True)
    # criar subpastas
    os.makedirs(os.path.join(out_root, OUT_MODELS), exist_ok=True)
    os.makedirs(os.path.join(out_root, OUT_ANIMS), exist_ok=True)
    os.makedirs(os.path.join(out_root, OUT_NPY), exist_ok=True)
    os.makedirs(os.path.join(out_root, OUT_META), exist_ok=True)
    os.makedirs(os.path.join(out_root, OUT_TEX), exist_ok=True)

    # buscar todos os arquivos suportados recursivamente
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(SUPPORTED_EXT):
                files.append(os.path.join(dirpath, f))
    print(f"Arquivos encontrados: {len(files)}")

    counters = {"model": 1, "accessory": 1, "scene": 1, "anim": 1}
    report_rows = []

    for i, f in enumerate(files):
        try:
            result = process_source_file(f, counters, out_root)
            if isinstance(result, tuple):
                counters, row = result
                report_rows.append(row)
            else:
                # old path: just got counters
                counters = result
        except Exception as e:
            print("Erro processando:", f, "->", e)

    # grava CSV
    csv_out = os.path.join(out_root, "report.csv")
    with open(csv_out, "w", newline='', encoding='utf-8') as cf:
        fieldnames = ["source","model_tag","model_glb","model_npy","accessory_glb","accessory_npy","scene_glb","scene_npy","has_animation","num_textures","voxel_status"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in report_rows:
            writer.writerow(r)

    print("=== Concluído ===")
    print("Outputs em:", out_root)
    if not HAS_TRIMESH:
        print("! Aviso: trimesh/numpy não disponível; voxelização não foi feita.")
    if not HAS_PIL:
        print("! Aviso: Pillow não disponível; processamento avançado de texturas limitado.")
    if not HAS_PYGLTF:
        print("! Aviso: pygltflib não disponível; detecção de animação em glb/gltf pode falhar.")
    if not BLENDER_BIN:
        print("! Aviso: Blender não detectado; conversão automática de .blend/.fbx não disponível.")

if __name__ == "__main__":
    main()
