"""
Blender batch processor v2
- varre recursivamente uma pasta escolhida
- separa modelo / anim / acessórios / cenário
- normaliza escala (TARGET_HEIGHT)
- exporta .glb normalizado (model) e .glb de animação (anim)
- voxeliza para .npy (VOXEL_RES) se numpy+trimesh instalados no Python do Blender
- organiza texturas: move para texturas_modelo_###/, classifica resoluções (4k/2k/1k/low)
- gera mixes (blend) entre texturas de diferentes resoluções opcionalmente
- gera .json por modelo com metadados e atualiza report.csv
Usage:
- Abra Blender
- Scripting -> New -> cole -> Run Script
- Selecione a pasta raiz quando solicitado
"""

import bpy, os, json, csv, shutil, math
from mathutils import Vector
from tkinter import Tk, filedialog

# ----------------------------
# CONFIGURAÇÕES (edite aqui)
# ----------------------------
VOXEL_RES = 80
TARGET_HEIGHT = 2.0           # você pediu 2.0 para corpos com objetos; ajuste se quiser 1.7 etc.
EXPORT_BASE = bpy.path.abspath("//processed_dataset")
PROCESS_ONLY_WITH_ANIM = True
DIST_ACCESSORY_LIMIT = 0.4    # relativo ao maior eixo do corpo (ajuste)
SCENE_SIZE_RATIO = 0.6
SUPPORTED_EXT = (".glb", ".gltf", ".fbx", ".obj")

# TEXTURAS
MIX_TEXTURES = True           # gera mixes (blend) entre resoluções
MOVE_TEXTURES = True          # se True -> move texturas originais para pasta do dataset; se False -> copia
MIX_RATIO = 0.5               # blend ratio para mix (0.0 = só primeiro, 1.0 = só segundo)
TEXTURE_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr")
RES_THRESHOLDS = {"4k":3000, "2k":1800, "1k":900}  # px thresholds

# KEYWORDS pra detectar acessórios (ajuste se quiser)
KEYWORD_ACCESSORIES = ["hat","cap","helmet","sword","blade","knife","umbrella","gun","pistol",
                       "shield","backpack","bag","axe","spear","staff","prop","weapon","glove","boots"]

# ----------------------------
# PREPARA PASTAS DE SAÍDA
# ----------------------------
os.makedirs(EXPORT_BASE, exist_ok=True)
EXPORT_MODELS = os.path.join(EXPORT_BASE, "models"); os.makedirs(EXPORT_MODELS, exist_ok=True)
EXPORT_ANIMS  = os.path.join(EXPORT_BASE, "anims");  os.makedirs(EXPORT_ANIMS, exist_ok=True)
EXPORT_NPY    = os.path.join(EXPORT_BASE, "npy");    os.makedirs(EXPORT_NPY, exist_ok=True)
EXPORT_JSON   = os.path.join(EXPORT_BASE, "meta");   os.makedirs(EXPORT_JSON, exist_ok=True)
EXPORT_TEX    = os.path.join(EXPORT_BASE, "textures");os.makedirs(EXPORT_TEX, exist_ok=True)

# ----------------------------
# Tenta importar libs opcionais
# ----------------------------
HAS_TRIMESH = True
HAS_PIL = True
try:
    import numpy as np
    import trimesh
except Exception as e:
    print("⚠️ numpy/trimesh não disponível — voxelização será pulada. Para habilitar, instale numpy+trimesh no Python do Blender.")
    HAS_TRIMESH = False

try:
    from PIL import Image
except Exception as e:
    print("⚠️ Pillow (PIL) não disponível — processamento avançado de texturas será limitado.")
    HAS_PIL = False

# ----------------------------
# UTIL: limpeza/ import
# ----------------------------
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError("Formato não suportado: " + ext)

def get_mesh_objects():
    return [o for o in bpy.data.objects if o.type == 'MESH']

def get_armatures():
    return [o for o in bpy.data.objects if o.type == 'ARMATURE']

def has_animation():
    if bpy.data.actions:
        return True
    for arm in get_armatures():
        if arm.animation_data and (arm.animation_data.action or arm.animation_data.nla_tracks):
            return True
    for o in bpy.data.objects:
        if o.animation_data and o.animation_data.action:
            return True
    return False

# ----------------------------
# GEOMETRIA: volume, groups
# ----------------------------
def compute_volume(obj):
    dims = obj.dimensions
    return max(0.0, dims.x * dims.y * dims.z)

def determine_groups(meshes):
    if not meshes:
        return None, [], []
    volumes = {m: compute_volume(m) for m in meshes}
    main = max(volumes, key=volumes.get)
    main_vol = volumes[main]
    main_centroid = main.location.copy()
    accessories = []
    scene = []
    for m, v in volumes.items():
        if m == main:
            continue
        dist = (m.location - main_centroid).length
        if v >= SCENE_SIZE_RATIO * main_vol:
            scene.append(m)
        elif dist > DIST_ACCESSORY_LIMIT * max(main.dimensions):
            scene.append(m)
        else:
            accessories.append(m)
    return main, accessories, scene

# ----------------------------
# NORMALIZAÇÃO E JUNÇÃO
# ----------------------------
def normalize_and_join(mesh_list, target_height=TARGET_HEIGHT):
    bpy.ops.object.select_all(action='DESELECT')
    for m in mesh_list:
        try:
            m.select_set(True)
        except Exception:
            pass
    bpy.context.view_layer.objects.active = mesh_list[0]
    try:
        bpy.ops.object.join()
    except Exception:
        pass
    obj = bpy.context.object
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    verts_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    zmin = min(v.z for v in verts_world)
    zmax = max(v.z for v in verts_world)
    height = zmax - zmin if (zmax - zmin) > 0 else 1.0
    scale = target_height / height
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj

# ----------------------------
# EXPORT HELPERS
# ----------------------------
def export_glb_all(out_path, use_animation=False, use_selection=False):
    try:
        bpy.ops.export_scene.gltf(filepath=out_path, export_format='GLB', use_selection=use_selection,
                                 export_apply=True, export_animation=use_animation)
        return True
    except Exception as e:
        print("Erro export GLB:", e)
        return False

# ----------------------------
# VOXELIZAÇÃO (trimesh)
# ----------------------------
def voxelize_glb_to_npy(glb_path, npy_path, res=VOXEL_RES):
    if not HAS_TRIMESH:
        return False, "trimesh_missing"
    try:
        mesh = trimesh.load(glb_path)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        pitch = mesh.extents.max() / float(res)
        vg = mesh.voxelized(pitch=pitch)
        mat = vg.matrix.astype(np.uint8)
        np.save(npy_path, mat)
        return True, None
    except Exception as e:
        return False, str(e)

# ----------------------------
# TEXTURAS: localizar e mover/copy + classify + mix
# ----------------------------
def collect_textures_from_blender():
    # percorre materiais e coleta images com filepath
    textures = []
    for img in bpy.data.images:
        if not img.filepath:
            continue
        fp = bpy.path.abspath(img.filepath)
        if os.path.exists(fp):
            textures.append(fp)
    # unique
    textures = list(dict.fromkeys(textures))
    return textures

def collect_textures_from_folder(model_folder):
    found = []
    for root,_,files in os.walk(model_folder):
        for f in files:
            if f.lower().endswith(TEXTURE_EXTS):
                found.append(os.path.join(root,f))
    return list(dict.fromkeys(found))

def classify_texture_size(path):
    if not HAS_PIL:
        return "unknown", (0,0)
    try:
        with Image.open(path) as im:
            w,h = im.size
    except Exception:
        return "unknown", (0,0)
    max_side = max(w,h)
    if max_side >= RES_THRESHOLDS["4k"]:
        return "4k", (w,h)
    elif max_side >= RES_THRESHOLDS["2k"]:
        return "2k", (w,h)
    elif max_side >= RES_THRESHOLDS["1k"]:
        return "1k", (w,h)
    else:
        return "low", (w,h)

def safe_move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if MOVE_TEXTURES:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)
        return True
    except Exception as e:
        print("Erro movendo/copiando textura:", e)
        return False

def mix_textures(path_a, path_b, out_path, ratio=MIX_RATIO):
    if not HAS_PIL:
        return False, "pil_missing"
    try:
        a = Image.open(path_a).convert("RGBA")
        b = Image.open(path_b).convert("RGBA")
        # resize both to same size (max of both)
        target = (max(a.width,b.width), max(a.height,b.height))
        a = a.resize(target, Image.LANCZOS)
        b = b.resize(target, Image.LANCZOS)
        blended = Image.blend(a, b, alpha=ratio)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        blended.save(out_path)
        return True, None
    except Exception as e:
        return False, str(e)

# ----------------------------
# auto label accessories by name
# ----------------------------
def detect_accessory_keywords_by_names(mesh_objs):
    found = []
    for m in mesh_objs:
        n = m.name.lower()
        for kw in KEYWORD_ACCESSORIES:
            if kw in n:
                found.append(kw)
    return list(set(found))

# ----------------------------
# PROCESSA UM ARQUIVO
# ----------------------------
def process_file(fullpath, counters, report_rows):
    print("\n>>> Processing:", fullpath)
    clean_scene()
    try:
        import_file(fullpath)
    except Exception as e:
        print("Import failed:", e)
        return counters

    anim_present = has_animation()
    if PROCESS_ONLY_WITH_ANIM and (not anim_present):
        print("ℹ️ Sem animação detectada — pulando (configurado para processar apenas com animação).")
        return counters

    meshes = get_mesh_objects()
    armatures = get_armatures()
    main, accessories, scene = determine_groups(meshes)

    if not main:
        print("⚠️ Nenhum mesh principal detectado. Pulando.")
        return counters

    # ========= EXPORT MODEL (main + accessories close) =========
    model_group = [main] + [m for m in accessories if m not in scene]
    model_obj = normalize_and_join(model_group, target_height=TARGET_HEIGHT)
    model_idx = counters['model']; model_name = f"modelo_{model_idx:03d}"
    model_glb_path = os.path.join(EXPORT_MODELS, model_name + ".glb")
    # export all (scene contains other objects; here we export entire scene after join)
    export_glb_all(model_glb_path, use_animation=False, use_selection=False)
    counters['model'] += 1

    # ========= EXPORT ACCESSORIES =========
    acc_glb_path = None
    if accessories:
        # reimport to get original accessory objects standalone
        clean_scene(); import_file(fullpath)
        meshes2 = get_mesh_objects()
        main2, accessories2, scene2 = determine_groups(meshes2)
        if accessories2:
            bpy.ops.object.select_all(action='DESELECT')
            for o in accessories2:
                try: o.select_set(True)
                except: pass
            bpy.context.view_layer.objects.active = accessories2[0]
            try:
                bpy.ops.object.join()
            except: pass
            acc_idx = counters['accessory']; acc_name = f"acessorios_{acc_idx:03d}"
            acc_glb_path = os.path.join(EXPORT_MODELS, acc_name + ".glb")
            try:
                bpy.ops.export_scene.gltf(filepath=acc_glb_path, export_format='GLB', use_selection=True, export_apply=True, export_animation=False)
                counters['accessory'] += 1
            except Exception as e:
                print("Erro export accessory:", e)
                acc_glb_path = None

    # ========= EXPORT SCENE =========
    scene_glb_path = None
    if scene:
        clean_scene(); import_file(fullpath)
        meshes3 = get_mesh_objects()
        main3, accessories3, scene3 = determine_groups(meshes3)
        if scene3:
            bpy.ops.object.select_all(action='DESELECT')
            for o in scene3:
                try: o.select_set(True)
                except: pass
            bpy.context.view_layer.objects.active = scene3[0]
            try:
                bpy.ops.object.join()
            except: pass
            scene_idx = counters['scene']; scene_name = f"cenario_{scene_idx:03d}"
            scene_glb_path = os.path.join(EXPORT_MODELS, scene_name + ".glb")
            try:
                bpy.ops.export_scene.gltf(filepath=scene_glb_path, export_format='GLB', use_selection=True, export_apply=True, export_animation=False)
                counters['scene'] += 1
            except Exception as e:
                print("Erro export scene:", e)
                scene_glb_path = None

    # ========= EXPORT ANIM =========
    anim_glb_path = None
    if anim_present and get_armatures():
        clean_scene(); import_file(fullpath)
        bpy.ops.object.select_all(action='DESELECT')
        for a in get_armatures():
            try: a.select_set(True)
            except: pass
        anim_idx = counters['anim']; anim_name = f"animacao_{anim_idx:03d}"
        anim_glb_path = os.path.join(EXPORT_ANIMS, anim_name + ".glb")
        try:
            bpy.ops.export_scene.gltf(filepath=anim_glb_path, export_format='GLB', use_selection=True, export_apply=False, export_animation=True)
            counters['anim'] += 1
        except Exception as e:
            print("Erro export anim:", e)
            anim_glb_path = None

    # ========= VOXELIZE (if available) =========
    voxel_model_status = "skipped"
    if os.path.exists(model_glb_path) and HAS_TRIMESH:
        npy_model = os.path.join(EXPORT_NPY, model_name + ".npy")
        ok, err = voxelize_glb_to_npy(model_glb_path, npy_model, VOXEL_RES)
        voxel_model_status = "ok" if ok else f"fail:{err}"
    voxel_acc_status = "skipped"
    if acc_glb_path and os.path.exists(acc_glb_path) and HAS_TRIMESH:
        npy_acc = os.path.join(EXPORT_NPY, acc_name + ".npy")
        ok, err = voxelize_glb_to_npy(acc_glb_path, npy_acc, VOXEL_RES)
        voxel_acc_status = "ok" if ok else f"fail:{err}"
    voxel_scene_status = "skipped"
    if scene_glb_path and os.path.exists(scene_glb_path) and HAS_TRIMESH:
        npy_scene = os.path.join(EXPORT_NPY, scene_name + ".npy")
        ok, err = voxelize_glb_to_npy(scene_glb_path, npy_scene, VOXEL_RES)
        voxel_scene_status = "ok" if ok else f"fail:{err}"

    # ========= TEXTURAS: coleta, move/copy, classify, mix =========
    textures_found = []
    tex_info = {}
    # 1) Desde materiais (caminhos absolutos)
    collected_from_blender = collect_textures_from_blender()
    textures_found.extend(collected_from_blender)
    # 2) Também verifica a pasta do arquivo por texturas soltas
    model_folder = os.path.dirname(fullpath)
    collected_from_folder = collect_textures_from_folder(model_folder)
    for t in collected_from_folder:
        if t not in textures_found:
            textures_found.append(t)

    # cria pasta do modelo para texturas
    tex_dest_folder = os.path.join(EXPORT_TEX, model_name)
    os.makedirs(tex_dest_folder, exist_ok=True)

    moved_list = []
    classified = {}
    mixes_created = []
    for tex_path in textures_found:
        try:
            base = os.path.basename(tex_path)
            # classify
            qual, size = classify_texture_size(tex_path)
            dest_name = f"{model_name}_{os.path.splitext(base)[0]}_{qual}{os.path.splitext(base)[1]}"
            dest_path = os.path.join(tex_dest_folder, dest_name)
            ok = safe_move(tex_path, dest_path)
            if ok:
                moved_list.append(dest_path)
                classified.setdefault(qual, []).append(dest_path)
                tex_info[os.path.basename(dest_path)] = {"quality": qual, "size": size, "orig": tex_path}
        except Exception as e:
            print("Erro handle texture:", e)

    # mixes: para cada par (e.g. 4k + 1k) criar mix se MIX_TEXTURES True
    if MIX_TEXTURES and HAS_PIL:
        quals = list(classified.keys())
        # prefer mixing largest with smallest present
        for high in ("4k","2k","1k"):
            for low in ("1k","low"):
                if high in classified and low in classified:
                    a = classified[high][0]
                    b = classified[low][0]
                    mix_name = f"{model_name}_mix_{high}_{low}.png"
                    mix_path = os.path.join(tex_dest_folder, mix_name)
                    ok, err = mix_textures(a,b,mix_path,ratio=MIX_RATIO)
                    if ok:
                        mixes_created.append(mix_path)
                        tex_info[os.path.basename(mix_path)] = {"quality":"mix", "size": None, "orig": [a,b]}

    # ========= JSON metadata =========
    meta = {
        "source_file": os.path.basename(fullpath),
        "model_glb": os.path.basename(model_glb_path) if model_glb_path else None,
        "model_npy": model_name + ".npy" if voxel_model_status.startswith("ok") else None,
        "accessory_glb": os.path.basename(acc_glb_path) if acc_glb_path else None,
        "accessory_npy": acc_name + ".npy" if voxel_acc_status.startswith("ok") else None,
        "scene_glb": os.path.basename(scene_glb_path) if scene_glb_path else None,
        "scene_npy": scene_name + ".npy" if voxel_scene_status.startswith("ok") else None,
        "has_animation": bool(anim_present),
        "animation_glb": os.path.basename(anim_glb_path) if anim_glb_path else None,
        "textures": tex_info,
        "detected_accessory_keywords": detect_accessory_keywords_by_names(accessories if accessories else []),
        "notes": ""
    }
    json_path = os.path.join(EXPORT_JSON, model_name + ".json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, ensure_ascii=False, indent=2)

    # ========= ADD TO REPORT =========
    report_rows.append({
        "source": fullpath,
        "model_glb": model_glb_path,
        "model_npy": meta["model_npy"],
        "accessory_glb": acc_glb_path,
        "accessory_npy": meta["accessory_npy"],
        "scene_glb": scene_glb_path,
        "scene_npy": meta["scene_npy"],
        "animation_glb": anim_glb_path,
        "has_animation": anim_present,
        "num_textures": len(textures_found),
        "texture_qualities": ",".join(sorted(list(tex_info[k]["quality"] if isinstance(tex_info[k]["quality"],str) else "mix" for k in tex_info))),
        "voxel_model_status": voxel_model_status,
        "voxel_accessory_status": voxel_acc_status,
        "voxel_scene_status": voxel_scene_status
    })

    return counters

# ----------------------------
# RUNNER principal
# ----------------------------
Tk().withdraw()
root = filedialog.askdirectory(title="Selecione a pasta raiz contendo seus modelos (varredura recursiva)")
if not root:
    raise SystemExit("Nenhuma pasta selecionada. Saindo.")

# lista recursiva de arquivos
files_to_process = []
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if f.lower().endswith(SUPPORTED_EXT):
            files_to_process.append(os.path.join(dirpath, f))

print(f"Arquivos encontrados: {len(files_to_process)}")

counters = {"model": 1, "accessory": 1, "scene": 1, "anim": 1}
report_rows = []

for f in files_to_process:
    try:
        counters = process_file(f, counters, report_rows)
    except Exception as e:
        print("Erro geral processando", f, e)

# write CSV
csv_path = os.path.join(EXPORT_BASE, "report.csv")
with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
    fieldnames = ["source","model_glb","model_npy","accessory_glb","accessory_npy","scene_glb","scene_npy","animation_glb","has_animation","num_textures","texture_qualities","voxel_model_status","voxel_accessory_status","voxel_scene_status"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in report_rows:
        writer.writerow(r)

print("\n✅ Processamento finalizado.")
print("Outputs em:", EXPORT_BASE)
if not HAS_TRIMESH:
    print("⚠️ Atenção: trimesh/numpy não estavam disponíveis; os arquivos .npy foram pulados.")
if not HAS_PIL:
    print("⚠️ Atenção: Pillow (PIL) não estava disponível; misturas/inspeção de texturas parcial.")
