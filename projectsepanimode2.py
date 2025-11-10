"""
Blender batch processor -> separates model/animation, normalizes, exports GLB,
detects accessories / scene parts, voxelizes (requires trimesh + numpy),
writes JSON metadata and report.csv.

Usage:
- Open Blender
- Scripting -> New script -> paste -> Run Script
- Select a folder (it will walk recursively)
"""

import bpy
import os
import math
import csv
import json
from mathutils import Vector

# ----------------------------
# CONFIGURAÇÕES (ajuste aqui)
# ----------------------------
VOXEL_RES = 80                # resolução escolhida (80³)
TARGET_HEIGHT = 1.0           # altura alvo em metros (1.0 => 1m)
EXPORT_BASE = bpy.path.abspath("//processed_dataset")  # pasta onde salvará os outputs
PROCESS_ONLY_WITH_ANIM = True  # se True: só processa arquivos que contenham animação
DIST_ACCESSORY_LIMIT = 0.4     # distância em unidades (proporcional após escala) para considerar "próximo" (ajuste se necessário)
SCENE_SIZE_RATIO = 0.6         # se objeto tem volume maior que SCENE_SIZE_RATIO * corpo_volume => classifica como 'cenario'
SUPPORTED_EXT = (".glb", ".gltf", ".fbx", ".obj")
KEYWORD_ACCESSORIES = ["hat","cap","helmet","sword","blade","knife","umbrella","gun","pistol",
                       "shield","backpack","bag","axe","spear","staff","prop","weapon"]

# ----------------------------
# Prepara pastas
# ----------------------------
os.makedirs(EXPORT_BASE, exist_ok=True)
EXPORT_MODELS = os.path.join(EXPORT_BASE, "models")
EXPORT_ANIMS  = os.path.join(EXPORT_BASE, "anims")
EXPORT_NPY    = os.path.join(EXPORT_BASE, "npy")
EXPORT_JSON   = os.path.join(EXPORT_BASE, "meta")
os.makedirs(EXPORT_MODELS, exist_ok=True)
os.makedirs(EXPORT_ANIMS, exist_ok=True)
os.makedirs(EXPORT_NPY, exist_ok=True)
os.makedirs(EXPORT_JSON, exist_ok=True)

# ----------------------------
# Tentativa de importar numpy/trimesh (voxelização)
# ----------------------------
HAS_TRIMESH = True
try:
    import numpy as np
    import trimesh
except Exception as e:
    print("⚠️ trimesh/numpy não disponíveis. Voxelização será pulada.")
    print("   Para habilitar voxelização: instale numpy e trimesh no Python do Blender.")
    print("   Exemplo (terminal): /path/to/blender/python -m pip install numpy trimesh")
    HAS_TRIMESH = False

# ----------------------------
# Utilitários Blender
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
    # detecta se há ações / keyframes ou armature com animation_data
    if bpy.data.actions:
        return True
    for arm in get_armatures():
        if arm.animation_data and (arm.animation_data.action or arm.animation_data.nla_tracks):
            return True
    # também checar objetos com animation_data
    for o in bpy.data.objects:
        if o.animation_data and o.animation_data.action:
            return True
    return False

# ----------------------------
# Separação baseado em volume e distância
# ----------------------------
def compute_volume(obj):
    # aproximação pelo produto das dimensões
    dims = obj.dimensions
    return max(0.0, dims.x * dims.y * dims.z)

def determine_groups(meshes):
    # escolhe o corpo principal como o mesh de maior volume
    volumes = {m: compute_volume(m) for m in meshes}
    if not volumes:
        return None, [], []
    main = max(volumes, key=volumes.get)
    main_vol = volumes[main]
    main_centroid = main.location.copy()

    accessories = []
    scene = []
    kept = [main]

    for m, v in volumes.items():
        if m == main:
            continue
        # distância ao centro do corpo
        dist = (m.location - main_centroid).length
        # normalized size ratio
        if v >= SCENE_SIZE_RATIO * main_vol:
            scene.append(m)
        elif dist > DIST_ACCESSORY_LIMIT * max(main.dimensions):
            # se muito longe, considera cena/plataforma
            scene.append(m)
        else:
            accessories.append(m)
            kept.append(m)
    return main, accessories, scene

# ----------------------------
# Normalização: centraliza, aplica transform e escala
# ----------------------------
def normalize_and_join(mesh_list):
    # Seleciona todos e junta
    bpy.ops.object.select_all(action='DESELECT')
    for m in mesh_list:
        m.select_set(True)
    bpy.context.view_layer.objects.active = mesh_list[0]
    try:
        bpy.ops.object.join()
    except Exception:
        pass
    obj = bpy.context.object
    # origin para bounds, centraliza
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # calcula altura e escala
    verts_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    zmin = min(v.z for v in verts_world)
    zmax = max(v.z for v in verts_world)
    height = zmax - zmin if (zmax - zmin) > 0 else 1.0
    scale = TARGET_HEIGHT / height
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj

# ----------------------------
# Export helpers
# ----------------------------
def export_glb_selection(out_path, use_animation=False, only_armature=False):
    bpy.ops.object.select_all(action='DESELECT')
    if only_armature:
        for o in get_armatures():
            o.select_set(True)
    else:
        # assume objetos já estão em cena e selecionados se necessário
        pass
    # Use export_scene.gltf with selection True if appropriate
    try:
        bpy.ops.export_scene.gltf(filepath=out_path, export_format='GLB', use_selection=False,
                                 export_apply=True, export_animation=use_animation)
        return True
    except Exception as e:
        print("Erro export GLB:", e)
        return False

# ----------------------------
# Voxelização via trimesh (se disponível)
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
# Auto-labeling accessories via names
# ----------------------------
def detect_accessory_keywords(mesh_list):
    found = []
    for m in mesh_list:
        name = m.name.lower()
        for kw in KEYWORD_ACCESSORIES:
            if kw in name:
                found.append(kw)
    return list(set(found))

# ----------------------------
# Main processing per file
# ----------------------------
def process_file(fullpath, counters, report_rows):
    print("\n>>> Processing:", fullpath)
    clean_scene()
    try:
        import_file(fullpath)
    except Exception as e:
        print("Import failed:", e)
        return counters

    # quick check for animation
    anim_present = has_animation()
    if PROCESS_ONLY_WITH_ANIM and (not anim_present):
        print("ℹ️ Sem animação detectada — pulando (configurado para processar apenas com animação).")
        return counters

    # get mesh objects & armatures
    meshes = get_mesh_objects()
    armatures = get_armatures()
    main, accessories, scene = determine_groups(meshes)

    # Export model: normalize + join main+near accessories? We will export model (only main+near accessories)
    model_group = [main] + [m for m in accessories if (m not in scene)]
    if not model_group:
        print("⚠️ Sem mesh válida para exportar como modelo.")
        return counters

    model_obj = normalize_and_join(model_group)
    # export model glb
    model_idx = counters['model']
    model_name = f"modelo_{model_idx:03d}"
    model_glb_path = os.path.join(EXPORT_MODELS, model_name + ".glb")
    export_glb_selection(model_glb_path, use_animation=False, only_armature=False)
    counters['model'] += 1

    # Export accessories group separately (if any)
    acc_idx = counters['accessory']
    acc_name = f"acessorios_{acc_idx:03d}"
    if accessories:
        # select accessories only
        bpy.ops.object.select_all(action='DESELECT')
        # we removed some accessories when joining model_group; re-import file to get raw objects?
        # Simpler: export the objects that remained in 'scene' or additional meshes previously separated.
        # Because we joined model into single object, reconstruct accessories by importing again a fresh scene:
        clean_scene()
        import_file(fullpath)
        meshes2 = get_mesh_objects()
        # recompute groups
        main2, accessories2, scene2 = determine_groups(meshes2)
        # select accessories2 only
        if accessories2:
            for o in accessories2:
                o.select_set(True)
            # join and normalize accessories
            bpy.context.view_layer.objects.active = accessories2[0]
            bpy.ops.object.join()
            acc_obj = bpy.context.object
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            # keep world position relative to original (we will center when voxelizing)
            acc_glb_path = os.path.join(EXPORT_MODELS, acc_name + ".glb")
            try:
                bpy.ops.export_scene.gltf(filepath=acc_glb_path, export_format='GLB', use_selection=True, export_apply=True, export_animation=False)
                counters['accessory'] += 1
            except Exception as e:
                print("Erro export acessorio:", e)
                acc_glb_path = None
        else:
            acc_glb_path = None
    else:
        acc_glb_path = None

    # Export scene/platform group separately (if any)
    scene_idx = counters['scene']
    scene_name = f"cenario_{scene_idx:03d}"
    if scene:
        clean_scene()
        import_file(fullpath)
        meshes3 = get_mesh_objects()
        main3, accessories3, scene3 = determine_groups(meshes3)
        if scene3:
            for o in scene3:
                o.select_set(True)
            bpy.context.view_layer.objects.active = scene3[0]
            bpy.ops.object.join()
            scene_glb_path = os.path.join(EXPORT_MODELS, scene_name + ".glb")
            try:
                bpy.ops.export_scene.gltf(filepath=scene_glb_path, export_format='GLB', use_selection=True, export_apply=True, export_animation=False)
                counters['scene'] += 1
            except Exception as e:
                print("Erro export cena:", e)
                scene_glb_path = None
        else:
            scene_glb_path = None
    else:
        scene_glb_path = None

    # Export animation (armature + actions) if present
    anim_idx = counters['anim']
    anim_name = f"animacao_{anim_idx:03d}"
    anim_glb_path = None
    if anim_present and get_armatures():
        # reimport clean and export only armature with animation
        clean_scene()
        import_file(fullpath)
        # select armature(s)
        bpy.ops.object.select_all(action='DESELECT')
        for a in get_armatures():
            a.select_set(True)
        try:
            anim_glb_path = os.path.join(EXPORT_ANIMS, anim_name + ".glb")
            bpy.ops.export_scene.gltf(filepath=anim_glb_path, export_format='GLB', use_selection=True, export_apply=False, export_animation=True)
            counters['anim'] += 1
        except Exception as e:
            print("Erro export anim:", e)
            anim_glb_path = None

    # Voxelize outputs if possible
    voxel_result_model = ("skipped", "trimesh_missing")
    voxel_result_accessory = ("skipped", "trimesh_missing")
    voxel_result_scene = ("skipped", "trimesh_missing")

    # Attempt voxelize model glb if exists
    if os.path.exists(model_glb_path) and HAS_TRIMESH:
        npy_model = os.path.join(EXPORT_NPY, model_name + ".npy")
        ok, err = voxelize_glb_to_npy(model_glb_path, npy_model, VOXEL_RES)
        voxel_result_model = ("ok", None) if ok else ("fail", err)
    # accessory
    if acc_glb_path and os.path.exists(acc_glb_path) and HAS_TRIMESH:
        npy_acc = os.path.join(EXPORT_NPY, acc_name + ".npy")
        ok, err = voxelize_glb_to_npy(acc_glb_path, npy_acc, VOXEL_RES)
        voxel_result_accessory = ("ok", None) if ok else ("fail", err)
    # scene
    if scene_glb_path and os.path.exists(scene_glb_path) and HAS_TRIMESH:
        npy_scene = os.path.join(EXPORT_NPY, scene_name + ".npy")
        ok, err = voxelize_glb_to_npy(scene_glb_path, npy_scene, VOXEL_RES)
        voxel_result_scene = ("ok", None) if ok else ("fail", err)

    # auto-generate JSON metadata (basic)
    meta = {
        "source_file": os.path.basename(fullpath),
        "model_glb": os.path.basename(model_glb_path) if model_glb_path else None,
        "model_npy": model_name + ".npy" if voxel_result_model[0] == "ok" else None,
        "accessory_glb": os.path.basename(acc_glb_path) if acc_glb_path else None,
        "accessory_npy": acc_name + ".npy" if voxel_result_accessory[0] == "ok" else None,
        "scene_glb": os.path.basename(scene_glb_path) if scene_glb_path else None,
        "scene_npy": scene_name + ".npy" if voxel_result_scene[0] == "ok" else None,
        "has_animation": bool(anim_present),
        "animation_glb": os.path.basename(anim_glb_path) if anim_glb_path else None,
        "detected_accessory_keywords": detect_accessory_keywords(accessories if accessories else []),
        "notes": ""
    }
    json_path = os.path.join(EXPORT_JSON, model_name + ".json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, ensure_ascii=False, indent=2)

    # Add row to report
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
        "voxel_model_status": voxel_result_model[0],
        "voxel_accessory_status": voxel_result_accessory[0],
        "voxel_scene_status": voxel_result_scene[0]
    })

    return counters

# ----------------------------
# Runner: escolhe pasta e varre recursivamente
# ----------------------------
from tkinter import Tk, filedialog
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

# Counters and report
counters = {"model": 1, "accessory": 1, "scene": 1, "anim": 1}
report_rows = []

for f in files_to_process:
    try:
        counters = process_file(f, counters, report_rows)
    except Exception as e:
        print("Erro geral processando", f, e)

# write CSV report
csv_path = os.path.join(EXPORT_BASE, "report.csv")
with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
    fieldnames = ["source","model_glb","model_npy","accessory_glb","accessory_npy","scene_glb","scene_npy","animation_glb","has_animation",
                  "voxel_model_status","voxel_accessory_status","voxel_scene_status"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in report_rows:
        writer.writerow(r)

print("\n✅ Processamento finalizado.")
print("Outputs em:", EXPORT_BASE)
if not HAS_TRIMESH:
    print("\n⚠️ Atenção: trimesh/numpy não estavam disponíveis; os arquivos .npy foram pulados.")
    print("Para ativar voxelização, instale numpy e trimesh no Python do Blender.")
