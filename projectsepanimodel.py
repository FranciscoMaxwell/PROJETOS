import bpy
import os
import numpy as np
import trimesh

# ===============================================================
# CONFIGURAÇÕES
# ===============================================================
VOXEL_RES = 80          # resolução da voxelização
TARGET_HEIGHT = 1.0     # altura total padrão (1.0 ≈ 1m)
EXPORT_DIR = bpy.path.abspath("//exportados")  # pasta de saída
os.makedirs(EXPORT_DIR, exist_ok=True)

# ===============================================================
# FUNÇÕES AUXILIARES
# ===============================================================
def limpar_cena():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def carregar_modelo(caminho):
    ext = os.path.splitext(caminho)[1].lower()
    limpar_cena()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=caminho)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=caminho)
    elif ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=caminho)
    else:
        print(f"Formato não suportado: {ext}")
        return False
    return True

def ajustar_escala_e_centro():
    objs = [o for o in bpy.data.objects if o.type == 'MESH']
    if not objs:
        return

    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()  # junta todas as partes
    obj = bpy.context.object

    # centraliza no mundo e normaliza escala
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
    z_min = min(v.z for v in bbox)
    z_max = max(v.z for v in bbox)
    altura = z_max - z_min
    esc = TARGET_HEIGHT / altura if altura > 0 else 1.0
    obj.scale = (esc, esc, esc)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj

def exportar_glb(obj, nome):
    path = os.path.join(EXPORT_DIR, nome + ".glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=False)
    print(f"✔ Exportado GLB: {path}")
    return path

def voxelizar_para_npy(glb_path, nome):
    mesh = trimesh.load(glb_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    voxel = mesh.voxelized(pitch=mesh.extents.max() / VOXEL_RES)
    npy = voxel.matrix.astype(np.uint8)
    np.save(os.path.join(EXPORT_DIR, nome + ".npy"), npy)
    print(f"✔ Voxel salvo: {nome}.npy")

def separar_e_exportar(caminho, idx_modelo, idx_anim):
    if not carregar_modelo(caminho):
        return idx_modelo, idx_anim

    # separa objetos
    objetos = bpy.data.objects
    tem_armature = any(o.type == 'ARMATURE' for o in objetos)

    # ===================================================
    # exporta modelo estático
    # ===================================================
    modelo_nome = f"modelo_{idx_modelo:03d}"
    obj = ajustar_escala_e_centro()
    if obj:
        glb_path = exportar_glb(obj, modelo_nome)
        voxelizar_para_npy(glb_path, modelo_nome)
        idx_modelo += 1

    # ===================================================
    # exporta animação (se houver)
    # ===================================================
    if tem_armature:
        anim_nome = f"animacao_{idx_anim:03d}"
        exportar_glb(objetos[0], anim_nome)
        idx_anim += 1

    return idx_modelo, idx_anim

# ===============================================================
# LOOP PRINCIPAL
# ===============================================================
from tkinter import Tk, filedialog
Tk().withdraw()
pasta = filedialog.askdirectory(title="Selecione a pasta com .obj/.fbx/.glb")
if not pasta:
    raise SystemExit("Nenhuma pasta selecionada.")

arquivos = [os.path.join(pasta, f) for f in os.listdir(pasta)
            if f.lower().endswith(('.glb', '.gltf', '.obj', '.fbx'))]

idx_modelo = 1
idx_anim = 1

for caminho in arquivos:
    print(f"\n>>> Processando: {os.path.basename(caminho)}")
    idx_modelo, idx_anim = separar_e_exportar(caminho, idx_modelo, idx_anim)

print("\n✅ Processo concluído!")
print(f"Total: {idx_modelo-1} modelos, {idx_anim-1} animações.")
