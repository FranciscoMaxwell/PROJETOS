import numpy as np
import pyvista as pv
from tkinter import Tk, filedialog

# ======================================================
# 🗂️ ABRIR O EXPLORER PARA ESCOLHER O ARQUIVO
# ======================================================
Tk().withdraw()  # Esconde a janela principal do Tkinter
caminho = filedialog.askopenfilename(
    title="Selecione um arquivo .npy para visualizar",
    filetypes=[("Arquivos NumPy", "*.npy")]
)

if not caminho:
    print("🚫 Nenhum arquivo selecionado. Saindo...")
    exit()

print(f"📁 Arquivo selecionado: {caminho}")

# ======================================================
# 📦 CARREGA O ARQUIVO E CONVERTE PARA MALHA 3D
# ======================================================
try:
    recon = np.load(caminho)
    print(f"✅ Arquivo carregado: {recon.shape}")
except Exception as e:
    print(f"❌ Erro ao carregar o arquivo: {e}")
    exit()

# ======================================================
# 🧱 CONVERSÃO PARA MESH E VISUALIZAÇÃO
# ======================================================
try:
    grid = pv.wrap(recon)
    mesh = grid.contour(isosurfaces=[0.5])

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="orange", show_edges=True)
    plotter.add_axes()
    plotter.add_text("Visualizador 3D - Reconstrução", font_size=12)
    plotter.show()

except Exception as e:
    print(f"⚠️ Erro ao visualizar: {e}")
