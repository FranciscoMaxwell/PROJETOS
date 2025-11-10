import os
import ctypes
import tempfile
from tkinter import Tk, filedialog, messagebox, Button, Label
from PIL import Image

# ============================
# Funções principais
# ============================

def imagem_para_cur(imagem_path, tamanho=(48, 48)):
    """
    Converte uma imagem (PNG, JPG etc.) em um cursor .cur temporário.
    """
    img = Image.open(imagem_path).convert("RGBA")
    img.thumbnail(tamanho, Image.Resampling.LANCZOS)

    temp_dir = tempfile.gettempdir()
    cur_path = os.path.join(temp_dir, "cursor_temp.cur")

    img.save(cur_path, format="ICO")  # salva como ICO
    os.rename(cur_path, cur_path.replace(".ico", ".cur"))
    cur_path = cur_path.replace(".ico", ".cur")

    return cur_path


def aplicar_cursor_global(cursor_path):
    """
    Aplica o cursor .cur globalmente no Windows.
    """
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    OCR_NORMAL = 32512  # Cursor padrão
    hcursor = user32.LoadImageW(0, cursor_path, 2, 0, 0, 0x00000010)
    if not hcursor:
        raise RuntimeError("Erro ao carregar cursor")

    user32.SetSystemCursor(hcursor, OCR_NORMAL)
    print("[OK] Cursor aplicado globalmente.")


def restaurar_cursor_padrao():
    """
    Restaura o cursor padrão do Windows.
    """
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SystemParametersInfoW(0x0057, 0, None, 0)
    print("[INFO] Cursor restaurado ao padrão.")


# ============================
# Interface gráfica (GUI)
# ============================

def escolher_imagem():
    caminho = filedialog.askopenfilename(
        title="Escolher imagem do cursor",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.ico *.cur *.bmp")]
    )
    if not caminho:
        return

    try:
        cursor_temp = imagem_para_cur(caminho)
        aplicar_cursor_global(cursor_temp)
        messagebox.showinfo("Sucesso", "Cursor aplicado com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", str(e))


def fechar():
    restaurar_cursor_padrao()
    root.destroy()


# ============================
# Janela principal
# ============================

root = Tk()
root.title("Alterar Cursor Global")
root.geometry("350x180")
root.resizable(False, False)

Label(root, text="Escolha uma imagem para usar como cursor", font=("Segoe UI", 10)).pack(pady=10)
Button(root, text="Selecionar Imagem", font=("Segoe UI", 10), command=escolher_imagem, width=25).pack(pady=10)
Button(root, text="Restaurar Cursor Padrão", font=("Segoe UI", 10), command=restaurar_cursor_padrao, width=25).pack(pady=10)
Button(root, text="Fechar", font=("Segoe UI", 10), command=fechar, width=25).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", fechar)
root.mainloop()
