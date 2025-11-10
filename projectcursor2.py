import os
import ctypes
import tempfile
from tkinter import Tk, filedialog, messagebox, Button, Label
from PIL import Image

# ============================
# Funções principais
# ============================

def imagem_para_cur(imagem_path, tamanho=(96, 96)):
    """
    Converte uma imagem (PNG, JPG etc.) em um cursor .cur temporário.
    """
    img = Image.open(imagem_path).convert("RGBA")
    img.thumbnail(tamanho, Image.Resampling.LANCZOS)

    temp_dir = tempfile.gettempdir()
    cur_path = os.path.join(temp_dir, os.path.basename(imagem_path).split('.')[0] + "_cursor.cur")

    img.save(cur_path, format="ICO")  # salva como ICO
    os.rename(cur_path, cur_path.replace(".ico", ".cur"))
    cur_path = cur_path.replace(".ico", ".cur")

    return cur_path


def aplicar_cursor_global(cursor_dict):
    """
    Aplica múltiplos cursores globalmente.
    """
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    # Códigos de cursores do Windows
    cursors = {
        "padrao": 32512,        # OCR_NORMAL
        "texto": 32513,         # OCR_IBEAM
        "botao": 32649,         # OCR_HAND
        "carregando": 32514     # OCR_WAIT
    }

    for tipo, caminho in cursor_dict.items():
        if not caminho:
            continue
        hcursor = user32.LoadImageW(0, caminho, 2, 0, 0, 0x00000010)
        if not hcursor:
            raise RuntimeError(f"Erro ao carregar cursor: {tipo}")
        user32.SetSystemCursor(hcursor, cursors[tipo])

    print("[OK] Todos os cursores aplicados globalmente.")


def restaurar_cursor_padrao():
    """
    Restaura o cursor padrão do Windows.
    """
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SystemParametersInfoW(0x0057, 0, None, 0)
    print("[INFO] Cursores restaurados ao padrão.")


# ============================
# Interface gráfica (GUI)
# ============================

cursor_paths = {
    "padrao": None,
    "texto": None,
    "botao": None,
    "carregando": None
}

def escolher_imagem(tipo):
    caminho = filedialog.askopenfilename(
        title=f"Escolher imagem do cursor ({tipo})",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.ico *.cur *.bmp")]
    )
    if not caminho:
        return

    try:
        cursor_temp = imagem_para_cur(caminho)
        cursor_paths[tipo] = cursor_temp
        messagebox.showinfo("OK", f"Imagem de '{tipo}' carregada com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", str(e))


def aplicar_todos():
    if not any(cursor_paths.values()):
        messagebox.showwarning("Atenção", "Escolha pelo menos um cursor antes de aplicar.")
        return
    try:
        aplicar_cursor_global(cursor_paths)
        messagebox.showinfo("Sucesso", "Cursores aplicados globalmente!")
    except Exception as e:
        messagebox.showerror("Erro", str(e))


def fechar():
    restaurar_cursor_padrao()
    root.destroy()


# ============================
# Janela principal
# ============================

root = Tk()
root.title("Customizador Global de Cursores")
root.geometry("380x350")
root.resizable(False, False)

Label(root, text="Escolha imagens para cada tipo de cursor", font=("Segoe UI", 10, "bold")).pack(pady=10)

Button(root, text="🖱️ Escolher Cursor Padrão", width=30, command=lambda: escolher_imagem("padrao")).pack(pady=5)
Button(root, text="✍️ Escolher Cursor sobre Texto", width=30, command=lambda: escolher_imagem("texto")).pack(pady=5)
Button(root, text="🔲 Escolher Cursor sobre Botão", width=30, command=lambda: escolher_imagem("botao")).pack(pady=5)
Button(root, text="⌛ Escolher Cursor de Carregamento", width=30, command=lambda: escolher_imagem("carregando")).pack(pady=5)

Label(root, text="Tamanho máximo: 3x (96x96 px)", font=("Segoe UI", 9, "italic")).pack(pady=5)

Button(root, text="✅ Aplicar Todos os Cursores", width=30, command=aplicar_todos).pack(pady=10)
Button(root, text="🔄 Restaurar Cursores Padrão", width=30, command=restaurar_cursor_padrao).pack(pady=5)
Button(root, text="❌ Fechar", width=30, command=fechar).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", fechar)
root.mainloop()
