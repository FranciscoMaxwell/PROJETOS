import os
import uuid
import ctypes
import tempfile
from tkinter import Tk, filedialog, messagebox, Button, Label, Scale, HORIZONTAL
from PIL import Image

# ============================
# Funções principais
# ============================

def limpar_fundo(img, sensibilidade=25):
    """
    Remove o fundo de uma imagem clara sem apagar o interior do objeto.
    Só remove pixels próximos às bordas que se assemelham ao fundo.
    """
    from PIL import ImageFilter

    img = img.convert("RGBA")
    largura, altura = img.size
    pixels = img.load()

    # 1️⃣ Detecta cor média do fundo (pelos 4 cantos)
    amostras = [
        pixels[0, 0],
        pixels[largura - 1, 0],
        pixels[0, altura - 1],
        pixels[largura - 1, altura - 1],
    ]
    media_fundo = tuple(sum(c[i] for c in amostras) // 4 for i in range(3))

    # 2️⃣ Cria uma máscara de bordas para saber onde o objeto começa
    #    Isso ajuda a não apagar pixels internos
    bordas = img.filter(ImageFilter.FIND_EDGES).convert("L")

    # 3️⃣ Converte tudo para edição direta
    mask_pixels = bordas.load()
    limite = max(15, min(120, sensibilidade * 1.2))

    for y in range(altura):
        for x in range(largura):
            r, g, b, a = pixels[x, y]
            if a < 10:
                continue

            # distância de cor em relação ao fundo
            dist = ((r - media_fundo[0]) ** 2 + (g - media_fundo[1]) ** 2 + (b - media_fundo[2]) ** 2) ** 0.5
            brilho = (r + g + b) / 3

            # quanto o pixel está próximo de uma borda detectada
            borda_intensa = mask_pixels[x, y] > 40

            # Apaga apenas se:
            # - for muito parecido com o fundo, e
            # - não for uma borda nem parte do interior do objeto
            if dist < limite and not borda_intensa and brilho > 160:
                pixels[x, y] = (r, g, b, 0)

    return img

def imagem_para_cur(imagem_path, fator_escala=3.0, sensibilidade=25):
    """
    Converte uma imagem (PNG, JPG etc.) em um cursor .cur temporário,
    limpando o fundo e redimensionando até 3x o tamanho padrão.
    Usa nome de arquivo único para evitar conflitos.
    """
    TAM_PADRAO = 32
    tamanho_alvo = int(TAM_PADRAO * fator_escala)
    tamanho_final = (tamanho_alvo, tamanho_alvo)

    img = Image.open(imagem_path).convert("RGBA")
    img = limpar_fundo(img, sensibilidade)
    img = img.resize(tamanho_final, Image.Resampling.LANCZOS)

    temp_dir = tempfile.gettempdir()
    unique_id = uuid.uuid4().hex[:8]
    base_name = os.path.basename(imagem_path).split('.')[0]
    ico_path = os.path.join(temp_dir, f"{base_name}_{unique_id}.ico")
    cur_path = ico_path.replace(".ico", ".cur")

    img.save(ico_path, format="ICO", sizes=[tamanho_final])
    os.rename(ico_path, cur_path)

    return cur_path


def aplicar_cursor_global(cursor_dict):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    cursors = {
        "padrao": 32512,
        "texto": 32513,
        "botao": 32649,
        "carregando": 32514
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

    sensibilidade = escala_sensibilidade.get()
    try:
        cursor_temp = imagem_para_cur(caminho, fator_escala=3.0, sensibilidade=sensibilidade)
        cursor_paths[tipo] = cursor_temp
        messagebox.showinfo(
            "OK",
            f"Imagem de '{tipo}' carregada, fundo limpo (força {sensibilidade}) e ampliada com sucesso!"
        )
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
root.geometry("400x400")
root.resizable(False, False)

Label(root, text="Escolha imagens para cada tipo de cursor", font=("Segoe UI", 10, "bold")).pack(pady=10)

Button(root, text="🖱️ Escolher Cursor Padrão", width=32, command=lambda: escolher_imagem("padrao")).pack(pady=5)
Button(root, text="✍️ Escolher Cursor sobre Texto", width=32, command=lambda: escolher_imagem("texto")).pack(pady=5)
Button(root, text="🔲 Escolher Cursor sobre Botão", width=32, command=lambda: escolher_imagem("botao")).pack(pady=5)
Button(root, text="⌛ Escolher Cursor de Carregamento", width=32, command=lambda: escolher_imagem("carregando")).pack(pady=5)

Label(root, text="Força da remoção de fundo:", font=("Segoe UI", 9, "italic")).pack(pady=(10, 0))
escala_sensibilidade = Scale(root, from_=0, to=100, orient=HORIZONTAL, length=250)
escala_sensibilidade.set(25)  # valor padrão equilibrado
escala_sensibilidade.pack()

Label(root, text="Fundo automaticamente removido e cursor até 3× maior", font=("Segoe UI", 9, "italic")).pack(pady=5)

Button(root, text="✅ Aplicar Todos os Cursores", width=32, command=aplicar_todos).pack(pady=10)
Button(root, text="🔄 Restaurar Cursores Padrão", width=32, command=restaurar_cursor_padrao).pack(pady=5)
Button(root, text="❌ Fechar", width=32, command=fechar).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", fechar)
root.mainloop()
