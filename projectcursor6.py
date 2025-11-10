import os
import uuid
import ctypes
import tempfile
import threading
import time
from tkinter import Tk, filedialog, messagebox, Button, Label, Scale, HORIZONTAL
from PIL import Image, ImageSequence

# ============================
# Funções principais
# ============================

def limpar_fundo(img, sensibilidade=25):
    from PIL import ImageFilter
    img = img.convert("RGBA")
    largura, altura = img.size
    pixels = img.load()

    amostras = [
        pixels[0, 0],
        pixels[largura - 1, 0],
        pixels[0, altura - 1],
        pixels[largura - 1, altura - 1],
    ]
    media_fundo = tuple(sum(c[i] for c in amostras) // 4 for i in range(3))

    bordas = img.filter(ImageFilter.FIND_EDGES).convert("L")
    mask_pixels = bordas.load()
    limite = max(15, min(120, sensibilidade * 1.2))

    for y in range(altura):
        for x in range(largura):
            r, g, b, a = pixels[x, y]
            if a < 10:
                continue
            dist = ((r - media_fundo[0]) ** 2 + (g - media_fundo[1]) ** 2 + (b - media_fundo[2]) ** 2) ** 0.5
            brilho = (r + g + b) / 3
            borda_intensa = mask_pixels[x, y] > 40
            if dist < limite and not borda_intensa and brilho > 160:
                pixels[x, y] = (r, g, b, 0)
    return img


def imagem_para_cur(imagem_path, fator_escala=3.0, sensibilidade=25):
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


def gif_para_curs(gif_path, fator_escala=3.0, sensibilidade=25):
    img = Image.open(gif_path)
    base = os.path.splitext(os.path.basename(gif_path))[0]
    temp_dir = os.path.join(tempfile.gettempdir(), f"frames_{base}")
    os.makedirs(temp_dir, exist_ok=True)

    TAM_PADRAO = 32
    tamanho_alvo = int(TAM_PADRAO * fator_escala)

    frames = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        frame = frame.convert("RGBA")
        frame = limpar_fundo(frame, sensibilidade)
        frame = frame.resize((tamanho_alvo, tamanho_alvo), Image.Resampling.LANCZOS)

        cur_path = os.path.join(temp_dir, f"{base}_{i:03d}.cur")
        ico_path = cur_path.replace(".cur", ".ico")
        frame.save(ico_path, format="ICO", sizes=[(tamanho_alvo, tamanho_alvo)])
        os.rename(ico_path, cur_path)
        frames.append(cur_path)

    return frames, temp_dir


def aplicar_cursor_global(cursor_dict):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    cursors = {"padrao": 32512, "texto": 32513, "botao": 32649, "carregando": 32514}
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

cursor_paths = {"padrao": None, "texto": None, "botao": None, "carregando": None}
frames_ativos = []
thread_animacao = None
animando = False


def escolher_imagem(tipo):
    caminho = filedialog.askopenfilename(
        title=f"Escolher imagem do cursor ({tipo})",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.ico *.cur *.bmp *.gif")]
    )
    if not caminho:
        return

    sensibilidade = escala_sensibilidade.get()

    if caminho.lower().endswith(".gif"):
        frames, pasta = gif_para_curs(caminho, fator_escala=3.0, sensibilidade=sensibilidade)
        cursor_paths[tipo] = frames[0]
        messagebox.showinfo("GIF convertido", f"GIF convertido em {len(frames)} frames.\nSalvos em:\n{pasta}")
        global frames_ativos
        frames_ativos = frames
        botao_teste_animacao.config(state="normal")
    else:
        cursor_temp = imagem_para_cur(caminho, fator_escala=3.0, sensibilidade=sensibilidade)
        cursor_paths[tipo] = cursor_temp
        messagebox.showinfo("OK", f"Imagem '{tipo}' carregada e convertida!")


def animar_cursor(frames, fps):
    global animando
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    cursors = 32512  # padrão
    intervalo = 1 / max(1, fps)

    while animando:
        for frame in frames:
            if not animando:
                break
            hcursor = user32.LoadImageW(0, frame, 2, 0, 0, 0x00000010)
            user32.SetSystemCursor(hcursor, cursors)
            time.sleep(intervalo)


def iniciar_animacao():
    global animando, thread_animacao
    if not frames_ativos:
        messagebox.showwarning("Aviso", "Nenhum GIF convertido ainda.")
        return
    if animando:
        animando = False
        botao_teste_animacao.config(text="▶️ Testar Animação (em tempo real)")
        return

    animando = True
    botao_teste_animacao.config(text="⏹️ Parar Animação")
    fps = escala_frames.get()
    thread_animacao = threading.Thread(target=animar_cursor, args=(frames_ativos, fps), daemon=True)
    thread_animacao.start()


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
    global animando
    animando = False
    restaurar_cursor_padrao()
    root.destroy()


# ============================
# Janela principal
# ============================

root = Tk()
root.title("Customizador Global de Cursores")
root.geometry("420x600")
root.resizable(False, False)

Label(root, text="Escolha imagens para cada tipo de cursor", font=("Segoe UI", 10, "bold")).pack(pady=10)

Button(root, text="🖱️ Cursor Padrão (GIF ou imagem)", width=35, command=lambda: escolher_imagem("padrao")).pack(pady=5)
Button(root, text="✍️ Cursor sobre Texto", width=35, command=lambda: escolher_imagem("texto")).pack(pady=5)
Button(root, text="🔲 Cursor sobre Botão", width=35, command=lambda: escolher_imagem("botao")).pack(pady=5)
Button(root, text="⌛ Cursor de Carregamento", width=35, command=lambda: escolher_imagem("carregando")).pack(pady=5)

Label(root, text="Força da remoção de fundo:", font=("Segoe UI", 9, "italic")).pack(pady=(10, 0))
escala_sensibilidade = Scale(root, from_=0, to=100, orient=HORIZONTAL, length=300)
escala_sensibilidade.set(25)
escala_sensibilidade.pack()

Label(root, text="Máx. de frames do GIF (FPS)", font=("Segoe UI", 9, "italic")).pack(pady=(15, 0))
escala_frames = Scale(root, from_=1, to=30, orient=HORIZONTAL, length=300)
escala_frames.set(10)
escala_frames.pack()

botao_teste_animacao = Button(root, text="▶️ Testar Animação (em tempo real)", width=35, state="disabled", command=iniciar_animacao)
botao_teste_animacao.pack(pady=10)

Button(root, text="✅ Aplicar Todos os Cursores", width=35, command=aplicar_todos).pack(pady=10)
Button(root, text="🔄 Restaurar Cursores Padrão", width=35, command=restaurar_cursor_padrao).pack(pady=5)
Button(root, text="❌ Fechar", width=35, command=fechar).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", fechar)
root.mainloop()
