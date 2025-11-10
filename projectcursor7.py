import os
import uuid
import ctypes
import tempfile
import threading
import time
import struct
import subprocess
from io import BytesIO
from tkinter import Tk, filedialog, messagebox, Button, Label, Scale, HORIZONTAL, Toplevel
from PIL import Image, ImageSequence

# -----------------------------
# Versão final com botão "Abrir pasta" após gerar .ANI
# -----------------------------

def limpar_fundo(img, sensibilidade=25):
    from PIL import ImageFilter
    img = img.convert("RGBA")

    # Se já tiver transparência, não tenta limpar
    if img.getextrema()[3][0] < 255:
        return img

    largura, altura = img.size
    pixels = img.load()
    amostras = [pixels[0, 0], pixels[largura - 1, 0], pixels[0, altura - 1], pixels[largura - 1, altura - 1]]
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


def imagem_para_cur(imagem_path, fator_escala=3.0, sensibilidade=25, temp_dir=None):
    TAM_PADRAO = 32
    tamanho_alvo = int(TAM_PADRAO * fator_escala)
    tamanho_final = (tamanho_alvo, tamanho_alvo)
    img = Image.open(imagem_path).convert("RGBA")
    img = limpar_fundo(img, sensibilidade)
    img = img.resize(tamanho_final, Image.Resampling.LANCZOS)
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    base_name = os.path.basename(imagem_path).split('.')[0]
    cur_path = os.path.join(temp_dir, f"{base_name}_{unique_id}.cur")
    # salva como ICO e converte binariamente para CUR
    buffer = BytesIO()
    img.save(buffer, format="ICO", sizes=[tamanho_final])
    ico_bytes = buffer.getvalue()
    cur_bytes = bytearray(ico_bytes)
    cur_bytes[2] = 2
    hotspot_x, hotspot_y = (0, 0)
    cur_bytes[10:12] = struct.pack('<H', hotspot_x)
    cur_bytes[12:14] = struct.pack('<H', hotspot_y)
    with open(cur_path, 'wb') as f:
        f.write(cur_bytes)
    return cur_path


def gif_para_curs(gif_path, fator_escala=3.0, sensibilidade=25, temp_dir=None, max_frames=None):
    img = Image.open(gif_path)
    base = os.path.splitext(os.path.basename(gif_path))[0]
    if temp_dir is None:
        temp_dir = os.path.join(tempfile.gettempdir(), f"frames_{base}")
    os.makedirs(temp_dir, exist_ok=True)
    TAM_PADRAO = 32
    tamanho_alvo = int(TAM_PADRAO * fator_escala)
    frames = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if max_frames is not None and i >= max_frames:
            break
        frame = frame.convert("RGBA")
        frame = limpar_fundo(frame, sensibilidade)
        frame = frame.resize((tamanho_alvo, tamanho_alvo), Image.Resampling.LANCZOS)
        cur_path = os.path.join(temp_dir, f"{base}_{i:03d}.cur")
        buffer = BytesIO()
        frame.save(buffer, format="ICO", sizes=[(tamanho_alvo, tamanho_alvo)])
        ico_bytes = buffer.getvalue()
        cur_bytes = bytearray(ico_bytes)
        cur_bytes[2] = 2
        cur_bytes[10:12] = struct.pack('<H', 0)
        cur_bytes[12:14] = struct.pack('<H', 0)
        with open(cur_path, 'wb') as f:
            f.write(cur_bytes)
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


def restaurar_cursor_padrao():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SystemParametersInfoW(0x0057, 0, None, 0)


def _make_chunk_bytes(chunk_id: bytes, data: bytes) -> bytes:
    size = len(data)
    pad = b'\x00' if (size % 2) == 1 else b''
    return chunk_id + struct.pack('<I', size) + data + pad


def frames_para_ani(frames_list, output_ani_path, fps=10, title=None, author=None):
    icons_data = []
    for fpath in frames_list:
        with open(fpath, 'rb') as f:
            icons_data.append(f.read())
    cFrames = len(icons_data)
    cbSizeof = 36
    jifRate = max(1, round(60 / max(1, fps)))
    anih = struct.pack('<9I', cbSizeof, cFrames, cFrames, 0, 0, 0, 0, jifRate, 1)
    rate_data = b''.join(struct.pack('<I', jifRate) for _ in range(cFrames))
    icon_chunks = b''.join(_make_chunk_bytes(b'icon', d) for d in icons_data)
    fram_chunk = _make_chunk_bytes(b'LIST', b'fram' + icon_chunks)
    subchunks = _make_chunk_bytes(b'anih', anih) + _make_chunk_bytes(b'rate', rate_data) + fram_chunk
    total_size = 4 + len(subchunks)
    with open(output_ani_path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', total_size))
        f.write(b'ACON')
        f.write(subchunks)
    return output_ani_path


cursor_paths = {"padrao": None, "texto": None, "botao": None, "carregando": None}
frames_ativos = []
thread_animacao = None
animando = False
temp_storage_dir = os.path.join(tempfile.gettempdir(), "custom_cursors")
os.makedirs(temp_storage_dir, exist_ok=True)


def escolher_imagem(tipo):
    caminho = filedialog.askopenfilename(title=f"Escolher imagem do cursor ({tipo})", filetypes=[("Imagens", "*.png *.jpg *.jpeg *.ico *.cur *.bmp *.gif")])
    if not caminho:
        return
    sensibilidade = escala_sensibilidade.get()
    if caminho.lower().endswith(".gif"):
        max_frames = escala_frames.get()
        frames, pasta = gif_para_curs(caminho, fator_escala=3.0, sensibilidade=sensibilidade, temp_dir=temp_storage_dir, max_frames=max_frames)
        cursor_paths[tipo] = frames[0]
        messagebox.showinfo("GIF convertido", f"GIF convertido em {len(frames)} frames.\nSalvos em:\n{pasta}")
        global frames_ativos
        frames_ativos = frames
        botao_teste_animacao.config(state="normal")
        botao_cur_para_ani.config(state="normal")
    else:
        cursor_temp = imagem_para_cur(caminho, fator_escala=3.0, sensibilidade=sensibilidade, temp_dir=temp_storage_dir)
        cursor_paths[tipo] = cursor_temp
        messagebox.showinfo("OK", f"Imagem '{tipo}' carregada e convertida!\nSalvo em:\n{cursor_temp}")


def animar_cursor(frames, fps):
    global animando
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    intervalo = 1 / max(1, fps)
    while animando:
        for frame in frames:
            if not animando:
                break
            hcursor = user32.LoadImageW(0, frame, 2, 0, 0, 0x00000010)
            if hcursor:
                user32.SetSystemCursor(hcursor, 32512)
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


def abrir_pasta(destino):
    pasta = os.path.dirname(destino)
    if os.path.exists(pasta):
        subprocess.Popen(f'explorer "{pasta}"')


def mostrar_sucesso_ani(caminho):
    win = Toplevel(root)
    win.title("Sucesso")
    Label(win, text=f".ANI salvo com sucesso!\n\n{caminho}", font=("Segoe UI", 9)).pack(padx=15, pady=10)
    Button(win, text="🔍 Abrir pasta", width=20, command=lambda: abrir_pasta(caminho)).pack(pady=5)
    Button(win, text="Fechar", width=20, command=win.destroy).pack(pady=5)


def cur_para_ani_salvar():
    global frames_ativos
    if not frames_ativos:
        pasta = filedialog.askdirectory(title="Selecione a pasta com arquivos .cur")
        if not pasta:
            return
        frames_ativos = [os.path.join(pasta, f) for f in sorted(os.listdir(pasta)) if f.lower().endswith(".cur")]
        if not frames_ativos:
            messagebox.showwarning("Aviso", "Nenhum arquivo .cur encontrado nessa pasta.")
            return
    save = filedialog.asksaveasfilename(defaultextension=".ani", filetypes=[("Animated Cursor", "*.ani")], title="Salvar arquivo .ani como")
    if not save:
        return
    fps = escala_frames.get()
    try:
        out = frames_para_ani(frames_ativos, save, fps=fps, title="Cursor personalizado", author="Python")
        mostrar_sucesso_ani(out)
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao criar .ANI: {e}")


def fechar():
    global animando
    animando = False
    restaurar_cursor_padrao()
    root.destroy()


# -----------------------------
# GUI
# -----------------------------
root = Tk()
root.title("Customizador Global de Cursores")
root.geometry("420x680")
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

botao_cur_para_ani = Button(root, text="🔁 Converter CURs (frames) → .ANI", width=35, state="disabled", command=cur_para_ani_salvar)
botao_cur_para_ani.pack(pady=6)

Button(root, text="✅ Aplicar Todos os Cursores", width=35, command=aplicar_todos).pack(pady=10)
Button(root, text="🔄 Restaurar Cursores Padrão", width=35, command=restaurar_cursor_padrao).pack(pady=5)
Button(root, text="❌ Fechar", width=35, command=fechar).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", fechar)
root.mainloop()
