import os
import uuid
import ctypes
import tempfile
import threading
import time
import struct
import subprocess
from io import BytesIO
from tkinter import Tk, filedialog, messagebox, Button, Label, Scale, HORIZONTAL
from PIL import Image, ImageSequence, ImageFilter

# ============================
# CONFIGURAÇÕES GERAIS
# ============================
PASTA_SAIDA_BASE = r"C:\Users\Maxwell Fernandes\Downloads\Textos\cursores"
os.makedirs(PASTA_SAIDA_BASE, exist_ok=True)


# ============================
# FUNÇÕES DE PROCESSAMENTO
# ============================

def limpar_fundo(img, sensibilidade=25):
    img = img.convert("RGBA")
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
    buffer = BytesIO()
    img.save(buffer, format="ICO", sizes=[tamanho_final])
    ico_bytes = buffer.getvalue()
    cur_bytes = bytearray(ico_bytes)
    cur_bytes[2] = 2
    cur_bytes[10:12] = struct.pack('<H', 0)
    cur_bytes[12:14] = struct.pack('<H', 0)
    with open(cur_path, 'wb') as f:
        f.write(cur_bytes)
    return cur_path

def gif_para_curs(gif_path, fator_escala=3.0, sensibilidade=25, temp_dir=None, max_frames=None):
    from PIL import ImageSequence, Image

    def cortar_transparente(img):
        """Recorta o conteúdo visível e centraliza sem deixar bordas transparentes."""
        bbox = img.getbbox()
        if not bbox:
            return img
        img_crop = img.crop(bbox)

        # cria novo quadro limpo do tamanho final
        TAM_PADRAO = 32
        tamanho_alvo = int(TAM_PADRAO * fator_escala)
        nova = Image.new("RGBA", (tamanho_alvo, tamanho_alvo), (0, 0, 0, 0))

        # redimensiona mantendo proporção
        img_crop.thumbnail((tamanho_alvo, tamanho_alvo), Image.Resampling.LANCZOS)

        # centraliza o conteúdo visível
        x = (tamanho_alvo - img_crop.width) // 2
        y = (tamanho_alvo - img_crop.height) // 2
        nova.paste(img_crop, (x, y), img_crop)
        return nova

    # garante caminho válido
    if not temp_dir or not isinstance(temp_dir, (str, bytes, os.PathLike)):
        temp_dir = os.path.join(tempfile.gettempdir(), f"frames_temp")

    img = Image.open(gif_path)
    base = os.path.splitext(os.path.basename(gif_path))[0]

    pasta_saida = os.path.join(temp_dir, base)
    os.makedirs(pasta_saida, exist_ok=True)

    frames = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if max_frames is not None and i >= max_frames:
            break

        frame = frame.convert("RGBA")
        frame = limpar_fundo(frame, sensibilidade)
        frame = cortar_transparente(frame)

        cur_path = os.path.join(pasta_saida, f"{base}_{i:03d}.cur")

        buffer = BytesIO()
        frame.save(buffer, format="ICO", sizes=[frame.size])
        ico_bytes = buffer.getvalue()

        cur_bytes = bytearray(ico_bytes)
        cur_bytes[2] = 2  # tipo CUR
        cur_bytes[10:12] = struct.pack('<H', 0)
        cur_bytes[12:14] = struct.pack('<H', 0)

        with open(cur_path, 'wb') as f:
            f.write(cur_bytes)

        frames.append(cur_path)

    return frames, pasta_saida


def _make_chunk_bytes(chunk_id: bytes, data: bytes) -> bytes:
    size = len(data)
    pad = b'\x00' if (size % 2) == 1 else b''
    return chunk_id + struct.pack('<I', size) + data + pad


def frames_para_ani(frames_list, output_dir, fps=10):
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

    # Gera nome resultado(n).ani
    nome_base = "resultado"
    ext = ".ani"
    contador = 0
    while True:
        nome_final = f"{nome_base}{'' if contador == 0 else f'({contador})'}{ext}"
        output_ani_path = os.path.join(output_dir, nome_final)
        if not os.path.exists(output_ani_path):
            break
        contador += 1

    with open(output_ani_path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', total_size))
        f.write(b'ACON')
        f.write(subchunks)
    return output_ani_path


# ============================
# FUNÇÕES DE CURSORES DO WINDOWS
# ============================

def aplicar_cursor_global(cursor_dict):
    import winreg
    reg_path = r"Control Panel\Cursors"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_SET_VALUE) as key:
            if cursor_dict.get("padrao"):
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, cursor_dict["padrao"])
            if cursor_dict.get("texto"):
                winreg.SetValueEx(key, "IBeam", 0, winreg.REG_SZ, cursor_dict["texto"])
            if cursor_dict.get("botao"):
                winreg.SetValueEx(key, "Hand", 0, winreg.REG_SZ, cursor_dict["botao"])
            if cursor_dict.get("carregando"):
                winreg.SetValueEx(key, "AppStarting", 0, winreg.REG_SZ, cursor_dict["carregando"])
        ctypes.windll.user32.SystemParametersInfoW(0x0057, 0, None, 0)
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao aplicar cursores: {e}")


def restaurar_cursor_padrao():
    try:
        ctypes.windll.user32.SystemParametersInfoW(0x0057, 0, None, 0)
        messagebox.showinfo("Restauração", "Cursores padrão restaurados com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao restaurar cursores padrão: {e}")


# ============================
# INTERFACE TKINTER
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
        max_frames = escala_frames.get()
        frames, pasta = gif_para_curs(caminho, 3.0, sensibilidade, max_frames)
        cursor_paths[tipo] = frames[0]
        global frames_ativos
        frames_ativos = frames
        botao_teste_animacao.config(state="normal")
        botao_cur_para_ani.config(state="normal")
        messagebox.showinfo("GIF convertido", f"Frames criados em:\n{pasta}")
    else:
        cursor_temp = imagem_para_cur(caminho, 3.0, sensibilidade)
        cursor_paths[tipo] = cursor_temp
        messagebox.showinfo("OK", f"Imagem '{tipo}' convertida e salva!\n{cursor_temp}")


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

    pasta = os.path.dirname(frames_ativos[0])
    fps = escala_frames.get()
    try:
        out = frames_para_ani(frames_ativos, pasta, fps)
        messagebox.showinfo("Gerado", f".ANI criado em:\n{out}")
        os.startfile(pasta)
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao criar .ANI: {e}")


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
        botao_teste_animacao.config(text="▶️ Testar Animação")
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
# GUI
# ============================

root = Tk()
root.title("Customizador Global de Cursores")
root.geometry("420x660")
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

Label(root, text="Máx. de frames / FPS:", font=("Segoe UI", 9, "italic")).pack(pady=(15, 0))
escala_frames = Scale(root, from_=1, to=30, orient=HORIZONTAL, length=300)
escala_frames.set(10)
escala_frames.pack()

botao_teste_animacao = Button(root, text="▶️ Testar Animação", width=35, state="disabled", command=iniciar_animacao)
botao_teste_animacao.pack(pady=10)

botao_cur_para_ani = Button(root, text="🔁 Converter CURs → .ANI", width=35, state="disabled", command=cur_para_ani_salvar)
botao_cur_para_ani.pack(pady=6)

Button(root, text="✅ Aplicar Todos os Cursores", width=35, command=aplicar_todos).pack(pady=10)
Button(root, text="🔄 Restaurar Cursores Padrão", width=35, command=restaurar_cursor_padrao).pack(pady=5)
Button(root, text="❌ Fechar", width=35, command=fechar).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", fechar)
root.mainloop()
