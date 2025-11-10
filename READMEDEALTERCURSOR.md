
PRIMEIRO = ESCOLHA A IMAGEM, OQUE VOCÊ VAI MUDAR
SEGUNDO = TESTE A IMAGEM(GIF OU NORMAL)
TERCEIRO = CONVERTA .CUR PARA .ANI(SE FOR GIF)
QUARTO = APLICAR

AO CONVERTER VOCÊ SO TERÁ ELA CONVERTIDA NA PASTA, TERA DE MUDAR NAS OPÇÕES DE MOUSE DO SEU PC PARA APLICA-LÁ



# ============================================================
# 🧩 Customizador Global de Cursores - Versão 9
# ============================================================
# Autor: Maxwell Fernandes
# Descrição:
#   - Converte imagens e GIFs em cursores animados (.CUR / .ANI)
#   - Aplica os cursores globalmente no Windows
#   - Remove bordas indesejadas e suaviza transições
# ============================================================

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
from PIL import Image, ImageSequence, ImageFilter, ImageChops

# ============================================================
# 🗂️ CONFIGURAÇÃO BASE
# ------------------------------------------------------------
# Local onde os cursores processados serão salvos automaticamente.
# Cria a pasta caso ainda não exista.
# ============================================================
BASE_DIR = r"C:\Users\Maxwell Fernandes\Downloads\Textos\cursores"
os.makedirs(BASE_DIR, exist_ok=True)

# ============================================================
# 🎨 UTILITÁRIOS DE IMAGEM (ANTI-BORDA)
# ------------------------------------------------------------
# Funções auxiliares para limpar, suavizar e centralizar imagens.
# ============================================================

def remover_pixels_quase_transparentes(img, threshold=20):
    """Remove pixels quase invisíveis (alpha baixo) para evitar contornos feios."""
    ...

def suavizar_borda(img, radius=1):
    """Aplica blur sutil na camada alpha para suavizar transições."""
    ...

def cortar_e_centralizar(img, tamanho_alvo):
    """Recorta o conteúdo visível e o centraliza dentro de um canvas quadrado."""
    ...

# ============================================================
# 🧠 CONVERSÕES PRINCIPAIS: IMAGEM → CURSOR
# ------------------------------------------------------------
# Funções que geram .cur reais (cursores) a partir de imagens.
# ============================================================

def imagem_para_cur(imagem_or_path, fator_escala=3.0, sensibilidade=25, temp_dir=None):
    """
    Converte imagem em .cur com correção anti-borda e centralização automática.
    Aceita tanto caminhos quanto objetos PIL.Image.
    """
    ...

def limpar_fundo(img, sensibilidade=25):
    """Remove fundos brancos ou sólidos com heurísticas inteligentes."""
    ...

def criar_subpasta_por_gif(gif_path):
    """Cria subpasta única para armazenar frames convertidos de um GIF."""
    ...

def gif_para_curs(gif_path, fator_escala=3.0, sensibilidade=25, temp_dir=None, max_frames=None):
    """
    Converte GIF animado em uma sequência de .cur com limpeza de bordas.
    Retorna a lista de frames gerados e a pasta de destino.
    """
    ...

# ============================================================
# 🌀 MONTAGEM DO ARQUIVO .ANI
# ------------------------------------------------------------
# Agrupa múltiplos .CUR em um único cursor animado (.ANI)
# compatível com o Windows.
# ============================================================

def frames_para_ani(frames_list, output_ani_path, fps=10, title=None, author=None):
    """
    Monta um arquivo .ANI (formato RIFF/ACON) a partir de vários frames .CUR.
    """
    ...

# ============================================================
# ⚙️ APLICAÇÃO GLOBAL DE CURSORES
# ------------------------------------------------------------
# Funções para aplicar ou restaurar cursores no sistema Windows.
# ============================================================

def aplicar_cursor_global(cursor_dict):
    """Aplica cursores personalizados em todo o sistema via registro do Windows."""
    ...

def restaurar_cursor_padrao():
    """Restaura os cursores padrões do Windows."""
    ...

# ============================================================
# 🪟 INTERFACE GRÁFICA (Tkinter)
# ------------------------------------------------------------
# Interface simples e funcional para conversão e testes em tempo real.
# ============================================================

# Variáveis globais
cursor_paths = {"padrao": None, "texto": None, "botao": None, "carregando": None}
frames_ativos = []
thread_animacao = None
animando = False
ultima_pasta = None

# ------------------------------------------------------------
# Funções da GUI
# ------------------------------------------------------------

def escolher_imagem(tipo):
    """Permite selecionar imagem ou GIF e converte automaticamente."""
    ...

def animar_cursor(frames, fps):
    """Anima os cursores no sistema em tempo real."""
    ...

def iniciar_animacao():
    """Controla início e parada da animação de teste."""
    ...

def aplicar_todos():
    """Aplica todos os cursores carregados de uma vez."""
    ...

def cur_para_ani_salvar():
    """Combina múltiplos .CUR em um .ANI e salva automaticamente."""
    ...

def fechar():
    """Fecha o programa e restaura o cursor padrão."""
    ...

# ============================================================
# 🧭 GUI PRINCIPAL
# ============================================================

root = Tk()
root.title("Customizador Global de Cursores")
root.geometry("420x680")
root.resizable(False, False)

# Cabeçalho
Label(root, text="Escolha imagens para cada tipo de cursor", font=("Segoe UI", 10, "bold")).pack(pady=10)

# Botões principais
Button(root, text="🖱️ Cursor Padrão (GIF ou imagem)", width=35, command=lambda: escolher_imagem("padrao")).pack(pady=5)
Button(root, text="✍️ Cursor sobre Texto", width=35, command=lambda: escolher_imagem("texto")).pack(pady=5)
Button(root, text="🔲 Cursor sobre Botão", width=35, command=lambda: escolher_imagem("botao")).pack(pady=5)
Button(root, text="⌛ Cursor de Carregamento", width=35, command=lambda: escolher_imagem("carregando")).pack(pady=5)

# Ajustes finos
Label(root, text="Força da remoção de fundo:", font=("Segoe UI", 9, "italic")).pack(pady=(10, 0))
...

# Encerramento](url)
root.protocol("WM_DELETE_WINDOW", fechar)
root.mainloop()
