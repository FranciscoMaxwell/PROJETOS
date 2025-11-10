import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os

class RecorteBorracha:
    def __init__(self, master):
        self.master = master
        self.master.title("🩶 Ferramenta de Recorte Manual (Borracha + IA)")
        self.master.geometry("1100x800")
        self.master.configure(bg="#1e1e1e")

        # Estado interno
        self.img_original = None
        self.img_edicao = None
        self.tk_img = None
        self.zoom = 1.0
        self.borracha_tamanho = 20
        self.arrastando = False

        # Canvas principal
        self.canvas = tk.Canvas(master, bg="#2b2b2b", cursor="circle")
        self.canvas.pack(fill="both", expand=True)

        # Painel superior
        painel = tk.Frame(master, bg="#1e1e1e")
        painel.pack(fill="x", pady=8)

        tk.Button(painel, text="📂 Abrir Imagem", command=self.abrir_imagem).pack(side="left", padx=10)
        tk.Button(painel, text="💾 Salvar PNG", command=self.salvar_imagem).pack(side="left", padx=10)
        tk.Button(painel, text="♻️ Restaurar", command=self.restaurar_imagem).pack(side="left", padx=10)

        tk.Label(painel, text="🩶 Tamanho da Borracha:", bg="#1e1e1e", fg="white").pack(side="left", padx=10)
        self.tamanho_var = tk.StringVar(value="Média")
        tk.OptionMenu(painel, self.tamanho_var, "Mini", "Pequena", "Média", "Grande", command=self.mudar_tamanho).pack(side="left")

        tk.Button(painel, text="🧠 Apagar Fundo Inteligente", command=self.apagar_fundo_inteligente).pack(side="left", padx=10)

        tk.Label(painel, text="🔍 Zoom: role o mouse", bg="#1e1e1e", fg="gray").pack(side="right", padx=10)
        tk.Button(painel, text="❌ Fechar", command=master.destroy).pack(side="right", padx=10)

        # Eventos
        self.canvas.bind("<ButtonPress-1>", self.iniciar_borracha)
        self.canvas.bind("<B1-Motion>", self.usar_borracha)
        self.canvas.bind("<ButtonRelease-1>", self.parar_borracha)
        self.canvas.bind("<MouseWheel>", self.rolar_zoom)

    # ---------------------------------------------------

    def abrir_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Escolher imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff")]
        )
        if not caminho:
            return

        self.img_path = caminho
        self.img_original = Image.open(caminho).convert("RGBA")
        self.img_edicao = self.img_original.copy()
        self.zoom = 1.0
        self.atualizar_imagem()
        messagebox.showinfo("✅", "Imagem carregada! Use a borracha ou o modo inteligente.")

    def atualizar_imagem(self):
        if not self.img_edicao:
            return
        w, h = self.img_edicao.size
        img_zoom = self.img_edicao.resize((int(w * self.zoom), int(h * self.zoom)), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_zoom)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.image = self.tk_img

    def mudar_tamanho(self, escolha):
        tamanhos = {
            "Mini": 4,
            "Pequena": 10,
            "Média": 25,
            "Grande": 50
        }
        self.borracha_tamanho = tamanhos.get(escolha, 20)

    def iniciar_borracha(self, event):
        self.arrastando = True
        self.apagar_pixel(event)

    def usar_borracha(self, event):
        if self.arrastando:
            self.apagar_pixel(event)

    def parar_borracha(self, event):
        self.arrastando = False

    def apagar_pixel(self, event):
        if not self.img_edicao:
            return
        x = int(event.x / self.zoom)
        y = int(event.y / self.zoom)

        draw = ImageDraw.Draw(self.img_edicao)
        draw.ellipse(
            (x - self.borracha_tamanho, y - self.borracha_tamanho,
             x + self.borracha_tamanho, y + self.borracha_tamanho),
            fill=(0, 0, 0, 0)
        )
        self.atualizar_imagem()

    def rolar_zoom(self, event):
        if not self.img_edicao:
            return
        if event.delta > 0:
            self.zoom = min(8.0, self.zoom * 1.1)
        else:
            self.zoom = max(0.2, self.zoom / 1.1)
        self.atualizar_imagem()

    def restaurar_imagem(self):
        if self.img_original:
            self.img_edicao = self.img_original.copy()
            self.zoom = 1.0
            self.atualizar_imagem()

    def apagar_fundo_inteligente(self):
        if not self.img_edicao:
            messagebox.showwarning("Aviso", "Abra uma imagem primeiro.")
            return

        img_np = np.array(self.img_edicao)
        r, g, b, a = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2], img_np[:, :, 3]

        # calcula brilho e remove tons muito claros (mas preserva tons coloridos)
        brilho = 0.299 * r + 0.587 * g + 0.114 * b
        mascara_fundo = (brilho > 235) & (np.abs(r - g) < 20) & (np.abs(r - b) < 20)
        img_np[mascara_fundo, 3] = 0  # torna transparente

        # limpa também cinzas neutros claros
        mascara_cinza = (brilho > 200) & (np.abs(r - g) < 15) & (np.abs(r - b) < 15)
        img_np[mascara_cinza, 3] = 0

        self.img_edicao = Image.fromarray(img_np)
        self.atualizar_imagem()
        messagebox.showinfo("🧠", "Fundo claro removido com sucesso!")

    def salvar_imagem(self):
        if not self.img_edicao:
            messagebox.showwarning("Aviso", "Nenhuma imagem para salvar.")
            return
        caminho = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("Imagem PNG transparente", "*.png")],
            title="Salvar imagem editada"
        )
        if not caminho:
            return
        self.img_edicao.save(caminho, format="PNG")
        messagebox.showinfo("Sucesso", f"Imagem salva como:\n{os.path.basename(caminho)}")

# ---------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = RecorteBorracha(root)
    root.mainloop()
