import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
from collections import deque

class RecorteBorracha:
    def __init__(self, master):
        self.master = master
        self.master.title("🩶 Ferramenta de Recorte (Borracha + IA + Varinha Mágica)")
        self.master.geometry("1100x800")
        self.master.configure(bg="#1e1e1e")

        # Estado interno
        self.img_original = None
        self.img_edicao = None
        self.tk_img = None
        self.zoom = 1.0
        self.borracha_tamanho = 20
        self.arrastando = False
        self.modo_varinha = False
        self.modo_multi_click = False
        self.tolerancia = 30
        self.linha_ids = []
        self.mascara_acumulada = None
        self.botao_parar = None
        self.animar = False

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
        tk.Button(painel, text="🪄 Varinha Mágica", command=self.ativar_varinha).pack(side="left", padx=10)

        tk.Label(painel, text="🎚️ Tolerância:", bg="#1e1e1e", fg="white").pack(side="left", padx=5)
        self.tolerancia_var = tk.Scale(painel, from_=5, to=100, orient="horizontal", length=150,
                                       bg="#1e1e1e", fg="white", highlightthickness=0,
                                       troughcolor="#3b3b3b", command=self.mudar_tolerancia)
        self.tolerancia_var.set(30)
        self.tolerancia_var.pack(side="left", padx=5)

        tk.Label(painel, text="🔍 Zoom: role o mouse", bg="#1e1e1e", fg="gray").pack(side="right", padx=10)
        tk.Button(painel, text="❌ Fechar", command=master.destroy).pack(side="right", padx=10)

        # Eventos
        self.canvas.bind("<ButtonPress-1>", self.clique_mouse)
        self.canvas.bind("<B1-Motion>", self.usar_borracha)
        self.canvas.bind("<ButtonRelease-1>", self.soltar_mouse)
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
        messagebox.showinfo("✅", "Imagem carregada! Use a borracha, IA ou varinha mágica.")

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
        tamanhos = {"Mini": 4, "Pequena": 10, "Média": 25, "Grande": 50}
        self.borracha_tamanho = tamanhos.get(escolha, 20)

    def mudar_tolerancia(self, valor):
        self.tolerancia = int(valor)

    # ---------------------------------------------------

    def clique_mouse(self, event):
        if not self.img_edicao:
            return

        if self.modo_varinha:
            self.usar_varinha(event)
        else:
            self.arrastando = True
            self.apagar_pixel(event)

    def usar_borracha(self, event):
        if self.arrastando and not self.modo_varinha:
            self.apagar_pixel(event)

    def soltar_mouse(self, event):
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

    # ---------------- VARINHA MÁGICA ----------------

    def ativar_varinha(self):
        self.modo_varinha = not self.modo_varinha
        estado = "Ativada 🪄" if self.modo_varinha else "Desativada"
        self.canvas.config(cursor="tcross" if self.modo_varinha else "circle")

        if self.modo_varinha:
            resp = messagebox.askyesno("Modo da Varinha", 
                "Deseja usar o modo Multi-Click?\n\nSim = Multi-Click (vários pontos)\nNão = Um-Click (um ponto por vez)")
            self.modo_multi_click = resp
            if self.modo_multi_click:
                self.iniciar_multi_click()
            else:
                messagebox.showinfo("Um-Click", "Clique em uma área do fundo e confirme para apagar.")
        else:
            self.encerrar_multi_click()

    def iniciar_multi_click(self):
        if self.mascara_acumulada is None and self.img_edicao:
            h, w = np.array(self.img_edicao).shape[:2]
            self.mascara_acumulada = np.zeros((h, w), dtype=bool)

        if not self.botao_parar:
            self.botao_parar = tk.Button(self.master, text="🛑 Parar Multi-Click", bg="#bb3333", fg="white",
                                         command=self.encerrar_multi_click)
            self.botao_parar.place(x=10, y=50)
        messagebox.showinfo("Multi-Click Ativo", "Clique em várias áreas do fundo.\nQuando terminar, clique em 'Parar Multi-Click'.")

    def encerrar_multi_click(self):
        if self.botao_parar:
            self.botao_parar.destroy()
            self.botao_parar = None

        if self.modo_multi_click and self.mascara_acumulada is not None:
            resp = messagebox.askyesno("Aplicar recorte", "Deseja apagar tudo que foi selecionado?")
            if resp:
                img_np = np.array(self.img_edicao)
                img_np[self.mascara_acumulada, 3] = 0
                self.img_edicao = Image.fromarray(img_np)
                self.atualizar_imagem()
            self.mascara_acumulada = None
            for lid in self.linha_ids:
                self.canvas.delete(lid)
            self.linha_ids.clear()
            self.animar = False
        self.modo_varinha = False
        self.canvas.config(cursor="circle")

    def usar_varinha(self, event):
        if not self.img_edicao:
            return

        x = int(event.x / self.zoom)
        y = int(event.y / self.zoom)
        img_np = np.array(self.img_edicao)
        h, w = img_np.shape[:2]

        if not (0 <= x < w and 0 <= y < h):
            return

        base_color = img_np[y, x, :3]
        tol = self.tolerancia
        visited = np.zeros((h, w), dtype=bool)
        mask = np.zeros((h, w), dtype=bool)
        q = deque([(x, y)])
        visited[y, x] = True

        while q:
            cx, cy = q.popleft()
            color = img_np[cy, cx, :3]
            if np.linalg.norm(color - base_color) < tol:
                mask[cy, cx] = True
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))

        # Mostrar borda pontilhada
        self.mostrar_borda(mask)

        if self.modo_multi_click:
            if self.mascara_acumulada is None:
                self.mascara_acumulada = mask
            else:
                self.mascara_acumulada |= mask
        else:
            resp = messagebox.askyesno("Aplicar recorte", "Deseja apagar o fundo selecionado?")
            for lid in self.linha_ids:
                self.canvas.delete(lid)
            self.linha_ids.clear()
            self.animar = False
            if resp:
                img_np[mask, 3] = 0
                self.img_edicao = Image.fromarray(img_np)
                self.atualizar_imagem()

    def mostrar_borda(self, mask):
        from scipy import ndimage
        contorno = mask ^ ndimage.binary_erosion(mask)
        ys, xs = np.where(contorno)
        coords = list(zip(xs * self.zoom, ys * self.zoom))
        if coords:
            lid = self.canvas.create_line(coords, fill="white", width=2, dash=(5, 3))
            self.linha_ids.append(lid)
            if not self.animar:
                self.animar = True
                self.animar_linhas()

    def animar_linhas(self):
        if not self.animar or not self.linha_ids:
            return
        for lid in self.linha_ids:
            dash = self.canvas.itemcget(lid, "dash")
            if dash:
                d = list(map(int, dash.split()))
                d = d[::-1]
                self.canvas.itemconfig(lid, dash=d)
        self.master.after(200, self.animar_linhas)

    # ---------------------------------------------------

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
        brilho = 0.299 * r + 0.587 * g + 0.114 * b
        mascara_fundo = (brilho > 235) & (np.abs(r - g) < 20) & (np.abs(r - b) < 20)
        img_np[mascara_fundo, 3] = 0
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
