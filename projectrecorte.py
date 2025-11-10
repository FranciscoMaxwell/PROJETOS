import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os

class RecorteBorracha:
    def __init__(self, master):
        self.master = master
        self.master.title("🩶 Ferramenta de Recorte Manual (Borracha)")
        self.master.geometry("1000x750")
        self.master.configure(bg="#1e1e1e")

        # --- Estado interno ---
        self.img_original = None
        self.img_edicao = None
        self.tk_img = None
        self.zoom = 1.0
        self.borracha_tamanho = 20
        self.arrastando = False

        # --- Canvas ---
        self.canvas = tk.Canvas(master, bg="#2b2b2b", cursor="circle")
        self.canvas.pack(fill="both", expand=True)

        # --- Painel de botões ---
        painel = tk.Frame(master, bg="#1e1e1e")
        painel.pack(fill="x", pady=8)

        tk.Button(painel, text="📂 Abrir Imagem", command=self.abrir_imagem).pack(side="left", padx=10)
        tk.Button(painel, text="💾 Salvar PNG", command=self.salvar_imagem).pack(side="left", padx=10)
        tk.Button(painel, text="♻️ Restaurar", command=self.restaurar_imagem).pack(side="left", padx=10)

        tk.Label(painel, text="🩶 Tamanho da Borracha:", bg="#1e1e1e", fg="white").pack(side="left", padx=10)
        self.tamanho_var = tk.StringVar(value="Média")
        tk.OptionMenu(painel, self.tamanho_var, "Pequena", "Média", "Grande", command=self.mudar_tamanho).pack(side="left")

        tk.Label(painel, text="🔍 Zoom: role o mouse", bg="#1e1e1e", fg="gray").pack(side="right", padx=10)
        tk.Button(painel, text="❌ Fechar", command=master.destroy).pack(side="right", padx=10)

        # --- Eventos ---
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
        messagebox.showinfo("✅", "Imagem carregada! Clique e arraste para apagar o fundo.")

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
        if escolha == "Pequena":
            self.borracha_tamanho = 10
        elif escolha == "Média":
            self.borracha_tamanho = 25
        elif escolha == "Grande":
            self.borracha_tamanho = 50

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
