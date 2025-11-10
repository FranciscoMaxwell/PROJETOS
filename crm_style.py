# crm_style.py
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import ttk as tkttk

# 🎨 Paleta principal — tons de amarelo e cinza neutro
PALETTE = {
    "primary": "#F7C948",   # Amarelo Salesforce
    "secondary": "#FFE08A",
    "background": "#FFFDF6",
    "text": "#2E2E2E",
    "accent": "#FFD43B",
    "border": "#E0C97F"
}


def apply_style(window):
    """Aplica o tema visual em toda a janela principal."""
    style = ttk.Style("flatly")

    # Configuração global
    style.configure(
        ".",
        background=PALETTE["background"],
        foreground=PALETTE["text"],
        font=("Segoe UI", 10)
    )

    # 🟨 Botões
    style.configure(
        "TButton",
        font=("Segoe UI Semibold", 10),
        foreground="#2E2E2E",
        background=PALETTE["primary"],
        borderwidth=0,
        focusthickness=3,
        focuscolor=PALETTE["accent"],
        padding=8
    )
    style.map(
        "TButton",
        background=[("active", PALETTE["accent"])],
        relief=[("pressed", "sunken")]
    )

    # Efeito de hover (opcional, mantém suave)
    def on_enter(e):
        e.widget.configure(cursor="hand2")

    def on_leave(e):
        e.widget.configure(cursor="")

    window.bind_class("TButton", "<Enter>", on_enter)
    window.bind_class("TButton", "<Leave>", on_leave)

    # 🧱 Treeview (listas e tabelas)
    style.configure(
        "Treeview",
        background="#FFF9E8",
        fieldbackground="#FFF9E8",
        foreground=PALETTE["text"],
        font=("Segoe UI", 9)
    )
    style.configure(
        "Treeview.Heading",
        background=PALETTE["primary"],
        foreground="#2E2E2E",
        font=("Segoe UI Semibold", 9)
    )

    # 🏷️ Labels e títulos
    style.configure(
        "TLabel",
        background=PALETTE["background"],
        foreground=PALETTE["text"],
        font=("Segoe UI", 10)
    )
    style.configure(
        "Title.TLabel",
        font=("Segoe UI Black", 16),
        foreground=PALETTE["primary"],
        background=PALETTE["background"]
    )

    # 📦 Frames
    style.configure("TFrame", background=PALETTE["background"])

    # ✏️ Entradas
    style.configure(
        "TEntry",
        fieldbackground="#FFF9E8",
        bordercolor=PALETTE["border"],
        lightcolor=PALETTE["accent"]
    )

    # 🟤 Sidebar refinada
    style.configure("Sidebar.TFrame", background="#F0EFEA")
    style.configure(
        "Sidebar.TLabel",
        background="#F0EFEA",
        foreground=PALETTE["primary"],
        font=("Segoe UI Semibold", 20, "bold")
    )

    window.configure(bg=PALETTE["background"])
    return style


# 🟡 Criação de botão estilizado (sem pack)
def styled_button(parent, text, command=None, icon=None):
    """Cria e retorna um botão estilizado, sem empacotar."""
    btn = ttk.Button(
        parent,
        text=text,
        bootstyle="warning",
        command=command
    )
    return btn


# 🟨 Treeview com rolagem (sem empacotar)
def styled_treeview(parent, columns, headings):
    """Cria e retorna um Treeview estilizado com scrollbar."""
    frame = ttk.Frame(parent)
    tree = ttk.Treeview(frame, columns=columns, show="headings", bootstyle="warning")

    for i, col in enumerate(columns):
        tree.heading(col, text=headings[i])
        tree.column(col, width=120, anchor="center")

    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=vsb.set)

    vsb.pack(side=RIGHT, fill=Y)
    tree.pack(fill=BOTH, expand=True)

    return frame, tree
