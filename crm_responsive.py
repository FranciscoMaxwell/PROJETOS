# crm_responsive.py
"""
Módulo de responsividade aprimorado para CRMApp.
- Compatível com pack() e grid()
- Não sobrepõe botões
- Ajusta espaçamento e layout suavemente
"""

import tkinter as tk
import ttkbootstrap as tb

COMPACT_WIDTH = 900
SMALL_WIDTH = 700

def enable_responsiveness(app, compact_width=COMPACT_WIDTH, small_width=SMALL_WIDTH):
    """
    Ativa comportamento responsivo em `app`.
    Deve ser chamado após a criação de sidebar e container.
    """

    if not hasattr(app, "sidebar") or not hasattr(app, "container"):
        print("[crm_responsive] Aviso: sidebar/container não encontrados.")
        return

    sidebar = app.sidebar
    container = app.container

    # estado inicial
    app._is_compact = None

    def _repack_sidebar(mode):
        """Reposiciona a sidebar sem sobrepor conteúdo"""
        sidebar.pack_forget()
        if mode == "compact":
            sidebar.pack(side="top", fill="x", pady=2)
            container.pack(side="top", fill="both", expand=True)
        else:
            sidebar.pack(side="left", fill="y", padx=2, pady=2)
            container.pack(side="right", fill="both", expand=True)

    def _adjust_children(width):
        """Ajusta largura e fonte dos botões"""
        for child in sidebar.winfo_children():
            cls = child.winfo_class().lower()
            try:
                if "button" in cls:
                    if width < small_width:
                        child.configure(width=12)
                    elif width < compact_width:
                        child.configure(width=16)
                    else:
                        child.configure(width=20)
                elif "label" in cls:
                    if width < small_width:
                        child.configure(font=("Segoe UI", 9, "bold"))
                    elif width < compact_width:
                        child.configure(font=("Segoe UI", 10, "bold"))
                    else:
                        child.configure(font=("Segoe UI", 12, "bold"))
            except Exception:
                pass

    def _on_resize(event=None):
        try:
            w = app.winfo_width()
        except Exception:
            return

        compact = w < compact_width
        if compact != app._is_compact:
            app._is_compact = compact
            _repack_sidebar("compact" if compact else "normal")

        _adjust_children(w)

    # vincula evento
    app.bind("<Configure>", lambda e: _on_resize())

    # aplica no início
    app.after(200, _on_resize)

    print("[crm_responsive] Responsividade ativada com sucesso.")
