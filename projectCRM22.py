# PARTE A - Base, DB, helpers, ModuleFrame e SalesModule
# Requisitos: pandas, matplotlib, ttkbootstrap

import sqlite3
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import datetime, time, threading, csv, os
import tkinter.ttk as ttk

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from crm_utils import multi_delete, db_execute
from crm_style import apply_style, styled_button, styled_treeview


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crm.db")
AUTOMATION_INTERVAL_SECONDS = 30

# --------------------------
# Database init & seed
# --------------------------
def db_connect():
    return sqlite3.connect(DB_PATH)

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    # core tables
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, phone TEXT, source TEXT, status TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, phone TEXT, account TEXT, tags TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, industry TEXT, website TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS opportunities (
        id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, contact_id INTEGER, account_id INTEGER, amount REAL, stage TEXT, close_date TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT, contact_id INTEGER, subject TEXT, description TEXT, priority TEXT, status TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id INTEGER, contact_id INTEGER, note TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS campaigns (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, channel TEXT, audience TEXT, sent_count INTEGER DEFAULT 0, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT, name TEXT, price REAL, stock INTEGER
    );
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT, contact_id INTEGER, product_id INTEGER, qty INTEGER, total REAL, status TEXT, order_date TEXT
    );
    CREATE TABLE IF NOT EXISTS automations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, action TEXT, last_run TEXT
    );
    CREATE TABLE IF NOT EXISTS automation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, automation_id INTEGER, message TEXT, created_at TEXT
    );
    """)
    conn.commit()
    conn.close()


def seed_demo_data(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM leads")
    if cur.fetchone()[0] > 0:
        conn.close(); return
    now = datetime.datetime.now().isoformat()
    leads = [
        ('Ana Souza','ana@mail.com','+55 11 90000-0001','Website','new',now),
        ('Carlos Lima','carlos@mail.com','+55 11 90000-0002','Evento','contacted',now),
    ]
    cur.executemany("INSERT INTO leads (name,email,phone,source,status,created_at) VALUES (?,?,?,?,?,?)", leads)
    contacts = [
        ('Ana Souza','ana@mail.com','+55 11 90000-0001','ACME','vip',now),
        ('Mariana Rocha','m.rocha@mail.com','+55 21 90000-0003','Beta SA','lead',now),
    ]
    cur.executemany("INSERT INTO contacts (name,email,phone,account,tags,created_at) VALUES (?,?,?,?,?,?)", contacts)
    products = [
        ('SKU-100','Produto Alpha',299.9,20),
        ('SKU-200','Produto Beta',79.5,100),
    ]
    cur.executemany("INSERT INTO products (sku,name,price,stock) VALUES (?,?,?,?)", products)
    cur.execute("INSERT INTO opportunities (title,contact_id,account_id,amount,stage,close_date,created_at) VALUES (?,?,?,?,?,?,?)",
                ('Venda Produto Alpha', 1, None, 299.9, 'negociação', (datetime.datetime.now()+datetime.timedelta(days=30)).date().isoformat(), now))
    conn.commit(); conn.close()

# --------------------------
# DB wrappers (simple)
# --------------------------
def db_connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def db_fetch(query, params=()):
    conn = db_connect(); cur = conn.cursor()
    cur.execute(query, params); rows = cur.fetchall(); conn.close(); return rows

def db_execute(query, params=()):
    conn = db_connect(); cur = conn.cursor()
    cur.execute(query, params); conn.commit(); last = cur.lastrowid; conn.close(); return last

def db_execute_many(query, rows):
    conn = db_connect(); cur = conn.cursor()
    cur.executemany(query, rows); conn.commit(); conn.close()


# --------------------------
# ModuleFrame base (ttkbootstrap)
# --------------------------
class ModuleFrame(tb.Frame):
    def __init__(self, master, app, title="Módulo"):
        super().__init__(master, padding=10)
        self.app = app
        self.title = title
        self.build_header()
    def build_header(self):
        tb.Label(self, text=self.title, font=('Inter', 16, 'bold')).pack(anchor='w', pady=(0,8))
    def refresh(self):
        pass

# --------------------------
# SalesModule (styled, functional)
# --------------------------
class SalesModule(ModuleFrame):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.build_ui()
        self.refresh()
        self.update_chart()

    # ----------------------------------------------------------
    # CONSTRUÇÃO DA INTERFACE
    # ----------------------------------------------------------
    def build_ui(self):
        title = ttk.Label(
            self,
            text="📈 Sales Cloud — Funil de Vendas Avançado",
            font=("Segoe UI", 16, "bold"),
            bootstyle="primary"
        )
        title.pack(pady=10)

        # --- Painel superior (formulário) ---
        frm_top = ttk.Labelframe(self, text="Cadastrar / Atualizar Lead", padding=10)
        frm_top.pack(fill="x", padx=15, pady=5)

        ttk.Label(frm_top, text="Nome Lead:").grid(row=0, column=0, sticky="w")
        self.entry_nome = ttk.Entry(frm_top, width=25)
        self.entry_nome.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(frm_top, text="Email:").grid(row=0, column=2, sticky="w")
        self.entry_email = ttk.Entry(frm_top, width=25)
        self.entry_email.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(frm_top, text="Status:").grid(row=0, column=4, sticky="w")
        self.combo_status = ttk.Combobox(
            frm_top,
            values=["Novo", "Contato Feito", "Negociação", "Proposta Enviada", "Ganho", "Perdido"],
            width=22,
            state="readonly",
        )
        self.combo_status.grid(row=0, column=5, padx=5, pady=2)
        self.combo_status.set("Novo")

        # --- Botões ---
        frm_btns = ttk.Frame(frm_top)
        frm_btns.grid(row=0, column=6, padx=5)
        ttk.Button(frm_btns, text="💾 Salvar", bootstyle="success", command=self.save_lead).grid(row=0, column=0, padx=2)
        ttk.Button(frm_btns, text="🗑️ Excluir", bootstyle="danger", command=self.del_lead).grid(row=0, column=1, padx=2)
        ttk.Button(frm_btns, text="🔁 Converter → Cliente", bootstyle="info", command=self.convert_to_client).grid(row=0, column=2, padx=2)

        # --- Lista de Leads ---
        frm_tree = ttk.Labelframe(self, text="Leads e Oportunidades", padding=10)
        frm_tree.pack(fill="both", expand=True, padx=15, pady=10)

        cols = ("id", "nome", "email", "status", "notas")
        self.tree = ttk.Treeview(frm_tree, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=150 if c != "id" else 50)
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_lead)

        # --- Painel de Notas ---
        frm_notas = ttk.Labelframe(self, text="🧠 Notas Internas / Atividades", padding=10)
        frm_notas.pack(fill="x", padx=15, pady=10)

        self.text_notas = tk.Text(frm_notas, height=5)
        self.text_notas.pack(fill="x", padx=5, pady=5)

        frm_btn_notas = ttk.Frame(frm_notas)
        frm_btn_notas.pack()
        ttk.Button(frm_btn_notas, text="💾 Salvar Nota", bootstyle="success", command=self.save_note).grid(row=0, column=0, padx=5)
        ttk.Button(frm_btn_notas, text="📜 Ver Histórico", bootstyle="secondary", command=self.show_history).grid(row=0, column=1, padx=5)

        # --- Painel do Gráfico ---
        frm_chart = ttk.Labelframe(self, text="📊 Oportunidades por Status", padding=10)
        frm_chart.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        self.chart = FigureCanvasTkAgg(fig, master=frm_chart)
        self.chart.get_tk_widget().pack(fill="both", expand=True)

    # ----------------------------------------------------------
    # FUNÇÕES CRUD
    # ----------------------------------------------------------
    def refresh(self):
        db_execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT,
            email TEXT,
            status TEXT,
            notas TEXT DEFAULT ''
        )
        """)
        db_execute("""
        CREATE TABLE IF NOT EXISTS interacoes_leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id INTEGER,
            data TEXT,
            descricao TEXT
        )
        """)
        # Forçar adição de colunas que faltam (caso o schema antigo exista)
        for col in ["nome", "email", "status", "notas"]:
            try:
                db_execute(f"ALTER TABLE leads ADD COLUMN {col} TEXT")
            except Exception:
                pass  # já existe, segue normal

        for i in self.tree.get_children():
            self.tree.delete(i)
        leads = db_fetch("SELECT id, nome, email, status, notas FROM leads ORDER BY id DESC")
        for lead in leads:
            self.tree.insert("", "end", values=lead)
        self.update_chart()

    def save_lead(self):
        nome = self.entry_nome.get().strip()
        email = self.entry_email.get().strip()
        status = self.combo_status.get().strip()
        if not nome:
            messagebox.showwarning("Aviso", "Nome obrigatório.")
            return
        sel = self.tree.selection()
        if sel:
            lid = self.tree.item(sel[0])["values"][0]
            db_execute("UPDATE leads SET nome=?, email=?, status=? WHERE id=?", (nome, email, status, lid))
            db_execute("INSERT INTO interacoes_leads (lead_id, data, descricao) VALUES (?, ?, ?)",
                       (lid, datetime.datetime.now().isoformat(), f"Status atualizado: {status}"))
        else:
            db_execute("INSERT INTO leads (nome, email, status) VALUES (?, ?, ?)", (nome, email, status))
        self.refresh()

    def del_lead(self):
        multi_delete(self.tree, "leads", self.refresh, label="lead")

    def on_select_lead(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0])["values"]
        self.entry_nome.delete(0, tk.END)
        self.entry_email.delete(0, tk.END)
        self.entry_nome.insert(0, vals[1])
        self.entry_email.insert(0, vals[2])
        self.combo_status.set(vals[3])
        self.text_notas.delete("1.0", tk.END)
        self.text_notas.insert("end", vals[4])

    # ----------------------------------------------------------
    # NOTAS E HISTÓRICO
    # ----------------------------------------------------------
    def save_note(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Aviso", "Selecione um lead primeiro.")
            return
        lid = self.tree.item(sel[0])["values"][0]
        nota = self.text_notas.get("1.0", tk.END).strip()
        db_execute("UPDATE leads SET notas=? WHERE id=?", (nota, lid))
        db_execute("INSERT INTO interacoes_leads (lead_id, data, descricao) VALUES (?, ?, ?)",
                   (lid, datetime.datetime.now().isoformat(), "Nota atualizada"))
        messagebox.showinfo("Sucesso", "Nota salva e registrada no histórico.")
        self.refresh()

    def show_history(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Aviso", "Selecione um lead.")
            return
        lid = self.tree.item(sel[0])["values"][0]
        interacoes = db_fetch("SELECT data, descricao FROM interacoes_leads WHERE lead_id=? ORDER BY id DESC", (lid,))
        hist_win = tk.Toplevel(self)
        hist_win.title("Histórico de Interações")
        hist_win.geometry("500x400")
        txt = tk.Text(hist_win)
        txt.pack(fill="both", expand=True)
        if interacoes:
            for data, desc in interacoes:
                txt.insert("end", f"[{data}]\n- {desc}\n\n")
        else:
            txt.insert("end", "Nenhum histórico encontrado.")
        txt.config(state="disabled")

    # ----------------------------------------------------------
    # CONVERSÃO LEAD → CLIENTE
    # ----------------------------------------------------------
    def convert_to_client(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Aviso", "Selecione um lead para converter.")
            return

        lid, nome, email, status, _ = self.tree.item(sel[0])["values"]

        if status != "Ganho":
            if not messagebox.askyesno("Confirmação", "Lead ainda não está como 'Ganho'. Converter mesmo assim?"):
                return

        # --- garante que tabela clientes exista ---
        db_execute("""
        CREATE TABLE IF NOT EXISTS clientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            email TEXT,
            telefone TEXT,
            status TEXT DEFAULT 'ativo',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # --- faz a conversão lead → cliente ---
        db_execute("""
            INSERT INTO clientes (nome, email, telefone, status)
            VALUES (?, ?, ?, ?)
        """, (nome, email, "", "Ativo"))

        # --- remove lead convertido ---
        db_execute("DELETE FROM leads WHERE id=?", (lid,))

        # --- adiciona histórico de conversão ---
        db_execute("""
            INSERT INTO interacoes_leads (lead_id, data, descricao)
            VALUES (?, ?, ?)
        """, (lid, datetime.datetime.now().isoformat(), "Lead convertido em cliente."))

        messagebox.showinfo("Conversão", f"Lead '{nome}' convertido em cliente com sucesso.")
        self.refresh()


    # ----------------------------------------------------------
    # GRÁFICO DINÂMICO
    # ----------------------------------------------------------
    def update_chart(self):
        data = db_fetch("SELECT status, COUNT(*) FROM leads GROUP BY status")
        self.ax.clear()
        if data:
            statuses, counts = zip(*data)
            self.ax.bar(statuses, counts)
            self.ax.set_title("Oportunidades por Status")
            self.ax.set_ylabel("Quantidade")
            self.ax.set_xlabel("Status do Funil")
        else:
            self.ax.text(0.5, 0.5, "Sem dados ainda", ha="center", va="center")
        self.chart.draw()

    
# ---- Dialogs (Sales) ----
class LeadDialog(tk.Toplevel):
    def __init__(self, parent, prefill=None):
        super().__init__(parent); self.title("Lead"); self.result=None; self.build(prefill); self.grab_set()
    def build(self, prefill):
        frm = tb.Frame(self, padding=8); frm.pack(fill='both', expand=True)
        tb.Label(frm, text="Nome").grid(row=0,column=0, sticky='e', padx=4, pady=4); self.e_name = tb.Entry(frm); self.e_name.grid(row=0,column=1)
        tb.Label(frm, text="Email").grid(row=1,column=0, sticky='e', padx=4, pady=4); self.e_email = tb.Entry(frm); self.e_email.grid(row=1,column=1)
        tb.Label(frm, text="Telefone").grid(row=2,column=0, sticky='e', padx=4, pady=4); self.e_phone = tb.Entry(frm); self.e_phone.grid(row=2,column=1)
        tb.Label(frm, text="Fonte").grid(row=3,column=0, sticky='e', padx=4, pady=4); self.e_source = tb.Entry(frm); self.e_source.grid(row=3,column=1)
        tb.Button(frm, text="Salvar", bootstyle="primary", command=self.on_save).grid(row=4,column=0,columnspan=2, pady=8)
        if prefill:
            try:
                self.e_name.insert(0, prefill[0]); self.e_email.insert(0, prefill[1]); self.e_phone.insert(0, prefill[2]); self.e_source.insert(0, prefill[3])
            except: pass
    def on_save(self):
        name = self.e_name.get().strip(); email = self.e_email.get().strip(); phone = self.e_phone.get().strip(); source = self.e_source.get().strip()
        if not name: messagebox.showwarning("Validação","Nome obrigatório"); return
        self.result = (name,email,phone,source); self.destroy()

class ContactDialog(tk.Toplevel):
    def __init__(self, parent, prefill=None):
        super().__init__(parent); self.title("Contato"); self.result=None; self.build(prefill); self.grab_set()
    def build(self, prefill):
        frm = tb.Frame(self, padding=8); frm.pack(fill='both', expand=True)
        tb.Label(frm,text="Nome").grid(row=0,column=0,sticky='e',padx=4,pady=4); self.e_name = tb.Entry(frm); self.e_name.grid(row=0,column=1)
        tb.Label(frm,text="Email").grid(row=1,column=0,sticky='e',padx=4,pady=4); self.e_email = tb.Entry(frm); self.e_email.grid(row=1,column=1)
        tb.Label(frm,text="Telefone").grid(row=2,column=0,sticky='e',padx=4,pady=4); self.e_phone = tb.Entry(frm); self.e_phone.grid(row=2,column=1)
        tb.Label(frm,text="Conta").grid(row=3,column=0,sticky='e',padx=4,pady=4); self.e_account = tb.Entry(frm); self.e_account.grid(row=3,column=1)
        tb.Label(frm,text="Tags").grid(row=4,column=0,sticky='e',padx=4,pady=4); self.e_tags = tb.Entry(frm); self.e_tags.grid(row=4,column=1)
        tb.Button(frm,text="Salvar", bootstyle="primary", command=self.on_save).grid(row=5,column=0,columnspan=2,pady=8)
        if prefill:
            try:
                self.e_name.insert(0,prefill[1]); self.e_email.insert(0,prefill[2]); self.e_phone.insert(0,prefill[3]); self.e_account.insert(0,prefill[4]); self.e_tags.insert(0,prefill[5])
            except: pass
    def on_save(self):
        name=self.e_name.get().strip(); email=self.e_email.get().strip(); phone=self.e_phone.get().strip()
        account=self.e_account.get().strip(); tags=self.e_tags.get().strip()
        if not name: messagebox.showwarning("Validação","Nome obrigatório"); return
        self.result=(name,email,phone,account,tags); self.destroy()

class OpportunityDialog(tk.Toplevel):
    def __init__(self, parent, prefill=None):
        super().__init__(parent); self.title("Oportunidade"); self.result=None; self.build(prefill); self.grab_set()
    def build(self, prefill):
        frm = tb.Frame(self, padding=8); frm.pack(fill='both', expand=True)
        tb.Label(frm,text="Título").grid(row=0,column=0,sticky='e',padx=4,pady=4); self.e_title=tb.Entry(frm); self.e_title.grid(row=0,column=1)
        tb.Label(frm,text="Contact ID").grid(row=1,column=0,sticky='e',padx=4,pady=4); self.e_contact=tb.Entry(frm); self.e_contact.grid(row=1,column=1)
        tb.Button(frm,text="Lookup by name", bootstyle="info-outline", command=self.lookup_contact_by_name).grid(row=1,column=2,padx=4)
        tb.Label(frm,text="Valor").grid(row=2,column=0,sticky='e',padx=4,pady=4); self.e_amount=tb.Entry(frm); self.e_amount.grid(row=2,column=1)
        tb.Label(frm,text="Stage").grid(row=3,column=0,sticky='e',padx=4,pady=4); self.e_stage=tb.Combobox(frm, values=['novo','negociação','ganho','perdido']); self.e_stage.grid(row=3,column=1)
        tb.Label(frm,text="Close Date (YYYY-MM-DD)").grid(row=4,column=0,sticky='e',padx=4,pady=4); self.e_close=tb.Entry(frm); self.e_close.grid(row=4,column=1)
        tb.Button(frm,text="Salvar", bootstyle="primary", command=self.on_save).grid(row=5,column=0,columnspan=3,pady=8)
        if prefill:
            try:
                if isinstance(prefill[0], int) and len(prefill)>=6:
                    _, title, contact_id, amount, stage, close_date = prefill
                    self.e_title.insert(0, title or ''); self.e_contact.insert(0, contact_id or ''); self.e_amount.insert(0, amount or ''); self.e_stage.set(stage or 'novo'); self.e_close.insert(0, close_date or '')
                else:
                    self.e_title.insert(0, prefill[0] or '')
            except: pass
    def lookup_contact_by_name(self):
        name = simpledialog.askstring("Lookup","Nome do contato:")
        if not name: return
        row = db_fetch("SELECT id,name FROM contacts WHERE name LIKE ?", (f"%{name}%",))
        if not row: messagebox.showinfo("Resultado","Nenhum contato encontrado"); return
        self.e_contact.delete(0,'end'); self.e_contact.insert(0, str(row[0][0])); messagebox.showinfo("Encontrado", f"{row[0][1]} (id {row[0][0]})")
    def on_save(self):
        title = self.e_title.get().strip(); contact_id = self.e_contact.get().strip()
        try: contact_id = int(contact_id) if contact_id else None
        except: contact_id = None
        try: amount = float(self.e_amount.get() or 0)
        except: amount = 0.0
        stage = self.e_stage.get().strip() or 'novo'; close_date = self.e_close.get().strip() or ''
        if not title: messagebox.showwarning("Validação","Título obrigatório"); return
        self.result = (title, contact_id, amount, stage, close_date); self.destroy()

# -------------------------
# ServiceModule (Tickets + Chatbot + Interações)
# -------------------------
class ServiceModule(ModuleFrame):
    def __init__(self, master, app):
        super().__init__(master, app, title="Service Cloud — Atendimento")
        self.build_ui()
        self.refresh()

    def build_ui(self):
        pan = tb.Panedwindow(self, orient='horizontal')
        pan.pack(fill='both', expand=True)

        left = tb.Frame(pan, padding=6)
        right = tb.Frame(pan, padding=6)
        pan.add(left, weight=1); pan.add(right, weight=2)

        # Tickets Treeview
        tb.Label(left, text="Tickets", font=('Inter',12,'bold')).pack(anchor='w')
        cols = ('id','contact','subject','priority','status','created_at')
        self.ticket_tree = ttk.Treeview(left, columns=cols, show='headings', height=20)
        for c in cols: self.ticket_tree.heading(c, text=c)
        self.ticket_tree.pack(fill='both', expand=True, pady=(6,4))

        tbtnf = tb.Frame(left); tbtnf.pack(fill='x', pady=6)
        tb.Button(tbtnf, text="Novo Ticket", bootstyle="success", command=self.new_ticket).pack(side='left', padx=4)
        tb.Button(tbtnf, text="Atualizar Status", bootstyle="warning", command=self.update_ticket_status).pack(side='left', padx=4)
        tb.Button(tbtnf, text="Adicionar Interação", bootstyle="info", command=self.add_interaction).pack(side='left', padx=4)
        tb.Button(tbtnf, text="Export CSV", bootstyle="secondary", command=self.export_tickets_csv).pack(side='left', padx=4)

        # Right: Chatbot + Details
        tb.Label(right, text="Chatbot (FAQ)", font=('Inter',12,'bold')).pack(anchor='w')
        self.chat_text = tk.Text(right, height=9)
        self.chat_text.pack(fill='x', pady=4)
        entry_frame = tb.Frame(right); entry_frame.pack(fill='x')
        self.chat_entry = tb.Entry(entry_frame); self.chat_entry.pack(side='left', fill='x', expand=True)
        tb.Button(entry_frame, text="Enviar", bootstyle="primary", command=self.chat_send).pack(side='left', padx=4)
        tb.Button(right, text="Limpar Chat", bootstyle="outline-secondary", command=lambda: self.chat_text.delete('1.0','end')).pack(pady=6)

        self.detail_frame = tb.Labelframe(right, text="Detalhes / Interações", padding=6)
        self.detail_frame.pack(fill='both', expand=True)
        self.detail_text = tk.Text(self.detail_frame)
        self.detail_text.pack(fill='both', expand=True)
        tb.Button(self.detail_frame, text="Ver Detalhes Selecionado", bootstyle="secondary", command=self.show_ticket_details).pack(pady=6)

    def refresh(self):
        # refresh ticket list
        for i in list(self.ticket_tree.get_children()): self.ticket_tree.delete(i)
        rows = db_fetch("""SELECT t.id, COALESCE(c.name,'(sem contato)'), t.subject, t.priority, t.status, t.created_at
                           FROM tickets t LEFT JOIN contacts c ON t.contact_id=c.id
                           ORDER BY t.created_at DESC""")
        for r in rows: self.ticket_tree.insert('', 'end', values=r)

    # Ticket operations
    def new_ticket(self):
        cid = simpledialog.askinteger("Contact ID", "Informe contact_id (opcional):")
        subject = simpledialog.askstring("Assunto", "Assunto do ticket:")
        if not subject:
            return
        desc = simpledialog.askstring("Descrição", "Descrição (opcional):")
        priority = simpledialog.askstring("Prioridade", "low/medium/high", initialvalue='low')
        now = datetime.datetime.now().isoformat()
        db_execute("INSERT INTO tickets (contact_id,subject,description,priority,status,created_at) VALUES (?,?,?,?,?,?)",
                   (cid,subject,desc,priority,'open',now))
        messagebox.showinfo("Criado", "Ticket criado.")
        self.refresh()

    def update_ticket_status(self):
        sel = self.ticket_tree.selection()
        if not sel:
            messagebox.showwarning("Atenção", "Selecione um ticket")
            return
        tid = self.ticket_tree.item(sel[0])['values'][0]
        status = simpledialog.askstring("Status", "Novo status (open/working/closed):", initialvalue='working')
        if not status:
            return
        db_execute("UPDATE tickets SET status=? WHERE id=?", (status, tid))
        self.refresh()

    def add_interaction(self):
        sel = self.ticket_tree.selection()
        if not sel:
            messagebox.showwarning("Atenção", "Selecione um ticket")
            return
        tid = self.ticket_tree.item(sel[0])['values'][0]
        cid = None
        try:
            cid = db_fetch("SELECT contact_id FROM tickets WHERE id=?", (tid,))[0][0]
        except:
            pass
        note = simpledialog.askstring("Interação", "Nota / resposta para registrar:")
        if not note:
            return
        now = datetime.datetime.now().isoformat()
        db_execute("INSERT INTO interactions (ticket_id,contact_id,note,created_at) VALUES (?,?,?,?)", (tid,cid,note,now))
        messagebox.showinfo("Registrado", "Interação salva.")
        self.show_ticket_details()

    def show_ticket_details(self):
        sel = self.ticket_tree.selection()
        if not sel:
            messagebox.showwarning("Atenção", "Selecione um ticket")
            return
        tid = self.ticket_tree.item(sel[0])['values'][0]
        ticket = db_fetch("SELECT t.id, COALESCE(c.name,'(sem contato)'), t.subject, t.description, t.priority, t.status, t.created_at FROM tickets t LEFT JOIN contacts c ON t.contact_id=c.id WHERE t.id=?", (tid,))
        if not ticket:
            self.detail_text.delete('1.0','end')
            self.detail_text.insert('end', "Ticket não encontrado.")
            return
        t = ticket[0]
        self.detail_text.delete('1.0','end')
        self.detail_text.insert('end', f"Ticket ID: {t[0]}\nContact: {t[1]}\nSubject: {t[2]}\nDesc: {t[3]}\nPriority: {t[4]}\nStatus: {t[5]}\nCreated: {t[6]}\n\n")
        interactions = db_fetch("SELECT note,created_at FROM interactions WHERE ticket_id=? ORDER BY created_at", (tid,))
        if interactions:
            self.detail_text.insert('end', "Interações:\n")
            for it in interactions:
                self.detail_text.insert('end', f"- {it[1]}: {it[0]}\n")
        else:
            self.detail_text.insert('end', "Nenhuma interação registrada.\n")

    def export_tickets_csv(self):
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        rows = db_fetch("""SELECT t.id, COALESCE(c.name,'(sem contato)'), t.subject, t.description, t.priority, t.status, t.created_at
                           FROM tickets t LEFT JOIN contacts c ON t.contact_id=c.id""")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['id','contact','subject','description','priority','status','created_at'])
            w.writerows(rows)
        messagebox.showinfo("Exportado", f"Tickets exportados para {path}")

    # Chatbot (simples rules)
    def chat_send(self):
        q = self.chat_entry.get().strip()
        if not q:
            return
        self.chat_text.insert('end', f"Usuário: {q}\n")
        resp = self.simple_bot_response(q)
        self.chat_text.insert('end', f"Bot: {resp}\n\n")
        self.chat_entry.delete(0,'end')

    def simple_bot_response(self, q):
        ql = q.lower()
        if 'pedido' in ql or 'entrega' in ql:
            return "Verifique seu pedido na aba Commerce > Orders ou informe o número do pedido."
        if 'preço' in ql or 'valor' in ql:
            return "Os preços estão no catálogo de produtos (Commerce). Posso exportar a lista para CSV."
        if 'suporte' in ql or 'ajuda' in ql:
            return "Você pode abrir um ticket usando o botão 'Novo Ticket' nesta tela."
        if 'oi' in ql or 'olá' in ql:
            return "Olá! Como posso ajudar?"
        return "Obrigado pela mensagem — encaminhei para atendimento (simulado)."


# -------------------------
# MarketingModule (Campanhas + Segmentação)
# -------------------------
class MarketingModule(ModuleFrame):
    def __init__(self, master, app):
        super().__init__(master, app, title="Marketing Cloud")
        self.build_ui()
        self.refresh()

    def build_ui(self):
        top = tb.Frame(self); top.pack(fill='x', pady=6)
        tb.Button(top, text="Nova Campanha", bootstyle="success", command=self.new_campaign).pack(side='left', padx=6)
        tb.Button(top, text="Enviar (simulado)", bootstyle="primary", command=self.send_campaign).pack(side='left', padx=6)
        tb.Button(top, text="Exportar Segmento", bootstyle="secondary", command=self.export_segment_csv).pack(side='left', padx=6)
        tb.Button(top, text="Importar Contatos", bootstyle="info", command=self.import_contacts_csv).pack(side='left', padx=6)

        self.c_tree = ttk.Treeview(self, columns=('id','name','channel','audience','sent_count','created_at'), show='headings')
        for h in ('id','name','channel','audience','sent_count','created_at'):
            self.c_tree.heading(h, text=h)
        self.c_tree.pack(fill='both', expand=True, padx=6, pady=6)

    def refresh(self):
        for i in list(self.c_tree.get_children()): self.c_tree.delete(i)
        rows = db_fetch("SELECT id,name,channel,audience,sent_count,created_at FROM campaigns ORDER BY created_at DESC")
        for r in rows: self.c_tree.insert('', 'end', values=r)

    def new_campaign(self):
        name = simpledialog.askstring("Nome da Campanha", "Nome:")
        if not name:
            return
        channel = simpledialog.askstring("Canal", "email/mobile/ads", initialvalue='email')
        audience = simpledialog.askstring("Audience", "Segmento (ex: vip,lead,all)", initialvalue='all')
        now = datetime.datetime.now().isoformat()
        db_execute("INSERT INTO campaigns (name,channel,audience,sent_count,created_at) VALUES (?,?,?,?,?)", (name,channel,audience,0,now))
        messagebox.showinfo("Criado","Campanha criada.")
        self.refresh()

    def send_campaign(self):
        sel = self.c_tree.selection()
        if not sel:
            messagebox.showwarning("Selecione","Selecione uma campanha")
            return
        cid = self.c_tree.item(sel[0])['values'][0]
        audience = db_fetch("SELECT audience FROM campaigns WHERE id=?", (cid,))[0][0]
        if audience == 'all':
            cnt = db_fetch("SELECT COUNT(*) FROM contacts")[0][0]
        else:
            cnt = db_fetch("SELECT COUNT(*) FROM contacts WHERE tags LIKE ?", (f"%{audience}%",))[0][0]
        # update sent_count
        db_execute("UPDATE campaigns SET sent_count = sent_count + ? WHERE id=?", (cnt, cid))
        # Log: in a real system you'd send emails here; we simulate by adding an automation_log
        now = datetime.datetime.now().isoformat()
        db_execute("INSERT INTO automation_logs (automation_id, message, created_at) VALUES (?,?,?)", (None, f"Campanha {cid} enviada para {cnt} contatos (simulação)", now))
        messagebox.showinfo("Simulado", f"Campanha enviada para {cnt} contatos (simulação).")
        self.refresh()

    def export_segment_csv(self):
        audience = simpledialog.askstring("Segmento", "Segmento (tag) para exportar (ex: vip):", initialvalue='vip')
        if not audience:
            return
        rows = db_fetch("SELECT name,email,phone FROM contacts WHERE tags LIKE ?", (f"%{audience}%",))
        if not rows:
            messagebox.showinfo("Nenhum", "Nenhum contato encontrado para esse segmento.")
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['name','email','phone'])
            w.writerows(rows)
        messagebox.showinfo("Exportado", f"Segmento exportado para {path}")

    def import_contacts_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao ler CSV: {e}")
            return
        now = datetime.datetime.now().isoformat()
        rows = []
        for _,r in df.iterrows():
            rows.append((r.get('name',''), r.get('email',''), r.get('phone',''), r.get('account',''), r.get('tags',''), now))
        if rows:
            db_execute_many("INSERT INTO contacts (name,email,phone,account,tags,created_at) VALUES (?,?,?,?,?,?)", rows)
            messagebox.showinfo("Importado", f"{len(rows)} contatos importados.")
            # refresh sales module contacts view if available
            try:
                self.app.modules['Sales'].refresh()
            except Exception:
                pass
#===================== MÓDULO COMMERCE CLOUD =====================
class CommerceCloud(ttk.Frame):
    def __init__(self, parent, db_path="crm.db"):
        super().__init__(parent)
        self.db_path = db_path
        self.config(padding=10)
        self.create_ui()
        self.create_tables()
        self.load_products()
        self.load_orders()

    def create_ui(self):
        # Seções
        title = ttk.Label(self, text="🛍️ Commerce Cloud", font=("Arial", 16, "bold"))
        title.pack(pady=5)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # --- PRODUTOS ---
        self.tab_products = ttk.Frame(notebook)
        notebook.add(self.tab_products, text="Produtos")

        self.tree_products = ttk.Treeview(self.tab_products, columns=("nome", "preco", "estoque"), show="headings")
        for col in ("nome", "preco", "estoque"):
            self.tree_products.heading(col, text=col.capitalize())
            self.tree_products.column(col, width=150)
        self.tree_products.pack(fill="both", expand=True, padx=5, pady=5)

        form_frame = ttk.Frame(self.tab_products)
        form_frame.pack(pady=5)
        ttk.Label(form_frame, text="Nome:").grid(row=0, column=0)
        ttk.Label(form_frame, text="Preço:").grid(row=0, column=2)
        ttk.Label(form_frame, text="Estoque:").grid(row=0, column=4)
        self.ent_nome = ttk.Entry(form_frame, width=20)
        self.ent_preco = ttk.Entry(form_frame, width=10)
        self.ent_estoque = ttk.Entry(form_frame, width=10)
        self.ent_nome.grid(row=0, column=1, padx=3)
        self.ent_preco.grid(row=0, column=3, padx=3)
        self.ent_estoque.grid(row=0, column=5, padx=3)
        ttk.Button(form_frame, text="Adicionar Produto", command=self.add_product).grid(row=0, column=6, padx=5)

        # --- PEDIDOS ---
        self.tab_orders = ttk.Frame(notebook)
        notebook.add(self.tab_orders, text="Pedidos")

        self.tree_orders = ttk.Treeview(self.tab_orders, columns=("cliente", "produto", "quantidade", "data"), show="headings")
        for col in ("cliente", "produto", "quantidade", "data"):
            self.tree_orders.heading(col, text=col.capitalize())
            self.tree_orders.column(col, width=150)
        self.tree_orders.pack(fill="both", expand=True, padx=5, pady=5)

        order_form = ttk.Frame(self.tab_orders)
        order_form.pack(pady=5)
        ttk.Label(order_form, text="Cliente:").grid(row=0, column=0)
        ttk.Label(order_form, text="Produto:").grid(row=0, column=2)
        ttk.Label(order_form, text="Qtd:").grid(row=0, column=4)

        self.ent_cliente = ttk.Entry(order_form, width=20)
        self.ent_produto = ttk.Entry(order_form, width=20)
        self.ent_qtd = ttk.Entry(order_form, width=5)
        self.ent_cliente.grid(row=0, column=1, padx=3)
        self.ent_produto.grid(row=0, column=3, padx=3)
        self.ent_qtd.grid(row=0, column=5, padx=3)

        ttk.Button(order_form, text="Registrar Pedido", command=self.add_order).grid(row=0, column=6, padx=5)

    def connect(self):
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS produtos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT,
                preco REAL,
                estoque INTEGER
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS pedidos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cliente TEXT,
                produto TEXT,
                quantidade INTEGER,
                data TEXT
            )""")
            conn.commit()

    def load_products(self):
        for i in self.tree_products.get_children():
            self.tree_products.delete(i)
        with self.connect() as conn:
            for row in conn.execute("SELECT nome, preco, estoque FROM produtos"):
                self.tree_products.insert("", "end", values=row)

    def load_orders(self):
        for i in self.tree_orders.get_children():
            self.tree_orders.delete(i)
        with self.connect() as conn:
            for row in conn.execute("SELECT cliente, produto, quantidade, data FROM pedidos"):
                self.tree_orders.insert("", "end", values=row)

    def add_product(self):
        nome = self.ent_nome.get().strip()
        preco = self.ent_preco.get().strip()
        estoque = self.ent_estoque.get().strip()
        if not nome or not preco or not estoque:
            messagebox.showerror("Erro", "Preencha todos os campos do produto!")
            return
        try:
            preco = float(preco)
            estoque = int(estoque)
        except ValueError:
            messagebox.showerror("Erro", "Preço e estoque devem ser numéricos!")
            return
        with self.connect() as conn:
            conn.execute("INSERT INTO produtos (nome, preco, estoque) VALUES (?, ?, ?)", (nome, preco, estoque))
            conn.commit()
        self.load_products()
        self.ent_nome.delete(0, tk.END)
        self.ent_preco.delete(0, tk.END)
        self.ent_estoque.delete(0, tk.END)
        messagebox.showinfo("Sucesso", f"Produto '{nome}' adicionado com sucesso!")

    def add_order(self):
        cliente = self.ent_cliente.get().strip()
        produto = self.ent_produto.get().strip()
        qtd = self.ent_qtd.get().strip()
        if not cliente or not produto or not qtd:
            messagebox.showerror("Erro", "Preencha todos os campos do pedido!")
            return
        try:
            qtd = int(qtd)
        except ValueError:
            messagebox.showerror("Erro", "Quantidade inválida!")
            return
        data = datetime.date.today().strftime("%Y-%m-%d")
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO pedidos (cliente, produto, quantidade, data) VALUES (?, ?, ?, ?)",
                        (cliente, produto, qtd, data))
            cur.execute("UPDATE produtos SET estoque = estoque - ? WHERE nome = ?", (qtd, produto))
            conn.commit()
        self.load_orders()
        self.load_products()
        messagebox.showinfo("Pedido", f"Pedido de {qtd}x {produto} registrado para {cliente}.")

# ===================== MÓDULO EXPERIENCE CLOUD =====================
class ExperienceCloud(ttk.Frame):
    def __init__(self, parent, db_path="crm.db"):
        super().__init__(parent)
        self.db_path = db_path
        self.config(padding=10)
        self.create_ui()

    def create_ui(self):
        ttk.Label(self, text="🌐 Experience Cloud - Portal do Cliente", font=("Arial", 16, "bold")).pack(pady=10)

        self.entry_cliente = ttk.Entry(self, width=30)
        self.entry_cliente.pack()
        self.entry_cliente.insert(0, "Digite o nome do cliente...")
        ttk.Button(self, text="Ver Dados", command=self.load_data).pack(pady=5)

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Pedidos Recentes:").grid(row=0, column=0, padx=10)
        ttk.Label(frame, text="Tickets de Suporte:").grid(row=0, column=1, padx=10)

        self.tree_pedidos = ttk.Treeview(frame, columns=("produto", "qtd", "data"), show="headings", height=10)
        for col in ("produto", "qtd", "data"):
            self.tree_pedidos.heading(col, text=col.capitalize())
        self.tree_pedidos.grid(row=1, column=0, padx=10, pady=5)

        self.tree_tickets = ttk.Treeview(frame, columns=("assunto", "status", "prioridade"), show="headings", height=10)
        for col in ("assunto", "status", "prioridade"):
            self.tree_tickets.heading(col, text=col.capitalize())
        self.tree_tickets.grid(row=1, column=1, padx=10, pady=5)

    def connect(self):
        return sqlite3.connect(self.db_path)

    def load_data(self):
        cliente = self.entry_cliente.get().strip()
        if not cliente:
            messagebox.showerror("Erro", "Informe o nome do cliente!")
            return
        for i in self.tree_pedidos.get_children():
            self.tree_pedidos.delete(i)
        for i in self.tree_tickets.get_children():
            self.tree_tickets.delete(i)
        with self.connect() as conn:
            for row in conn.execute("SELECT produto, quantidade, data FROM pedidos WHERE cliente = ?", (cliente,)):
                self.tree_pedidos.insert("", "end", values=row)
            for row in conn.execute("SELECT assunto, status, prioridade FROM tickets WHERE cliente = ?", (cliente,)):
                self.tree_tickets.insert("", "end", values=row)
        messagebox.showinfo("Sucesso", f"Dados carregados para {cliente}")

# -------------------------
# AnalyticsModule (embeds matplotlib, adaptive to table names)
# -------------------------
class AnalyticsModule(tb.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        tb.Label(self, text="Analytics / Dashboards", font=('Inter',16,'bold')).pack(anchor='w', pady=(0,8))
        self.build_ui()

    def build_ui(self):
        toolbar = tb.Frame(self); toolbar.pack(fill='x', pady=6)
        tb.Button(toolbar, text="Atualizar", bootstyle="primary", command=self.draw_charts).pack(side='left', padx=6)
        tb.Button(toolbar, text="Exportar Vendas CSV", bootstyle="secondary", command=self.export_sales_csv).pack(side='left', padx=6)
        self.plot_frame = tb.Frame(self); self.plot_frame.pack(fill='both', expand=True, padx=6, pady=6)

    def _read_orders_df(self):
        """Tenta ler tanto a tabela 'orders' quanto 'pedidos' (compatibilidade)."""
        try:
            conn = db_connect()
            # try orders
            try:
                df = pd.read_sql_query("SELECT order_date AS date, total FROM orders", conn)
                conn.close()
                return df
            except Exception:
                # try pedidos/pedidos table from Parte C
                try:
                    df = pd.read_sql_query("SELECT data AS date, quantidade * 1.0 AS total FROM pedidos", conn)
                    conn.close()
                    # note: pedidos doesn't have total; using quantidade as proxy (user can extend)
                    return df
                except Exception:
                    conn.close()
                    return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def draw_charts(self):
        for w in self.plot_frame.winfo_children(): w.destroy()
        df = self._read_orders_df()
        if df.empty:
            tb.Label(self.plot_frame, text="Sem dados de vendas para exibir").pack()
            return
        # normalize date and aggregate
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            pass
        try:
            summary = df.groupby(df['date'].dt.date)['total'].sum()
        except Exception:
            # fallback simple plot
            summary = df['total']
        # build plot
        fig = Figure(figsize=(6,3))
        ax = fig.add_subplot(111)
        summary.plot(kind='bar', ax=ax)
        ax.set_title('Vendas por período')
        ax.set_ylabel('Total')
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

        # KPIs box
        kpi_frame = tb.Frame(self.plot_frame, padding=12)
        kpi_frame.pack(side='left', fill='both', expand=True)
        total_revenue = float(df['total'].sum())
        # try counts in different possible table names
        try:
            total_orders = db_fetch("SELECT COUNT(*) FROM orders")[0][0]
        except Exception:
            try:
                total_orders = db_fetch("SELECT COUNT(*) FROM pedidos")[0][0]
            except Exception:
                total_orders = 0
        try:
            total_customers = db_fetch("SELECT COUNT(*) FROM contacts")[0][0]
        except Exception:
            total_customers = 0
        tb.Label(kpi_frame, text=f"Receita total: R$ {total_revenue:.2f}", font=('Inter',12,'bold')).pack(anchor='w', pady=6)
        tb.Label(kpi_frame, text=f"Total pedidos: {total_orders}").pack(anchor='w', pady=3)
        tb.Label(kpi_frame, text=f"Total clientes: {total_customers}").pack(anchor='w', pady=3)

    def export_sales_csv(self):
        # try multiple table names to export best available info
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        rows = []
        try:
            rows = db_fetch("""SELECT o.order_date, c.name, p.name, o.qty, o.total, o.status
                               FROM orders o LEFT JOIN contacts c ON o.contact_id=c.id LEFT JOIN products p ON o.product_id=p.id""")
        except Exception:
            try:
                rows = db_fetch("SELECT data, cliente, produto, quantidade, '' as total, '' as status FROM pedidos")
            except Exception:
                rows = []
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if rows:
                w.writerow(['date','contact','product','qty','total','status'])
                w.writerows(rows)
        messagebox.showinfo("Exportado", f"Vendas exportadas para {path}")

# -------------------------
# Integration & Automation Module
# -------------------------
class IntegrationModule(tb.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=10)
        self.app = app
        self.running = False
        self.thread = None
        tb.Label(self, text="Integração & Automação", font=('Inter',16,'bold')).pack(anchor='w', pady=(0,8))
        self.build_ui()

    def build_ui(self):
        tb.Button(self, text="Exportar Leads CSV", bootstyle="secondary", command=self.exportar_leads).pack(anchor='w', pady=4)
        tb.Button(self, text="Importar Leads CSV", bootstyle="info", command=self.importar_leads).pack(anchor='w', pady=4)
        self.toggle_btn = tb.Button(self, text="Iniciar Automações", bootstyle="primary", command=self.toggle_automations)
        self.toggle_btn.pack(anchor='w', pady=4)
        tb.Button(self, text="Ver logs de automação (DB)", bootstyle="secondary", command=self.show_automation_logs).pack(anchor='w', pady=4)
        self.log = tk.Text(self, height=12)
        self.log.pack(fill='both', expand=True, pady=6)

    def exportar_leads(self):
        rows = db_fetch("SELECT id,name,email,phone,status,created_at FROM leads")
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['id','name','email','phone','status','created_at']); w.writerows(rows)
        messagebox.showinfo("Exportado", f"Leads exportados para {path}")

    def importar_leads(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao ler CSV: {e}")
            return
        now = datetime.datetime.now().isoformat(); rows=[]
        for _,r in df.iterrows():
            rows.append((r.get('name',''), r.get('email',''), r.get('phone',''), r.get('source',''), r.get('status',''), now))
        if rows:
            db_execute_many("INSERT INTO leads (name,email,phone,source,status,created_at) VALUES (?,?,?,?,?,?)", rows)
            messagebox.showinfo("Importado", f"{len(rows)} leads importados.")

    def toggle_automations(self):
        if not self.running:
            self.running = True
            self.toggle_btn.config(text="Parar Automações", bootstyle="danger")
            self.thread = threading.Thread(target=self._automation_loop, daemon=True)
            self.thread.start()
            messagebox.showinfo("Automação", "Automações iniciadas.")
        else:
            self.running = False
            self.toggle_btn.config(text="Iniciar Automações", bootstyle="primary")
            messagebox.showinfo("Automação", "Automações paradas.")

    def _automation_loop(self):
        while self.running:
            try:
                now = datetime.datetime.now()
                # Exemplo: lembrete para leads 'new' > 7 dias
                rows = db_fetch("SELECT id,name,email,created_at FROM leads WHERE status='new'")
                reminders = []
                for r in rows:
                    try:
                        created = datetime.datetime.fromisoformat(r[3]); days = (now - created).days
                        if days >= 7:
                            reminders.append((r[0], r[1], r[2], days))
                    except:
                        pass
                if reminders:
                    for rem in reminders:
                        msg = f"[{datetime.datetime.now().isoformat()}] Reminder: Lead {rem[1]} ({rem[2]}) criado há {rem[3]} dias\n"
                        self.log.insert('end', msg); self.log.see('end')
                        db_execute("INSERT INTO automation_logs (automation_id, message, created_at) VALUES (?,?,?)", (None, msg, datetime.datetime.now().isoformat()))
                else:
                    msg = f"[{datetime.datetime.now().isoformat()}] Lembrete: nenhum lead antigo encontrado.\n"
                    self.log.insert('end', msg); self.log.see('end')
                time.sleep(AUTOMATION_INTERVAL_SECONDS)
            except Exception as e:
                self.log.insert('end', f"Erro na automação: {e}\n"); self.log.see('end'); time.sleep(5)

    def show_automation_logs(self):
        rows = db_fetch("SELECT created_at, message FROM automation_logs ORDER BY created_at DESC LIMIT 200")
        dlg = tk.Toplevel(self)
        dlg.title("Automation Logs")
        txt = tk.Text(dlg, width=100, height=30)
        txt.pack(fill='both', expand=True)
        for r in rows:
            txt.insert('end', f"{r[0]} - {r[1]}\n")

# -------------------------
# App Main (connects modules and sidebar)
# -------------------------
class CRMApp(tb.Window):
    def __init__(self):
        super().__init__(themename="flatly")
        self.title("Mini CRM — Desktop (v2.0)")
        self.geometry("1200x700")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # 🎨 aplica o estilo personalizado
        apply_style(self)
                
        # layout: sidebar left + container right
        self.sidebar = tb.Frame(self, width=240, bootstyle="secondary")
        self.sidebar.pack(side='left', fill='y')
        self.container = tb.Frame(self)
        self.container.pack(side='left', fill='both', expand=True)

        self.modules = {}
        self._build_sidebar()
        self._build_modules()
        self.show_module('Sales')

        # # Criar frame principal
        # self.main_frame = tb.Frame(self)
        # self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # # # Criar botões
        # # self.btn_salvar = styled_button(self.main_frame, "💾 Salvar Nota", self.sales.save_note)
        # # self.btn_salvar.pack(side="left", padx=5)

        # # self.btn_historico = styled_button(self.main_frame, "🕓 Ver Histórico", self.ver_historico)
        # # self.btn_historico.pack(side="left", padx=5)

        # # Criar tabela
        # tree_frame, self.tree = styled_treeview(
        #     self.main_frame,
        #     ["ID", "Nome", "Email", "Status"],
        #     ["ID", "Nome", "Email", "Status"]
        # )
        # tree_frame.pack(fill="both", expand=True, pady=10)

    def _build_sidebar(self):
        tb.Label(self.sidebar, text=" CRM ", style="Sidebar.TLabel").pack(pady=(12,6))
        # menu buttons (key used by show_module)
        menu = [
            ('Sales','Sales','success'),
            ('Service','Service','info'),
            ('Marketing','Marketing','primary'),
            ('Commerce','Commerce','warning'),
            ('Experience','Experience','secondary'),
            ('Analytics','Analytics','dark'),
            ('Automation','Automation','danger'),
        ]
        for label, key, style in menu:
            btn = tb.Button(self.sidebar, text=label, bootstyle=f"{style}-outline", width=22, command=lambda k=key: self.show_module(k))
            btn.pack(pady=6, padx=12)
        tb.Separator(self.sidebar).pack(fill='x', pady=10, padx=10)
        tb.Button(self.sidebar, text="Backup DB", bootstyle="outline-primary", width=22, command=self.backup_db).pack(pady=6, padx=12)
        tb.Button(self.sidebar, text="Sair", bootstyle="outline-secondary", width=22, command=self.on_close).pack(pady=6, padx=12)

    def _build_modules(self):
        # instantiate modules that should exist from previous parts
        # Sales/Service/Marketing are expected from Part A/B
        # Commerce/Experience may be either classes from Part C or the earlier module names;
        # attempt to wrap/instantiate them safely.
        container = self.container

        # Sales, Service, Marketing should already be defined (from Part A/B)
        try:
            self.modules['Sales'] = SalesModule(container, self)
        except Exception as e:
            print("SalesModule not available:", e)
        try:
            self.modules['Service'] = ServiceModule(container, self)
        except Exception as e:
            print("ServiceModule not available:", e)
        try:
            self.modules['Marketing'] = MarketingModule(container, self)
        except Exception as e:
            print("MarketingModule not available:", e)

        # Commerce: prefer to wrap CommerceCloud (Part C) if exists, else CommerceModule from earlier versions
        # We'll create a simple wrapper that embeds either class.
        commerce_frame = tb.Frame(container, padding=10)
        try:
            if 'CommerceModule' in globals():
                # tenta com (frame, app), senão tenta só (frame)
                try:
                    inst = CommerceModule(commerce_frame, self)
                except TypeError:
                    inst = CommerceModule(commerce_frame)
                inst.pack(fill='both', expand=True)

            elif 'CommerceCloud' in globals():
                # aceita apenas 1 argumento
                cc = CommerceCloud(commerce_frame)
                cc.pack(fill='both', expand=True)
            else:
                tb.Label(commerce_frame, text="Commerce not implemented").pack()
        except Exception as e:
            tb.Label(commerce_frame, text=f"Erro ao carregar Commerce: {e}").pack()
        self.modules['Commerce'] = commerce_frame

        # Experience wrapper: prefer ExperienceModule then ExperienceCloud
        experience_frame = tb.Frame(container, padding=10)
        try:
            if 'ExperienceModule' in globals():
                try:
                    em = ExperienceModule(experience_frame, self)
                except TypeError:
                    em = ExperienceModule(experience_frame)
                em.pack(fill='both', expand=True)

            elif 'ExperienceCloud' in globals():
                ec = ExperienceCloud(experience_frame)
                ec.pack(fill='both', expand=True)
            else:
                tb.Label(experience_frame, text="Experience not implemented").pack()
        except Exception as e:
            tb.Label(experience_frame, text=f"Erro ao carregar Experience: {e}").pack()
        self.modules['Experience'] = experience_frame

        # Analytics & Automation from this part
        self.modules['Analytics'] = AnalyticsModule(container, self)
        self.modules['Automation'] = IntegrationModule(container, self)

        # place all modules in container (only one visible)
        for mname, frame in list(self.modules.items()):
            # frame might be a tb.Frame subclass (we ensure it's a widget)
            try:
                frame.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
                frame.lower()
            except Exception as e:
                print("error placing module", mname, e)

    def show_module(self, name):
        # lower all
        for v in self.modules.values():
            try: v.lower()
            except: pass
        m = self.modules.get(name)
        if not m:
            messagebox.showerror("Erro", f"Módulo {name} não encontrado")
            return
        try:
            m.lift()
            # if module has refresh method call it
            if hasattr(m, 'refresh'):
                try: m.refresh()
                except Exception as e: print("refresh error:", e)
            # if wrapped widget (e.g., frame containing CommerceCloud) contains a child with refresh => call
            for child in m.winfo_children():
                if hasattr(child, 'refresh'):
                    try: child.refresh()
                    except: pass
        except Exception as e:
            messagebox.showerror("Erro ao exibir módulo", str(e))

    def backup_db(self):
        import shutil
        ts = datetime.date.today().isoformat()
        dst = f"crm_backup_{ts}.db"
        try:
            shutil.copy(DB_PATH, dst)
            messagebox.showinfo("Backup", f"Backup salvo: {dst}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar backup: {e}")


    def on_close(self):
        # stop automations gracefully
        try:
            auto = self.modules.get('Automation')
            if auto and hasattr(auto, 'running') and getattr(auto, 'running'):
                auto.running = False
                time.sleep(0.2)
        except:
            pass
        if messagebox.askokcancel("Sair", "Deseja sair?"):
            self.destroy()

        # ============================================================
# PARTE E - COMMERCE CLOUD (CATÁLOGO E PEDIDOS)
# ============================================================

class CommerceModule(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Label(self, text="🛒 Commerce Cloud - Catálogo e Pedidos", font=("Segoe UI", 14, "bold")).pack(pady=10)

        # Notebook (abas)
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.frame_produtos = ttk.Frame(notebook)
        self.frame_pedidos = ttk.Frame(notebook)

        notebook.add(self.frame_produtos, text="Produtos")
        notebook.add(self.frame_pedidos, text="Pedidos")

        self.create_produtos_tab()
        self.create_pedidos_tab()

    def create_table(self):
        try:
            db_execute("""
            CREATE TABLE IF NOT EXISTS produtos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                preco REAL NOT NULL,
                estoque INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)
            messagebox.showinfo("Sucesso", "Tabela 'produtos' criada com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao criar tabela: {e}")


    # ---------------------------------------------------------
    # ABA PRODUTOS
    # ---------------------------------------------------------
    def create_produtos_tab(self):
        frm_top = ttk.Frame(self.frame_produtos)
        frm_top.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm_top, text="Nome:").grid(row=0, column=0, sticky="w")
        self.prod_nome = ttk.Entry(frm_top, width=30)
        self.prod_nome.grid(row=0, column=1, padx=5)

        ttk.Label(frm_top, text="Preço:").grid(row=0, column=2, sticky="w")
        self.prod_preco = ttk.Entry(frm_top, width=10)
        self.prod_preco.grid(row=0, column=3, padx=5)

        ttk.Label(frm_top, text="Estoque:").grid(row=0, column=4, sticky="w")
        self.prod_estoque = ttk.Entry(frm_top, width=10)
        self.prod_estoque.grid(row=0, column=5, padx=5)

        ttk.Button(frm_top, text="Adicionar / Atualizar", command=self.add_produto).grid(row=0, column=6, padx=5)
        ttk.Button(frm_top, text="Excluir", command=self.del_produto).grid(row=0, column=7, padx=5)
        ttk.Button(frm_top, text="Criar Tabela Produtos", command=self.create_table).grid(row=0, column=8, padx=5)
        
        # Treeview de produtos
        cols = ("id", "nome", "preco", "estoque")
        self.tree_prod = ttk.Treeview(self.frame_produtos, columns=cols, show="headings")
        for c in cols:
            self.tree_prod.heading(c, text=c.capitalize())
            self.tree_prod.column(c, width=150 if c != "id" else 50)
        self.tree_prod.pack(fill="both", expand=True, padx=10, pady=5)
        self.tree_prod.bind("<<TreeviewSelect>>", self.load_produto_selected)

        self.load_produtos()

    def load_produtos(self):
        for i in self.tree_prod.get_children():
            self.tree_prod.delete(i)
        produtos = db_fetch("SELECT id, nome, preco, estoque FROM produtos ORDER BY nome")
        for p in produtos:
            self.tree_prod.insert("", "end", values=p)

    def add_produto(self):
        nome = self.prod_nome.get().strip()
        preco = self.prod_preco.get().strip()
        estoque = self.prod_estoque.get().strip()
        if not nome or not preco:
            messagebox.showwarning("Aviso", "Preencha nome e preço.")
            return
        try:
            preco = float(preco)
            estoque = int(estoque) if estoque else 0
        except ValueError:
            messagebox.showerror("Erro", "Preço e estoque devem ser numéricos.")
            return
        sel = self.tree_prod.selection()
        if sel:
            pid = self.tree_prod.item(sel[0])["values"][0]
            db_execute("UPDATE produtos SET nome=?, preco=?, estoque=? WHERE id=?", (nome, preco, estoque, pid))
        else:
            db_execute("INSERT INTO produtos (nome, preco, estoque) VALUES (?, ?, ?)", (nome, preco, estoque))
        self.load_produtos()
        self.prod_nome.delete(0, tk.END)
        self.prod_preco.delete(0, tk.END)
        self.prod_estoque.delete(0, tk.END)

    def del_produto(self):
        multi_delete(self.tree, "produtos", self.refresh, label="produto")

    def load_produto_selected(self, event):
        sel = self.tree_prod.selection()
        if sel:
            vals = self.tree_prod.item(sel[0])["values"]
            self.prod_nome.delete(0, tk.END)
            self.prod_nome.insert(0, vals[1])
            self.prod_preco.delete(0, tk.END)
            self.prod_preco.insert(0, vals[2])
            self.prod_estoque.delete(0, tk.END)
            self.prod_estoque.insert(0, vals[3])

    # ---------------------------------------------------------
    # ABA PEDIDOS
    # ---------------------------------------------------------
    def create_pedidos_tab(self):
        frm_top = ttk.Frame(self.frame_pedidos)
        frm_top.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm_top, text="Cliente:").grid(row=0, column=0)
        self.combo_cliente = ttk.Combobox(frm_top, width=25)
        self.combo_cliente.grid(row=0, column=1, padx=5)

        ttk.Label(frm_top, text="Produto:").grid(row=0, column=2)
        self.combo_produto = ttk.Combobox(frm_top, width=25)
        self.combo_produto.grid(row=0, column=3, padx=5)

        ttk.Label(frm_top, text="Qtd:").grid(row=0, column=4)
        self.entry_qtd = ttk.Entry(frm_top, width=5)
        self.entry_qtd.grid(row=0, column=5, padx=5)

        ttk.Button(frm_top, text="Registrar Pedido", command=self.add_pedido).grid(row=0, column=6, padx=5)

        # Treeview de pedidos
        cols = ("id", "cliente", "produto", "quantidade", "total")
        self.tree_pedidos = ttk.Treeview(self.frame_pedidos, columns=cols, show="headings")
        for c in cols:
            self.tree_pedidos.heading(c, text=c.capitalize())
            self.tree_pedidos.column(c, width=150 if c != "id" else 50)
        self.tree_pedidos.pack(fill="both", expand=True, padx=10, pady=5)

        self.load_clientes_produtos()
        self.load_pedidos()

    def load_clientes_produtos(self):
        clientes = db_fetch("SELECT nome FROM clientes")
        produtos = db_fetch("SELECT nome FROM produtos")
        self.combo_cliente["values"] = [c[0] for c in clientes]
        self.combo_produto["values"] = [p[0] for p in produtos]

    def add_pedido(self):
        cliente = self.combo_cliente.get().strip()
        produto = self.combo_produto.get().strip()
        qtd = self.entry_qtd.get().strip()
        if not cliente or not produto or not qtd:
            messagebox.showwarning("Aviso", "Preencha todos os campos.")
            return
        try:
            qtd = int(qtd)
        except ValueError:
            messagebox.showerror("Erro", "Quantidade inválida.")
            return
        p = db_fetch("SELECT id, preco, estoque FROM produtos WHERE nome=?", (produto,))
        if not p:
            messagebox.showerror("Erro", "Produto não encontrado.")
            return
        pid, preco, estoque = p[0]
        if qtd > estoque:
            messagebox.showerror("Erro", "Estoque insuficiente.")
            return
        total = preco * qtd
        db_execute("INSERT INTO pedidos (cliente, produto, quantidade, total) VALUES (?, ?, ?, ?)",
                   (cliente, produto, qtd, total))
        db_execute("UPDATE produtos SET estoque=? WHERE id=?", (estoque - qtd, pid))
        self.load_pedidos()
        self.load_produtos()
        messagebox.showinfo("Sucesso", f"Pedido registrado. Total: R${total:.2f}")

    def load_pedidos(self):
        for i in self.tree_pedidos.get_children():
            self.tree_pedidos.delete(i)
        pedidos = db_fetch("SELECT id, cliente, produto, quantidade, total FROM pedidos ORDER BY id DESC")
        for p in pedidos:
            self.tree_pedidos.insert("", "end", values=p)

        
        # ============================================================
# PARTE F - EXPERIENCE CLOUD (PORTAL DO CLIENTE)
# ============================================================

class ExperienceModule(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Label(self, text="🌐 Experience Cloud - Portal do Cliente", font=("Segoe UI", 14, "bold")).pack(pady=10)

        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm_top, text="Selecione o Cliente:").grid(row=0, column=0, sticky="w")
        self.combo_cliente = ttk.Combobox(frm_top, width=30)
        self.combo_cliente.grid(row=0, column=1, padx=5)
        ttk.Button(frm_top, text="Carregar", command=self.load_cliente_data).grid(row=0, column=2, padx=5)

        # Frame principal dividido em abas
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_perfil = ttk.Frame(notebook)
        self.tab_pedidos = ttk.Frame(notebook)
        self.tab_tickets = ttk.Frame(notebook)

        notebook.add(self.tab_perfil, text="Perfil")
        notebook.add(self.tab_pedidos, text="Pedidos")
        notebook.add(self.tab_tickets, text="Tickets")

        # --- PERFIL ---
        self.lbl_info = tk.Text(self.tab_perfil, height=10, width=70)
        self.lbl_info.pack(padx=10, pady=10, fill="both", expand=True)
        self.lbl_info.insert("end", "Selecione um cliente para visualizar o perfil.\n")
        self.lbl_info.config(state="disabled")

        # --- PEDIDOS ---
        cols = ("id", "produto", "quantidade", "total")
        self.tree_pedidos = ttk.Treeview(self.tab_pedidos, columns=cols, show="headings")
        for c in cols:
            self.tree_pedidos.heading(c, text=c.capitalize())
            self.tree_pedidos.column(c, width=150 if c != "id" else 50)
        self.tree_pedidos.pack(fill="both", expand=True, padx=10, pady=10)

        # --- TICKETS ---
        cols2 = ("id", "titulo", "status", "prioridade")
        self.tree_tickets = ttk.Treeview(self.tab_tickets, columns=cols2, show="headings")
        for c in cols2:
            self.tree_tickets.heading(c, text=c.capitalize())
            self.tree_tickets.column(c, width=150 if c != "id" else 50)
        self.tree_tickets.pack(fill="both", expand=True, padx=10, pady=10)

        self.load_clientes()

    # ---------------------------------------------------------
    # FUNÇÕES
    # ---------------------------------------------------------
    def load_clientes(self):
        clientes = db_fetch("SELECT nome FROM clientes ORDER BY nome")
        self.combo_cliente["values"] = [c[0] for c in clientes]

    def load_cliente_data(self):
        nome = self.combo_cliente.get().strip()
        if not nome:
            messagebox.showwarning("Aviso", "Selecione um cliente.")
            return

        # PERFIL
        cliente = db_fetch("SELECT id, nome, email, telefone, status FROM clientes WHERE nome=?", (nome,))
        if not cliente:
            messagebox.showerror("Erro", "Cliente não encontrado.")
            return
        cid, nome, email, telefone, status = cliente[0]
        info = f"""
📋 Nome: {nome}
📧 Email: {email}
📞 Telefone: {telefone}
🎯 Status: {status}
"""
        self.lbl_info.config(state="normal")
        self.lbl_info.delete("1.0", tk.END)
        self.lbl_info.insert("end", info)
        self.lbl_info.config(state="disabled")

        # PEDIDOS
        for i in self.tree_pedidos.get_children():
            self.tree_pedidos.delete(i)
        pedidos = db_fetch("SELECT id, produto, quantidade, total FROM pedidos WHERE cliente=?", (nome,))
        for p in pedidos:
            self.tree_pedidos.insert("", "end", values=p)

        # TICKETS
        for i in self.tree_tickets.get_children():
            self.tree_tickets.delete(i)
        tickets = db_fetch("SELECT id, titulo, status, prioridade FROM tickets WHERE cliente=?", (nome,))
        for t in tickets:
            self.tree_tickets.insert("", "end", values=t)


# -------------------------
# Entrypoint (final)
# -------------------------
def main():
    # ensure DB and demo data exist (Part A's functions)
    try:
        init_db(DB_PATH)
        seed_demo_data(DB_PATH)
    except Exception as e:
        print("Erro init DB:", e)
    app = CRMApp()
    app.mainloop()

if __name__ == "__main__":
    main()
    

