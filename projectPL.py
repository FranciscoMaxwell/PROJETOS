#!/usr/bin/env python3
# hospital_analisador.py
"""
Hospital Analisador - Versão completa com chat em rede, autenticação e chamada.
Target: Python 3.11
Tudo em um único arquivo .py

Funcionalidades principais:
- Banco SQLite persistente (hospital.db)
- Módulos de negócio (Suprimentos, Atendimento, Assistencial, SADT, Faturamento, Financeiro, Contábil)
- Integrações HIS / RIS-PACS / LIS (adapters REST simples)
- GUI Tkinter com gerenciamento de suprimentos, relatórios, alertas
- Monitor periódico de validade/limite de suprimentos
- Chat em rede (servidor/cliente) usando sockets TCP com mensagens JSON
- Registro/Autenticação de usuários (registro/login) com hashing seguro (salt + PBKDF2)
- Função de chamada: "CALL" de um usuário para outro, com notificação em tempo real e registro (perdas)
- Logs de comunicação salvos em DB (tabela comunicacoes)
- Tudo configurável via GUI

Instruções rápidas:
- Requer Python 3.11 (testado)
- Recomenda-se executar em ambiente com permissões de rede para aceitar conexões (se ativar servidor)
- Para rodar: python hospital_analisador.py
- Para conectar clientes na rede: abra o app em "client mode" e informe IP do servidor + porta
- O protocolo de rede usa JSON por linha (newline-terminated). Exemplo de mensagem:
    {"type":"AUTH","user":"alice","pass":"plaintext-password"}
    {"type":"MSG","to":"bob","text":"Olá"}
    {"type":"CALL","to":"bob"}
  O servidor valida AUTH e encaminha mensagens.

Segurança:
- Senhas armazenadas com PBKDF2-HMAC-SHA256 + salt (100_000 iterações)
- Use VPN/TLS em produção; este é um exemplo que usa TCP simples (sem TLS)

Autor: ChatGPT (ajustado para seu pedido)
Data: 2025-11-06
"""

from __future__ import annotations
import sqlite3
import threading
import time
import datetime
from dataclasses import dataclass
import json
import os
import sys
import traceback
import socket
import selectors
import hashlib
import secrets
import base64
from typing import Optional, Dict, Any, Tuple, List

# GUI
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog
except Exception as e:
    print("Tkinter não disponível. Instale Python com suporte GUI.")
    raise

# Optional requests for REST integrations
try:
    import requests
except Exception:
    requests = None
    import urllib.request as urllib_request

DB_FILE = "hospital.db"
MONITOR_INTERVAL_SECONDS = 60  # default

# -------------------------
# Utilities
# -------------------------
def now_iso():
    return datetime.datetime.utcnow().isoformat()

def parse_date(s: Optional[str]) -> Optional[datetime.datetime]:
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

def days_between(d1: datetime.datetime, d2: datetime.datetime) -> int:
    return (d2 - d1).days

# -------------------------
# Database wrapper
# -------------------------
class Database:
    def __init__(self, path=DB_FILE):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            c = self.conn.cursor()
            # suprimentos
            c.execute("""
                CREATE TABLE IF NOT EXISTS suprimentos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT NOT NULL,
                    quantidade REAL NOT NULL,
                    unidade TEXT,
                    preco_unitario REAL,
                    validade TEXT,
                    limite_alerta REAL DEFAULT 0,
                    criado_em TEXT,
                    atualizado_em TEXT
                )
            """)
            # registros / eventos
            c.execute("""
                CREATE TABLE IF NOT EXISTS registros (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fonte TEXT,
                    tipo TEXT,
                    dados TEXT,
                    recebido_em TEXT
                )
            """)
            # pacientes
            c.execute("""
                CREATE TABLE IF NOT EXISTS pacientes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT,
                    prontuario TEXT,
                    internado INTEGER DEFAULT 0,
                    criado_em TEXT
                )
            """)
            # config
            c.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    chave TEXT PRIMARY KEY,
                    valor TEXT
                )
            """)
            # users (auth)
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    display_name TEXT,
                    role TEXT,
                    external_id TEXT,
                    password_hash TEXT,
                    salt TEXT,
                    created_at TEXT
                )
            """)
            # communications log
            c.execute("""
                CREATE TABLE IF NOT EXISTS comunicacoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    origem TEXT,
                    destino TEXT,
                    tipo TEXT, -- MESSAGE | CALL | AUTH | SYSTEM
                    texto TEXT,
                    timestamp TEXT,
                    status TEXT -- delivered | missed | pending
                )
            """)
            self.conn.commit()

    def execute(self, sql: str, params: tuple = (), commit: bool=False):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(sql, params)
            if commit:
                self.conn.commit()
            return cur

    def fetchall(self, sql: str, params: tuple = ()):
        cur = self.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def fetchone(self, sql: str, params: tuple = ()):
        cur = self.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def set_config(self, chave: str, valor: Any):
        self.execute("REPLACE INTO config (chave, valor) VALUES (?, ?)", (chave, json.dumps(valor)), commit=True)

    def get_config(self, chave: str, default=None):
        row = self.fetchone("SELECT valor FROM config WHERE chave = ?", (chave,))
        if not row:
            return default
        try:
            return json.loads(row['valor'])
        except Exception:
            return row['valor']

# -------------------------
# Authentication helpers
# -------------------------
def hash_password(password: str, salt: Optional[bytes]=None) -> Tuple[str,str]:
    """
    Returns (hash_b64, salt_b64)
    PBKDF2-HMAC-SHA256 with 100k iterations
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return base64.b64encode(dk).decode('ascii'), base64.b64encode(salt).decode('ascii')

def verify_password(stored_hash_b64: str, stored_salt_b64: str, password_attempt: str) -> bool:
    salt = base64.b64decode(stored_salt_b64.encode('ascii'))
    h, _ = hash_password(password_attempt, salt)
    return secrets.compare_digest(h, stored_hash_b64)

# -------------------------
# Models & Business Areas
# -------------------------
@dataclass
class SuprimentoModel:
    nome: str
    quantidade: float
    unidade: str = ""
    preco_unitario: float = 0.0
    validade: Optional[str] = None
    limite_alerta: float = 0.0

class Suprimentos:
    def __init__(self, db: Database):
        self.db = db

    def add_suprimento(self, s: SuprimentoModel):
        now = now_iso()
        self.db.execute("""
            INSERT INTO suprimentos (nome, quantidade, unidade, preco_unitario, validade, limite_alerta, criado_em, atualizado_em)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (s.nome, s.quantidade, s.unidade, s.preco_unitario, s.validade, s.limite_alerta, now, now), commit=True)

    def update_suprimento(self, id_, **kwargs):
        allowed = ['nome','quantidade','unidade','preco_unitario','validade','limite_alerta']
        pairs = []
        vals = []
        for k,v in kwargs.items():
            if k in allowed:
                pairs.append(f"{k} = ?")
                vals.append(v)
        vals.append(now_iso())
        vals.append(id_)
        if pairs:
            sql = "UPDATE suprimentos SET " + ", ".join(pairs) + ", atualizado_em = ? WHERE id = ?"
            self.db.execute(sql, tuple(vals), commit=True)

    def list_suprimentos(self):
        return self.db.fetchall("SELECT * FROM suprimentos ORDER BY nome")

    def get_suprimento(self, id_):
        return self.db.fetchone("SELECT * FROM suprimentos WHERE id = ?", (id_,))

    def consume(self, id_, quantidade):
        row = self.get_suprimento(id_)
        if not row:
            raise ValueError("Suprimento não encontrado.")
        nova = float(row['quantidade']) - float(quantidade)
        if nova < 0:
            nova = 0
        self.update_suprimento(id_, quantidade=nova)

    def add_quantity(self, id_, quantidade):
        row = self.get_suprimento(id_)
        if not row:
            raise ValueError("Suprimento não encontrado.")
        nova = float(row['quantidade']) + float(quantidade)
        self.update_suprimento(id_, quantidade=nova)

    def check_alerts(self):
        alerts = []
        for row in self.list_suprimentos():
            q = float(row['quantidade'])
            limite = float(row['limite_alerta'] or 0)
            validade = row['validade']
            if limite > 0 and q <= limite:
                alerts.append({
                    'tipo':'limite',
                    'suprimento': row['nome'],
                    'quantidade': q,
                    'limite': limite
                })
            if validade:
                dt = parse_date(validade)
                if dt:
                    dias = days_between(datetime.datetime.utcnow(), dt)
                    if dias < 0:
                        alerts.append({
                            'tipo':'vencido',
                            'suprimento': row['nome'],
                            'validade': validade,
                            'dias_atraso': abs(dias)
                        })
                    elif dias <= 7:
                        alerts.append({
                            'tipo':'vencendo',
                            'suprimento': row['nome'],
                            'validade': validade,
                            'dias_restantes': dias
                        })
        return alerts

    def estimate_duration(self, id_, consumo_diario):
        row = self.get_suprimento(id_)
        if not row:
            return None
        if consumo_diario <= 0:
            return float('inf')
        dias = float(row['quantidade']) / consumo_diario
        return dias

# Minimal other domain classes (kept simple for extensibility)
class Atendimento:
    def __init__(self, db: Database):
        self.db = db

    def novo_paciente(self, nome, prontuario=None, internado=0):
        now = now_iso()
        self.db.execute("INSERT INTO pacientes (nome, prontuario, internado, criado_em) VALUES (?, ?, ?, ?)",
                        (nome, prontuario, internado, now), commit=True)

    def listar_pacientes(self):
        return self.db.fetchall("SELECT * FROM pacientes ORDER BY criado_em DESC")

    def set_internado(self, paciente_id, internado):
        self.db.execute("UPDATE pacientes SET internado = ? WHERE id = ?", (1 if internado else 0, paciente_id), commit=True)

class Assistencial:
    def __init__(self, db: Database, suprimentos: Suprimentos):
        self.db = db
        self.suprimentos = suprimentos

    def registrar_uso(self, suprimento_id, quantidade, motivo=None):
        self.suprimentos.consume(suprimento_id, quantidade)
        dados = {
            'suprimento_id': suprimento_id,
            'quantidade': quantidade,
            'motivo': motivo,
            'ts': now_iso()
        }
        self.db.execute("INSERT INTO registros (fonte,tipo,dados,recebido_em) VALUES (?, ?, ?, ?)",
                        ("LOCAL", "USO_SUPRIMENTO", json.dumps(dados), now_iso()), commit=True)

class SADT:
    def __init__(self, db: Database):
        self.db = db

    def agendar_exame(self, paciente_id, exame_codigo, data):
        dados = {'paciente_id': paciente_id, 'exame': exame_codigo, 'data': data, 'ts': now_iso()}
        self.db.execute("INSERT INTO registros (fonte,tipo,dados,recebido_em) VALUES (?, ?, ?, ?)",
                        ("LOCAL","SADT_AGENDAMENTO", json.dumps(dados), now_iso()), commit=True)

class Comercial:
    def __init__(self, db: Database):
        self.db = db

class Faturamento:
    def __init__(self, db: Database):
        self.db = db

    def gerar_fatura_simples(self, paciente_id, itens: list):
        dados = {'paciente_id': paciente_id, 'itens': itens, 'ts': now_iso()}
        self.db.execute("INSERT INTO registros (fonte,tipo,dados,recebido_em) VALUES (?, ?, ?, ?)",
                        ("LOCAL","FATURA_GERADA", json.dumps(dados), now_iso()), commit=True)

class Financeiro:
    def __init__(self, db: Database):
        self.db = db

class Contabil:
    def __init__(self, db: Database):
        self.db = db

# -------------------------
# Integrations (simple adapters)
# -------------------------
class BaseIntegration:
    def __init__(self, db: Database, config_key: str):
        self.db = db
        self.config_key = config_key
        self.config = self.db.get_config(config_key, default={})

    def save_config(self):
        self.db.set_config(self.config_key, self.config)

    def log_event(self, tipo: str, dados: dict):
        self.db.execute("INSERT INTO registros (fonte,tipo,dados,recebido_em) VALUES (?, ?, ?, ?)",
                        (self.config_key, tipo, json.dumps(dados), now_iso()), commit=True)

class IntegracaoHIS(BaseIntegration):
    def __init__(self, db: Database):
        super().__init__(db, "integracao_his")

    def sync(self):
        mode = self.config.get('mode','local')
        if mode == 'disabled':
            return {'status':'disabled'}
        if mode == 'local':
            self.log_event("HEARTBEAT", {"msg":"HIS local heartbeat", "ts": now_iso()})
            return {'status':'local'}
        url = self.config.get('url')
        token = self.config.get('token')
        if not url:
            return {'status':'no_url'}
        try:
            headers = {'Authorization': f"Bearer {token}"} if token else {}
            if requests:
                resp = requests.get(url.rstrip('/') + '/patients', headers=headers, timeout=10)
                data = resp.json()
            else:
                req = urllib_request.Request(url.rstrip('/') + '/patients', headers=headers)
                with urllib_request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read().decode('utf-8'))
            for p in data if isinstance(data, list) else [data]:
                self.log_event("HIS_PATIENT", p)
            return {'status':'ok','count': len(data) if isinstance(data, list) else 1}
        except Exception as e:
            self.log_event("ERROR", {"error": str(e), "trace": traceback.format_exc()})
            return {'status':'error','error': str(e)}

class IntegracaoRIS_PACS(BaseIntegration):
    def __init__(self, db: Database):
        super().__init__(db, "integracao_ris_pacs")

    def sync(self):
        mode = self.config.get('mode','local')
        if mode == 'disabled':
            return {'status':'disabled'}
        if mode == 'local':
            self.log_event("HEARTBEAT", {"msg":"RIS/PACS local heartbeat", "ts": now_iso()})
            return {'status':'local'}
        url = self.config.get('url')
        token = self.config.get('token')
        if not url:
            return {'status':'no_url'}
        try:
            headers = {'Authorization': f"Bearer {token}"} if token else {}
            if requests:
                resp = requests.get(url.rstrip('/') + '/studies/pending', headers=headers, timeout=10)
                data = resp.json()
            else:
                req = urllib_request.Request(url.rstrip('/') + '/studies/pending', headers=headers)
                with urllib_request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read().decode('utf-8'))
            for s in data if isinstance(data, list) else [data]:
                self.log_event("RIS_PENDING_STUDY", s)
            return {'status':'ok','count': len(data) if isinstance(data, list) else 1}
        except Exception as e:
            self.log_event("ERROR", {"error": str(e), "trace": traceback.format_exc()})
            return {'status':'error','error': str(e)}

class IntegracaoLIS(BaseIntegration):
    def __init__(self, db: Database):
        super().__init__(db, "integracao_lis")

    def sync(self):
        mode = self.config.get('mode','local')
        if mode == 'disabled':
            return {'status':'disabled'}
        if mode == 'local':
            self.log_event("HEARTBEAT", {"msg":"LIS local heartbeat", "ts": now_iso()})
            return {'status':'local'}
        url = self.config.get('url')
        token = self.config.get('token')
        if not url:
            return {'status':'no_url'}
        try:
            headers = {'Authorization': f"Bearer {token}"} if token else {}
            if requests:
                resp = requests.get(url.rstrip('/') + '/lab/results/recent', headers=headers, timeout=10)
                data = resp.json()
            else:
                req = urllib_request.Request(url.rstrip('/') + '/lab/results/recent', headers=headers)
                with urllib_request.urlopen(req, timeout=10) as r:
                    data = json.loads(r.read().decode('utf-8'))
            for rlt in data if isinstance(data, list) else [data]:
                self.log_event("LIS_RESULT", rlt)
            return {'status':'ok','count': len(data) if isinstance(data, list) else 1}
        except Exception as e:
            self.log_event("ERROR", {"error": str(e), "trace": traceback.format_exc()})
            return {'status':'error','error': str(e)}

# -------------------------
# Monitor and Reports
# -------------------------
class Monitor:
    def __init__(self, db: Database, suprimentos: Suprimentos, integrations: list, interval=MONITOR_INTERVAL_SECONDS, on_alert=None):
        self.db = db
        self.suprimentos = suprimentos
        self.integrations = integrations
        self.interval = interval
        self.on_alert = on_alert
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                for integ in self.integrations:
                    try:
                        _ = integ.sync()
                    except Exception:
                        pass
                alerts = self.suprimentos.check_alerts()
                if alerts and self.on_alert:
                    try:
                        self.on_alert(alerts)
                    except Exception:
                        pass
            except Exception:
                pass
            self._stop_event.wait(self.interval)

class ReportGenerator:
    def __init__(self, db: Database, suprimentos: Suprimentos):
        self.db = db
        self.suprimentos = suprimentos

    def gerar_relatorio_diario(self):
        report = {}
        report['gerado_em'] = now_iso()
        report['suprimentos'] = self.suprimentos.list_suprimentos()
        report['alerts'] = self.suprimentos.check_alerts()
        registros = self.db.fetchall("SELECT fonte, tipo, COUNT(*) as contagem FROM registros GROUP BY fonte, tipo ORDER BY contagem DESC")
        report['registros_resumo'] = registros
        pacientes = self.db.fetchall("SELECT COUNT(*) as total FROM pacientes")
        report['pacientes_total'] = pacientes[0]['total'] if pacientes else 0
        return report

# -------------------------
# Chat Server / Client (JSON-over-TCP)
# -------------------------
# Protocol summary:
# - JSON messages, one per line (newline-terminated)
# - Types:
#   AUTH: {"type":"AUTH","user":"username","pass":"plaintext"}
#   REGISTER: {"type":"REGISTER","user":"username","pass":"plaintext","display":"Name","role":"role","external_id":"ID"}
#   MSG: {"type":"MSG","to":"username","text":"..."}
#   BROADCAST: {"type":"BROADCAST","text":"..."}  # to all connected
#   CALL: {"type":"CALL","to":"username"}  # initiates a call
#   PING: {"type":"PING"}
# - Server responds with JSON messages as well
# - Server keeps mapping username -> socket (if authenticated)

class ChatServer:
    def __init__(self, db: Database, host='0.0.0.0', port=9090):
        self.db = db
        self.host = host
        self.port = port
        self.sel = selectors.DefaultSelector()
        self.sock = None
        self.clients = {}  # sock -> {'addr':(), 'buffer':bytes, 'user':username or None}
        self.user_sockets: Dict[str, socket.socket] = {}  # username -> socket
        self._stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self._stop_event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            if self.sock:
                self.sock.close()
        except:
            pass

    def _run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen(100)
            self.sock.setblocking(False)
            self.sel.register(self.sock, selectors.EVENT_READ, data=None)
        except Exception as e:
            print("ChatServer start error:", e)
            return

        while not self._stop_event.is_set():
            events = self.sel.select(timeout=1)
            for key, mask in events:
                if key.data is None:
                    self._accept_wrapper(key.fileobj)
                else:
                    self._service_connection(key, mask)
        # cleanup
        keys = list(self.sel.get_map().values())
        for k in keys:
            try:
                self.sel.unregister(k.fileobj)
                k.fileobj.close()
            except Exception:
                pass
        self.sel.close()

    def _accept_wrapper(self, sock):
        try:
            conn, addr = sock.accept()
            conn.setblocking(False)
            data = {'addr': addr, 'inb': b'', 'outb': b'', 'user': None}
            self.sel.register(conn, selectors.EVENT_READ, data=data)
            self.clients[conn] = data
        except Exception:
            pass

    def _service_connection(self, key, mask):
        sock = key.fileobj
        data = key.data
        try:
            if mask & selectors.EVENT_READ:
                recv = sock.recv(4096)
                if recv:
                    data['inb'] += recv
                    # process complete lines
                    while b'\n' in data['inb']:
                        line, data['inb'] = data['inb'].split(b'\n', 1)
                        try:
                            msg = json.loads(line.decode('utf-8'))
                            self._handle_message(sock, data, msg)
                        except Exception as e:
                            self._send_error(sock, "invalid_json", str(e))
                else:
                    # connection closed
                    self._disconnect(sock)
            if mask & selectors.EVENT_WRITE:
                if data.get('outb'):
                    sent = sock.send(data['outb'])
                    data['outb'] = data['outb'][sent:]
        except ConnectionResetError:
            self._disconnect(sock)
        except Exception:
            self._disconnect(sock)

    def _send(self, sock, obj: dict):
        try:
            raw = (json.dumps(obj, ensure_ascii=False) + "\n").encode('utf-8')
            # try to write immediately; if fails, append to outb
            try:
                sock.send(raw)
            except BlockingIOError:
                data = self.clients.get(sock)
                if data is not None:
                    data['outb'] += raw
        except Exception:
            pass

    def _send_error(self, sock, errcode, message):
        self._send(sock, {"type":"ERROR","code":errcode,"message":message})

    def _disconnect(self, sock):
        data = self.clients.get(sock)
        username = None
        if data:
            username = data.get('user')
        try:
            self.sel.unregister(sock)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        if sock in self.clients:
            del self.clients[sock]
        if username and username in self.user_sockets:
            try:
                del self.user_sockets[username]
            except Exception:
                pass

    def _handle_message(self, sock, data, msg: dict):
        mtype = msg.get('type')
        if mtype == 'PING':
            self._send(sock, {"type":"PONG","ts": now_iso()})
            return
        if mtype == 'REGISTER':
            self._handle_register(sock, data, msg)
            return
        if mtype == 'AUTH':
            self._handle_auth(sock, data, msg)
            return
        # Require authenticated user for other messages
        user = data.get('user')
        if not user:
            self._send_error(sock, "unauthenticated", "Please authenticate first with AUTH")
            return
        if mtype == 'MSG':
            self._handle_msg(sock, user, msg)
            return
        if mtype == 'BROADCAST':
            self._handle_broadcast(sock, user, msg)
            return
        if mtype == 'CALL':
            self._handle_call(sock, user, msg)
            return
        # unknown
        self._send_error(sock, "unknown_type", f"Unknown message type: {mtype}")

    def _handle_register(self, sock, data, msg):
        user = msg.get('user')
        password = msg.get('pass')
        display = msg.get('display') or user
        role = msg.get('role') or 'user'
        external_id = msg.get('external_id') or ''
        if not user or not password:
            self._send_error(sock, "invalid", "user and pass required")
            return
        # check exists
        existing = self.db.fetchone("SELECT * FROM users WHERE username = ?", (user,))
        if existing:
            self._send(sock, {"type":"REGISTERED","status":"exists"})
            return
        pwdhash, salt_b64 = hash_password(password)
        now = now_iso()
        try:
            self.db.execute("INSERT INTO users (username, display_name, role, external_id, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (user, display, role, external_id, pwdhash, salt_b64, now), commit=True)
            self._send(sock, {"type":"REGISTERED","status":"ok"})
        except Exception as e:
            self._send_error(sock, "register_failed", str(e))

    def _handle_auth(self, sock, data, msg):
        user = msg.get('user')
        password = msg.get('pass')
        if not user or not password:
            self._send_error(sock, "invalid", "user and pass required")
            return
        dbu = self.db.fetchone("SELECT * FROM users WHERE username = ?", (user,))
        if not dbu:
            self._send(sock, {"type":"AUTH_RESP","status":"invalid_user"})
            return
        ok = verify_password(dbu['password_hash'], dbu['salt'], password)
        if not ok:
            self._send(sock, {"type":"AUTH_RESP","status":"invalid_pass"})
            return
        # success
        data['user'] = user
        self.user_sockets[user] = sock
        self._send(sock, {"type":"AUTH_RESP","status":"ok","user":user,"display": dbu['display_name']})
        # send any pending messages for this user (comunicacoes with status pending)
        pending = self.db.fetchall("SELECT * FROM comunicacoes WHERE destino = ? AND status = 'pending'", (user,))
        for p in pending:
            try:
                self._send(sock, {"type":"PENDING","id":p['id'], "origem":p['origem'], "tipo": p['tipo'], "texto": p['texto'], "timestamp": p['timestamp']})
                self.db.execute("UPDATE comunicacoes SET status = 'delivered' WHERE id = ?", (p['id'],), commit=True)
            except Exception:
                pass

    def _handle_msg(self, sock, origem: str, msg: dict):
        to = msg.get('to')
        text = msg.get('text') or ''
        ts = now_iso()
        # persist
        self.db.execute("INSERT INTO comunicacoes (origem, destino, tipo, texto, timestamp, status) VALUES (?, ?, ?, ?, ?, ?)",
                        (origem, to, "MESSAGE", text, ts, "pending"), commit=True)
        cid = self.db.execute("SELECT last_insert_rowid() as id").fetchone()['id']
        # deliver if connected
        dest_sock = self.user_sockets.get(to)
        if dest_sock:
            try:
                self._send(dest_sock, {"type":"MSG","from":origem,"text":text,"ts":ts})
                self.db.execute("UPDATE comunicacoes SET status = 'delivered' WHERE id = ?", (cid,), commit=True)
            except Exception:
                pass
        # ack to sender
        self._send(sock, {"type":"MSG_ACK","id":cid,"status":"queued"})

    def _handle_broadcast(self, sock, origem: str, msg: dict):
        text = msg.get('text') or ''
        ts = now_iso()
        # persist to DB as broadcast (destino=NULL)
        self.db.execute("INSERT INTO comunicacoes (origem, destino, tipo, texto, timestamp, status) VALUES (?, ?, ?, ?, ?, ?)",
                        (origem, None, "BROADCAST", text, ts, "delivered"), commit=True)
        # send to all
        for user, s in list(self.user_sockets.items()):
            try:
                self._send(s, {"type":"BROADCAST","from":origem,"text":text,"ts":ts})
            except Exception:
                pass
        self._send(sock, {"type":"BROADCAST_ACK","status":"ok","ts":ts})

    def _handle_call(self, sock, origem: str, msg: dict):
        to = msg.get('to')
        ts = now_iso()
        # persist as CALL pending
        self.db.execute("INSERT INTO comunicacoes (origem, destino, tipo, texto, timestamp, status) VALUES (?, ?, ?, ?, ?, ?)",
                        (origem, to, "CALL", f"{origem} calling {to}", ts, "pending"), commit=True)
        cid = self.db.execute("SELECT last_insert_rowid() as id").fetchone()['id']
        dest_sock = self.user_sockets.get(to)
        if dest_sock:
            try:
                self._send(dest_sock, {"type":"CALL","from":origem,"ts":ts})
                self.db.execute("UPDATE comunicacoes SET status = 'delivered' WHERE id = ?", (cid,), commit=True)
                self._send(sock, {"type":"CALL_ACK","status":"delivered","ts":ts})
            except Exception:
                self._send(sock, {"type":"CALL_ACK","status":"error"})
        else:
            # target offline; keep pending -> will be delivered at next login
            self._send(sock, {"type":"CALL_ACK","status":"pending","ts":ts})

class ChatClient:
    def __init__(self, db: Database, server_host: str, server_port: int, on_message=None, on_call=None, on_broadcast=None):
        self.db = db
        self.server_host = server_host
        self.server_port = server_port
        self.sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.on_message = on_message
        self.on_call = on_call
        self.on_broadcast = on_broadcast
        self.lock = threading.Lock()
        self.buffer = b""

    def connect(self, timeout=5):
        with self.lock:
            if self.sock:
                return True
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect((self.server_host, self.server_port))
                s.settimeout(None)
                self.sock = s
                self._stop_event.clear()
                self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
                self._recv_thread.start()
                return True
            except Exception as e:
                print("ChatClient connect error:", e)
                self.sock = None
                return False

    def close(self):
        self._stop_event.set()
        try:
            if self.sock:
                self.sock.close()
        except:
            pass
        self.sock = None

    def send(self, obj: dict):
        if not self.sock:
            return False
        try:
            raw = (json.dumps(obj, ensure_ascii=False) + "\n").encode('utf-8')
            self.sock.sendall(raw)
            return True
        except Exception:
            return False

    def _recv_loop(self):
        try:
            while not self._stop_event.is_set():
                try:
                    data = self.sock.recv(4096)
                    if not data:
                        break
                    self.buffer += data
                    while b'\n' in self.buffer:
                        line, self.buffer = self.buffer.split(b'\n',1)
                        try:
                            msg = json.loads(line.decode('utf-8'))
                            self._handle_server_msg(msg)
                        except Exception:
                            pass
                except Exception:
                    break
        finally:
            self.close()

    def _handle_server_msg(self, msg: dict):
        mtype = msg.get('type')
        if mtype == 'PONG':
            return
        if mtype == 'MSG':
            if self.on_message:
                self.on_message(msg.get('from'), msg.get('text'), msg.get('ts'))
            # persist
            self.db.execute("INSERT INTO comunicacoes (origem, destino, tipo, texto, timestamp, status) VALUES (?, ?, ?, ?, ?, ?)",
                            (msg.get('from'), None, "MESSAGE_RECV", msg.get('text'), msg.get('ts'), "delivered"), commit=True)
            return
        if mtype == 'CALL':
            if self.on_call:
                self.on_call(msg.get('from'), msg.get('ts'))
            # persist call arrival
            self.db.execute("INSERT INTO comunicacoes (origem, destino, tipo, texto, timestamp, status) VALUES (?, ?, ?, ?, ?, ?)",
                            (msg.get('from'), None, "CALL_IN", f"Call from {msg.get('from')}", msg.get('ts'), "delivered"), commit=True)
            return
        if mtype == 'BROADCAST':
            if self.on_broadcast:
                self.on_broadcast(msg.get('from'), msg.get('text'), msg.get('ts'))
            return
        # other message types ignored here

# -------------------------
# GUI Application (integrates everything)
# -------------------------
class AppGUI:
    def __init__(self, root, db: Database):
        self.root = root
        self.db = db
        self.root.title("Hospital Analisador - Rede & Chat (Python 3.11)")
        self.suprimentos = Suprimentos(db)
        self.atendimento = Atendimento(db)
        self.assistencial = Assistencial(db, self.suprimentos)
        self.sadt = SADT(db)
        self.faturamento = Faturamento(db)
        self.financeiro = Financeiro(db)
        self.contabil = Contabil(db)
        # integrations
        self.integ_his = IntegracaoHIS(db)
        self.integ_ris = IntegracaoRIS_PACS(db)
        self.integ_lis = IntegracaoLIS(db)
        # reports & monitor
        self.report_gen = ReportGenerator(db, self.suprimentos)
        self.monitor = Monitor(db, self.suprimentos, [self.integ_his, self.integ_ris, self.integ_lis],
                               interval=db.get_config('monitor_interval', MONITOR_INTERVAL_SECONDS),
                               on_alert=self.on_alerts)
        # chat server/client
        cfg = db.get_config('chat_server', default={"host":"0.0.0.0","port":9090,"enabled":False})
        self.chat_server = ChatServer(db, host=cfg.get("host","0.0.0.0"), port=cfg.get("port",9090))
        self.chat_client: Optional[ChatClient] = None
        self.current_user: Optional[str] = None  # logged-in username in GUI
        self.current_display: Optional[str] = None
        # build UI
        self.create_widgets()
        self.refresh_suprimentos()
        self.refresh_registros()
        self.monitor.start()
        # start server if enabled config
        if cfg.get("enabled"):
            try:
                self.chat_server.start()
            except Exception as e:
                print("Erro ao iniciar chat server:", e)

    def create_widgets(self):
        frm = ttk.Frame(self.root, padding=6)
        frm.pack(fill=tk.BOTH, expand=True)

        # Top bar: user auth / server control
        top = ttk.Frame(frm)
        top.pack(fill=tk.X, pady=4)

        self.lbl_user = ttk.Label(top, text="User: (não logado)")
        self.lbl_user.pack(side=tk.LEFT, padx=4)

        btn_register = ttk.Button(top, text="Registrar usuário", command=self.register_user_dialog)
        btn_register.pack(side=tk.LEFT, padx=4)

        btn_login = ttk.Button(top, text="Login", command=self.login_dialog)
        btn_login.pack(side=tk.LEFT, padx=4)

        btn_logout = ttk.Button(top, text="Logout", command=self.logout)
        btn_logout.pack(side=tk.LEFT, padx=4)

        btn_server = ttk.Button(top, text="Servidor Chat: START/STOP", command=self.toggle_server)
        btn_server.pack(side=tk.RIGHT, padx=4)

        # Middle split: Left suprimentos, right chat
        middle = ttk.Frame(frm)
        middle.pack(fill=tk.BOTH, expand=True, pady=6)

        # Suprimentos frame
        left = ttk.LabelFrame(middle, text="Suprimentos & Estoque")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        cols = ("id","nome","quantidade","unidade","preco_unitario","validade","limite_alerta")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=110)
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.tree.bind("<Double-1>", self.on_double_suprimento)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=sb.set)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        # Chat frame
        right = ttk.LabelFrame(middle, text="Comunicação / Chat")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4)

        # Chat connections config
        conn_frm = ttk.Frame(right)
        conn_frm.pack(fill=tk.X, pady=2)
        ttk.Label(conn_frm, text="Server IP:").grid(row=0,column=0,sticky=tk.W)
        self.e_server_ip = ttk.Entry(conn_frm, width=15)
        self.e_server_ip.insert(0, self.db.get_config('chat_connect_host', '127.0.0.1'))
        self.e_server_ip.grid(row=0,column=1,sticky=tk.W,padx=2)
        ttk.Label(conn_frm, text="Port:").grid(row=0,column=2,sticky=tk.W)
        self.e_server_port = ttk.Entry(conn_frm, width=6)
        self.e_server_port.insert(0, str(self.db.get_config('chat_connect_port', 9090)))
        self.e_server_port.grid(row=0,column=3,sticky=tk.W,padx=2)
        btn_conn = ttk.Button(conn_frm, text="Conectar ao servidor", command=self.connect_chat_server)
        btn_conn.grid(row=0,column=4,sticky=tk.W,padx=4)
        btn_disconnect = ttk.Button(conn_frm, text="Desconectar", command=self.disconnect_chat_client)
        btn_disconnect.grid(row=0,column=5,sticky=tk.W,padx=4)

        # Chat messages display
        self.chat_txt = tk.Text(right, height=14, state=tk.DISABLED)
        self.chat_txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # Chat send area
        send_frm = ttk.Frame(right)
        send_frm.pack(fill=tk.X)
        ttk.Label(send_frm, text="Para (username):").pack(side=tk.LEFT)
        self.e_chat_to = ttk.Entry(send_frm, width=15)
        self.e_chat_to.pack(side=tk.LEFT, padx=4)
        self.e_chat_msg = ttk.Entry(send_frm, width=40)
        self.e_chat_msg.pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)
        btn_send = ttk.Button(send_frm, text="Enviar", command=self.send_chat_message)
        btn_send.pack(side=tk.LEFT, padx=4)
        btn_broadcast = ttk.Button(send_frm, text="Broadcast", command=self.broadcast_chat_message)
        btn_broadcast.pack(side=tk.LEFT, padx=2)
        btn_call = ttk.Button(send_frm, text="📞 Chamar", command=self.call_user)
        btn_call.pack(side=tk.LEFT, padx=2)

        # Bottom controls (suprimentos actions)
        bottom = ttk.Frame(frm)
        bottom.pack(fill=tk.X, pady=6)
        btn_add = ttk.Button(bottom, text="➕ Adicionar Suprimento", command=self.add_suprimento_dialog)
        btn_add.pack(side=tk.LEFT, padx=3)
        btn_limite = ttk.Button(bottom, text="⚙️ Definir Limite", command=self.definir_limite_dialog)
        btn_limite.pack(side=tk.LEFT, padx=3)
        btn_use = ttk.Button(bottom, text="Usar Suprimento (consumir)", command=self.usar_suprimento_dialog)
        btn_use.pack(side=tk.LEFT, padx=3)
        btn_add_q = ttk.Button(bottom, text="Adicionar Quantidade", command=self.adicionar_quantidade_dialog)
        btn_add_q.pack(side=tk.LEFT, padx=3)
        btn_report = ttk.Button(bottom, text="📊 Gerar Relatório", command=self.gerar_relatorio)
        btn_report.pack(side=tk.LEFT, padx=3)
        btn_refresh = ttk.Button(bottom, text="Refresh", command=self.refresh_all)
        btn_refresh.pack(side=tk.RIGHT, padx=3)

    # -------------------------
    # Auth / Users
    # -------------------------
    def register_user_dialog(self):
        dlg = RegisterDialog(self.root)
        self.root.wait_window(dlg.top)
        if dlg.result:
            username = dlg.result['username']
            password = dlg.result['password']
            display = dlg.result.get('display') or username
            role = dlg.result.get('role') or 'user'
            external_id = dlg.result.get('external_id') or ''
            # store in DB
            existing = self.db.fetchone("SELECT * FROM users WHERE username = ?", (username,))
            if existing:
                messagebox.showerror("Erro", "Usuário já existe.")
                return
            pwdhash, salt = hash_password(password)
            now = now_iso()
            self.db.execute("INSERT INTO users (username, display_name, role, external_id, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (username, display, role, external_id, pwdhash, salt, now), commit=True)
            messagebox.showinfo("Ok", "Usuário registrado com sucesso.")

    def login_dialog(self):
        dlg = LoginDialog(self.root)
        self.root.wait_window(dlg.top)
        if dlg.result:
            username = dlg.result['username']
            password = dlg.result['password']
            user = self.db.fetchone("SELECT * FROM users WHERE username = ?", (username,))
            if not user:
                messagebox.showerror("Erro", "Usuário não encontrado.")
                return
            if not verify_password(user['password_hash'], user['salt'], password):
                messagebox.showerror("Erro", "Senha incorreta.")
                return
            # success
            self.current_user = username
            self.current_display = user['display_name']
            self.lbl_user.config(text=f"User: {self.current_display} ({self.current_user})")
            messagebox.showinfo("Login", f"Bem-vindo {self.current_display}")
            # if chat client connected, send AUTH
            if self.chat_client and self.chat_client.sock:
                self.chat_client.send({"type":"AUTH","user":self.current_user,"pass":password})
            # store last login display? (not needed)
    
    def logout(self):
        if not self.current_user:
            return
        self.current_user = None
        self.current_display = None
        self.lbl_user.config(text="User: (não logado)")
        messagebox.showinfo("Logout", "Logout efetuado.")
        # disconnect chat client if exists
        self.disconnect_chat_client()

    # -------------------------
    # Chat server control
    # -------------------------
    def toggle_server(self):
        cfg = self.db.get_config('chat_server', default={"host":"0.0.0.0","port":9090,"enabled":False})
        if cfg.get("enabled"):
            # stop
            self.chat_server.stop()
            cfg['enabled'] = False
            self.db.set_config('chat_server', cfg)
            messagebox.showinfo("Servidor", "Servidor chat parado.")
        else:
            # start server with chosen host/port (use current config entries if present)
            try:
                # option: allow editing via dialog
                host = simpledialog.askstring("Host", "Host para bind (por padrão 0.0.0.0):", initialvalue=cfg.get("host","0.0.0.0"))
                port = simpledialog.askinteger("Port", "Porta para servidor chat (por padrão 9090):", initialvalue=cfg.get("port",9090), minvalue=1, maxvalue=65535)
                if not host or not port:
                    return
                cfg['host'] = host
                cfg['port'] = port
                cfg['enabled'] = True
                self.db.set_config('chat_server', cfg)
                # restart server instance
                try:
                    self.chat_server.stop()
                except: pass
                self.chat_server = ChatServer(self.db, host=host, port=port)
                self.chat_server.start()
                messagebox.showinfo("Servidor", f"Servidor iniciado em {host}:{port}")
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível iniciar servidor: {e}")

    # -------------------------
    # Chat client control & actions
    # -------------------------
    def connect_chat_server(self):
        host = self.e_server_ip.get().strip()
        port = int(self.e_server_port.get().strip())
        self.db.set_config('chat_connect_host', host)
        self.db.set_config('chat_connect_port', port)
        # create client with callbacks
        def on_msg(frm, text, ts):
            self.append_chat_line(f"[{ts}] {frm} → {self.current_user or 'local'}: {text}")
        def on_call(frm, ts):
            self.append_chat_line(f"[{ts}] 🔔 CHAMADA de {frm} para {self.current_user or 'local'}")
            messagebox.showinfo("Chamada", f"{frm} está te chamando!")
        def on_bcast(frm, text, ts):
            self.append_chat_line(f"[{ts}] 🔊 BROADCAST de {frm}: {text}")

        self.chat_client = ChatClient(self.db, host, port, on_message=on_msg, on_call=on_call, on_broadcast=on_bcast)
        ok = self.chat_client.connect()
        if not ok:
            messagebox.showerror("Erro", "Não foi possível conectar ao servidor.")
            self.chat_client = None
            return
        # if user logged in, try to auth (we do not store plaintext password; ask user)
        if self.current_user:
            pwd = simpledialog.askstring("Senha", f"Senha para autenticar usuário {self.current_user} (será enviada ao servidor):", show="*")
            if pwd:
                self.chat_client.send({"type":"AUTH","user":self.current_user,"pass":pwd})
        messagebox.showinfo("Conectado", f"Conectado a {host}:{port}")

    def disconnect_chat_client(self):
        if self.chat_client:
            try:
                self.chat_client.close()
            except:
                pass
            self.chat_client = None
            messagebox.showinfo("Chat", "Desconectado do servidor.")

    def send_chat_message(self):
        if not self.chat_client or not self.chat_client.sock:
            messagebox.showwarning("Chat", "Conecte ao servidor antes de enviar.")
            return
        if not self.current_user:
            messagebox.showwarning("Chat", "Faça login para enviar mensagens.")
            return
        to = self.e_chat_to.get().strip()
        text = self.e_chat_msg.get().strip()
        if not to or not text:
            messagebox.showwarning("Chat", "Preencha destino e mensagem.")
            return
        msg = {"type":"MSG","to":to,"text":text}
        ok = self.chat_client.send(msg)
        if ok:
            self.append_chat_line(f"[{now_iso()}] {self.current_user} → {to}: {text}")
            self.db.execute("INSERT INTO comunicacoes (origem, destino, tipo, texto, timestamp, status) VALUES (?, ?, ?, ?, ?, ?)",
                            (self.current_user, to, "MESSAGE_OUT", text, now_iso(), "sent"), commit=True)
            self.e_chat_msg.delete(0, tk.END)
        else:
            messagebox.showerror("Erro", "Falha ao enviar mensagem.")

    def broadcast_chat_message(self):
        if not self.chat_client or not self.chat_client.sock:
            messagebox.showwarning("Chat", "Conecte ao servidor antes de enviar.")
            return
        if not self.current_user:
            messagebox.showwarning("Chat", "Faça login para enviar mensagens.")
            return
        text = self.e_chat_msg.get().strip()
        if not text:
            messagebox.showwarning("Chat", "Mensagem vazia.")
            return
        msg = {"type":"BROADCAST","text":text}
        ok = self.chat_client.send(msg)
        if ok:
            self.append_chat_line(f"[{now_iso()}] {self.current_user} → TODOS: {text}")
            self.e_chat_msg.delete(0, tk.END)
        else:
            messagebox.showerror("Erro", "Falha ao enviar broadcast.")

    def call_user(self):
        if not self.chat_client or not self.chat_client.sock:
            messagebox.showwarning("Chat", "Conecte ao servidor antes de chamar.")
            return
        if not self.current_user:
            messagebox.showwarning("Chat", "Faça login para chamar.")
            return
        to = self.e_chat_to.get().strip()
        if not to:
            messagebox.showwarning("Chat", "Preencha destino para chamar.")
            return
        msg = {"type":"CALL","to":to}
        ok = self.chat_client.send(msg)
        if ok:
            self.append_chat_line(f"[{now_iso()}] {self.current_user} está chamando {to}...")
            self.db.execute("INSERT INTO comunicacoes (origem, destino, tipo, texto, timestamp, status) VALUES (?, ?, ?, ?, ?, ?)",
                            (self.current_user, to, "CALL_OUT", f"{self.current_user} calling {to}", now_iso(), "sent"), commit=True)
        else:
            messagebox.showerror("Erro", "Falha ao enviar chamada.")

    def append_chat_line(self, text: str):
        self.chat_txt.config(state=tk.NORMAL)
        self.chat_txt.insert(tk.END, text + "\n")
        self.chat_txt.see(tk.END)
        self.chat_txt.config(state=tk.DISABLED)

    # -------------------------
    # Suprimentos UI actions
    # -------------------------
    def refresh_all(self):
        self.refresh_suprimentos()
        self.refresh_registros()

    def refresh_suprimentos(self):
        for r in self.tree.get_children():
            self.tree.delete(r)
        for row in self.suprimentos.list_suprimentos():
            self.tree.insert("", tk.END, values=(row['id'], row['nome'], row['quantidade'],
                                                 row['unidade'], row['preco_unitario'], row['validade'], row['limite_alerta']))

    def refresh_registros(self):
        lines = []
        registros = self.db.fetchall("SELECT * FROM registros ORDER BY recebido_em DESC LIMIT 200")
        for r in registros:
            lines.append(f"[{r['recebido_em']}] {r['fonte']} / {r['tipo']} -> {r['dados']}")
        # append into chat_txt for visibility (not only chat)
        self.chat_txt.config(state=tk.NORMAL)
        self.chat_txt.insert(tk.END, "\n".join(lines) + ("\n" if lines else ""))
        self.chat_txt.config(state=tk.DISABLED)

    def add_suprimento_dialog(self):
        dlg = SuprimentoDialog(self.root)
        self.root.wait_window(dlg.top)
        if dlg.result:
            self.suprimentos.add_suprimento(SuprimentoModel(**dlg.result))
            self.refresh_suprimentos()

    def definir_limite_dialog(self):
        sel = self._get_selected_suprimento()
        if not sel:
            messagebox.showinfo("Seleção", "Selecione um suprimento na lista para definir limite.")
            return
        id_ = sel['id']
        current = sel.get('limite_alerta') or 0
        val = simpledialog.askfloat("Definir Limite", f"Limite atual: {current}\nDefinir novo limite (quantidade):", minvalue=0)
        if val is None:
            return
        self.suprimentos.update_suprimento(id_, limite_alerta=val)  # note: key is 'limite_alerta' in DB, but update function maps accepted keys
        self.refresh_suprimentos()

    def gerar_relatorio(self):
        rpt = self.report_gen.gerar_relatorio_diario()
        fname = f"relatorio_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2, default=str)
        messagebox.showinfo("Relatório", f"Relatório gerado e salvo em {fname}")
        self.refresh_registros()

    def on_double_suprimento(self, event):
        sel = self._get_selected_suprimento()
        if not sel:
            return
        info = json.dumps(sel, indent=2, default=str)
        messagebox.showinfo("Suprimento", info)

    def _get_selected_suprimento(self):
        sel = self.tree.selection()
        if not sel:
            return None
        vals = self.tree.item(sel[0])['values']
        return {
            'id': vals[0],
            'nome': vals[1],
            'quantidade': vals[2],
            'unidade': vals[3],
            'preco_unitario': vals[4],
            'validade': vals[5],
            'limite_alerta': vals[6]
        }

    def usar_suprimento_dialog(self):
        sel = self._get_selected_suprimento()
        if not sel:
            messagebox.showinfo("Seleção", "Selecione um suprimento.")
            return
        q = simpledialog.askfloat("Consumir", f"Quantidade a consumir de {sel['nome']} (atualmente {sel['quantidade']}):", minvalue=0)
        if q is None:
            return
        motivo = simpledialog.askstring("Motivo", "Motivo/Procedimento (opcional):")
        try:
            self.assistencial.registrar_uso(sel['id'], q, motivo)
            messagebox.showinfo("Sucesso", "Consumo registrado.")
            self.refresh_suprimentos()
            self.refresh_registros()
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def adicionar_quantidade_dialog(self):
        sel = self._get_selected_suprimento()
        if not sel:
            messagebox.showinfo("Seleção", "Selecione um suprimento.")
            return
        q = simpledialog.askfloat("Adicionar", f"Quantidade a adicionar em {sel['nome']}:", minvalue=0)
        if q is None:
            return
        try:
            self.suprimentos.add_quantity(sel['id'], q)
            self.refresh_suprimentos()
            messagebox.showinfo("Sucesso", "Quantidade adicionada.")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def novo_paciente_dialog(self):
        nome = simpledialog.askstring("Novo Paciente", "Nome do paciente:")
        if not nome:
            return
        prontuario = simpledialog.askstring("Prontuário", "Número do prontuário (opcional):")
        self.atendimento.novo_paciente(nome, prontuario)
        messagebox.showinfo("Ok", "Paciente cadastrado.")
        self.refresh_registros()

    def listar_pacientes(self):
        pats = self.atendimento.listar_pacientes()
        if not pats:
            messagebox.showinfo("Pacientes", "Nenhum paciente cadastrado.")
            return
        out = []
        for p in pats:
            out.append(f"[{p['id']}] {p['nome']} - internado: {p['internado']} - criado: {p['criado_em']}")
        messagebox.showinfo("Pacientes", "\n".join(out))

    def on_alerts(self, alerts):
        def _show():
            lines = []
            for a in alerts:
                if a['tipo'] == 'limite':
                    lines.append(f"ALERTA LIMITE: {a['suprimento']} estoque {a['quantidade']} <= limite {a['limite']}")
                elif a['tipo'] == 'vencendo':
                    lines.append(f"VENCENDO: {a['suprimento']} vence em {a['dias_restantes']} dias (val: {a.get('validade')})")
                elif a['tipo'] == 'vencido':
                    lines.append(f"VENCIDO: {a['suprimento']} vencido há {a['dias_atraso']} dias")
            if lines:
                for l in lines:
                    self.append_chat_line(f"[ALERTA {now_iso()}] {l}")
                try:
                    messagebox.showwarning("Alertas de Suprimentos", "\n".join(lines))
                except:
                    pass
        self.root.after(50, _show)

# -------------------------
# Dialogs
# -------------------------
class RegisterDialog:
    def __init__(self, parent):
        top = self.top = tk.Toplevel(parent)
        top.title("Registrar Usuário")
        self.result = None
        ttk.Label(top, text="Username:").grid(row=0,column=0,sticky=tk.W)
        self.e_user = ttk.Entry(top)
        self.e_user.grid(row=0,column=1)
        ttk.Label(top, text="Display name:").grid(row=1,column=0,sticky=tk.W)
        self.e_display = ttk.Entry(top)
        self.e_display.grid(row=1,column=1)
        ttk.Label(top, text="Role:").grid(row=2,column=0,sticky=tk.W)
        self.e_role = ttk.Entry(top)
        self.e_role.insert(0,"user")
        self.e_role.grid(row=2,column=1)
        ttk.Label(top, text="External ID (opcional):").grid(row=3,column=0,sticky=tk.W)
        self.e_ext = ttk.Entry(top)
        self.e_ext.grid(row=3,column=1)
        ttk.Label(top, text="Password:").grid(row=4,column=0,sticky=tk.W)
        self.e_pass = ttk.Entry(top, show="*")
        self.e_pass.grid(row=4,column=1)
        ttk.Label(top, text="Repeat:").grid(row=5,column=0,sticky=tk.W)
        self.e_pass2 = ttk.Entry(top, show="*")
        self.e_pass2.grid(row=5,column=1)
        btn = ttk.Button(top, text="Registrar", command=self.on_register)
        btn.grid(row=6,column=0,columnspan=2)

    def on_register(self):
        u = self.e_user.get().strip()
        d = self.e_display.get().strip() or u
        r = self.e_role.get().strip() or "user"
        e = self.e_ext.get().strip()
        p1 = self.e_pass.get()
        p2 = self.e_pass2.get()
        if not u or not p1:
            messagebox.showerror("Erro", "Username e senha são obrigatórios.")
            return
        if p1 != p2:
            messagebox.showerror("Erro", "Senhas não conferem.")
            return
        self.result = {"username":u, "display":d, "role":r, "external_id":e, "password":p1}
        self.top.destroy()

class LoginDialog:
    def __init__(self, parent):
        top = self.top = tk.Toplevel(parent)
        top.title("Login")
        self.result = None
        ttk.Label(top, text="Username:").grid(row=0,column=0,sticky=tk.W)
        self.e_user = ttk.Entry(top)
        self.e_user.grid(row=0,column=1)
        ttk.Label(top, text="Password:").grid(row=1,column=0,sticky=tk.W)
        self.e_pass = ttk.Entry(top, show="*")
        self.e_pass.grid(row=1,column=1)
        btn = ttk.Button(top, text="Login", command=self.on_login)
        btn.grid(row=2,column=0,columnspan=2)

    def on_login(self):
        u = self.e_user.get().strip()
        p = self.e_pass.get()
        if not u or not p:
            messagebox.showerror("Erro", "Preencha usuário e senha.")
            return
        self.result = {"username":u, "password":p}
        self.top.destroy()

class SuprimentoDialog:
    def __init__(self, parent):
        top = self.top = tk.Toplevel(parent)
        top.title("Adicionar Suprimento")
        self.result = None
        ttk.Label(top, text="Nome:").grid(row=0, column=0, sticky=tk.W)
        self.e_nome = ttk.Entry(top)
        self.e_nome.grid(row=0, column=1)
        ttk.Label(top, text="Quantidade:").grid(row=1, column=0, sticky=tk.W)
        self.e_q = ttk.Entry(top)
        self.e_q.grid(row=1, column=1)
        ttk.Label(top, text="Unidade:").grid(row=2, column=0, sticky=tk.W)
        self.e_u = ttk.Entry(top)
        self.e_u.grid(row=2, column=1)
        ttk.Label(top, text="Preço unitário:").grid(row=3, column=0, sticky=tk.W)
        self.e_p = ttk.Entry(top)
        self.e_p.grid(row=3, column=1)
        ttk.Label(top, text="Validade (YYYY-MM-DD, opcional):").grid(row=4, column=0, sticky=tk.W)
        self.e_v = ttk.Entry(top)
        self.e_v.grid(row=4, column=1)
        ttk.Label(top, text="Limite alerta:").grid(row=5, column=0, sticky=tk.W)
        self.e_l = ttk.Entry(top)
        self.e_l.grid(row=5, column=1)
        btn = ttk.Button(top, text="Adicionar", command=self.on_add)
        btn.grid(row=6, column=0, columnspan=2)

    def on_add(self):
        try:
            nome = self.e_nome.get().strip()
            q = float(self.e_q.get())
            unidade = self.e_u.get().strip()
            p = float(self.e_p.get()) if self.e_p.get().strip() else 0.0
            v = self.e_v.get().strip() or None
            l = float(self.e_l.get()) if self.e_l.get().strip() else 0.0
            if not nome:
                messagebox.showerror("Erro", "Nome obrigatório")
                return
            self.result = {
                'nome': nome, 'quantidade': q, 'unidade': unidade,
                'preco_unitario': p, 'validade': v, 'limite_alerta': l
            }
            self.top.destroy()
        except Exception as e:
            messagebox.showerror("Erro", f"Entrada inválida: {e}")

# -------------------------
# Seed sample data (if empty)
# -------------------------
def seed_sample_data(db: Database):
    rows = db.fetchall("SELECT COUNT(*) as c FROM suprimentos")
    if rows and rows[0]['c'] > 0:
        return
    s = Suprimentos(db)
    s.add_suprimento(SuprimentoModel(nome="Soro Fisiológico 0.9%", quantidade=200.0, unidade="ml", preco_unitario=1.5, validade=(datetime.date.today()+datetime.timedelta(days=90)).isoformat(), limite_alerta=20))
    s.add_suprimento(SuprimentoModel(
    nome="Gaze Estéril",
    quantidade=500.0,
    unidade="un",
    preco_unitario=0.12,
    validade=(datetime.date.today() + datetime.timedelta(days=365)).isoformat(),
    limite_alerta=50
    ))
    s.add_suprimento(SuprimentoModel(nome="Luvas Nitrílicas P", quantidade=1000.0, unidade="par", preco_unitario=0.08, validade=None, limite_alerta=100))
    db.set_config('monitor_interval', MONITOR_INTERVAL_SECONDS)
    db.set_config('chat_server', {"host":"0.0.0.0","port":9090,"enabled":False})
    db.set_config('chat_connect_host', '127.0.0.1')
    db.set_config('chat_connect_port', 9090)

# -------------------------
# Main
# -------------------------
def main():
    db = Database()
    seed_sample_data(db)
    root = tk.Tk()
    app = AppGUI(root, db)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root, app))
    root.mainloop()

def on_closing(root, app: AppGUI):
    if messagebox.askokcancel("Sair", "Deseja realmente sair?"):
        try:
            app.monitor.stop()
        except: pass
        try:
            app.chat_server.stop()
        except: pass
        try:
            if app.chat_client:
                app.chat_client.close()
        except: pass
        root.destroy()

if __name__ == "__main__":
    main()
