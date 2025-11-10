import os
import json
import hashlib
import secrets
import binascii
from datetime import datetime
from getpass import getpass

USERS_FILE = "usuarios_secure.json"
LOG_FILE = "log_tentativas.txt"
LIMITE_TENTATIVAS = 3
PBKDF2_ITER = 100_000  # número de iterações (bom padrão)


# ---------- utilitários de persistência / log ----------
def carregar_usuarios():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def salvar_usuarios(usuarios):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(usuarios, f, indent=4, ensure_ascii=False)


def registrar_log(usuario, status):
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} - Usuário: {usuario} - {status}\n")
    except Exception as e:
        print(f"⚠️ Erro ao gravar log: {e}")


# ---------- hashing seguro (PBKDF2) ----------
def hash_password(password: str, salt: bytes = None, iterations: int = PBKDF2_ITER):
    """
    Retorna dict com salt (hex), hash (hex) e iterations.
    Se salt não for passado, gera um novo.
    """
    if salt is None:
        salt = secrets.token_bytes(16)  # 128-bit salt
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return {
        "salt": binascii.hexlify(salt).decode("ascii"),
        "hash": binascii.hexlify(dk).decode("ascii"),
        "iterations": iterations,
    }


def verify_password(stored: dict, password_attempt: str) -> bool:
    salt = binascii.unhexlify(stored["salt"].encode("ascii"))
    iterations = stored.get("iterations", PBKDF2_ITER)
    dk_attempt = hashlib.pbkdf2_hmac("sha256", password_attempt.encode("utf-8"), salt, iterations)
    return secrets.compare_digest(binascii.unhexlify(stored["hash"].encode("ascii")), dk_attempt)


# ---------- funcionalidades: cadastro e login ----------
def inicializar_usuarios():
    """
    Se não existir arquivo, cria um com um usuário padrão.
    """
    if not os.path.exists(USERS_FILE):
        usuarios = {}
        # usuário padrão (apenas para demo). Em produção, não incluir senha em texto.
        default_pass = "157"
        usuarios["desodorante"] = {
            "cred": hash_password(default_pass),
            "failed_attempts": 0,
            "locked": False,
        }
        salvar_usuarios(usuarios)
        print(f"Arquivo '{USERS_FILE}' criado com usuário padrão.")


def cadastrar_usuario():
    usuarios = carregar_usuarios()
    username = input("Escolha um nome de usuário: ").strip()
    if not username:
        print("Nome de usuário não pode ser vazio.")
        return

    if username in usuarios:
        print("Usuário já existe.")
        return

    while True:
        senha = getpass("Escolha uma senha (mín. 6 caracteres): ").strip()
        if len(senha) < 6:
            print("Senha muito curta. Tente novamente.")
            continue
        senha2 = getpass("Repita a senha: ").strip()
        if senha != senha2:
            print("Senhas não conferem. Tente novamente.")
            continue
        break

    usuarios[username] = {
        "cred": hash_password(senha),
        "failed_attempts": 0,
        "locked": False,
    }
    salvar_usuarios(usuarios)
    print(f"Usuário '{username}' cadastrado com sucesso.")
    registrar_log(username, "Usuário cadastrado")


def tentar_login():
    usuarios = carregar_usuarios()
    username = input("Usuário: ").strip()
    if username not in usuarios:
        print("Usuário inexistente.")
        registrar_log(username, "Login falhou - usuário inexistente")
        return

    user_record = usuarios[username]
    if user_record.get("locked"):
        print("Conta bloqueada. Contate o administrador.")
        registrar_log(username, "Login falhou - conta bloqueada")
        return

    senha = getpass("Senha: ").strip()
    if verify_password(user_record["cred"], senha):
        print("✅ Login bem-sucedido!")
        user_record["failed_attempts"] = 0
        salvar_usuarios(usuarios)
        registrar_log(username, "Login bem-sucedido")
    else:
        user_record["failed_attempts"] = user_record.get("failed_attempts", 0) + 1
        salvar_usuarios(usuarios)
        tent = user_record["failed_attempts"]
        print(f"❌ Senha incorreta. Tentativa {tent} de {LIMITE_TENTATIVAS}.")
        registrar_log(username, f"Senha incorreta ({tent}/{LIMITE_TENTATIVAS})")
        if tent >= LIMITE_TENTATIVAS:
            user_record["locked"] = True
            salvar_usuarios(usuarios)
            print("⛔ Conta bloqueada por muitas tentativas incorretas.")
            registrar_log(username, "Conta bloqueada")


# ---------- menu simples ----------
def menu_principal():
    inicializar_usuarios()
    while True:
        print("\n--- MENU ---")
        print("1) Login")
        print("2) Cadastrar usuário")
        print("3) Sair")
        opc = input("Escolha: ").strip()
        if opc == "1":
            tentar_login()
        elif opc == "2":
            cadastrar_usuario()
        elif opc == "3":
            print("Tchau.")
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    menu_principal()
