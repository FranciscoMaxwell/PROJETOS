import os
import json
import hashlib
import secrets
import binascii
from datetime import datetime
import pwinput

users_file = 'usuarios_secure.json'
logfile = 'log_tentativas.txt'
limite_tentativas = 3
pbkdf2_iter = 100_000


def carregar_usuarios():
    if not os.path.exists(users_file):
        return {}
    # encoding correto: "utf-8"
    with open(users_file, 'r', encoding="utf-8") as hot:
        return json.load(hot)


def salvar_usuarios(usuarios):
    # encoding correto: "utf-8"
    with open(users_file, 'w', encoding='utf-8') as hot:
        json.dump(usuarios, hot, indent=4, ensure_ascii=False)


def registrar_log(usuario, status):
    ts = datetime.now().isoformat(sep=' ', timespec='seconds')
    try:
        # spelling correto: encoding
        with open(logfile, 'a', encoding='utf-8') as hot:
            hot.write(f'{ts} - Usuário: {usuario} - {status}\n')
    except Exception as e:
        print(f'Erro ao gravar log: {e}')


def hash_password(password: str, salt: bytes = None, iterations: int = pbkdf2_iter):
    if salt is None:
        salt = secrets.token_bytes(16)
    dko = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return {
        'salt': binascii.hexlify(salt).decode('ascii'),
        'hash': binascii.hexlify(dko).decode('ascii'),
        'iterations': iterations,
    }


def verify_password(stored: dict, password_attempt: str) -> bool:
    salt = binascii.unhexlify(stored['salt'].encode('ascii'))
    iterations = stored.get('iterations', pbkdf2_iter)
    dko_attempt = hashlib.pbkdf2_hmac('sha256', password_attempt.encode('utf-8'), salt, iterations)
    return secrets.compare_digest(binascii.unhexlify(stored['hash'].encode('ascii')), dko_attempt)


def inicializar_usuarios():
    if not os.path.exists(users_file):
        usuarios = {}
        default_pass = '157'
        usuarios['desodorante'] = {
            'cred': hash_password(default_pass),
            'failed_attempts': 0,
            'locked': False,
        }
        salvar_usuarios(usuarios)
        print(f"Arquivo '{users_file}' criado com usuario padrão.")


def cadastrar_usuario():
    usuarios = carregar_usuarios()
    username = input('Escolha um nome de usuario: ').strip()
    if not username:
        print('Nome de usuario não pode ser vazio.')
        return

    if username in usuarios:
        print('Usuario já existe.')
        return

    while True:
        senha = pwinput.pwinput('Escolha uma senha (mín. 6 caracteres): ', mask="*").strip()
        if len(senha) < 6:
            print('Senha muito curta. Tente novamente.')
            continue
        senha2 = pwinput.pwinput('Repita a senha: ', mask="*").strip()
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
    print(f"Usuario '{username}' cadastrado com sucesso.")
    registrar_log(username, 'Usuario cadastrado')


def tentar_login():
    usuarios = carregar_usuarios()
    username = input('usuario: ').strip()
    if username not in usuarios:
        print('Usuario inexistente.')
        registrar_log(username, 'login falhou - usuario inexistente')
        return

    user_record = usuarios[username]
    if user_record.get('locked'):
        print('Conta bloqueada. Contate o administrador.')
        registrar_log(username, 'login falhou - conta bloqueada')
        return

    senha = pwinput.pwinput('senha: ', mask="*").strip()
    if verify_password(user_record['cred'], senha):
        print('Login bem-sucedido!')
        user_record['failed_attempts'] = 0
        salvar_usuarios(usuarios)
        registrar_log(username, 'login bem-sucedido')
    else:
        # correção: nome da chave era 'failed_attempts' (typo corrigido)
        user_record['failed_attempts'] = user_record.get('failed_attempts', 0) + 1
        salvar_usuarios(usuarios)
        tent = user_record['failed_attempts']
        print(f'Senha incorreta. tentativa {tent} de {limite_tentativas}.')
        registrar_log(username, f"Senha incorreta ({tent}/{limite_tentativas})")
        # correção lógica: usar >= (ou >=) em vez de ">-"
        if tent >= limite_tentativas:
            user_record['locked'] = True
            salvar_usuarios(usuarios)
            print('Conta bloqueada por muitas tentativas incorretas.')
            registrar_log(username, 'conta bloqueada')


def menu_principal():
    inicializar_usuarios()
    while True:
        print("\n---------------MENU---------------")
        print('1) Login')
        print('2) Cadastrar usuário')
        print('3) Sair')
        opc = input('Escolha: ').strip()
        if opc == '1':
            tentar_login()
        elif opc == '2':
            cadastrar_usuario()
        elif opc == '3':
            print('Tchau')
            break
        else:
            print('Opção inválida.')


if __name__ == '__main__':
    menu_principal()
