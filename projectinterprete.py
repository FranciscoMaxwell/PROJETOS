# interprete_local.py
# Intérprete local com tradução e fala entre dois idiomas (modo voz ou texto)

from googletrans import Translator
from gtts import gTTS
import speech_recognition as sr
import playsound
import os
import random

# ========== FUNÇÕES ==========

def traduzir(texto, destino):
    """Traduz texto para o idioma de destino usando googletrans."""
    tradutor = Translator()
    try:
        traducao = tradutor.translate(texto, dest=destino)
        return traducao.text
    except Exception as e:
        print(f"[ERRO na tradução]: {e}")
        return texto


def ouvir(lang='pt-BR'):
    """Capta áudio do microfone e converte em texto (SpeechRecognition + Google)."""
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"\n🎤 Fale ({lang}): ")
        audio = rec.listen(source)
    try:
        texto = rec.recognize_google(audio, language=lang)
        print("🗣️ Você disse:", texto)
        return texto
    except sr.UnknownValueError:
        print("🤔 Não entendi o que foi dito.")
        return ""
    except sr.RequestError:
        print("⚠️ Erro de conexão com o serviço de voz.")
        return ""


def falar(texto, lang='pt'):
    """Transforma texto em fala com gTTS."""
    if not texto:
        return
    nome = f"voz_{random.randint(0,9999)}.mp3"
    try:
        tts = gTTS(text=texto, lang=lang)
        tts.save(nome)
        playsound.playsound(nome)
    except Exception as e:
        print(f"[ERRO na fala]: {e}")
    finally:
        if os.path.exists(nome):
            os.remove(nome)

# ========== PROGRAMA PRINCIPAL ==========

print("🌍 Intérprete Local Multilíngue (Protótipo)")
print("Use microfone ou digitação. Diga 'sair' ou 'exit' para encerrar.\n")

# Idiomas: códigos ISO (ex: pt, en, es, fr, de, it, ja, ko)
idioma_1 = input("Pessoa 1 - Qual idioma você fala (ex: pt, en, es)? ").strip()
modo_1 = input("Pessoa 1 - Deseja digitar ou falar? (d/f): ").strip().lower()

idioma_2 = input("\nPessoa 2 - Qual idioma você fala (ex: pt, en, es)? ").strip()
modo_2 = input("Pessoa 2 - Deseja digitar ou falar? (d/f): ").strip().lower()

print(f"\n🟢 Interpretador iniciado entre {idioma_1.upper()} ↔ {idioma_2.upper()}.\n")

while True:
    # --- Turno Pessoa 1 ---
    print("\n🎧 Turno da Pessoa 1:")
    if modo_1 == 'f':
        msg_1 = ouvir(lang=idioma_1)
    else:
        msg_1 = input("💬 Digite sua mensagem: ")

    if msg_1.lower() in ['sair', 'exit']:
        print("👋 Encerrando sessão...")
        break

    traduzido_1 = traduzir(msg_1, destino=idioma_2)
    print(f"💬 Traduzido para {idioma_2}: {traduzido_1}")
    falar(traduzido_1, lang=idioma_2)

    # --- Turno Pessoa 2 ---
    print("\n🎧 Turno da Pessoa 2:")
    if modo_2 == 'f':
        msg_2 = ouvir(lang=idioma_2)
    else:
        msg_2 = input("💬 Digite sua mensagem: ")

    if msg_2.lower() in ['sair', 'exit']:
        print("👋 Encerrando sessão...")
        break

    traduzido_2 = traduzir(msg_2, destino=idioma_1)
    print(f"💬 Traduzido para {idioma_1}: {traduzido_2}")
    falar(traduzido_2, lang=idioma_1)
