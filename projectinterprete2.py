import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
import speech_recognition as sr
from gtts import gTTS
import playsound
from googletrans import Translator
import os
import time

translator = Translator()

class RoboTradutorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Intérprete de Idiomas")
        self.root.geometry("500x600")
        self.root.configure(bg="#1e1e1e")

        self.frame = ttk.Frame(root)
        self.frame.pack(pady=20)

        self.title = tk.Label(root, text="🤖 Intérprete IA", font=("Arial", 20, "bold"), fg="#00ffcc", bg="#1e1e1e")
        self.title.pack(pady=10)

        self.status_label = tk.Label(root, text="Escolha os idiomas para cada pessoa:", fg="white", bg="#1e1e1e", font=("Arial", 12))
        self.status_label.pack(pady=5)

        self.lang_frame = tk.Frame(root, bg="#1e1e1e")
        self.lang_frame.pack(pady=10)

        tk.Label(self.lang_frame, text="Pessoa 1:", bg="#1e1e1e", fg="white").grid(row=0, column=0, padx=10)
        tk.Label(self.lang_frame, text="Pessoa 2:", bg="#1e1e1e", fg="white").grid(row=1, column=0, padx=10)

        self.lang1 = ttk.Combobox(self.lang_frame, values=["pt", "en", "es", "fr", "de", "it"], width=10)
        self.lang2 = ttk.Combobox(self.lang_frame, values=["pt", "en", "es", "fr", "de", "it"], width=10)
        self.lang1.set("pt")
        self.lang2.set("en")
        self.lang1.grid(row=0, column=1)
        self.lang2.grid(row=1, column=1)

        self.method_label = tk.Label(root, text="Escolha o modo de entrada:", fg="white", bg="#1e1e1e", font=("Arial", 12))
        self.method_label.pack(pady=5)

        self.method_var = tk.StringVar(value="texto")
        self.text_btn = ttk.Radiobutton(root, text="Digitar", variable=self.method_var, value="texto")
        self.voice_btn = ttk.Radiobutton(root, text="Falar (microfone)", variable=self.method_var, value="voz")
        self.text_btn.pack()
        self.voice_btn.pack()

        self.chat_box = tk.Text(root, height=15, width=55, bg="#2e2e2e", fg="white", font=("Arial", 10))
        self.chat_box.pack(pady=15)
        self.chat_box.insert(tk.END, "🤖 Pronto para iniciar!\n")
        self.chat_box.config(state="disabled")

        self.start_btn = ttk.Button(root, text="Iniciar Conversa", command=self.iniciar_conversa)
        self.start_btn.pack(pady=10)

        self.animation_label = tk.Label(root, text="⬤", font=("Arial", 32), fg="#003333", bg="#1e1e1e")
        self.animation_label.pack(pady=20)

        self.animating = False

    def speak(self, text, lang):
        try:
            tts = gTTS(text=text, lang=lang)
            filename = "voice.mp3"
            tts.save(filename)
            playsound.playsound(filename)
            os.remove(filename)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao reproduzir áudio: {e}")

    def listen(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            self.log("🎙️ Ouvindo...")
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language=self.lang1.get())
            self.log(f"🗣️ Você disse: {text}")
            return text
        except sr.UnknownValueError:
            self.log("❌ Não entendi o que foi dito.")
            return ""
        except sr.RequestError:
            self.log("⚠️ Erro ao acessar o serviço de reconhecimento.")
            return ""

    def translate_and_speak(self, text, src, dest, person):
        if not text:
            return
        translated = translator.translate(text, src=src, dest=dest).text
        self.log(f"💬 {person} ({dest}): {translated}")
        self.speak(translated, dest)

    def log(self, message):
        self.chat_box.config(state="normal")
        self.chat_box.insert(tk.END, message + "\n")
        self.chat_box.config(state="disabled")
        self.chat_box.see(tk.END)

    def animate_robot(self):
        colors = ["#003333", "#006666", "#00cccc", "#00ffcc", "#00cccc", "#006666"]
        while self.animating:
            for color in colors:
                if not self.animating:
                    break
                self.animation_label.config(fg=color)
                time.sleep(0.15)

    def iniciar_conversa(self):
        Thread(target=self._iniciar_conversa_thread).start()

    def _iniciar_conversa_thread(self):
        self.start_btn.config(state="disabled")
        self.animating = True
        Thread(target=self.animate_robot, daemon=True).start()

        method = self.method_var.get()
        lang1 = self.lang1.get()
        lang2 = self.lang2.get()

        self.log(f"🌐 Iniciando conversa entre {lang1.upper()} ↔ {lang2.upper()} usando modo: {method.upper()}")

        for turno in range(2):
            self.log("\n🧍 Pessoa 1:")
            if method == "voz":
                msg1 = self.listen()
            else:
                msg1 = self.input_popup("Pessoa 1 - Digite sua mensagem:")
            self.translate_and_speak(msg1, src=lang1, dest=lang2, person="Pessoa 1")

            self.log("\n🧍 Pessoa 2:")
            if method == "voz":
                msg2 = self.listen()
            else:
                msg2 = self.input_popup("Pessoa 2 - Digite sua mensagem:")
            self.translate_and_speak(msg2, src=lang2, dest=lang1, person="Pessoa 2")

        self.animating = False
        self.animation_label.config(fg="#003333")
        self.start_btn.config(state="normal")
        self.log("\n✅ Conversa encerrada.")

    def input_popup(self, prompt):
        popup = tk.Toplevel(self.root)
        popup.title("Entrada de Texto")
        popup.geometry("400x200")
        tk.Label(popup, text=prompt, font=("Arial", 12)).pack(pady=10)
        entry = tk.Entry(popup, width=40)
        entry.pack(pady=10)
        result = []

        def submit():
            result.append(entry.get())
            popup.destroy()

        ttk.Button(popup, text="Enviar", command=submit).pack(pady=10)
        popup.wait_window()
        return result[0] if result else ""


if __name__ == "__main__":
    root = tk.Tk()
    app = RoboTradutorApp(root)
    root.mainloop()
