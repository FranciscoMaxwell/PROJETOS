#!/usr/bin/env python3
# avaliador_mistral_local.py
# Captura telas, OCR, envia para Mistral local via ollama, salva relatório TXT e (opcional) cola no editor.

import os, sys, time, json, shutil, subprocess, tempfile, re
from datetime import datetime
from typing import List, Optional

try:
    from PIL import Image, ImageChops
    import pytesseract, pyautogui, pyperclip
except ImportError:
    print("Instale dependências: pip install pillow pytesseract pyautogui pyperclip")
    sys.exit(1)

OUT_DIR = r"C:\Users\Maxwell Fernandes\.vscode\avaliador_resultados"
DEFAULT_INTERVAL = 10
DEFAULT_MAX_PAGES = 12
DEFAULT_DIFF_THRESHOLD = 2000
OLLAMA_EXEC = shutil.which("ollama") or "ollama"
MISTRAL_MODEL = "mistral:latest"

def now_filename_ts(): return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def ensure_outdir(p): os.makedirs(p, exist_ok=True); return p

def capture_screenshot(path, region=None):
    im = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()
    im.save(path); return path

def img_diff_pixels(a, b):
    a, b = Image.open(a).convert("L"), Image.open(b).convert("L")
    if a.size != b.size: b = b.resize(a.size)
    diff = ImageChops.difference(a, b)
    hist = diff.histogram()
    return sum(hist[1:])

def ocr_image(path): 
    try: return pytesseract.image_to_string(Image.open(path))
    except Exception as e: return f"[OCR ERROR] {e}"

def dedupe_text_pages(pages: List[str]) -> str:
    seen, out = set(), []
    for page in pages:
        for ln in [l.strip() for l in page.splitlines() if l.strip()]:
            if ln not in seen:
                seen.add(ln); out.append(ln)
    return "\n".join(out)

def call_mistral_via_ollama(prompt, model=MISTRAL_MODEL, timeout=90):
    if not shutil.which(OLLAMA_EXEC):
        raise RuntimeError("Ollama não encontrado no PATH.")
    proc = subprocess.run([OLLAMA_EXEC, "run", model],
                          input=prompt.encode("utf-8"),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8"))
    return proc.stdout.decode("utf-8", errors="ignore")

def extract_code_from_output(text):
    m = re.search(r"```(?:python)?\n([\s\S]+?)```", text, re.I)
    return m.group(1).strip() if m else text.strip()

def main():
    ensure_outdir(OUT_DIR)
    print("Avaliador Mistral local iniciado.")
    print("Abra a página do problema e clique nela para focar.")
    input("Pressione Enter para começar as capturas...")

    ts = now_filename_ts()
    paths, texts = [], []
    last = None; same = 0

    try:
        for i in range(DEFAULT_MAX_PAGES):
            img_path = os.path.join(OUT_DIR, f"snap_{ts}_{i}.png")
            capture_screenshot(img_path)
            print(f"[{i}] Capturada -> {img_path}")
            paths.append(img_path)

            if last:
                diff = img_diff_pixels(last, img_path)
                print(f"  diff vs prev: {diff}")
                same = same + 1 if diff < DEFAULT_DIFF_THRESHOLD else 0
            last = img_path

            txt = ocr_image(img_path)
            texts.append(txt)
            if same >= 2: break
            time.sleep(DEFAULT_INTERVAL)
    except KeyboardInterrupt:
        print("\nInterrompido, processando OCR acumulado...")

    # OCR pronto
    full_text = dedupe_text_pages(texts)
    print("\n--- OCR final ---\n", full_text[:800], "...\n")

    # chama o modelo
    prompt = (
        "Gere código Python resolvendo a questão abaixo. "
        "Entregue apenas o código entre ```python``` blocos.\n\n"
        + full_text
    )
    print("Chamando Mistral local via Ollama...")
    try:
        output = call_mistral_via_ollama(prompt)
        print("✅ Resposta recebida.")
    except Exception as e:
        print("Erro chamando Ollama:", e)
        output = str(e)

    code = extract_code_from_output(output)
    name = now_filename_ts()
    sol_py = os.path.join(OUT_DIR, f"solucao_{name}.py")
    sol_txt = os.path.join(OUT_DIR, f"relatorio_{name}.txt")

    with open(sol_py, "w", encoding="utf-8") as f: f.write(code)
    with open(sol_txt, "w", encoding="utf-8") as f:
        f.write("=== TEXTO OCR ===\n\n" + full_text + "\n\n")
        f.write("=== RESPOSTA MISTRAL ===\n\n" + output + "\n\n")
        f.write("=== CÓDIGO FINAL ===\n\n" + code + "\n")

    print(f"\nArquivos salvos em:\n - {sol_py}\n - {sol_txt}")

    # colagem opcional
    choice = input("\nDigite 'paste' para colar automaticamente, ou Enter para sair: ").strip().lower()
    if choice == "paste":
        input("Clique no campo de código e pressione Enter aqui...")
        pyperclip.copy(code)
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "v")
        print("✅ Código colado na página.")

    print("\nConcluído.")

if __name__ == "__main__":
    main()
