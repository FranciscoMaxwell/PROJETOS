import os
import re
import json
import requests

BACKEND_PATH = "app.py"
OUTPUT_DIR = "frontend/gerados"
MODEL = "mistral"  # pode trocar por codellama ou phi3

# Garante que a pasta de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extrair_endpoints(caminho):
    """Lê o arquivo do backend e tenta encontrar rotas (FastAPI, Flask, etc.)"""
    with open(caminho, "r", encoding="utf-8") as f:
        codigo = f.read()

    # Expressões simples para capturar endpoints
    padroes = [
        r'@app\.get\("([^"]+)"', 
        r'@app\.post\("([^"]+)"',
        r'@app\.put\("([^"]+)"',
        r'@app\.delete\("([^"]+)"'
    ]

    endpoints = []
    for padrao in padroes:
        endpoints += re.findall(padrao, codigo)

    return endpoints

def gerar_componente(endpoint):
    """Cria prompt e gera componente React usando Ollama"""
    prompt = f"""
Crie um componente React funcional e responsivo que consome a rota '{endpoint}' do backend local.
Use fetch() e mostre o resultado na tela. Use Tailwind para estilizar.
"""

    print(f"🧠 Gerando front para endpoint: {endpoint}")

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL, "prompt": prompt},
        stream=True
    )

    texto = ""
    for linha in response.iter_lines():
        if linha:
            data = json.loads(linha.decode())
            if "response" in data:
                texto += data["response"]
            if data.get("done"):
                break

    return texto.strip()

def salvar_componente(endpoint, codigo):
    """Salva o componente gerado em arquivo .jsx"""
    nome = endpoint.strip("/").replace("/", "_") or "index"
    caminho = os.path.join(OUTPUT_DIR, f"{nome}.jsx")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(codigo)
    print(f"✅ Componente salvo: {caminho}\n")

def main():
    endpoints = extrair_endpoints(BACKEND_PATH)
    if not endpoints:
        print("⚠️ Nenhum endpoint encontrado no backend!")
        return

    for ep in endpoints:
        codigo = gerar_componente(ep)
        salvar_componente(ep, codigo)

if __name__ == "__main__":
    main()
