import os
import json
import subprocess
from datetime import datetime

# ===============================
# 🔧 Configurações principais
# ===============================

# Diretório raiz do projeto (onde estão os arquivos a analisar)
PROJECT_DIR = "."

# Nome do relatório final
REPORT_FILE = "quality_report.json"

# ===============================
# 🧩 Funções auxiliares
# ===============================

def run_command(command, description):
    """
    Executa um comando no terminal e retorna a saída (stdout).
    """
    print(f"[INFO] Executando: {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR
        )
        return {
            "description": description,
            "command": command,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode
        }
    except Exception as e:
        return {
            "description": description,
            "error": str(e)
        }

# ===============================
# 🧰 Ferramentas de análise
# ===============================

tools = {
    "Ruff (lint)": "ruff check . --output-format text",
    "Black (formatação)": "black --check .",
    "Mypy (tipagem)": "mypy . --pretty --no-error-summary",
    "Pytest (testes rápidos)": "pytest --maxfail=3 --disable-warnings -q",
    "Bandit (segurança)": "bandit -r . -f txt"
}

# ===============================
# 🚀 Execução das ferramentas
# ===============================

def main():
    print("=" * 60)
    print("🔍 Iniciando análise automática de qualidade de código")
    print("=" * 60)

    all_results = {
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "diretorio": os.path.abspath(PROJECT_DIR),
        "resultados": []
    }

    for name, cmd in tools.items():
        result = run_command(cmd, name)
        all_results["resultados"].append(result)

    # ===============================
    # 💾 Salvando o relatório
    # ===============================

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("\n✅ Análise concluída!")
    print(f"📄 Relatório salvo em: {os.path.abspath(REPORT_FILE)}")

# ===============================
# ▶️ Execução direta
# ===============================
if __name__ == "__main__":
    main()
