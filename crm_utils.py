import sqlite3
import os
from tkinter import messagebox

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crm.db")

def db_connect():
    return sqlite3.connect(DB_PATH)

def db_execute(query, params=()):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id


# 🔧 Regras de dependência (exclusão em cascata)
# chave = tabela principal
# valor = lista de (tabela_dependente, campo_relacionado)
CASCADE_MAP = {
    "leads": [("interacoes_leads", "lead_id")],
    "clientes": [("tickets", "cliente_id"), ("pedidos", "cliente_id")],
    "produtos": [("pedidos_itens", "produto_id")],
    # adicione mais se quiser que apague junto
}


def multi_delete(tree, table, refresh_callback=None, label="registro"):
    """
    Exclui vários registros de uma vez de uma tabela específica,
    com suporte a deleção em cascata (dependências definidas em CASCADE_MAP).
    """
    selection = tree.selection()
    if not selection:
        messagebox.showwarning("Aviso", f"Selecione um ou mais {label}(s) para excluir.")
        return

    if not messagebox.askyesno("Confirmação", f"Deseja excluir {len(selection)} {label}(s)?"):
        return

    for item in selection:
        rec_id = tree.item(item)["values"][0]  # ID é sempre a 1ª coluna

        # 🔄 Excluir registros dependentes (se existirem)
        if table in CASCADE_MAP:
            for dep_table, dep_field in CASCADE_MAP[table]:
                db_execute(f"DELETE FROM {dep_table} WHERE {dep_field}=?", (rec_id,))

        # 🗑️ Excluir registro principal
        db_execute(f"DELETE FROM {table} WHERE id=?", (rec_id,))

    messagebox.showinfo("Sucesso", f"{len(selection)} {label}(s) excluído(s) com sucesso.")

    if refresh_callback:
        refresh_callback()
