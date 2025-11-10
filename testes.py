import sqlite3
conn = sqlite3.connect("crm.db")
cur = conn.cursor()

print("\n--- ESTRUTURA DE OPORTUNIDADES ---")
cur.execute("PRAGMA table_info(oportunidades)")
print(cur.fetchall())

print("\n--- ESTRUTURA DE TICKETS ---")
cur.execute("PRAGMA table_info(tickets)")
print(cur.fetchall())

conn.close()
