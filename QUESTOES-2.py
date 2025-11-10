# Escreva um código em Python que crie uma conexão com um banco SQLite 
# chamado empresa.db. Se o arquivo não existir, 
# ele deve ser criado. Imprima "Conexão estabelecida com sucesso".

# import sqlite3

# hot = None

# try:
#     hot = sqlite3.connect('empresa.db')
#     print('conexão estabelecida om sucesso!!')

# except sqlite3.Error as h:
#     print(f'Ocorreu um erro {h}')

# finally:
#     if hot:
#         hot.close()




# # Conecta ou cria o banco de dados
# with sqlite3.connect('empresa.db') as conn:

# import sqlite3


#     cursor = conn.cursor()
    
#     # Cria a tabela funcionarios
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS funcionarios (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             nome TEXT NOT NULL,
#             idade INTEGER,
#             salario REAL
#         )
#     """)
    
    # print("Tabela 'funcionarios' criada com sucesso!")


import sqlite3

# Conecta ou cria o banco de dados
with sqlite3.connect('empresa.db') as conn:
    cursor = conn.cursor()
    
    # Cria a tabela funcionarios se não existir
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS funcionarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            idade INTEGER,
            salario REAL
        )
    """)
    
    print("Tabela 'funcionarios' criada ou já existente!")
    
    # Inserir alguns registros (exemplo)
    funcionarios = [
        ("João Silva", 30, 2500.0),
        ("Maria Santos", 25, 3000.0),
        ("Carlos Oliveira", 40, 4000.0)
    ]
    
    # Insere cada funcionário
    for f in funcionarios:
        cursor.execute("""
            INSERT INTO funcionarios (nome, idade, salario)
            VALUES (?, ?, ?)
        """, f)
    
    # Salva as alterações
    conn.commit()
    print("Registros inseridos com sucesso!")
    
    # Consulta e mostra todos os dados
    cursor.execute("SELECT * FROM funcionarios")
    todos = cursor.fetchall()
    
    print("\nTodos os funcionários:")
    for linha in todos:
        print(linha)

        
# Crie uma tabela clientes com colunas: id (inteiro, PK), nome (varchar), email (varchar), idade (int).

# Insira 10 registros na tabela clientes.

# Selecione todos os clientes (SELECT * FROM clientes).

# Selecione clientes com idade maior que 30.

# Selecione apenas os nomes e emails dos clientes.

# Conte quantos clientes existem (COUNT(*)).

# Encontre o cliente mais velho (MAX(idade)).

# Encontre o cliente mais novo (MIN(idade)).

# Calcule a idade média dos clientes (AVG(idade)).

# Exiba clientes com idade entre 20 e 40 (BETWEEN).

# Selecione clientes cujo nome comece com “A” (LIKE 'A%').

# Ordene clientes por nome ascendente e por idade descendente.

# Atualize o email de um cliente específico.

# Exclua um cliente com certo id.

# Crie uma tabela pedidos com colunas: id, cliente_id (FK), data_pedido (date), valor (decimal).

# Insira registros em pedidos, relacionando clientes.

# Selecione todos os pedidos de um cliente específico (JOIN simples).

# Some o valor total dos pedidos por cliente (GROUP BY cliente_id).

# Use HAVING para filtrar clientes com valor total de pedidos > 1000.

# Exiba cliente e total de pedidos ordenado do maior para o menor.

# Nível Intermediário

# Use INNER JOIN, LEFT JOIN, RIGHT JOIN entre clientes e pedidos.

# Encontre clientes que não tenham nenhum pedido (LEFT JOIN + IS NULL).

# Encontre o pedido mais recente de cada cliente (GROUP BY cliente_id + MAX(data_pedido)).

# Use CROSS JOIN entre clientes e uma tabela meses para ver combinatórias.

# Subconsulta simples: selecione clientes cujo id está em uma subconsulta de pedidos com valor > X.

# Subconsultas correlacionadas: para cada cliente, selecione o valor máximo de pedido.

# Use WITH (CTE) para organizar consultas complexas.

# Crie uma view que mostra cliente + total de pedidos.

# Use CASE WHEN para categorizar clientes por faixa de gasto (ex: “baixo”, “médio”, “alto”).

# Atualize registros usando um JOIN entre tabelas.

# Delete registros em pedidos que atendam a uma condição baseada em outra tabela.

# Trabalhe com funções de janela (window functions): ROW_NUMBER(), RANK(), OVER (PARTITION BY ...).

# Encontre o top 3 pedidos por cliente usando funções de janela.

# Use LAG() ou LEAD() para comparar pedidos anteriores/próximos de um cliente.

# Calcule soma acumulada de valor de pedidos por data (função de janela).

# Particione dados para calcular média de valor de pedidos dentro de cada faixa de clientes.

# Truncar datas, extrair mês/ano de uma coluna de data_pedido.

# Nível Avançado / Sênior

# Otimize uma consulta lenta: uso de índices, EXPLAIN PLAN, analisar plano de execução.

# Criar índices (simples e compostos) nas tabelas para acelerar consultas frequentes.

# Usar FULL TEXT SEARCH (dependendo do SGDB) para buscas de texto.

# Particionamento de tabelas (por data ou por cliente).

# Tabelas temporárias e variáveis de tabela para cálculos intermediários.

# Manipulação de transações: BEGIN, COMMIT, ROLLBACK, isolamentos (READ COMMITTED, SERIALIZABLE).

# Controle de concorrência e bloqueios (locks) — problema de deadlock, como evitá-lo.

# Upssert (inserir ou atualizar em um só comando) — usando MERGE ou ON DUPLICATE KEY UPDATE.

# Procedures / Stored Procedures: criar uma procedure que insira um pedido validando estoque.

# Funções definidas pelo usuário (UDF): criar função que retorna status baseado em valor de pedido.

# Triggers (gatilhos): antes ou depois da inserção de pedido, atualizar um campo de resumo.

# Snapshot isolation / versões de dados no banco (dependendo do SGDB).

# Replicação, particionamento ou cluster de banco de dados (conceito).

# Análise de performance e tuning de queries complexas (joins múltiplos, subconsultas profundas).

# Utilizar CURSOR em T-SQL para iteração (com cuidado).

# Integração de SQL e Python: executar consultas SQL via Python (biblioteca pyodbc, pymssql, sqlalchemy).

# Automatizar processamentos via Python: rodar consulta, tratar dados, salvar resultados em outro banco ou arquivo.

# ETL simples com Python + SQL: extrair dados, transformar (Pandas) e carregar em outra tabela.

# Monitoramento e logging de execução de queries, tempo de execução, alertas.

# Escreva um script Python que se conecta ao SQL Server, executa uma consulta e imprime resultados formatados.

# Use Python para ler um CSV e inserir registros em uma tabela SQL.

# Em Python, execute uma query que retorne os top 5 clientes e gere um gráfico (matplotlib) desses valores.

# Use SQLAlchemy (ou outro ORM) para mapear classes Python às tabelas do banco e fazer consultas via objetos.

# Crie uma função em Python que recebe parâmetros do usuário e monta uma consulta SQL parametrizada (evitando SQL injection).

# Em um job Python, faça uma consulta que agregue dados no SQL, então recupere e salve em arquivo JSON.

# Combine Pandas + SQL: use pd.read_sql_query para trazer dados do SQL, manipule-os no Pandas e então escreva de volta no banco.