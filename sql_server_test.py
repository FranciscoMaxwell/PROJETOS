# import pyodbc

# # Configuração de conexão
# server = r'localhost\SQLEXPRESS'  # nome do servidor e instância
# database = 'EmpresaDB'            # nome do banco que você vai criar
# username = ''                     # se usar autenticação do Windows, deixe vazio
# password = ''                     # se usar autenticação do Windows, deixe vazio

# # # String de conexão (autenticação do Windows)
# conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

# # Conecta ao SQL Server
# try:
#     conn = pyodbc.connect(conn_str)
#     cursor = conn.cursor()
#     print("Conexão estabelecida com sucesso!")

#     # Criar tabela funcionarios (se não existir)
#     cursor.execute("""
#     IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='funcionarios' AND xtype='U')
#     CREATE TABLE funcionarios (
#         id INT IDENTITY(1,1) PRIMARY KEY,
#         nome NVARCHAR(100) NOT NULL,
#         idade INT,
#         salario DECIMAL(10,2)
#     )
#     """)
#     conn.commit()
#     print("Tabela 'funcionarios' criada ou já existente!")

#     # Inserir alguns registros
#     funcionarios = [
#         ("João Silva", 30, 2500.00),
#         ("Maria Santos", 25, 3000.00),
#         ("Carlos Oliveira", 40, 4000.00)
#     ]

#     for f in funcionarios:
#         cursor.execute("INSERT INTO funcionarios (nome, idade, salario) VALUES (?, ?, ?)", f)

#     conn.commit()
#     print("Registros inseridos com sucesso!")

#     # Consultar dados
#     cursor.execute("SELECT * FROM funcionarios")
#     todos = cursor.fetchall()
#     print("\nTodos os funcionários:")
#     for linha in todos:
#         print(f"ID: {linha.id} | Nome: {linha.nome} | Idade: {linha.idade} | Salário: R${linha.salario:.2f}")

# except Exception as e:
#     print("Erro:", e)

# finally:
#     if 'conn' in locals():
#         conn.close()








# import pyodbc

# #Configurar conexão
# server = r'localhost\SQLEXPRESS'
# database = "laba"
# username = ''
# password = ''

# jorge =  f'DRIVER={{SQL Server}};server={server};database={database};trusted_connection=yes;'

# try:
#     lancha = pyodbc.connect(jorge)
#     jato = lancha.cursor()
#     print('Conexão estabelecida com sucesso!')

#     jato.execute("DROP TABLE IF EXISTS noram") #apaga a tabela antiga
#     lancha.commit()

#     jato.execute("""
#     CREATE TABLE noram (
#         NUMERACAO int identity(1,1) primary key,
#         NOME nvarchar(100) not null,
#         IDADE int,
#         SALARIO decimal(10,3),
#         BUSTO decimal(10,2)
# )""")

#     lancha.commit()
#     print(f'tabela criada ou já existente: ')


#     jato.execute("delete from noram")  # apaga todos os registros dentro da tabela
#     lancha.commit()

#     jato.execute("dbcc checkident('noram', reseed, 1)") #reseta a 0 o ID
#     lancha.commit()

#     noram = [
# ('maru', 60, 2900.00, 70),
# ('jubibjubao', 80, 3000.00, 80),
# ('caralhiudo', 110, 50000.00, 90),
# ('mocreia', 40, 1000.00, 55),
# ('jarao', 50, 5000.00, 45),
# ('jorbes', 2, 40000.00, 47),
# ('gotica rabuda', 40, 500.00, 150)
# ]

#     for dor in noram:
#         jato.execute('insert into noram (nome,idade,salario,busto) values(?,?,?,?)', dor)

#     lancha.commit()
#     print('REGISTROS INSERIDOS COM SUCESSO')


#     jato.execute('select * from noram')
#     mega = jato.fetchall()
#     print('\nTodos os funcionarios')
#     for boga in mega:
#         print(f'ID: {boga.NUMERACAO} | NOME: {boga.NOME} | idade: {boga.IDADE} | salario: {boga.SALARIO} | busto: {boga.BUSTO}')

# except Exception as e:
#     print("deu erro man", e)

# finally:
#     if 'lancha' in locals():
#         lancha.close()




# Como você criaria uma tabela simples chamada 
# clientes com colunas para id, nome e email?

# import pyodbc


# server = r'localhost\SQLEXPRESS'
# database = 'lore'
# username = ''
# password = ''

# move = f'DRIVER={{SQL server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

# try:
#     caju = pyodbc.connect(move)
#     cho = caju.cursor()
#     print('Conexão feita com sucesso!!')

#     cho.execute('DROP TABLE if exists clientes')
#     caju.commit()


#     cho.execute("""
#     create table clientes(
#     Idnumber int identity(1,1),
#     Nome nvarchar(100) not null,
#     Email nvarchar(100) not null  
# )""")
    
#     caju.commit()
#     print('tabela criada ou já existente...')

#     clientes=[
#         ('lina', 'lina@gmail.com'),
#         ('jockey', 'jockeylima@gmail.com'),
#         ('jarem', 'jaremco@gmail.com')
#     ]


#     for tro in clientes:
#         cho.execute('insert into clientes(nome, email) values (?,?)', tro)

#     caju.commit()
#     print('os registros foram adicionados com sucesso!!')


#     cho.execute('select * from clientes')
#     uru = cho.fetchall()
#     print('\nMostrando todos')
#     for zx in uru:
#         print(f' IDNUMBER: {zx[0]} | NOME: {zx[1]} | EMAIL: {zx[2]}')


# except Exception as o:
#     print('deu erro man', o)

# finally:
#     if 'caju' in locals():
#         caju.close()


import pyodbc

server = r'localhost\SQLEXPRESS'
database = 'hot'
username = ''
password = ''

rock = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes'

try:
    quente = pyodbc.connect(rock)
    frio = quente.cursor()
    print('Conexão feita com sucesso.....')

    frio.execute("""
    IF OBEJECT ID('pacman', 'U') IS NOT NULL DROP TABLE pacman;
        CREAT TABLE pacman(
    codigo int indentity(1,1) PRIMARY KEY,
    nome nvarchar(100) not null,
    peito decimal(10,2) not null,
    bunda decimal(10,2) not null,
    email nvarchar(100) not null,
    idade int            
)""")
    
    quente.commit()

    pacman=[
('rapariga', 50.8, 75.7, 'rapariga7777@gmail.com', 20),
('medonha', 2, 5, 'medonhaeeeee@gmail.com', 90),
('jucicleide', 100, 150, 'gigablast@gmail.com', 18),
('prostituta', 50, 100, 'prosti@gmail.com', 43)
]

    
    for bin in pacman:
        frio.execute('INSERT INTO pacman (nome,peito,bunda,email,idade) values (?,?,?,?,?)', bin)
    quente.commit()
    print('Registrado com sucesso....')


    frio.execute('SELECT * FROM pacman')
    bina = frio.fetchall()
    print('\nMostrando todas as pessoas')
    for shock in bina:
        print(f' ID: {shock[0]} | NOME: {shock[1]} | PEITO: {shock[2]} | BUNDA: {shock[3]} | EMAIL: {shock[4]} | IDADE: {shock[5]}')


    frio.execute('UPDATE pacman SET nome = ? WHERE codigo = ?', ('roger', 1))
    quente.commit()


    