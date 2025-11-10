# Dicionarios
elemento = {
    'Z': 6,
    'RAPADURA': 'doce',
    'grupo': ' metais alcalinos',
    'densidade': 0.670
}

print(f'Elemento: {elemento['grupo']}')
print(f'Densidade: {elemento['densidade']}')
print(f'COISA ESTRANHA: {elemento['RAPADURA']}')
print(F'O dicionario possui {len(elemento)} nessa porra.')

#Atualizar uma entrada
elemento['grupo'] = 'Alcalinos'
print(elemento)

#Adicionar uma entrada
elemento['putaria'] =1
print(elemento)

# #Exclusão de itens em dicionarios
# del elemento['grupo']
# print(elemento)

# #Apagar tudo, mas o dicionario continua existindo no computador
# elemento.clear()
# print(elemento)

# #Apagar tudo, o dicionario deixa de existir no computador
# del elemento
# print(elemento)

print(elemento.items())
for separa in elemento.items():
    print(separa)

print(elemento.keys())
for chaves in elemento.keys():
    print(chaves)

print(elemento.values())
for valores in elemento.values():
    print(valores)

for coisadin, melhorzim in elemento.items():
    print(f' {coisadin}: {melhorzim}')