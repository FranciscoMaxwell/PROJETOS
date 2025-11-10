#Lista: representa uma sequencia de valores

# sintaxe: nome_lista = [valores]

# notas = [5,6,7,8,9]
# notas2 = [7,6,8,4,1,353,500,57,59,51,53]
# valores = notas + notas2
# print(valores)

# n1 = [0,19,3,57,8,4]
# n2 = [46,0.5,7,9,3,113,14]

# valores = n1+n2
# valores[0] = 9
# print(len(valores))
# print(sorted(valores, reverse = True))
# print(sum(valores))
# print(max(valores))
# print(min(valores))

# n1 = [0,19,3,57,8,4]
# n2 = [46,0.5,7,9,3,113,14]
# valores = n1+n2
# valores.append(13)
# print(valores)
# valores.pop()
# print(valores)
# valores.pop(12)
# print(valores)
# valores.insert(5,6) 
# print(valores)
# print(113 in valores)

# planeta = ['terra', 'mercurio', 'marte', 'b2b', 'urano']
# for planetas in planeta:
#     print(planeta)

bebidas = []

for ou in range(5):
    print(f'digite uma bebida: ')
    bebida = input()
    bebidas.append(bebida)
    
bebidas.sort()

print(f'\nbebidas escolhidas: ')
for bebida in bebidas:
    print(bebida)

print(f'\nSaude!!!')

        
    

        