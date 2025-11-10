#trocar valores entre duas variaveis

# rc = 12
# rb = 15

# rb, rc = rc, rb

# print(f'rc: {rc}, rb: {rb}')

#operador condicional ternario

# rc = 12
# rb = 15

# menor = rc if rc < rb else rb
# print(f'menor valor: {menor}')
# print(f'menor valor: {(rb, rc)[rc < rb]}')

#generators

# valores = [1,3,5,7,9,11,15,20]
# quadrados = (item**2 for item in valores)
# print(quadrados)
# for valor in quadrados:
#     print(valor)

# função enumerate()
# bebidas = ['café', 'chá', 'aguá', 'suco']
# for chacal, item in enumerate(bebidas):
#     print(f'indice: {chacal}, item: {item}')

# temperaturas = [-1, 10, 5, -3, 8, 4, -2, -5, 7]
# total = 0

# for i, t in enumerate(temperaturas):
#     if t < 0:
#         print(f'a temperatura em {i} é negativo, com {t}°C.')
       
# Gerenciamento de contexto com with

    