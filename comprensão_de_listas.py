# numeros = [1,4,7,9,10,12,21]

# quadrados = [list(map(lambda x: x**2 , numeros))]
# print(quadrados)


numeros = [1,4,7,9,10,12,21]
quadrados = [max**2 for max in numeros]
print(quadrados)


# criar uma lista de números pares dfe 0 a 10
# pares = [num for num in range(120) if num % 2 == 0]
# print(pares)


# ENREDO = "A LOGICA É APENAS O PRINCIPIO DA SABEDORIA, E NÃO O SEU FIM!"
# VOGAIS = ['A','E','I','O','U','Á', 
#           'É', 'Í','Ó','Ú']
# LISTA_DE_VOGAIS = [V for V in ENREDO if V in VOGAIS]
# print(F'A FRASE POSSUI {len(LISTA_DE_VOGAIS)} VOGAIS: ')
# print(LISTA_DE_VOGAIS)


#Distrivutiva entre valores de duas listas

# distributiva = [multi * plique for multi, plique in zip([6,8,9], [10,23,40])]
# print(distributiva)

distributiva = [multi*plique for multi in [6,8,9] for plique in [10,23,40]]
print(distributiva)