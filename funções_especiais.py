# Funções Lambda (anônimas)
# Sintaxe:
#Lambda argumentos: expressão

# quadrado = lambda x: x**2

# for i in range(1,11):
#     print(quadrado(i))

# par = lambda x: x %2 == 0
# print(par(102))

# f_c = lambda f: (f-32) / 9 * 5    # 212 fareheights -32 e divide por 9 e multiplica por 5
# print(f_c(32))

# Função map()
# Sintaxe
# map(fubnção, iterável)

# num = [1,2,3,4,5,6,7,8]
# dobro = list(map(lambda x: x*2, num))
# print(dobro)

# palavras = ['Python', 'é', 'uma', 'linguagem', 'de', 'programação'] # map strigs
# maiusculas = list(map(str.upper, palavras))
# print(maiusculas)

#Função filter()
#Sintaxe:
#Filter(função, sequencia)

# def numeros_pares(n):
#     return n % 2 == 0

# numeros = [1,2,3,4,5,6,7,8,9,10,11,12,123]
# num_par = list(filter(numeros_pares, numeros))  # filter numeros inteiros
# print(num_par)

# numeros = [1,2,3,4,5,6,7,8,9,10,11,12,123]
# num_impar =list(filter(lambda x: x % 2 != 0, numeros))
# print(num_impar)

#Função reduce()
#Sintaxe
#reduce(função, sequencia, valor_inicial)

# from functools import reduce
# def mult(x,y):
#     return x * y


# num = [1,2,3,4,5,6]

# total = reduce(mult, num)
# print(total)

# Soma cumulativa dos quadrados de valores, usando expressão lambda

# from functools import reduce
# # ((1²+2²)² + 3²)² + 4²
# numeros = [1,2,3,4]
# total = reduce(lambda ac, cu: ac**2 + cu**2, numeros)
# # você pode usar qualquer coisa no "ac" ou "cu", tipo x e y
# print(total)

# numeros = [1, 2, 3, 4]

# resultado = list(map(lambda x: x * x * 2, numeros))
# print(resultado)  