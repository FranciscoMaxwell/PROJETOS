#Funções
#Modularização, Reúso de Código, Legibilidade


# def <nome_função>  ([argumentos]):
#     <instruções>

# def mensagem():
#     print('3d maxwell blender navios')
#     print('super navios 3d mega upeer lindesas e incriveis')

# mensagem()

# Função com argumentos
# def soma(a,b):
#     print(a+b)

# soma(12, 8)

# def mult(x,y):
#     return x*y

# vadia = 5
# puta = 8
# rapariga = mult(vadia,puta)
# print(f' o produto de {vadia} e {puta} é igual a {rapariga}')

# def div(k, j):
#     if j != 0:
#         return k/j
#     else:
#         return ' impossivel dividir por zero!'

# if __name__ == '__main__':
#     a = int(input(' Digite um numero: '))
#     b = int(input('Digite outro numero: '))

#     r = div(a,b)
#     print(f'{a} dividido por {b} é igual a {r}')

# def quadrado(val):
#     quadrados = []
#     for x in val:
#         quadrados.append(x**2)
#     return quadrados

# if __name__ == '__main__':
#     valores = [2,5,7,9,12]
#     resultados = quadrado(valores)
#     for g in resultados:
        
#         print(g)

# def contar(caractere,num=11):
#     for i in range(1, num):
#         print(caractere)

# if __name__ == '__main__': #oque conta é oque põe neste aqui como print
#     contar('$', 6)

x = 5
y = 7
z = 9

def soma_mult(a, b, c = 0):
    if c == 0:
        return a * b
    else:
        return a + b + c
    
if __name__ == '__main__':
    res = soma_mult(x,y, z)
    print(res)