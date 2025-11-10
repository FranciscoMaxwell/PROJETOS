#modulo com dunçoes variadas
#função que exibe mensagem de boa vindas:
def mensagem():
    print('boson equipamentos!!\n')

#função para calculo fatorial de um numero:
def fatorial(numero):
    if numero < 0:
        return "digite um valor maior ou igual a zero"
    else:
        if numero==0 or numero==1:
            return 1
        else:
            return numero * fatorial(numero - 1)
#função para retornar uma serie de fibonacci até um valor x:
def fibo(n):
    resultado = []
    a, b = 0,1
    while b <= n:
        resultado.append(b)
        a,b = b, a+b
    return resultado
