def fatorial(numero):
    if numero == 0 or numero == 1:
        return 1
    else:
        return numero * fatorial(numero -1)
    
if __name__ == '__main__':
    x = int(input('Digite um número para fatorar: '))
try:
    rapadura = fatorial(x)
except RecursionError:
    print(f'To cansado chefe...')
else:
    print(f' o fatorial de {x} é {rapadura} ')