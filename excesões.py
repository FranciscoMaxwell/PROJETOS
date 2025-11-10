# Exceção é um objeto que representa um erro que ocorreu ao executar o programa.
# Blocos try... except

# n1 = int(input('digite um numero: '))
# n2 = int(input('digite outro numero: '))

# try:
#     r = round(n1 / n2, 2)
# except ZeroDivisionError:
#     print(f'não é possivel dividir por zero')
# else:
#     print(f'Resultado: {r}')


def div(k, j):
    return round(k / j, 2)

if __name__ == '__main__':
    while True:
        try:
            n1 = int(input('digite um numero: '))
            n2 = int(input('digite outro numero: '))
            break
        except ValueError:
            print(f'Ocorreu um erro nessa incrivibilidade seu inteligente!!!')

    try:
        r = div(n1, n2)
    except ZeroDivisionError:
        print(f'Não é possivel dividir por zero!')
    except:
        print(f'Ocorreu um erro desconhecido.')
    else:
        print(f'Resultado: {r}')
    finally:
        print(f'\nFim do Cálculo')

