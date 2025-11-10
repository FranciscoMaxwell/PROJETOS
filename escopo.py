#Escopo Global e local

var_global = ' curso completo de python'

def escreve_texto():
    global var_global
    var_global = 'Banco de rapadura'
    var_local = 'Senhor assasino'
    print(f'Variável Global: {var_global}')
    print(f'Variavel Local: {var_local}')

if __name__ == '__main__':
    print(f'executar a função escreve_texto()')
    escreve_texto()

    print('tentar acessar as variaveis diretamente')
    print(f'Variável Global: {var_global}')
    # print(f'Varivel global: {var_global}')
    # print(f'Variavel Local: {var_local}')