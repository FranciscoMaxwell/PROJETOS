#manipulação de arquivos de texto

# manipulador =  open('arquivo.txt', 'w', encoding='utf-8')

# print(f'\Método read():\n')
# print(manipulador.read())

# print(f'\nMetodo readline():\n')
# print(manipulador.readline())
# print(manipulador.readline())

# print(f'\nmetodo readlines():\n')
# print(manipulador.readlines())
# texto = input('qual termo deseja procurar no arquivo?  ')
# try:
#     manipulador = open('C:\\Users\\Maxwell Fernandes\\OneDrive\\Documentos\\arquivo2.txt', 'r', encoding='utf-8')
#     for linha in manipulador:
#         linha = linha.rstrip()
#         if texto in linha:
#             print(f'a string foi encontrada!') 
#             print(linha)
#         else:
#             print(f'a string não foi encontrada!  ')
# except IOError:
#     print(f'Não foi possivel abrir o arquivo')
# else:
#     manipulador.close()


oleos = ['morango\n', 'uva\n', 'caju\n', 'graviola\n']
try:
    manipulador = open('oleos.dat', 'w', encoding='utf-8')
    manipulador.write('rapadura putarias\n')
    manipulador.write('como criar casas de shows??rpa')
    manipulador.writelines(oleos)
except IOError:
    print(f'Não foi possivel abrir o arquivo')
else:
    manipulador.close()

#Ler arquivo criado:
try:
    manipulador = open('oleos.dat', 'r', encoding='utf-8')
    print(manipulador.read())
except IOError:
    print(f'Não foi possivel abrir o arquivo')
else:
    manipulador.close()