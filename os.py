import os

os.chdir('C:\\teste')
print('diretorio atual: {os.getcwd()}')

teuboga = input('qual o padrão de nomes de arquivos a usar (sem a extensão)?')

for bloco, nor in enumerate(os.listdir()):
    if os.path.isfile(nor):
        bloco2, bloco3 = os.path.splitext(nor)
        bloco2 = teuboga + '' + str(bloco)

        nadahaver = f'{bloco2}{bloco3}'
        os.rename(nor, nadahaver)

print(f'\npqp renomiados nesta bagaça')

    
