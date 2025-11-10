# num = 1

# while (num <= 10):
#     print(num)
#     num += 1
# print('laço encerrado')

nome = None

while True:
    print('digite seu nome!!')
    nome = input()
    if nome == 'x' or nome == 'X':
        break
    else:
        print('bem vindo', nome)
    

