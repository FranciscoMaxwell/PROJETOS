# nomes = [2, 4, 6, 8]
# raticat = 0
# for ralo in nomes:
#     raticat += ralo
    
# print(f'a soma de todos é, {raticat}', )


# #ou


# numbers = [2,4,6,8]
# print(sum(numbers))


# #Escreva um programa que peça ao usuário 5 números e, usando for, 
# #armazene esses números em uma lista. Depois, imprima a lista completa.


# lista = []

# for bibo in range(5):
#     print('digite um número ')
#     porn = input()
#     lista.append(porn)

# print(f'\n5 coisas escolhidas!!')
# for porn in lista:
#     print(f'{porn}')


# Crie uma lista de palavras ["gato", "cachorro", "passaro"] 
# e use um for para imprimir cada palavra em maiúsculas.


# lore = ['jato', 'ameixa', 'jorgu']

# for i in lore:
#     print(i.upper(), end=' ')


# Faça um programa que conte quantas vogais e
# xistem na palavra digitada pelo usuário usando um laço for.


# jor = input('digite sua palavra aqui!!!  ')
# lor = [i for i in jor if i in 'aeiou']
# print(len(lor))


# Peça ao usuário 5 números, guarde em uma lista e 
# depois use um for para mostrar somente os números pares digitados.


# mira = []
# for jor in range(5):
#     jor = int(input('digite um numero: '))
#     mira.append(jor)

# for min in mira:
#     if min %2 == 0:
#         print(min, end=' ')


# Imprima a tabuada de 1 a 10, usando dois laços for aninhados.


# mira = int(input('digite um numero: '))

# for lin in range(11):
#     print(f'{mira} x {lin} = {lin*mira}')


# ou


# for lin in range(1,11):
#     for lor in range(1,11):
#         print(f'{lin} x {lor} = {lin*lor}')
#     print('-' *50)


# Dada uma matriz (lista de listas), como [[1,2,3],[4,5,6],[7,8,9]], 
# use for para imprimir todos os elementos, linha por linha


# lore =  [[1,2,3],[4,5,6],[7,8,9]]
# for lin in lore:
#     for min in lin:
#         print(min, end=' ')
#     print()


#Como fazer a soma de todos os números com for


# lista1 = [[1,2,3],[4,5,6],[7,8,9]]
# mary = 0
# for role in lista1:
#     for char in role:
#         mary+=char
    
# print(mary)


#somando a diagonal 1,5,9


# lista1 = [[1,2,3],[4,5,6],[7,8,9]]
# mor = 0
# for mary in range(len(lista1)):
#     mor += lista1[mary][mary]
    
# print(f'a soma total é {mor}')


#somando diagonal invertida 3,5,7


# lista1 = [[1,2,3],[4,5,6],[7,8,9]]
# ro = 0
# for lin in range(len(lista1)):
#     ro += lista1[lin][len(lista1) - 1 - lin]

# print(ro)


# Crie um programa que peça uma frase e, usando for, substitua todas as 
# letras "a" por "*", imprimindo o resultado final.


# frase = input('digite sua frase aqui:   ')
# chor = frase.replace('a', '*').replace('A', '*')
    
# print(chor)


# Faça um programa que gere uma lista com os 
# quadrados dos números de 1 a 20 usando um for.


# ro = []
# for lo in range(1,21):
#     ma = lo**2
#     ro.append(ma)

# print(ro)


# ou


# ro = [x**2 for x in range(1,21)]
# print(ro)


# Crie um laço for que percorra os números de 1 a 50 e imprima:
# "Fizz" se for múltiplo de 3
# "Buzz" se for múltiplo de 5
# "FizzBuzz" se for múltiplo de 3 e 5
# O próprio número caso contrário.


# for mim in range(1,51):
#     if mim %5 == 0 and mim %3 ==0:
#         print(f'{mim} fizzbuzz')
#     elif mim %5 == 0:
#         print(f'{mim} buzz')
#     elif mim %3 == 0:
#         print(f'{mim} fizz')
#     else:
#         print(mim)


# Crie uma variável nome que receba seu nome, e 
# outra idade que receba sua idade. Em seguida, imprima a frase:


# nome = input('Digite seu nome aqui: ')
# idade = int(input('Digite sua idade aqui: '))
# print(f'como vai {nome}, a sua idade é {idade} anos!!')


# Crie três variáveis com números inteiros e calcule 
# a média deles.


# numero1 = int(input('Digite o primeiro número:  '))
# numero2 = int(input('Digite o segundo número:  '))
# numero3 = int(input('Digite o terceiro número:  '))
# soma = numero1+numero2+numero3
# print(soma / 3)


#ou


# nv = []
# for lor in range(5):
#     lor = int(input('digite um numero '))
#     nv.append(lor)

# sh = sum(nv) / len(nv)
# print(sh)


# Peça ao usuário dois números e imprima: 
# soma, subtração, multiplicação, divisão e resto da divisão.


# numeros = []

# for char in range(2):
#     char = int(input('Digite seu número: '))
#     numeros.append(char)

# num1, num2 = numeros

# print(f'A Soma deles: {num1+num2}')
# print(f'A Divisão deles: {num1/num2}')
# print(f'A Subtração deles: {num1-num2}')
# print(f'A multiplicação deles: {num1*num2}')
# print(f'O resto deles: {num1%num2}')


# Um carro percorre 240 km em 4 horas. Calcule a velocidade média.


# distancia = int(input('Quantos km percorreu o carro?: '))
# tempo = int(input('Quantas horas percorreu o carro?: '))

# velocidade_media = distancia / tempo
# print(f'Média de velocidade: {velocidade_media} km/h')


# Peça ao usuário dois números e 
# diga qual é maior, ou se são iguais.


# num1 = int(input('Digite seu primeiro numero: '))
# num2 = int(input('Digite seu segundo número: '))

# if num1 > num2:
#     print('Número 1 é maior que o número 2: ')
# elif num1 == num2:
#     print('Número 1 é igual ao número 2: ')
# else:
#     num1 < num2
#     print('Número 1 é menor que o número 2')


# Verifique se um número dado pelo usuário é par e positivo.


# num = int(input('Digite um número: '))

# if num %2 == 0 and num > 0:
#     print('O número é positivo e par!!')
# elif num > 0:
#     print('O número é positivo')
# elif num == 0:
#     print('O número é zero (neutro, nem positivo nem negativo).')
# else: 
#     print('O número é negativo')


# Pergunte ao usuário se ele possui carteira de motorista 
# (sim/não) e idade. Determine se ele pode dirigir 
# legalmente (idade ≥ 18 e carteira válida).


# try:
#     idade = int(input('Qual a sua idade?: '))
#     carteira = str(input('Você possui carteira? responda(sim/não): ')).lower()

#     if carteira not in('sim' or 'não'):
#         print('Digite apenas sim ou não!!')
    
#     if idade >= 18 and carteira == 'sim':
#         print('Motorista Habilitado')
#     elif idade >= 18 and carteira == 'não':
#         print('Não pode dirigir, mas pode tirar carteira')
#     else:
#         idade < 18
#         print('Não pode dirigir e nem tirar carteira')

# except ValueError:
#     print(f'Coloque direito!!')


# Determine se um número é divisível por 3 ou 5.


# numero = int(input('Digite um número: '))

# if numero % 3 == 0 and numero % 5 == 0:
#     print(f'O número é divisivel por 3 e por 5!')
# elif numero % 3 == 0:
#     print(f'O número é divisivel por 3!')
# elif numero % 5 == 0:
#     print(f'O número é divisivel por 5!')
# else:
#     print(f'o número não é divisivel nem por 3 ou por 5!')


# Peça uma nota de 0 a 10 e imprima a situação:
# 0–4 → "Reprovado"
# 5–6 → "Recuperação"
# 7–10 → "Aprovado"


# nota1 = float(input('Digite sua primeira nota semestral: '))
# nota2 = float(input('Digite sua segunda nota semestral: '))
# media = (nota1+nota2) / 2

# if media >= 7:
#     print('Você foi aprovado. ')
# elif media >= 5:
#     print('Você foi para a recuperação. ')
# else:
#     print('Você foi reprovado. ')


# Imprima uma tabela de preços com pelo menos 3 produtos usando f-strings.


# nome = []
# preço = []
# print('Digite o NOME e o PREÇO logo em seguida.')
# for rage in range(5):
#     nome_do_produto = str(input(f'Digite o nome do produto: '))
#     preço_do_produto = float(input(f'Digite o preço do produto: '))
#     nome.append(nome_do_produto)
#     preço.append(preço_do_produto)


# print('\n-------TABELA DE PREÇOS-------')
# for num1, num2 in zip(nome,preço):  #zip() combina com duas listas ou mais em pares
#     print(f'{num1:<15} R$ {num2:>7.2f}')


# Faça o mesmo usando format().


# nome = []
# preço = []
# print('Adicione o preço logo após por o nome do produto.')
# for ragen in range(5):
#     nome_p = str(input('Digite aqui o nome do produto: '))
#     preço_p = float(input('Digite aqui o preço do produto: '))
#     nome.append(nome_p)
#     preço.append(preço_p)

# print('======TABELA DE PREÇOS======')
# for naur, paur in zip(nome,preço):
#     print('{:<15} R$ {:>7.2f}'.format(naur,paur))


# Faça um programa que peça números até o usuário digitar 0. 
# Imprima a soma de todos os números digitados.

#SOMA
# soma = 0
# while True:
#     numbers = int(input('Digite quantos numeros quiser '))
#     if numbers == 0:
#         break
#     soma += numbers


# print(f'A soma de todos é = {soma}')


#MULTIPLICAÇÃO
# soma = 1
# while True:
#     numbers = int(input('Digite quantos numeros quiser '))
#     if numbers == 0:
#         break
#     soma *= numbers


# print(f'A soma de todos é = {soma}')


#DIVISÃO
# numero1 = float(input('Digite um número para ser divido: '))
# divisor = numero1

# while True:
#     numbers = float(input('Digite quantos numeros quiser '))
#     if numbers == 0:
#         break
#     divisor /= numbers


# print(f'A soma de todos é = {divisor}')


# SUBTRAÇÃO
# Crie um menu que fique em loop até o usuário escolher a opção "sair".


# sub = float(input('digite número para ser subtraido:   '))
# while True:
#     numbers = input('Digite quantos numeros quiser ')
#     if numbers.lower() == 'sair':
#         break
#     else:
#         sub -= float(numbers)

# print(f'oque sobrou é = {sub}')


# Imprima os números de 1 a 50 que sejam múltiplos de 7.


# for tlep in range(1, 51):
#     if tlep % 7 == 0:
#         print(f'o {tlep} é multiplo de 7 ')


# Gere uma tabuada completa de um número informado pelo usuário.


# urubu = int(input('Digite o numero para a tabuada: '))

# print(f'====TABUADA DO {urubu}====')
# for puta in range(1,21):
#     print(f'{urubu} x {puta} = {urubu*puta}')


# Dada uma matriz [[1,2,3],[4,5,6],[7,8,9]], 
# imprima todos os elementos.


# lista = [[1,2,3],[4,5,6],[7,8,9]]
# for lin in lista:
#     print('  '.join(str(jar) for jar in lin))


# Calcule a soma de todos os elementos dessa matriz.

# lista = [[1,2,3], [4,5,6,], [7,8,9]]
# soma = 0
# for loro in lista:
#     for ruby in loro:
#         soma += ruby
    
# print(soma)

# #ou

# lista = [[1,2,3], [4,5,6], [7,8,9]]
# soma = sum(sum(linha) for linha in lista)
# print(soma)

# ou


# lista = [[1,2,3], [4,5,6], [7,8,9]]
# soma = sum(na for linha in lista for na in linha)
# print(soma)


# algo


# import requests

# url = input('digite seu site como (https://www.google.com):  ')

# try:
#     lar = requests.get(url)
#     if lar.status_code == 200:
#         print(f' o site {url} está online! (status {lar.status_code})')
#     else:
#         print(f'o site {url} não responde, esta com status {lar.status_code}')
# except requests.exceptions.RequestException as ro:
#     print(f'Não foi possivel verificar site {url}, erro: {ro}')


# Gere um número aleatório entre 1 e 100 
# e peça para o usuário adivinhar até acertar.


# import random
# print('Acerte o número aleatorio de 1 a 100 para parar o programa.')
# while True:
#     for lorena in range(1):
#         tor = random.randint(1,101)
#         tor = int(input('escolha um número de 1 a 100 (Acerte o):'))
#     if tor == 51:
#         break
    
# print(f'Parabens você acertou, era {tor}') 


#ou


# import random
# print('Acerte o número de 1 a 100 para parar o programa!!!')
# tor = random.randint(1,101)
# while True:
#     hor = int(input('Escolha seu número:  '))
#     if hor == 51:
#         print(f'Parabéns! Você acertou, o número era {hor} 🎉')
#         break
#     elif hor < 51:
#         print('O número é maior')
#     else:
#         print('O número é menor')


# Simule o lançamento de dois 
# dados 10 vezes e imprima os resultados.


# import random
# print(F'lançando os dois dados 10 vezes...')
# for yor in range(10):
#     tor = random.randint(1,6)
#     tor2 = random.randint(1,6)
#     print(f'Jogada {yor+1}: Dado 1 = {tor} Dado 2 = {tor2} Soma = {tor+tor2}')


#or:


# import random
# print(f'Lançando os dois dado 10 vezes...')
# for uo in enumerate(range(10), start=1):
#     tor1 = random.randint(1,6)
#     tor2 = random.randint(1,6)
#     print(f'Jogada {uo} Dado 1 = {tor1} e Dado 2 = {tor2} Soma = {tor1+tor2}')


# Crie uma lista de 5 frutas e imprima cada uma usando um for.


# lito = ['abacaxi', 'maça', 'jabuticaba', 'uhuumelão', 'abacate']
# for rogue in lito:
#     print(rogue)


# Adicione uma fruta nova no final da lista e outra no início.


# lito = ['abacaxi', 'maça', 'jabuticaba', 'uhuumelão', 'abacate']
# lito.append('amendoa')
# lito.insert(0,'acerola')
# print(lito)


#Remova a fruta do meio e imprima a lista atualizada.


# lito = ['abacaxi', 'maça', 'jabuticaba', 'uhuumelão', 'abacate']
# lito.append('amendoa')
# lito.insert(0,'acerola')
# lito.remove('jabuticaba')
# print(lito)


#ou


# lore = ['abacate', ' amendoa', 'gerimum', 'ratatuille', ' abarca']
# lore.append('jumento')
# lore.insert(0, 'loucura')
# lore.pop(len(lore)//2)
# print(lore)


# Crie uma tupla com 5 cores e imprima a primeira e a última.


# tur = ('azul', 'amarelo', 'vermelho', 'rosa', 'cinza')
# print('primeira cor', tur[0])
# print('segunda cor', tur[-1])


# Tente adicionar um elemento à tupla (o que acontece?).

# tur = ('azul', 'amarelo', 'vermelho', 'rosa', 'cinza')
# mar = list(tur)
# mar.append('grana')
# mar = tuple(mar) #fazendo voltar a ser tupla com o novo valor
# print(mar)


# Calcule o valor absoluto, arredondamento, 
# raiz quadrada e seno de um número.


# import math
# a = 36.20
# tor1 = math.sqrt(a)
# tor = abs(a)
# tor2 = round(a)
# tor3 = math.radians(a)
# tor4 = math.sin(tor3)
# print(f'absoluto: {tor}  raiz quadrada: {tor1:.2f}  arren: {tor2}  seno: {tor4:.2f}')



# # Peça ao usuário uma frase e conte quantas letras 
# "a" existem nela.


# usu = input('Escreva uma frase:  ')
# mar = usu.count('a') + usu.count('A')
# print(f'A letra "a" aparece {mar} vezes na frase')


# Inverta a frase digitada pelo usuário.


# usu = input('Escreva uma frase: ')
# lor = usu[::-1]
# print(lor)


# Crie um dicionário com 3 pessoas 
# (nome, idade) e imprima todas as chaves e valores.


# dic= {'bruno': 30,
#       'mathrus': 23,
#       'bruns': 450
# }

# for tho, tha in dic.items():
#     print(f'Nome: {tho}   Idade: {tha}')


# Atualize a idade de uma pessoa e remova outra do dicionário.


# di = {'breno': 30,
#       'matheus': 40,
#       'rodrigo': 50
# }
# di['breno'] = 45
# del di['rodrigo']
# for tho, tha in di.items():
#     print(f'Nome:  {tho}   Idade:  {tha}')


# Crie dois sets com alguns números 
# e calcule: união, interseção e diferença.


# se1 = {1,2,3,4,5,6, 11,23}
# se2 = {11,23,57, 15, 6}

# jar = (se1.intersection(se2))
# char = (se1 | se2)
# meriu = (se1 ^ se2)
# choque = (se1 - se2)

# print(jar, char, meriu, choque)


# Crie uma função que receba dois números e retorne a soma.


# xa =5
# ye =7
# def soma(a,b):
#     return a + b

# jor = soma(xa,ye)
# print(jor)


# sem return:


# def doma(a,b):
#     print(a+b)

# doma(5,7)


# Crie uma função que calcule o fatorial de um número usando for.


# shock = int(input('Escreve um número para fatorar: '))
# lu = 1
# for rake in range(1, shock+1):
#     lu *= rake
    
# print(lu)


#com função


# def seuboga(nao):
#     lor = 1
#     for rake in range(1, nao+1):
#         lor *= rake
#     return lor

# shock = int(input('Escreve um número para fatorar: '))
# print(f'O fatorial é {seuboga(shock)}')


# Faça uma função que imprime uma saudação, recebendo nome 
# obrigatório e mensagem opcional ("Olá" por padrão).


# def shorck():
#     return shock

# shock = str(input('Digite seu nome:  '))
# print(f'Olá, {shock}')


#ou


# def shorck(n, m= 'Olá'):
#     print(f'{m}. {n}!')

# l = input('Digite seu nome aqui: ')
# shorck(l)
# shorck(l, 'Sejá Bem-vindo')


# Demonstre uma variável global 
# e outra local dentro de uma função.


# var = 'core'

# def lordinho():
#     vor = 'care'
#     print(f'Variavel local é {vor}')
    

# print(f'Variavel global é {var}')
# lordinho()


#Peça para o usuário 
# digitar um número e trate erros caso ele digite texto.


# try:
#     usu = int(input('Digite um número:  '))
#     print(usu)
# except ValueError:
#     print('Digite apenas numeros!!')


# Faça uma função recursiva para calcular fatorial.


# def lorfi(um):
#     if um == 0 or um == 1:
#         return 1
#     else:
#         return um*lorfi (um -1)


# nu = int(input('Digite um número: '))
# print(lorfi(nu))


# Faça uma função recursiva para 
# somar todos os números de 1 até n.


# def lorfi(um):
#     if um == 0:
#         return 0
#     else:
#         return um+lorfi (um -1)
    
# nu = int(input('Digite um número: '))
# print(lorfi(nu))


# Use lambda + map para elevar ao quadrado todos os 
# números de uma lista.


# lore = [2,3,4,5,6,7,8,9,10]
# roque = list(map(lambda x: x**2, lore))
# print(roque)


# Use filter para selecionar 
# apenas números pares de uma lista.


# lore = [1,2,3,4,5,6,7,8,9,10]
# roque = list(filter(lambda x:x %2 == 0, lore))
# print(roque)


# Use reduce para 
# multiplicar todos os elementos de uma lista.


# from functools import reduce

# lore = [1,2,3,4,5,6,7,8,9,10]
# loki = reduce(lambda x, y: x*y, lore)
# print(f'A multiplicação da lista é: {loki}')


# Crie uma lista com os quadrados
# de 1 a 20 usando list comprehension.


# num = [mae**7 for mae in range(1,21)]
# print(num)


# Crie uma lista com as 
# palavras de uma frase que tenham mais de 3 letras


# lu = input('digite uma frase: ')
# lar = lu.split()
# lore = [ma for ma in lar if len(ma) > 3]
# print(lore)


# Crie uma classe Pessoa com atributos 
# nome e idade, e método cumprimentar().


# class Pessoa():
#     def __init__(self, nome, idade):
#         self.__nome = nome
#         self.__idade = idade

#     def cumpri(self):
#         print(f'O meu nomé é {self.__nome} e minha idade é {self.__idade}')

#     def get(self):
#         return f'Nome: {self.__nome} Idade: {self.__idade}'

# jar = Pessoa('Gabi', 26)
# jar.cumpri()
# print(jar.get())


# Crie uma classe Aluno que herda 
# de Pessoa e adiciona atributo curso.


# class pessoa:
#     def __init__(self, idade, nome,):
#         self.__nome = nome
#         self.__idade = idade
    
#     def cumpri(self):
#         print(f'Olá meu nome é {self.__nome} e minha idade é {self.__idade} anos ')

#     def getnomeidade(self):
#         return f'Nome: {self.__nome}, idade: {self.__idade}'

# jar = pessoa(24, 'JOANA')
# jar.cumpri()
# print(jar.getnomeidade())

# class Aluno(pessoa):
#     def __init__(self, idade,nome, curso):
#         super().__init__(idade,nome)
#         self.__curso = curso

#     def cumpri(self):
#         print(f'Sou um aluno novo e faço {self.__curso}')

# mar = Aluno(18, 'PRISCILLA', 'curso de putaria')
# mar.cumpri()
# print(mar.getnomeidade())


# Abra o arquivo e imprima cada linha com um for.


# haram = open('arquivo2.txt', 'r', encoding = 'utf-8')
# print(haram.read())


# Use enumerate para imprimir 
# índice e valor de uma lista de nomes.


# girls = ['bunda', 'peitos', 'peitão', 'buceta', 'rabão']
# for g, ass in enumerate(girls):
#     print(f'Numeros: {g}, parte: {ass}')


# Abra um arquivo usando with e leia seu conteúdo.


# try:
#     with open('oleos.dat', 'r', encoding = 'utf-8') as mo:
#         print(mo.read())
# except IOError:
#     print(' Não foi possivel abrir este arquivo')
    

# Crie variáveis para armazenar nome, idade 
# e altura de uma pessoa e exiba esses valores.


# nome = 'jaca'
# idade = 24
# altura = 1.80

# print(f'NOME: {nome} IDADE: {idade} ALTURA: {altura}')


# ----------------------------------#Converta uma temperatura em Celsius para Fahrenheit e Kelvin.


# chop = lambda oi: oi * 1.8+32 #celsius para fahrenheit
# tor = lambda toi: toi + 273.15 #celsius para kelvin

# celsius = 1200
# print(f'{celsius}°C É IGUAL A {chop(celsius)} FAHRENHEITH')
# print(f'{celsius}°C É IGUAL A {tor(celsius)} KELVIN')


# ----------------------------------#Crie um sistema de cadastro de produto com 
# # nome (str), preço (float), quantidade (int) e disponível 
# # (bool), validando os tipos automaticamente e exibindo em JSON.


# import json

# nome = str(input('digite o nome do produto: '))

# try:
#     while True:
#         preco = float(input('escreva o preço do produto: '))
#         break
# except ValueError:
#     print('digite um valor valido: ')

# try:
#     while True:
#         quantidade = int(input('digite um valor valido: '))
#         break
# except ValueError:
#     print('digite um valor valido: ')

# disponivel = quantidade > 0

# dic = {
#     'nome': nome,
#     'preco': preco,
#     'quantidade': quantidade,
#     'disponivel': disponivel
# }
# print('\nProdutos cadastrados')
# print(json.dumps(dic, indent=4, ensure_ascii=False))


# Peça dois números e mostre todas as operações básicas.


# while True:
#         try:
#             num1=int(input('Digite seu primeiro número: '))
#             break
#         except ValueError:
#               print('Digite um valor valido!')

# while True:
#         try:        
#             num2=int(input('Digite seu segundo número: '))
#             break
#         except ValueError:
#               print('Digite um valor valido!')

# char = (f'SOMA: {num1+num2} // MULTI: {num1*num2} // DIV: {num1/num2} // SUB: {num1-num2}') 

# print(char)


#OU


# while True:
#     try:
#         num1 = int(input('Digite seu primeiro numero: '))
#         break
#     except:
#         print('Digite um valor valido!')

# while True:
#     try:
#         num2 = int(input('Digite seu segundo número: '))
#         break
#     except:
#         print('Digite um valor valido!')

# soma = num1+num2
# sub = num1-num2
# multi = num1*num2
# div = num1/num2

# if num2 != 0:
#     div = num1/num2
# else:
#     ('Seu número não pode ser dividido por 0!')

# print(f"""
#      SOMATORIA: {soma}
#      MULTIPLICAÇÃO: {multi}
#      SUBTRAÇÃO: {sub}
#      DIVISÃO: {div}    
# """)


# Faça uma calculadora que suporte potência e raiz quadrada


# import math
# print(f'Coloque o Número que será repetido')
# potencia = int(input('Digite um número para potenciação:  '))

# print(f'Quantas vezes este número será repetido? ')
# quadrada = int(input('Digite um número para a raiz:  '))

# char = potencia**quadrada
# print(f'Este é o resultado da potenciação {char}')


# chor = math.sqrt(quadrada)
# print(chor)


#OU


# import math

# print('====CALCULADORA====')

# num1 = int(input('digite a base para a potenciação: '))
# num2 = int(input('digite uo expoente: '))

# expo = num1**num2
# print(f'a exponenciação dos numeros {num1}^{num2} é {expo}')

# raiq = float(input('digite um numero para a raiz quadrada: '))
# if raiq >= 0:
#     lor = math.sqrt(raiq)
#     print(f'a raiz do numero {raiq} é ({lor})')
# else:
#     print('numeros negativos não possuem raiz')


# Desenvolva um avaliador de expressão matemática (ex.: "3 + 5 * 2") sem eval().


# import ast
# import operator

# sub = {
# ast.Mod: operator.mod,
# ast.Pow: operator.pow,
# ast.Div: operator.truediv,
# ast.Sub: operator.sub,
# ast.Add: operator.add,
# ast.Mult: operator.mul
# }

# class avalir(ast.NodeVisitor):
#     def visit_BinOp(self, node):
#         left = self.visit(node.left)
#         right = self.visit(node.right)
#         return sub[type(node.op)](left,right)
    
#     def visit_Constant(self, node):
#         return node.value
    
# def vali_exp(exor):
#     tor = ast.parse(exor, mode='eval')
#     return avalir().visit(tor.body)

# print(vali_exp('3+5'))


#OU ESTILO MANUAL


# def anal(expq):
#     return alam_l(expq.split())

# def alam_l(mira):
#     while '**' in mira:
#         for lin, kan in enumerate(mira):
#             if kan == '**':
#                 ko = float(mira[lin-1]) ** float(mira[lin+1])
#                 mira[lin-1:lin+2] = [ko]
#                 break

#     while '*' in mira or "/" in mira:
#         for lin, kan in enumerate(mira):
#             if kan == '*':
#                 ko = float(mira[lin-1]) * float(mira[lin+1])
#                 mira[lin-1:lin+2] = [ko]
#                 break
#             elif kan == "/":
#                 ko = float(mira[lin-1]) / float(mira[lin+1])
#                 mira[lin-1:lin+2] = [ko]
#                 break

#     lort = float(mira[0])
#     lin = 1
#     while lin<len(mira):
#         if mira[lin] == '+':
#             lort += float(mira[lin+1])
#         elif mira[lin] == '-':
#             lort -= float(mira[lin+1])
#         lin+=2
#     return lort

# print(anal('2 + 5 ** 2'))


# Informe se um número é positivo, negativo ou zero.


# num = int(input('Digite um número: '))

# if num > 0:
#     print('numero positivo')
# elif num == 0:
#     print('é um 0')
# else:
#     print('numero negativo')


# # Sistema de notas: aprovado, recuperação ou reprovado.


# num1 = float(input('Digite um número: '))
# num2 = float(input('Digite um número: '))

# media = num1+num2 
# hot = media / 2


# if hot >= 7:
#     print('você passou')
# elif hot >= 4:
#     print('recuperação')
# else:
#     print('reprovado')


# # Sistema de login com até 3 tentativas antes de bloquear.


# usua = 'aerio'
# senha = 1237

# n = 0
# t = 3

# while n < t:
#     usu = input('digite seu usuario:  ')
#     sen = int(input('digite sua senha:  '))

#     if usu == usua and sen == senha:
#         print('logado com sucesso')
#         break
#     else:
#         n +=1
#         print(f'tente de novo {n}/{t}')

#         if n == t:
#             print('esgotou chances')
            

# #Tabuada de um número.


# hot = int(input('Digite um número para a tabuada: '))

# print(f'tabuada do {hot}')
# for cor in range(1,11):
#     print(f'{hot} X {cor} = {hot*cor}')


# Some números digitados até o usuário entrar com 0.


# lo = 0

# try:
#     while True:
#         nu = int(input('digite um numero até cansar: '))
#         if nu == 0:
#             print('cabous')
#             break
#         else:
#             lo += nu
#             print(f'a soma até o momento {lo}')
# except ValueError:
#     print('tente denovo')


# Jogo de adivinhação (1 a 100), com dicas e limite de tentativas.


# import random

# print('====adivinhe o numero de 1 a 100====')
# print('você tem 10 chances')

# nua = random.randint(1,101)

# nu = 0
# lu = 10

# while True:
#     try:
#         chu = int(input('digite o numero que acha: '))
#     except ValueError:
#         print('um numerooooo cara')

#     if not 1 <= chu <=100:
#         print('fora de intervalo, digite um numero dentro do limite de 1 a 100')
#         continue

#     nu +=1

#     if chu == nua:
#         print(f'você acertou parabens em {nu} de {lu}')
#     elif chu < nua:
#         print(f'suba mais, ta frio, você ainda tem {nu} de {lu}')
#     else:
#         print(f'desça mais, apenas desça {nu} de {lu}')

#         if nu == lu:
#             print(f'acabou suas chances {nu} de {lu}')
#             break


# Crie uma lista de 5 frutas e exiba a segunda e a última.


# frutas = ['abacaxi', 'xana', 'maça', 'roma', 'banana']
# print(frutas[1], frutas[4])


# # Dicionário com nome, notas, média e situação de um aluno.


# dicionario = {
# 'notas': [7,7,7],
# 'media': 7.0,
# 'situacao': 'aprovado'
# }

# print(f'notas: {dicionario["notas"]}')
# print(f'media: {dicionario["media"]}')
# print(f'situação: {dicionario["situacao"]}')


# OU MELHORADO


# doc = {
#     'nome': 'armando',
#     'notas': [8, 8.4, 5.3]
# }

# doc['media'] = sum(doc['notas']) / len(doc['notas'])

# if doc['media'] > 7:
#     doc['situacao'] = 'aprovado'
# elif doc['media'] >= 4:
#     doc['situacao'] = 'está de recuperação'
# else:
#     doc['situacao'] = 'reprovado'

# print(f"Nome: {doc['nome']}")
# print(f"Notas: {doc['notas']}")
# print(f"Media: {doc['media']}")
# print(f"Situação: {doc['situacao']}")


# Programa que conta quantas vezes cada palavra aparece em um texto.


# noa ="""
# 'Vai Descendo' é, portanto, uma celebração da dança, da música e da 
# cultura das festas de rua, onde a comunidade se reúne 
# para se divertir e expressar sua alegria através do movimento e da música
# """.lower()

# char = input('digite qual palavra quer saber quantas vezes aparece: ')
# lor = noa.count(char)

# print(f'a palavra "{char}" aparece {lor} vez(es)')


# Receba uma lista de palavras e retorne apenas 
# as únicas, ordenadas alfabeticamente.


# lista = ['ovo', 'amora', 'jubilubi', 'baitola', 'amora', 'ovo']
# lore = set(lista)
# lare = sorted(lore)
# print(f'aqui alfabeticamente: {lare}')


# OU:


# lista = ['ovo', 'amora', 'jubilubi', 'baitola', 'amora', 'ovo']
# ol = sorted({char for char in lista if char.isalpha()})
# print(ol)


# Função que retorna o dobro de um número.


# def ado(num):
#     return num*2

# nume = int(input('digite um numero: '))
# resultado = ado(nume)
# print(resultado)


# Função recursiva que calcula o fatorial:


# def cor(nar):
#     if nar <= 1:
#         return 1
#     else:
#         return nar * cor (nar-1)
    
# jar = int(input('digite um numero: '))
# result = cor(jar)
# print(result)


# Função que aceita parâmetros ilimitados, 
# filtra apenas inteiros e retorna a soma.


# def lan(*args):
#     lon = [do for do in args if isinstance(do, int)]
#     return sum(lon)

# jarq = lan(1,2,3,4,5,6.7,'@#$%','amendoas')
# print(f'numeros inteiros apenas somados: {jarq}')


# Função recursiva que resolve o problema da Torre de Hanói.


# def hanoi(n,origem, fim, auxilio):
#     if n== 1:

#         print(f'mova o disco 1 de {origem} ao {fim}')
#     else:
#         # Move n-1 discos de origem para auxiliar
#         hanoi(n-1,origem, auxilio, fim)
#         # Move o disco restante de origem para destino
#         print(f'mova o disco {n} de {origem} ao {fim}')
#          # Move os n-1 discos do auxiliar para destino
#         hanoi(n-1, auxilio, fim, origem)

# numerodedisco = int(input('Digite o número de discos: '))
# hanoi(numerodedisco, 'A', 'C', 'B')


#OU


# def jo(j,o,f,a,r):
#     if j == 1:
#         m = r[o].pop()
#         r[f].append(m)
#         print(f'o {m} foi de {o} a {f}')
#     else:
#         jo(j-1,o,a,f,r)
#         jo(1,o,f,a,r)
#         jo(j-1,a,f,o,r)

# no = 3
# r = {
#     'A': list(range(no, 0,-1)),
#     'B': [],
#     'C': []
# }

# print(no, 'A', 'B', 'C', r)
# jo(no,'A', 'C', 'B', r)
# print('Estado final: ', r)


# Tratamento de divisão por zero.


# try:
#     no4 = float(input('digite o dividendo: '))
#     no3 = float(input('digite o divisor: '))
#     o = no4/no3
#     print(o)

# except ValueError:
#     print('coloque um numero')
# except ZeroDivisionError:
#     print('o divisor não pode ser zero')


# Leia um número do usuário e trate entradas inválidas (strings, vazio etc.).


# try:
#     num1 = float(input('diga um numero para dividir '))
#     print(num1)
# except ValueError:
#     print('apenas numeros porfavor')


# Função que tenta abrir um arquivo e mostra mensagem clara caso não exista.


# def jor(arquivo):
#     try:
#         with open('arquivo.txt', 'r', encoding='utf-8') as ror:
#             jar = ror.read()
#             print('Coisas do arquivo')
#             print(jar)
#     except FileNotFoundError:
#         print(f'o arquivo {arquivo} não foi encontrado')
#     except Exception:
#         print('Nada com o nome foi encontrado')

# ou = input('digite o nome do arquivo: ')
# jor(ou)


# Sistema de login que trata erros de entrada e bloqueio.


# usuario = 'desodorante'
# senha = '157'

# usu = input('digite usuario: ')
# sen = input('digite senha: ')

# if usu == usuario and sen == senha:
#     print('login realizado com sucesso')
# else:
#     print('não foi possivel realizar login')


# Crie uma calculadora robusta que trate erros de sintaxe e operações inválidas.


# def lan():
#     print('calculadora')

#     while True:
#         io = input('>>>>>>  ').strip()
#         if io.lower() == 'sair':
#             print('estamos saindo')
#             break

#         try:
#             resu = eval(io,{'__builtins__': None}, {'abs': abs, 'pow':pow, 'round':round})
#             print(f'o resultado é {resu}')
#         except ZeroDivisionError:
#             print('não pode ser dividido por zero')
#         except SyntaxError:
#             print('um numero ou simbolo valido')
#         except ValueError:
#             print('um numero ou simbolo valido')
#         except Exception as e:
#             print(f'o erro foi {e}')

# if __name__ == '__main__':
#     lan()


# Use lambda para calcular o quadrado de um número.


# num2 = lambda x:x**2
# num1 = float(input('>>>>>  '))
# print(f'{num2(num1)}')


# Dada uma lista de idades, use filter para obter apenas maiores de 18.


# lista = [1,7,9,10,18,19]
# no = list(filter(lambda xo: xo >= 18, lista))
# print(no)


# Use map para transformar uma lista de Celsius em Fahrenheit.


# lista = [20, 40, 50, 103]
# cpf = list(map(lambda x:x*1.8+32, lista))
# print(f'celsius para fahrenheit: {cpf}')


# Use reduce para calcular o produto de todos os elementos de uma lista.


# from functools import reduce
# lista = [1,2,3,5,6,7,8,9,11]
# num1 = reduce(lambda x,y:x*y, lista)
# print(num1)


# Combine map, filter e reduce para analisar uma lista de transações financeiras 
# (positivas e negativas) 
# e gerar: total de depósitos, total de retiradas e saldo final.


# from functools import reduce

# o = [10000,-1405, 60401, -70304, 23400]

# l =  list(map(float, o))

# n = filter(lambda f:f < 0, l)
# m = filter(lambda f:f > 0, l)

# g = reduce(lambda f,i: f+i, n)
# h = reduce(lambda f,i: f+i, m)

# print(f'Positivo somado: {h}')
# print(f'Negativo somado: {g}')
# print(f'Valor descontado: {g+h}')


# Crie uma lista com os quadrados de 1 a 10 usando list comprehension.


# list = [1,2,3,4,5,6,7,8,9,10]
# nor = [f**2 for f in list]
# print(nor)


# Gere uma lista apenas com números pares de 0 a 50.


# list = [chor for chor in range(0,51) if chor %2 == 0]
# print(list)


# Transforme uma lista de strings em uma lista apenas das que começam com vogal.


# listo = ['IODO','charizar', 'macaco', 'mariposa', 'amendoim', 'aranha']
# lo = [d for d in listo if d[0] in 'AEIOUaeiou']
# print(lo)


# Crie um dicionário via dict comprehension que mapeie cada 
# palavra de um texto ao seu número de ocorrências.


# lore =' a rapadura foi despedaçada por um bicho que foi a opera'
# shakra = lore.split()
# kan = {mira: shakra.count(mira) for mira in shakra}
# print(kan)


#  Classe Carro com marca e ano.


# class car():
#     def __init__(self, marca, ano):
#         self.__marca = marca
#         self.__ano = ano

#     def lor(self, eo):
#         self.__eo = eo
#         return f'{self.__marca} e {self.__ano} e {self.__eo}'

# io = car('tyotef', 1002)
# print(io.lor(5))


#ou publico


# class car():
#     def __init__(self, marca, ano):
#         self.marca = marca
#         self.ano = ano

# lor = car('toyut', 2004)
# print(lor.marca)
# print(lor.ano)


# Classe ContaBancaria com depósito, saque e saldo. 


# class conta:
#     def __init__(self,vi = 0):
#         self.__vi = vi

#     def depo(self, dep):
#         if dep > 0:
#             self.__vi += dep
#             print(f'foi adicionado {dep} com sucesso')
#         else:
#             print('valor invalido')

#     def sake(self, sal):
#         if sal > self.__vi:
#             print(f'tem {self.__vi} saldo insuficiente')
#         elif sal <= 0:
#             ('valor invalido')
#         else:
#             self.__vi -= sal
#             print(f'{sal} retirado com sucesso')

#     def mos(self):
#         print(f'saldo atual {self.__vi}')
#         return self.__vi

# i = conta(30000)
# i.depo(3000)
# i.mos()
# i.sake(7500)
# i.mos()


# Classe Funcionario com subclasses Gerente e Desenvolvedor com bônus diferentes.


# class funcionario():
#     def __init__(self, clt):
#         self.__clt = clt

#     def mo(self):
#         return f'SALARIO: {self.__clt}'
    
# class gerente(funcionario):
#     def __init__(self,clt,sardinha):
#         self.__sardinha = sardinha
#         super().__init__(clt)

#     def mo(self):
#         print('\ngerente: ')
#         return f'GERENTE {self.__sardinha} e {super().mo()}'

# class desenvolvedor(funcionario):
#     def __init__(self,clt, caro):
#         self.__caro = caro
#         super().__init__(clt)

#     def mo(self):
#         print('\ndesenvolvedor: ')
#         return f'DESENVOLVEDOR DE {self.__caro} e {super().mo()}'
    
# Loo = funcionario('SALARIO DE 45.000')
# mar = gerente('50.000','FINANCEIRO')
# je = desenvolvedor('400.000', 'DEV PYTHON')
# print(Loo.mo())
# print(mar.mo())
# print(je.mo())


#OU JEITO MAIS CERTO----


# class func():
#     def __init__(self, funci):
#         self._funci = funci

#     def movi(self):
#         return f'Eu sou um {self._funci}'
    
# class gerente(func):
#     def __init__(self, funci, B_gen):
#         self._B_gen = B_gen
#         super().__init__(funci)

#     def movie(self):
#         return f'{self.movi()} e recebo {self._B_gen}'

# class desen(func):
#     def __init__(self, funci, C_de):
#         self._C_de = C_de
#         super().__init__(funci)

#     def movie(self):
#         return f'{self.movi()} e recebo {self._C_de}'
    
# lo = func('funcionario')
# la = gerente('super funcionario', '10.800 R$')
# le = desen('master funcionario', '28.000 R$')
# print(lo.movi())
# print(la.movie())
# print(le.movie())


# Classe Biblioteca que gerencia livros (adicionar, remover, listar).


# class biblio:
#     def __init__(self,biblior = 0):
#         self._biblio = biblior

#     def moe(self):
#         return f'a quantidade é {self._biblio}'

#     def depo(self, dep):
#         if dep > 0:
#             self._biblio += dep
#             print(f'{dep} adicionados.')
#         else:
#             print('valor invalido')

#     def sak(self, sal):
#         if sal > self._biblio:
#             print(f'não existe está quantidade de livros, oque tem é {self._biblio}')
#         elif sal <= 0:
#             print('você não retirou nada ou valor negativo')
#         else:
#             self._biblio -= sal
#             print(f'você retirou {sal} livros com sucesso') 
    
#     def mos(self):
#         print(f'QUANTIDADE ATUAL:')
#         return f'quantidade atual da biblioteca {self._biblio}'
  
# o = biblio(12)
# print(o.moe())
# o.depo(7)
# print(o.mos())
# o.sak(6)
# print(o.mos())


#OU JEITO CERTO


# class biblio:
#     def __init__(self):
#         self._livros = []
    
#     def depos(self, *dep):
#         for l in dep:
#             self._livros.append(l)
#             print(f'livro \{l}/ adicionado com sucesso')

#     def sake(self, *sat):
#         for k in sat:
#             if k in self._livros:
#                 self._livros.remove(k)
#                 print(f'livro "{k}" removido com sucesso')
#             else:
#                 print(f'o livro {k} não existe na biblioteca')
    
#     def mos(self):
#         if self._livros:
#             print(f'estes são os livros \ {self._livros} / disponiveis')
#             for jar, lor in enumerate(self._livros, start=1):
#                 print(f' {jar} : {lor}')
#         else:
#             print('biblioteca está vazia')

# o = biblio()
# o.mos()
# o.depos('rubia safada', 'rubia latejante')
# o.depos('rubia gulosa')
# o.mos()
# o.sake('rubia safada', 'rubia latejante')
# o.mos()


# Implemente um sistema de locadora de filmes usando POO 
# com herança e polimorfismo (ex.: Filme, Serie, Documentario, 
# todos herdando de Midia).


# class mateo:
#     def __init__(self, titulo, ano):
#         self._titulo = titulo
#         self._ano = ano

#     def info(self):
#         return f'{self._titulo} {self._ano}'
    
#     def __str__(self):
#         return self.info()
        
# class filme(mateo):
#     def __init__(self, titulo, ano, film):
#         self._film = film
#         super().__init__(titulo, ano)

#     def info(self):
#         return f'Filmes: {self._titulo} / {self._film} / {self._ano}'
    
# class serie(mateo):
#     def __init__(self, titulo, ano, ser):
#         self._ser = ser
#         super().__init__(titulo, ano)

#     def info(self):
#         return f'Series: {self._titulo} / {self._ser} / {self._ano}'
    
# class doc(mateo):
#     def __init__(self, titulo, ano, docu):
#         self._docu = docu
#         super().__init__(titulo, ano)

#     def info(self):
#         return f'Documentario: {self._titulo} / {self._docu} / {self._ano}'
    
# class loc:
#     def __init__(self):
#         self._loca = []

#     def dep(self, *depo):
#         for l in depo:
#             self._loca.append(l)
#             print(f'{l} >>>>>> video guardado com sucesso')

#     def rem(self, remov):
#         if remov in self._loca:
#             self._loca.remove(remov)
#             print(f'retirado {remov} com sucesso')
#         else:
#             print(f'não existe o conteudo {remov}')

#     def most(self):
#         if self._loca:
#             print(f'aqui está os filmes')
#             for lan, kan in enumerate(self._loca, start=1):
#                 print(f'o VIDEO é {lan} : {kan}')
#         else:
#             print(f'locadora fechada')

# lo = filme('AÇÃO', '2020', 'os padrinhos')
# jo = serie('DRAMA', '2023', 'Os boiolas')
# u = doc('SEXY', '1995', 'Rubia a gulosa')

# loa = loc()
# print(mateo('LOCADORA', '1400'))
# loa.most()
# loa.dep(lo, jo, u)
# loa.dep('the gothic','DARKWEB / latejante / 1999')
# loa.most()
# loa.rem(lo)
# loa.most()

    
# Use POO para modelar um sistema de RPG com classes Personagem, Mago, 
# Guerreiro, Arqueiro, cada um com habilidades próprias.


# from abc import ABC, abstractmethod

# class persona(ABC):
#     def __init__(self, nome, vida, ataque):
#         self._nome = nome
#         self._vida = vida
#         self._ataque = ataque

#     def inf(self):
#         return f'PIRIGUETE: {self._nome} / Vida: {self._vida} / Ataque: {self._ataque}'
    
#     @abstractmethod
#     def hab(self):
#         pass

# class guer(persona):
#     def __init__(self, nome):
#         super().__init__(nome, vida = 70, ataque = 50)

#     def hab(self):
#         return f'{self._nome} causa dano de ({self._ataque*0.5})'
    
# class arq(persona):
#     def __init__(self, nome):
#         super().__init__(nome, vida = 80, ataque = 80)
    
#     def hab(self):
#         return f'{self._nome} causa dano de ({self._ataque*7})'
    
# class mag(persona):
#     def __init__(self, eomagoo):
#         super().__init__(eomagoo, vida = 1700, ataque = 77)

#     def hab(self):
#         return f'{self._nome} causa dano de ({self._ataque*50})'
    
# o = guer('rubia safada')
# i = mag('rubia saborosa')
# j = arq('rubia gostosa')

# print(o.inf())
# print(o.hab())
# print(i.inf())
# print(i.hab())
# print(j.inf())
# print(j.hab())


# Crie e escreva “Olá, Python!” em um .txt.


# bordel = ['puta\n', 'prostituta\n', 'nega maluca\n', 'soquete\n']
# try:
#     textlin = open('borde.txt', 'w', encoding='utf-8')
#     textlin.write('puta dormiu no sofa\n')
#     textlin.write('prostituta esta na mesa\n')
#     textlin.write('Olá, Python!')
#     textlin.writelines(bordel)
# except IOError:
#     print(f' não foi possivel abrir o arquivo')
# else:
#     textlin.close()

# try:
#     textlin = open('borde.txt', 'r', encoding='utf-8')
#     print(textlin.read())
# except IOError:
#     print(f'não foi possivel abrir o arquivo')
# else:
#     textlin.close()


# Leia um .txt e conte as palavras.


# try:
#     lor = open('borde.txt', 'r', encoding='utf-8')
#     mar = lor.read()
#     hor = mar.split()
#     lur = len(hor)
#     print(f'total de palavras: {lur}')
# except IOError:
#     print('Não foi possivel ler')
# else:
#     lor.close()


# crie um csv


# import csv

# linha = [
# ['NOME', 'IDADE', 'JAGUAR'],
# ['GAGO', 67, 'PINTADO'],
# ['MILLENA', 89,'GAY'],
# ['MARCAS', 7, 'NEGO']]

# try:
#     with open('jacs.csv', 'w', newline='', encoding='utf-8') as b:
#         me = csv.writer(b)
#         me.writerows(linha)
#         print(f'CSV criado com sucesso {linha}')
# except IOError:
#     print('Não foi possivel escrever arquivo')


# Leia um CSV de produtos e calcule o valor total do estoque.


# import csv
# try:
#     with open('arquivo.csv', 'r', encoding='utf-8') as m:
#         ler = csv.DictReader(m)
#         totalvalor = 0

#         for linha in ler:
#             quant = int(linha['Idade'])
#             preço = float(linha['valor'])
#             totalvalor += quant*preço
        
#         print(f'O valor de todo o estoque: R$ {totalvalor}')

# except IOError:
#     print('não foi possivel ler arquivo')


# Desenvolva um programa que gerencie um cadastro de usuários em JSON 
# (inserir, atualizar, remover, salvar em arquivo).


# import json
# import os

# lin = 'mega.json'

# def corega():
#     "carregar dados"
#     if not os.path.exists(lin):
#         return []
#     try:
#         with open(lin, 'r', encoding='utf-8')as b:
#             return json.load(b)
#     except (json.JSONDecodeError, IOError):
#         return []

# def salve(dead):
#     "salvar dados"
#     with open(lin, 'w', encoding='utf-8')as b:
#         json.dump(dead, b, ensure_ascii=False, indent=4)

# def lord(nome, email):
#     lor = corega()
#     lor.append({'nome': nome,'email': email })
#     salve(lor)
#     print(f'o {nome} foi adicionado com sucesso')

# def atuw(email, newnan=None, newema=None):
#     lor = corega()
#     for w in lor:
#         if w['email'] == email:
#             if newema:
#                 w['email'] = newema
#             if newnan:
#                 w['nome'] = newnan
#             salve(lor)
#             print(f'o {email} foi atualizado com sucesso')
#             return
#     print('o usuario não foi encontrado')

# def apaga(email):
#     lor = corega()
#     lore = [l for l in lor if l['email'] != email]
#     if len(lore) != len(lor):
#         salve(lore)
#         print(f'o usuario {email} foi removido com sucesso')
#     else:
#         print('usuario não foi encontrado')   

# def most():
#     lor = corega()
#     if not lor:
#         print('não existe este usuario')
#     else:
#         print('usuarios cadastrados')
#         for g, j in enumerate(lor, start=1):
#             print(f"{g} : {j['email']} : {j['nome']}")

# if __name__ == '__main__':
#     while True:
#         print('\n1 - Inserir usuario')
#         print('2 - Atualizar usuario')
#         print('3 - Remover usuario')
#         print('4 - Listar usuarios')
#         print('5 - sair')
#         op = input('>>>>> escolha opção:   ').strip()

#         if op == '1':
#             email = input('digite o email para entrar:  ')
#             nome = input('digite o usuario: ')
#             lord(nome, email)
#         elif op == '2':
#             email = input('digite o email atual: ')
#             newnan = input('digite o novo usuario (ou Enter p/ deixar): ') or None
#             newema = input('digite o novo email (ou Enter p/ deixar): ') or None
#             atuw(email, newnan, newema)
#         elif op == '3':
#             email = input('digite o email para remover: ')
#             apaga(email)
#         elif op == '4':
#             print('aqui todos os usuarios')
#             most()
#         elif op == '5':
#             print('>>>> saindo...')
#             break
#         else:
#             print('opção invalida')


# Use enumerate para imprimir índices e valores de uma lista.


# list = [1,2,3,4,5,6,7,8,9,10]
# for i ,j in enumerate(list):
#     print(i, j)
    
  
# # Use with para abrir um arquivo de forma segura.


# import csv
# try:
#     with open('jacs.csv', 'r', encoding='utf-8') as b:
#         lin = csv.DictReader(b)
#         for lor in lin:
#             print(lor)
# except IOError:
#     print('não deu para abrir')


# Crie um generator que produza os primeiros 100 números da 
# sequência de Fibonacci.


# def fibo(n):
#     a,b = 0,1
#     for _ in range(n):
#         yield a
#         a,b =b, a+b

# for i in fibo(20):
#     print(i)


# OU GUARDAR NA LISTA


# def jor(n):
#     man = []
#     a,b = 0,1
#     for _ in range(n):
#         man.append(b)
#         a,b = b, a+b
#     return man

# for b in jor(20):
#     print(b)


# Crie um generator infinito de números primos.


# def lin():
#     n = 2
#     while True:
#         for l in range(2, int(n**0.5)+1):
#             if n % l ==0:
#                 break
#         else:
#             yield n
#         n += 1
        

# jar = lin()
# for j in range(20):
#     print(next(jar))


# Implemente um sistema que consome dados de um generator em streaming 
# (simulando um fluxo de sensores) e aplique filtros com funções lambda.


# import random
# import time

# def sensor():
#     while True:
#         valor = random.uniform(20.0, 35.0)
#         yield valor
#         time.sleep(0.1)

# def consu(stream, filtro= None, n=20):
#     for _ in range(n):
#         valor = next(stream)
#         if filtro:
#             if filtro(valor):
#                 print(f'valor: {valor:.2f}')
#         else:
#             print(f'valor: {valor:.2f}')

# if __name__ == '__main__':
#     senx = sensor()

#     print('todos os valores:')
#     consu(senx, n =5)

#     print('\n valores acima de 30:')
#     consu(senx, filtro=lambda x:x >30, n=5)


# Leia um número e diga se é par ou ímpar.


# nora = int(input('digite o numero: '))
# if nora %2 == 0:
#     print(f'é par: {nora}')
# else:
#     print('não é par')


# Calcule a soma dos números de 1 a 100.


# soma = sum(range(1,101))
# print(soma)


# # Conte quantos caracteres existem em uma string


# lore ='rapidinha violenta'
# print(len(lore))


# Inverta uma string sem usar slicing.


# lare ='ele gosxxta, ai mamâeee'
# lore = ''.join(reversed(lare))
# print(lore)


# ou número


# lar =12345
# lore = int(str(lar)[::-1])
# print(lore)


#outros jeitos


# num = 12345
# inv = 0

# while num > 0:
#     resto = num % 10
#     inv = inv *10 + resto
#     num //=10
#     print(inv)

# def inverter(n, inv = 0):
#     if n == 0:
#         return inv
#     return inverter(n // 10, inv*10+n % 10)

# print(inverter(12345))


# Leia 3 números e imprima o maior deles.


# lore = int(input('Digite um número: '))
# amar = int(input('Digite outro número: '))
# loe = int(input('Digite mais um número: '))

# if lore >= amar and lore >= loe:
#     print(f'{lore} é o maior número')
# elif amar >= lore and amar >= loe:
#     print(f'{amar} é o maior número')
# else:
#     print(f'{loe} é o maior número')


# Gere a tabuada de um número escolhido pelo usuário.


# sus = int(input('Digite numero para a tabuada: '))
# print(f'====TABUADA DO {sus}====')
# for lin in range(1,11):
#     print(f'{lin} X {sus} = {lin*sus}')


#Crie uma lista de 10 números e imprima apenas os pares.



# for lin in range(1,11):
#     if lin %2 == 0:
#         print(f'{lin} é um numero par')
#     else:
#         print(f'{lin} é um numero impar')


# OU


# NUM = list(range(1,11))
# paso =  [n for n in NUM if n %2 == 0]
# print(paso)


# Crie um programa que conta quantas vogais existem em uma frase.


# lore = 'jara na ficando muito doido'
# lark = [x for x in lore if x in 'aeiou']
# print(len(lark))


# Leia um número e calcule seu fatorial.


# mari = int(input('Digite um numero: '))
# lara = 1
# for roge in range(1, mari+1):
#     lara *= roge

# print(lara)


# Leia uma lista de números e calcule a média.


# lore = []
# for lan in range(10):
#     lan = int(input('digite um numero para a media: '))
#     lore.append(lan)

# laurk = sum(lore) / len(lore)
# print(laurk)


# Gere os primeiros 20 números da sequência de Fibonacci.


# def loi(n):
#     lore = []
#     a,b = 0,1
#     for _ in range(n):
#         lore.append(b)
#         a,b = b, a+b
#     return lore

# for jor in loi(20):
#     print(jor)


# Conte quantas vezes cada palavra aparece em um texto.


# doit = """Em linguística, a noção de texto é ampla e ainda aberta a uma 
# definição mais precisa. Grosso modo, pode ser entendido como manifestação 
# linguística das ideias de um autor, que serão interpretadas pelo leitor de acordo 
# com seus conhecimentos linguísticos e culturais. Seu tamanho é variável."""

# lark = doit.lower().split()
# dicio = {}

# for lot in lark:
#     lot = lot.strip(',.?!:;')
#     dicio[lot] = dicio.get(lot, 0) +1

# for lot, rabo in dicio.items():
#     print(f'{lot}: {rabo}')


# Remova os duplicados de uma lista.


# lore = [1,2,3,4,5,7,3,2,6,8,0,9,10,6,7,8,10]
# larkk = set(lore)
# print(larkk)


# Verifique se uma palavra é palíndromo.


# jaca = ['ovo', 'radar', 'ana', 'jokenpo', 'aniki']
# for jor in jaca:
#     if jor[::-1] == jor:
#         print(f'{jor}: temos um palindromo')
#     else:
#         print(f'{jor}: não é um palindromo')


# Ordene uma lista de números sem usar sort().


# lore = [3,1,7,10,9,4,2,6,5,8]
# n = len(lore)
# for i in range(n):
#     for man in range(0, n -i -1):
#         if lore[man] > lore[man+1]:
#             lore[man], lore[man+1] = lore[man+1], lore[man]

# print(lore)


# Leia um arquivo .txt e conte as linhas.


# try:
#     lore = open('arquivo.txt', 'r', encoding='utf-8')
#     mar = lore.read()
#     lar = mar.splitlines()
#     print(len(lar))
# except IOError:
#     print('Erro desconhecido')
# else:
#     lore.close()


# Escreva uma lista de nomes em um arquivo .txt.


# nomes = ['rubia rabosa','bunda latejante','nadegas gulosas']
# try:
#     with open('arquivo.txt', 'w', encoding='utf-8') as foda:
#         for n in nomes:
#             foda.write(n + '\n')
#     print('Deu certo')
# except IOError:
#     print('Erro desconhecido')


# Converta Celsius em Fahrenheit.


# cel = int(input('Escreva quanto graus celsius quer converter: '))
# lor = cel*1.8 +32
# print(f'{cel}°C = {lor} Fahrenheit')


# Gere 10 números aleatórios e armazene em uma lista.


# import random
# ko = []
# for rar in range(10):
#     tor = random.randint(1,100)
#     ko.append(tor)

# print(ko)


# Implemente uma calculadora com as operações básicas.


# def soma(a,b):
#     return a+b

# def subtrei(a,b):
#     return a-b

# def multipli(a,b):
#     return a*b

# def divide(a,b):
#     if b == 0:
#         return 'Erro: divisor não pode ser zero'
#     return a / b

# def calculadaro():
#     print('====calculadora básica====')
#     try:
#         a = float(input('Digite o primeiro número: '))
#         operador = input('Digite a operação (+. -, *, /): ')
#         b = float(input('Digite o segundo número: '))
    
#         if operador == '+':
#             resultado = soma(a, b)
#         elif operador == '-':
#             resultado = subtrei(a,b)
#         elif operador == '*':
#             resultado = multipli(a,b)
#         elif operador == '/':
#             resultado = divide(a,b)
#         else:
#             print('Operador inválido!')
#             return
        
#         print(f'Resultado: {resultado}')

#     except ValueError:
#         print('Erro: digite apenas os números validos.')

# if __name__ == '__main__':
#     calculadaro()


# Leia um número e diga se ele é primo.


# def primo(n):
#     if n<2:
#         return False
#     for ola in range(2, int(n**0.5)+ 1):
#         if n % ola == 0:
#             return False
#     return True

# num = int(input('Digite um número: '))
# if primo(num):
#     print(f'{num} é primo')
# else:
#     print(f'{num} não é primo')


# Leia uma lista e encontre o segundo maior número.


# numeros = input('Digite os numeros separados por espaço: ').split()
# numeros = [int(n) for n in numeros]

# uninum = list(set(numeros))

# if len(uninum) < 2:
#     print('Não há um segundo maior número')
# else:
#     uninum.sort(reverse=True)
#     print(f'O segundo maior número é: {uninum[1]}')


# Inverta a ordem de uma lista sem usar reverse().


# lore = [1,24,5,2,7,8,5,8,9]
# num = []

# for lin in range(len(lore) -1, -1, -1):
#     num.append(lore[lin])

# print(num)


# OU


# lore = [1, 2, 3, 5, 6, 7, 89, 10]
# invertida = []

# while lore:
#     invertida.append(lore.pop())  

# print(invertida)


#OU


# lore = [1, 2, 3, 5, 6, 7, 89, 10]
# n = len(lore)

# for i in range(n // 2):  # só até a metade
#     lore[i], lore[n - i -1] = lore[n - i -1], lore[i]

# print(lore)


# Leia números até o usuário digitar 0, então calcule a soma.

# soma = 0
# while True:
#     try:
#         a = int(input('digite um numero: '))
#         if a == 0:
#             break
#         soma += a
      
#     except ValueError:
#         print('bota numero ')

# print(soma)


# Gere uma senha aleatória de 8 caracteres.


# import random
# import string
# senha = ''.join(random.choices(string.ascii_letters + string.digits +string
# .punctuation, k=8))
# login = 'rabadu'

# print(f'senha aleatorio: {senha}')

# logi= input('login ai ')
# sen = input('senha ai ')

# if logi == login and sen == senha:
#     print('login com sucesso')
# else:
#     print('deu não')


# Crie um dicionário que conte a frequência das letras em um texto.


# text = input('Digite uma frase ou texto: ')
# lor = {}

# for lar in text:
#     if lar.isalpha():
#         if lar in lor:
#             lor[lar] +=1
#         else:
#             lor[lar] = 1

# print(lor)


# Leia um nome e imprima apenas as iniciais.


# nome = input('insira nome completo: ' ).split()

# for lor in nome:
#     print(lor[0].capitalize(), end='.')


# Leia uma lista de números e remova todos os negativos.


# lir = [-1,-4,-10, 1, 2,26,82]
# posi = []
# for lor in lir:
#     if lor > 0:
#         posi.append(lor)

#         print(f'aqui numeros positivos {lor}' )


# Crie um programa que simula um dado (1 a 6).


# import random

# for lin in range(2):
#     var1 = random.randint(1,6)
#     var2 = random.randint(1,6)
#     print(f' Jogada{lin} = dado [1] = {var1} e dado [2] = {var2} deu {var1+var2} pontos')


# Leia uma frase e conte quantas palavras têm mais de 5 letras.


# frase = input('digite a frase:  ').split()
# mar = 1
# for lor in frase:
#     if len(lor) >= 5:
#         print(f'frase tem mais de 5 letras {lor} e {mar}')
#         mar +=1


# Gere uma matriz 3x3 com números aleatórios.


# import random
# lis = []
# for lor in range(9):
#     lis.append(random.randint(1,100))

# for lar in range(0, 9, 3):
#     print(lis[lar:lar+3])


#OU


# Some todos os elementos de uma matriz 3x3.


# import random
# lore = []
# soma = 0
# for _ in range(9):
#     lore.append(random.randint(1,100))


# for min in range(0, 9, 3):
#     print(lore[min:min+3])
#     soma += sum(lore[min:min+3])

# print(f'soma de todos os elementos: {soma}')
    

# Leia uma string e substitua todas as vogais por *.


# frase = input('Digite uma frase: ')

# lotrra = 'AEIOUaeiou'
# resultado = ''

# for char in frase:
#     if char in lotrra:
#         resultado += '*'
#     else:
#         resultado += char

# print(resultado)


# Crie uma lista de 1 a 100 e filtre apenas múltiplos de 7.


# lor = []
# for a in range(1, 101):
#     if a % 7 == 0:
#         lor.append(a)

# print(lor)


# ou


# lor = [a for a in range(1,101) if a %7 ==0]
# print(lor)


# Leia um número e mostre sua tabuada ao contrário.


# num = int(input('Digite um numero:  '))
# print(f'====TABUADA DO {num}====')
# for rar in range(10, 0, -1):
#     print(f'{num} x {rar} = {num*rar}')
    

# Verifique se duas strings são anagramas.


# nomes = ['amor', 'roma', 'melado', 'modela', 'jaca', 'caja', 'rapadura','acaj']

# for lor in range(len(nomes)):
#     for lar in range(lor+1, len(nomes)):
#         if sorted(nomes[lor]) == sorted(nomes[lar]):
#             print(f'o {nomes[lor]} e {nomes[lar]} são anagramas')


# Crie um programa que converta decimal em binário.


# num = int(input('digite um numero decimal: '))
# bin = ''

# if num == 0:
#     bin == '0'
# else:
#     while num > 0:
#         r = num % 2
#         num //= 2
#         bin = str(r) + bin
    
# print(f'Binario: {bin}')


# Leia uma lista e multiplique todos os valores entre si.


# lore = []
# mult = 1
# for nar in range(5):
#     nar = int(input('digite numeros para a lista: '))
#     lore.append(nar)
#     mult *= nar
    
# print(f'lista: {lore}, Resultado de todos multiplicados "{mult}"')


# Gere um triângulo de números no console.
# Triângulo direito simples (cada linha com 1..i)


# lin = int(input('digite a altura do triangulo: '))

# for jan in range(1, lin+1):
#     for lon in range(1, jan+1):
#         print(f'{lon}', end="")
#     print()


# ou Triângulo de Floyd (números consecutivos)


# n = int(input("Altura do triângulo: "))
# num = 1
# for i in range(1, n+1):
#     for _ in range(i):
#         print(num, end=" ")
#         num += 1
#     print()


# ou Triângulo (piramidal) centralizado com números da linha


# n = int(input("Altura do triângulo: "))
# for i in range(1, n+1):
#     print(" " * (n - i), end="")               # espaçamento para centralizar
#     for j in range(1, i+1):
#         print(j, end=" ")
#     print()


# ou Triângulo de Pascal (valores binomiais)


# def pascal(n):
#     row = [1]
#     for _ in range(n):
#         print(" ".join(map(str, row)).center(4*n))
#         row = [1] + [row[i] + row[i+1] for i in range(len(row)-1)] + [1]

# n = int(input("Número de linhas de Pascal: "))
# pascal(n)


# Crie uma classe ContaBancaria com métodos de depósito e saque.


# class contabancaria:
#     def __init__(self, valorinicia = 0):
#         self.__saldo = valorinicia

#     def deposi(self, deposit):
#         if deposit > 0:
#             self.__saldo += deposit
#             print(f' deposito do {deposit} realizado com sucesso')
#         else:
#             print('deposito invalido')

#     def saque(self, sakee):
#         if sakee > self.__saldo:
#             print('saldo insuficiente')
#         elif sakee <= 0:
#             print('valor invalido')
#         else:
#             self.__saldo -= sakee
#             print(f'saque de {sakee} foi retirado com sucesso.')
        
#     def saldo(self):
#         print(f'saldo atual {self.__saldo}')
#         return self.__saldo
    
# ioio = contabancaria(100000)
# ioio.deposi(10100)
# ioio.saldo()
# ioio.saque(57534.56)
# ioio.saldo()


# Implemente um jogo de adivinhação (número secreto).


# import random

# print("=== JOGO DE ADIVINHAÇÃO ===")
# print("Escolha a dificuldade: ")
# print("1 - Fácil (1 a 20)")
# print("2 - Médio (1 a 50)")
# print("3 - Difícil (1 a 100)")

# nivel = int(input('Digite sua escolha: '))
# if nivel ==1:
#     limite, tentativas = 20, 7
# elif nivel ==2:
#     limite, tentativas = 50,6
# else:
#     limite, tentativas = 100,5

# sec = random.randint(1, limite)

# print(f'\nadivinhe o numero entre 1 e {limite}. você tem {tentativas} tentativas!')

# for rar in range(tentativas):
#     palpite = int(input(f'Tentativa {rar+1}/{tentativas}: '))
#     if palpite == sec:
#         print(f'você venceu! o numero era {sec}')
#         break
#     elif palpite < sec:
#         print('Dica: o numero é maior')
#     else:
#         print('Dica: o numero é menor')
# else:
#     print(f' Game Over! o numero era {sec}')


# Leia um CSV de produtos e calcule o valor total do estoque.


# import csv
# with open('arquivo.csv', 'r', newline='', encoding='utf-8') as lor:
#     jar = csv.DictReader(lor)
#     tot = 0
    
#     for to in jar:
#         lire = int(to['Idade'])
#         jor = float(to['valor'])
#         tot += lire*jor
    
#     print(tot)


# Implemente uma função que recebe uma lista e retorna apenas os números únicos.


# def unicos(self):
#     return list(set(self))

# nor = [1,2,2,2,3,4,4,4,5,5]
# print(unicos(nor))


# Crie um programa que leia JSON e imprima formatado.


# import json

# with open('usuarios.json', 'r', encoding='utf-8') as g:
#     dad = json.load(g)
# #formatação
# print(json.dumps(dad, indent=4, ensure_ascii= False))


# Implemente um decorador que conte o tempo de execução de uma função.


# import time

#Essa técnica de usar *args e **kwargs em uma função decoradora
# é chamada de argumentos arbitrários ou parâmetros variádicos.

# def tempo(func):
#     def wrar(*arg, **kwanza):
#         i= time.time()
#         print(f'Tempo de execução: {fi - ini:.6f}')
#         retni = time.time()
#         resu = func(*arg, **kwanza)
#         fi urn resu
#     return wrar

# @tempo
# def cal():
#     som = 0
#     for rar in range(1,10**6):
#         som += rar
#     return som

# print(cal())


# Crie um gerador que produza números pares infinitamente.


# def par():
#     n=0
#     while True:
#         yield n
#         n+=2

# for p in par():
#     print(p)
#     if p >= 20:
#         break


# Leia um arquivo .txt grande e conte quantas vezes aparece uma palavra.


# with open('arquivo2.txt', 'r', encoding='utf-8') as l:
#     lor = l.read()

# jar = lor.lower().split()
# digi = {}

# for gri in jar:
#     gri = gri.strip(',.:;?!')
#     digi[gri] = digi.get(gri, 0)+1

# for gri,bun in digi.items():
#     print(f'{gri}: {bun}')

# Implemente uma função que calcule o MMC de dois números.


# def lun(a,b):
#     while b != 0:
#         a, b =b, a%b
#     return a

# def lan(a,b):
#     return (a*b) // lun(a,b)

# x = 12
# y = 18
# print(f'O MMC de {x} e {y} é {lan(x,y)}')


# Crie um programa que valide CPF (formato e dígitos verificadores).


# import re

# def validar_cpf(cpf:str) -> bool:
#     """
#     Valida um CPF
#     Aceita CPF com pontuação ('123.456.789-00') ou apenas digitos ('12345678901')
#     retorna true se o cpf for invalido, false caos contrario.
#     """

    # cpfl = re.sub(r'\D', '', str(cpf))

#     if len(cpfl) != 11:
#         return False
    
#     if cpfl == cpfl[0] * 11:
#         return False
    
#     ints =[int(cha) for cha in cpfl]

#     soma1 = sum((10-i) * ints[i] for i in range(9))
#     resto1 = soma1 % 11
#     dig1 = 0 if resto1 < 2 else 11 - resto1

#     if ints[9] != dig1:
#         return False
    
#     soma2 = sum((11-i) * ints[i] for i in range(10))
#     resto2 = soma2 % 11
#     dig2 = 0 if resto2 < 2 else 11 - resto2

#     if ints[10] != dig2:
#         return False
    
#     return True

# if __name__ == '__main__':
#     testes = [
#         "529.982.247-25",
#         "12345678910",
#         '111.111.111.11',
#         '00000000000',
#         '862.883.667-57'
#     ]

#     for t in testes:
#         print(f'{t} -> {validar_cpf(t)}')

    
# Conecte-se a um banco SQLite e cadastre usuários.


# import sqlite3

# conn = sqlite3.connect('usuarios.db')
# cursor = conn.cursor()

# cursor.execute("""
# CREATE TABLE IF NOT EXISTS usuarios(
#                id INTEGER PRIMARY KEY AUTOINCREMENT,
#                nome TEXT NOT NULL,
#                email TEXT UNIQUE NOT NULL
#                )
# """)
# conn.commit()

# def cadastrar_usuario(nome, email):
#     try:
#         cursor.execute('INSERT INTO usuarios (nome, email) VALUES (?,?)', (nome, email))
#         conn.commit()
#         print(f'Usuario {nome} cadastrado com sucesso!')
#     except sqlite3.IntegrityError:
#         print('Erro: Email já cadastrado!')

# cadastrar_usuario('Alice', 'alice@gmail.com')
# cadastrar_usuario('BOV', 'bobdw@gmail.com')

# cursor.execute('SELECT * FROM usuarios')
# for usuario in cursor.fetchall():
#     print(usuario)

# conn.close()


# Implemente uma classe Funcionario e uma subclasse Gerente.


# class funcionario:
#     def __init__(self, nome, salario):
#         self.nome = nome
#         self.salario = salario

#     def mostrar_info(self):
#         print(f'Nome {self.nome}, Salário: R${self.salario:2f}')

#     def aumentar_salario(self, percentual):
#         self.salario += self.salario * (percentual / 100)
#         print(f'Salario de {self.nome} atualizado para R${self.salario:.2f}')

# class gerente(funcionario):
#     def __init__(self, nome, salario, setor):
#         super().__init__(nome,salario)
#         self.setor = setor
    
#     def mostrar_info(self):
#         super().mostrar_info()
#         print(f'Setor: {self.setor}')

#     def aumentar_salario(self, percentual):
#         bonus = 10
#         total_percentual =percentual +bonus
#         super().aumentar_salario(total_percentual)

# f1 = funcionario('alice', 4000)
# f1.mostrar_info()
# f1.aumentar_salario(10)

# g1=gerente('bob', 5000, 'financeiro')
# g1.mostrar_info()
# g1.aumentar_salario(10)


# Simule uma fila de atendimento usando collections.deque.


# from collections import deque

# fila = deque()

# fila.append('alice')
# fila.append('rubia')
# fila.append('raichu')
# fila.append('vitoria')

# print('fila inicial:', list(fila))

# while fila:
#     cliente = fila.popleft()
#     print(f'atendendo {cliente}')

# print('todos os clientes foram atendidos!')


# Use map() para elevar ao quadrado todos os elementos de uma lista.


# numeros = [1,2,3,4,5]

# quadrados = list(map(lambda x:x**2, numeros))

# print('original:', numeros)
# print('quadrados', quadrados)


# Use filter() para extrair apenas os números primos de uma lista.


# def primo(n):
#     if n < 2:
#         return False
#     for i in range(2, int(n**0.5)+1):
#         if n % i == 0:
#             return False
#     return True

# num = list(range(1,31))

# primos = list(filter(primo,num))

# print('original:', num)
# print('primos', primos)


# Implemente uma função que valide senhas (mínimo 8 chars, número, maiúscula).


# def validar_senha(senha):
#     especiais = '!@#$%^¨&*()-_=+'

#     if len(senha) <8:
#         return False, 'A senha deve ter pelo menos 8 caracteres.'
    
#     if not any(ch.isupper() for ch in senha):
#         return False, 'A senha deve conter pelo menos uma letra maiúscula.'
    
#     if not any(ch.isdigit() for ch in senha):
#         return False, 'A senha deve conter pelo menos um número.'
    
#     if not any(ch in especiais for ch in senha):
#         return False, 'A senha deve conter pelo menos um caractere especial.'
    
#     return True, 'senha valida'

# senhas= ['abc123', 'senhaforte1', 'Senhaboa1', 'Senhaboa1@', '123456789707070']
# for s in senhas:
#     ro,lu = validar_senha(s)
#     print(f'{s} -> {lu}')


# Crie um programa que faça scraping em uma página da web.


# import requests
# from bs4 import BeautifulSoup

# url = 'https://google.com'
# resposta = requests.get(url)
# if resposta.status_code == 200:
#     soup = BeautifulSoup(resposta.text, 'html.parser')
#     paragrafos = soup.find_all('p')
#     print('paragrafos encontrados: ')
#     for p in paragrafos:
#         print(p.get_text())
# else:
#     print(f'Erro ao acessar a pagina. Status code: {resposta.status_code}')


# Implemente uma API REST simples com Flask ou FastAPI.

# Exemplo com Flask
# from flask import Flask, jsonify, request

# app = Flask(__name__)

# items = []

# @app.route("/items", methods=['GET'])
# def get_items():
#     return jsonify(items)


# @app.route('/items', methods=['POST'])
# def create_item():
#     data =request.get_json()
#     item = {'id': len(items) + 1, 'name': data['name']}
#     items.append(item)
#     return jsonify(item), 201

# @app.route('/items/<int:item_id>', methods=['GET'])
# def get_item(item_id):
#     item = next((i for i in items if i['id'] == item_id), None)
#     if item:
#         return jsonify(item)
#     return  jsonify({'error': 'item not found'}), 404

# @app.route('/items/<int:item_id>', methods=['DELETE'])
# def delete_item(item_id):
#     global items
#     items = [i for i in items if i['id'] != item_id]
#     return jsonify({'message': 'item deleted'})

# if __name__ == '__main__':
#     app.run(debug=True)


#OU Exemplo com FastAPI


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# app = FastAPI()

# class Item(BaseModel):
#     name: str

# items = []

# @app.get('/items')
# def get_items():
#     return items

# @app.post('/items')
# def create_item(item: Item):
#     new_item = {'id': len(items)+1, 'name': item.name}
#     items.append(new_item)
#     return new_item

# @app.get('/items/{item_id}')
# def get_item(item_id: int):
#     for i in items:
#         if i['id'] == item_id:
#             return i
#     raise HTTPException(status_code=404, detail='Item not found')
    
# @app.delete('/items/{item_id}')
# def delete_item(item_id: int):
#     global items
#     items = [i for i in items if i['id'] != item_id]
#     return {'message': 'Item deleted'}


# Simule um sistema de estoque com entrada e saída de produtos.


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# app = FastAPI()

# class Produto(BaseModel):
#     nome: str
#     quantidade: int

# estoque = []

# @app.get('/produtos')
# def listar_produtos():
#     return estoque

# @app.post('/produtos')
# def adicionar_produto(produto: Produto):
#     existente = next((p for p in estoque if p['nome'].lower() == produto.nome.lower()), None)
#     if existente:
#         existente['quantidade'] += produto.quantidade
#     else:
#         novo = {'id': len(estoque)+1, 'nome': produto.nome, 'quantidade': produto.quantidade}
#         estoque.append(novo)
#     return {'mensagem': "Produto adicionado/atualizado com sucesso"}

# @app.post('/produtos/retirar')
# def retirar_produto(produto: Produto):
#     existente = next((p for p in estoque if p['nome'].lower() == produto.nome.lower()), None)
#     if not existente:
#         raise HTTPException(status_code=404, detail='Produto não encontrado no estoque.')
#     if existente['quantidade'] < produto.quantidade:
#         raise HTTPException(status_code=400, detail='Quantidade insuficiente no estoque')
#     existente['quantidade'] -= produto.quantidade
#     return {'mensagem': 'Produto retirado com sucesso'}

# @app.get('/produtos/{produto_id}')
# def consultar_prudto(produto_id: int):
#     p = next((p for p in estoque if p['id'] == produto_id), None)
#     if not p:
#         raise HTTPException(status_code=404, detail='Produto não encontrado')
#     return p


# Crie uma calculadora científica com funções (math).


#!/usr/bin/env python3
# """
# Calculadora cientifica simples (CLI) - usa math.
# Salvo como questoes1 e rodando como questoes1.py
# """

# import math
# import readline
# from collections import deque

# history_size=50
# mode_radians = True
# history = deque(maxlen=history_size)

# def _deg_to_rad(x):
#     return math.radians(x)

# def _rad_to_deg(x):
#     return math.degrees(x)

# safe_symbols = {
#     'pi': math.pi,
#     'e': math.e,
#     'abs': abs,
#     'round': round,
#     'sqrt': math.sqrt,
#     'pow': pow,
#     'exp': math.exp,
#     'ln': math.log,
#     'log': math.log,
#     'log10': math.log10,
#     'fact': math.factorial,
#     'factorial': math.factorial,

# }

# def _sin(x):
#     return math.sin(x if mode_radians else _deg_to_rad(x))

# def _cos(x):
#     return math.cos(x if mode_radians else _deg_to_rad(x))

# def _tan(x):
#     return math.tan(x if mode_radians else _deg_to_rad(x))

# def _asin(x):
#     val = math.asin(x)
#     return val if mode_radians else _rad_to_deg(val)

# def _acos(x):
#     val = math.acos(x)
#     return val if mode_radians else _rad_to_deg(val)

# def _atan(x):
#     val = math.atan(x)
#     return val if mode_radians else _rad_to_deg(val)


# safe_symbols.update({
#     'sin': _sin,
#     'cos': _cos,
#     'tan': _tan,
#     'asin': _asin,
#     'acos': _acos,
#     'atan': _atan,
#     'deg': math.degrees,
#     'rad': math.radians,
# })

# def safe_eval(expr: str):
#     """
#     avalia expressão matematica simples usando apenas nomes e funções em safe_symbols.
#     não habilita builtins, usa eval com um dicionario restrito.
#     """
#     local_safe=dict(safe_symbols)
#     try:
#         value = eval(expr, {"__builtins__": None}, local_safe)
#         return value
#     except Exception as e:
#         raise

# def print_help():
#     print("""
# Calculadora Cientifica (CLI)
# Digite expressões matemáticas usando:
# - operadores: +, -, *, /, %, //, **, ()
# - constantes: pi,e
# - funções: sin(x), cos(x), tan(x), asin(x), acos(x), atan(X), sqrt(x), ln(x),
#            log10, exp(x), fact(x) ou factorial(x), abs(x), pow(x,y)
# Comandos especiais:
# - mode            : mostra modo atual (RAD ou DEG)
# - mode rad        : define modo radiano
# - mode deg        : define modo graus
# - history         : mostra historico das ultimas operações
# - clear           : limpa a tela
# - help            : mostra esta ajuda
# - quit / exit     : sai da calculadora

# Exemplos:
#     sin(pi/2)
#     cos(60)
#     2**10 + sqrt(15)
#     ln(e)
#     fact(5)
# """)

# def repl():
#     global mode_radians
#     print('Calculadora Cientifica - fast e simple')
#     print('Digite "help" para ver comandos. Modo inicial:', "RAD" if 
#           mode_radians else "DEG")
#     while True:
#         try:
#             raw = input('calc> ').strip()
#         except (EOFError, KeyboardInterrupt):
#             print('\nSaindo...')
#             break

#         if not raw:
#             continue

#         cmd = raw.lower().strip()
#         if cmd in ('quit', 'exit'):
#             print('Saindo...')
#             break
#         if cmd == 'help':
#             print_help()
#             continue
#         if cmd == 'history':
#             if not history:
#                 print('Historico vazio.')
#             else:
#                 for i, (expr, result) in enumerate(history, 1):
#                     print(f'{i}: {expr} => {result}')
#             continue
#         if cmd.startswith('mode'):
#             parts = cmd.split()
#             if len(parts) == 1:
#                 print('Modo atual:', 'RAD' if mode_radians else 'DEG')
#             elif parts[1] in ('rad', 'r'):
#                 mode_radians = False
#                 print('modo alterado para DEG (graus).')
#             else:
#                 print('Uso: mode [rad|deg]')
#             continue
#         if cmd == 'clear':
#             import os
#             os.system('cls' if os.name == 'nt' else 'clear')
#             continue

#         try:
#             result = safe_eval(raw)
#         except Exception as e:
#             print('Erro ao avaliar expressão: ', e)
#             continue

#         history.append((raw, result))
#         print('=>', result)

# if __name__ == "__main__":
#     repl()


# Implemente um programa que leia logs e filtre os erros.


# def filtrar_erros(arquivo_log, arquivo_saida=None):
#     """
#     le um arquivo de log e filtra apenas as linhas que contem erros.

#     :param arquivo_log: caminho do arquivo do log de entrada
#     :param arquivo_saida: (opcional) caminho do arquivo onde salvar os erros.
#     """

#     try:
#         with open(arquivo_log, 'r', encoding='utf-8') as lor:
#             linhas = lor.readlines()
#     except FileNotFoundError:
#         print(f'Arquivo {arquivo_log} não encontrado')
#         return
    
#     erros = [linha for linha in linhas if 'ERROR' in linha.upper() or 'ERRO'
#              in linha.upper()]
    
#     if not erros:
#         print('nenhum erro encontrado no log')
#         return
    
#     print('=== Erros encontrados ===')
#     for e in erros:
#         print(e.strip())

#     if arquivo_saida:
#         with open(arquivo_saida, 'w', encoding='utf-8') as lor:
#             lor.writelines(erros)
#         print(f'\nErros salvos em {arquivo_saida}')

# if __name__ == '__main__':
#     filtrar_erros('app.log', "erros.log")


# Crie uma classe Carro com atributos e métodos (acelerar, frear).


# class carro:
#     def __init__(self, marca, modelo, ano, velocidade_maxima):
#         self.marca = marca
#         self.modelo = modelo
#         self.ano = ano
#         self.velocidade_maxima = velocidade_maxima
#         self.velocidade_atual = 0

#     def acelerar(self, incremento):
#         """
#         Aumenta a velocidade do carro, sem ultrapassar a velocidade maxima
#         """
#         if self.velocidade_atual + incremento > self.velocidade_maxima:
#             self.velocidade_atual = self.velocidade_maxima
#         else:
#             self.velocidade_atual += incremento
#         print(f'O carro acelerou. velocidade atual: {self.velocidade_atual} km/h')

#     def frear(self, decremento):
#         """diminui a velocidade do carro, sem ficar negativo"""
#         if self.velocidade_atual -  decremento < 0:
#             self.velocidade_atual = 0
#         else:
#             self.velocidade_atual -= decremento
#         print(f'O carro freiou. velocidade atual: {self.velocidade_atual} km/h')

#     def __str__(self):
#         return f'{self.marca} {self.modelo} ({self.ano}) - velocidade: {self.velocidade_atual} km/h' 

# if __name__ =='__main__':
#     lorn = carro('bolo', 'bolota', '1425', 300)
#     print(lorn)
#     lorn.acelerar(100)
#     lorn.acelerar(50)
#     lorn.acelerar(150)
#     lorn.frear(50)
#     lorn.frear(100)
#     print(lorn)


# Leia uma lista de dicionários (JSON) e filtre usuários maiores de 18 anos.


# import json

# def filtrar_maiores(json_usuarios):
#     """
#     recebe uma string JSON com lista de usuarios e retorna apenas os maiores de 
#     18 anos.
#     """
#     try:
#         usuarios = json.loads(json_usuarios)
#     except json.JSONDecodeError:
#         print('Erro: JSON inválido')
#         return []
    
#     maiores = [u for u in usuarios if u.get('idade', 0) > 18]
#     return maiores

# if __name__ == "__main__":
#     dados = """
#     [
#     {"nome": "ana", "idade": 17},
#     {"nome": "Carlos", "idade": 20},
#     {"nome": "joe", "idade": 21},
#     {"nome": "meruchan", "idade": 27}
#     ]
# """
#     maiores = filtrar_maiores(dados)

#     print('=== usuarios maiores de 18 anos ===')
#     for usuario in maiores:
#         print(f" {usuario['nome']} - {usuario['idade']} anos")


# Crie um gerador que produza a sequência de números primos.


# from itertools import islice

# def primos():
#     primo = []
#     l =2
#     while True:
#         eprime = True
#         for p in primo:
#             if p * p > l:
#                 break
#             if l % p == 0:
#                 eprime = False
#                 break
#         if eprime:
#             primo.append(l)
#             yield l
#         l += 1
            
# lar = primos()
# print(list(islice(lar, 100)))


# Implemente uma função que use reduce para multiplicar todos os números.


# from functools import reduce
# import operator

# def multi(num):
#     if not num:
#         return None
#     return reduce(operator.mul, num)

# print(multi([6,3,21,68,4]))
# print(multi([6,4]))


# Crie uma agenda em JSON (inserir, atualizar, remover contatos).


# import json
# import os

# arqagenda = 'agenda.json'

# def carregar_agenda():
#     if os.path.exists(arqagenda):
#         with open(arqagenda, 'r', encoding='utf-8') as f:
#             try:
#                 return json.load(f)
#             except json.JSONDecodeError:
#                 return []
#     return []

# # --- Função para salvar a agenda ---
# def salvar_agenda(agenda):
#     with open(arqagenda, 'w', encoding='utf-8') as f:
#         json.dump(agenda, f, indent=4, ensure_ascii=False)

# # --- Inserir contato ---
# def inserir_contato(nome, telefone, email=None):
#     agenda = carregar_agenda()

#     for contato in agenda:
#         if contato['nome'].lower() == nome.lower():
#             print(f"contato '{nome}' já existe.")
#             return
#      # adiciona novo contato
#     agenda.append({'nome': nome, 'telefone': telefone, 'email': email})
#     salvar_agenda(agenda)
#     print(f"contato '{nome}' inserido com sucesso!")

# # --- Atualizar contato ---
# def atualizar_contato(nome, telefone=None, email=None):
#     agenda = carregar_agenda()
#     for contato in agenda:
#         if contato['nome'].lower() == nome.lower():
#             if telefone:
#                 contato['telefone'] = telefone
#             if email is not None:
#                 contato['email'] = email
#             salvar_agenda(agenda)
#             print(f"contato '{nome}' atualizado")
#             return
#     print(f"contato '{nome}' não encontrado")

# # --- Remover contato ---
# def remover_contato(nome):
#     agenda =carregar_agenda()
#     nova_agenda = [c for c in agenda if c ['nome'].lower() != nome.lower()]
#     if len(nova_agenda) == len(agenda):
#         print(f"contato '{nome}' não encontrado.")
#         return
#     salvar_agenda(nova_agenda)
#     print(f"contato '{nome}' removido!")

# # --- Listar contatos ---
# def listar_contatos():
#     agenda = carregar_agenda()
#     if not agenda:
#         print('agenda vazia.')
#         return
#     for c in agenda:
#         print(f"Nome: {c['nome']}, Telefone: {c['telefone']}, Email: {c.get('email', '-')}")

# if __name__ == '__main__':
#     inserir_contato('ana', '99999-0001', 'anabombas@gmail.com')
#     inserir_contato('carlos', '99999-0001')
#     listar_contatos()
#     atualizar_contato('ana', "88888-1171")
#     remover_contato('carlos')
#     listar_contatos()


# Implemente um sistema de notas de alunos em CSV e calcule médias.


# import csv

# def salvar_notas(nome_arquivo, notas):
#     with open(nome_arquivo, 'w', newline='', encoding='utf-8') as arq:
#         escritor = csv.DictWriter(arq, fieldnames=['nome', 'nota1', 'nota2', 'nota3'])
#         escritor.writeheader()
#         escritor.writerows(notas)

# def carregar_notas(nome_arquivo):
#     with open(nome_arquivo, 'r', encoding='utf-8') as arq:
#         leitor = csv.DictReader(arq)
#         return list(leitor)
    
# def calcular_medias(notas):
#     medias = []
#     for aluno in notas:
#         n1 = float(aluno['nota1'])
#         n2 = float(aluno['nota2'])
#         n3 = float(aluno['nota2'])
#         media =(n1+n2+n3) / 3
#         medias.append({'nome': aluno['nome'], 'media': round(media,2)})
#     return medias

# if __name__ == "__main__":
#     alunos = [
#     {'nome': 'ana', 'nota1': 8, 'nota2': 7.5, 'nota3':9},
#     {'nome': 'carlos', 'nota1': 6, 'nota2': 8, 'nota3':5},
#     {'nome': 'joão', 'nota1': 6, 'nota2': 7.5, 'nota3':10},
#     ]

#     salvar_notas('notas.csv', alunos)
#     dados = carregar_notas('notas.csv')
#     medias = calcular_medias(dados)

#     print('=== Medias dos aluno ===')
#     for aluno in medias:
#         print(f' {aluno['nome']}: {aluno['media']}')


# Crie uma classe Pessoa com __str__ para imprimir informações.


# class pessoa:
#     def __init__(self, nome, idade, email):
#         self.nome = nome
#         self.idade = idade
#         self.email = email

#     def __str__(self):
#         return f"Nome: {self.nome}, Idade: {self.idade}, Email: {self.email}"

# if __name__ == '__main__':
#     p1 = pessoa('ana', '25', 'ana@gmail.com')
#     p2 = pessoa('anatoli', '26', 'anore@gmail.com')

# print(p1)
# print(p2)


# Crie um programa que envie e-mails automaticamente.


# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# EMAIL = 'maxwellgodhero@gmail.com'
# SENHA = 'ispr unmq xmex poda'
# DESTINATARIO = 'maxwellfernandesbanc@gmail.com'

# msg = MIMEMultipart()
# msg['from'] = EMAIL
# msg['to'] = DESTINATARIO
# msg['subject'] = 'teste de envio automatico'

# corpo = 'você está com virus'
# msg.attach(MIMEText(corpo, 'plain'))

# try:
#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.starttls()
#     server.login(EMAIL, SENHA)
#     server.sendmail(EMAIL, DESTINATARIO, msg.as_string())
#     server.quit()
#     print('Email enviado com sucesso!')
# except Exception as e:
#     print('Erro ao enviar email:', e)


# Implemente um sistema de login simples com dicionário (usuário/senha).


# usuarios = {
#     'admin': '1234',
#     'joao': 'senha1234',
#     'maria': 'abcde'
# }

# def login(nome_usuario, senha):
#     if nome_usuario in usuarios:
#         if usuarios[nome_usuario] == senha:
#             return True
#         else:
#             return False
        
# def main():
#     print('===sistema de login===')

#     nome = input('digite nome de usuario: ')
#     senha = input('digite uma senha: ')

#     if login(nome, senha):
#         print(f'Login bem sucedido! Bem bindo, {nome}')
#     else:
#         print('usuario não incorreto.')


# if __name__ == '__main__':
#     main()


# Crie uma API que retorne dados de clima em tempo real.


# import requests
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# API_KEY = ''

# @app.route('/clima', methods=['GET'])
# def clima():
#     cidade = request.args.get('cidade', 'são paulo')
#     url = f'http://api.openweathermap.org/data/2.5/weather?q={cidade}&appid={API_KEY}&lang=pt_br&units=metric'

#     resposta = requests.get(url)
#     if resposta.status_code ==200:
#         dados = resposta.json()
#         resultado = {
#             'cidade': dados['name'],
#             'temperatura': dados['main']['temp'],
#             'sensacao': dados['main']['feels_like'],
#             'umidade': dados['main']['humidity'],
#             'descricao': dados['weather'][0]['description']
#         }
#         return jsonify(resultado)
#     else:
#         return jsonify({'erro': 'não foi possivel obter o clima'}), 400
    
# if __name__=='__main__':
#     app.run(port=5000, debug=True)


# Implemente uma simulação de dados de sensor em streaming.


# import time
# import random 
# from flask import Flask, Response


# app = Flask(__name__)

# def gerar_dados_sensor():
#     while True:
#         temperatura = round(random.uniform(20.0,30.0), 2)
#         umidade = round(random.uniform(40.0, 70.0), 2)
#         dado = f"data: {{'temperatura': {temperatura}, 'umidade': {umidade}}}\n\n"
#         yield dado
#         time.sleep(1)

# @app.route('/stream')
# def stream():
#     return Response(gerar_dados_sensor(), mimetype = 'text/event-stream')

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)


# Crie um programa que converta números romanos em inteiros.


# def romano_para_inteiro(romano):
#     valores = {
#         'I': 1,
#         'V': 5,
#         'X': 10,
#         'L': 50,
#         'C': 100,
#         'D': 500,
#         'M': 1000
#     }
#     total = 0
#     i=0
#     while i < len(romano):
#         if romano[i] not in valores:
#             raise ValueError(f'caracter invalido: {romano[i]}')
#         if i +1< len(romano) and valores[romano[i]] < valores[romano[i+1]]:
#             total += valores[romano[i+1]] - valores[romano[i]]
#             i += 2
#         else:
#             total += valores[romano[i]]
#             i += 1
#     return total

# numero_romano = input('Digite um número romano: ').upper()
# resultado = romano_para_inteiro(numero_romano)
# print(f'O numero romano {numero_romano} equivale a {resultado}')


# Implemente uma pilha (stack) com lista em Python.


# class Pilha:
#     def __init__(self):
#         self.itens = []

#     def esta_vazia(self):
#         return len(self.itens) == 0
    
#     def empilhar(self, item):
#         self.itens.append(item)

#     def desempilhar(self):
#         if self.esta_vazia():
#             raise IndexError('Desempilhar de uma pilha vazia')
#         return self.itens.pop()
    
#     def topo(self):
#         if self.esta_vazia():
#             raise IndexError('topo de umapilha vazia')
#         return self.itens[-1]
    
#     def tamanho(self):
#         return len(self.itens)
    
#     def __str__(self):
#         return f'Pilha: {self.itens}'
    
# pilha = Pilha()
# print('Pilha es´ta vazia?', pilha.esta_vazia())

# pilha.empilhar(10)
# pilha.empilhar(20)
# pilha.empilhar(55)
# print(pilha)

# print('topo da pilha: ', pilha.topo())

# print('Desempilhando: ', pilha.desempilhar())
# print(pilha)

# print('Tamanho da pilha: ', pilha.tamanho())


# Crie um programa que salve logs em arquivo com logging.


# import logging
# import os

# if not os.path.exists('logs'):
#     os.makedirs('logs')

# logging.basicConfig (filename='logs/meu_log.log',
#                      format ='%(asctime)s = %(levelname)s - %(message)s',
#                      datefmt = '%y-%m-%d %H:%M:%S',
#                      encoding = 'utf-8'
# )
# logging.debug('isso é um log de DEBUG')
# logging.info('Isso é um log de INFO')
# logging.warning('Isso é um log de WARNING')
# logging.error('Isso é um log de ERROR')
# logging.critical('Isso é um log de CRITICAL')

# def dividir(a,b):
#     try:
#         resultado = a / b
#         logging.info(f'Divisão realizada: {a} / {b} = {resultado}')
#         return resultado
#     except ZeroDivisionError:
#         logging.error(f'Tentativa de dividir {a} por zero!')
#         return None
    
# dividir(10, 2)
# dividir(5,0)

# print("logs salvos em 'logs/meu_log.log'.")


# Implemente uma árvore binária com inserção, busca e remoção.


# class Node:
#     def __init__(self, valor):
#         self.valor = valor
#         self.esq = None
#         self.dir = None

# class ArvoreBinaria:
#     def __init__(self):
#         self.raiz = None

#     def inserir(self, valor):
#         if self.raiz is None:
#             self.raiz = Node(valor)
#         else:
#             self._inserir(valor, self.raiz)

#     def _inserir(self, valor, nodo):
#         if valor < nodo.valor:
#             if nodo.esq is None:
#                 nodo.esq = Node(valor)
#             else:
#                 self._inserir(valor, nodo.esq)

#         else:
#             if nodo.dir is None:
#                 nodo.dir = Node(valor)
#             else:
#                 self._inserir(valor, nodo.dir)

#     def buscar(self,valor):
#         "Busca um valor na árvore"
#         return self._buscar(valor, self.raiz)
    
#     def _buscar(self, valor, nodo):
#         if nodo is None:
#             return False
#         if valor == nodo.valor:
#             return True
#         elif valor < nodo.valor:
#             return self._buscar(valor, nodo.esq)
#         else:
#             return self._buscar(valor, nodo.dir)
        
#     def remover(self, valor):
#         self.raiz = self._remover(valor, self.raiz)

#     def _remover(self, valor, nodo):
#         if nodo is None:
#             return nodo
#         if valor < nodo.valor:
#             nodo.esq = self._remover(valor, nodo.esq)
#         elif valor > nodo.valor:
#             nodo.dir = self._remover(valor, nodo.dir)
#         else:
#             if nodo.esq is None and nodo.dir is None:
#                 return None
#             elif nodo.esq is None:
#                 return nodo.dir
#             elif nodo.dir is None:
#                 return nodo.esq
#             menor = self._minimo(nodo.dir)
#             nodo.valor = menor.valor
#             nodo.dir = self._remover(menor.valor, nodo.dir)
#         return nodo
    
#     def _minimo(self, nodo):
#         atual = nodo
#         while atual.esq is not None:
#             atual = atual.esq
#         return atual
    
#     def em_ordem(self, nodo=None):
#         if nodo is None:
#             nodo = self.raiz
#             if nodo is None:
#                 return
            
#         if nodo.esq:
#             self.em_ordem(nodo.esq)
           
#         print(nodo.valor, end=' ')
            
#         if nodo.dir:
#             self.em_ordem(nodo.dir)


# if __name__ == '__main__':
#     arv = ArvoreBinaria()
#     for v in [50, 30,70,20,40,60,80]:
#         arv.inserir(v)

#     print('Arvore em ordem:')
#     arv.em_ordem()
#     print('\nBuscar 40:', arv.buscar(40))
#     print('Buscar 100:', arv.buscar(100))

#     print('\nRemover 30 (tem 1 filhos): ')
#     arv.remover(30)
#     arv.em_ordem()

#     print('\nRemover 50 (tem 2 filhos): ')
#     arv.remover(50)
#     arv.em_ordem()


# Ordene uma lista grande usando quick sort sem usar sort().


# import random
# import time

# def _median_of_three(a,b,c):
#     if (a <= b and b <= c) or (c <= b and b <= a):
#         return b
#     if (b <= a and a <= c) or (c <= a and a <= b):
#         return a
#     return c

# def _partition_hoare(arr, lo, hi):
#     mid = (lo + hi) // 2
#     pivot = _median_of_three(arr[lo], arr[mid], arr[hi])
#     i = lo -1
#     j = hi +1
#     while True:
#         i += 1
#         while arr[i] < pivot:
#             i += 1
#         j -= 1
#         while arr[j] > pivot:
#             j -= 1
#         if i >= j:
#             return j
#         arr[i], arr[j] = arr[j], arr[i]

# def quicksort_inplace(arr):
#     if len(arr) < 2:
#         return arr
#     stack = [(0, len(arr) - 1)]
#     while stack:
#         lo, hi = stack.pop()
#         if lo < hi:
#             p = _partition_hoare(arr, lo, hi)
#             if p - lo > hi - (p + 1):
#                 stack.append((lo,p))
#                 stack.append((p + 1, hi))
#             else:
#                 stack.append((p + 1, hi))
#                 stack.append((lo, p))
#     return arr

# if __name__ == '__main__':
#     n = 1_000_000
#     arr = list(range(n))
#     random.shuffle(arr)

#     arr_copy = arr.copy()

#     t0 = time.time()
#     quicksort_inplace(arr)
#     t1 = time.time()
#     print('Quicksort in-place terminou em {:.3f}s'.format(t1 - t0))

#     if arr == sorted(arr_copy):
#         print('ok: ordenado corretamente')
#     else:
#         print('erro: resultado incorreto')


# Resolva um labirinto representado por matriz usando BFS ou DFS.


# from collections import deque

# def bfs_labirinto(matriz, inicio, fim):
#     n,m = len(matriz), len(matriz[0])
#     visitado = [[False]*m for _ in range(n)]
#     pai = {inicio: None}

#     fila = deque([inicio])
#     visitado[inicio[0]][inicio[1]] = True

#     movimentos = [(1,0), (-1,0), (0,1), (0,-1)]

#     while fila:
#         x,y = fila.popleft()
#         if (x,y) == fim:
#             caminho = []
#             atual = fim
#             while atual is not None:
#                 caminho.append(atual)
#                 atual = pai[atual]
#             return caminho[::-1]
        
#         for dx, dy in movimentos:
#             nx, ny = x+dx, y+dy
#             if 0 <= nx < n and 0 <= ny < m and not visitado[nx][ny] and matriz[nx][ny] == 0:
#                 fila.append((nx,ny))
#                 visitado[nx][ny] = True
#                 pai[(nx,ny)] = (x,y)
#     return None

# if __name__ == '__main__':
#     labirinto = [
#         [0,0,1,0,0],
#         [1,0,1,0,1],
#         [0,0,0,0,0],
#         [0,1,1,1,0],
#         [0,0,0,1,0]
#     ]

#     inicio = (0,0)
#     fim = (4,4)

#     caminho = bfs_labirinto(labirinto, inicio, fim)

#     if caminho:
#         print('Caminho encontrado: ', caminho)
#     else:
#         print('Não existe caminho.')


# Compare tempos de execução de bubble sort, quick sort e merge sort.


# import random
# import time

# def bubble_sort(arr):
#     n = len(arr)
#     for i in range(n):
#         for j in range(0, n - i -1):
#             if arr[j] > arr[j + 1]:
#                 arr[j], arr[j + 1] = arr[j + 1], arr[j]
#     return arr

# def quick_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivo = arr[len(arr) // 2]
#     esquerda = [x for x in arr if x < pivo]
#     meio = [x for x in arr if x == pivo]
#     direita = [x for x in arr if x > pivo]
#     return quick_sort(esquerda) + meio + quick_sort(direita)

# def merge_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     meio = len(arr) // 2
#     esquerda = merge_sort(arr[:meio])
#     direita = merge_sort(arr[meio:])
#     return merge(esquerda, direita)

# def merge(esquerda, direita):
#     resultado = []
#     i = j = 0
#     while i < len(esquerda) and j < len(direita):
#         if esquerda[i] < direita[j]:
#             resultado.append(esquerda[i])
#             i += 1
#         else:
#             resultado.append(direita[j])
#             j += 1
#     resultado.extend(esquerda[i:])
#     resultado.extend(direita[j:])
#     return resultado

# tamanho = 10000
# lista = [random.randint(0, 100000) for _ in range(tamanho)]

# lista_bubble = lista.copy()
# lista_quick = lista.copy()
# lista_merge = lista.copy()

# inicio = time.time()
# bubble_sort(lista_bubble)
# print(f'Bubble Sort: {time.time() - inicio:.4f} segundos')

# inicio = time.time()
# quick_sort(lista_quick)
# print(f'Quick Sort: {time.time() - inicio:.4f} segundos')

# inicio = time.time()
# merge_sort(lista_merge)
# print(f'Merge Sort: {time.time() - inicio:.4f} segundos')


# Implemente busca binária em uma lista ordenada.


# def busca_binaria(lista, alvo):
#     esquerda = 0
#     direita = len(lista) -1

#     while esquerda <= direita:
#         meio = (esquerda + direita) // 2
#         chute = lista[meio]

#         print(f'checando indice {meio}: {chute}')

#         if chute == alvo:
#             return meio
#         elif chute < alvo:
#             esquerda = meio +1
#         else:
#             direita =  meio -1
#     return -1

# if __name__ == '__main__':
#     numeros = list(range(0, 100, 5))
#     print('lista: ', numeros)

#     alvo = int(input('Digite o Número para buscar: '))
#     indice = busca_binaria(numeros, alvo)

#     if indice != -1:
#         print(f' Valor {alvo} encontrado no indice {indice}.')
#     else:
#         print(f' Valor {alvo} não encontrado.')


# Calcule a complexidade de diferentes funções e explique O(n), O(log n), O(n²).


# # O(n) - Linear
# def soma_lista(lista):
#     total = 0
#     for x in lista:
#         total += x
#     return total

# lista = [i for i in range(10**6)]
# print(soma_lista(lista))


# # O(log n) = Logaritmico
# def busca_binaria(lista, alvo):
#     esquerda, direita = 0, len(lista) - 1
#     while esquerda <= direita:
#         meio = (esquerda + direita) // 2
#         if lista[meio] == alvo:
#             return meio
#         elif lista[meio] < alvo:
#             esquerda = meio + 1
#         else:
#             direita = meio - 1
#     return -1

# lista = list(range(1000000))
# print(busca_binaria(lista, 987456))


# # O(n²) - Quadrática
# def bubble_sort(arr):
#     n = len(arr)
#     for i in range(n):
#         for j in range(0, n-i-1):
#             if arr[j] > arr[j+1]:
#                 arr[j], arr[j+1] = arr[j+1], arr[j]
#     return arr

# arr = [5, 2, 9, 1, 5, 6]
# print(bubble_sort(arr))


# Crie uma classe Banco com clientes, contas e métodos de depósito/saque.


# class Cliente:
#     def __init__(self, nome, cpf):
#         self.nome = nome
#         self.cpf = cpf

#     def __str__(self):
#         return f'{self.nome} (CPF: {self.cpf})'
    
# class Conta:
#     def __init__(self, numero, cliente, saldo=0):
#         self.numero = numero
#         self.cliente = cliente
#         self.saldo = saldo

#     def depositar(self, valor):
#         if valor <= 0:
#             print('valor de deposito invalido.')
#             return
#         self.saldo += valor
#         print(f'Depósito de R${valor:.2f} realizado com sucesso.')

#     def sacar(self, valor):
#         if valor <= 0:
#             print('valor de saque inválido.')
#             return
#         if valor > self.saldo:
#             print('saldo insuficiente para saque.')
#             return
#         self.saldo -= valor
#         print(f'saque de R${valor:.2f} realizado com sucesso.')

#     def __str__(self):
#         return f'Conta {self.numero} | Cliente: {self.cliente.nome} | Saldo: R${self.saldo:.2f}'
    
# class Banco:
#     def __init__(self, nome):
#         self.nome = nome
#         self.clientes = []
#         self.contas = []
    
#     def adicionar_cliente(self, cliente):
#         self.clientes.append(cliente)
#         print(f'Cliente {cliente.nome} adicionado ao banco.')

#     def abrir_conta(self, cliente, numero_conta):
#         if cliente not in self.clientes:
#             print('Cliente não encontrado no banco.')
#             return
#         conta = Conta(numero_conta, cliente)
#         self.contas.append(conta)
#         print(f'Conta {numero_conta} criada para {cliente.nome}.')
#         return conta
    
#     def buscar_conta(self, numero):
#         for conta in self.contas:
#             if conta.numero ==  numero:
#                 return conta
#         print('conta não encontrada.')
#         return None
    
#     def __str__(self):
#         return f'Banco {self.nome} | {len(self.clientes)} clientes | {len(self.contas)} contas'
    
# banco = Banco('banco gpt')

# cliente1 = Cliente('MAXWELL', '134-455-224-90')
# banco.adicionar_cliente(cliente1)

# conta1 = banco.abrir_conta(cliente1, 1001)

# conta1.depositar(1000)
# conta1.sacar(250)
# conta1.sacar(900)
# print(conta1)
# print(banco)


# Implemente classes de veículos (Carro, Caminhão, Moto) usando herança e polimorfismo.


# class Veiculo:
#     def __init__(self, marca, modelo, ano):
#         self.marca = marca
#         self.modelo = modelo
#         self.ano = ano
#         self.ligado = False

#     def ligar(self):
#         if self.ligado:
#             print(f'O {self.modelo} já está ligado.')
#         else:
#             self.ligado = True
#             print(f'O {self.modelo} foi ligado.')

#     def desligar(self):
#         if not self.ligado:
#             print(f'O {self.modelo} já está desligado.')
#         else:
#             self.ligado = False
#             print(f'O {self.modelo} foi desligado.')

#     def acelerar(self):
#         if self.ligado:
#             print(f'O {self.modelo} está acelerando...')
#         else:
#             print(f'Não é possivel acelerar - o {self.modelo} desligado')

#     def __str__(self):
#         status = 'Ligado' if self.ligado else 'Desligado'
#         return f'{self.marca} {self.modelo} ({self.ano}) - {status}'
    
# class Carro(Veiculo):
#     def __init__(self, marca, modelo, ano, portas):
#         super().__init__(marca, modelo, ano)
#         self.portas = portas

#     def acelerar(self):
#         if self.ligado:
#             print(f'O carro {self.modelo} acelera suavemente!')
#         else:
#             print(f'O carro {self.modelo} precisa ser ligado primeiro.')

#     def __str__(self):
#         return f'Carro: {super().__str__()}  Portas: {self.portas}'
    
# class Caminhao(Veiculo):
#     def __init__(self, marca, modelo, ano, carga_maxima):
#         super().__init__(marca, modelo, ano)
#         self.carga_maxima = carga_maxima

#     def acelerar(self):
#         if self.ligado:
#             print(f'O caminhão {self.modelo} acelera com força, carregando até {self.carga_maxima}kg!')
#         else:
#             print(f'Ligue o caminhão {self.modelo} antes de acelerar.')

#     def __str__(self):
#         return f'Caminhão: {super().__str__()} | Carga maxima: {self.carga_maxima}kg'
    
# class Moto(Veiculo):
#     def __init__(self, marca, modelo, ano, cilindradas):
#         super().__init__(marca, modelo, ano)
#         self.cilindradas = cilindradas

#     def acelerar(self):
#         if self.ligado:
#             print(f'A moto {self.modelo} acelera rapidamente com {self.cilindradas}cc')
#         else:
#             print(f'A moto {self.modelo} está desligada, não pode acelerar.')

#     def __str__(self):
#         return f'Moto: {super().__str__()} | {self.cilindradas}cc'
    
# veiculos = [
#     Carro('toyota', 'corolla', 2022, 5),
#     Caminhao('Voldo', 'gdr56', 2053, 60500),
#     Moto('minamaha','MT-55', 2034, 5786)

# ]

# for v in veiculos:
#     print('\n---')
#     print(v)
#     v.ligar()
#     v.acelerar()
#     v.desligar()


# Crie um sistema escolar com alunos, professores e turmas usando composição de classes.


# class Aluno:
#     def __init__(self, nome, matricula):
#         self.nome = nome
#         self.matricula = matricula
    
#     def __str__(self):
#         return f'Aluno: {self.nome} (Matricula: {self.matricula})'
    
# class Professor:
#     def __init__(self, nome, disciplina):
#         self.nome = nome
#         self.disciplina = disciplina

#     def __str__(self):
#         return f'Professor: {self.nome} | Disciplina: {self.disciplina}'
    
# class Turma:
#     def __init__(self, codigo, professor):
#         self.codigo = codigo
#         self.professor = professor
#         self.alunos =[]

#     def adicionar_aluno(self, aluno):
#         self.alunos.append(aluno)
#         print(f'Aluno {aluno.nome} adicionado a turma {self.codigo}')

#     def listar_alunos(self):
#         if not self.alunos:
#             print(f'Nenhum aluno na turma {self.codigo}.')
#         else:
#             print(f'Alunos da turma {self.codigo}:')
#             for aluno in self.alunos:
#                 print(f' - {aluno.nome}')

#     def __str__(self):
#         return (f'Turma {self.codigo} | Professor: {self.professor.nome}'
#                 f' | Total de alunos: {len(self.alunos)}')
    
# class Escola:
#     def __init__(self, nome):
#         self.nome = nome
#         self.turmas = []

#     def adicionar_turma(self, turma):
#         self.turmas.append(turma)
#         print(f'Turma {turma.codigo} adicionada á escola {self.nome}.')

#     def listar_turmas(self):
#         print(f'\nEscola: {self.nome} - Turmas cadastradas: ')
#         for turma in self.turmas:
#             print(f' - {turma}')

#     def __str__(self):
#         return f'Escola {self.nome} | Total de turmas: {len(self.turmas)}'
    
# prof = Professor('Carlos Silva', 'Matematica')

# a1 = Aluno('Ana Souza', 'A001')
# a2 = Aluno('Bruno lima', 'A002')
# a3 = Aluno('Carla', 'A004')

# turma1 = Turma('T1', prof)
# turma1.adicionar_aluno(a1)
# turma1.adicionar_aluno(a2)
# turma1.adicionar_aluno(a3)

# escola = Escola('RAPADURA ESCOLAE')
# escola.adicionar_turma(turma1)

# print('\n===Detalhes===')
# turma1.listar_alunos()
# escola.listar_turmas()
# print(escola)
    

# Utilize métodos especiais como __str__, __repr__, __eq__ em classes criadas.


# class Aluno:
#     def __init__(self, nome, matricula):
#         self.nome = nome
#         self.matricula = matricula

#     def __str__(self):
#         return f'Aluno: {self.nome} (Matricula: {self.matricula})'
    
#     def __repr__(self):
#         return f'Aluno:("{self.nome}", "{self.matricula}")'
    
#     def __eq__(self, outro):
#         if isinstance(outro, Aluno):
#             return self.matricula == outro.matricula
#         return False
    
# class Professor:
#     def __init__(self, nome, disciplina):
#         self.nome = nome
#         self.disciplina = disciplina

#     def __str__(self):
#         return f'Professor: {self.nome} | Disciplina: {self.disciplina}'
    
#     def __repr__(self):
#         return f'Professor: "{self.nome}", "{self.disciplina}" '
    
#     def __eq__(self, outro):
#         if isinstance(outro, Professor):
#             return self.nome == outro.nome and self.disciplina == outro.disciplina
#         return False
    
# class Turma:
#     def __init__(self, codigo, professor):
#         self.codigo = codigo
#         self.professor = professor
#         self.alunos = []

#     def adicionar_aluno(self, aluno):
#         if aluno in self.alunos:
#             print(f'Aluno {aluno.nome} já está na turma {self.codigo}.')
#         else:
#             self.alunos.append(aluno)
#             print(f'Aluno {aluno.nome} adicionado á turma {self.codigo}')

#     def __str__(self):
#         return (f'Turma {self.codigo} | '
#                 f'Professor: {self.professor.nome} | ' 
#                 f'Alunos: {len(self.alunos)}'
#     )

#     def __repr__(self):
#         return f'Turma("{self.codigo}, {repr(self.professor)}")'
    
# a1 = Aluno('Ana souza', 'A1')
# a2 = Aluno('Bruninha', 'A6')
# a3 = Aluno('Amendoas naros', 'A1')

# prof = Professor('Carlos silva', 'Matematica')

# turma = Turma('T1', prof)
# turma.adicionar_aluno(a1)
# turma.adicionar_aluno(a2)
# turma.adicionar_aluno(a3)

# print()
# print(a1)
# print(repr(a1))

# print()
# print(turma)
# print(repr(turma))


# Implemente propriedades com @property e encapsulamento de atributos.


# class contaBancaria:
#     def __init__(self, titular, saldo = 0):
#         self.titular = titular
#         self.__saldo = saldo

#     @property
#     def saldo(self):
#         return self.__saldo
    
#     @saldo.setter
#     def saldo(self, valor):
#         if valor < 0:
#             print('Não é permitido saldo negativo.')
#         else:
#             self.__saldo = valor

#     def depositar(self, valor):
#         if valor > 0:
#             self.__saldo += valor
#             print(f'Deposito de R${valor:.2f} realizado. Novo saldo: '
#                   f'R${self.__saldo:.2f}')
#         else:
#             print('Valor de deposito invalido.')

#     def sacar(self, valor):
#         if 0 < valor <= self.__saldo:
#             self.__saldo -= valor
#             print(f'Saque de R${valor:.2f} realizado.'
#                   f'Saldo restante: R${self.__saldo:.2f}')
#         else:
#             print('saque inválido ou saldo insuficiente')

# conta = contaBancaria('Maxwell', 400)

# print(conta.saldo)
# conta.depositar(200)
# conta.sacar(100)
# conta.saldo = 1000
# print(conta.saldo)
# conta.saldo = -500


# Crie uma classe abstrata e implemente subclasses que a estendem.


# from abc import ABC, abstractmethod

# class Animal(ABC):
#     def __init__(self, nome):
#         self.nome = nome

#     @abstractmethod
#     def falar(self):
#         pass

#     @abstractmethod
#     def mover(self):
#         pass

# class Cachorro(Animal):
#     def falar(self):
#         return f'{self.nome} diz: Au Au'
    
#     def mover(self):
#         return f'{self.nome} está correndo pelo quintal.'
    
# class Gato(Animal):
#     def falar(self):
#         return f'{self.nome} diz: Miau'
    
#     def mover(self):
#         return f'{self.nome} está pulando sobre o criar do mudo'
    
# class Passaro(Animal):
#     def falar(self):
#         return f'{self.nome} diz: Piu Piu'
    
#     def mover(self):
#         return f'{self.nome} está voando pelo céu'
    
# animais = [Cachorro('Rex'), Gato('Mimi'), Passaro('Piu')]

# for animal in animais:
#     print(animal.falar())
#     print(animal.mover())
#     print('---')


# Leia um CSV de vendas, calcule a média de vendas por produto e salve em outro CSV.


# import pandas as pd

# vendas = pd.read_csv('arquivo.csv')

# print('dados originais')
# print(vendas.head())

# media_vendas = vendas.groupby('Idade')['valor'].mean().reset_index()
# media_vendas.rename(columns={'valor': 'Media_Vendas'}, inplace=True)

# print('\nMedia de vendas por produto: ')
# print(media_vendas)

# media_vendas.to_csv('media_vendas.csv', index=False)
# print("\nArquivo 'media_vendas.csv' criado com sucesso!")


# Filtre um JSON de clientes para encontrar apenas os ativos ou com idade > 18.


# import json

# # Exemplo de arquivo JSON (poderia ser um arquivo .json externo também)
# dados_json = '''
# [
#     {"nome": "Ana", "idade": 17, "ativo": false},
#     {"nome": "Carlos", "idade": 25, "ativo": true},
#     {"nome": "Maria", "idade": 30, "ativo": false},
#     {"nome": "João", "idade": 15, "ativo": true}
# ]
# '''

# # Carrega o JSON em formato de lista de dicionários
# clientes = json.loads(dados_json)

# # Filtra clientes ativos ou com idade > 18
# clientes_filtrados = [
#     cliente for cliente in clientes
#     if cliente['ativo'] or cliente['idade'] > 18
# ]

# # Exibe o resultado formatado
# print("Clientes ativos ou maiores de 18:")
# for cliente in clientes_filtrados:
#     print(f"- {cliente['nome']} (Idade: {cliente['idade']}, Ativo: {cliente['ativo']})")


# Crie gráficos simples (matplotlib ou seaborn) a partir de dados de CSV.


# import pandas as pd
# import matplotlib.pyplot as plt

# # Lendo o arquivo CSV
# vendas = pd.read_csv('arquivo.csv')

# # Exibindo os dados
# print(vendas)

# # Criando um gráfico de barras
# plt.bar(vendas['Idade'], vendas['valor'])
# plt.title('Valor por Produto')
# plt.xlabel('Produto')
# plt.ylabel('Valor (R$)')
# plt.show()

# # Criando um gráfico de linha
# plt.plot(vendas['Produto'], vendas['Quantidade'], marker='o', color='green')
# plt.title('Quantidade vendida por produto')
# plt.xlabel('Produto')
# plt.ylabel('Quantidade')
# plt.grid(True)
# plt.show()


# OU


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Lendo o arquivo CSV
# vendas = pd.read_csv('vendas.csv')

# # Gráfico de barras
# sns.barplot(data=vendas, x='Produto', y='Valor', palette='viridis')
# plt.title('Valor por Produto (Seaborn)')
# plt.show()

# # Gráfico de dispersão
# sns.scatterplot(data=vendas, x='Quantidade', y='Valor', hue='Produto', s=100)
# plt.title('Relação entre Quantidade e Valor')
# plt.show()


# Use Numpy para criar matrizes, aplicar máscaras e operações matemáticas avançadas.


# import numpy as np

# # Criando uma matriz 3x3 com valores aleatórios entre 1 e 10
# matriz = np.random.randint(1, 11, (3, 3))
# print("Matriz original:")
# print(matriz)

# # 🧮 Operações matemáticas
# soma = np.sum(matriz)
# media = np.mean(matriz)
# raiz = np.sqrt(matriz)

# print("\nSoma dos elementos:", soma)
# print("Média dos elementos:", media)
# print("Raiz quadrada de cada elemento:")
# print(np.round(raiz, 2))

# # 🎯 Máscaras (filtros)
# # Criando uma máscara para elementos maiores que 5
# mascara = matriz > 5
# print("\nMáscara (True = valores > 5):")
# print(mascara)

# # Aplicando a máscara
# print("Valores maiores que 5:", matriz[mascara])

# # 🧠 Operações avançadas
# # Multiplicação de matrizes (produto escalar)
# outra_matriz = np.random.randint(1, 11, (3, 3))
# print("\nOutra matriz:")
# print(outra_matriz)

# produto_matricial = np.dot(matriz, outra_matriz)
# print("\nProduto matricial:")
# print(produto_matricial)

# # Transposta da matriz
# print("\nTransposta da matriz original:")
# print(matriz.T)

# # Determinante e inversa (somente se matriz for invertível)
# det = np.linalg.det(matriz)
# print("\nDeterminante:", round(det, 2))

# if det != 0:
#     inversa = np.linalg.inv(matriz)
#     print("Inversa da matriz:")
#     print(np.round(inversa, 2))
# else:
#     print("Esta matriz não é invertível (determinante = 0).")


# Leia e processe arquivos grandes linha por linha usando streaming.


# import csv
# from collections import defaultdict

# def calcular_media_vendas(arquivo_entrada, arquivo_saida):
#     # Dicionários para somar e contar
#     soma_vendas = defaultdict(float)
#     contagem = defaultdict(int)

#     # Lendo linha a linha (streaming)
#     with open(arquivo_entrada, 'r', encoding='utf-8', newline='') as f:
#         leitor = csv.DictReader(f)
#         for i, linha in enumerate(leitor, start=1):
#             try:
#                 produto = linha['produto']
#                 valor = float(linha['valor'])
#                 soma_vendas[produto] += valor
#                 contagem[produto] += 1
#             except (ValueError, KeyError):
#                 continue  # ignora linhas corrompidas

#             if i % 10000 == 0:
#                 print(f"Processadas {i} linhas...")

#     # Calcula médias
#     medias = {p: soma_vendas[p] / contagem[p] for p in soma_vendas}

#     # Salva resultado
#     with open(arquivo_saida, 'w', encoding='utf-8', newline='') as f:
#         escritor = csv.writer(f)
#         escritor.writerow(['produto', 'media_vendas'])
#         for produto, media in medias.items():
#             escritor.writerow([produto, f"{media:.2f}"])

#     print(f"\n✅ Arquivo salvo com sucesso: {arquivo_saida}")
#     print(f"Produtos processados: {len(medias)}")


# # Exemplo de uso
# if __name__ == "__main__":
#     calcular_media_vendas("vendas.csv", "media_vendas.csv")


# Crie um decorador que valide os parâmetros de uma função.


# from inspect import signature

# def validar_parametros(regras):
#     """
#     Decorador para validar tipos e condições de parâmetros.
    
#     Exemplo de uso:
#     @validar_parametros({
#         "nome": (str, lambda v: len(v) > 0),
#         "idade": (int, lambda v: v >= 18)
#     })
#     """
#     def decorador(func):
#         def wrapper(*args, **kwargs):
#             sig = signature(func)
#             argumentos = sig.bind(*args, **kwargs)
#             argumentos.apply_defaults()

#             for nome, (tipo, condicao) in regras.items():
#                 if nome in argumentos.arguments:
#                     valor = argumentos.arguments[nome]
                    
#                     # Valida tipo
#                     if not isinstance(valor, tipo):
#                         raise TypeError(
#                             f"'{nome}' deve ser do tipo {tipo.__name__}, mas recebeu {type(valor).__name__}."
#                         )
                    
#                     # Valida condição adicional
#                     if condicao and not condicao(valor):
#                         raise ValueError(f"'{nome}' possui valor inválido: {valor}.")
#             return func(*args, **kwargs)
#         return wrapper
#     return decorador

# @validar_parametros({
#     "nome": (str, lambda v: len(v) > 0),
#     "idade": (int, lambda v: v >= 18),
#     "saldo": (float, lambda v: v >= 0)
# })
# def criar_conta(nome, idade, saldo):
#     print(f"Conta criada com sucesso! Nome: {nome}, Idade: {idade}, Saldo: R${saldo:.2f}")


# Implemente um gerador que produza números primos infinitamente.


# def primos_infinitos():
#     """Generator infinito de números primos."""
#     primos = []
#     n = 2
#     while True:
#         # Verifica se n é divisível por algum primo anterior
#         if all(n % p != 0 for p in primos):
#             primos.append(n)
#             yield n
#         n += 1

# # Exemplo de uso:
# gen = primos_infinitos()
# for _ in range(10):  # Mostra os 10 primeiros primos
#     print(next(gen))


# Use funções lambda para manipular listas e DataFrames.


# import pandas as pd

# # Lista para armazenar dados
# pessoas = []

# # Quantas pessoas deseja cadastrar
# n = int(input("Quantas pessoas deseja cadastrar? "))

# # Coletando dados
# for i in range(n):
#     nome = input(f"Nome da pessoa {i+1}: ")
#     idade = int(input(f"Idade de {nome}: "))
#     salario = float(input(f"Salário de {nome}: "))
#     pessoas.append((nome, idade, salario))

# # Criar DataFrame
# df = pd.DataFrame(pessoas, columns=["Nome", "Idade", "Salario"])

# print("\nDataFrame original:")
# print(df)

# # Manipulação com lambda
# df_final = (
#     df
#     .loc[lambda df: df["Idade"] > 20]  # Filtra idade > 20
#     .assign(
#         Salario=lambda df: df["Salario"] * 1.1,  # Aumenta salário em 10%
#         Categoria=lambda df: df["Idade"].apply(lambda x: "Jovem" if x < 30 else "Adulto")  # Cria categoria
#     )
#     .sort_values(by="Salario", ascending=False)  # Ordena pelo salário
# )

# print("\nDataFrame manipulado:")
# print(df_final)


# Utilize context managers (with) para ler e escrever arquivos grandes.


# import pandas as pd

# # Nome do arquivo grande
# arquivo_entrada = "dados.csv"
# arquivo_saida = "arquivo.csv"

# # Definimos o tamanho do chunk (quantas linhas ler de cada vez)
# tamanho_chunk = 10000

# # Abrir arquivo de saída usando context manager
# with open(arquivo_saida, "w", encoding="utf-8") as saida:
#     # Ler o arquivo em chunks
#     for chunk in pd.read_csv(arquivo_entrada, chunksize=tamanho_chunk):
#         # Exemplo de processamento: filtrar linhas
#         filtrado = chunk[chunk["Idade"] > 20]  # só pessoas com idade > 20
        
#         # Escrever no arquivo de saída
#         # header=True apenas no primeiro chunk
#         filtrado.to_csv(saida, index=False, header=saida.tell()==0)


# Escreva iteradores personalizados com __iter__ e __next__.


# class Contador:
#     def __init__(self, limite):
#         self.limite = limite
#         self.atual = 1

#     def __iter__(self):
#         return self  # retorna o próprio iterador

#     def __next__(self):
#         if self.atual > self.limite:
#             raise StopIteration  # sinaliza fim da iteração
#         valor = self.atual
#         self.atual += 1
#         return valor

# # Usando o iterador
# contador = Contador(5)
# for numero in contador:
#     print(numero)


# Crie uma API simples com Flask ou FastAPI que retorne dados de clima.


# from fastapi import FastAPI
# from pydantic import BaseModel
# from datetime import datetime

# app = FastAPI(title="API de Clima Simples")

# # Modelo de dados para resposta
# class Clima(BaseModel):
#     cidade: str
#     temperatura: float
#     umidade: int
#     descricao: str
#     data: str

# # Dados de exemplo
# dados_clima = {
#     "São Paulo": {"temperatura": 26.5, "umidade": 60, "descricao": "Ensolarado"},
#     "Rio de Janeiro": {"temperatura": 28.3, "umidade": 70, "descricao": "Parcialmente nublado"},
#     "Belo Horizonte": {"temperatura": 24.0, "umidade": 65, "descricao": "Chuvoso"},
# }

# @app.get("/clima/{cidade}", response_model=Clima)
# def obter_clima(cidade: str):
#     cidade_formatada = cidade.title()  # Formata nome da cidade
#     if cidade_formatada in dados_clima:
#         info = dados_clima[cidade_formatada]
#         return Clima(
#             cidade=cidade_formatada,
#             temperatura=info["temperatura"],
#             umidade=info["umidade"],
#             descricao=info["descricao"],
#             data=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         )
#     return {"erro": "Cidade não encontrada"}


# Faça scraping de preços de produtos de uma página web e salve em CSV.


# import requests
# from bs4 import BeautifulSoup
# import pandas as pd

# # URL da página (exemplo)
# url = "https://pt.aliexpress.com/?src=google&albch=fbrnd&acnt=907-325-7776&isdl=y&aff_short_key=UneMJZVf&albcp=22465909199&albag=178972697235&slnk=&trgt=kwd-22976535628&plac=&crea=737535531803&netw=g&device=c&mtctp=e&memo1=&albbt=Google_7_fbrnd&aff_platform=google&albagn=888888&isSmbActive=false&isSmbAutoCall=false&needSmbHouyi=false&gad_source=1&gad_campaignid=22465909199&gbraid=0AAAAAD2TwoFRgtfejXIlWPiTYxRc4zGBO&gclid=Cj0KCQjwl5jHBhDHARIsAB0Yqjwio2XIODCCx8OtkqNR0gIe6rEv3VYCIkNiK1rrlIqBuiEbuQ_WRDcaArhQEALw_wcB&gatewayAdapt=glo2bra"

# # Fazer requisição HTTP
# response = requests.get(url)
# if response.status_code != 200:
#     raise Exception("Falha ao acessar a página")

# # Parse do HTML
# soup = BeautifulSoup(response.text, "html.parser")

# Encontrar todos os produtos
# produtos_html = soup.find_all("div", class_="produto")

# Extrair nome e preço
# dados = []
# for produto in produtos_html:
#     nome = produto.find("h2", class_="nome").text.strip()
#     preco_texto = produto.find("span", class_="preco").text.strip()
    
#     # Converter preço para float (remover R$ e substituir vírgula)
#     preco = float(preco_texto.replace("R$", "").replace(",", "."))
    
#     dados.append({"Produto": nome, "Preco": preco})

# # Criar DataFrame
# df = pd.DataFrame(dados)

# # Salvar em CSV
# df.to_csv("produtos.csv", index=False, encoding="utf-8-sig")

# print("Scraping concluído! Dados salvos em produtos.csv")


# Automatize preenchimento de formulários simples com Selenium ou PyAutoGUI.


# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# import time

# # Configuração do navegador
# options = Options()
# # options.add_argument("--headless")  # comente para ver o navegador
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# try:
#     # Página de teste
#     driver.get("https://www.google.com")

#     wait = WebDriverWait(driver, 10)

#     # Espera o campo de pesquisa aparecer
#     search_box = wait.until(EC.presence_of_element_located((By.NAME, "q")))

#     # Digita algo
#     search_box.send_keys("Python Selenium")

#     # Submete o formulário
#     search_box.submit()

#     # Espera resultados carregarem
#     wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g")))

#     # Mostra quantos resultados foram encontrados
#     results = driver.find_elements(By.CSS_SELECTOR, "div.g")
#     print(f"Encontrados {len(results)} resultados.")

#     time.sleep(5)  # pausa para você ver a página

# finally:
#     driver.quit()


# Consuma uma API externa REST e filtre os dados retornados.


# exemplo_jsonplaceholder.py
# import requests
# import pandas as pd

# URL = "https://jsonplaceholder.typicode.com/posts"

# def fetch_posts():
#     resp = requests.get(URL, timeout=10)
#     resp.raise_for_status()
#     return resp.json()  # retorna lista de dicts

# def filter_posts(posts, user_id=None, keyword=None):
#     # aplica filtros encadeados com lambdas
#     resultado = posts
#     if user_id is not None:
#         resultado = list(filter(lambda p: p.get("userId") == user_id, resultado))
#     if keyword:
#         resultado = list(filter(lambda p: keyword.lower() in p.get("title", "").lower(), resultado))
#     return resultado

# def to_csv(rows, out_file="posts_filtrados.csv"):
#     df = pd.DataFrame(rows)
#     # usa context manager internamente, mas mostramos explicitamente
#     with open(out_file, "w", encoding="utf-8-sig", newline="") as f:
#         df.to_csv(f, index=False)
#     print(f"Salvo em {out_file} ({len(df)} linhas)")

# if __name__ == "__main__":
#     posts = fetch_posts()
#     filtrados = filter_posts(posts, user_id=1, keyword="sunt")
#     to_csv(filtrados)


# Implemente autenticação simples em sua API (token ou JWT).


# # api_jwt_swagger.py
# from fastapi import FastAPI, Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
# from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
# from fastapi.security.oauth2 import OAuth2
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from datetime import datetime, timedelta
# from typing import Optional, Dict

# # --------------------
# # Configurações
# # --------------------
# SECRET_KEY = "minha_chave_super_secreta"
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 30

# pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# # Usuários "fake"
# fake_users_db = {
#     "maxwell": {
#         "username": "maxwell",
#         "full_name": "Maxwell Fernandes",
#         "hashed_password": pwd_context.hash("senha123"),
#     }
# }

# # --------------------
# # Segurança OAuth2
# # --------------------
# class OAuth2PasswordBearerWithSwagger(OAuth2):
#     """
#     Permite aparecer o botão 'Authorize' no Swagger UI
#     """
#     def __init__(self, tokenUrl: str):
#         flows = OAuthFlowsModel(password={"tokenUrl": tokenUrl})
#         super().__init__(flows=flows, scheme_name="JWT Bearer")

# oauth2_scheme = OAuth2PasswordBearerWithSwagger(tokenUrl="/token")

# # --------------------
# # Inicializa app
# # --------------------
# app = FastAPI(title="API Clima JWT", description="Exemplo de API com JWT e Swagger Authorize")

# # --------------------
# # Funções utilitárias
# # --------------------
# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)

# def authenticate_user(username: str, password: str) -> Optional[Dict]:
#     user = fake_users_db.get(username)
#     if not user:
#         return None
#     if not verify_password(password, user["hashed_password"]):
#         return None
#     return user

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Token inválido ou expirado",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception
#     user = fake_users_db.get(username)
#     if user is None:
#         raise credentials_exception
#     return user

# # --------------------
# # Rotas
# # --------------------
# @app.post("/token", summary="Login e obtenção do JWT")
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Usuário ou senha incorretos",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     access_token = create_access_token(data={"sub": user["username"]})
#     return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/dados-clima", summary="Retorna dados de clima protegidos por JWT")
# async def dados_clima(current_user: dict = Depends(get_current_user)):
#     return {
#         "cidade": "São Paulo",
#         "temperatura": 26,
#         "umidade": 60,
#         "usuario": current_user["full_name"]
#     }


# Crie um programa que registre erros em um arquivo log com níveis de gravidade.


# import logging
# from datetime import datetime

# # -----------------------------
# # Configuração do logger
# # -----------------------------
# log_filename = "app.log"  # nome do arquivo de log

# logging.basicConfig(
#     filename=log_filename,          # arquivo onde os logs serão salvos
#     level=logging.DEBUG,            # nível mínimo de log que será registrado
#     format="%(asctime)s - %(levelname)s - %(message)s",  # formato da mensagem
#     datefmt="%Y-%m-%d %H:%M:%S"    # formato da data/hora
# )

# # -----------------------------
# # Função para registrar logs
# # -----------------------------
# def registrar_erro(mensagem: str, nivel: str = "INFO"):
#     """
#     Registra uma mensagem no arquivo de log com o nível especificado.
#     Níveis possíveis: DEBUG, INFO, WARNING, ERROR, CRITICAL
#     """
#     nivel = nivel.upper()
#     if nivel == "DEBUG":
#         logging.debug(mensagem)
#     elif nivel == "INFO":
#         logging.info(mensagem)
#     elif nivel == "WARNING":
#         logging.warning(mensagem)
#     elif nivel == "ERROR":
#         logging.error(mensagem)
#     elif nivel == "CRITICAL":
#         logging.critical(mensagem)
#     else:
#         logging.info(f"Nível inválido '{nivel}' informado: {mensagem}")

# # -----------------------------
# # Exemplo de uso
# # -----------------------------
# if __name__ == "__main__":
#     registrar_erro("Aplicativo iniciado", "INFO")
#     registrar_erro("Este é um aviso de teste", "WARNING")
#     registrar_erro("Erro ao conectar ao banco de dados", "ERROR")
#     registrar_erro("Detalhes para debug", "DEBUG")
#     registrar_erro("Falha crítica no sistema!", "CRITICAL")

#     print(f"Logs registrados no arquivo {log_filename}")


# Implemente tratamento de exceções refinado (try/except/else/finally).


# def ler_arquivo(caminho):
#     try:
#         arquivo = open(caminho, "r", encoding="utf-8")
#         conteudo = arquivo.read()
#     except FileNotFoundError:
#         print("❌ Erro: Arquivo não encontrado.")
#     except PermissionError:
#         print("⚠️ Erro: Permissão negada para acessar o arquivo.")
#     except Exception as e:
#         print(f"🚨 Erro inesperado: {e}")
#     else:
#         print("✅ Arquivo lido com sucesso!")
#         print("Conteúdo:")
#         print(conteudo)
#     finally:
#         # Fecha o arquivo, se ele foi aberto
#         try:
#             arquivo.close()
#             print("🔒 Arquivo fechado com segurança.")
#         except NameError:
#             print("ℹ️ Nenhum arquivo foi aberto, nada para fechar.")

# # Teste:
# ler_arquivo("dados.txt")


# Escreva testes unitários para funções matemáticas ou manipulação de listas.


# import unittest
# from operacoes import soma, multiplicacao, media, ordenar_lista

# class TestOperacoes(unittest.TestCase):

#     def test_soma(self):
#         self.assertEqual(soma(2, 3), 5)
#         self.assertEqual(soma(-1, 1), 0)
#         self.assertNotEqual(soma(2, 2), 5)

#     def test_multiplicacao(self):
#         self.assertEqual(multiplicacao(3, 4), 12)
#         self.assertEqual(multiplicacao(0, 100), 0)
#         self.assertEqual(multiplicacao(-2, 3), -6)

#     def test_media(self):
#         self.assertEqual(media([2, 4, 6]), 4)
#         self.assertAlmostEqual(media([1, 2, 3]), 2.0)
#         with self.assertRaises(ValueError):
#             media([])  # Testa exceção para lista vazia

#     def test_ordenar_lista(self):
#         self.assertEqual(ordenar_lista([3, 1, 2]), [1, 2, 3])
#         self.assertEqual(ordenar_lista([]), [])
#         self.assertEqual(ordenar_lista([5, 5, 5]), [5, 5, 5])

# if __name__ == '__main__':
#     unittest.main()


# Refatore seu código para seguir boas práticas de PEP8 e clareza.


# """
# Módulo de testes unitários para o arquivo operacoes.py.
# Segue boas práticas de PEP8 e organização com unittest.
# """

# import unittest
# from operacoes import soma, multiplicacao, media, ordenar_lista


# class TestOperacoes(unittest.TestCase):
#     """Classe de testes para as funções do módulo operacoes."""

#     def test_soma(self):
#         self.assertEqual(soma(2, 3), 5)
#         self.assertEqual(soma(-1, 1), 0)

#     def test_multiplicacao(self):
#         self.assertEqual(multiplicacao(2, 3), 6)
#         self.assertEqual(multiplicacao(-2, 4), -8)

#     def test_media(self):
#         self.assertAlmostEqual(media([1, 2, 3, 4]), 2.5)
#         with self.assertRaises(ValueError):
#             media([])

#     def test_ordenar_lista(self):
#         self.assertEqual(ordenar_lista([3, 1, 2]), [1, 2, 3])
#         self.assertEqual(ordenar_lista([]), [])


# if __name__ == "__main__":
#     unittest.main()


# Implemente KNN para classificar um dataset real (CSV).


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # 1️⃣ Carregar CSV
# arquivo = "notas.csv"
# dados = pd.read_csv(arquivo)
# print("Dados carregados:")
# print(dados)

# # 2️⃣ Criar coluna target: aprovado (1) ou reprovado (0)
# # Considerando média >= 7 como aprovado
# dados["media"] = dados[["nota1", "nota2", "nota3"]].mean(axis=1)
# dados["aprovado"] = (dados["media"] >= 7).astype(int)

# # 3️⃣ Separar features (notas) e target (aprovado)
# X = dados[["nota1", "nota2", "nota3"]]
# y = dados["aprovado"]

# # 4️⃣ Dividir em treino e teste
# X_treino, X_teste, y_treino, y_teste = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 5️⃣ Normalizar os dados
# scaler = StandardScaler()
# X_treino = scaler.fit_transform(X_treino)
# X_teste = scaler.transform(X_teste)

# # 6️⃣ Treinar KNN
# knn = KNeighborsClassifier(n_neighbors=2)
# knn.fit(X_treino, y_treino)

# # 7️⃣ Fazer previsões
# y_pred = knn.predict(X_teste)

# # 8️⃣ Avaliar resultados
# print("\nResultados das previsões:")
# print(pd.DataFrame({"nome": dados.loc[y_teste.index, "nome"], "aprovado_real": y_teste, "aprovado_pred": y_pred}))

# print("\nRelatório de Classificação:")
# print(classification_report(y_teste, y_pred))
# print(f"Acurácia: {accuracy_score(y_teste, y_pred):.2f}")

# # 5️⃣ Treinar modelo KNN
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_treino, y_treino)

# # 6️⃣ Fazer previsões
# y_pred = knn.predict(X_teste)

# # 7️⃣ Avaliar resultados
# print("\nRelatório de Classificação:")
# print(classification_report(y_teste, y_pred))
# print(f"Acurácia: {accuracy_score(y_teste, y_pred):.2f}")


# Crie uma árvore de decisão simples para classificar clientes como “bom” ou “risco”.


# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, export_text

# # 1️⃣ Criar dataset fictício
# dados = pd.DataFrame({
#     "idade": [25, 45, 35, 50, 23, 40, 60, 30],
#     "renda_mensal": [2000, 5000, 3000, 4000, 1500, 3500, 6000, 2800],
#     "dividas": [0, 1, 0, 1, 0, 0, 1, 0],
#     "cliente": ["bom", "risco", "bom", "risco", "bom", "bom", "risco", "bom"]
# })

# # 2️⃣ Separar features e target
# X = dados[["idade", "renda_mensal", "dividas"]]
# y = dados["cliente"]

# # 3️⃣ Criar e treinar a árvore de decisão
# arvore = DecisionTreeClassifier(max_depth=3, random_state=42)
# arvore.fit(X, y)

# # 4️⃣ Fazer previsões (exemplo)
# y_pred = arvore.predict(X)

# # 5️⃣ Mostrar resultados
# resultados = dados.copy()
# resultados["previsao"] = y_pred
# print("Resultados das previsões:")
# print(resultados)

# # 6️⃣ Mostrar a árvore em formato de texto
# print("\nÁrvore de decisão:")
# arvore_texto = export_text(arvore, feature_names=list(X.columns))
# print(arvore_texto)


# Normalizar dados antes de aplicar um algoritmo de ML e comparar resultados.


# import unittest
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# class TestKNNNormalization(unittest.TestCase):
#     def setUp(self):
#         # Dataset fictício
#         self.dados = pd.DataFrame({
#             "idade": [25, 45, 35, 50, 23, 40, 60, 30],
#             "renda_mensal": [2000, 5000, 3000, 4000, 1500, 3500, 6000, 2800],
#             "dividas": [0, 1, 0, 1, 0, 0, 1, 0],
#             "cliente": ["bom", "risco", "bom", "risco", "bom", "bom", "risco", "bom"]
#         })
#         self.X = self.dados[["idade", "renda_mensal", "dividas"]]
#         self.y = self.dados["cliente"]

#         # Divisão treino/teste
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             self.X, self.y, test_size=0.25, random_state=42
#         )

#     def test_knn_with_and_without_normalization(self):
#         # --- Sem normalização ---
#         knn = KNeighborsClassifier(n_neighbors=3)
#         knn.fit(self.X_train, self.y_train)
#         y_pred = knn.predict(self.X_test)
#         acc_no_scaling = accuracy_score(self.y_test, y_pred)

#         # --- Com normalização ---
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(self.X_train)
#         X_test_scaled = scaler.transform(self.X_test)

#         knn_scaled = KNeighborsClassifier(n_neighbors=3)
#         knn_scaled.fit(X_train_scaled, self.y_train)
#         y_pred_scaled = knn_scaled.predict(X_test_scaled)
#         acc_scaled = accuracy_score(self.y_test, y_pred_scaled)

#         print(f"Acurácia sem normalização: {acc_no_scaling:.2f}")
#         print(f"Acurácia com normalização: {acc_scaled:.2f}")

#         # Verifica se normalização não piora o modelo
#         self.assertGreaterEqual(acc_scaled, acc_no_scaling, 
#             "Normalização piorou o desempenho do KNN")

# if __name__ == "__main__":
#     unittest.main()


# Calcule métricas de avaliação: accuracy, precision e recall.


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score

# # Dataset de exemplo
# dados = pd.DataFrame({
#     "idade": [25, 45, 35, 50, 23, 40, 60, 30],
#     "renda_mensal": [2000, 5000, 3000, 4000, 1500, 3500, 6000, 2800],
#     "dividas": [0, 1, 0, 1, 0, 0, 1, 0],
#     "cliente": ["bom", "risco", "bom", "risco", "bom", "bom", "risco", "bom"]
# })

# # Features e target
# X = dados[["idade", "renda_mensal", "dividas"]]
# y = dados["cliente"]

# # Normalização
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Divisão treino/teste
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# # Treinamento do KNN
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)

# # Cálculo das métricas
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, pos_label="bom")
# recall = recall_score(y_test, y_pred, pos_label="bom")

# # Exibição dos resultados
# print(f"Acurácia: {accuracy:.2f}")
# print(f"Precision (bom): {precision:.2f}")
# print(f"Recall (bom): {recall:.2f}")


# Treine um modelo simples para prever uma saída binária e explique overfitting/underfitting.


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# # Dataset fictício de alunos aprovados ou não
# dados = pd.DataFrame({
#     "nota1": [8, 6, 6, 9, 5, 7, 10, 4, 8, 6],
#     "nota2": [7.5, 8, 7.5, 8.5, 6, 7, 9.5, 5, 9, 5.5],
#     "nota3": [9, 5, 10, 9, 4, 7, 10, 6, 8.5, 6],
#     "aprovado": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]  # 1 = aprovado, 0 = reprovado
# })

# X = dados[["nota1", "nota2", "nota3"]]
# y = dados["aprovado"]

# # Normalização
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Divisão treino/teste
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# # Treinamento do KNN
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)

# # Métricas
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Acurácia no teste: {accuracy:.2f}")
# print("Previsões:", y_pred)


# 🔴 Nível Sênior (76–101) – Avançado

# criar um mini-sistema de “memorizar textos” usando PyTorch e Flask, 
# onde os textos são aprendidos nos pesos da rede e não ficam no código.

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from flask import Flask, request, jsonify

# textos = [
#     'olá mundo!',
#     'aprendizado de maquina é incrivel.',
#     'Python é divertido',
# ]
# # --- Preparar vocabulário ---
# all_chars = sorted(list(set(''.join(textos))))
# char_to_idx = {c: i for i, c in enumerate(all_chars)}
# idx_to_char = {i: c for i, c in enumerate(all_chars)}

# # Função para converter texto em índices
# def texto_para_indices(texto):
#     return [char_to_idx[c] for c in texto]

# # --- Rede neural simples ---
# class MemoriaTexto(nn.Module):
#     def __init__(self, n_chars, hidden_size=64):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.embed = nn.Embedding(n_chars, hidden_size)
#         self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, n_chars)

#     def forward(self, x, hidden=None):
#         x = self.embed(x)
#         out, hidden = self.rnn(x, hidden)
#         out = self.fc(out)
#         return out, hidden
    
# # --- Inicializar rede ---
# n_chars = len(all_chars)
# net = MemoriaTexto(n_chars)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr = 0.01)


# if os.path.exists('modelo.pth'):
#     net.load_state_dict(torch.load('modelo.pth'))
#     net.eval()
#     print('modelo carregado de modelo.pth')
# else:
#     # --- Treinar a rede em todos os textos ---
#     for texto in textos:
#         data = texto_para_indices(texto)
#         input_seq = torch.tensor(data[:-1]).unsqueeze(0)
#         target_seq = torch.tensor(data[1:]).unsqueeze(0)

#         for epoch in range(1000):
#             optimizer.zero_grad()
#             output, _ = net(input_seq)
#             loss = criterion(output.squeeze(0), target_seq.squeeze(0))
#             loss.backward()
#             optimizer.step()

#     torch.save(net.state_dict(), 'modelo.pth')
#     print('Modelo treinado e salvo em modelo.pth')

# # --- Função para gerar texto a partir do primeiro caractere ---
# def gerar_texto(primeiro_char, tamanho=50):
#     net.eval()
#     input_char = torch.tensor([char_to_idx[primeiro_char]]).unsqueeze(0)
#     hidden = None
#     resultado = [input_char.item()]
#     for _ in range(tamanho-1):
#         out, hidden = net(input_char, hidden)
#         prob = torch.softmax(out[0,0], dim=0)
#         idx = torch.argmax(prob).item()
#         resultado.append(idx)
#         input_char = torch.tensor([idx]).unsqueeze(0)
#     return ''.join([idx_to_char[i] for i in resultado])

# # --- Criar API Flask ---
# app = Flask(__name__)

# @app.route("/gerar", methods= ['GET'])
# def gerar():
#     primeiro_char = request.args.get("primeiro_char", 'o')
#     tamanho = int(request.args.get("tamanho", 50))
#     texto_gerado = gerar_texto(primeiro_char, tamanho)
#     return jsonify({'texto': texto_gerado})

# if __name__ =='__main__':
#     app.run(port=5000, debug=True)


# Implemente um sistema assíncrono que faça 10 requisições HTTP em paralelo.


# import asyncio
# import aiohttp

# url = 'https://httpbin.org/get'

# async def fetch(session, i):
#     async with session.get(url) as response:
#         data = await response.json()
#         print(f'resposta {i}: {data['url']}')
#         return data

# async def main():
#     async with aiohttp.ClientSession() as session:
#         tasks = [fetch(session, i) for i in range(10)]
#         results = await asyncio.gather(*tasks)
#         return results
    
# if __name__ == "__main__":
#     asyncio.run(main())

# Crie um cache com functools.lru_cache.


# import functools
# import time

# @functools.lru_cache(maxsize=128)
# def fib(n: int) -> int:
#     if n < 2:
#         return n
#     return fib(n -1) + fib(n - 2)

# def medir():
#     start = time.time()
#     print('fib(35) =', fib(35))
#     t1 = time.time() - start

#     start =time.time()
#     print('fibt(35) novamente =', fib(35))
#     t2 = time.time() - start

#     print(f' tempo  primeira: {t1:.4f}s, segunda (cache): {t2:.6f}s')
#     print('info do cache:', fib.cache_info())

# if __name__ == '__main__':
#     medir()


# Implemente um sistema de chat com socket.

# Crie um sistema de filas usando RabbitMQ ou Kafka.

# Implemente um servidor web com asyncio.

# Crie uma metaclasse que registre todas as subclasses criadas.

# Implemente o padrão Observer em Python.

# Crie um decorador que retente uma função falha até 3 vezes.

# Implemente uma API GraphQL em Python.

# Crie uma classe com __slots__ para otimizar memória.

# Implemente um analisador de logs em tempo real (streaming).

# Crie um ETL que leia CSV → transforme → grave em banco SQL.

# Implemente um sistema de recomendação simples com similaridade de cosseno.

# Crie um motor de busca em Python (índice invertido).

# Implemente um parser de expressões matemáticas.

# Crie um sistema de workers com concurrent.futures.

# Implemente um sistema que detecte deadlocks em threads.

# Crie um compilador simples para uma linguagem fictícia.

# Implemente um sistema de versionamento de arquivos.

# Crie uma API que consuma e processe dados em tempo real (websockets).

# Implemente um algoritmo de machine learning básico do zero (ex: regressão linear).

# Crie um sistema distribuído de votação (múltiplos processos).

# Implemente um sistema de cache distribuído em Python.

# Crie uma ferramenta de profiling que meça tempo/memória de funções.

# Implemente um microframework estilo Flask do zero.


