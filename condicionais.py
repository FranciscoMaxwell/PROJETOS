# simples, composto e encadeado

n1 = n2 = 0.0
media = 0.0

n1 = float(input('Digite sua primeira nota: '))
n2 = float(input('Digite sua segunda nota: '))

#calcular a media aritmetica das notas
media = (n1 + n2) / 2

if (media >= 7):
    print ("o aluno passou esse arrombado")
    print ('Parabens')
elif (media >= 5):
    print('você vai pra recuperação, seu gordoooo>>>>>')
else:
    print('vocÊ foi pras cucuias')
    print('não passaste')

print ('Sua media foi de {}'.format(media))





