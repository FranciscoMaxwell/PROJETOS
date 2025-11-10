# Set, colocar multiplos itens como conjunto dentro de um objeto

planeta_anao = {'Plutão', 'Haumea', 'Europa'}
# print(len(planeta_anao))
# print('lua' in planeta_anao)

# for astronauta in planeta_anao:
#     print(astronauta.upper(), end= " ")

# astros = ['Lua', 'Sol', 'Jupiter', 'Saturno', 'Lua', 'Lua']
# print(astros, end=" ")
# astros_set = set(astros)
# print(astros_set)

astros1 = {'Lua', 'Sol', 'Jupiter', 'Sarturno'}
astros2 = {'Lua', 'Sol', 'Jupiter', 'Sarturno', 'cometa halley'}
# print(astros1 <= astros2)
# print(astros1 | astros2)
# print(astros1.union(astros2))

# print(astros1 & astros2)
# print(astros1.intersection(astros2))

# print(astros1 ^ astros2)
# print(astros1.symmetric_difference(astros2))

astros1.add('Uraniano')
astros1.add('Barriga')
astros1.discard('Lua')
astros1.remove('Sol')
astros1.pop()
# astros1.clear()
print(astros1)
