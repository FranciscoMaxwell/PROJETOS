#import math

#print(math.sqrt(81))

#or

#from math import sqrt, sin
#print(sqrt(81))

#import math as m
#print(m.sqrt(144))

import mod_func as mf

if __name__  == '__main__':
    mf.mensagem()

    num = int(input('Digite o numero inteiro aqui!  '))

    print(f'\nCalcular fatorial do número: ')
    fat = mf.fatorial(num)
    print(f'O fatorial é {fat}')

    print(f'\nCalcular sequencia de fibonacci: ')
    fib = mf.fibo(num)
    print(f'o fatorial é {fib}')

import numpy as np

if __name__ == '__main__':
    L = np.array([1,2,3,4,5,6,7,8,9])
    print(L)