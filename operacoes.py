def soma(a, b):
    """Retorna a soma de dois números."""
    return a + b

def multiplicacao(a, b):
    """Retorna a multiplicação de dois números."""
    return a * b

def media(lista):
    """Retorna a média dos valores de uma lista."""
    if not lista:
        raise ValueError("A lista não pode estar vazia.")
    return sum(lista) / len(lista)

def ordenar_lista(lista):
    """Retorna a lista em ordem crescente."""
    return sorted(lista)


"""
Módulo: operacoes.py
Descrição: Contém funções matemáticas simples e manipulação de listas.
"""

def soma(a: float, b: float) -> float:
    """Retorna a soma de dois números."""
    return a + b


def multiplicacao(a: float, b: float) -> float:
    """Retorna o produto de dois números."""
    return a * b


def media(lista: list[float]) -> float:
    """Calcula a média dos elementos de uma lista."""
    if not lista:
        raise ValueError("A lista não pode estar vazia.")
    return sum(lista) / len(lista)


def ordenar_lista(lista: list) -> list:
    """Retorna uma nova lista ordenada em ordem crescente."""
    return sorted(lista)
