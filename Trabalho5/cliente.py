import numpy as np
import pandas as pd


def probabilidades(epsilon, k):
    e = np.exp(epsilon)
    d = e + k -1

    p = e / d
    q = 1/ d

    return p, q

def grr(valor, dominio, epsilon, rng):
    k = len(dominio)
    p, q = probabilidades(epsilon, k)
    prob = rng.random()

    if prob < p:
        return valor

    else:
        categorias = [categoria for categoria in dominio if categoria != valor]
        indice = rng.integers(0, len(categorias))

        return categorias[indice]

def gerar_relatorios(dados, dominio, epsilon, semente):
    rng = np.random.default_rng(semente)

    relatorios = [grr(val, dominio, epsilon, rng) for val in dados]
    return relatorios


def agrupar_idades(dados, largura):
    bins = [(inicio, min(inicio + largura - 1, 90)) for inicio in range(17, 91, largura)]

    agrupados = [((idade - 17) // largura, salario) for idade, salario in dados]

    return bins, agrupados