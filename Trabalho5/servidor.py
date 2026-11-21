import numpy as np
import pandas as pd
from collections import Counter

def contar_relatorios(relatorios, dominio):
    contagens = Counter(relatorios)

    frequencias = {categoria: contagens.get(categoria, 0) for categoria in dominio}

    return frequencias


def estimar_frequencias(relatorios, dominio, epsilon):

    n = len(relatorios)
    k = len(dominio)

    exp_epsilon = np.exp(epsilon)

    p = exp_epsilon / (exp_epsilon + k - 1)
    q = 1 / (exp_epsilon + k - 1)

    contagens = contar_relatorios(relatorios, dominio)

    estimativas = {}

    for categoria in dominio:
        cy = contagens[categoria]
        ny = (cy - n * q) / (p - q)
        estimativas[categoria] = ny

    return estimativas

def estimar_consulta(estimativas,idade_min=None,idade_max=None,salario=None,truncamento=False,n=None):
    resultado = 0

    for (idade, sal), frequencia in estimativas.items():
        if idade_min is not None and idade < idade_min:
            continue

        if idade_max is not None and idade > idade_max:
            continue

        if salario is not None and sal != salario:
            continue

        resultado += frequencia

    if truncamento:
        resultado = max(0, min(resultado, n))

    return resultado



def estimar_consulta_bins(estimativas, bins, idade_min=None, idade_max=None, salario=None):
    total = 0

    for (indice, sal), freq in estimativas.items():
        if salario is not None and sal != salario:
            continue

        inicio, fim = bins[indice]
        menor = max(inicio, idade_min) if idade_min is not None else inicio
        maior = min(fim, idade_max) if idade_max is not None else fim

        if menor <= maior:
            total += freq * (maior - menor + 1) / (fim - inicio + 1)

    return total
