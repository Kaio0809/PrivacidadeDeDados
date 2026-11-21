
import numpy as np
import pandas as pd
from collections import Counter

def contar_relatorios(relatorios, dominio):
    """
    Conta a frequência de cada categoria nos relatórios perturbados.

    relatorios: valores recebidos dos clientes após a aplicação do GRR.
    dominio: conjunto de todas as categorias possíveis.

    Retorna um dicionário com a contagem observada de cada categoria.
    Categorias que não aparecem nos relatórios recebem contagem zero.
    """
    contagens = Counter(relatorios)

    frequencias = {categoria: contagens.get(categoria, 0) for categoria in dominio}

    return frequencias


def estimar_frequencias(relatorios, dominio, epsilon):
    """
    Estima as frequências verdadeiras a partir dos relatórios perturbados.

    Utiliza o estimador não viesado do GRR:
        ny = (cy - n*q) / (p-q)

    n: quantidade de relatórios recebidos.
    k: tamanho do domínio.
    cy: frequência observada da categoria.
    p e q: probabilidades do mecanismo GRR.

    Retorna um dicionário com as frequências estimadas.
    """
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
    """
    Estima a resposta de uma consulta sobre idade e salário na E1.

    Os filtros de idade e salário são opcionais, permitindo consultas
    bidimensionais e marginais.

    estimativas: frequências estimadas pelo GRR.
    idade_min e idade_max: limites inclusivos do intervalo de idades.
    salario: categoria de salário selecionada.
    truncamento: indica se a resposta deve ser limitada a [0, n].
    n: quantidade de indivíduos, usada no truncamento.

    Retorna a soma das estimativas das categorias selecionadas.
    """
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
        # O truncamento é aplicado à resposta final, não a cada categoria.
        resultado = max(0, min(resultado, n))

    return resultado



def estimar_consulta_bins(estimativas, bins, idade_min=None, idade_max=None, salario=None):
    """
    Estima consultas sobre idades agrupadas em bins na estratégia E2.

    Cada bin contribui proporcionalmente à quantidade de idades
    pertencentes ao intervalo consultado.

    estimativas: frequências estimadas das categorias (bin, salário).
    bins: lista de intervalos de idade representados por (início, fim).
    idade_min e idade_max: limites inclusivos da consulta.
    salario: categoria de salário selecionada, quando aplicável.

    A aproximação assume distribuição uniforme das idades dentro
    de cada bin parcialmente selecionado.

    Retorna a estimativa da consulta considerando essa proporção.
    """
    total = 0

    for (indice, sal), freq in estimativas.items():
        if salario is not None and sal != salario:
            continue

        inicio, fim = bins[indice]
        menor = max(inicio, idade_min) if idade_min is not None else inicio
        maior = min(fim, idade_max) if idade_max is not None else fim

        if menor <= maior:
            # Considera somente a fração do bin coberta pela consulta.
            total += freq * (maior - menor + 1) / (fim - inicio + 1)

    return total
