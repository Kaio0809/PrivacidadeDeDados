import numpy as np

def probabilidades(epsilon, k):
    """
    Calcula as probabilidades do mecanismo GRR.

    epsilon: orçamento de privacidade.
    k: quantidade de categorias do domínio.

    Retorna p, a probabilidade de manter o valor verdadeiro,
    e q, a probabilidade de reportar cada categoria diferente.
    """
    e = np.exp(epsilon)
    d = e + k -1

    p = e / d
    q = 1/ d

    return p, q

def grr(valor, dominio, epsilon, rng):
    """
    Aplica o Generalized Randomized Response a um único valor.

    O mecanismo mantém a categoria verdadeira com probabilidade p.
    Caso contrário, sorteia uniformemente uma das outras categorias.

    rng: gerador de números aleatórios utilizado no sorteio.
    """
    k = len(dominio)
    p, q = probabilidades(epsilon, k)
    prob = rng.random()

    if prob < p:
        return valor

    else:
        # Exclui a categoria verdadeira das alternativas do sorteio.
        categorias = [categoria for categoria in dominio if categoria != valor]
        indice = rng.integers(0, len(categorias))

        return categorias[indice]

def gerar_relatorios(dados, dominio, epsilon, semente):
    """
    Gera os relatórios perturbados de todos os indivíduos.

    Cada valor é processado independentemente pelo GRR.
    A semente permite reproduzir os resultados do experimento.

    Retorna uma lista com um relatório por indivíduo.
    """
    rng = np.random.default_rng(semente)

    relatorios = [grr(val, dominio, epsilon, rng) for val in dados]
    return relatorios


def agrupar_idades(dados, largura):
    """
    Agrupa as idades em intervalos para a estratégia E2.

    largura: quantidade de anos em cada intervalo.
    O último bin é limitado à idade máxima de 90 anos.

    Retorna os intervalos criados e os dados com as idades
    substituídas pelos índices dos respectivos bins.
    """
    bins = [(inicio, min(inicio + largura - 1, 90)) for inicio in range(17, 91, largura)]

    agrupados = [((idade - 17) // largura, salario) for idade, salario in dados]

    return bins, agrupados
