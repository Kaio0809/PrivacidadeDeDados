import pandas as pd
import numpy as np
import time
import random

random.seed(42)
np.random.seed(42)

url = 'https://raw.githubusercontent.com/Kaio0809/datasets/refs/heads/main/adult.csv'
df = pd.read_csv(url, sep=',')

df = df.drop(columns=['fnlwgt', 'education', 'capital-gain', 'capital-loss', 'hours-per-week'])

df = df[~(df == '?').any(axis=True)]

def codificacao(df_in):
    df_cod = df_in.copy()
    mapeamentos = {}
    for coluna in df_in.columns:
        if df_in[coluna].dtype in ['str', 'object']:
            valores = sorted(df_in[coluna].unique())
            mapa = {valor: i for i, valor in enumerate(valores)}
            df_cod[coluna] = df_in[coluna].map(mapa)
            mapeamentos[coluna] = mapa
    return df_cod.astype(int), mapeamentos

df_codificado, mapeamentos = codificacao(df)

df_codificado.to_csv('df_pronto.csv', index=False)

indices = list(range(len(df_codificado)))
random.shuffle(indices)

limite = int(len(indices) * 0.70)
indices_treino = indices[:limite]
indices_teste = indices[limite:]

df_treino = df_codificado.iloc[indices_treino]
df_teste = df_codificado.iloc[indices_teste]

X_treino = df_treino.drop(columns=['income']).to_numpy(dtype=np.float64)
y_treino = df_treino['income'].to_numpy(dtype=np.int32)

X_teste = df_teste.drop(columns=['income']).to_numpy(dtype=np.float64)
y_teste = df_teste['income'].to_numpy(dtype=np.int32)

print(f"Registros de treino: {len(X_treino)}")
print(f"Registros de teste: {len(X_teste)}")

def prever_knn_tradicional(X_treino, y_treino, X_teste, k=10, tamanho_bloco=256):
    n_teste = len(X_teste)
    previsoes = np.zeros(n_teste, dtype=np.int32)
    norma_treino = np.sum(X_treino ** 2, axis=1)

    for i in range(0, n_teste, tamanho_bloco):
        fim = min(i + tamanho_bloco, n_teste)
        bloco = X_teste[i:fim]
        norma_teste = np.sum(bloco ** 2, axis=1)
        
        dist_sq = norma_teste[:, None] + norma_treino[None, :] - 2 * (bloco @ X_treino.T)
        np.maximum(dist_sq, 0, out=dist_sq)

        idx_vizinhos = np.argpartition(dist_sq, k, axis=1)[:, :k]
        
        for j, idxs in enumerate(idx_vizinhos):
            classes_vizinhos = y_treino[idxs]
            counts = np.bincount(classes_vizinhos)
            previsoes[i + j] = np.argmax(counts)

    return previsoes

if __name__ == '__main__':
    inicio = time.time()
    previsoes = prever_knn_tradicional(X_treino, y_treino, X_teste, k=10)
    tempo = time.time() - inicio

    resultado = df_teste.copy()
    resultado['y_pred'] = previsoes
    resultado['Corretude'] = resultado['income'] == resultado['y_pred']
    resultado.to_csv('resultados/df_resultados_testes.csv', index=False)

    acertos = np.sum(previsoes == y_teste)
    acuracia = acertos / len(y_teste)

    print(f"Acurácia do kNN Tradicional: {acuracia * 100:.2f}%")
    print(f"Tempo de execução: {tempo:.2f} segundos")

    with open("resultados/resultado.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write('Resultados - kNN Tradicional (sem privacidade): \n\n')
        arquivo.write(f'A acurácia do teste foi de {acuracia:.4f}, o que equivale a {acuracia * 100:.2f}% de acertos.\n')
        arquivo.write(f'Feitas {len(previsoes)} previsões em {tempo:.2f} segundos.\n')