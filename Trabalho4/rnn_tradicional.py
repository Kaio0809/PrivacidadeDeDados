
import pandas as pd
import numpy as np
import time
import random
import os


url = 'https://raw.githubusercontent.com/Kaio0809/datasets/refs/heads/main/adult.csv'
df = pd.read_csv(url, sep=',')

df = df.drop(columns=['fnlwgt', 'education', 'capital-gain', 'capital-loss', 'hours-per-week'])
df = df[~(df == '?').any(axis=True)].copy()


def separar_balanceado(df):
    classe_0 = df[df['income'] == '<=50K']
    classe_1 = df[df['income'] == '>50K']

    if len(classe_0) < 500 or len(classe_1) < 500:
        raise ValueError("Não existem registros suficientes para formar um teste balanceado.")

    teste_0 = classe_0.sample(n=500, random_state=42)
    teste_1 = classe_1.sample(n=500, random_state=42)

    df_teste = pd.concat([teste_0, teste_1])
    df_teste = df_teste.sample(frac=1, random_state=42)

    df_treino = df.drop(index=df_teste.index)

    return df_treino.copy(), df_teste.copy()


def codificacao(df_treino, df_teste):
    treino = df_treino.copy()
    teste = df_teste.copy()

    treino['income'] = (treino['income'] == '>50K').astype(int)
    teste['income'] = (teste['income'] == '>50K').astype(int)

    categoricas = ['workclass','marital-status','occupation','relationship','race','gender','native-country']

    treino = pd.get_dummies(treino, columns=categoricas, dtype=int)
    teste = pd.get_dummies(teste, columns=categoricas, dtype=int)

    teste = teste.reindex(columns=treino.columns, fill_value=0)

    return treino, teste


def normalizar(df_treino, df_teste):
    treino = df_treino.copy()
    teste = df_teste.copy()

    for coluna in ['age', 'educational-num']:
        minimo = treino[coluna].min()
        maximo = treino[coluna].max()

        if minimo != maximo:
            treino[coluna] = (treino[coluna] - minimo) / (maximo - minimo)
            teste[coluna] = (teste[coluna] - minimo) / (maximo - minimo)
        else:
            treino[coluna] = 0
            teste[coluna] = 0

    return treino, teste


def r_N(X_treino, y_treino, X_teste, raio=6.0, tamanho_bloco=256):
    previsoes = []
    norma_treino = np.sum(X_treino ** 2, axis=1)
    raio_sq = raio ** 2
    classes = np.unique(y_treino)
    classe_majoritaria = classes[np.argmax([np.sum(y_treino == c) for c in classes])]

    for i in range(0, len(X_teste), tamanho_bloco):
        fim = min(i + tamanho_bloco, len(X_teste))
        bloco = X_teste[i:fim]
        norma_teste = np.sum(bloco ** 2, axis=1)

        dist_sq = norma_teste[:, None] + norma_treino[None, :] - 2 * (bloco @ X_treino.T)
        np.maximum(dist_sq, 0, out=dist_sq)

        dentro_raio = dist_sq <= raio_sq

        for j in range(len(bloco)):
            vizinhos = y_treino[dentro_raio[j]]

            if len(vizinhos) == 0:
                previsoes.append(classe_majoritaria)
            else:
                contagens = [np.sum(vizinhos == c) for c in classes]
                previsoes.append(classes[np.argmax(contagens)])

    return np.array(previsoes)


df_treino, df_teste = separar_balanceado(df)
df_treino, df_teste = codificacao(df_treino, df_teste)
dfNormTreino, dfNormTeste = normalizar(df_treino, df_teste)

os.makedirs('resultados', exist_ok=True)

dfNormTreino.to_csv('df_treino.csv', index=False)
dfNormTeste.to_csv('df_teste.csv', index=False)

X_treino = dfNormTreino.drop(columns=['income']).to_numpy(dtype=np.float64)
y_treino = dfNormTreino['income'].to_numpy(dtype=np.int32)

X_teste = dfNormTeste.drop(columns=['income']).to_numpy(dtype=np.float64)
y_teste = dfNormTeste['income'].to_numpy(dtype=np.int32)

print(f"Registros de treino: {len(X_treino)}")
print(f"Registros de teste: {len(X_teste)}")

if __name__ == '__main__':
    raios = [0.5, 1, 1.5, 2, 6]

    os.makedirs("resultados", exist_ok=True)

    resultados = []

    for r in raios:
        inicio = time.time()
        previsoes = r_N(X_treino, y_treino, X_teste, raio=r)
        tempo = time.time() - inicio

        resultado = dfNormTeste.copy()
        resultado["y_pred"] = previsoes
        resultado["Corretude"] = resultado["income"] == resultado["y_pred"]
        resultado.to_csv(f"resultados/previsoes/previsoes_tradicional_r_{r}.csv", index=False)

        acertos = np.sum(previsoes == y_teste)
        acuracia = acertos / len(y_teste)

        resultados.append({
            "raio": r,
            "acuracia": acuracia,
            "tempo": tempo
        })

        print(f"\nRaio: {r}")
        print(f"Acurácia do r-N Tradicional: {acuracia * 100:.2f}%")
        print(f"Tempo de execução: {tempo:.2f} segundos")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("resultados/acuracias_tradicional.csv", index=False)

    with open("resultados/resultados_tradicionais.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write("Resultados - r-N Tradicional (sem privacidade)\n\n")

        for resultado in resultados:
            arquivo.write(f"Raio: {resultado['raio']}\n")
            arquivo.write(f"Acurácia: {resultado['acuracia']:.4f} ({resultado['acuracia'] * 100:.2f}%)\n")
            arquivo.write(f"Tempo de execução: {resultado['tempo']:.2f} segundos\n")
            arquivo.write("-" * 40 + "\n")