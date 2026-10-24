import pandas as pd
import numpy as np
import time
import random

random.seed(42)
np.random.seed(42)

df = pd.read_csv('df_pronto.csv', sep=',')

indices = list(range(len(df)))
random.shuffle(indices)

limite = int(len(indices) * 0.70)
indices_treino = indices[:limite]
indices_teste = indices[limite:]

df_treino = df.iloc[indices_treino].copy()
df_teste = df.iloc[indices_teste].copy()

X_treino = df_treino.drop(columns=['income']).to_numpy(dtype=np.float64)
y_treino = df_treino['income'].to_numpy(dtype=np.int32)

X_teste = df_teste.drop(columns=['income']).to_numpy(dtype=np.float64)
y_teste = df_teste['income'].to_numpy(dtype=np.int32)

classes = np.unique(y_treino)
n_classes = len(classes)
n_teste = len(X_teste)

raio = 6.0
epsilons = [0.5, 1.0, 5.0, 10.0]

def calcular_contagens_raio(X_treino, y_treino, X_teste, classes, raio=6.0, tamanho_bloco=256):
    contagens = np.zeros((len(X_teste), len(classes)), dtype=np.int32)
    norma_treino = np.sum(X_treino ** 2, axis=1)
    raio_sq = raio ** 2
    mascaras_classes = [y_treino == c for c in classes]

    for i in range(0, len(X_teste), tamanho_bloco):
        fim = min(i + tamanho_bloco, len(X_teste))
        bloco = X_teste[i:fim]
        norma_teste = np.sum(bloco ** 2, axis=1)

        dist_sq = norma_teste[:, None] + norma_treino[None, :] - 2 * (bloco @ X_treino.T)
        np.maximum(dist_sq, 0, out=dist_sq)

        dentro_raio = dist_sq <= raio_sq

        for j, mascara in enumerate(mascaras_classes):
            contagens[i:fim, j] = np.count_nonzero(dentro_raio[:, mascara], axis=1)

    return contagens

def estimar_sensibilidade_mcs(X_teste, raio=6.0, tamanho_bloco=512):
    
    n = len(X_teste)
    graus = np.zeros(n, dtype=np.int32)
    raio_cruzado_sq = (2 * raio) ** 2
    normas = np.sum(X_teste ** 2, axis=1)

    for i in range(0, n, tamanho_bloco):
        fim = min(i + tamanho_bloco, n)
        bloco = X_teste[i:fim]
        norma_bloco = normas[i:fim]

        dist_sq = norma_bloco[:, None] + normas[None, :] - 2 * (bloco @ X_teste.T)
        np.maximum(dist_sq, 0, out=dist_sq)

        sobreposicao = dist_sq <= raio_cruzado_sq
        graus[i:fim] = np.sum(sobreposicao, axis=1) - 1

    mcs_estimado = int(np.max(graus)) + 1
    return max(mcs_estimado, 1)

if __name__ == '__main__':
    print("Calculando contagens reais de vizinhos no raio r=6...")
    inicio_contagem = time.time()
    contagens_reais = calcular_contagens_raio(X_treino, y_treino, X_teste, classes, raio=raio)
    tempo_contagem = time.time() - inicio_contagem

    print("Construindo o Grafo de Sobreposição e estimando o MCS (Sensibilidade)...")
    mcs = estimar_sensibilidade_mcs(X_teste, raio=raio)
    print(f"Sensibilidade (MCS estimado): {mcs}")

    for epsilon in epsilons:
        inicio_p = time.time()

        epsilon_efetivo = epsilon / (mcs * n_classes)
        escala_laplace = 1.0 / epsilon_efetivo

        ruido = np.random.laplace(loc=0.0, scale=escala_laplace, size=contagens_reais.shape)
        contagens_ruidosas = contagens_reais + ruido

        indices_previstos = np.argmax(contagens_ruidosas, axis=1)
        previsoes_privadas = classes[indices_previstos]

        acuracia = np.mean(previsoes_privadas == y_teste)
        tempo_predicao = time.time() - inicio_p

        resultado = df_teste.copy()
        resultado['y_pred'] = previsoes_privadas
        resultado['Corretude'] = resultado['income'] == resultado['y_pred']
        resultado.to_csv(f'resultados/resultados_privados_epsilon_{epsilon}.csv', index=False)

        with open(f'resultados/resultado_privado_epsilon_{epsilon}.txt', 'w', encoding='utf-8') as arquivo:
            arquivo.write(f'Resultados kNN Privado - Laplace (Epsilon={epsilon})\n\n')
            arquivo.write(f'Epsilon total: {epsilon}\n')
            arquivo.write(f'Epsilon por consulta (efetivo): {epsilon_efetivo:.10f}\n')
            arquivo.write(f'Sensibilidade (MCS): {mcs}\n')
            arquivo.write(f'Escala do ruído de Laplace: {escala_laplace:.2f}\n')
            arquivo.write(f'Acurácia: {acuracia:.4f} ({acuracia * 100:.2f}%)\n')
            arquivo.write(f'Total de previsões: {n_teste}\n')
            arquivo.write(f'Tempo total de execução: {tempo_contagem + tempo_predicao:.2f} segundos\n')

        print(f"Epsilon = {epsilon:4.1f} | Acurácia = {acuracia * 100:6.2f}% | Escala do Ruído = {escala_laplace:8.2f}")

    print("\nProcessamento do kNN Privado concluído com sucesso!")