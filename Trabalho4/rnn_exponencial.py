
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

caminhoTreino = "df_treino.csv"
caminhoTeste = "df_teste.csv"

raios = [0.5, 1, 1.5, 2, 6]
epsilons = [0.1, 0.5, 1, 5, 10, 100]

SEMENTE = 42
TAMANHO_BLOCO = 256

df_tradicional = pd.read_csv("resultados/acuracias_tradicional.csv")
acuracias_tradicional = dict(zip(df_tradicional["raio"], df_tradicional["acuracia"]))

rng = np.random.default_rng(SEMENTE)


def calcular_contagens_raio(X_treino, y_treino, X_teste, classes, raio, tamanho_bloco=256):

    contagens = np.zeros((len(X_teste), len(classes)),dtype=np.int32)

    norma_treino = np.sum(X_treino ** 2, axis=1)
    raio_sq = raio ** 2

    mascaras_classes = [y_treino == classe for classe in classes]

    for i in range(0, len(X_teste), tamanho_bloco):
        fim = min(i + tamanho_bloco, len(X_teste))
        bloco = X_teste[i:fim]
        norma_teste = np.sum(bloco ** 2, axis=1)
        dist_sq = (norma_teste[:, None] + norma_treino[None, :] - 2 * (bloco @ X_treino.T))

        np.maximum(dist_sq, 0, out=dist_sq)
        dentro_raio = dist_sq <= raio_sq

        for j, mascara in enumerate(mascaras_classes):
            contagens[i:fim, j] = np.count_nonzero(dentro_raio[:, mascara], axis=1)

    return contagens


def r_N_exponencial(contagens, classes, epsilon, rng):
    previsoes = []
    epsilon_individual = epsilon / len(contagens)

    for contagem in contagens:
        pontuacoes = (epsilon_individual * contagem) / 2.0
        pontuacoes -= np.max(pontuacoes)

        pesos = np.exp(pontuacoes)
        probabilidades = pesos / np.sum(pesos)

        indice = rng.choice(len(classes),p=probabilidades)
        previsoes.append(classes[indice])

    return np.array(previsoes)


def calcular_acuracia(y_real, y_pred):
    return np.mean(y_real == y_pred)


def main():
    os.makedirs("resultados", exist_ok=True)

    df_treino = pd.read_csv(caminhoTreino)
    df_teste = pd.read_csv(caminhoTeste)

    X_treino = df_treino.drop(columns=['income']).to_numpy(dtype=np.float64)
    y_treino = df_treino["income"].to_numpy(dtype=np.int32)

    X_teste = df_teste.drop(columns=['income']).to_numpy(dtype=np.float64)
    y_teste = df_teste["income"].to_numpy(dtype=np.int32)

    classes = np.unique(y_treino)

    print("Quantidade de registros de treino:", len(X_treino))
    print("Quantidade de registros de teste:", len(X_teste))
    print("Classes:", classes)

    resultados = []

    for raio in raios:
        print("\n" + "=" * 50)
        print(f"RAIO: {raio}")
        print("=" * 50)

        inicio_contagem = time.time()
        contagens = calcular_contagens_raio(X_treino, y_treino, X_teste, classes, raio, TAMANHO_BLOCO)
        tempo_contagem = time.time() - inicio_contagem

        for epsilon in epsilons:
            print(f"\nEpsilon: {epsilon}")

            inicio = time.time()
            previsoes = r_N_exponencial(contagens, classes, epsilon, rng)
            tempo_mecanismo = time.time() - inicio
            epsilon_individual = epsilon / len(X_teste)

            acuracia = calcular_acuracia(y_teste, previsoes)
            tempo_total = tempo_contagem + tempo_mecanismo

            print(f"Epsilon individual: {epsilon_individual:.6f}")
            print(f"Acurácia: {acuracia * 100:.2f}%")
            print(f"Tempo de Execução: {tempo_total:.2f} segundos")

            resultados.append({"raio": raio, "epsilon": epsilon, "epsilon_individual": epsilon_individual, "acuracia": acuracia, "tempo_contagem": tempo_contagem, "tempo_mecanismo": tempo_mecanismo, "tempo_total": tempo_total})

            resultado = df_teste.copy()
            resultado['y_pred'] = previsoes
            resultado['Corretude'] = resultado['income'] == resultado['y_pred']

            resultado.to_csv(f"resultados/previsoes/previsoes_r_{raio}_eps_{epsilon}.csv", index=False)

    df_resultados = pd.DataFrame(resultados)

    with open("resultados/resultados.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write("RESULTADOS DO MECANISMO EXPONENCIAL\n")
        arquivo.write("=" * 90 + "\n\n")
        arquivo.write(f"Registros de treino: {len(X_treino)}\n")
        arquivo.write(f"Registros de teste: {len(X_teste)}\n")
        arquivo.write(f"Classes: {classes.tolist()}\n\n")

        for _, resultado in df_resultados.iterrows():
            arquivo.write(f"Raio: {resultado['raio']} | Epsilon: {resultado['epsilon']}\n")
            arquivo.write(f"Epsilon individual: {resultado['epsilon_individual']:.6f}\n")
            arquivo.write(f"Acurácia: {resultado['acuracia'] * 100:.2f}%\n")
            arquivo.write(f"Tempo de contagem: {resultado['tempo_contagem']:.2f} segundos\n")
            arquivo.write(f"Tempo do mecanismo: {resultado['tempo_mecanismo']:.2f} segundos\n")
            arquivo.write(f"Tempo total: {resultado['tempo_total']:.2f} segundos\n")
            arquivo.write("-" * 90 + "\n")

    print("\n" + "=" * 50)
    print("RESULTADOS FINAIS")
    print("=" * 50)
    print(df_resultados.to_string(index=False))

    df_grafico = df_resultados.copy()
    df_grafico["acuracia_percentual"] = df_grafico["acuracia"] * 100

    plt.figure(figsize=(13, 8))

    cores = plt.cm.tab10(np.linspace(0, 1, len(raios)))

    for raio, cor in zip(raios, cores):
        dados = df_grafico[df_grafico["raio"] == raio].sort_values("epsilon")

        plt.plot(dados["epsilon"], dados["acuracia_percentual"], marker="o", linewidth=2.5, markersize=7, color=cor, label=f"Privado - r = {raio}")

        plt.axhline(y=acuracias_tradicional[raio] * 100, color=cor, linestyle="--", linewidth=1.8, alpha=0.8, label=f"Tradicional - r = {raio}")

    plt.xscale("log")
    plt.xticks(epsilons, [str(e) for e in epsilons])
    plt.xlabel("Valor de ε (epsilon)", fontsize=12)
    plt.ylabel("Acurácia (%)", fontsize=12)
    plt.title("Comparação da Acurácia: r-N Tradicional vs. Mecanismo Exponencial", fontsize=15, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.ylim(0, 100)
    plt.legend(title="Método e raio", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig("resultados/grafico_acuracia.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
