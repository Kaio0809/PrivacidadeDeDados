import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# 1. CARREGAMENTO E LIMPEZA DOS DADOS

nome_arquivo_json = "mapeamento_generalizacao.json"
with open(nome_arquivo_json, "r", encoding="utf-8") as f:
    mapeamento = json.load(f)

mapeamento_municipio = mapeamento["municipio"]
mapeamento_data = mapeamento["data"]

colunas = ["municipioCaso", "dataNascimento", "racaCor"]

df_covid = pd.read_csv(
    "Datasets/covid.csv",
    usecols=lambda c: c.strip() in colunas,
    dtype=str,
)
df_covid.columns = df_covid.columns.str.strip()

df_covid = df_covid.dropna(subset=colunas)
for col in colunas:
    df_covid = df_covid[df_covid[col].str.strip() != ""]

df_covid = df_covid[df_covid["racaCor"] != "Sem Informacao"]
df_covid = df_covid[colunas].reset_index(drop=True)

def diagnosticar_mapeamento(df):
    for nome, col, mapa in [("municipio", "municipioCaso", mapeamento_municipio),
                            ("data", "dataNascimento", mapeamento_data)]:
        for nv in sorted(mapa, key=lambda x: int(x.replace("nivel", ""))):
            faltando = (~df[col].isin(mapa[nv].keys())).sum()
            if faltando:
                print(f"AVISO: {faltando} registros com '{col}' ausente em {nome}/{nv} (viram '*')")


diagnosticar_mapeamento(df_covid)

# 2. NÍVEIS DO RETICULADO

max_nm = max(int(k.replace("nivel", "")) for k in mapeamento_municipio if k.startswith("nivel")) + 1
max_nd = max(int(k.replace("nivel", "")) for k in mapeamento_data if k.startswith("nivel")) + 1


# 3. FUNÇÕES AUXILIARES

def obter_classes_equivalencia(df):
    return (
        df.groupby(["municipioCaso", "dataNascimento"], dropna=False)
        .size()
        .reset_index(name="tamanho")
    )


def verificar_k_anonimato(df, k):
    classes = obter_classes_equivalencia(df)
    menor_classe = classes["tamanho"].min()
    return menor_classe >= k, menor_classe, classes


def _generalizar_coluna(serie, mapa, n_niveis, inicios, tamanhos):
    """
    Para cada grupo (fatia contígua), acha o menor nível que o torna uniforme
    e devolve a coluna generalizada + nível de cada linha. Tudo vetorizado.
    """
    n = len(serie)
    gen, ok = [], []
    for nv in range(n_niveis):
        g = serie.map(mapa[f"nivel{nv}"]).fillna("*")
        gen.append(g.to_numpy(dtype=object))
        cod = pd.factorize(g)[0]
        ok.append(np.minimum.reduceat(cod, inicios) == np.maximum.reduceat(cod, inicios))
    ok = np.array(ok)
    tem_solucao = ok.any(axis=0)
    nivel_grupo = np.where(tem_solucao, ok.argmax(axis=0), n_niveis - 1)
    nivel_linha = np.repeat(nivel_grupo, tamanhos)

    saida = np.empty(n, dtype=object)
    for nv in range(n_niveis):
        m = nivel_linha == nv
        saida[m] = gen[nv][m]
    # Nenhum nível uniformiza o grupo (JSON incompleto): suprime o valor
    saida[np.repeat(~tem_solucao, tamanhos)] = "*"
    return saida, nivel_linha


# 4. ALGORITMO K-ANONIMATO E PRECISÃO

def anonimizar(df, k):
    dados = df.copy()
    dados["_regiao"] = dados["municipioCaso"].map(mapeamento_municipio["nivel1"])
    dados = dados.sort_values(
        by=["_regiao", "municipioCaso", "dataNascimento"]
    ).reset_index(drop=True)

    n = len(dados)
    n_grupos = max(n // k, 1)
    inicios = np.arange(n_grupos) * k
    tamanhos = np.diff(np.append(inicios, n))

    mun, nm = _generalizar_coluna(dados["municipioCaso"], mapeamento_municipio, max_nm, inicios, tamanhos)
    dat, nd = _generalizar_coluna(dados["dataNascimento"], mapeamento_data, max_nd, inicios, tamanhos)

    dados["municipioCaso"] = mun
    dados["dataNascimento"] = dat
    dados["_nivel_municipio"] = nm
    dados["_nivel_data"] = nd
    dados = garantir_k(dados, k)
    return dados.drop(columns=["_regiao"])


def garantir_k(dados, k):
    chave = ["municipioCaso", "dataNascimento"]
    if len(dados) < k:
        return dados
    for _ in range(10000):
        tam = dados.groupby(chave, dropna=False)["municipioCaso"].transform("size")
        ruim = (tam < k).to_numpy()
        estrela = ((dados["municipioCaso"] == "*") & (dados["dataNascimento"] == "*")).to_numpy()
        if not ruim.any() and not (0 < estrela.sum() < k):
            break
        marca = ruim.copy()
        falta = k - (estrela | marca).sum()
        if 0 < (estrela | marca).sum() < k:
            candidatos = np.where(~(estrela | marca))[0]
            extras = candidatos[np.argsort(tam.to_numpy()[candidatos], kind="stable")[:falta]]
            marca[extras] = True
        dados.loc[marca, chave] = "*"
        dados.loc[marca, "_nivel_municipio"] = max_nm - 1
        dados.loc[marca, "_nivel_data"] = max_nd - 1
    return dados


def calcular_precisao(df_anonimizado):
    n_a = 2
    n_d = len(df_anonimizado)
    hg_municipio = max(max_nm - 1, 1)
    hg_data = max(max_nd - 1, 1)

    soma_perda = (
        df_anonimizado["_nivel_municipio"].sum() / hg_municipio
        + df_anonimizado["_nivel_data"].sum() / hg_data
    )
    return 1.0 - soma_perda / (n_d * n_a)


# 5. EXECUÇÃO

if __name__ == "__main__":
    k = int(input("Digite o k = {2,4,8,16}: "))

    print(f"EXECUTANDO K-ANONIMATO (K = {k})")

    df_anon = anonimizar(df_covid, k=k)
    precisao = calcular_precisao(df_anon)

    df_salvar = df_anon[colunas]
    df_salvar.to_csv(f"Datasets/covid_{k}.csv", index=False)

    df_lido = pd.read_csv(f"Datasets/covid_{k}.csv", dtype=str, keep_default_na=False)
    eh_k_anon, menor_tam, classes = verificar_k_anonimato(df_lido, k)
    tamanho_medio_classes = len(df_salvar) / len(classes)

    print(f"Dataset gerado: covid_{k}.csv")
    print("Níveis (nm, nd) usados -> nº de registros:")
    print(df_anon.groupby(["_nivel_municipio", "_nivel_data"]).size().to_string())
    print(f"Precisão (Prec): {precisao:.4f}")
    print(f"É {k}-anônimo? {eh_k_anon} (menor classe = {menor_tam})")
    print(f"Tamanho Médio das Classes de Equivalência: {tamanho_medio_classes:.2f}")
    print(f"Total de classes: {len(classes)}")

    plt.figure(figsize=(8, 5))
    plt.hist(classes["tamanho"], bins=30, edgecolor="black", alpha=0.7)
    plt.title(f"Distribuição do Tamanho das Classes de Equivalência (K = {k})")
    plt.xlabel("Tamanho da Classe de Equivalência")
    plt.ylabel("Frequência")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"Histogramas/histograma_k_{k}.png")
    plt.close()

    print("\nProcesso concluído com sucesso!")