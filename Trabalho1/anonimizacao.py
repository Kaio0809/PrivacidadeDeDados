import pandas as pd
import unicodedata
import json
import matplotlib.pyplot as plt

df_covid = pd.read_csv('covid.csv',sep=',', encoding='UTF-8')
df_covid = df_covid.dropna(subset=['municipioCaso'])
df_covid = df_covid.dropna(subset=['dataNascimento'])
df_covid.info()

df_ibge = pd.read_csv('map_municipio_regiao.csv')
df_ibge

def remover_acento(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

regioes = {}
def mapamun(coluna, dicionario):
    for item in coluna:
        regiao = df_ibge.loc[df_ibge['municipio'] == remover_acento(item), 'regiao']
        if not regiao.empty:
            dicionario[item] = regiao.iloc[0]
        else:
            dicionario[item] = None

mapamun(df_covid['municipioCaso'].unique(), regioes)

datas = list(df_covid['dataNascimento'].unique())
datan1 = lambda x: x[:-3]
datan2 = lambda x: x[:-6]
def datan3(dt):
    intdt = int((int(dt) / 10) + 0.5)
    if (int(dt) % 10) < 5:
        return str((intdt*10) + 1) + ' - ' + str((intdt*10) + 5)
    elif (int(dt) % 10 == 5):
        return str(((intdt - 1)*10) + 1) + ' - ' + str(((intdt - 1)*10) + 5)
    else:
        return str((intdt*10) - 4) + ' - ' + str((intdt*10))

mapeamento_municipio = {
    'nivel0': dict(zip(regioes.keys(), regioes.keys())),
    'nivel1': regioes,
    'nivel2': {m: 'CEARA' for m in regioes.keys()}
}

mapeamento_data = {
    'nivel0': dict(zip(datas, datas)),
    'nivel1': {dt: datan1(dt) for dt in datas if isinstance(dt, str)},
    'nivel2': {dt: datan2(dt) for dt in datas if isinstance(dt, str)},
    'nivel3': {dt: datan3(dt[:-6]) for dt in datas if isinstance(dt, str)}
}
mapeamentos = {'municipio': mapeamento_municipio, 'data': mapeamento_data}

with open('mapeamento_generalizacao.json', 'w', encoding='utf-8') as arquivo:
    json.dump(mapeamentos, arquivo, ensure_ascii=False, indent=4)

binss_municipio = {'nivel0': len(regioes.keys()), 'nivel1': len(set(regioes.values())), 'nivel2': 1}
binss_data = {'nivel0': len(set(datas)), 'nivel1': len(set(mapeamento_data['nivel1'].values())), 'nivel2': len(set(mapeamento_data['nivel2'].values())), 'nivel3': len(set(mapeamento_data['nivel3'].values()))}

def gerar_histograma(coluna,bins, titulo, original=False):
    plt.figure(figsize=(50,5))
    plt.hist(coluna, bins=bins, color='skyblue', edgecolor='black')
    plt.title(titulo)
    plt.xlabel('Atributo')
    plt.ylabel('Quantidade')
    if original:
        plt.xticks([])
    plt.xticks(rotation=45)
    plt.savefig(f'Histogramas/{titulo}.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    #gerar_histograma(df_covid['municipioCaso'], len(regioes.keys()), f'HIST_Municipio_Original', original=True)
    #gerar_histograma(df_covid['dataNascimento'], len(datas), f'HIST_data_Original', original=True)
    while True:
        nm = int(input('Digite o nível de abstração desejado para município (0,1,2): '))
        nd = int(input('Digite o nível de abstração desejado para data de nascimento (0,1,2,3): '))
        if (nm == -1) or (nd == -1):
            break
        if (nm not in [0,1,2]) or (nd not in [0,1,2,3]):
            print("Nível Inválido! - Tente Novamente")
            continue

        DT_nm_nd = df_covid.copy()
        DT_nm_nd['municipioCaso'] = DT_nm_nd['municipioCaso'].map(mapeamento_municipio[f'nivel{nm}'])
        DT_nm_nd['dataNascimento'] = DT_nm_nd['dataNascimento'].map(mapeamento_data[f'nivel{nd}'])
        print(f"Gerando DT_nivel{nm}_nivel{nd}...",'\n')
        DT_nm_nd.to_csv(f'Datasets/DT_municipio(n{nm})_data(n{nd}).csv')
        nmzero = False
        ndzero = False
        if nm == 0:
            nmzero = True
        if nd == 0:
            ndzero = True
        gerar_histograma(DT_nm_nd['municipioCaso'], binss_municipio[f'nivel{nm}'], titulo=f'HIST_municipioCaso_{nm}', original=nmzero)
        gerar_histograma(DT_nm_nd['dataNascimento'], binss_data[f'nivel{nd}'], titulo=f'HIST_dataNascimento_{nd}', original=ndzero)


    return

main()