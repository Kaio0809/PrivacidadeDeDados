import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

resultados = [
    {'e': 0.5, 'df': None, 'acuracia': None},
    {'e': 1.0, 'df': None, 'acuracia': None},
    {'e': 5.0, 'df': None, 'acuracia': None},
    {'e': 10.0, 'df': None, 'acuracia': None}
]


def acuracia(df):
    return sum(df['Corretude']) / df.shape[0]


for dic in resultados:
    dic['df'] = pd.read_csv(f'resultados/resultados_privados_epsilon_{dic['e']}.csv', sep=',')
    dic['acuracia'] = acuracia(dic['df'])

knnTrad = pd.read_csv('resultados/df_resultados_testes.csv', sep=',')
acuracia_knnT = acuracia(knnTrad)

plot = {x['e']: x['acuracia'] for x in resultados}

es = plot.keys() 
acs = plot.values()

fig, ax = plt.subplots()

ax.plot(es, acs, marker='o', label='Acuracia KNN Privado')
ax.axhline(y=acuracia_knnT, color='r', linestyle='--', label='Acuracia KNN Tradicional')
ax.set_ylim(0, 1)

plt.legend()
plt.savefig('grafico_acuracias.png', dpi=300, bbox_inches='tight')
plt.show()

