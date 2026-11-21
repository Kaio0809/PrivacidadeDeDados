
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cliente import gerar_relatorios, agrupar_idades
from servidor import estimar_frequencias, estimar_consulta, estimar_consulta_bins

pasta = Path(__file__).resolve().parent
saida = pasta / 'resultados'
saida.mkdir(exist_ok=True)

df = pd.read_csv(pasta / 'dados_censo.csv')

df['idade'] = pd.to_numeric(df['idade'], errors='raise')
df['salario'] = df['salario'].astype('string').str.strip().str.replace(r'\s+', '', regex=True)



df['idade'] = df['idade'].astype(int)
dados = list(zip(df['idade'], df['salario']))
dom_sal = ['<=50K', '>50K']
dom_conj = [(idade, sal) for idade in range(17, 91) for sal in dom_sal]

consultas = {
    'Q1': (30, 50, '>50K'), 'Q2': (None, 19, '<=50K'),
    'Q3': (60, None, '>50K'), 'Q4': (20, 30, '>50K'),
    'Q5': (40, 60, '>50K'), 'Q6': (None, 30, '<=50K'),
    'Q7': (40, None, '<=50K'), 'Q8': (30, 40, '<=50K'),
    'Q9': (50, None, None), 'Q10': (None, None, '>50K')
}


def resposta_verdadeira(df, idade_min=None, idade_max=None, salario=None):
    dados = df
    if idade_min is not None:
        dados = dados[dados['idade'] >= idade_min]
    if idade_max is not None:
        dados = dados[dados['idade'] <= idade_max]
    if salario is not None:
        dados = dados[dados['salario'] == salario]
    return len(dados)


reais = {nome: resposta_verdadeira(df, *parametros) for nome, parametros in consultas.items()}
epsilons = [0.5, 1.0, 2.0, 4.0]
larguras = [2, 5, 10]
sementes = np.random.default_rng(2026).choice(1_000_000, 10, replace=False).tolist()

configuracoes = [('E1', 0, dados, dom_conj, None)]
for largura in larguras:
    bins, dados_bins = agrupar_idades(dados, largura)
    dominio = [(i, sal) for i in range(len(bins)) for sal in dom_sal]
    configuracoes.append(('E2', largura, dados_bins, dominio, bins))

execucoes = []
consultas_avaliadas = []
n = len(df)

print(f'Indivíduos: {n} | Domínio E1: {len(dom_conj)}')
print(f'Sementes: {sementes}')
for estrategia, largura, entrada, dominio, bins in configuracoes:
    print(f'\n{estrategia} | bin: {largura if largura else "sem bins"} | domínio: {len(dominio)}')

    for epsilon in epsilons:
        for repeticao, semente in enumerate(sementes, 1):
            relatorios = gerar_relatorios(entrada, dominio, epsilon, semente)
            estimativas = estimar_frequencias(relatorios, dominio, epsilon)
            erros_sem, erros_com = [], []

            for nome, parametros in consultas.items():
                if estrategia == 'E1':
                    estimada = estimar_consulta(estimativas, *parametros)
                else:
                    estimada = estimar_consulta_bins(estimativas, bins, *parametros)

                estimada_com = max(0, min(n, estimada))
                real = reais[nome]
                erro_sem = abs(real - estimada)
                erro_com = abs(real - estimada_com)
                erros_sem.append(erro_sem)
                erros_com.append(erro_com)

                consultas_avaliadas.append({
                    'estrategia': estrategia, 'bin': largura, 'epsilon': epsilon,
                    'repeticao': repeticao, 'semente': semente, 'consulta': nome,
                    'real': real, 'estimada_sem': estimada, 'estimada_com': estimada_com,
                    'erro_sem': erro_sem, 'erro_com': erro_com
                })

            execucoes.append({
                'estrategia': estrategia, 'bin': largura, 'epsilon': epsilon,
                'repeticao': repeticao, 'semente': semente,
                'mae_sem': np.mean(erros_sem), 'mae_com': np.mean(erros_com)
            })

        print(f'  epsilon {epsilon}: 10 repetições concluídas')

df_exec = pd.DataFrame(execucoes)
df_cons = pd.DataFrame(consultas_avaliadas)
resumo = df_exec.groupby(['estrategia', 'bin', 'epsilon']).agg(
    mae_medio_sem=('mae_sem', 'mean'), desvio_sem=('mae_sem', 'std'),
    mae_medio_com=('mae_com', 'mean'), desvio_com=('mae_com', 'std')
).reset_index()
resumo_cons = df_cons.groupby(['estrategia', 'bin', 'epsilon', 'consulta']).agg(
    mae_medio_sem=('erro_sem', 'mean'), mae_medio_com=('erro_com', 'mean'),
    desvio_sem=('erro_sem', 'std'), desvio_com=('erro_com', 'std')
).reset_index()

melhores = resumo[resumo['estrategia'] == 'E2']
melhores = melhores.loc[melhores.groupby('epsilon')['mae_medio_com'].idxmin()].sort_values('epsilon')

print('\nRESUMO DO MAE')
print(resumo.round(2).to_string(index=False))
print('\nMELHOR BIN POR EPSILON (COM TRUNCAMENTO)')
print(melhores[['epsilon', 'bin', 'mae_medio_com']].round(2).to_string(index=False))
print(f'\nExecuções: {len(df_exec)} | Consultas avaliadas: {len(df_cons)} | Sementes: {df_exec.semente.nunique()}')

assert len(df_exec) == (1 + len(larguras)) * len(epsilons) * len(sementes)
assert len(df_cons) == len(df_exec) * len(consultas)
assert df_exec['semente'].nunique() == 10
assert (df_exec.groupby(['estrategia', 'bin', 'epsilon']).size() == 10).all()

df_exec.to_csv(saida / 'execucoes.csv', index=False)
df_cons.to_csv(saida / 'consultas.csv', index=False)
resumo.to_csv(saida / 'resumo_mae.csv', index=False)
resumo_cons.to_csv(saida / 'resumo_consultas.csv', index=False)
melhores.to_csv(saida / 'melhores_bins.csv', index=False)
pd.DataFrame({'semente': sementes}).to_csv(saida / 'sementes.csv', index=False)

for tipo in ['sem', 'com']:
    plt.figure(figsize=(9, 5))
    for (estrategia, largura), grupo in resumo.groupby(['estrategia', 'bin'], sort=False):
        nome = 'E1' if estrategia == 'E1' else f'E2 - bin {largura}'
        plt.errorbar(grupo['epsilon'], grupo[f'mae_medio_{tipo}'], yerr=grupo[f'desvio_{tipo}'],
                     marker='o', capsize=3, label=nome)
    plt.title(f'MAE por epsilon - {"sem" if tipo == "sem" else "com"} truncamento')
    plt.xlabel('Epsilon')
    plt.ylabel('MAE médio (escala linear)')
    plt.xticks(epsilons)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(saida / f'mae_epsilon_{tipo}.png', dpi=300)
    plt.close()

plt.figure(figsize=(9, 5))
for epsilon in epsilons:
    grupo = resumo[(resumo['estrategia'] == 'E2') & (resumo['epsilon'] == epsilon)].sort_values('bin')
    plt.plot(grupo['bin'], grupo['mae_medio_com'], marker='o', label=f'epsilon {epsilon}')
plt.title('E2 - MAE por largura dos bins (com truncamento)')
plt.xlabel('Largura do bin (anos)')
plt.ylabel('MAE médio')
plt.xticks(larguras)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(saida / 'mae_bins.png', dpi=300)
plt.close()

bin_escolhido = int(melhores.loc[melhores['epsilon'] == 1.0, 'bin'].iloc[0])
ordem = [f'Q{i}' for i in range(1, 11)]
e1 = resumo_cons[(resumo_cons['estrategia'] == 'E1') & (resumo_cons['epsilon'] == 1.0)]
e2 = resumo_cons[(resumo_cons['estrategia'] == 'E2') & (resumo_cons['epsilon'] == 1.0) & (resumo_cons['bin'] == bin_escolhido)]
e1 = e1.set_index('consulta').loc[ordem]
e2 = e2.set_index('consulta').loc[ordem]

x = np.arange(10)
plt.figure(figsize=(11, 5))
plt.bar(x - 0.2, e1['mae_medio_com'], 0.4, label='E1')
plt.bar(x + 0.2, e2['mae_medio_com'], 0.4, label=f'E2 - bin {bin_escolhido}')
plt.title('MAE por consulta - epsilon 1 (com truncamento)')
plt.xlabel('Consulta')
plt.ylabel('MAE médio')
plt.xticks(x, ordem)
plt.legend()
plt.tight_layout()
plt.savefig(saida / 'mae_consultas.png', dpi=300)
plt.close()

print(f'\nArquivos salvos em: {saida}')
