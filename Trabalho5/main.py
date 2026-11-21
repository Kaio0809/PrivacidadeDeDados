
import pandas as pd
from pathlib import Path

# Localiza os resultados produzidos pelo módulo de avaliação.
pasta = Path(__file__).resolve().parent / 'resultados'

resumo = pd.read_csv(pasta / 'resumo_mae.csv')
consultas = pd.read_csv(pasta / 'resumo_consultas.csv')

# Apresenta o MAE médio e o desvio padrão de cada configuração experimental.
print('\n===== RESULTADOS GERAIS =====')
print(resumo.round(2).to_string(index=False))

# Compara as larguras dos bins da E2 utilizando a média dos MAEs
# obtidos nos quatro valores de epsilon, com e sem truncamento.
print('\n===== COMPARAÇÃO DOS BINS =====')
bins = resumo[resumo['estrategia'] == 'E2']
comparacao_bins = bins.groupby('bin')[['mae_medio_sem', 'mae_medio_com']].mean().sort_values('mae_medio_com')
print(comparacao_bins.round(2))

# Seleciona como largura principal aquela que apresentou o menor
# MAE médio com truncamento, considerando todos os epsilons.
bin_principal = int(comparacao_bins.index[0])
print(f'\nMelhor bin pela média dos epsilons: {bin_principal} anos')

# Identifica a melhor largura de bin para cada orçamento de privacidade.
print('\n===== MELHOR BIN POR EPSILON =====')
for epsilon in sorted(resumo['epsilon'].unique()):
    grupo = bins[bins['epsilon'] == epsilon].sort_values('mae_medio_com')
    melhor = grupo.iloc[0]
    print(f"Epsilon {epsilon}: bin {int(melhor['bin'])} | MAE: {melhor['mae_medio_com']:.2f}")

# Compara a E1 com a E2 utilizando uma largura fixa de bin.
# O critério de comparação é o menor MAE médio com truncamento.
print('\n===== E1 X E2 =====')
comparacao = []

for epsilon in sorted(resumo['epsilon'].unique()):
    grupo = resumo[resumo['epsilon'] == epsilon]
    e1 = grupo[grupo['estrategia'] == 'E1'].iloc[0]
    e2 = grupo[(grupo['estrategia'] == 'E2') & (grupo['bin'] == bin_principal)].iloc[0]

    mae_e1 = e1['mae_medio_com']
    mae_e2 = e2['mae_medio_com']
    vencedor = 'E1' if mae_e1 < mae_e2 else 'E2' if mae_e2 < mae_e1 else 'Empate'

    # Armazena os resultados para a comparação final entre estratégias.
    comparacao.append({
        'epsilon': epsilon,
        'mae_e1': mae_e1,
        'desvio_e1': e1['desvio_com'],
        'mae_e2': mae_e2,
        'desvio_e2': e2['desvio_com'],
        'melhor': vencedor
    })

    print(f'\nEpsilon: {epsilon}')
    print(f"E1: {mae_e1:.2f} ± {e1['desvio_com']:.2f}")
    print(f"E2 (bin {bin_principal}): {mae_e2:.2f} ± {e2['desvio_com']:.2f}")
    print(f'Menor MAE: {vencedor}')

# Verifica a influência do truncamento sobre o MAE.
# A redução corresponde à diferença entre os erros médios
# antes e depois de limitar as respostas ao intervalo [0, n].
print('\n===== EFEITO DO TRUNCAMENTO =====')
for _, linha in resumo.iterrows():
    nome = 'E1' if linha['estrategia'] == 'E1' else f"E2 (bin {int(linha['bin'])})"
    diferenca = linha['mae_medio_sem'] - linha['mae_medio_com']
    print(f"{nome} | epsilon {linha['epsilon']} | Sem: {linha['mae_medio_sem']:.2f} | Com: {linha['mae_medio_com']:.2f} | Redução: {diferenca:.2f}")

# Analisa individualmente as dez consultas para epsilon igual a 1.
# As consultas são ordenadas pelo MAE médio com truncamento.
print('\n===== MAE POR CONSULTA (EPSILON = 1) =====')
dados = consultas[consultas['epsilon'] == 1.0]

for (estrategia, largura), grupo in dados.groupby(['estrategia', 'bin']):
    grupo = grupo.sort_values('mae_medio_com')
    nome = 'E1' if estrategia == 'E1' else f'E2 (bin {int(largura)})'

    print(f'\n{nome}')
    print(grupo[['consulta', 'mae_medio_sem', 'mae_medio_com']].round(2).to_string(index=False))
    print(f"Menor erro: {grupo.iloc[0]['consulta']} ({grupo.iloc[0]['mae_medio_com']:.2f})")
    print(f"Maior erro: {grupo.iloc[-1]['consulta']} ({grupo.iloc[-1]['mae_medio_com']:.2f})")

# Resume quantos valores de epsilon favoreceram cada estratégia,
# considerando a largura principal escolhida para a E2.
print('\n===== RESUMO FINAL =====')
melhores = pd.DataFrame(comparacao)

print(f"Bin principal: {bin_principal} anos")
print(f"E1 venceu em {(melhores['melhor'] == 'E1').sum()} epsilons")
print(f"E2 venceu em {(melhores['melhor'] == 'E2').sum()} epsilons")

# Salva as comparações para utilização no relatório e no notebook.
comparacao_bins.to_csv(pasta / 'comparacao_bins.csv')
melhores.to_csv(pasta / 'comparacao_estrategias.csv', index=False)

print('\nAnálise concluída.')
