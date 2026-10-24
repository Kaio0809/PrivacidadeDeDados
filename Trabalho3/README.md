# Trabalho 3: Privacidade Diferencial em kNN usando o Mecanismo de Laplace

**Disciplina:** Privacidade de Dados (2026)  
**Professores:** Javam Machado

---

## Alunos

**Kaio Edmar Castro Barreto - 557246**   
**Cauã de Freitas Souza - 554583**


## Descrição do Projeto

Este projeto tem como objetivo implementar e comparar o desempenho do algoritmo de classificação **k-Nearest Neighbors (kNN)** tradicional (*sem privacidade*) contra uma versão **diferencialmente privada ($\epsilon$-DP)** que utiliza o **Mecanismo de Laplace**.

O estudo baseia-se na metodologia proposta por *Gursoy et al. (2017)* (*"Differentially private nearest neighbor classification"*), adaptando a contagem de vizinhos em um raio fixo $r=6.0$ e estimando a sensibilidade global por meio do **Grafo de Sobreposição de Regiões (Region Overlap Graph)**.

O dataset utilizado é o **Adult Income Dataset**, onde a tarefa de classificação consiste em prever se a renda de um indivíduo excede 50K dólares por ano (`income`).

---

## Pré-processamento dos Dados

Conforme exigido nas especificações do trabalho:
1. **Remoção de Atributos Irrelevantes:** Eliminação das colunas `fnlwgt`, `education`, `capital-gain`, `capital-loss` e `hours-per-week`.
2. **Tratamento de Dados Faltantes:** Remoção de todas as linhas contendo valores faltantes marcados como `?`.
3. **Codificação Categórica:** Mapeamento de atributos textuais/categóricos para valores inteiros ordenados.
4. **Divisão da Base:** Particionamento aleatório determinístico (seed = 42) em **70% para treino** ($X_{train}$) e **30% para teste** ($X_{test}$).

---

## Estrutura do Projeto e Resultados

```text
.
├── knn_tradicional.py                  # Script 1: Pré-processamento e kNN sem privacidade
├── knn_privado.py                      # Script 2: kNN com Mecanismo de Laplace e Grafo de Sobreposição
├── gerar_grafico.py                    # Script 3: Geração do gráfico comparativo de acurácias
├── df_pronto.csv                       # Dataset limpo e codificado gerado pelo Script 1
├── grafico_acuracias.png               # Gráfico gerado contendo a curva DP vs Tradicional
└── resultados/                         # Diretório onde são gravados os outputs
    ├── df_resultados_testes.csv        # Predições do kNN tradicional
    ├── resultado.txt                   # Relatório do kNN tradicional
    ├── resultados_privados_epsilon_*.csv # Predições para epsilon = 0.5, 1.0, 5.0, 10.0
    └── resultado_privado_epsilon_*.txt   # Detalhes de execução e ruído para cada epsilon
```

---

## Como Executar o Projeto

### 1. Pré-requisitos
Instale as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```

### 2. Execução do kNN Tradicional
Baixa o dataset, realiza o pré-processamento, salva o `df_pronto.csv` e executa o kNN padrão ($k=10$):
```bash
python knn_tradicional.py
```

### 3. Execução do kNN Diferencialmente Privado
Lê o `df_pronto.csv`, calcula as contagens dentro do raio $r=6.0$, constrói o Grafo de Sobreposição para estimar a sensibilidade ($MCS$) e aplica o ruído de Laplace para $\epsilon \in \{0.5, 1.0, 5.0, 10.0\}$:
```bash
python knn_privatizado_rnn.py
```

### 4. Geração do Gráfico Comparativo
Lê os arquivos salvos em `resultados/` e gera o gráfico comparativo `grafico_acuracias.png`:
```bash
python graficos.py
```

---


## Referências
- **Gursoy, Mehmet Emre, et al.** *"Differentially private nearest neighbor classification."* Data Mining and Knowledge Discovery 31 (2017): 1544-1575.