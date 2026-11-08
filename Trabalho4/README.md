# Trabalho 4: Privacidade Diferencial em Classificação r-N usando o Mecanismo Exponencial

**Disciplina:** Privacidade de Dados (2026)  
**Professor:** Javam Machado

---

## Alunos

**Kaio Edmar Castro Barreto - 557246**  
**Cauã de Freitas Souza - 554583**

---

## Descrição do Projeto

Este projeto tem como objetivo implementar e comparar o desempenho do algoritmo de classificação **r-N tradicional (sem privacidade)** com uma versão **diferencialmente privada epsilon-DP** que utiliza o **Mecanismo Exponencial**.

O estudo é baseado no artigo *Differentially Private Nearest Neighbor Classification*, de Gursoy et al. (2017). A implementação utiliza o conceito de vizinhança por raio fixo, no qual são considerados vizinhos os registros cuja distância Euclidiana até a instância de teste seja menor ou igual a um determinado raio \(r\).

Na versão tradicional, a classe prevista é aquela que aparece com maior frequência entre os vizinhos encontrados. Já na versão privada, a escolha da classe é realizada probabilisticamente por meio do Mecanismo Exponencial, considerando a quantidade de vizinhos pertencentes a cada classe.

O dataset utilizado é o **Adult Income Dataset**, cuja tarefa de classificação consiste em prever se a renda anual de um indivíduo é superior a 50 mil dólares, utilizando o atributo `income` como variável-alvo.

---

## Pré-processamento dos Dados

O pré-processamento é realizado antes da execução dos algoritmos, seguindo as seguintes etapas:

1. **Remoção de atributos:** exclusão das colunas `fnlwgt`, `education`, `capital-gain`, `capital-loss` e `hours-per-week`.

2. **Tratamento de valores ausentes:** remoção das linhas que apresentam o caractere `?` em qualquer atributo.

3. **Codificação categórica:** aplicação de One-Hot Encoding nos atributos `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `gender` e `native-country`. Essa técnica transforma cada categoria em uma coluna binária, representada por 0 ou 1.

4. **Codificação da variável-alvo:** transformação do atributo `income` em valores numéricos, sendo 0 para rendas menores ou iguais a 50 mil dólares e 1 para rendas superiores a 50 mil dólares.

5. **Normalização:** aplicação da normalização Min-Max nos atributos numéricos `age` e `educational-num`, utilizando os valores mínimo e máximo do conjunto de treinamento. Os mesmos parâmetros são aplicados aos dados de teste.

6. **Divisão dos dados:** separação dos registros em conjuntos de treinamento e teste. O conjunto de teste contém exatamente **1.000 registros**, sendo 500 de cada classe, garantindo uma distribuição balanceada entre as duas categorias de renda. Os registros restantes são utilizados para treinamento.

A semente aleatória utilizada na seleção dos registros é 42, permitindo reproduzir a divisão dos dados.

---

## Algoritmo r-N

O algoritmo r-N realiza a classificação considerando todos os registros do conjunto de treinamento que estejam dentro de um raio \(r\) da instância que será classificada.

A distância utilizada é a **distância Euclidiana**, calculada por:

$$
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

Em que \(x\) representa a instância de teste e \(y\) representa uma instância de treinamento.

São considerados vizinhos os registros que satisfazem:

$$
d(x,y)\leq r
$$

A classe prevista é aquela que apresenta a maior quantidade de vizinhos dentro do raio definido. Caso não sejam encontrados vizinhos, a implementação tradicional utiliza a classe majoritária do conjunto de treinamento.

Foram utilizados os seguintes valores de raio:

$$
r \in \{0.5,\ 1,\ 1.5,\ 2,\ 6\}
$$

---

## Mecanismo Exponencial

O Mecanismo Exponencial permite selecionar uma resposta de maneira probabilística, favorecendo as alternativas que apresentam maior utilidade, sem revelar diretamente os resultados das consultas.

Neste trabalho, a utilidade de cada classe corresponde à quantidade de registros daquela classe que estão dentro do raio \(r\) da instância de teste:

$$
q(x,c)=|\{y\in X_{treino}:d(x,y)\leq r \land classe(y)=c\}|
$$

Assim, quanto maior a quantidade de vizinhos de uma classe, maior será a probabilidade de ela ser selecionada.

### Sensibilidade Global

A sensibilidade global mede o quanto a função de utilidade pode variar com a inclusão ou remoção de um único registro do conjunto de dados.

Para a função de utilidade utilizada, a inclusão ou remoção de um registro pode alterar a contagem de vizinhos de uma classe em, no máximo, uma unidade.

Portanto, considerando a relação de adjacência por inclusão ou remoção de um registro:

$$
\Delta q=1
$$

### Aplicação do Mecanismo Exponencial

A probabilidade de selecionar uma classe \(c\) é calculada por:

$$
P(c)=
\frac{\exp\left(\frac{\epsilon_i q(x,c)}{2\Delta q}\right)}
{\sum_{c'\in C}\exp\left(\frac{\epsilon_i q(x,c')}{2\Delta q}\right)}
$$

Em que:

- \(q(x,c)\): quantidade de vizinhos da classe \(c\) dentro do raio;
- \(\epsilon_i\): orçamento de privacidade utilizado naquela previsão;
- \(\Delta q\): sensibilidade global da função de utilidade;
- \(C\): conjunto de classes possíveis.

Como a sensibilidade global é igual a 1, ela é substituída diretamente na fórmula.

Dessa forma, classes com mais vizinhos tendem a apresentar maior probabilidade de seleção. Entretanto, a escolha não é determinística, pois existe uma possibilidade de selecionar uma classe com menor quantidade de vizinhos.

Caso nenhuma instância seja encontrada dentro do raio, todas as classes possuem utilidade igual a zero e, consequentemente, apresentam a mesma probabilidade de seleção.

### Orçamento de Privacidade

Foram utilizados os seguintes valores de orçamento de privacidade:

$$
\epsilon \in \{0.1,\ 0.5,\ 1,\ 5,\ 10,\ 100\}
$$

Como são realizadas 1.000 previsões, o orçamento total é dividido igualmente entre elas, considerando a composição sequencial:

$$
\epsilon_i=\frac{\epsilon}{1000}
$$

Assim, cada previsão utiliza apenas uma parcela do orçamento total de privacidade.

Quanto menor o valor de \(\epsilon\), maior é a proteção oferecida e mais aleatória tende a ser a seleção da classe. Já valores maiores permitem que o mecanismo favoreça mais intensamente as classes com maior utilidade.

**Observação:** neste trabalho, o orçamento individual é calculado dividindo o orçamento total pelo número de previsões. Não é realizada uma divisão adicional pelo número de classes.


São realizadas 5 configurações de raio para o r-N tradicional e 30 combinações de raio e epsilon para o mecanismo privado.

---

## Estrutura do Projeto e Resultados

```text
.
├── rnn_tradicional.py
│   └── Pré-processamento dos dados e execução do r-N sem privacidade
│
├── rnn_exponencial.py
│   └── Execução do r-N com Mecanismo Exponencial
│
├── df_treino.csv
│   └── Dados de treinamento pré-processados
│
├── df_teste.csv
│   └── Dados de teste pré-processados e balanceados
│
└── resultados/
    ├── previsoes_tradicional_r_*.csv
    │   └── Previsões do r-N tradicional para cada raio
    │
    ├── acuracias_tradicional.csv
    │   └── Acurácias e tempos de execução do r-N tradicional
    │
    ├── resultados_tradicionais.txt
    │   └── Relatório dos resultados tradicionais
    │
    ├── previsoes_r_*_eps_*.csv
    │   └── Previsões do mecanismo exponencial para cada combinação
    │
    ├── resultados.txt
    │   └── Acurácias, orçamentos individuais e tempos de execução
    │
    └── grafico_acuracia.png
        └── Gráfico comparativo entre os métodos tradicional e privado
```

Cada arquivo de previsões contém os registros do conjunto de teste, a classe real (`income`), a classe prevista (`y_pred`) e a indicação de acerto ou erro (`Corretude`).
OBS: Sabemos que em hipótese alguma disponibilizariamos um dataset resultado com o real e o previsto e a corretude, mas para facilitar os testes e a vizualização dos senhores, fizemos assim.

---

## Como Executar o Projeto

### 1. Pré-requisitos

É necessário possuir Python instalado e as bibliotecas utilizadas no projeto:

```bash
pip install pandas numpy matplotlib
```

### 2. Execução do r-N Tradicional

Executa o pré-processamento do Adult Income Dataset, realiza a divisão balanceada dos dados, normaliza os atributos e executa o algoritmo r-N para todos os raios definidos.

```bash
python rnn_tradicional.py
```

Ao final, são gerados os arquivos `df_treino.csv` e `df_teste.csv`, além dos resultados das previsões tradicionais no diretório `resultados/`.

### 3. Execução do Mecanismo Exponencial

Lê os conjuntos de treinamento e teste gerados na etapa anterior, calcula as quantidades de vizinhos para cada raio e aplica o Mecanismo Exponencial para todos os valores de epsilon definidos.

```bash
python rnn_exponencial.py
```

São geradas 30 configurações de previsões, correspondentes às combinações entre os cinco valores de raio e os seis valores de epsilon.

### 4. Geração do Gráfico

O gráfico é gerado ao final da execução do script do mecanismo exponencial.

Ele apresenta a acurácia em função de epsilon, contendo:

- Uma linha contínua para cada raio do mecanismo exponencial;
- Uma linha tracejada para cada raio do r-N tradicional;
- Escala logarítmica no eixo horizontal, facilitando a comparação dos diferentes valores de epsilon.

O arquivo gerado é:

```text
resultados/grafico_acuracia.png
```

---

## Referências

**Gursoy, Mehmet Emre, et al.** "Differentially private nearest neighbor classification." *Data Mining and Knowledge Discovery*, 31 (2017): 1544–1575.

**Adult Income Dataset:** [UCI Machine Learning Repository - Adult](https://archive.ics.uci.edu/dataset/2/adult).  
OBS: A url utilizada é de um repositório público no github, onde só colocamos o dataset acima para facilitar o acesso. 