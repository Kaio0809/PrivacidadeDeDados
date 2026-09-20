# Trabalho 2 - k-Anonimato

Disciplina: Privacidade de Dados (2026-2)

Implementação do modelo k-anonimato, conforme Sweeney (2002), por meio da generalização de valores de atributos semi-identificadores sobre o conjunto de dados `covid.csv`. O programa recebe o valor de k (2, 4, 8 ou 16), gera o dataset anonimizado correspondente e calcula a precisão, o tamanho médio das classes de equivalência e o histograma dos tamanhos das classes.

## Atributos considerados

| Atributo | Papel |
|---|---|
| `municipioCaso` | semi-identificador |
| `dataNascimento` (formato `aaaa-mm-dd`) | semi-identificador |
| `racaCor` | atributo sensível (mantido sem alteração) |

O k-anonimato é garantido sobre o par (`municipioCaso`, `dataNascimento`). O atributo `racaCor` não faz parte do semi-identificador e, portanto, não é generalizado.

## Requisitos

- Python 3.9 ou superior
- pandas
- numpy
- matplotlib

Instalação das dependências:

```
pip install pandas numpy matplotlib
```

## Estrutura de arquivos esperada

```
.
├── k_anonimato.py
├── mapeamento_generalizacao.json
├── README.md
└── Datasets/
    └── covid.csv
└── Histogramas/
    └── histograma_k.csv    
```

Após a execução são gerados:

- `Datasets/covid_{k}.csv`: dataset anonimizado para o k informado.
- `Datasets/histograma_k_{k}.png`: histograma dos tamanhos das classes de equivalência.

## Como executar

```
python k_anonimato.py
```

O programa solicita o valor de k pelo terminal:

```
Digite o k = {2,4,8,16}:
```

Para gerar os resultados de todos os valores de k, execute o programa uma vez para cada valor.

## Formato do arquivo de hierarquias

O arquivo `mapeamento_generalizacao.json` (construído no Trabalho 1) contém as hierarquias de generalização de domínio de cada semi-identificador. Cada chave `nivelN` mapeia o valor original do atributo para o seu valor generalizado naquele nível:

```
{
  "municipio": { "nivel0": {...}, "nivel1": {...}, ... },
  "data":      { "nivel0": {...}, "nivel1": {...}, ... }
}
```

O nível 0 é o domínio original (sem generalização) e o nível de maior índice é o topo da hierarquia. O nível `nivel1` de `municipio` associa cada município à sua região e é usado para ordenar os registros por proximidade geográfica.

## Pré-processamento

1. Leitura apenas das colunas de interesse (`municipioCaso`, `dataNascimento`, `racaCor`).
2. Remoção de registros com valores nulos, vazios ou compostos apenas por espaços.
3. Remoção de registros com `racaCor` igual a `Sem Informacao`.
4. Descarte das demais colunas.

Se algum município ou data do CSV não existir no arquivo de hierarquias, o programa imprime um aviso indicando o atributo, o nível e a quantidade de registros afetados. Esses valores são tratados como `*` (suprimidos).

## Algoritmo

### Reticulado de generalização

O reticulado é formado pelo produto das hierarquias de município e de data. Cada nó é um par (nm, nd) com custo nm + nd, em que nm e nd são os níveis de generalização de cada atributo. Nós de menor custo representam menor generalização.

### Construção das classes de equivalência

Para evitar generalizar um atributo em todos os registros de uma vez, o algoritmo generaliza apenas grupos de registros próximos, com o reticulado servindo de guia:

1. Os registros são ordenados por região (`nivel1` de município), município e data de nascimento. Registros geograficamente e temporalmente próximos ficam adjacentes.
2. A sequência ordenada é dividida em grupos de k registros consecutivos. Se sobrarem menos de k registros ao final, eles são incorporados ao último grupo. Se o dataset tiver menos de k registros, forma-se um único grupo.
3. Para cada grupo, escolhe-se o nó de menor custo do reticulado que torna todos os seus registros equivalentes, isto é, com o mesmo município generalizado e a mesma data generalizada. Grupos cujos registros já são idênticos permanecem no nível (0, 0).
4. Os valores do grupo são substituídos pelos valores do nível escolhido.

A escolha do nó de menor custo é feita de forma vetorizada. Como a uniformidade do município depende apenas de nm e a da data apenas de nd, o primeiro nó válido percorrendo o reticulado em ordem crescente de custo é o par formado pelo menor nm válido e pelo menor nd válido. O resultado é idêntico ao de uma busca sequencial sobre o reticulado, sem percorrer os registros linha a linha.

### Garantia final de k-anonimato

Se nenhum nível do reticulado uniformizar um grupo (por exemplo, por lacunas no arquivo de hierarquias), o valor do atributo é suprimido (`*`). Ao final, uma etapa de garantia verifica todas as classes de equivalência. Qualquer classe com menos de k registros é suprimida em ambos os atributos, e, se a classe suprimida ainda tiver menos de k registros, são incorporados registros das menores classes restantes. O processo se repete até que todas as classes tenham ao menos k registros.

## Métricas

### Precisão

```
Prec(D) = 1 - ( sum_i sum_j  h / |HG_Ai| ) / ( |D| * Na )
```

Em que:

- Na = 2 é o número de atributos semi-identificadores.
- |D| é o número de registros do dataset.
- h é a altura da hierarquia do atributo após a generalização do registro (o nível nm ou nd aplicado a ele).
- |HG_Ai| é a altura máxima da hierarquia do atributo Ai (índice do último nível).

A precisão vale 1 quando nenhum valor é generalizado e 0 quando todos os valores estão no topo da hierarquia.

### Tamanho médio das classes de equivalência

```
tamanho médio = número total de registros / número de classes de equivalência
```

### Histograma

O histograma mostra a distribuição dos tamanhos das classes de equivalência do dataset gerado para cada k, salvo em `histograma_k_{k}.png`.

## Saída do programa

Para o k informado, o programa imprime:

- O nome do dataset gerado (`covid_{k}.csv`).
- Os pares de níveis (nm, nd) utilizados e a quantidade de registros em cada par.
- A precisão (Prec).
- O resultado da verificação de k-anonimato e o tamanho da menor classe.
- O tamanho médio das classes de equivalência e o total de classes.

A verificação de k-anonimato é feita relendo o arquivo `covid_{k}.csv` gravado em disco e agrupando por (`municipioCaso`, `dataNascimento`).

## Complexidade

O custo é dominado pela ordenação dos registros, O(n log n), mais O(n * L) para aplicar os L níveis das hierarquias. A escolha do nível de cada grupo e a aplicação dos valores generalizados são feitas com operações vetorizadas (numpy e pandas), sem laços em Python sobre os registros.

## Limitações

- A formação de grupos de k registros consecutivos é uma heurística. Ela não garante a distorção mínima global (como faria o algoritmo MinGen de Sweeney), mas evita a generalização de um atributo inteiro e tem custo O(n log n).
- A qualidade da precisão depende das hierarquias do Trabalho 1 e da ordenação geográfica derivada de `nivel1` de município.
- O k-anonimato protege contra ataques de ligação ao registro sobre os semi-identificadores. Não oferece, por si só, proteção contra divulgação do atributo sensível (`racaCor`) quando uma classe de equivalência possui pouca diversidade nesse atributo.

## Referência

L. Sweeney. Achieving k-anonymity privacy protection using generalization and suppression. International Journal on Uncertainty, Fuzziness and Knowledge-based Systems, 10 (5), 2002; 571-588.