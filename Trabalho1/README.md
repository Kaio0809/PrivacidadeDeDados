# Generalização de Dados — COVID-19

Projeto em Python para **generalizar dados de uma base de casos de COVID-19**, reduzindo o nível de detalhe de algumas informações.

## Autores
Kaio Edmar Castro Barreto - 557246  
Cauã de Freitas Souza - 554583

#
A ideia é simples: pegar valores muito específicos e transformá-los em categorias mais gerais.

Por exemplo:

```text
Município específico
        ↓
     Região
        ↓
      Ceará
```

Assim, é possível gerar diferentes versões da mesma base e comparar como os dados mudam conforme o nível de generalização.

---

## O que o projeto faz?

O código trabalha com dois atributos:

- `municipioCaso`
- `dataNascimento`

Para cada um deles existem diferentes níveis de generalização.

### Município

```text
Nível 0 → Município
Nível 1 → Região
Nível 2 → Estado
```

### Data de nascimento

```text
Nível 0 → DD-MM-AAAA
Nível 1 → MM-AAAA
Nível 2 → AAAA
Nível 3 → Intervalo de 5 anos
```

O usuário escolhe os níveis e o programa gera uma nova base CSV.

---

## Estrutura

```text
.
├── anonimização.py
├── covid.csv
├── map_municipio_regiao.csv
└── README.md
```

Depois da execução, também serão criados os arquivos:

```text
mapeamento_generalizacao.json

DT_municipio(n0)_data(n0).csv
DT_municipio(n1)_data(n2).csv
...

HIST_Municipio_Original.png
HIST_data_Original.png
...
```

---

# Como funciona

## 1. Carregamento dos dados

A base de COVID-19 é carregada usando Pandas:

```python
df_covid = pd.read_csv(
    'covid.csv',
    sep=',',
    encoding='UTF-8'
)
```

Em seguida, são removidas as linhas que não possuem município ou data de nascimento:

```python
df_covid = df_covid.dropna(
    subset=['municipioCaso', 'dataNascimento']
)
```

---

# 2. Mapeamento dos municípios

O arquivo `map_municipio_regiao.csv` é usado para descobrir a região de cada município.

Exemplo conceitual:

```text
municipio              regiao
-----------------------------------------
Fortaleza              ...
Caucaia                ...
Sobral                 ...
```

Como os nomes podem aparecer com ou sem acento, o código possui uma função para removê-los:

```python
def remover_acento(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
```

Por exemplo:

```text
Maracanaú → Maracanau
```

Isso facilita a comparação entre os arquivos.

---

# 3. Generalização do município

O município possui três níveis.

## Nível 0

Mantém o município original.

```text
Fortaleza → Fortaleza
Caucaia → Caucaia
Sobral → Sobral
```

Ou seja, não existe generalização nesse nível.

---

## Nível 1

O município é substituído pela região encontrada no arquivo `map_municipio_regiao.csv`.

```text
Fortaleza → Grande Fortaleza
Caucaia → Grande Fortaleza
```

---

## Nível 2

Todos os municípios são generalizados para:

```text
CEARA
```

Por exemplo:

```text
Fortaleza         → CEARA
Caucaia           → CEARA
Sobral            → CEARA
Juazeiro do Norte → CEARA
```

---

# 4. Generalização da data

A coluna `dataNascimento` também possui níveis diferentes.

O código utiliza:

```python
datan1 = lambda x: x[:-3]
datan2 = lambda x: x[:-6]
```

E uma função específica para o nível 3:

```python
def datan3(dt):
    intdt = int((int(dt) / 10) + 0.5)

    if (int(dt) % 10) < 5:
        return str((intdt * 10) + 1) + ' - ' + str((intdt * 10) + 5)

    elif int(dt) % 10 == 5:
        return str(((intdt - 1) * 10) + 1) + ' - ' + str(((intdt - 1) * 10) + 5)

    else:
        return str((intdt * 10) - 4) + ' - ' + str(intdt * 10)
```

O nível 3 transforma o valor em um intervalo.

Por exemplo, considerando a parte usada pela função:

```text
2018 → 2016 - 2020
1919 → 1916 - 1920
2020 → 2016 - 2020
1821 → 1821 - 1825
2022 → 2021 - 2025
```

---

# 5. Mapeamentos

Depois de descobrir como cada valor deve ser generalizado, o programa cria dicionários.

Para município:

```python
mapeamento_municipio = {
    'nivel0': ...,
    'nivel1': ...,
    'nivel2': ...
}
```

Para data:

```python
mapeamento_data = {
    'nivel0': ...,
    'nivel1': ...,
    'nivel2': ...,
    'nivel3': ...
}
```

Esses mapeamentos são reunidos em:

```python
mapeamentos = {
    'municipio': mapeamento_municipio,
    'data': mapeamento_data
}
```

---

# 6. Arquivo JSON

Todos os mapeamentos são salvos em:

```text
mapeamento_generalizacao.json
```

Isso permite guardar as regras utilizadas pelo programa e reutilizá-las posteriormente.

A estrutura fica aproximadamente assim:

```json
{
    "municipio": {
        "nivel0": {
            "Fortaleza": "Fortaleza", ...
        },
        "nivel1": {
            "Fortaleza": "Grande Fortaleza", ...
        },
        "nivel2": {
            "Fortaleza": "CEARA", ...
        }
    },
    "data": {
        "nivel0": {},
        "nivel1": {},
        "nivel2": {},
        "nivel3": {}
    }
}
```

---

# 7. Escolhendo os níveis

Ao executar o programa:

```bash
python anonimizacao.py
```

ele pergunta:

```text
Digite o nível de abstração desejado para município (0,1,2):
Digite o nível de abstração desejado para data de nascimento (0,1,2,3):
```

Por exemplo:

```text
Município: 1
Data: 3
```

Isso significa:

```text
municipioCaso → nível 1
dataNascimento → nível 3
```

O programa então cria uma nova versão da base.

---

# 8. Exemplo

Imagine que a base tenha:

| municipioCaso | dataNascimento |
| ------------- | -------------- |
| Fortaleza     | 20-12-2001     |
| Caucaia       | 02-02-1997     |
| Sobral        | 15-07-2012     |

Se escolher:

```text
Município = 2
Data = 3
```

o município será generalizado para estado e os valores da data serão transformados conforme a regra definida em `datan3()`.:

```text
Fortaleza → CEARA
Caucaia → CEARA
Sobral → CEARA
```

```
20-12-2001 -> 2001 - 2005
02-02-1997 -> 1996 - 2000
15-07-2012 -> 2011 - 2015
```

O resultado será salvo em um arquivo como:

```text
DT_municipio(n2)_data(n3).csv
```

---

# 9. Histogramas

O programa também gera histogramas para visualizar a quantidade de categorias antes e depois da generalização.

Por exemplo:

```text
HIST_Municipio_Original.png
HIST_municipioCaso_1.png
HIST_data_Original.png
HIST_dataNascimento_3.png
```

A ideia é poder visualizar algo como:

```text
Antes:

Município A  █████████
Município B  █████
Município C  ███
Município D  ███████


Depois:

Região A     █████████████████
Região B     ███████████
```

Quanto maior a generalização, menos categorias tendem a existir.

---

# 10. Quantidade de categorias

O código também calcula quantas categorias existem em cada nível.

Para município:

```python
binss_municipio = {
    'nivel0': len(regioes.keys()),
    'nivel1': len(set(regioes.values())),
    'nivel2': 1
}
```

A lógica é:

```text
Nível 0 → vários municípios
Nível 1 → regiões
Nível 2 → 1 estado
```

Para a data:

```python
binss_data = {
    'nivel0': len(set(datas)),
    'nivel1': len(set(mapeamento_data['nivel1'].values())),
    'nivel2': len(set(mapeamento_data['nivel2'].values())),
    'nivel3': len(set(mapeamento_data['nivel3'].values()))
}
```

Isso permite observar diretamente a redução da quantidade de valores distintos.

---

# 11. Arquivos gerados

Dependendo dos níveis escolhidos, o programa pode gerar arquivos como:

```text
DT_municipio(n0)_data(n0).csv
DT_municipio(n0)_data(n1).csv
DT_municipio(n1)_data(n2).csv
DT_municipio(n2)_data(n3).csv
```

Além dos histogramas correspondentes.

O arquivo:

```text
mapeamento_generalizacao.json
```

também é criado automaticamente.

---

# 12. Tecnologias

- Python
- Pandas
- Matplotlib
- JSON
- Unicode (`unicodedata`)

Instalação:

```bash
pip install pandas matplotlib
```

---

# 13. Ideia do projeto

A ideia principal é testar diferentes níveis de generalização e observar o que acontece com os dados.

Em resumo:

```text
       DADOS ORIGINAIS
              |
              v
     +------------------+
     |   Generalização  |
     +------------------+
              |
       +------+------+
       |             |
       v             v
   Município        Data
       |             |
   N0 → N1 → N2   N0 → N1 → N2 → N3
       |             |
       +------+------+
              |
              v
       NOVA BASE CSV
              |
              v
        HISTOGRAMAS
```

A graça do projeto está justamente em poder experimentar:

```text
Município 0 + Data 0
Município 0 + Data 1
Município 1 + Data 2
Município 2 + Data 3
...
```

e comparar como a base muda conforme os dados ficam mais generalizados.

------

