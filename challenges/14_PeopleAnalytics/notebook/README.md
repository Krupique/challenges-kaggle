<h1>PEOPLE ANALYTICS - PREVENDO SE O COLABORADOR VAI DEIXAR A EMPRESA</h1>

---

<h2>SUMÁRIO</h2>

* **1) INTRODUÇÃO**
	* 1.1) O que é People Analytics 
	* 1.2) Objetivo do projeto 
	* 1.3) Sobre o Dataset 
<br/><br/>

* **2) ANÁLISE EXPLORATÓRIA DOS DADOS** 
	* 2.1) Visão Geral dos Dados 
<br/><br/>

* **3) ANÁLISE UNIVARIADA**
	* 3.1) Deixou a Empresa (deixou_empresa) 
	* 3.2) Nível de Satisfação (nivel_satisfacao) 
	* 3.3) Última Avaliação (ultima_avaliacao) 
	* 3.4) Quantidade de Projetos (numero_projetos) 
	* 3.5) Média de horas trabalhadas por mês (horas_medias_por_mes) 
	* 3.6) Tempo como funcionário da empresa (tempo_empresa) 
	* 3.7) Acidente de Trabalho (acidente_trabalho) 
	* 3.8) Promoção do funcionário nos últimos 5 anos (ultima_promocao_5anos) 
	* 3.9) Área de atuação (area) 
	* 3.10) Salário categórico (salario) 
<br/><br/>

* **4) ANÁLISE MULTIVARIADA** 
	* 4.1) Deixou a empresa com as demais variáveis 
	* 4.2) Nível de satisfação com as demais variáveis 
<br/><br/>

* **5) PRÉ PROCESSAMENTO DOS DADOS** 
	* 5.1) Importação das bibliotecas 
	* 5.2) Tratamento de outliers 
	* 5.3) Primeira seleção de atributos e one hot encoder 
	* 5.4) Modelagem base para primeira avaliação 
		* 5.4.1) KNN 
		* 5.4.2) Regressão Logística 
		* 5.4.3) Random Forest 
		* 5.4.4) Seleção das features mais importantes 
<br/><br/>

* **6) MODELAGEM** 
	* 6.1) Avaliando os resultados
	<br/><br/>
* **7) EXPORTANDO O MODELO** 
* **8) CONSIDERAÇÕES SOBRE OS RESULTADOS OBTIDOS** 

---
<h2>1) INTRODUÇÃO</h2>

<h3>1.1) O que é People Analytics</h3>

People analytics é um processo de coleta e análise de dados voltado para a gestão de pessoas em empresas. O conceito nasce a partir da ideia de big data, que consiste na coleta, armazenamento e análise de um volume imenso de dados. Com tantas informações disponíveis ou passíveis de serem coletadas, as empresas têm identificado oportunidades de aproveitá-las para melhorar seus processos.

É bom deixar claro desde o começo que o People Analytics não é uma ferramenta ou um software. Trata-se de uma metodologia cujo princípio é a coleta, a organização e a análise de dados aplicada à gestão de pessoas para se ter uma visão mais estratégica do papel de cada colaborador dentro de uma empresa. Seu objetivo é melhorar a qualidade da tomada de decisão sobre os profissionais a partir da coleta e do cruzamento de informações relacionadas a eles. Assim, é possível tanto reconhecer um funcionário que se destaca quanto identificar problemas que estejam ocorrendo, como baixa produtividade, pouco engajamento, insatisfação, alto índice de rotatividade, entre outros.

<h3>1.2) Objetivo do projeto</h3>
Este projeto tem por objetivo aplicar técnicas de análise descritiva, estatística e preditiva dos dados para entender o comportamento dos colaboradores e os principais fatores que levam o mesmo a deixar ou não a empresa. No final, espera-se um modelo preditivo capaz de prever se o colaborador vai deixar ou não a empresa e um relatório completo sobre os resultados da análise.

<h3>1.3) Sobre o Dataset</h3>

O dataset se trata de um conjunto de dados fictícios de uma empresa fictícia. É um dataset próprio para estudo deste tema e contempla os seguintes atributos:
* **nivel_satisfacao**: O nível de satisfação é uma nota que representa o quão satisfeito o colaborador está trabalhando na empresa.
* **ultima_avaliacao**: A última avaliação representa a nota atribuída pelo usuário na última pesquisa de avaliação feita pela empresa.
* **numero_projetos**: Quantidade de projetos que o colaborador já atuou.
* **horas_medias_por_mes**: Média de horas trabalhadas por mês.
* **tempo_empresa**: Quantidade em anos que o colaborador está na empresa.
* **acidente_trabalho**: Indica se o colaborador já sofreu acidente de trabalho.
* **deixou_empresa**: Variável target e representa se o colaborador saiu da empresa.
* **ultima_promocao_5anos**: Atributo categórico que mostra se o colaborador teve promoção nos últimos 5 anos.
* **area**: Área de atuação dentro da empresa.
* **salario**: Faixa salarial categórica. Como o salário é uma informação sensível, o atributo está apenas dividio em salário: `baixo`, `médio` e `alto`. 


---
<h2>2) ANÁLISE EXPLORATÓRIA DOS DADOS</h2>


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import warnings
import re

warnings.filterwarnings("ignore")
```


```python
df = pd.read_csv('data/dados_func.csv')
```


```python
df.head()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
      <th>area</th>
      <th>salario</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>



<h3>2.1) Visão Geral dos Dados</h3>

Primeiro irei realizar um breve estudo sobre como os dados estão concentrados. Irei verificar coisas como: Shape dos dados, isto é, quantidade de registros e atributos, irei verificar o nome das colunas, se há dados nulos, dentre outros passos.


```python
df_eda = df.copy()
```


```python
print(f'O dataset possui {df_eda.shape[0]} registros e {df_eda.shape[1]} colunas')
```

    O dataset possui 14999 registros e 10 colunas
    


```python
#Nome das colunas
df_eda.columns
```




    Index(['nivel_satisfacao', 'ultima_avaliacao', 'numero_projetos',
           'horas_medias_por_mes', 'tempo_empresa', 'acidente_trabalho',
           'deixou_empresa', 'ultima_promocao_5anos', 'area', 'salario'],
          dtype='object')




```python
#Verificação da ocorrência de dados missing
df_eda.isnull().sum()
```




    nivel_satisfacao         0
    ultima_avaliacao         0
    numero_projetos          0
    horas_medias_por_mes     0
    tempo_empresa            0
    acidente_trabalho        0
    deixou_empresa           0
    ultima_promocao_5anos    0
    area                     0
    salario                  0
    dtype: int64




```python
#Breve descrição estatística dos atributos. 
df_eda.describe()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.612834</td>
      <td>0.716102</td>
      <td>3.803054</td>
      <td>201.050337</td>
      <td>3.498233</td>
      <td>0.144610</td>
      <td>0.238083</td>
      <td>0.021268</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.248631</td>
      <td>0.171169</td>
      <td>1.232592</td>
      <td>49.943099</td>
      <td>1.460136</td>
      <td>0.351719</td>
      <td>0.425924</td>
      <td>0.144281</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.090000</td>
      <td>0.360000</td>
      <td>2.000000</td>
      <td>96.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.440000</td>
      <td>0.560000</td>
      <td>3.000000</td>
      <td>156.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.640000</td>
      <td>0.720000</td>
      <td>4.000000</td>
      <td>200.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.820000</td>
      <td>0.870000</td>
      <td>5.000000</td>
      <td>245.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>310.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.skew()
```




    nivel_satisfacao        -0.476360
    ultima_avaliacao        -0.026622
    numero_projetos          0.337706
    horas_medias_por_mes     0.052842
    tempo_empresa            1.853319
    acidente_trabalho        2.021149
    deixou_empresa           1.230043
    ultima_promocao_5anos    6.636968
    dtype: float64



**O que temos até agora?**<br/>
O dataset possui aproximadamente 15000 mil registros. É uma boa quantidade de dados para a construção de um modelo preditivo. O dataset está bastante limpo, portanto, iremos economizar bastante tempo na etapa de tratamento de valores missing.

Observando a descrição estatística, podemos tirar algumas conclusões:
* Na média, a empresa tem um nível de satisfação 6. É um nível razoável para bom;
* A última avaliação dos colaboradores teve média 7. Isso indica que a empresa teve uma melhora em relação ao nível de satisfação;
* O colaborador trabalha em média em 4 projetos.
* O colaborador fica em média por 4 anos na empresa.
* 14% dos colaboradores já sofreram acidente de trabalho.
* Os colaboradores trabalham em média 200 horas por mês.

Agora observando o skewness dos dados temos:
* Os atributos ultima_avaliacao e horas_medias_por_mes estão bastante próximos de zero, indicando que estas colunas muito provavelmente estão respeitando uma distribuição normal;
* Porém, a maioria dos atributos distoam bastante de zero, o que isso quer dizer? Que estes atributos muito provalvelmente não estão normalmente distribuídos;

> NOTA: A maioria das inferências estatísticas e também dos algoritmos de Machine Learning esperam que os dados estejam normalmente distribuídos, portanto, se aplicarmos estas técnicas com o conjunto de dados do jeito que está, com certeza teremos problemas. O que fazer?
> 1. Tratar outliers;
> 2. Realizar transformações, seja logarítmica, normalização e/ou escala dos dados;


---

<h2>3) ANÁLISE UNIVARIADA</h2>

Agora eu irei fazer uma análise mais detalhada para cada atributo, afim de compreender o que cada atributo representa individualmente.

<h3>3.1) Deixou a Empresa (deixou_empresa)</h3>


```python
#Cópia do dataset para não perder o dataset original
df_ind = df.copy()
```


```python
df_ind['deixou_empresa'].value_counts()
```




    0    11428
    1     3571
    Name: deixou_empresa, dtype: int64




```python
#Proporção de colaboradores que deixaram a empresa
proporcao = pd.DataFrame(round(df_ind.groupby(['deixou_empresa'])['deixou_empresa'].count() / len(df_ind) * 100, 2))
proporcao
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deixou_empresa</th>
    </tr>
    <tr>
      <th>deixou_empresa</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>76.19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.81</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.pie(proporcao, values='deixou_empresa', names=['Não', 'Sim'], title='Proporção de colaboradores que deixaram a empresa')

fig.update_traces(marker=dict(colors=['#424456', '#e85151']), textinfo='percent', textfont_size=16)

fig.update_layout(
    autosize=False,
    width=600,
    height=400
)

fig.show()
```


    
![png](assets/output_20_0.png)
    


Vemos que aproximadamente 1 a cada 4 funcionários saem da empresa. Isso é um valor bastante elevado e no que isso impacta para o crescimento da empresa
> 1. A empresa gasta muitos recursos financeiros e humanos em admissões e demissões;
> 2. Gasta tempo que poderia ser melhor utilizado aplicando processos seletivos;
> 3. Gasta muito recursos com treinamento;
> 4. Frea o crescimento, pois frequentemente a empresa precisa contratar novos colaboradores para as vagas que foram desocupadas, fazendo com que a empresa precise gastar recursos nos tópicos acima.

<h3>3.2) Nível de Satisfação (nivel_satisfacao)</h3>


```python
#Cabeçalho
df_ind.head()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
      <th>area</th>
      <th>salario</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Descrição dos dados
df_ind['nivel_satisfacao'].describe()
```




    count    14999.000000
    mean         0.612834
    std          0.248631
    min          0.090000
    25%          0.440000
    50%          0.640000
    75%          0.820000
    max          1.000000
    Name: nivel_satisfacao, dtype: float64




```python
#Histograma com o nível de satisfação dos colaboradores
plt.figure(figsize=(10, 6))
sns.histplot(df_ind['nivel_satisfacao'] , kde=True, bins=20)
```




    <AxesSubplot:xlabel='nivel_satisfacao', ylabel='Count'>




    
![png](assets/output_25_1.png)
    



```python
#função para automatizar o processo de exibição do gráfico de barras
def exibeBarplot(df, coluna, xlabel = 'x', ylabel = 'y', title = 'default', width=800, height=400):
    values = pd.DataFrame(df[coluna].value_counts().sort_index())
    values['percent'] = values[coluna].apply(lambda x: round(x / np.sum(values[coluna].values) * 100, 2))
    values['text1'] = values[coluna].astype(str)
    values['text2'] = values['percent'].astype(str)

    values['text'] = values['text1'] + ' [' + values['text2'] + '%]' 



    fig = px.bar(x=values.index, y=values[coluna], text=values['text'])

    fig.update_traces(marker_color='#424456')

    fig.update_layout(
        title=f'<span>{title}</span>', 
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(title=f'<span>{xlabel}</span>'),
        yaxis=dict(title=f'<span>{ylabel}</span>')
    )

    fig.show()
```


```python
df_ind['nivel_satisfacao_bins'] = pd.cut(df_ind['nivel_satisfacao'], [-0.1, 0.2, 0.4, 0.6, 0.8, 1], 
                                        labels=['péssimo [0-20]', 'baixo [20-40]', 'médio [40-60]', 'bom [60-80]', 'ótimo [80-100]'])
```

Para melhorar a nossa compreensão sobre os dados, eu decidi separar as avaliações em grupos, dessa forma teremos uma ideia mais clara de como os dados estão dispostos:
* **Péssimo**: Avaliações entre: 0 e 0.2;
* **Baixo**: Entre 0.2 e 0.4;
* **Médio**: Entre 0.4 e 0.6;
* **Bom**: Entre 0.6 e 0.8;
* **Ótimo**: Entre 0.8 e 1.


```python
exibeBarplot(df_ind, 'nivel_satisfacao_bins', title='Quantidade de nível de satisfação dos colaboradores', xlabel='Satisfação', ylabel='Quantidade')
```


    
![png](assets/output_29_0.png)
    


**Considerações:**
* A média geral de satisfação fica em torno de 6, isso é um valor considerado razoável para uma empresa. Porém, para uma empresa que almeja ser o `great place to work` esse valor é um tanto quanto preocupante;
* Pelo histograma podemos ver que aproximadamente 1000 colaboradores não estão nada satisfeitos trabalhando na empresa;
* Vendo o gráfico por categorias temos que mais de 50% dos colaboradores estão com o nível de satisfação bom ou ótimo;

<h3>3.3) Última Avaliação (ultima_avaliacao)</h3>


```python
df['ultima_avaliacao'].describe()
```




    count    14999.000000
    mean         0.716102
    std          0.171169
    min          0.360000
    25%          0.560000
    50%          0.720000
    75%          0.870000
    max          1.000000
    Name: ultima_avaliacao, dtype: float64




```python
plt.figure(figsize=(10, 6))
sns.histplot(df_ind['ultima_avaliacao'], kde=True, bins=20).set(title='Histograma da última avaliação')
```




    [Text(0.5, 1.0, 'Histograma da última avaliação')]




    
![png](assets/output_33_1.png)
    



```python
df_ind['ultima_avaliacao_bins'] = pd.cut(df_ind['ultima_avaliacao'], [0, 0.2, 0.4, 0.6, 0.8, 1], 
                                        labels=['péssimo [0-20]', 'baixo [20-40]', 'médio [40-60]', 'bom [60-80]', 'ótimo [80-100]'])
```


```python
exibeBarplot(df_ind, 'ultima_avaliacao_bins', xlabel='Satisfação', ylabel='Quantidade', title='Quantidade de satisfação na última avaliação')
```


    
![png](assets/output_35_0.png)
    


**Considerações:**
* A média da última avaliação teve uma melhora em relação ao nível de satisfação, foi de 0.7;
* Outro ponto muito importante foi que a empresa não teve avaliações consideradas péssima e a quantidade de avaliações consideradas baixo foi de apenas 1.5%.
* A maioria das avaliações foi classificadas como ótima;

Isso mostra que a empresa vem melhorando o seu nível no que diz respeito em ser um local que os colaborades gostam de trabalhar. 

<h3>3.4) Quantidade de Projetos (numero_projetos)</h3>


```python
df_ind['numero_projetos'].describe()
```




    count    14999.000000
    mean         3.803054
    std          1.232592
    min          2.000000
    25%          3.000000
    50%          4.000000
    75%          5.000000
    max          7.000000
    Name: numero_projetos, dtype: float64




```python
df_ind['numero_projetos'].value_counts()
```




    4    4365
    3    4055
    5    2761
    2    2388
    6    1174
    7     256
    Name: numero_projetos, dtype: int64




```python
exibeBarplot(df_ind, coluna='numero_projetos', xlabel='Número de projetos' , ylabel='Quantidade', title='Quantidade por números de projeto', width=1000, height=600)
```


    
![png](assets/output_40_0.png)
    


**Considerações**:
* A maioria dos colaboradores atuam/atuaram em 3 ou 4 projetos. Algums poucos atuaram em 7 projetos.
* Esse é um atributo que sozinho não diz muita coisa, pois não temos como mensurar o impacto que atuar em poucos ou vários projetos têm sobre a quantidade de colaboradores que deixaram a empresa.

<h3>3.5) Média de horas trabalhadas por mês (horas_medias_por_mes)</h3>


```python
df_ind['horas_medias_por_mes'].describe()
```




    count    14999.000000
    mean       201.050337
    std         49.943099
    min         96.000000
    25%        156.000000
    50%        200.000000
    75%        245.000000
    max        310.000000
    Name: horas_medias_por_mes, dtype: float64




```python
week_mean = round(df_ind['horas_medias_por_mes'].describe()['mean'] / 4.5)
print(f'A média de horas trabalhadas por semana é de {week_mean} horas')
```

    A média de horas trabalhadas por semana é de 45 horas
    


```python
df_ind.sort_values(by=['horas_medias_por_mes'], ascending=False).head(20)
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
      <th>area</th>
      <th>salario</th>
      <th>nivel_satisfacao_bins</th>
      <th>ultima_avaliacao_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>803</th>
      <td>0.10</td>
      <td>0.77</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>809</th>
      <td>0.11</td>
      <td>0.97</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>1717</th>
      <td>0.11</td>
      <td>0.96</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>14975</th>
      <td>0.10</td>
      <td>0.79</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>0.11</td>
      <td>0.97</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>0.10</td>
      <td>0.79</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>0.09</td>
      <td>0.89</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>14972</th>
      <td>0.11</td>
      <td>0.97</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>566</th>
      <td>0.11</td>
      <td>0.79</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>low</td>
      <td>péssimo [0-20]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>0.11</td>
      <td>0.84</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>1578</th>
      <td>0.10</td>
      <td>0.96</td>
      <td>6</td>
      <td>310</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>1596</th>
      <td>0.36</td>
      <td>0.48</td>
      <td>5</td>
      <td>310</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>product_mng</td>
      <td>medium</td>
      <td>baixo [20-40]</td>
      <td>médio [40-60]</td>
    </tr>
    <tr>
      <th>14777</th>
      <td>0.11</td>
      <td>0.79</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>low</td>
      <td>péssimo [0-20]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>1545</th>
      <td>0.10</td>
      <td>0.78</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>1482</th>
      <td>0.09</td>
      <td>0.77</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>low</td>
      <td>péssimo [0-20]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>0.70</td>
      <td>0.93</td>
      <td>2</td>
      <td>310</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>product_mng</td>
      <td>low</td>
      <td>bom [60-80]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>12566</th>
      <td>0.11</td>
      <td>0.79</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>low</td>
      <td>péssimo [0-20]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>1228</th>
      <td>0.10</td>
      <td>0.85</td>
      <td>7</td>
      <td>310</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>marketing</td>
      <td>low</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0.10</td>
      <td>0.94</td>
      <td>6</td>
      <td>309</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>897</th>
      <td>0.30</td>
      <td>0.56</td>
      <td>3</td>
      <td>309</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>baixo [20-40]</td>
      <td>médio [40-60]</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ind.sort_values(by=['horas_medias_por_mes'], ascending=True).head(5)
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
      <th>area</th>
      <th>salario</th>
      <th>nivel_satisfacao_bins</th>
      <th>ultima_avaliacao_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10020</th>
      <td>0.50</td>
      <td>0.58</td>
      <td>4</td>
      <td>96</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>médio [40-60]</td>
      <td>médio [40-60]</td>
    </tr>
    <tr>
      <th>2678</th>
      <td>0.79</td>
      <td>0.61</td>
      <td>5</td>
      <td>96</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>marketing</td>
      <td>medium</td>
      <td>bom [60-80]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>8605</th>
      <td>0.48</td>
      <td>0.43</td>
      <td>3</td>
      <td>96</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>médio [40-60]</td>
      <td>médio [40-60]</td>
    </tr>
    <tr>
      <th>7492</th>
      <td>0.15</td>
      <td>0.98</td>
      <td>2</td>
      <td>96</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>high</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>9221</th>
      <td>0.92</td>
      <td>0.82</td>
      <td>4</td>
      <td>96</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>IT</td>
      <td>medium</td>
      <td>ótimo [80-100]</td>
      <td>ótimo [80-100]</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.box(y=df['horas_medias_por_mes'])

fig.update_layout(
        title=f'<span>Boxplot de horas médias trabalhadas por mês</span>', 
        autosize=False,
        width=500,
        height=400,
        xaxis=dict(title=f'<span></span>'),
        yaxis=dict(title=f'<span>Horas</span>')
    )

fig.show()
```


    
![png](assets/output_47_0.png)
    


**Considerações**:
* Considerando que o mês tenha mais ou menos 4.5 semanas, temos que a média de horas trabalhadas por semana é de 45 horas;
* Há vários colaboradores que trabalham 310 horas por mês, isso dá algo em por volta de 12 horas diárias durante 6 dias da semana;
    * Ao analisar o boxplot, percebemos que esses valores não indicam outliers e que há vários colabodores que tem essa carga horária mensal;

<h3>3.6) Tempo como funcionário da empresa (tempo_empresa)</h3>


```python
df_ind['tempo_empresa'].describe()
```




    count    14999.000000
    mean         3.498233
    std          1.460136
    min          2.000000
    25%          3.000000
    50%          3.000000
    75%          4.000000
    max         10.000000
    Name: tempo_empresa, dtype: float64




```python
display(df_ind.sort_values(by=['tempo_empresa'], ascending=True).head(5))
display(df_ind.sort_values(by=['tempo_empresa'], ascending=False).head(5))
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
      <th>area</th>
      <th>salario</th>
      <th>nivel_satisfacao_bins</th>
      <th>ultima_avaliacao_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7499</th>
      <td>0.93</td>
      <td>0.58</td>
      <td>5</td>
      <td>238</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>ótimo [80-100]</td>
      <td>médio [40-60]</td>
    </tr>
    <tr>
      <th>3966</th>
      <td>0.82</td>
      <td>0.58</td>
      <td>4</td>
      <td>202</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>RandD</td>
      <td>medium</td>
      <td>ótimo [80-100]</td>
      <td>médio [40-60]</td>
    </tr>
    <tr>
      <th>9324</th>
      <td>0.57</td>
      <td>0.66</td>
      <td>4</td>
      <td>158</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>médio [40-60]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>9325</th>
      <td>0.55</td>
      <td>0.54</td>
      <td>5</td>
      <td>168</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>médio [40-60]</td>
      <td>médio [40-60]</td>
    </tr>
    <tr>
      <th>3960</th>
      <td>0.50</td>
      <td>0.55</td>
      <td>3</td>
      <td>188</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>product_mng</td>
      <td>medium</td>
      <td>médio [40-60]</td>
      <td>médio [40-60]</td>
    </tr>
  </tbody>
</table>
</div>



<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
      <th>area</th>
      <th>salario</th>
      <th>nivel_satisfacao_bins</th>
      <th>ultima_avaliacao_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11630</th>
      <td>0.68</td>
      <td>0.67</td>
      <td>4</td>
      <td>162</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>management</td>
      <td>high</td>
      <td>bom [60-80]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>11230</th>
      <td>0.99</td>
      <td>0.84</td>
      <td>4</td>
      <td>142</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>technical</td>
      <td>high</td>
      <td>ótimo [80-100]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>13742</th>
      <td>0.67</td>
      <td>0.72</td>
      <td>2</td>
      <td>134</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>management</td>
      <td>high</td>
      <td>bom [60-80]</td>
      <td>bom [60-80]</td>
    </tr>
    <tr>
      <th>11232</th>
      <td>0.65</td>
      <td>0.85</td>
      <td>4</td>
      <td>201</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>bom [60-80]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>13741</th>
      <td>0.60</td>
      <td>0.65</td>
      <td>2</td>
      <td>225</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>management</td>
      <td>high</td>
      <td>médio [40-60]</td>
      <td>bom [60-80]</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_ind['tempo_empresa'].value_counts()
```




    3     6443
    2     3244
    4     2557
    5     1473
    6      718
    10     214
    7      188
    8      162
    Name: tempo_empresa, dtype: int64




```python
exibeBarplot(df_ind, coluna='tempo_empresa', xlabel='Tempo médio como funcionário da empresa', ylabel='Quantidade', 
                width=1300, height=600, title='Quantidade de funcionários por tempo médio em anos na empresa')
```


    
![png](assets/output_53_0.png)
    


**Considerações**:
* A maioria dos funcionários ficam cerca de 3 anos na empresa;
* A grande maioria fica no máximo por 4 anos;
* Poucos funcionárias ficaram mais do que 6 anos; 

<h3>3.7) Acidente de Trabalho (acidente_trabalho)</h3>


```python
df_ind['acidente_trabalho'].value_counts()
```




    0    12830
    1     2169
    Name: acidente_trabalho, dtype: int64




```python
exibeBarplot(df_ind, coluna='acidente_trabalho', xlabel='Acidentes de trabalho', ylabel='Quantidade', 
                title='Quantidade de pessoas que sofreram acidades de trabalho', width=600)
```


    
![png](assets/output_57_0.png)
    


**Considerações**: 
* 14% dos colaboradores já sofreram acidente de trabalho. É como estar no meio de uma guerra, você não sabe se vai voltar vivo para casa 😂. Eu fiz uma brincadeira, mas esses dados são alarmantes, mas para frente veremos a relação que isso tem com as demais variáveis.

<h3>3.8) Promoção do funcionário nos últimos 5 anos (ultima_promocao_5anos)</h3>


```python
df_ind['ultima_promocao_5anos'].value_counts()
```




    0    14680
    1      319
    Name: ultima_promocao_5anos, dtype: int64




```python
exibeBarplot(df_ind, coluna='ultima_promocao_5anos', xlabel='Promoção [0 = Não|1 = Sim]', width=600, height=400,
                title='Quantidade de promoções por quantidade de funcionários nos últimos 5 anos')
```


    
![png](assets/output_61_0.png)
    


**Considerações**:
* De 15000 mil colaboradores, apenas 300 tiveream promoções nos últimos 5 anos. Talvez isso explique porque tantos funcionários têm o tempo de empresa tão baixo e porque tantas demissões;
* Se a pessoa não vê perspectiva de crescimento, é muito difícil que ela continue estagnada no mesmo lugar;

<h3>3.9) Área de atuação (area)</h3>


```python
exibeBarplot(df_ind, coluna='area', xlabel='Áreas de atuação', ylabel='Quantidade', 
            title='Quantidade de pessoas por cargo', width=1300, height=500)
```


    
![png](assets/output_64_0.png)
    


**Considerações**:
* A maioria dos colaboradores trabalham nas áreas de: `vendas`, `suporte` e `técnica`. É normal que empresa tenha mais colaboradores em áreas consideradas operacionais;
* As demais áreas possuem menos colaborades, pois se tratam de áreas mais estratégicas para a empresa;

<h3>3.10) Salário categórico (salario)</h3>


```python
exibeBarplot(df_ind, coluna='salario', xlabel='Categorias de salário', ylabel='Quantidade', 
            title='Quantidade por categoria de salário', width=900, height=500)
```



**Considerações**:
* A maioria dos colaboradores possuem salários considerados `baixo` ou `médio`;
* Olhando apenas para o contexto de categorias de salários por quantidade de colaboradores por área, pode até ser considerado normal essa distribuição. Pois é natural que a área operacional (que representa a maioria dos colaboradores, como visto anteriormente) recebam menos que as áreas estratégicas;
* Porém, é natural pensar que um lugar que tenha muitos `baixo salários` tambem terá muitos colaboradores que deixarão a empresa;


---

<h2>4) ANÁLISE MULTIVARIADA</h2>

Após analisar cada atributo, agora irei verificar como os atributos se correlacionam, para enteder quais fatores explicam os fenômenos apresentados.


```python
df_mul = df_ind.copy()
df_mul.head()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
      <th>area</th>
      <th>salario</th>
      <th>nivel_satisfacao_bins</th>
      <th>ultima_avaliacao_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>baixo [20-40]</td>
      <td>médio [40-60]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>bom [60-80]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>bom [60-80]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>baixo [20-40]</td>
      <td>médio [40-60]</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Criação de uma função para automatizar a apresentação do gráfico de barras
def exibeBarplotMult(df, colunaIndex, labels, x, y, z, title='Default', xlabel='x_default', ylabel='y_default', width=1000, height=600):
    fig = go.Figure()

    for key, value in enumerate(labels):

        color = '#424456'
        if key == 1:
            color = '#e85151'

        _df = df.loc[df[colunaIndex] == key]

        fig.add_traces(go.Bar(
            x=_df[x],
            y=_df[y],
            name=value,
            text=_df[z],
            marker={'color': color}
        ))

    fig.update_layout(
        title=f'<span>{title}</span>', 
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(title=f'<span>{xlabel}</span>'),
        yaxis=dict(title=f'<span>{ylabel}</span>')
    )

    fig.show()

    #return fig
```

<h3>4.1) Deixou a empresa com as demais variáveis</h3>

A primeira análise será a cerca da variável target. Vamos verificar como o atributo `deixou_empresa` se correlacionam com os demais atributos.

**Matriz de Correlação**<br/>


```python
df_corr = df_mul[['deixou_empresa', 'ultima_avaliacao', 'numero_projetos',
       'horas_medias_por_mes', 'tempo_empresa', 'acidente_trabalho',
       'ultima_promocao_5anos', 'nivel_satisfacao', 'area', 'salario', 'nivel_satisfacao_bins', 'ultima_avaliacao_bins']].corr()
plt.figure(figsize=(12, 6))
sns.heatmap(df_corr, annot=True)
```




    <AxesSubplot:>




    
![png](assets/output_73_1.png)
    


Uma análise interessante a ser feita é a matriz de correlação entre os atritubos, vemos que:
* O atributo nivel_satisfacao é o que mais se relaciona com o atributo deixou_empresa, nota que a correlação é negativa, isto é, à medida que os colaboradores deixam a empresa, o nivel de satisfação diminui;

Outras correlações interessantes:
* Última avaliação aumenta conforme o Números de projetos também aumenta;
* Última avaliação aumenta conforme a quantidade de horas médias trabalhadas aumenta;
* Números de horas trabalhadas aumenta à medida que a quantidade de projetos também aumenta (Essa aqui era esperado de acontecer);



```python
def compararAtributos(df, coluna1, coluna2, colunaAlvo, aggFunc, percent=False):
    df_temp = df.copy()
    df_temp = df_temp.groupby([coluna1, coluna2], as_index=False)[colunaAlvo].agg(aggFunc)

    df_temp = df_temp.reset_index()
    if percent:
        #Porcentagem Total
        df_temp['% total'] = df_temp['count'].apply(lambda x: round(x / np.sum(df_temp['count']) * 100, 2))

        #Porcentagem por Grupo
        total = df_temp.groupby([coluna2], as_index=False)['count'].transform('sum')
        df_temp['% do grupo'] = df_temp['count'].div(total['count'])
        df_temp['% do grupo'] = df_temp['% do grupo'].apply(lambda x: round(x * 100, 2))
    
    df_temp = df_temp.sort_values(by=[coluna1, coluna2], ascending=True)

    return df_temp
```

**Deixou a empresa com Nível de Satisfação**<br/>


```python
df_temp = compararAtributos(df_mul, 'deixou_empresa', 'nivel_satisfacao_bins', 'nivel_satisfacao', ['count'], percent=True)
df_temp = df_temp.sort_values(by=['nivel_satisfacao_bins', 'deixou_empresa'], ascending=[True, False])
df_temp
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deixou_empresa</th>
      <th>nivel_satisfacao_bins</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>péssimo [0-20]</td>
      <td>924</td>
      <td>6.16</td>
      <td>62.52</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>péssimo [0-20]</td>
      <td>554</td>
      <td>3.69</td>
      <td>37.48</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>baixo [20-40]</td>
      <td>811</td>
      <td>5.41</td>
      <td>49.27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>baixo [20-40]</td>
      <td>835</td>
      <td>5.57</td>
      <td>50.73</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>médio [40-60]</td>
      <td>865</td>
      <td>5.77</td>
      <td>23.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>médio [40-60]</td>
      <td>2740</td>
      <td>18.27</td>
      <td>76.01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>bom [60-80]</td>
      <td>422</td>
      <td>2.81</td>
      <td>9.89</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>bom [60-80]</td>
      <td>3846</td>
      <td>25.64</td>
      <td>90.11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>ótimo [80-100]</td>
      <td>549</td>
      <td>3.66</td>
      <td>13.72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>ótimo [80-100]</td>
      <td>3453</td>
      <td>23.02</td>
      <td>86.28</td>
    </tr>
  </tbody>
</table>
</div>




```python
exibeBarplotMult(df_temp, colunaIndex='deixou_empresa', 
                x='nivel_satisfacao_bins', 
                y='count', z='% do grupo', labels=['Não', 'Sim'], 
                width=1200, height=400, 
                title='Quantidade de demissões e não demissões por grupo de satisfação dos colaboradores',
                xlabel='Nível de satisfação', ylabel='Quantidade')
```



**Deixou a empresa com Última Avaliação**<br/>


```python
df_temp = compararAtributos(df_mul, 'deixou_empresa', 'ultima_avaliacao_bins', 'ultima_avaliacao', ['count', 'mean'], percent=True)
display(df_temp)

exibeBarplotMult(df_temp, colunaIndex='deixou_empresa', 
                x='ultima_avaliacao_bins', 
                y='count', z='% do grupo', labels=['Não', 'Sim'], 
                width=1200, height=400, 
                title='Quantidade de demissões e não demissões por última avaliação dos colaboradores',
                xlabel='Última avaliação', ylabel='Quantidade')
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deixou_empresa</th>
      <th>ultima_avaliacao_bins</th>
      <th>count</th>
      <th>mean</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>péssimo [0-20]</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>baixo [20-40]</td>
      <td>236</td>
      <td>0.382839</td>
      <td>1.57</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>médio [40-60]</td>
      <td>3112</td>
      <td>0.529865</td>
      <td>20.75</td>
      <td>65.83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>bom [60-80]</td>
      <td>4282</td>
      <td>0.703206</td>
      <td>28.55</td>
      <td>94.17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>ótimo [80-100]</td>
      <td>3798</td>
      <td>0.902056</td>
      <td>25.32</td>
      <td>69.19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>péssimo [0-20]</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>baixo [20-40]</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>médio [40-60]</td>
      <td>1615</td>
      <td>0.512087</td>
      <td>10.77</td>
      <td>34.17</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>bom [60-80]</td>
      <td>265</td>
      <td>0.751887</td>
      <td>1.77</td>
      <td>5.83</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>ótimo [80-100]</td>
      <td>1691</td>
      <td>0.909586</td>
      <td>11.27</td>
      <td>30.81</td>
    </tr>
  </tbody>
</table>
</div>




Curioso analisar que vários dos colaboradores que fizeram a última avaliação de satisfação como ótimo deixaram a empresa logo em seguida.


**Deixou a Empresa por Área de atuação**<br/>


```python
df_temp = compararAtributos(df_mul, 'deixou_empresa', 'area', 'ultima_avaliacao', ['count'], percent=True)
df_temp
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deixou_empresa</th>
      <th>area</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>IT</td>
      <td>954</td>
      <td>6.36</td>
      <td>77.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>RandD</td>
      <td>666</td>
      <td>4.44</td>
      <td>84.63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>accounting</td>
      <td>563</td>
      <td>3.75</td>
      <td>73.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>hr</td>
      <td>524</td>
      <td>3.49</td>
      <td>70.91</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>management</td>
      <td>539</td>
      <td>3.59</td>
      <td>85.56</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>marketing</td>
      <td>655</td>
      <td>4.37</td>
      <td>76.34</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>product_mng</td>
      <td>704</td>
      <td>4.69</td>
      <td>78.05</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>sales</td>
      <td>3126</td>
      <td>20.84</td>
      <td>75.51</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>support</td>
      <td>1674</td>
      <td>11.16</td>
      <td>75.10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>technical</td>
      <td>2023</td>
      <td>13.49</td>
      <td>74.38</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>IT</td>
      <td>273</td>
      <td>1.82</td>
      <td>22.25</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>RandD</td>
      <td>121</td>
      <td>0.81</td>
      <td>15.37</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>accounting</td>
      <td>204</td>
      <td>1.36</td>
      <td>26.60</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>hr</td>
      <td>215</td>
      <td>1.43</td>
      <td>29.09</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>management</td>
      <td>91</td>
      <td>0.61</td>
      <td>14.44</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>marketing</td>
      <td>203</td>
      <td>1.35</td>
      <td>23.66</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>product_mng</td>
      <td>198</td>
      <td>1.32</td>
      <td>21.95</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>sales</td>
      <td>1014</td>
      <td>6.76</td>
      <td>24.49</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>support</td>
      <td>555</td>
      <td>3.70</td>
      <td>24.90</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>technical</td>
      <td>697</td>
      <td>4.65</td>
      <td>25.62</td>
    </tr>
  </tbody>
</table>
</div>




```python
exibeBarplotMult(df_temp, colunaIndex='deixou_empresa', 
                x='area', 
                y='count', z='% do grupo', labels=['Não', 'Sim'], 
                width=1600, height=400, 
                title='Quantidade de demissões e não demissões por area de atuação',
                xlabel='Area', ylabel='Quantidade')
```



De maneira geral, a maioria dos colaboradores que deixaram a empresa são das áreas de vendas, suporte e técnica. Isso é normal, já que são as areas com a maior quantidade de colaboradores;

**Deixou a empresa com Número de Projetos**<br/>


```python
df_temp = compararAtributos(df_mul, 'deixou_empresa', 'numero_projetos', 'area', ['count'], percent=True)
display(df_temp)

exibeBarplotMult(df_temp, colunaIndex='deixou_empresa', 
                x='numero_projetos', 
                y='count', z='% do grupo', labels=['Não', 'Sim'], 
                width=1200, height=400, 
                title='Quantidade de demissões e não demissões por quantidade de projetos',
                xlabel='Quantidade de projetos', ylabel='Quantidade')
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deixou_empresa</th>
      <th>numero_projetos</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>821</td>
      <td>5.47</td>
      <td>34.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>3983</td>
      <td>26.56</td>
      <td>98.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>4</td>
      <td>3956</td>
      <td>26.38</td>
      <td>90.63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>5</td>
      <td>2149</td>
      <td>14.33</td>
      <td>77.83</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>6</td>
      <td>519</td>
      <td>3.46</td>
      <td>44.21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>2</td>
      <td>1567</td>
      <td>10.45</td>
      <td>65.62</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>3</td>
      <td>72</td>
      <td>0.48</td>
      <td>1.78</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>4</td>
      <td>409</td>
      <td>2.73</td>
      <td>9.37</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>5</td>
      <td>612</td>
      <td>4.08</td>
      <td>22.17</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>6</td>
      <td>655</td>
      <td>4.37</td>
      <td>55.79</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>7</td>
      <td>256</td>
      <td>1.71</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>




**Deixou a empresa com Tempo de Empresa**<br/>

Um fenômeno interessante:
* A medida que a quantidade de projetos aumenta, a quantidade de deixou a empresa também aumenta, até o ponto de que todos os colaboradores que atuaram em 7 projetos sairam da empresa;
* Porém, a maioria todos os colaboradores que atuaram em apenas dois projetos deixaram a empresa;


```python
df_temp = compararAtributos(df_mul, 'deixou_empresa', 'tempo_empresa', 'area', ['count'], percent=True)
display(df_temp)

exibeBarplotMult(df_temp, colunaIndex='deixou_empresa', 
                x='tempo_empresa', 
                y='count', z='% do grupo', labels=['Não', 'Sim'], 
                width=1200, height=400, 
                title='Quantidade de demissões e não demissões por tempo de empresa',
                xlabel='Tempo de empresa', ylabel='Quantidade')
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deixou_empresa</th>
      <th>tempo_empresa</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>3191</td>
      <td>21.27</td>
      <td>98.37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>4857</td>
      <td>32.38</td>
      <td>75.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>4</td>
      <td>1667</td>
      <td>11.11</td>
      <td>65.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>5</td>
      <td>640</td>
      <td>4.27</td>
      <td>43.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>6</td>
      <td>509</td>
      <td>3.39</td>
      <td>70.89</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>7</td>
      <td>188</td>
      <td>1.25</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>8</td>
      <td>162</td>
      <td>1.08</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>10</td>
      <td>214</td>
      <td>1.43</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>2</td>
      <td>53</td>
      <td>0.35</td>
      <td>1.63</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>3</td>
      <td>1586</td>
      <td>10.57</td>
      <td>24.62</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>4</td>
      <td>890</td>
      <td>5.93</td>
      <td>34.81</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>5</td>
      <td>833</td>
      <td>5.55</td>
      <td>56.55</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>6</td>
      <td>209</td>
      <td>1.39</td>
      <td>29.11</td>
    </tr>
  </tbody>
</table>
</div>




**Deixou a empresa com Acidente de Trabalho**<br/>

O fator tempo também é importante aqui:
* Quanto mais tempo na empresa, menor a chance do colaborador deixar a empresa;


```python
df_temp = compararAtributos(df_mul, 'deixou_empresa', 'acidente_trabalho', 'area', ['count'], percent=True)
display(df_temp)

exibeBarplotMult(df_temp, colunaIndex='deixou_empresa', 
                x='acidente_trabalho', 
                y='count', z='% do grupo', labels=['Não', 'Sim'], 
                width=600, height=400, 
                title='Quantidade de demissões e não demissões por ocorrência de acidentes de trabalho',
                xlabel='Acidente', ylabel='Quantidade')
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deixou_empresa</th>
      <th>acidente_trabalho</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>9428</td>
      <td>62.86</td>
      <td>73.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2000</td>
      <td>13.33</td>
      <td>92.21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>3402</td>
      <td>22.68</td>
      <td>26.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>169</td>
      <td>1.13</td>
      <td>7.79</td>
    </tr>
  </tbody>
</table>
</div>




Apenas 1% de todos os colaboradores sofreram acidentes de trabalha E deixaram a empresa, isso mostra que talvez o evento sofrer acidente não seja tão relevante para o colaborador deixar a empresa.

**Deixou a empresa com Última Promoção nos últimos 5 anos**<br/>


```python
df_temp = compararAtributos(df_mul, 'deixou_empresa', 'ultima_promocao_5anos', 'area', ['count'], percent=True)
display(df_temp)

exibeBarplotMult(df_temp, colunaIndex='deixou_empresa', 
                x='ultima_promocao_5anos', 
                y='count', z='% do grupo', labels=['Não', 'Sim'], 
                width=800, height=400, 
                title='Quantidade de demissões e não demissões por última promoção nos últimos 5 anos',
                xlabel='Promoção', ylabel='Quantidade')
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>11128</td>
      <td>74.19</td>
      <td>75.80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>300</td>
      <td>2.00</td>
      <td>94.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>3552</td>
      <td>23.68</td>
      <td>24.20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>0.13</td>
      <td>5.96</td>
    </tr>
  </tbody>
</table>
</div>




Veja que praticamente nenhum colaborador que teve promoção nos últimos 5 anos saiu da empresa. Contudo, nota que apenas 2.13% dos colaboradores tiveram promoção nos últimos 5 anos.
Vemos que a empresa não tem um plano de carreira para seus colaboradores, é natural que as pessoas se desmotivem com o tempo; 

**Deixou a empresa com Salário**<br/>


```python
df_temp = compararAtributos(df_mul, 'deixou_empresa', 'salario', 'area', ['count'], percent=True)
display(df_temp)

exibeBarplotMult(df_temp, colunaIndex='deixou_empresa', 
                x='salario', 
                y='count', z='% do grupo', labels=['Não', 'Sim'], 
                width=800, height=400, 
                title='Quantidade de demissões e não demissões por categoria de salario',
                xlabel='Promoção', ylabel='Quantidade')
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>deixou_empresa</th>
      <th>salario</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>high</td>
      <td>1155</td>
      <td>7.70</td>
      <td>93.37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>low</td>
      <td>5144</td>
      <td>34.30</td>
      <td>70.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>medium</td>
      <td>5129</td>
      <td>34.20</td>
      <td>79.57</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>high</td>
      <td>82</td>
      <td>0.55</td>
      <td>6.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>low</td>
      <td>2172</td>
      <td>14.48</td>
      <td>29.69</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>medium</td>
      <td>1317</td>
      <td>8.78</td>
      <td>20.43</td>
    </tr>
  </tbody>
</table>
</div>




Um fenômeno que já era esperado:
* Pessoas que tem um salário na categoria baixo, seguido de médio são mais propensos a deixaram a empresa;

**Deixou a empresa com Horas Médias por Mês**<br/>


```python
fig = px.histogram(df_mul, 
                    x=df_mul['horas_medias_por_mes'], 
                    marginal='box', 
                    barmode ='overlay', 
                    histnorm ='density', 
                    color='deixou_empresa',
                    labels =[{1: 'Sim'}, {0: 'Não'}],
                    color_discrete_map={0: '#424456', 1: '#e85151'}
                    )  
fig.update_layout(
    title_text="Distribuição de horas médias trabalhadas por mês"
)
fig.show()
```



Temos duas situações aqui:
* Os colaboradores que trabalham entre 125~164 horas semanais são mais propensos a deixarem a empresa;
* A partir das 215 horas semanais o histograma de colaboradores que deixaram a empresa volta a crescer;
* Poucos funcionários que fazem mais de 275 horas semanais continuam na empresa;

<h3>4.2) Nível de satisfação com as demais variáveis</h3>

**Nível de Satisfação com Última Avaliação**<br/>


```python
df_temp = df_mul.groupby(['nivel_satisfacao_bins', 'ultima_avaliacao_bins'], as_index=False)['deixou_empresa'].agg(['count'])
df_temp['percent'] = df_temp['count'].apply(lambda x: round(x / np.sum(df_temp['count']) * 100, 2))

display(df_temp.sort_values(by=['percent'], ascending=False))
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>count</th>
      <th>percent</th>
    </tr>
    <tr>
      <th>nivel_satisfacao_bins</th>
      <th>ultima_avaliacao_bins</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ótimo [80-100]</th>
      <th>ótimo [80-100]</th>
      <td>1756</td>
      <td>11.71</td>
    </tr>
    <tr>
      <th>bom [60-80]</th>
      <th>ótimo [80-100]</th>
      <td>1714</td>
      <td>11.43</td>
    </tr>
    <tr>
      <th>médio [40-60]</th>
      <th>médio [40-60]</th>
      <td>1620</td>
      <td>10.80</td>
    </tr>
    <tr>
      <th>bom [60-80]</th>
      <th>bom [60-80]</th>
      <td>1492</td>
      <td>9.95</td>
    </tr>
    <tr>
      <th>ótimo [80-100]</th>
      <th>bom [60-80]</th>
      <td>1311</td>
      <td>8.74</td>
    </tr>
    <tr>
      <th>médio [40-60]</th>
      <th>bom [60-80]</th>
      <td>1049</td>
      <td>6.99</td>
    </tr>
    <tr>
      <th>baixo [20-40]</th>
      <th>médio [40-60]</th>
      <td>1038</td>
      <td>6.92</td>
    </tr>
    <tr>
      <th>bom [60-80]</th>
      <th>médio [40-60]</th>
      <td>1003</td>
      <td>6.69</td>
    </tr>
    <tr>
      <th>péssimo [0-20]</th>
      <th>ótimo [80-100]</th>
      <td>917</td>
      <td>6.11</td>
    </tr>
    <tr>
      <th>ótimo [80-100]</th>
      <th>médio [40-60]</th>
      <td>906</td>
      <td>6.04</td>
    </tr>
    <tr>
      <th>médio [40-60]</th>
      <th>ótimo [80-100]</th>
      <td>861</td>
      <td>5.74</td>
    </tr>
    <tr>
      <th>péssimo [0-20]</th>
      <th>bom [60-80]</th>
      <td>387</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">baixo [20-40]</th>
      <th>bom [60-80]</th>
      <td>308</td>
      <td>2.05</td>
    </tr>
    <tr>
      <th>ótimo [80-100]</th>
      <td>241</td>
      <td>1.61</td>
    </tr>
    <tr>
      <th>péssimo [0-20]</th>
      <th>médio [40-60]</th>
      <td>160</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>médio [40-60]</th>
      <th>baixo [20-40]</th>
      <td>75</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>bom [60-80]</th>
      <th>baixo [20-40]</th>
      <td>59</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>baixo [20-40]</th>
      <th>baixo [20-40]</th>
      <td>59</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>ótimo [80-100]</th>
      <th>baixo [20-40]</th>
      <td>29</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>péssimo [0-20]</th>
      <th>baixo [20-40]</th>
      <td>14</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>médio [40-60]</th>
      <th>péssimo [0-20]</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>bom [60-80]</th>
      <th>péssimo [0-20]</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>ótimo [80-100]</th>
      <th>péssimo [0-20]</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>baixo [20-40]</th>
      <th>péssimo [0-20]</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>péssimo [0-20]</th>
      <th>péssimo [0-20]</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


Nesta tabela vemos a relação do nível de satisfação com a última avaliação. Nota que a maioria das últimas avaliações são melhores do que o nível de satisfação, isso demonstra que a empresa vem melhorando.

**Nível de Satisfação com Números de Projetos**<br/>


```python
df_temp = compararAtributos(df_mul, 'nivel_satisfacao_bins', 'numero_projetos', 'deixou_empresa', ['count'], percent=True)

display(df_temp.sort_values(by='% do grupo', ascending=False))

df_temp['numero_projetos'] = df_temp['numero_projetos'].astype(str)
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao_bins</th>
      <th>numero_projetos</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>péssimo [0-20]</td>
      <td>7</td>
      <td>241</td>
      <td>1.61</td>
      <td>94.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>péssimo [0-20]</td>
      <td>6</td>
      <td>746</td>
      <td>4.97</td>
      <td>63.54</td>
    </tr>
    <tr>
      <th>12</th>
      <td>médio [40-60]</td>
      <td>2</td>
      <td>1073</td>
      <td>7.15</td>
      <td>44.93</td>
    </tr>
    <tr>
      <th>6</th>
      <td>baixo [20-40]</td>
      <td>2</td>
      <td>876</td>
      <td>5.84</td>
      <td>36.68</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ótimo [80-100]</td>
      <td>5</td>
      <td>998</td>
      <td>6.65</td>
      <td>36.15</td>
    </tr>
    <tr>
      <th>19</th>
      <td>bom [60-80]</td>
      <td>3</td>
      <td>1449</td>
      <td>9.66</td>
      <td>35.73</td>
    </tr>
    <tr>
      <th>20</th>
      <td>bom [60-80]</td>
      <td>4</td>
      <td>1557</td>
      <td>10.38</td>
      <td>35.67</td>
    </tr>
    <tr>
      <th>26</th>
      <td>ótimo [80-100]</td>
      <td>4</td>
      <td>1493</td>
      <td>9.95</td>
      <td>34.20</td>
    </tr>
    <tr>
      <th>21</th>
      <td>bom [60-80]</td>
      <td>5</td>
      <td>922</td>
      <td>6.15</td>
      <td>33.39</td>
    </tr>
    <tr>
      <th>25</th>
      <td>ótimo [80-100]</td>
      <td>3</td>
      <td>1253</td>
      <td>8.35</td>
      <td>30.90</td>
    </tr>
    <tr>
      <th>13</th>
      <td>médio [40-60]</td>
      <td>3</td>
      <td>1034</td>
      <td>6.89</td>
      <td>25.50</td>
    </tr>
    <tr>
      <th>14</th>
      <td>médio [40-60]</td>
      <td>4</td>
      <td>950</td>
      <td>6.33</td>
      <td>21.76</td>
    </tr>
    <tr>
      <th>15</th>
      <td>médio [40-60]</td>
      <td>5</td>
      <td>447</td>
      <td>2.98</td>
      <td>16.19</td>
    </tr>
    <tr>
      <th>10</th>
      <td>baixo [20-40]</td>
      <td>6</td>
      <td>146</td>
      <td>0.97</td>
      <td>12.44</td>
    </tr>
    <tr>
      <th>18</th>
      <td>bom [60-80]</td>
      <td>2</td>
      <td>237</td>
      <td>1.58</td>
      <td>9.92</td>
    </tr>
    <tr>
      <th>22</th>
      <td>bom [60-80]</td>
      <td>6</td>
      <td>102</td>
      <td>0.68</td>
      <td>8.69</td>
    </tr>
    <tr>
      <th>16</th>
      <td>médio [40-60]</td>
      <td>6</td>
      <td>97</td>
      <td>0.65</td>
      <td>8.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>péssimo [0-20]</td>
      <td>5</td>
      <td>208</td>
      <td>1.39</td>
      <td>7.53</td>
    </tr>
    <tr>
      <th>24</th>
      <td>ótimo [80-100]</td>
      <td>2</td>
      <td>175</td>
      <td>1.17</td>
      <td>7.33</td>
    </tr>
    <tr>
      <th>28</th>
      <td>ótimo [80-100]</td>
      <td>6</td>
      <td>83</td>
      <td>0.55</td>
      <td>7.07</td>
    </tr>
    <tr>
      <th>9</th>
      <td>baixo [20-40]</td>
      <td>5</td>
      <td>186</td>
      <td>1.24</td>
      <td>6.74</td>
    </tr>
    <tr>
      <th>7</th>
      <td>baixo [20-40]</td>
      <td>3</td>
      <td>221</td>
      <td>1.47</td>
      <td>5.45</td>
    </tr>
    <tr>
      <th>8</th>
      <td>baixo [20-40]</td>
      <td>4</td>
      <td>207</td>
      <td>1.38</td>
      <td>4.74</td>
    </tr>
    <tr>
      <th>11</th>
      <td>baixo [20-40]</td>
      <td>7</td>
      <td>10</td>
      <td>0.07</td>
      <td>3.91</td>
    </tr>
    <tr>
      <th>2</th>
      <td>péssimo [0-20]</td>
      <td>4</td>
      <td>158</td>
      <td>1.05</td>
      <td>3.62</td>
    </tr>
    <tr>
      <th>1</th>
      <td>péssimo [0-20]</td>
      <td>3</td>
      <td>98</td>
      <td>0.65</td>
      <td>2.42</td>
    </tr>
    <tr>
      <th>17</th>
      <td>médio [40-60]</td>
      <td>7</td>
      <td>4</td>
      <td>0.03</td>
      <td>1.56</td>
    </tr>
    <tr>
      <th>0</th>
      <td>péssimo [0-20]</td>
      <td>2</td>
      <td>27</td>
      <td>0.18</td>
      <td>1.13</td>
    </tr>
    <tr>
      <th>23</th>
      <td>bom [60-80]</td>
      <td>7</td>
      <td>1</td>
      <td>0.01</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ótimo [80-100]</td>
      <td>7</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


A maioria das pessoas que têm satisfação nível bom e ótimo veêm uma boa quantidade de projetos como sendo 3 ou 4 projetos.

**Nível de satisfação por horas médias por mês**<br/>


```python
fig = px.histogram(df_mul, 
                    x=df_mul['horas_medias_por_mes'], 
                    marginal='box', 
                    barmode ='overlay', 
                    histnorm ='density', 
                    color='nivel_satisfacao_bins',
                    #labels =[{1: 'Sim'}, {0: 'Não'}],
                    color_discrete_map={'baixo [20-40]': '#fcba03', 'péssimo [0-20]': '#eb290c', 'médio [40-60]': '#949494', 'bom [60-80]': '#00f239', 'ótimo [80-100]': '#0576e8'}
                    )  
fig.update_layout(
    title_text="Distribuição de horas médias trabalhadas por mês",
    width=1200,
    height=700
)
fig.show()
```



Pela visualização por boxplot fica nítido que a quantidade de horas trabalhadas por mês afeta diretamente na satisfação dos colaboradores, observa que:
* **Satisfação péssima**: São os colaboradores que trabalham muitas horas semanais, são os trabalhadores que fazem muita hora extra no serviço;
* **Satisfação baixa**: Se concentra mais nos colaboradores que trabalham poucas horas por semana;
* **Satisfação média**: Colaboradores que trabalham em média 40 horas semanais;
* **Satisfação boa e ótima**: Se concentra principalmente nos trabalhadores que tem a jornada semanal de 44~ horas em média.


**Nível de satisfação por tempo de empresa**<br/>


```python
df_temp = compararAtributos(df_mul, 'nivel_satisfacao_bins', 'tempo_empresa', 'deixou_empresa', ['count'], percent=False)

df_temp = df_temp.sort_values(by='tempo_empresa', ascending=False)

df_temp
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao_bins</th>
      <th>tempo_empresa</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>ótimo [80-100]</td>
      <td>10</td>
      <td>58</td>
    </tr>
    <tr>
      <th>7</th>
      <td>péssimo [0-20]</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15</th>
      <td>baixo [20-40]</td>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <th>23</th>
      <td>médio [40-60]</td>
      <td>10</td>
      <td>64</td>
    </tr>
    <tr>
      <th>31</th>
      <td>bom [60-80]</td>
      <td>10</td>
      <td>66</td>
    </tr>
    <tr>
      <th>6</th>
      <td>péssimo [0-20]</td>
      <td>8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>30</th>
      <td>bom [60-80]</td>
      <td>8</td>
      <td>56</td>
    </tr>
    <tr>
      <th>14</th>
      <td>baixo [20-40]</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>38</th>
      <td>ótimo [80-100]</td>
      <td>8</td>
      <td>48</td>
    </tr>
    <tr>
      <th>22</th>
      <td>médio [40-60]</td>
      <td>8</td>
      <td>40</td>
    </tr>
    <tr>
      <th>5</th>
      <td>péssimo [0-20]</td>
      <td>7</td>
      <td>16</td>
    </tr>
    <tr>
      <th>21</th>
      <td>médio [40-60]</td>
      <td>7</td>
      <td>50</td>
    </tr>
    <tr>
      <th>29</th>
      <td>bom [60-80]</td>
      <td>7</td>
      <td>44</td>
    </tr>
    <tr>
      <th>37</th>
      <td>ótimo [80-100]</td>
      <td>7</td>
      <td>62</td>
    </tr>
    <tr>
      <th>13</th>
      <td>baixo [20-40]</td>
      <td>7</td>
      <td>16</td>
    </tr>
    <tr>
      <th>36</th>
      <td>ótimo [80-100]</td>
      <td>6</td>
      <td>214</td>
    </tr>
    <tr>
      <th>28</th>
      <td>bom [60-80]</td>
      <td>6</td>
      <td>198</td>
    </tr>
    <tr>
      <th>20</th>
      <td>médio [40-60]</td>
      <td>6</td>
      <td>104</td>
    </tr>
    <tr>
      <th>4</th>
      <td>péssimo [0-20]</td>
      <td>6</td>
      <td>102</td>
    </tr>
    <tr>
      <th>12</th>
      <td>baixo [20-40]</td>
      <td>6</td>
      <td>100</td>
    </tr>
    <tr>
      <th>11</th>
      <td>baixo [20-40]</td>
      <td>5</td>
      <td>179</td>
    </tr>
    <tr>
      <th>3</th>
      <td>péssimo [0-20]</td>
      <td>5</td>
      <td>247</td>
    </tr>
    <tr>
      <th>35</th>
      <td>ótimo [80-100]</td>
      <td>5</td>
      <td>526</td>
    </tr>
    <tr>
      <th>19</th>
      <td>médio [40-60]</td>
      <td>5</td>
      <td>119</td>
    </tr>
    <tr>
      <th>27</th>
      <td>bom [60-80]</td>
      <td>5</td>
      <td>402</td>
    </tr>
    <tr>
      <th>18</th>
      <td>médio [40-60]</td>
      <td>4</td>
      <td>377</td>
    </tr>
    <tr>
      <th>2</th>
      <td>péssimo [0-20]</td>
      <td>4</td>
      <td>942</td>
    </tr>
    <tr>
      <th>34</th>
      <td>ótimo [80-100]</td>
      <td>4</td>
      <td>487</td>
    </tr>
    <tr>
      <th>10</th>
      <td>baixo [20-40]</td>
      <td>4</td>
      <td>180</td>
    </tr>
    <tr>
      <th>26</th>
      <td>bom [60-80]</td>
      <td>4</td>
      <td>571</td>
    </tr>
    <tr>
      <th>9</th>
      <td>baixo [20-40]</td>
      <td>3</td>
      <td>991</td>
    </tr>
    <tr>
      <th>33</th>
      <td>ótimo [80-100]</td>
      <td>3</td>
      <td>1566</td>
    </tr>
    <tr>
      <th>25</th>
      <td>bom [60-80]</td>
      <td>3</td>
      <td>1787</td>
    </tr>
    <tr>
      <th>17</th>
      <td>médio [40-60]</td>
      <td>3</td>
      <td>1984</td>
    </tr>
    <tr>
      <th>1</th>
      <td>péssimo [0-20]</td>
      <td>3</td>
      <td>115</td>
    </tr>
    <tr>
      <th>8</th>
      <td>baixo [20-40]</td>
      <td>2</td>
      <td>148</td>
    </tr>
    <tr>
      <th>32</th>
      <td>ótimo [80-100]</td>
      <td>2</td>
      <td>1041</td>
    </tr>
    <tr>
      <th>24</th>
      <td>bom [60-80]</td>
      <td>2</td>
      <td>1144</td>
    </tr>
    <tr>
      <th>16</th>
      <td>médio [40-60]</td>
      <td>2</td>
      <td>867</td>
    </tr>
    <tr>
      <th>0</th>
      <td>péssimo [0-20]</td>
      <td>2</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>



Veja só, há funcionários que estão na empresa há 10 anos e possuem nível de satisfação péssimo.

**Nível de satisfação por acidente de trabalho**<br/>


```python
fig = px.histogram(df_mul, 
                    x=df_mul['nivel_satisfacao_bins'], 
                    
                    barmode ='overlay', 
                    histnorm ='density', 
                    color='acidente_trabalho',
                    #labels =[{1: 'Sim'}, {0: 'Não'}],
                    color_discrete_map={'baixo [20-40]': '#fcba03', 'péssimo [0-20]': '#eb290c', 'médio [40-60]': '#949494', 'bom [60-80]': '#00f239', 'ótimo [80-100]': '#0576e8'}
                    )  
fig.update_layout(
    title_text="Nível de satisfação por quantidade de ocorrências de acidentes de trabalho",
    width=800,
    height=400
)
fig.show()
```



O fato de o colaborador ter sofrido acidente de trabalho parece não ter relação com seu nível de satisfação trabalhando na empresa. Mais um indício de que sofrer acidente no trabalho não impacta na taxa de demissões na empresa.

**Nível de satisfação por promoção nos últimos 5 anos**<br/>


```python
fig = px.histogram(df_mul, 
                    x=df_mul['nivel_satisfacao_bins'], 
                    
                    barmode ='overlay', 
                    histnorm ='density', 
                    color='ultima_promocao_5anos',
                    #labels =[{1: 'Sim'}, {0: 'Não'}],
                    color_discrete_map={'baixo [20-40]': '#fcba03', 'péssimo [0-20]': '#eb290c', 'médio [40-60]': '#949494', 'bom [60-80]': '#00f239', 'ótimo [80-100]': '#0576e8'}
                    )  
fig.update_layout(
    title_text="Nível de satisfação por promoções nos últimos 5 anos",
    width=800,
    height=400
)
fig.show()
```



Agora uma tendência que já era esperada. Os colaboradores que tiverem promoção em sua maioria têm um nível de satisfação `médio`, `bom` ou `ótimo`. Em contrapartida, poucos colaboradores que tiveram promoção possuem um nível de satisfação `baixo` ou `péssimo`.

**Nível de satisfação por area de atuação**<br/>


```python
df_temp = compararAtributos(df_mul, 'nivel_satisfacao_bins', 'area', 'deixou_empresa', ['count'], percent=True)
df_temp = df_temp.sort_values(by=['area', '% do grupo'], ascending=[True, False])
df_temp
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao_bins</th>
      <th>area</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>bom [60-80]</td>
      <td>IT</td>
      <td>373</td>
      <td>2.49</td>
      <td>30.40</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ótimo [80-100]</td>
      <td>IT</td>
      <td>333</td>
      <td>2.22</td>
      <td>27.14</td>
    </tr>
    <tr>
      <th>20</th>
      <td>médio [40-60]</td>
      <td>IT</td>
      <td>268</td>
      <td>1.79</td>
      <td>21.84</td>
    </tr>
    <tr>
      <th>0</th>
      <td>péssimo [0-20]</td>
      <td>IT</td>
      <td>127</td>
      <td>0.85</td>
      <td>10.35</td>
    </tr>
    <tr>
      <th>10</th>
      <td>baixo [20-40]</td>
      <td>IT</td>
      <td>126</td>
      <td>0.84</td>
      <td>10.27</td>
    </tr>
    <tr>
      <th>31</th>
      <td>bom [60-80]</td>
      <td>RandD</td>
      <td>230</td>
      <td>1.53</td>
      <td>29.22</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ótimo [80-100]</td>
      <td>RandD</td>
      <td>219</td>
      <td>1.46</td>
      <td>27.83</td>
    </tr>
    <tr>
      <th>21</th>
      <td>médio [40-60]</td>
      <td>RandD</td>
      <td>188</td>
      <td>1.25</td>
      <td>23.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>péssimo [0-20]</td>
      <td>RandD</td>
      <td>75</td>
      <td>0.50</td>
      <td>9.53</td>
    </tr>
    <tr>
      <th>11</th>
      <td>baixo [20-40]</td>
      <td>RandD</td>
      <td>75</td>
      <td>0.50</td>
      <td>9.53</td>
    </tr>
    <tr>
      <th>32</th>
      <td>bom [60-80]</td>
      <td>accounting</td>
      <td>195</td>
      <td>1.30</td>
      <td>25.42</td>
    </tr>
    <tr>
      <th>42</th>
      <td>ótimo [80-100]</td>
      <td>accounting</td>
      <td>189</td>
      <td>1.26</td>
      <td>24.64</td>
    </tr>
    <tr>
      <th>22</th>
      <td>médio [40-60]</td>
      <td>accounting</td>
      <td>182</td>
      <td>1.21</td>
      <td>23.73</td>
    </tr>
    <tr>
      <th>12</th>
      <td>baixo [20-40]</td>
      <td>accounting</td>
      <td>114</td>
      <td>0.76</td>
      <td>14.86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>péssimo [0-20]</td>
      <td>accounting</td>
      <td>87</td>
      <td>0.58</td>
      <td>11.34</td>
    </tr>
    <tr>
      <th>33</th>
      <td>bom [60-80]</td>
      <td>hr</td>
      <td>198</td>
      <td>1.32</td>
      <td>26.79</td>
    </tr>
    <tr>
      <th>23</th>
      <td>médio [40-60]</td>
      <td>hr</td>
      <td>192</td>
      <td>1.28</td>
      <td>25.98</td>
    </tr>
    <tr>
      <th>43</th>
      <td>ótimo [80-100]</td>
      <td>hr</td>
      <td>185</td>
      <td>1.23</td>
      <td>25.03</td>
    </tr>
    <tr>
      <th>13</th>
      <td>baixo [20-40]</td>
      <td>hr</td>
      <td>93</td>
      <td>0.62</td>
      <td>12.58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>péssimo [0-20]</td>
      <td>hr</td>
      <td>71</td>
      <td>0.47</td>
      <td>9.61</td>
    </tr>
    <tr>
      <th>34</th>
      <td>bom [60-80]</td>
      <td>management</td>
      <td>208</td>
      <td>1.39</td>
      <td>33.02</td>
    </tr>
    <tr>
      <th>24</th>
      <td>médio [40-60]</td>
      <td>management</td>
      <td>174</td>
      <td>1.16</td>
      <td>27.62</td>
    </tr>
    <tr>
      <th>44</th>
      <td>ótimo [80-100]</td>
      <td>management</td>
      <td>144</td>
      <td>0.96</td>
      <td>22.86</td>
    </tr>
    <tr>
      <th>14</th>
      <td>baixo [20-40]</td>
      <td>management</td>
      <td>54</td>
      <td>0.36</td>
      <td>8.57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>péssimo [0-20]</td>
      <td>management</td>
      <td>50</td>
      <td>0.33</td>
      <td>7.94</td>
    </tr>
    <tr>
      <th>35</th>
      <td>bom [60-80]</td>
      <td>marketing</td>
      <td>253</td>
      <td>1.69</td>
      <td>29.49</td>
    </tr>
    <tr>
      <th>45</th>
      <td>ótimo [80-100]</td>
      <td>marketing</td>
      <td>230</td>
      <td>1.53</td>
      <td>26.81</td>
    </tr>
    <tr>
      <th>25</th>
      <td>médio [40-60]</td>
      <td>marketing</td>
      <td>201</td>
      <td>1.34</td>
      <td>23.43</td>
    </tr>
    <tr>
      <th>15</th>
      <td>baixo [20-40]</td>
      <td>marketing</td>
      <td>99</td>
      <td>0.66</td>
      <td>11.54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>péssimo [0-20]</td>
      <td>marketing</td>
      <td>75</td>
      <td>0.50</td>
      <td>8.74</td>
    </tr>
    <tr>
      <th>46</th>
      <td>ótimo [80-100]</td>
      <td>product_mng</td>
      <td>252</td>
      <td>1.68</td>
      <td>27.94</td>
    </tr>
    <tr>
      <th>36</th>
      <td>bom [60-80]</td>
      <td>product_mng</td>
      <td>240</td>
      <td>1.60</td>
      <td>26.61</td>
    </tr>
    <tr>
      <th>26</th>
      <td>médio [40-60]</td>
      <td>product_mng</td>
      <td>236</td>
      <td>1.57</td>
      <td>26.16</td>
    </tr>
    <tr>
      <th>16</th>
      <td>baixo [20-40]</td>
      <td>product_mng</td>
      <td>93</td>
      <td>0.62</td>
      <td>10.31</td>
    </tr>
    <tr>
      <th>6</th>
      <td>péssimo [0-20]</td>
      <td>product_mng</td>
      <td>81</td>
      <td>0.54</td>
      <td>8.98</td>
    </tr>
    <tr>
      <th>37</th>
      <td>bom [60-80]</td>
      <td>sales</td>
      <td>1163</td>
      <td>7.75</td>
      <td>28.09</td>
    </tr>
    <tr>
      <th>47</th>
      <td>ótimo [80-100]</td>
      <td>sales</td>
      <td>1119</td>
      <td>7.46</td>
      <td>27.03</td>
    </tr>
    <tr>
      <th>27</th>
      <td>médio [40-60]</td>
      <td>sales</td>
      <td>1003</td>
      <td>6.69</td>
      <td>24.23</td>
    </tr>
    <tr>
      <th>17</th>
      <td>baixo [20-40]</td>
      <td>sales</td>
      <td>445</td>
      <td>2.97</td>
      <td>10.75</td>
    </tr>
    <tr>
      <th>7</th>
      <td>péssimo [0-20]</td>
      <td>sales</td>
      <td>410</td>
      <td>2.73</td>
      <td>9.90</td>
    </tr>
    <tr>
      <th>38</th>
      <td>bom [60-80]</td>
      <td>support</td>
      <td>652</td>
      <td>4.35</td>
      <td>29.25</td>
    </tr>
    <tr>
      <th>48</th>
      <td>ótimo [80-100]</td>
      <td>support</td>
      <td>594</td>
      <td>3.96</td>
      <td>26.65</td>
    </tr>
    <tr>
      <th>28</th>
      <td>médio [40-60]</td>
      <td>support</td>
      <td>527</td>
      <td>3.51</td>
      <td>23.64</td>
    </tr>
    <tr>
      <th>18</th>
      <td>baixo [20-40]</td>
      <td>support</td>
      <td>253</td>
      <td>1.69</td>
      <td>11.35</td>
    </tr>
    <tr>
      <th>8</th>
      <td>péssimo [0-20]</td>
      <td>support</td>
      <td>203</td>
      <td>1.35</td>
      <td>9.11</td>
    </tr>
    <tr>
      <th>39</th>
      <td>bom [60-80]</td>
      <td>technical</td>
      <td>756</td>
      <td>5.04</td>
      <td>27.79</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ótimo [80-100]</td>
      <td>technical</td>
      <td>737</td>
      <td>4.91</td>
      <td>27.10</td>
    </tr>
    <tr>
      <th>29</th>
      <td>médio [40-60]</td>
      <td>technical</td>
      <td>634</td>
      <td>4.23</td>
      <td>23.31</td>
    </tr>
    <tr>
      <th>9</th>
      <td>péssimo [0-20]</td>
      <td>technical</td>
      <td>299</td>
      <td>1.99</td>
      <td>10.99</td>
    </tr>
    <tr>
      <th>19</th>
      <td>baixo [20-40]</td>
      <td>technical</td>
      <td>294</td>
      <td>1.96</td>
      <td>10.81</td>
    </tr>
  </tbody>
</table>
</div>



Podemos ver que a maioria dos colaboradores por area classificam o nível de satisfeito como `bom`. A única area que a maioria dos funcionários são `ótimo` é a área de `product_mng`.
* Por que a maioria dos colaboradores classificam como ótimo? O que fazer nas outras áreas para que sejam como a de `product_mng`?

Mas agora vamos nos concetrar nos valores `baixo` ou `péssimo`.


```python
df_ruim = df_temp.loc[(df_temp['nivel_satisfacao_bins'] == 'péssimo [0-20]') | (df_temp['nivel_satisfacao_bins'] == 'baixo [20-40]')]
df_ruim
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao_bins</th>
      <th>area</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>péssimo [0-20]</td>
      <td>IT</td>
      <td>127</td>
      <td>0.85</td>
      <td>10.35</td>
    </tr>
    <tr>
      <th>10</th>
      <td>baixo [20-40]</td>
      <td>IT</td>
      <td>126</td>
      <td>0.84</td>
      <td>10.27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>péssimo [0-20]</td>
      <td>RandD</td>
      <td>75</td>
      <td>0.50</td>
      <td>9.53</td>
    </tr>
    <tr>
      <th>11</th>
      <td>baixo [20-40]</td>
      <td>RandD</td>
      <td>75</td>
      <td>0.50</td>
      <td>9.53</td>
    </tr>
    <tr>
      <th>12</th>
      <td>baixo [20-40]</td>
      <td>accounting</td>
      <td>114</td>
      <td>0.76</td>
      <td>14.86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>péssimo [0-20]</td>
      <td>accounting</td>
      <td>87</td>
      <td>0.58</td>
      <td>11.34</td>
    </tr>
    <tr>
      <th>13</th>
      <td>baixo [20-40]</td>
      <td>hr</td>
      <td>93</td>
      <td>0.62</td>
      <td>12.58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>péssimo [0-20]</td>
      <td>hr</td>
      <td>71</td>
      <td>0.47</td>
      <td>9.61</td>
    </tr>
    <tr>
      <th>14</th>
      <td>baixo [20-40]</td>
      <td>management</td>
      <td>54</td>
      <td>0.36</td>
      <td>8.57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>péssimo [0-20]</td>
      <td>management</td>
      <td>50</td>
      <td>0.33</td>
      <td>7.94</td>
    </tr>
    <tr>
      <th>15</th>
      <td>baixo [20-40]</td>
      <td>marketing</td>
      <td>99</td>
      <td>0.66</td>
      <td>11.54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>péssimo [0-20]</td>
      <td>marketing</td>
      <td>75</td>
      <td>0.50</td>
      <td>8.74</td>
    </tr>
    <tr>
      <th>16</th>
      <td>baixo [20-40]</td>
      <td>product_mng</td>
      <td>93</td>
      <td>0.62</td>
      <td>10.31</td>
    </tr>
    <tr>
      <th>6</th>
      <td>péssimo [0-20]</td>
      <td>product_mng</td>
      <td>81</td>
      <td>0.54</td>
      <td>8.98</td>
    </tr>
    <tr>
      <th>17</th>
      <td>baixo [20-40]</td>
      <td>sales</td>
      <td>445</td>
      <td>2.97</td>
      <td>10.75</td>
    </tr>
    <tr>
      <th>7</th>
      <td>péssimo [0-20]</td>
      <td>sales</td>
      <td>410</td>
      <td>2.73</td>
      <td>9.90</td>
    </tr>
    <tr>
      <th>18</th>
      <td>baixo [20-40]</td>
      <td>support</td>
      <td>253</td>
      <td>1.69</td>
      <td>11.35</td>
    </tr>
    <tr>
      <th>8</th>
      <td>péssimo [0-20]</td>
      <td>support</td>
      <td>203</td>
      <td>1.35</td>
      <td>9.11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>péssimo [0-20]</td>
      <td>technical</td>
      <td>299</td>
      <td>1.99</td>
      <td>10.99</td>
    </tr>
    <tr>
      <th>19</th>
      <td>baixo [20-40]</td>
      <td>technical</td>
      <td>294</td>
      <td>1.96</td>
      <td>10.81</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ruim = pd.DataFrame(df_ruim.groupby(['area'])['% do grupo'].sum()).sort_values(by=['% do grupo'], ascending=False)
df_ruim
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>% do grupo</th>
    </tr>
    <tr>
      <th>area</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accounting</th>
      <td>26.20</td>
    </tr>
    <tr>
      <th>hr</th>
      <td>22.19</td>
    </tr>
    <tr>
      <th>technical</th>
      <td>21.80</td>
    </tr>
    <tr>
      <th>sales</th>
      <td>20.65</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>20.62</td>
    </tr>
    <tr>
      <th>support</th>
      <td>20.46</td>
    </tr>
    <tr>
      <th>marketing</th>
      <td>20.28</td>
    </tr>
    <tr>
      <th>product_mng</th>
      <td>19.29</td>
    </tr>
    <tr>
      <th>RandD</th>
      <td>19.06</td>
    </tr>
    <tr>
      <th>management</th>
      <td>16.51</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ruim.to_excel('15_satisfacao_ruim_area.xlsx')
```

Aqui temos a porcentagem por área de nível de satisfação `baixo` e `péssimo`.
* 26% dos colaboradores da area de accounting não estão satisfeitos.
* 22% dos colaboradores da area de RH não estão satisfeitos.

**Nível de satisfação por salário**<br/>


```python
df_temp = compararAtributos(df_mul, 'nivel_satisfacao_bins', 'salario', 'deixou_empresa', ['count'], percent=True)

df_temp = df_temp.sort_values(by=['salario', '% do grupo'], ascending=[True, False])
df_temp
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao_bins</th>
      <th>salario</th>
      <th>count</th>
      <th>% total</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>bom [60-80]</td>
      <td>high</td>
      <td>410</td>
      <td>2.73</td>
      <td>33.14</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ótimo [80-100]</td>
      <td>high</td>
      <td>320</td>
      <td>2.13</td>
      <td>25.87</td>
    </tr>
    <tr>
      <th>6</th>
      <td>médio [40-60]</td>
      <td>high</td>
      <td>315</td>
      <td>2.10</td>
      <td>25.46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>baixo [20-40]</td>
      <td>high</td>
      <td>112</td>
      <td>0.75</td>
      <td>9.05</td>
    </tr>
    <tr>
      <th>0</th>
      <td>péssimo [0-20]</td>
      <td>high</td>
      <td>80</td>
      <td>0.53</td>
      <td>6.47</td>
    </tr>
    <tr>
      <th>10</th>
      <td>bom [60-80]</td>
      <td>low</td>
      <td>1970</td>
      <td>13.13</td>
      <td>26.93</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ótimo [80-100]</td>
      <td>low</td>
      <td>1925</td>
      <td>12.83</td>
      <td>26.31</td>
    </tr>
    <tr>
      <th>7</th>
      <td>médio [40-60]</td>
      <td>low</td>
      <td>1727</td>
      <td>11.51</td>
      <td>23.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>baixo [20-40]</td>
      <td>low</td>
      <td>886</td>
      <td>5.91</td>
      <td>12.11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>péssimo [0-20]</td>
      <td>low</td>
      <td>808</td>
      <td>5.39</td>
      <td>11.04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>bom [60-80]</td>
      <td>medium</td>
      <td>1888</td>
      <td>12.59</td>
      <td>29.29</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ótimo [80-100]</td>
      <td>medium</td>
      <td>1757</td>
      <td>11.71</td>
      <td>27.26</td>
    </tr>
    <tr>
      <th>8</th>
      <td>médio [40-60]</td>
      <td>medium</td>
      <td>1563</td>
      <td>10.42</td>
      <td>24.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>baixo [20-40]</td>
      <td>medium</td>
      <td>648</td>
      <td>4.32</td>
      <td>10.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>péssimo [0-20]</td>
      <td>medium</td>
      <td>590</td>
      <td>3.93</td>
      <td>9.15</td>
    </tr>
  </tbody>
</table>
</div>



Por salário temos que 6% dos colaboradores que possuem o salário alto não estão nada satisfeitos com a empresa. 23% dos colaboradores que possuem salário baixo estão nada satisfeitos ou pouco satisfeitos com a empresa.


```python
df_ruim = df_temp.loc[(df_temp['nivel_satisfacao_bins'] == 'péssimo [0-20]') | (df_temp['nivel_satisfacao_bins'] == 'baixo [20-40]')]

df_ruim = pd.DataFrame(df_ruim.groupby(['salario'], as_index=False)['% do grupo'].sum()).sort_values(by=['% do grupo'], ascending=False)

df_ruim['% do grupo'] = df_ruim['% do grupo'].apply(lambda x: round(x, 2))

df_ruim
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salario</th>
      <th>% do grupo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>low</td>
      <td>23.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>medium</td>
      <td>19.20</td>
    </tr>
    <tr>
      <th>0</th>
      <td>high</td>
      <td>15.52</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.bar(x=df_ruim['salario'], y=df_ruim['% do grupo'], text=df_ruim['% do grupo'])

fig.update_traces(marker_color='#424456')

fig.update_layout(
        title='<span>Porcentagem por grupo de salario de colaboradores instatisfeitos</span>', 
        autosize=False,
        width=700,
        height=500,
        xaxis=dict(title='<span>Faixa de salário</span>'),
        yaxis=dict(title='<span>Porcentagem</span>')
    )

fig.show()
```



Aparentemente podemos ver que quanto maior o salário, mais o colaborador tende a se sentir satisfeito com o emprego.

---

<h2>5) PRÉ PROCESSAMENTO DOS DADOS</h2>

Agora que fizemos todas as análises e temos vários insights sobre os dados, vamos realizar a etapa de pré-processamento. Nesta etapa iremos tratar os outliers, padronizar e aplicar transformações de escala nos dados.

<h3>5.1) Importação das bibliotecas</h3>


```python
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler, RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC


# Bibliotecas para computação estatística
from scipy import stats
```

Abaixo irei criar uma classe de apoio para auxiliar no one hot encoder do nosso dataset. Esta classe aplicar o one-hot-encoder e também nomeia os atributos que foram transformados.


```python
class MyHotEncoder:
    def __init__(self, df, cols):
        self.df = df
        self.list_oneHotEncoder = []
        self.list_col = []

        if isinstance(cols, list):
            self.list_col = cols
        else:
            self.list_col.append(cols)

    #Treina os dados passados.
    def fit_transform(self):

        self.df.reset_index(drop=True, inplace=True) #Reseta o index do dataset para não causar problemas
        X = self.df.values

        #Para cada coluna crie uma string contendo o nome do atributo e o nome da categoria.
        #Exemplo de saída: [sex_male]
        for col in self.list_col: 
            valores = list(self.df[col].sort_values().unique())
            colunas = []
            for i in valores:
                colunas.append(col + '_' + str(i))
            colunas = colunas + list(self.df.columns)
            
            #One hot encoder para treinar
            onehotencoder = OneHotEncoder(handle_unknown='ignore')
            self.list_oneHotEncoder.append(onehotencoder)

            enc_df = onehotencoder.fit_transform(self.df[[col]]).toarray()
            enc_df = pd.DataFrame(enc_df)
            
            #União do dataset gerado ao dataset original
            self.df = enc_df.join(self.df)
            self.df.columns = colunas
            self.df.drop(col, axis=1, inplace=True)
        
        return self.df

    #Apenas aplica a transformação a novos dados.
    def transform(self, df):
        
        for i in range(len(self.list_col)):
            col = self.list_col[i]

            valores = list(df[col].sort_values().unique())
            colunas = []
            for j in valores:
                colunas.append(col + '_' + str(j))
            colunas = colunas + list(df.columns)

            #One hot encoder para transformar
            enc_df = self.list_oneHotEncoder[i].transform(df[[col]]).toarray()
            enc_df = pd.DataFrame(enc_df)

            #União do dataset gerado ao dataset original
            df = enc_df.join(df)
            df.columns = colunas
            df.drop(col, axis=1, inplace=True)


        return df   
```

<h3>5.2) Tratamento de outliers</h3>


```python
if df_ind['nivel_satisfacao'].dtypes == 'float64':
    print(True)

for coluna in df_ind.columns:
    if df_ind[coluna].dtypes == 'float64' or df_ind[coluna].dtypes == 'int64':
        fig = px.box(y=df_ind[coluna])
        fig.update_layout(
            title_text=f"Boxplot: {coluna}",
            width=400,
            height=300
        )
        fig.show()
```

    True
    

















**Considerações**:
* Pelas análises prévias, consideramos que o atributo sobre ocorrência de acidentes de trabalho não impacta no evento de o colaborador deixar a empresa;
* Colaboradores com mais do que 6 anos de empresa são mínimos e a quantidade de colaboradores com MAIS de 6 anos E que deixaram a empresa é zero. Porém a quantidade colaboradores com exatos 6 anos de empresa e que deixaram a empresa é significativa, por isso, mesmo com o valor 6 anos sendo considerado outlier, eu irei apenas remover os colaboradores com tempo_empresa MAIOR que 6 anos.


```python
df_ind.loc[df_ind['tempo_empresa'] >= 6, 'deixou_empresa'].value_counts()
```




    0    1073
    1     209
    Name: deixou_empresa, dtype: int64



Exemplo do que eu estou falando. 

Agora para remover os outliers eu poderia simplesmente utilizar o drop do pandas considerando os tempo_empresa maior que 6. Mas eu quero realizar essa etapa de maneira quantitativa, por isso irei aplicar o calculo do z-score para obter os dados outliers novamente. O Z-Score também é chamado de pontuação padrão. Este valor/pontuação ajuda a entender a que distância o ponto de dados está da média. E depois de definir um valor limite, pode-se utilizar os valores de pontuação z dos pontos de dados para definir os valores discrepantes.

Agora, para definir um valor limite de outlier é escolhido geralmente o valor 3. Como 99,7% dos pontos de dados estão entre +/- 3 desvios padrão (usando a abordagem de distribuição de Gauss).


```python
#Cálculo do Z-score
z = np.abs(stats.zscore(df_ind['tempo_empresa']))

#Definindo o threshold igual a 3 desvios padrão
threshold = 3
indices = np.where(z > 3)

#Visualizando os outliers
df_temp = df_ind.iloc[indices]


df_temp['tempo_empresa'].value_counts()
```




    10    214
    8     162
    Name: tempo_empresa, dtype: int64



Observe que pelo calculo do z-score obtemos como outliers somente os valores tempo_empresa que são 8 e 10 anos, portanto, agora é só aplicar o drop dos registros na tabela.

> **NOTA**: Por que a visualização do boxplot considera os valores 6 anos de empresa como outliers e a pontuação por z-score não?<br/><br/>
> **Resposta**: Isso acontece porque a pontuação Z-Score considera como sendo outliers dados que estão a mais de 3 desvios-padrão de distância do centro. Por outro lado, o cálculo utilizado para os boxplot é o calculo de IQR que considera 2,68 desvios-padrão de distância do centro. 


```python
#Removendo os outliers
df_ind.drop(index=np.array(indices).reshape(-1), inplace=True)
df_ind.reset_index(inplace=True, drop=True)
df_ind.head()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>deixou_empresa</th>
      <th>ultima_promocao_5anos</th>
      <th>area</th>
      <th>salario</th>
      <th>nivel_satisfacao_bins</th>
      <th>ultima_avaliacao_bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>baixo [20-40]</td>
      <td>médio [40-60]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>bom [60-80]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>péssimo [0-20]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>bom [60-80]</td>
      <td>ótimo [80-100]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>baixo [20-40]</td>
      <td>médio [40-60]</td>
    </tr>
  </tbody>
</table>
</div>



<h3>5.3) Primeira seleção de atributos e one hot encoder</h3>

Agora irei selecionar os atributos que serão utilizados como variáveis preditoras e o atributo target. Irei selecionar todos os atributos iniciais para a construção de um modelo base.<br/>
Após concluído o modelo base, irei testar e avaliar quais são os atributos mais relevantes para o modelo.



```python
#Seleção dos atributos preditores
X = df_ind[['nivel_satisfacao', 'ultima_avaliacao', 'numero_projetos', 'horas_medias_por_mes', 'tempo_empresa', 'acidente_trabalho',
       'ultima_promocao_5anos', 'area', 'salario']]

#Seleção do atributo target
Y = df_ind['deixou_empresa']

X.head()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>ultima_promocao_5anos</th>
      <th>area</th>
      <th>salario</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Aplicando a classe que foi definida MyHotEncoder para realizar o One Hot Encoding dos atributos.
my_encoder = MyHotEncoder(X,['area', 'salario'])
x_encoded = my_encoder.fit_transform()
```


```python
'''salario_map = {
    'low': 0,
    'medium': 0.5,
    'high': 1
}

x_encoded['salario'] = x_encoded['salario'].map(salario_map)'''
```




    "salario_map = {\n    'low': 0,\n    'medium': 0.5,\n    'high': 1\n}\n\nx_encoded['salario'] = x_encoded['salario'].map(salario_map)"




```python
display(x_encoded.head(3))
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salario_high</th>
      <th>salario_low</th>
      <th>salario_medium</th>
      <th>area_IT</th>
      <th>area_RandD</th>
      <th>area_accounting</th>
      <th>area_hr</th>
      <th>area_management</th>
      <th>area_marketing</th>
      <th>area_product_mng</th>
      <th>area_sales</th>
      <th>area_support</th>
      <th>area_technical</th>
      <th>nivel_satisfacao</th>
      <th>ultima_avaliacao</th>
      <th>numero_projetos</th>
      <th>horas_medias_por_mes</th>
      <th>tempo_empresa</th>
      <th>acidente_trabalho</th>
      <th>ultima_promocao_5anos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
#Convertendo de dataframe para array de valores.
x_encoded_val = x_encoded.values
```

<h3>5.4) Modelagem base para primeira avaliação</h3>

Aqui irei criar um modelo base com o objetivo de avaliar os melhores parâmetros e como os dados se saem quando expostos a um algoritmo de Machine Learning.


```python
scaler = StandardScaler()

x_encoded_scaler = scaler.fit_transform(x_encoded_val)
```

**Separando os dados em treino e teste e Balanceando o dataset**


```python
#Separando os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(x_encoded_scaler, Y, test_size = 0.3, random_state = 0)
```

Quando se trabalha com Machine Learning, um dos problemas é o fato de termos datasets desbalanceados, isto é, uma classe tem maior quantidade do que outro. Esse é um problema que pode acarretar em um modelo enviesado.

Para resolver isso irei utilizar um pacote chamado RandomUnderSampler que irá realizar esse trbalho de balancear o dataset. 


```python
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X_train, Y_train)
```

**5.4.1) KNN**

Como primeira modelo, irei utilizar um dos mais simples de todos, o KNN. Vamos verificar se a relação por proximidade de vizinhança se mostra boa o suficiente;


```python
knn = KNeighborsClassifier()
knn.fit(X_rus, y_rus)

preds = knn.predict(X_test)
f_score = f1_score(preds, Y_test)
ac_score = accuracy_score(preds, Y_test)

print('F1 score em Teste = {:.2f}%'.format(f_score * 100))
print('Acurácia em Teste = {:.2f}%'.format(ac_score * 100))
```

    F1 score em Teste = 81.20%
    Acurácia em Teste = 90.11%
    

**5.4.2) Regressão Logística**

O próximo modelo será um método linear generalizado. A regressão logística é útil para modelar a probabilidade de um evento ocorrer em função de outros fatores.


```python
logisticReg = LogisticRegression()
logisticReg.fit(X_rus, y_rus)

preds = logisticReg.predict(X_test)
f_score = f1_score(preds, Y_test)
ac_score = accuracy_score(preds, Y_test)

print('F1 score em Teste = {:.2f}%'.format(f_score * 100))
print('Acurácia em Teste = {:.2f}%'.format(ac_score * 100))
```

    F1 score em Teste = 67.19%
    Acurácia em Teste = 80.08%
    

**5.4.3) Random Forest**

Agora vamos utilizar as árvores de devisão.


```python
randomForest = RandomForestClassifier()
randomForest.fit(X_rus, y_rus)

preds = randomForest.predict(X_test)
f_score = f1_score(preds, Y_test)
ac_score = accuracy_score(preds, Y_test)

print('F1 score em Teste = {:.2f}%'.format(f_score * 100))
print('Acurácia em Teste = {:.2f}%'.format(ac_score * 100))
```

    F1 score em Teste = 96.62%
    Acurácia em Teste = 98.40%
    

Nosso primeiro modelo base teve uma ótima performance aplicando o algoritmo de Random Forest, agora vamos selecionar as principais features para aplicar no nosso próximo modelo.

**5.4.4) Seleção das features mais importantes**


```python
df_temp = pd.DataFrame(randomForest.feature_importances_, index=x_encoded.columns, columns=['importance']).sort_values(by='importance', ascending=False)

display(df_temp)
colunas_importantes = list(df_temp.loc[df_temp['importance'] > 0.1].index)


plt.figure(figsize=(12,8))
sns.barplot(x=df_temp['importance'], y=df_temp.index)


```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nivel_satisfacao</th>
      <td>0.247995</td>
    </tr>
    <tr>
      <th>tempo_empresa</th>
      <td>0.236409</td>
    </tr>
    <tr>
      <th>horas_medias_por_mes</th>
      <td>0.165476</td>
    </tr>
    <tr>
      <th>numero_projetos</th>
      <td>0.157009</td>
    </tr>
    <tr>
      <th>ultima_avaliacao</th>
      <td>0.133104</td>
    </tr>
    <tr>
      <th>acidente_trabalho</th>
      <td>0.013833</td>
    </tr>
    <tr>
      <th>salario_low</th>
      <td>0.007538</td>
    </tr>
    <tr>
      <th>salario_high</th>
      <td>0.005639</td>
    </tr>
    <tr>
      <th>area_technical</th>
      <td>0.004825</td>
    </tr>
    <tr>
      <th>area_sales</th>
      <td>0.004483</td>
    </tr>
    <tr>
      <th>salario_medium</th>
      <td>0.004251</td>
    </tr>
    <tr>
      <th>area_support</th>
      <td>0.003968</td>
    </tr>
    <tr>
      <th>ultima_promocao_5anos</th>
      <td>0.002915</td>
    </tr>
    <tr>
      <th>area_RandD</th>
      <td>0.002359</td>
    </tr>
    <tr>
      <th>area_hr</th>
      <td>0.002284</td>
    </tr>
    <tr>
      <th>area_IT</th>
      <td>0.001923</td>
    </tr>
    <tr>
      <th>area_accounting</th>
      <td>0.001776</td>
    </tr>
    <tr>
      <th>area_management</th>
      <td>0.001676</td>
    </tr>
    <tr>
      <th>area_product_mng</th>
      <td>0.001307</td>
    </tr>
    <tr>
      <th>area_marketing</th>
      <td>0.001231</td>
    </tr>
  </tbody>
</table>
</div>





    <AxesSubplot:xlabel='importance'>




    
![png](assets/output_168_2.png)
    


**Considerações**:
O modelo considerou apenas 5 parâmetros como sendo muito importantes, são eles: `nivel_satisfacao`, `tempo_empresa`, `horas_medias_por_mes`, `numero_projetos`, `ultima_avaliacao`. Portanto, serão estes os parâmetros uqe iremos considerar para o próximo modelo.

---

<h2>6) MODELAGEM</h2>

Agora sim, iremos realmente criar o modelo final do projeto. Primeiro irei criar uma lista contendo várias técnicas e vários parâmetros. Todos estes parâmetros serão testados utilizando o GridSearch. Além disso, todos os algoritmos serão testados utilizando três técnicas diferentes de escala dos dados. O Normalizer, o Standard Scaler e o Robust Scaler.


```python
list_estimators = [
    {'name': 'Logistic Regression',
     'model': LogisticRegression(),
     'params': {
                'penalty':['l1', 'l2', 'elasticnet'],
                'fit_intercept': [True, False],
                'solver': ['newton-cg', 'lbfgs', 'liblinear']}
    },
    
    {
       'name': 'Naive Bayes',
       'model': GaussianNB(),
       'params': {}
    },

     {'name': 'KNN',
     'model': KNeighborsClassifier(),
     'params': {
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'p': [1, 2]
     }
    },

    {'name': 'SVM - Support Vector Machine',
     'model': SVC(),
     'params': {}
    },

    {'name': 'Random Forest',
     'model': RandomForestClassifier(),
     'params': {
            'n_estimators': [100],
            'bootstrap': [True, False], 
            'criterion': ['entropy'], 
            'min_samples_leaf': [1, 5, 10], 
            'min_samples_split': [10, 20, 30]
     }
    },

    {
       'name': 'XGB - Extreme Boosting Classifier',
       'model': XGBClassifier(),
       'params': {
          'eta': [0.3],
          'max_depth': [3],
          'gamma': [0, 1],
          'scale_pos_weight': [0, 2]
          #'alpha': [0.5, 1, 1.5],
          #'lambda': [0.5, 1, 1.5]
       }
    }
    
    ]
```


```python
x_axes = ['Não Deixou', 'Deixou']
y_axes =  ['Deixou', 'Não Deixou']

def conf_matrix(z, x=x_axes, y=y_axes):
    
    z = np.flip(z, 0)

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='ice')

    # add title
    fig.update_layout(
        title_text='<span><b>Confusion matrix</b></span>',
        xaxis = dict(title='Valores previstos'),
        yaxis = dict(title='Valores reais'),
        width=300,
        height=300
        )

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()
    #return fig
```

Abaixo eu crio uma função que automatiza todo o processo de treinamento, teste e avaliação dos modelos. A função utiliza a lista que definimos anteriormente com todos os modelos e todas os parâmetros.


```python
def autoModel(X_train, Y_train, X_test, Y_test):

    #Inicializando variáveis
    list_all_results = []
    list_standardization = [Normalizer(), StandardScaler(), RobustScaler()]

    best_acc = 0
    best_model = {
        'name': 'null',
        'model_grid': 'null',
        'best_params': 'null',
        'stand': 'null',
        'acc_train': 0,
        'acc_test': 0
    }

    #Para cada tipo de padronização iteramos
    for stand in list_standardization:

        #Para cada algoritmo na lista de algoritmos
        for estimator in list_estimators:
            
            #Criação do objeto GridSearch
            model_grid = GridSearchCV(estimator=estimator['model'], param_grid=estimator['params'], cv=10)
            
            #Criação do pipeline de execução
            pipe = Pipeline(steps=[('scale', stand), ('grid', model_grid)])
            pipe.fit(X_train, Y_train)

            #Realizando as previsões para cada dataset. Treino e teste
            preds_train = pipe.predict(X_train)
            preds_test = pipe.predict(X_test)
            
            #Otendo os resultados para cada métrica
            acc_train = accuracy_score(preds_train, Y_train)
            acc_test = accuracy_score(preds_test, Y_test)
            f1_train = f1_score(preds_train, Y_train)
            f1_test = f1_score(preds_test, Y_test)

            #Exibindo os resultados por acurácia, f1 score e matriz de confusão de treino e teste para cada algoritmo.
            print('Algoritmo: [{}]'.format(estimator['name'])) 
            print('stand:[{}] [ACCURACY] train:[{}%]  test:[{}%]'.format(stand, round(acc_train * 100, 2), round(acc_test * 100, 2)))
            print('stand:[{}] [F1 SCORE] train:[{}%]  test:[{}%]'.format(stand, round(f1_train * 100, 2), round(f1_test * 100, 2)))
            conf_matrix(confusion_matrix(preds_test, Y_test))

            #Armazenando sempre o melhor resultado
            if acc_test > best_model['acc_test']:
                best_model['name'] = estimator['name'],
                best_model['model_grid'] = pipe['grid'].estimator,
                best_model['best_params'] =  pipe[1].best_params_,
                best_model['stand'] = stand
                best_model['acc_train'] = acc_train
                best_model['acc_test'] = acc_test

            #Armazenando todos os resultados em uma lista para exibir em um dataframe final
            list_all_results.append(
                {'name': estimator['name'],
                'standardization': re.sub(r'[()]', '', str(stand)),
                'f1 score': round(f1_test * 100, 2)
                })

    return best_model, list_all_results
```

**Aplicando novamente a divisão em treino e teste e balanceamento do dataset** 


```python
#Selecionando somente as colunas que foram definidas anteriormente
X_transformed = x_encoded[colunas_importantes].values

#Divisão em treino e testes
X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, test_size = 0.3, random_state = 0)

#Balanceamento do dataset
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X_train, Y_train)
```


```python
#Treinando, testando e avaliando os algoritmos
best_model, list_all_results = autoModel(X_rus, y_rus, X_test, Y_test)
```

    Algoritmo: [Logistic Regression]
    stand:[Normalizer()] [ACCURACY] train:[82.18%]  test:[78.73%]
    stand:[Normalizer()] [F1 SCORE] train:[83.34%]  test:[66.28%]
    



    Algoritmo: [Naive Bayes]
    stand:[Normalizer()] [ACCURACY] train:[79.11%]  test:[70.94%]
    stand:[Normalizer()] [F1 SCORE] train:[81.67%]  test:[59.92%]
    



    Algoritmo: [KNN]
    stand:[Normalizer()] [ACCURACY] train:[100.0%]  test:[93.91%]
    stand:[Normalizer()] [F1 SCORE] train:[100.0%]  test:[88.18%]
    



    Algoritmo: [SVM - Support Vector Machine]
    stand:[Normalizer()] [ACCURACY] train:[64.67%]  test:[59.77%]
    stand:[Normalizer()] [F1 SCORE] train:[67.74%]  test:[46.76%]
    



    Algoritmo: [Random Forest]
    stand:[Normalizer()] [ACCURACY] train:[99.9%]  test:[97.38%]
    stand:[Normalizer()] [F1 SCORE] train:[99.9%]  test:[94.53%]
    



    Algoritmo: [XGB - Extreme Boosting Classifier]
    stand:[Normalizer()] [ACCURACY] train:[96.97%]  test:[93.37%]
    stand:[Normalizer()] [F1 SCORE] train:[97.01%]  test:[87.22%]
    



    Algoritmo: [Logistic Regression]
    stand:[StandardScaler()] [ACCURACY] train:[82.18%]  test:[79.14%]
    stand:[StandardScaler()] [F1 SCORE] train:[83.2%]  test:[66.4%]
    



    Algoritmo: [Naive Bayes]
    stand:[StandardScaler()] [ACCURACY] train:[79.62%]  test:[80.81%]
    stand:[StandardScaler()] [F1 SCORE] train:[79.07%]  test:[65.46%]
    



    Algoritmo: [KNN]
    stand:[StandardScaler()] [ACCURACY] train:[100.0%]  test:[96.06%]
    stand:[StandardScaler()] [F1 SCORE] train:[100.0%]  test:[92.02%]
    



    Algoritmo: [SVM - Support Vector Machine]
    stand:[StandardScaler()] [ACCURACY] train:[94.87%]  test:[95.74%]
    stand:[StandardScaler()] [F1 SCORE] train:[94.72%]  test:[90.96%]
    



    Algoritmo: [Random Forest]
    stand:[StandardScaler()] [ACCURACY] train:[99.43%]  test:[98.54%]
    stand:[StandardScaler()] [F1 SCORE] train:[99.43%]  test:[96.88%]
    



    Algoritmo: [XGB - Extreme Boosting Classifier]
    stand:[StandardScaler()] [ACCURACY] train:[97.07%]  test:[95.35%]
    stand:[StandardScaler()] [F1 SCORE] train:[97.06%]  test:[90.6%]
    



    Algoritmo: [Logistic Regression]
    stand:[RobustScaler()] [ACCURACY] train:[81.86%]  test:[79.58%]
    stand:[RobustScaler()] [F1 SCORE] train:[82.72%]  test:[66.64%]
    



    Algoritmo: [Naive Bayes]
    stand:[RobustScaler()] [ACCURACY] train:[79.62%]  test:[80.81%]
    stand:[RobustScaler()] [F1 SCORE] train:[79.07%]  test:[65.46%]
    



    Algoritmo: [KNN]
    stand:[RobustScaler()] [ACCURACY] train:[100.0%]  test:[95.56%]
    stand:[RobustScaler()] [F1 SCORE] train:[100.0%]  test:[91.11%]
    



    Algoritmo: [SVM - Support Vector Machine]
    stand:[RobustScaler()] [ACCURACY] train:[94.81%]  test:[95.26%]
    stand:[RobustScaler()] [F1 SCORE] train:[94.67%]  test:[90.08%]
    



    Algoritmo: [Random Forest]
    stand:[RobustScaler()] [ACCURACY] train:[99.45%]  test:[98.68%]
    stand:[RobustScaler()] [F1 SCORE] train:[99.45%]  test:[97.18%]
    



    Algoritmo: [XGB - Extreme Boosting Classifier]
    stand:[RobustScaler()] [ACCURACY] train:[97.07%]  test:[95.35%]
    stand:[RobustScaler()] [F1 SCORE] train:[97.06%]  test:[90.6%]
    



<h3>6.1) Avaliando os resultados</h3>

Agora que temos a construção de todos os modelos vamos avaliar os resultados. O método que eu criei tem dois retornos, o primeiro contém o melhor modelo com todos os melhores parâmetros, o segundo é uma lista que contém o nome do algoritmo, o tipo de padronização e a pontuação da avaliação utilizando o f1 score. <br/>

Vamos exibir essas informações em um dataframe ordenado do melhor resultado para o pior para comparar os resultados:


```python
df_results = pd.DataFrame(list_all_results)
df_results = df_results.sort_values(by='f1 score', ascending=False)
display(df_results)
```


<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>standardization</th>
      <th>f1 score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>Random Forest</td>
      <td>RobustScaler</td>
      <td>97.18</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Random Forest</td>
      <td>StandardScaler</td>
      <td>96.88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Random Forest</td>
      <td>Normalizer</td>
      <td>94.53</td>
    </tr>
    <tr>
      <th>8</th>
      <td>KNN</td>
      <td>StandardScaler</td>
      <td>92.02</td>
    </tr>
    <tr>
      <th>14</th>
      <td>KNN</td>
      <td>RobustScaler</td>
      <td>91.11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SVM - Support Vector Machine</td>
      <td>StandardScaler</td>
      <td>90.96</td>
    </tr>
    <tr>
      <th>11</th>
      <td>XGB - Extreme Boosting Classifier</td>
      <td>StandardScaler</td>
      <td>90.60</td>
    </tr>
    <tr>
      <th>17</th>
      <td>XGB - Extreme Boosting Classifier</td>
      <td>RobustScaler</td>
      <td>90.60</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SVM - Support Vector Machine</td>
      <td>RobustScaler</td>
      <td>90.08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>Normalizer</td>
      <td>88.18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>XGB - Extreme Boosting Classifier</td>
      <td>Normalizer</td>
      <td>87.22</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Logistic Regression</td>
      <td>RobustScaler</td>
      <td>66.64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Logistic Regression</td>
      <td>StandardScaler</td>
      <td>66.40</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>Normalizer</td>
      <td>66.28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Naive Bayes</td>
      <td>StandardScaler</td>
      <td>65.46</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Naive Bayes</td>
      <td>RobustScaler</td>
      <td>65.46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Naive Bayes</td>
      <td>Normalizer</td>
      <td>59.92</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVM - Support Vector Machine</td>
      <td>Normalizer</td>
      <td>46.76</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_results.to_excel('resultados_algoritmos.xlsx', index=False)
```

Vemos que o melhor modelo acabou sendo a RandomForest, com um score de 97.17%. Com isso, podemos dizer que com os dados que temos, podemos prever 97% dos casos se o colaborador vai deixar ou não a empresa.

---
<h2>7) EXPORTANDO O MODELO</h2>

Após todos estes passos, vamos exportar o nosso modelo para que ele possa ser usado por outros sistemas.

**Obtendo o melhor modelo e treinando com todos os dados**


```python
#Selecionando somente as colunas que foram definidas anteriormente
X_transformed = x_encoded[colunas_importantes].values

#Balanceamento do dataset
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X_transformed, Y)
```


```python
#Selecionando o melhor algoritmo e seus melhores parâmetros
estimator = best_model['model_grid'][0]
params = best_model['best_params'][0].copy()

#Seleção da melhor padronização
stand = best_model['stand']

#Convertendo para o formato que o GridSearch aceita. Formato de lista.
for key in params:
    lst = [params[key]]
    params[key] = lst

    
#Criação do modelo GridSearch final
final_model = GridSearchCV(estimator=estimator, param_grid=params, cv=10)

#Criação do pipeline de execução
pipe = Pipeline(steps=[('scale', stand), ('grid', final_model)])
```


```python
#Treinamento do modelo final
pipe.fit(X_rus, y_rus)
preds = pipe.predict(X_transformed)

#Avaliando o resultado final
f1_test = f1_score(preds, Y)
print('O F1 SCORE do modelo final foi de: {}%'.format(round(f1_test * 100, 2)))
```

    O F1 SCORE do modelo final foi de: 98.72%
    

Vale ressaltar que essa avaliação final não deve ser levada ao pé da letra, pois realimos treino e teste com os mesmos dados.

**Exportando o modelo**


```python
import joblib
import time
```


```python
timestr = time.strftime("%Y%m%d-%H%M%S")

# Salva o modelo em disco
joblib.dump(pipe, f'modelo/modelo-{timestr}.pkl')
```




    ['modelo/modelo-20220728-223256.pkl']



<h2>8) CONSIDERAÇÕES SOBRE OS RESULTADOS OBTIDOS</h2>

<h3>Análise Descritiva</h3>

Na análise descritiva, observamos que:
* A empresa possui nota 6 em média de nível de satisfação dentre todos os colaboradores;
* A última avaliação obteve nota 7 em média, indicando uma melhora em relação ao nível de satisfação;
    * Além disso, na última avaliação a empresa não teve avaliações consideradas péssimas e apenas 1.57% dos colaboradores avaliaram como ruim.
* Os colaboradores ficam em média 3.5 anos na empresa até deixarem a mesma;
* 14% dos colaboradores já sofreram acidentes de trabalho.
    * Embora este atributo não tenha impacto sobre o evento de o colaborador deixar ou não a empresa, é um dado bastante elevado e alarmante.

Analisando a matriz de correlação e a seleção de atributos com o Random Forest, notamos que o principal atributo que impacata no evento de o colaborador deixar ou não a empresa é o `nível de satisfação`.

Dito isso, temos que:
* Quanto pior o nível de satisfação, mais o colaborador tende a deixar a empresa;
* 9.85% do total de colaboradores estão insatisfeitos e 20% possuem avaliação negativa;
* A quantidade de horas trabalhadas por mês afeta diretamente na satisfação dos colaboradores:
   * **Satisfação péssima**: São os colaboradores que trabalham muitas horas semanais, são os trabalhadores que fazem muita hora extra no serviço;
   * **Satisfação baixa**: Se concentra mais nos colaboradores que trabalham poucas horas por semana;
   * **Satisfação média**: Colaboradores que trabalham em média 40 horas semanais;
   * **Satisfação boa e ótima**: Se concentra principalmente nos trabalhadores que tem a jornada semanal de 44~ horas em média.

* Ao analisar o nível de insatisfação por área, vemos que a maioria das áreas ultrapassam os 20% de insatisfação dos colaboradores;
* Há também uma relação direta entre categoria salarial e insatisfação dos colaboradores;
    * A categoria salarial impacta diretamente no evento do colaborador deixar ou não a empresa;

<h3>Análise Preditiva</h3>

Para a realização deste projeto, eu utilizei vários algoritmos, mas o que se sobressaiu foi o Random Forest com uma precisão fina de 97%. 

Aplicando o feature selection utilizando o próprio Random Forest, obtive que os principais atributos para o modelo foram: `nivel_satisfacao`, `tempo_empresa`, `horas_medias_por_mes`, `numero_projetos` e `ultima_avaliacao`.

---

<h3>Agradecimentos</h3>

Quem chegou até fica os meus agradecimentos, qualquer dúvida estou a disposição no email: **`krupck@outlook.com`**.

Atenciosamente, 

Henrique K. Secchi.

Abraços.

<h2>FIM</h2>
