# Utilizando Machine Learning para prever os jogos do Campeonato Brasileiro de futebol

O intuito deste projeto é desenvolver um modelo preditivo capaz de prever o resultado das partidas do campeonato brasileiro de futebol. 
O modelo tem como saída os seguintes resultados:
* 1: Vitória da equipe mandante;
* 0: Empate;
* -1: Vitória da equipe visitante

O dataset foi obtido do site Football-Data. O Football-Data é um portal de apostas de futebol gratuito que fornece resultados históricos e probabilidades para ajudar os entusiastas de apostas de futebol a analisar muitos anos de dados de forma rápida e eficiente.
> Link disponível em: https://www.football-data.co.uk/brazil.php

A organização deste notebook está dividida em:
* 1) Análise exploratória dos dados;
* 2) Limpeza e pré-processamento dos dados;
* 3) Modelagem;
* 4) Aplicação dos algoritmos de Machine Learning;
* 5) Utilizando o modelo para prever resultados do campeonato
* 6) Considerações finais
<br/><br/>

**Considerações sobre os resultados obtidos:**
<br/>
O modelo final obteve uma acurácia de 52.11%. O resultado pode ser considerado razoável, dado a dificuldade que é realizar esse tipo de previsão. Pontos positivos e negativos do modelo:
<br/>
Como positivo podemos considerar:
* O modelo consegue dizer de maneira geral quais são os 'níveis' das equipes. Por exemplo: Ele consegue dizer que Flamego RJ, Atlético MG e Palmeiras são as melhores equipes, tanto em 2020 quanto em 2021. E quando olhamos para o passado, em 2013 o Cruzeiro e Corinthians estavam entre os melhores. O que se confirma se olharmos a tabela daquele ano.
* Além disso podemos notar que o modelo consegue dizer com precisão quaise são os clubes que lutam para não cair. Exemplo de 2020 e 2021, onde o modelo acertou 5 dos 8 clubes que foram rebaixados nestes dois anos, e quase acertou mais dois. Botafogo RJ em 2020 e Juventude em 2021.
* O modelo foi bom em dizer quais foram as equipes que figuraram entre os 10 melhores do Brasil, acertando em 7 dos 10 casos.
<br/><br/>

Como avaliação negativa do modelo podemos considerar:
* O modelo não foi bom em dizer exatamente qual a posição que a equipe terminou o campeonato.
* O modelo também previu poucos empates, o que foi bastante negativo, dado que as equipes de meio de tabela empatam bastante seus jogos.

**Vamos começar importando as bibliotecas e carregando os dados**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import warnings
from matplotlib import style
```

Esta célula são configurações sobre os estilos que serão utilizados neste projeto.


```python
sns.set_theme()
# Criação de um array contendo cores no formato hexadecimal
colors = ['#fffb2d', '#23ad2a','#ebebeb', '#36469c']
# Setando o palette de cores personalizadas
customPalette = sns.set_palette(sns.color_palette(colors))
#Mapeamento de estilos que serão utilizados para esse projeto
style.use('estilos/style_brasil.mplstyle')
plt.rcParams["figure.figsize"] = (14,8)

```


```python
data = pd.read_csv('data/BRA.csv')
```

### 1) Análise exploratória dos dados

A primeira análise a ser feita é uma análise mais com teor de ver informações gerais mais superficiais sobre os dados. Informações sobre os tipos de dados, sobre as colunas, a quantidade de registros, distribuição dos dados, dentre outras informações.


```python
data.dtypes
```




    Country     object
    League      object
    Season       int64
    Date        object
    Time        object
    Home        object
    Away        object
    HG         float64
    AG         float64
    Res         object
    PH         float64
    PD         float64
    PA         float64
    MaxH       float64
    MaxD       float64
    MaxA       float64
    AvgH       float64
    AvgD       float64
    AvgA       float64
    dtype: object




```python
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>League</th>
      <th>Season</th>
      <th>Date</th>
      <th>Time</th>
      <th>Home</th>
      <th>Away</th>
      <th>HG</th>
      <th>AG</th>
      <th>Res</th>
      <th>PH</th>
      <th>PD</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brazil</td>
      <td>Serie A</td>
      <td>2012</td>
      <td>19/05/2012</td>
      <td>22:30</td>
      <td>Palmeiras</td>
      <td>Portuguesa</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>D</td>
      <td>1.75</td>
      <td>3.86</td>
      <td>5.25</td>
      <td>1.76</td>
      <td>3.87</td>
      <td>5.31</td>
      <td>1.69</td>
      <td>3.50</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brazil</td>
      <td>Serie A</td>
      <td>2012</td>
      <td>19/05/2012</td>
      <td>22:30</td>
      <td>Sport Recife</td>
      <td>Flamengo RJ</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>D</td>
      <td>2.83</td>
      <td>3.39</td>
      <td>2.68</td>
      <td>2.83</td>
      <td>3.42</td>
      <td>2.70</td>
      <td>2.59</td>
      <td>3.23</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Serie A</td>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>01:00</td>
      <td>Figueirense</td>
      <td>Nautico</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>H</td>
      <td>1.60</td>
      <td>4.04</td>
      <td>6.72</td>
      <td>1.67</td>
      <td>4.05</td>
      <td>7.22</td>
      <td>1.59</td>
      <td>3.67</td>
      <td>5.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>Serie A</td>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>20:00</td>
      <td>Botafogo RJ</td>
      <td>Sao Paulo</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>H</td>
      <td>2.49</td>
      <td>3.35</td>
      <td>3.15</td>
      <td>2.49</td>
      <td>3.39</td>
      <td>3.15</td>
      <td>2.35</td>
      <td>3.26</td>
      <td>2.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>Serie A</td>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>20:00</td>
      <td>Corinthians</td>
      <td>Fluminense</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>A</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.89</td>
      <td>3.33</td>
      <td>3.89</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (3849, 19)




```python
data.columns
```




    Index(['Country', 'League', 'Season', 'Date', 'Time', 'Home', 'Away', 'HG',
           'AG', 'Res', 'PH', 'PD', 'PA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD',
           'AvgA'],
          dtype='object')



De início vemos que o dataset possui algumas informações irrelevantes e outras que devem ser retiradas. O dataset possui 19 colunas, sendo RES o atributo target.
* Country: País (Irrelevante, pois todos os jogos ocorrem no mesmo país)
* League: Campeonato (Irrelevante, pois todos os jogos são do mesmo campeonato)
* Season: Temporada que ocorreu a partida.
* Date: Data que ocorreu a partida
* Time: Horário do jogo
* Home: Clube mandante
* Away: Clube visitante
* HG: Home Gols - Gols do clube mandante durante o tempo regulamentado. (Deve ser retirada, pois não é uma informação que se tem de antemão, e sim somente depois que o jogo acontece).
* AG: Away Gols - Gols do clube visitante durante o tempo regulamentado. (Deve ser retirada, pois não é uma informação que se tem de antemão, e sim somente depois que o jogo acontece).

Informações sobre os sites de apostas:
* PH: Probabilidade de vitória do clube mandante segundo a Pinnacle.
* PD: Probabilidade de empate segundo a Pinnacle.
* PA: Probabilidade de vitória do clube visitante segundo a Pinnacle.
* MaxH: Probabilidades máxima de vitória do clube mandante segundo o mercado.
* MaxD: Probabilidades máxima de empata segundo o mercado.
* MaxA: Probabilidades máxima de vitória do clube vistante segundo o mercado.
* AvgH: Probabilidades média de vitória do clube mandante segundo o mercado.
* AvgD: Probabilidades média de empate segundo o mercado.
* AvgA: Probabilidades média de vitória do clube visitante segundo o mercado.

* RES: Resultado (H - vitória do clube mandante; D - Empate; A - Vitória do clube visitante).


```python
data.isnull().sum() #O dataset possui alguns valores missing que devemos tratar.
```




    Country    0
    League     0
    Season     0
    Date       0
    Time       0
    Home       0
    Away       0
    HG         1
    AG         1
    Res        1
    PH         1
    PD         1
    PA         1
    MaxH       0
    MaxD       0
    MaxA       0
    AvgH       0
    AvgD       0
    AvgA       0
    dtype: int64




```python
data[data.duplicated()] #Não há dados duplicados
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>League</th>
      <th>Season</th>
      <th>Date</th>
      <th>Time</th>
      <th>Home</th>
      <th>Away</th>
      <th>HG</th>
      <th>AG</th>
      <th>Res</th>
      <th>PH</th>
      <th>PD</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>HG</th>
      <th>AG</th>
      <th>PH</th>
      <th>PD</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3849.000000</td>
      <td>3848.000000</td>
      <td>3848.000000</td>
      <td>3848.000000</td>
      <td>3848.000000</td>
      <td>3848.000000</td>
      <td>3849.000000</td>
      <td>3849.000000</td>
      <td>3849.000000</td>
      <td>3849.000000</td>
      <td>3849.000000</td>
      <td>3849.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2016.570018</td>
      <td>1.414501</td>
      <td>0.943867</td>
      <td>2.300218</td>
      <td>3.657747</td>
      <td>4.708545</td>
      <td>2.380688</td>
      <td>3.760483</td>
      <td>4.969028</td>
      <td>2.222055</td>
      <td>3.484100</td>
      <td>4.334679</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.920169</td>
      <td>1.157684</td>
      <td>0.973776</td>
      <td>0.957730</td>
      <td>0.651623</td>
      <td>2.504508</td>
      <td>1.076403</td>
      <td>0.678611</td>
      <td>2.800519</td>
      <td>0.893556</td>
      <td>0.562054</td>
      <td>2.137199</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2012.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.080000</td>
      <td>2.540000</td>
      <td>1.150000</td>
      <td>1.090000</td>
      <td>2.630000</td>
      <td>0.000000</td>
      <td>1.070000</td>
      <td>2.510000</td>
      <td>1.150000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2014.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.690000</td>
      <td>3.260000</td>
      <td>2.990000</td>
      <td>1.740000</td>
      <td>3.350000</td>
      <td>3.070000</td>
      <td>1.650000</td>
      <td>3.150000</td>
      <td>2.860000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2017.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.080000</td>
      <td>3.460000</td>
      <td>4.020000</td>
      <td>2.150000</td>
      <td>3.540000</td>
      <td>4.160000</td>
      <td>2.030000</td>
      <td>3.300000</td>
      <td>3.740000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2019.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.600000</td>
      <td>3.860000</td>
      <td>5.760000</td>
      <td>2.700000</td>
      <td>3.970000</td>
      <td>6.090000</td>
      <td>2.510000</td>
      <td>3.660000</td>
      <td>5.310000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2022.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>17.680000</td>
      <td>12.250000</td>
      <td>29.580000</td>
      <td>29.000000</td>
      <td>12.500000</td>
      <td>36.000000</td>
      <td>20.230000</td>
      <td>10.820000</td>
      <td>26.580000</td>
    </tr>
  </tbody>
</table>
</div>



Com essa descrição dos dados podemos perceber algumas coisas:
* A média de gols da equipe mandante é bem maior do que a média de gols da equipe visitante. Isso significa que muito provavelmente o fator jogar em casa é determinante para a vitória da equipe;
* De maneira geral os atributos relacionados a probabilidades (PH, PD, PA, MaxH, MaxD, MaxA, AvgH, AvgD, AvgA) estão bem distribuídos, pois as médias se aproximam bastante das suas respectivas medianas.
* A diferença entre valores mínimos e valores máximos é bastante alta, indício de que teremos aplicar a normalização destes dados.


```python
#Biblioteca muito interessante para visualizar informações geradas automaticamente.
warnings.filterwarnings('ignore')
my_report = sv.analyze(data)
my_report.show_html()
```

    Done! Use 'show' commands to display/save.   |██████████| [100%]   00:00 -> (00:00 left)
    

    Report SWEETVIZ_REPORT.html was generated! NOTEBOOK/COLAB USERS: the web browser MAY not pop up, regardless, the report IS saved in your notebook/colab files.
    

**Proporção de vitórias, empates e derrotas em casa**<br/>
Ao analisar as vitórias, é possível observar que em quase metade das partidas o clube mandante se saiu vencedor, sendo o fator jogar em casa muito importante no contexto dos dados. 


```python
res = pd.DataFrame(round(data['Res'].value_counts() / data['Res'].count() * 100, 2))
res.rename(columns={'Res': 'Porcentagem'}, inplace = True)
res.reset_index(inplace=True)
res
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Porcentagem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>H</td>
      <td>48.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D</td>
      <td>26.95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>24.06</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(3,3))
ax = sns.barplot(x="index", y="Porcentagem", data=res)
```


    
![png](PrevisorFutebol_files/PrevisorFutebol_21_0.png)
    


**Quais clubes mais jogaram?**


```python
home = data['Home'].value_counts().to_frame()
away = data['Away'].value_counts().to_frame()
```


```python
home['Away_qtd'] = away
total = home
```


```python
total.reset_index(inplace=True)
total.rename(columns={'index': 'Team', 'home_team': 'home_qtd'}, inplace = True)
```


```python
total['Away_qtd'].fillna(0, inplace=True)
```


```python
# Estes são os clubes que mais jogaram jogos como mandantes.
total.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Home</th>
      <th>Away_qtd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Flamengo RJ</td>
      <td>193</td>
      <td>192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Santos</td>
      <td>193</td>
      <td>192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Atletico-MG</td>
      <td>193</td>
      <td>192</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fluminense</td>
      <td>192</td>
      <td>193</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sao Paulo</td>
      <td>192</td>
      <td>193</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Corinthians</td>
      <td>192</td>
      <td>193</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gremio</td>
      <td>190</td>
      <td>190</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Palmeiras</td>
      <td>174</td>
      <td>173</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Internacional</td>
      <td>173</td>
      <td>174</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Botafogo RJ</td>
      <td>154</td>
      <td>155</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Estes são os clubes que mais jogaram jogos como visitantes na história
total.sort_values(by='Away_qtd', ascending=False).head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Home</th>
      <th>Away_qtd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Sao Paulo</td>
      <td>192</td>
      <td>193</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Corinthians</td>
      <td>192</td>
      <td>193</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fluminense</td>
      <td>192</td>
      <td>193</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Flamengo RJ</td>
      <td>193</td>
      <td>192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Santos</td>
      <td>193</td>
      <td>192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Atletico-MG</td>
      <td>193</td>
      <td>192</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gremio</td>
      <td>190</td>
      <td>190</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Internacional</td>
      <td>173</td>
      <td>174</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Palmeiras</td>
      <td>174</td>
      <td>173</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Botafogo RJ</td>
      <td>154</td>
      <td>155</td>
    </tr>
  </tbody>
</table>
</div>




```python
total['Total'] = total.loc[total['Home'] >= 0,['Home','Away_qtd']].sum(axis=1)
#Estes são os clubes que mais jogaram jogos na história
total.sort_values(by='Total', ascending=False).head(15)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Home</th>
      <th>Away_qtd</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Flamengo RJ</td>
      <td>193</td>
      <td>192</td>
      <td>385</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sao Paulo</td>
      <td>192</td>
      <td>193</td>
      <td>385</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Corinthians</td>
      <td>192</td>
      <td>193</td>
      <td>385</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Santos</td>
      <td>193</td>
      <td>192</td>
      <td>385</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fluminense</td>
      <td>192</td>
      <td>193</td>
      <td>385</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Atletico-MG</td>
      <td>193</td>
      <td>192</td>
      <td>385</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gremio</td>
      <td>190</td>
      <td>190</td>
      <td>380</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Palmeiras</td>
      <td>174</td>
      <td>173</td>
      <td>347</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Internacional</td>
      <td>173</td>
      <td>174</td>
      <td>347</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Botafogo RJ</td>
      <td>154</td>
      <td>155</td>
      <td>309</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cruzeiro</td>
      <td>152</td>
      <td>152</td>
      <td>304</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Bahia</td>
      <td>152</td>
      <td>152</td>
      <td>304</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sport Recife</td>
      <td>152</td>
      <td>152</td>
      <td>304</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Coritiba</td>
      <td>135</td>
      <td>136</td>
      <td>271</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Chapecoense-SC</td>
      <td>133</td>
      <td>133</td>
      <td>266</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = sns.barplot(y=total['Team'], x=total['Total'], color=colors[1])
plt.title('Quantidade total de jogos por equipe')
plt.show()
```


    
![png](PrevisorFutebol_files/PrevisorFutebol_30_0.png)
    



```python
data.groupby('Res')['Home', 'Away'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">Home</th>
      <th colspan="4" halign="left">Away</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>Res</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>926</td>
      <td>37</td>
      <td>Botafogo RJ</td>
      <td>48</td>
      <td>926</td>
      <td>37</td>
      <td>Flamengo RJ</td>
      <td>68</td>
    </tr>
    <tr>
      <th>D</th>
      <td>1037</td>
      <td>37</td>
      <td>Corinthians</td>
      <td>62</td>
      <td>1037</td>
      <td>37</td>
      <td>Atletico-MG</td>
      <td>61</td>
    </tr>
    <tr>
      <th>H</th>
      <td>1885</td>
      <td>37</td>
      <td>Atletico-MG</td>
      <td>126</td>
      <td>1885</td>
      <td>37</td>
      <td>Fluminense</td>
      <td>92</td>
    </tr>
  </tbody>
</table>
</div>



Aqui vemos informações importantes e relevantes que com certeza impactarão no resultado do modelo preditivo:
* Atlético-MG é o clube com mais vitórias jogando como mandante os seus jogos, enquanto que o Flamengo RJ é o clube com mais vitórias jogando como visitante.
* Corinthians é o clube com mais empates jogando como mandante, já o Atlético-MG é o clube com mais empates jogando como visitante.
* O Botafogo RJ é a equipe que mais perde jogos sendo mandante no campeonato, enquanto que o Fluminense é a equipe que mais perde jogos sendo visitante. Curiosidade que os dois clubes que mais perdem jogos são do estado do Rio de Janeiro. 


```python
maiorHome = data.loc[data['Res'] == 'H', ['Home']]
sns.barplot(x=maiorHome['Home'].value_counts().values, y=maiorHome['Home'].value_counts().index, color=colors[0])
plt.title('Maiores vencedores como mandantes')
```




    Text(0.5, 1.0, 'Maiores vencedores como mandantes')




    
![png](PrevisorFutebol_files/PrevisorFutebol_33_1.png)
    



```python
maiorAway = data.loc[data['Res'] == 'A', ['Away']]
sns.barplot(x=maiorAway['Away'].value_counts().values, y=maiorAway['Away'].value_counts().index, color=colors[3])
plt.title('Maiores vencedores como visitantes')
```




    Text(0.5, 1.0, 'Maiores vencedores como visitantes')




    
![png](PrevisorFutebol_files/PrevisorFutebol_34_1.png)
    


### 2) Limpeza e Pré processamento dos dados

**Dropando colunas desnecessárias**<br/>
Como dito anteriormente, alguns atributos são irrelevantes para o modelo, enquanto que outros são impossíveis de se obter antes do jogo acontecer, então iremos descartá-los.


```python
colunas = ['Country', 'League', 'HG', 'AG']
data.drop(columns=colunas, inplace=True)
```


```python
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Date</th>
      <th>Time</th>
      <th>Home</th>
      <th>Away</th>
      <th>Res</th>
      <th>PH</th>
      <th>PD</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>19/05/2012</td>
      <td>22:30</td>
      <td>Palmeiras</td>
      <td>Portuguesa</td>
      <td>D</td>
      <td>1.75</td>
      <td>3.86</td>
      <td>5.25</td>
      <td>1.76</td>
      <td>3.87</td>
      <td>5.31</td>
      <td>1.69</td>
      <td>3.50</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>19/05/2012</td>
      <td>22:30</td>
      <td>Sport Recife</td>
      <td>Flamengo RJ</td>
      <td>D</td>
      <td>2.83</td>
      <td>3.39</td>
      <td>2.68</td>
      <td>2.83</td>
      <td>3.42</td>
      <td>2.70</td>
      <td>2.59</td>
      <td>3.23</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>01:00</td>
      <td>Figueirense</td>
      <td>Nautico</td>
      <td>H</td>
      <td>1.60</td>
      <td>4.04</td>
      <td>6.72</td>
      <td>1.67</td>
      <td>4.05</td>
      <td>7.22</td>
      <td>1.59</td>
      <td>3.67</td>
      <td>5.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>20:00</td>
      <td>Botafogo RJ</td>
      <td>Sao Paulo</td>
      <td>H</td>
      <td>2.49</td>
      <td>3.35</td>
      <td>3.15</td>
      <td>2.49</td>
      <td>3.39</td>
      <td>3.15</td>
      <td>2.35</td>
      <td>3.26</td>
      <td>2.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>20:00</td>
      <td>Corinthians</td>
      <td>Fluminense</td>
      <td>A</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.89</td>
      <td>3.33</td>
      <td>3.89</td>
    </tr>
  </tbody>
</table>
</div>



**Tratando valores missing**


```python
data.isnull().sum()
```




    Season    0
    Date      0
    Time      0
    Home      0
    Away      0
    Res       1
    PH        1
    PD        1
    PA        1
    MaxH      0
    MaxD      0
    MaxA      0
    AvgH      0
    AvgD      0
    AvgA      0
    dtype: int64



O dataset possui apenas um registro missing. Na verdade a partida que não possui os dados foi o jogo entre Atlético-MG e Chapecoense-SC. Esta partida foi cancelada, pois foi na época que aconteceu o acidente que envolveu os jogadores do clube de Santa Catarina.
A melhor solução será dropar esse registro, já que se trata de uma partida que não aconteceu.


```python
#A partida entre Chapecoense-SC e Atletico-MG foi cancelada,
#Foi o ano que houve o acidente envolvendo os jogadores do clube de SC
##Vamos dropar esse registro
data.loc[data.PH.isnull() == True]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Date</th>
      <th>Time</th>
      <th>Home</th>
      <th>Away</th>
      <th>Res</th>
      <th>PH</th>
      <th>PD</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1891</th>
      <td>2016</td>
      <td>11/12/2016</td>
      <td>19:00</td>
      <td>Chapecoense-SC</td>
      <td>Atletico-MG</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.85</td>
      <td>3.3</td>
      <td>2.68</td>
      <td>2.85</td>
      <td>3.3</td>
      <td>2.67</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dropna(inplace=True)
```

**Mapeando os dados categóricos**<br/>
A grande maioria dos algoritmos de Machine Learning não preveem e nem aceitam dados categóricos, portanto devemos tratar esses dados de alguma forma.<br/>
Vamos mapear primeiro a variável target, transformar o valores categóricos em numéricos. Portanto a saída ficará assim:
* 1: Vitória da equipe mandante;
* 0: Empate;
* -1: Vitória da equipe visitante.


```python
res_map = {'H': 1,
           'D': 0,
           'A': -1}

data['Res'] = data['Res'].map(res_map)
```

Criação de uma função que faz o mapeamento automático do atributo passado como parâmetro


```python
#Criando um mapeamento de maneira automática
def criar_dict(df, coluna):
    temp = pd.DataFrame(df[coluna].value_counts()).sort_index()
    temp['cont'] = range(1, len(temp) + 1)
    temp = temp.drop(columns=coluna)
    dic = temp.to_dict()
    return dic['cont']

map_teams = criar_dict(data, 'Home')
```

Aqui eu decidi mapear os atributos Home e Away, uma vez que são atributos categóricos e devemos tratar de alguma forma.


```python
data['Home_categorical'] = data['Home'].map(map_teams)
data['Away_categorical'] = data['Away'].map(map_teams)
```

**Vamos categorizar os atributos Time e Date seguindo a mesma lógica dos atributos Home e Away**<br/>
Tavez esses atributos da forma como foram mapeados mais atrapalhem do que ajudem, mas vamos testar várias combinações para analisar o impacto que esse tipo de mapeamento causará no modelo.


```python
map_time = criar_dict(data, 'Time')
map_date = criar_dict(data, 'Date')
```


```python
data['Time_categorical'] = data['Time'].map(map_time)
data['Date_categorical'] = data['Date'].map(map_date)
```


```python
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Date</th>
      <th>Time</th>
      <th>Home</th>
      <th>Away</th>
      <th>Res</th>
      <th>PH</th>
      <th>PD</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
      <th>Home_categorical</th>
      <th>Away_categorical</th>
      <th>Time_categorical</th>
      <th>Date_categorical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>19/05/2012</td>
      <td>22:30</td>
      <td>Palmeiras</td>
      <td>Portuguesa</td>
      <td>0</td>
      <td>1.75</td>
      <td>3.86</td>
      <td>5.25</td>
      <td>1.76</td>
      <td>3.87</td>
      <td>5.31</td>
      <td>1.69</td>
      <td>3.50</td>
      <td>4.90</td>
      <td>28</td>
      <td>31</td>
      <td>32</td>
      <td>646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>19/05/2012</td>
      <td>22:30</td>
      <td>Sport Recife</td>
      <td>Flamengo RJ</td>
      <td>0</td>
      <td>2.83</td>
      <td>3.39</td>
      <td>2.68</td>
      <td>2.83</td>
      <td>3.42</td>
      <td>2.70</td>
      <td>2.59</td>
      <td>3.23</td>
      <td>2.58</td>
      <td>35</td>
      <td>19</td>
      <td>32</td>
      <td>646</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>01:00</td>
      <td>Figueirense</td>
      <td>Nautico</td>
      <td>1</td>
      <td>1.60</td>
      <td>4.04</td>
      <td>6.72</td>
      <td>1.67</td>
      <td>4.05</td>
      <td>7.22</td>
      <td>1.59</td>
      <td>3.67</td>
      <td>5.64</td>
      <td>18</td>
      <td>27</td>
      <td>9</td>
      <td>686</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>20:00</td>
      <td>Botafogo RJ</td>
      <td>Sao Paulo</td>
      <td>1</td>
      <td>2.49</td>
      <td>3.35</td>
      <td>3.15</td>
      <td>2.49</td>
      <td>3.39</td>
      <td>3.15</td>
      <td>2.35</td>
      <td>3.26</td>
      <td>2.84</td>
      <td>8</td>
      <td>34</td>
      <td>22</td>
      <td>686</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>20:00</td>
      <td>Corinthians</td>
      <td>Fluminense</td>
      <td>-1</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.89</td>
      <td>3.33</td>
      <td>3.89</td>
      <td>13</td>
      <td>20</td>
      <td>22</td>
      <td>686</td>
    </tr>
  </tbody>
</table>
</div>



**One Hot Enconding**<br/>
Um ponto muito importante que deve ser levado em consideração ao transformar variáveis categóricas em variáveis numéricas é tomar cuidado para o significado dos dados não mude.<br/>
Por exemplo: Ao mapear o atributo Home, como sendo: 1-Flamengo, 2-Palmeiras, 3-Atlético-MG, 4-Corinthians, etc.. acontece de que a maioria dos algoritmos de Machine Learning vão entender estes dados como sendo sequenciais e ordenados, afetando negativamente o modelo.<br/>
Por isso decidi criar um novo dataset aplicando o one-hot-enconding nos atributos Home e Away.


```python
#Função que aplica o one hot enconding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def encoder(df, col):
    X = df.values
    valores_Pclass = list(df[col].sort_values().unique())
    colunas = []
    for i in valores_Pclass:
        colunas.append(col + '_' + str(i))
    colunas = colunas + list(df.columns)
    
    labelencoder_Pclass = LabelEncoder()
    X[:, 0] = labelencoder_Pclass.fit_transform(X[:, 0])
    
    onehotencoder = OneHotEncoder(handle_unknown='ignore')
    enc_df = onehotencoder.fit_transform(df[[col]]).toarray()
    enc_df = pd.DataFrame(enc_df)
    
    df = enc_df.join(df)
    df.columns = colunas
    df.drop(col, axis=1, inplace=True)
    
    return df
```


```python
data2 = encoder(data, 'Home')
data2 = encoder(data2, 'Away')
```


```python
data2.dropna(inplace=True)
```


```python
data2.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Away_America MG</th>
      <th>Away_Athletico-PR</th>
      <th>Away_Atletico GO</th>
      <th>Away_Atletico-MG</th>
      <th>Away_Atletico-PR</th>
      <th>Away_Avai</th>
      <th>Away_Bahia</th>
      <th>Away_Botafogo RJ</th>
      <th>Away_Bragantino</th>
      <th>Away_CSA</th>
      <th>...</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
      <th>Home_categorical</th>
      <th>Away_categorical</th>
      <th>Time_categorical</th>
      <th>Date_categorical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.76</td>
      <td>3.87</td>
      <td>5.31</td>
      <td>1.69</td>
      <td>3.50</td>
      <td>4.90</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>32.0</td>
      <td>646.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.83</td>
      <td>3.42</td>
      <td>2.70</td>
      <td>2.59</td>
      <td>3.23</td>
      <td>2.58</td>
      <td>35.0</td>
      <td>19.0</td>
      <td>32.0</td>
      <td>646.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.67</td>
      <td>4.05</td>
      <td>7.22</td>
      <td>1.59</td>
      <td>3.67</td>
      <td>5.64</td>
      <td>18.0</td>
      <td>27.0</td>
      <td>9.0</td>
      <td>686.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.49</td>
      <td>3.39</td>
      <td>3.15</td>
      <td>2.35</td>
      <td>3.26</td>
      <td>2.84</td>
      <td>8.0</td>
      <td>34.0</td>
      <td>22.0</td>
      <td>686.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.89</td>
      <td>3.33</td>
      <td>3.89</td>
      <td>13.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>686.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 92 columns</p>
</div>




```python
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Date</th>
      <th>Time</th>
      <th>Home</th>
      <th>Away</th>
      <th>Res</th>
      <th>PH</th>
      <th>PD</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
      <th>Home_categorical</th>
      <th>Away_categorical</th>
      <th>Time_categorical</th>
      <th>Date_categorical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012</td>
      <td>19/05/2012</td>
      <td>22:30</td>
      <td>Palmeiras</td>
      <td>Portuguesa</td>
      <td>0</td>
      <td>1.75</td>
      <td>3.86</td>
      <td>5.25</td>
      <td>1.76</td>
      <td>3.87</td>
      <td>5.31</td>
      <td>1.69</td>
      <td>3.50</td>
      <td>4.90</td>
      <td>28</td>
      <td>31</td>
      <td>32</td>
      <td>646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>19/05/2012</td>
      <td>22:30</td>
      <td>Sport Recife</td>
      <td>Flamengo RJ</td>
      <td>0</td>
      <td>2.83</td>
      <td>3.39</td>
      <td>2.68</td>
      <td>2.83</td>
      <td>3.42</td>
      <td>2.70</td>
      <td>2.59</td>
      <td>3.23</td>
      <td>2.58</td>
      <td>35</td>
      <td>19</td>
      <td>32</td>
      <td>646</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>01:00</td>
      <td>Figueirense</td>
      <td>Nautico</td>
      <td>1</td>
      <td>1.60</td>
      <td>4.04</td>
      <td>6.72</td>
      <td>1.67</td>
      <td>4.05</td>
      <td>7.22</td>
      <td>1.59</td>
      <td>3.67</td>
      <td>5.64</td>
      <td>18</td>
      <td>27</td>
      <td>9</td>
      <td>686</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>20:00</td>
      <td>Botafogo RJ</td>
      <td>Sao Paulo</td>
      <td>1</td>
      <td>2.49</td>
      <td>3.35</td>
      <td>3.15</td>
      <td>2.49</td>
      <td>3.39</td>
      <td>3.15</td>
      <td>2.35</td>
      <td>3.26</td>
      <td>2.84</td>
      <td>8</td>
      <td>34</td>
      <td>22</td>
      <td>686</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>20/05/2012</td>
      <td>20:00</td>
      <td>Corinthians</td>
      <td>Fluminense</td>
      <td>-1</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.89</td>
      <td>3.33</td>
      <td>3.89</td>
      <td>13</td>
      <td>20</td>
      <td>22</td>
      <td>686</td>
    </tr>
  </tbody>
</table>
</div>



### 3) Modelagem

Agora vamos modelar o dataset para aplicar os algoritmos de Machine Learning.<br/>
As etapas consistem em:
* Selecionar somente os atributos que serão utilizados na versão do modelo;
* Padronizar os dados;
* Dividir o conjunto de dados em treino, teste e validação;
* Aplicar os algoritmos de Machine Learning.

**Seleção de atributos**<br/>
A primeira parte da modelagem dos dados é selecionar os melhores atributos para a construção do modelo. Para isso vamos utilizar a função SelectKBest do Scikit-Learning que utiliza além da correlação das variáveis, mais algumas métricas para nos dizer quais atributos são os mais relevantes para o modelo.


```python
from sklearn.feature_selection import SelectKBest
```


```python
colunas_drop = ['Date', 'Time', 'Home', 'Away', 'Res']
features_list = ['Season', 'PH', 'PD', 'PA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD',
       'AvgA', 'Home_categorical', 'Away_categorical', 'Time_categorical',
       'Date_categorical']
features = data.drop(colunas_drop, axis=1) #A função SelectKBest  funciona somente com atributos numéricos, por isso dropamos os atributos categóricos
target = data['Res']

k_best_features = SelectKBest(k='all')
k_best_features.fit_transform(features, target)
k_best_scores = k_best_features.scores_
raw_pairs = zip(features_list, k_best_scores)
ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))

k_best_features_final = dict(ordered_pairs[:15])
best_features = k_best_features_final.keys()

dicionario = {
    'Atributo': list(k_best_features_final.keys()),
    'Valor': list(k_best_features_final.values()) 
}

pd.DataFrame(dicionario)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Atributo</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH</td>
      <td>154.497585</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AvgH</td>
      <td>149.681919</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AvgA</td>
      <td>143.278938</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PA</td>
      <td>136.859756</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MaxA</td>
      <td>135.307882</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MaxH</td>
      <td>135.254897</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MaxD</td>
      <td>67.108791</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AvgD</td>
      <td>65.887455</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PD</td>
      <td>64.678581</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Season</td>
      <td>1.393195</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Time_categorical</td>
      <td>1.007850</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Date_categorical</td>
      <td>0.469567</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Home_categorical</td>
      <td>0.265985</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Away_categorical</td>
      <td>0.189104</td>
    </tr>
  </tbody>
</table>
</div>



Aqui podemos perceber que aqueles atributos que nós mapeamos não são relevantes para o modelo, por outro lado, os atributos que referem-se as estatísticas e probabilidades são muito importantes para o modelo.


```python
target = data['Res']
colunas_drop1  = ['Res', 
                 'Date', 'Time',
                 'Away_categorical',
                 'Home_categorical',
                 'Date_categorical',
                 'Time_categorical',
                 'Season'
                 ]

colunas_drop2  = ['Res', 
                 'Date', 'Time',
                 'Away_categorical',
                 'Home_categorical',
                 'Date_categorical',
                 'Time_categorical',
                 'Season',
                 'PD',
                 'AvgD',
                 'MaxD'
                 ]


dataV1 = data.drop(columns=colunas_drop1)
dataV2 = data.drop(columns=colunas_drop2)
dataV3 = data2.drop(columns=colunas_drop1)
dataV4 = data2.drop(columns=colunas_drop2)

dataV1.drop(columns=['Home', 'Away'], inplace=True)
dataV2.drop(columns=['Home', 'Away'], inplace=True)
```

**Agora temos 4 datasets**
Tínhamos dividido nosso dataset em dois: O primeiro que possuia apenas a conversão das variáveis categóricas para numéricas e o segundo possuia além dessa conversão, a aplicação do método de one-hot-enconding.<br/>
Ao aplicar o SelectKBest e ele nos dizer quais são os melhores atributos para o modelo, eu decidi pegar dois grupos:
* O primeiro com somente os atributos com a pontuação acima de 100;
* O segundo com a pontuação acima de 50;

Aplicando essa divisão de grupos nos dois datasets, temos agora 4 datasets que serão utilizados para os testes.


```python
dataV1.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PH</th>
      <th>PD</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.75</td>
      <td>3.86</td>
      <td>5.25</td>
      <td>1.76</td>
      <td>3.87</td>
      <td>5.31</td>
      <td>1.69</td>
      <td>3.50</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.83</td>
      <td>3.39</td>
      <td>2.68</td>
      <td>2.83</td>
      <td>3.42</td>
      <td>2.70</td>
      <td>2.59</td>
      <td>3.23</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.60</td>
      <td>4.04</td>
      <td>6.72</td>
      <td>1.67</td>
      <td>4.05</td>
      <td>7.22</td>
      <td>1.59</td>
      <td>3.67</td>
      <td>5.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.49</td>
      <td>3.35</td>
      <td>3.15</td>
      <td>2.49</td>
      <td>3.39</td>
      <td>3.15</td>
      <td>2.35</td>
      <td>3.26</td>
      <td>2.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.89</td>
      <td>3.33</td>
      <td>3.89</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataV2.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PH</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.75</td>
      <td>5.25</td>
      <td>1.76</td>
      <td>5.31</td>
      <td>1.69</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.83</td>
      <td>2.68</td>
      <td>2.83</td>
      <td>2.70</td>
      <td>2.59</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.60</td>
      <td>6.72</td>
      <td>1.67</td>
      <td>7.22</td>
      <td>1.59</td>
      <td>5.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.49</td>
      <td>3.15</td>
      <td>2.49</td>
      <td>3.15</td>
      <td>2.35</td>
      <td>2.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.96</td>
      <td>4.41</td>
      <td>1.96</td>
      <td>4.41</td>
      <td>1.89</td>
      <td>3.89</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataV3.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Away_America MG</th>
      <th>Away_Athletico-PR</th>
      <th>Away_Atletico GO</th>
      <th>Away_Atletico-MG</th>
      <th>Away_Atletico-PR</th>
      <th>Away_Avai</th>
      <th>Away_Bahia</th>
      <th>Away_Botafogo RJ</th>
      <th>Away_Bragantino</th>
      <th>Away_CSA</th>
      <th>...</th>
      <th>Home_Vitoria</th>
      <th>PH</th>
      <th>PD</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxD</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgD</th>
      <th>AvgA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.75</td>
      <td>3.86</td>
      <td>5.25</td>
      <td>1.76</td>
      <td>3.87</td>
      <td>5.31</td>
      <td>1.69</td>
      <td>3.50</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.83</td>
      <td>3.39</td>
      <td>2.68</td>
      <td>2.83</td>
      <td>3.42</td>
      <td>2.70</td>
      <td>2.59</td>
      <td>3.23</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.60</td>
      <td>4.04</td>
      <td>6.72</td>
      <td>1.67</td>
      <td>4.05</td>
      <td>7.22</td>
      <td>1.59</td>
      <td>3.67</td>
      <td>5.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.49</td>
      <td>3.35</td>
      <td>3.15</td>
      <td>2.49</td>
      <td>3.39</td>
      <td>3.15</td>
      <td>2.35</td>
      <td>3.26</td>
      <td>2.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.96</td>
      <td>3.53</td>
      <td>4.41</td>
      <td>1.89</td>
      <td>3.33</td>
      <td>3.89</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>




```python
dataV4.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Away_America MG</th>
      <th>Away_Athletico-PR</th>
      <th>Away_Atletico GO</th>
      <th>Away_Atletico-MG</th>
      <th>Away_Atletico-PR</th>
      <th>Away_Avai</th>
      <th>Away_Bahia</th>
      <th>Away_Botafogo RJ</th>
      <th>Away_Bragantino</th>
      <th>Away_CSA</th>
      <th>...</th>
      <th>Home_Sao Paulo</th>
      <th>Home_Sport Recife</th>
      <th>Home_Vasco</th>
      <th>Home_Vitoria</th>
      <th>PH</th>
      <th>PA</th>
      <th>MaxH</th>
      <th>MaxA</th>
      <th>AvgH</th>
      <th>AvgA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.75</td>
      <td>5.25</td>
      <td>1.76</td>
      <td>5.31</td>
      <td>1.69</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.83</td>
      <td>2.68</td>
      <td>2.83</td>
      <td>2.70</td>
      <td>2.59</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.60</td>
      <td>6.72</td>
      <td>1.67</td>
      <td>7.22</td>
      <td>1.59</td>
      <td>5.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.49</td>
      <td>3.15</td>
      <td>2.49</td>
      <td>3.15</td>
      <td>2.35</td>
      <td>2.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.96</td>
      <td>4.41</td>
      <td>1.96</td>
      <td>4.41</td>
      <td>1.89</td>
      <td>3.89</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



**Padronização dos dados**<br/>
Após selecionar so atributos que serão utilizados vamos padronizar os dados. A etapa de padronização tem alguns propósitos:
* Deixar todos os atributos na mesma escala, evitando que os dados fiquem muito dispersos;
* Normalmente as padronizações transformam os dados em um mapa de valores entre -1 e 1 (Dependendo da técnica), com isso o processamento computacional se torna menos custoso, pois a máquina não precisa realizar cálculos com valores exorbitantes;
* A principal função da padronização é aproximar os dados de uma distribuição normal, dado que vários algoritmos de Machine Learning se saem melhor com dados normalmente distribuídos.

Vamos utilizar duas técnicas, a StandardScaler, para uma padronização mais padrão, assim como o nome sugere e a Normalizer, que é uma técnica para normalizar os dados e deixá-los normalmente distribuídos.



```python
#from sklearn.preprocessing import QuantileTransformer
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
```


```python
X_standard1 = dataV1.copy()
X_standard1 = StandardScaler().fit_transform(dataV1)

X_standard2 = dataV2.copy()
X_standard2 = StandardScaler().fit_transform(dataV2)

X_standard3 = dataV3.copy()
X_standard3 = StandardScaler().fit_transform(dataV3)

X_standard4 = dataV4.copy()
X_standard4 = StandardScaler().fit_transform(dataV4)
```


```python
X_normalizer1 = dataV1.copy()
X_normalizer1 = Normalizer().fit_transform(dataV1)

X_normalizer2 = dataV2.copy()
X_normalizer2 = Normalizer().fit_transform(dataV2)

X_normalizer3 = dataV3.copy()
X_normalizer3 = Normalizer().fit_transform(dataV3)

X_normalizer4 = dataV4.copy()
X_normalizer4 = Normalizer().fit_transform(dataV4)
```

Aplicando as duas técnicas para cada um dos nossos datasets, temos agora oito datasets.


```python
list_X = {
    'X_standard1': X_standard1,
    'X_standard2': X_standard2,
    'X_standard3': X_standard3,
    'X_standard4': X_standard4,
    'X_normalizer1': X_normalizer1,
    'X_normalizer2': X_normalizer2,
    'X_normalizer3': X_normalizer3,
    'X_normalizer4': X_normalizer4
}

Y = target
```

**Divisão treino e teste**

Esta é a etapa onde dividimos nossos dados em um conjunto de Treino, outro para Testes, e ainda um terceiro conjunto para Validações. Poderíamos utilizar o train_test_split que é uma excelente função para realizar uma divisão nos conjuntos de dados. Porém, vamos realizar o treinamento com os dados sequenciais até a temporada de 2020, e separar a temporada de 2021 para ralizarmos as previsões.


```python
#from sklearn.model_selection import GridSearchCV, train_test_split
#x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.33, random_state = 101)
```


```python
#Fazendo uma divisão de treino e teste por temporada
print(data['Season'].iloc[3415:3420]) #Onde exatamente começa a temporada 2021
print(data['Season'].iloc[3795:3800]) #Onde exatamente começa a temporada 2022
```

    3416    2020
    3417    2020
    3418    2020
    3419    2020
    3420    2021
    Name: Season, dtype: int64
    3796    2021
    3797    2021
    3798    2021
    3799    2021
    3800    2022
    Name: Season, dtype: int64
    

Os nossos index para separar nosso conjunto de dados foram definidos:
* 0 - 3419: Dados da temporada 2012 até 2020.
* 3419 - 3799: Dados da temporada 2021.


```python
index_ini_treino = 0
index_fim_treino = 3419
index_ini_teste = 3419
index_fim_teste = 3799
```

**Cross Validation (K-Fold)**
Neste caso nós também não iremos aplicar a validação cruzada, pelos motivos que eu citei no item acima. Mas fica a sugestão para estudos posteriores.


```python
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
```

### 4) Aplicação dos algoritmos de Machine Learning


```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time
```

Vamos criar uma lista para guardar as informações detelhadas do nosso treinamento.


```python
lst_detalhes = []

#obj = {
#    'Algoritmo': O algoritmo que será utilizado, 
#    'Dataset': Qual dentre os nossos 8 datasets,
#    'Acurácia' A precisão do modelo: 
#}
```

##### **Criando método GridSearch**

O GridSearch é o principal método para testar diferentes combinações de parâmetros dos modelos, parâmetros estes que demandariam muito tempo se fossem testados manualmente.<br/>
Por isso decidi criar um método de GridSearch que recebe o modelo, um dicionário com os parâmetros do modelo, os dados de treino e teste e um parâmetro que serve como label para indicar qual é o modelo/conjunto de dados que está sendo testado.


```python
def modeloGrid(modelo, parametros, x_train, y_train, x_test, y_test, nome_modelo, nome_dataset):
    modeloGrid = GridSearchCV(modelo, parametros, cv = 10)
    
    modeloGrid.fit(x_train, y_train)
    
    acuracia = round(accuracy_score(y_test, modeloGrid.predict(x_test)) * 100, 2)
    print('Melhor performance do modelo {} {}: {:.2f}%'.format(nome_modelo, nome_dataset, acuracia))

    obj = {
        'Algoritmo': nome_modelo, 
        'Dataset': nome_dataset,
        'Acurácia' : acuracia
    }
    lst_detalhes.append(obj)
    
    return modeloGrid
```

##### **KNN**

O KNN é um dos algoritmos mais simples para Machine Learning, sendo um algoritmo do tipo "lazy", ou seja, nenhuma computação é realizada no dataset até que um novo ponto de dado seja alvo de teste.<br/>

Ele é um algoritmo não-linear que utiliza uma métrica de distância para encontrar o valor de K mais adequado as instâncias do dataset de treino.


```python
#Criando o modelo KNN
KNN = KNeighborsClassifier()

# Definição de parâmetros
k = np.arange(20) + 1
parametros = {'n_neighbors':k}
```


```python
for item in list_X.items():
    X = item[1]

    x_train = X[index_ini_treino :index_fim_treino]
    y_train = Y[index_ini_treino :index_fim_treino]
    x_test = X[index_ini_teste :index_fim_teste]
    y_test = Y[index_ini_teste :index_fim_teste]
    modelo = modeloGrid(KNN, parametros, x_train, y_train, x_test, y_test, 'KNN', item[0])
```

    Melhor performance do modelo KNN X_standard1: 47.89%
    Melhor performance do modelo KNN X_standard2: 47.63%
    Melhor performance do modelo KNN X_standard3: 45.26%
    Melhor performance do modelo KNN X_standard4: 45.79%
    Melhor performance do modelo KNN X_normalizer1: 45.79%
    Melhor performance do modelo KNN X_normalizer2: 47.63%
    Melhor performance do modelo KNN X_normalizer3: 44.21%
    Melhor performance do modelo KNN X_normalizer4: 45.53%
    


```python
pd.DataFrame(lst_detalhes)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algoritmo</th>
      <th>Dataset</th>
      <th>Acurácia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KNN</td>
      <td>X_standard1</td>
      <td>47.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>X_standard2</td>
      <td>47.63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>X_standard3</td>
      <td>45.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNN</td>
      <td>X_standard4</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNN</td>
      <td>X_normalizer1</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KNN</td>
      <td>X_normalizer2</td>
      <td>47.63</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KNN</td>
      <td>X_normalizer3</td>
      <td>44.21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>KNN</td>
      <td>X_normalizer4</td>
      <td>45.53</td>
    </tr>
  </tbody>
</table>
</div>



##### **Decision Tree**

É um tipo de algoritmo que cria uma estrutura de árvore até que uma decisão seja feita para um determinado registro. Inicia sempre de um nó raiz, a partir daí selecionando os outros atributos e tomando decisões.


```python
#Criando o modelo de Árvore de Decisão
Decision_Tree = DecisionTreeClassifier()

parametros = {
    'max_depth' : ([1, 2, 3, 4, 5, 6]),
    'max_features': ([None, 'sqrt', 'log2']),
    'criterion': (['entropy', 'gini']),
    'min_samples_leaf': ([1, 2, 3]), 
    'min_samples_split': ([2, 4, 8, 10])

}
```


```python
for item in list_X.items():
    X = item[1]
    nome = 'Decision Tree ' + item[0]

    x_train = X[0:3419]
    y_train = Y[0:3419]
    x_test = X[3419:3799]
    y_test = Y[3419:3799]
    modelo = modeloGrid(Decision_Tree, parametros, x_train, y_train, x_test, y_test, 'Decision Tree', item[0])
```

    Melhor performance do modelo Decision Tree X_standard1: 47.11%
    Melhor performance do modelo Decision Tree X_standard2: 51.58%
    Melhor performance do modelo Decision Tree X_standard3: 49.47%
    Melhor performance do modelo Decision Tree X_standard4: 50.79%
    Melhor performance do modelo Decision Tree X_normalizer1: 50.79%
    Melhor performance do modelo Decision Tree X_normalizer2: 51.32%
    Melhor performance do modelo Decision Tree X_normalizer3: 48.16%
    Melhor performance do modelo Decision Tree X_normalizer4: 42.63%
    


```python
pd.DataFrame(lst_detalhes).sort_values(by='Acurácia', ascending=False)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algoritmo</th>
      <th>Dataset</th>
      <th>Acurácia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Decision Tree</td>
      <td>X_standard2</td>
      <td>51.58</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Decision Tree</td>
      <td>X_normalizer2</td>
      <td>51.32</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Decision Tree</td>
      <td>X_standard4</td>
      <td>50.79</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Decision Tree</td>
      <td>X_normalizer1</td>
      <td>50.79</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Decision Tree</td>
      <td>X_standard3</td>
      <td>49.47</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Decision Tree</td>
      <td>X_normalizer3</td>
      <td>48.16</td>
    </tr>
    <tr>
      <th>0</th>
      <td>KNN</td>
      <td>X_standard1</td>
      <td>47.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>X_standard2</td>
      <td>47.63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KNN</td>
      <td>X_normalizer2</td>
      <td>47.63</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Decision Tree</td>
      <td>X_standard1</td>
      <td>47.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNN</td>
      <td>X_standard4</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNN</td>
      <td>X_normalizer1</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>7</th>
      <td>KNN</td>
      <td>X_normalizer4</td>
      <td>45.53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>X_standard3</td>
      <td>45.26</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KNN</td>
      <td>X_normalizer3</td>
      <td>44.21</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Decision Tree</td>
      <td>X_normalizer4</td>
      <td>42.63</td>
    </tr>
  </tbody>
</table>
</div>



##### **Random Forest**

Modelos compostos por múltiplos modelos mais fracos que são independentemente treinados e a previsão combina esse modelos de alguma forma; Criam vários modelos de árvores diferentes, selecionando diferentes atributos com diferentes configurações. A partir daí o melhor modelo será selecionado através de um processo de votação.


```python
RandomForest = RandomForestClassifier()

parametros = {'n_estimators': [5],
              'max_depth': [10, 25, 50],
              'bootstrap': ([True, False]),
              'criterion': (['entropy', 'gini']),
              'min_samples_leaf': ([1, 2, 4, 6]),
              'min_samples_split': ([2, 10, 22])
             }
```


```python
for item in list_X.items():
    X = item[1]

    x_train = X[0:3419]
    y_train = Y[0:3419]
    x_test = X[3419:3799]
    y_test = Y[3419:3799]
    modelo = modeloGrid(RandomForest, parametros, x_train, y_train, x_test, y_test, 'Random Forest', item[0])
```

    Melhor performance do modelo Random Forest X_standard1: 47.37%
    Melhor performance do modelo Random Forest X_standard2: 48.68%
    Melhor performance do modelo Random Forest X_standard3: 46.84%
    Melhor performance do modelo Random Forest X_standard4: 48.95%
    Melhor performance do modelo Random Forest X_normalizer1: 51.05%
    Melhor performance do modelo Random Forest X_normalizer2: 50.26%
    Melhor performance do modelo Random Forest X_normalizer3: 49.74%
    Melhor performance do modelo Random Forest X_normalizer4: 46.05%
    


```python
pd.DataFrame(lst_detalhes).sort_values(by='Acurácia', ascending=False)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algoritmo</th>
      <th>Dataset</th>
      <th>Acurácia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Decision Tree</td>
      <td>X_standard2</td>
      <td>51.58</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Decision Tree</td>
      <td>X_normalizer2</td>
      <td>51.32</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Random Forest</td>
      <td>X_normalizer1</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Decision Tree</td>
      <td>X_normalizer1</td>
      <td>50.79</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Decision Tree</td>
      <td>X_standard4</td>
      <td>50.79</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Random Forest</td>
      <td>X_normalizer2</td>
      <td>50.26</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Random Forest</td>
      <td>X_normalizer3</td>
      <td>49.74</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Decision Tree</td>
      <td>X_standard3</td>
      <td>49.47</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Random Forest</td>
      <td>X_standard4</td>
      <td>48.95</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Random Forest</td>
      <td>X_standard2</td>
      <td>48.68</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Decision Tree</td>
      <td>X_normalizer3</td>
      <td>48.16</td>
    </tr>
    <tr>
      <th>0</th>
      <td>KNN</td>
      <td>X_standard1</td>
      <td>47.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>X_standard2</td>
      <td>47.63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KNN</td>
      <td>X_normalizer2</td>
      <td>47.63</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Random Forest</td>
      <td>X_standard1</td>
      <td>47.37</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Decision Tree</td>
      <td>X_standard1</td>
      <td>47.11</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Random Forest</td>
      <td>X_standard3</td>
      <td>46.84</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Random Forest</td>
      <td>X_normalizer4</td>
      <td>46.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNN</td>
      <td>X_normalizer1</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNN</td>
      <td>X_standard4</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>7</th>
      <td>KNN</td>
      <td>X_normalizer4</td>
      <td>45.53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>X_standard3</td>
      <td>45.26</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KNN</td>
      <td>X_normalizer3</td>
      <td>44.21</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Decision Tree</td>
      <td>X_normalizer4</td>
      <td>42.63</td>
    </tr>
  </tbody>
</table>
</div>



##### **Naive Bayes**

O Naive Bayes é um tipo de classificador que desconsidera a correlação entre os atributos, cada atributo é tratado individualmente.

A resolução de problemas relacionados a texto é muito bem resolvida com a utilização do Naive Bayes. Classificação de textos, filtragem de SPAM e análise de sentimento em redes sociais são algumas das muitas aplicações para o algoritmo.


```python
from sklearn.naive_bayes import GaussianNB
```


```python
#Treinando

for item in list_X.items():
    NaiveBayes = GaussianNB()
    X = item[1]

    x_train = X[0:3419]
    y_train = Y[0:3419]
    x_test = X[3419:3799]
    y_test = Y[3419:3799]
    NaiveBayes.fit(x_train, y_train)
    
    acuracia = round(accuracy_score(y_test, NaiveBayes.predict(x_test)) * 100, 2)
    print('Melhor performance do modelo Naive Bayes: {:.2f}%'.format(acuracia))

    obj = {
        'Algoritmo': 'Naive Bayes', 
        'Dataset': item[0],
        'Acurácia' : acuracia
    }
    lst_detalhes.append(obj)

```

    Melhor performance do modelo Naive Bayes: 42.11%
    Melhor performance do modelo Naive Bayes: 47.63%
    Melhor performance do modelo Naive Bayes: 32.89%
    Melhor performance do modelo Naive Bayes: 31.84%
    Melhor performance do modelo Naive Bayes: 46.84%
    Melhor performance do modelo Naive Bayes: 47.37%
    Melhor performance do modelo Naive Bayes: 35.00%
    Melhor performance do modelo Naive Bayes: 32.37%
    


```python
pd.DataFrame(lst_detalhes).sort_values(by='Acurácia', ascending=False)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algoritmo</th>
      <th>Dataset</th>
      <th>Acurácia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Decision Tree</td>
      <td>X_standard2</td>
      <td>51.58</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Decision Tree</td>
      <td>X_normalizer2</td>
      <td>51.32</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Random Forest</td>
      <td>X_normalizer1</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Decision Tree</td>
      <td>X_standard4</td>
      <td>50.79</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Decision Tree</td>
      <td>X_normalizer1</td>
      <td>50.79</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Random Forest</td>
      <td>X_normalizer2</td>
      <td>50.26</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Random Forest</td>
      <td>X_normalizer3</td>
      <td>49.74</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Decision Tree</td>
      <td>X_standard3</td>
      <td>49.47</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Random Forest</td>
      <td>X_standard4</td>
      <td>48.95</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Random Forest</td>
      <td>X_standard2</td>
      <td>48.68</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Decision Tree</td>
      <td>X_normalizer3</td>
      <td>48.16</td>
    </tr>
    <tr>
      <th>0</th>
      <td>KNN</td>
      <td>X_standard1</td>
      <td>47.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>X_standard2</td>
      <td>47.63</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Naive Bayes</td>
      <td>X_standard2</td>
      <td>47.63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KNN</td>
      <td>X_normalizer2</td>
      <td>47.63</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Naive Bayes</td>
      <td>X_normalizer2</td>
      <td>47.37</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Random Forest</td>
      <td>X_standard1</td>
      <td>47.37</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Decision Tree</td>
      <td>X_standard1</td>
      <td>47.11</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Random Forest</td>
      <td>X_standard3</td>
      <td>46.84</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Naive Bayes</td>
      <td>X_normalizer1</td>
      <td>46.84</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Random Forest</td>
      <td>X_normalizer4</td>
      <td>46.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNN</td>
      <td>X_normalizer1</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNN</td>
      <td>X_standard4</td>
      <td>45.79</td>
    </tr>
    <tr>
      <th>7</th>
      <td>KNN</td>
      <td>X_normalizer4</td>
      <td>45.53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>X_standard3</td>
      <td>45.26</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KNN</td>
      <td>X_normalizer3</td>
      <td>44.21</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Decision Tree</td>
      <td>X_normalizer4</td>
      <td>42.63</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Naive Bayes</td>
      <td>X_standard1</td>
      <td>42.11</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Naive Bayes</td>
      <td>X_normalizer3</td>
      <td>35.00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Naive Bayes</td>
      <td>X_standard3</td>
      <td>32.89</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Naive Bayes</td>
      <td>X_normalizer4</td>
      <td>32.37</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Naive Bayes</td>
      <td>X_standard4</td>
      <td>31.84</td>
    </tr>
  </tbody>
</table>
</div>



##### **Logistic Regression**

A regressão logística é um algoritmo de aprendizagem de máquina supervisionado utilizado para classificação, apesar de ter a palavra regressão em seu nome.<br/>
Esta consiste em analisar cada classe de forma separada contra todas as outras, criando um classificador para cada possibilidade, dessa forma se calcula a probabilidade de uma dada instância pertencer a classe em questão ou não. Ao final, a classe selecionada será aquela que apresentar a maior probabilidade.


```python
logistRegression = LogisticRegression()

parametros = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'intercept_scaling': [0, 0.5, 1],
    'multi_class': ['multinomial']
}

for item in list_X.items():
    X = item[1]

    x_train = X[0:3419]
    y_train = Y[0:3419]
    x_test = X[3419:3799]
    y_test = Y[3419:3799]
    modelo = modeloGrid(logistRegression, parametros, x_train, y_train, x_test, y_test, 'Logistic Regression', item[0])
```

    Melhor performance do modelo Logistic Regression X_standard1: 52.11%
    Melhor performance do modelo Logistic Regression X_standard2: 51.05%
    Melhor performance do modelo Logistic Regression X_standard3: 47.63%
    Melhor performance do modelo Logistic Regression X_standard4: 49.21%
    Melhor performance do modelo Logistic Regression X_normalizer1: 51.05%
    Melhor performance do modelo Logistic Regression X_normalizer2: 51.32%
    Melhor performance do modelo Logistic Regression X_normalizer3: 50.26%
    Melhor performance do modelo Logistic Regression X_normalizer4: 50.26%
    


```python
pd.DataFrame(lst_detalhes).sort_values(by='Acurácia', ascending=False).head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algoritmo</th>
      <th>Dataset</th>
      <th>Acurácia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>Logistic Regression</td>
      <td>X_standard1</td>
      <td>52.11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Decision Tree</td>
      <td>X_standard2</td>
      <td>51.58</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Logistic Regression</td>
      <td>X_normalizer2</td>
      <td>51.32</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Decision Tree</td>
      <td>X_normalizer2</td>
      <td>51.32</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Random Forest</td>
      <td>X_normalizer1</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Logistic Regression</td>
      <td>X_normalizer1</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>36</th>
      <td>SVM</td>
      <td>X_normalizer1</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Logistic Regression</td>
      <td>X_standard2</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>37</th>
      <td>SVM</td>
      <td>X_normalizer2</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>44</th>
      <td>SVM</td>
      <td>X_normalizer3</td>
      <td>51.05</td>
    </tr>
  </tbody>
</table>
</div>



##### **SVM**

O algoritmo Support Vector Machine traça uma reta e tenta separar linearmente e classificar o conjunto de dados. O algoritmo tenta encontrar a reta que tenha maior distância dentre as classes prevista.<br/>

Para aplicar o SVC em conjuntos de dados não linearmente separáveis é necessário configurar o parâmetro Kernel. Este parâmetro é responsável por traçar não somente retas, mas também outros tipos de linhas no conjunto de dados.<br/>

> Possibilidades: kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}


```python
SVM = SVC()

# Template para utilizar o GridSearch e experimentar diferentes parâmetros
# Porém a execução vai ser bem mais demorada
#parametros = {'kernel': ['linear', 'rbf', 'sigmoid'], 
#              'C': [1, 2, 4],
#              'gamma': [0.25, 0.75, 1]}

parametros = {}

for item in list_X.items():
    X = item[1]

    x_train = X[0:3419]
    y_train = Y[0:3419]
    x_test = X[3419:3799]
    y_test = Y[3419:3799]
    modelo = modeloGrid(SVM, parametros, x_train, y_train, x_test, y_test, 'SVM', item[0])
    

```

    Melhor performance do modelo SVM X_standard1: 50.79%
    Melhor performance do modelo SVM X_standard2: 51.05%
    Melhor performance do modelo SVM X_standard3: 48.95%
    Melhor performance do modelo SVM X_standard4: 48.42%
    Melhor performance do modelo SVM X_normalizer1: 50.53%
    Melhor performance do modelo SVM X_normalizer2: 50.79%
    Melhor performance do modelo SVM X_normalizer3: 51.05%
    Melhor performance do modelo SVM X_normalizer4: 51.05%
    


```python
pd.DataFrame(lst_detalhes).sort_values(by='Acurácia', ascending=False).head(15)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algoritmo</th>
      <th>Dataset</th>
      <th>Acurácia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>Logistic Regression</td>
      <td>X_standard1</td>
      <td>52.11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Decision Tree</td>
      <td>X_standard2</td>
      <td>51.58</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Logistic Regression</td>
      <td>X_normalizer2</td>
      <td>51.32</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Decision Tree</td>
      <td>X_normalizer2</td>
      <td>51.32</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Random Forest</td>
      <td>X_normalizer1</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Logistic Regression</td>
      <td>X_normalizer1</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>36</th>
      <td>SVM</td>
      <td>X_normalizer1</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Logistic Regression</td>
      <td>X_standard2</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>37</th>
      <td>SVM</td>
      <td>X_normalizer2</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>44</th>
      <td>SVM</td>
      <td>X_normalizer3</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>45</th>
      <td>SVM</td>
      <td>X_normalizer4</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>39</th>
      <td>SVM</td>
      <td>X_standard2</td>
      <td>51.05</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Random Forest</td>
      <td>X_normalizer1</td>
      <td>50.79</td>
    </tr>
    <tr>
      <th>43</th>
      <td>SVM</td>
      <td>X_normalizer2</td>
      <td>50.79</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Decision Tree</td>
      <td>X_standard2</td>
      <td>50.79</td>
    </tr>
  </tbody>
</table>
</div>



**Considerações sobre os modelos criados**<br/><br/>
Dentre todos os modelos criados, o que obteve o melhor desempenho seguindo as métricas de acurácia foram Logistic Regression, Decision Tree e SVM. <br/><br/>
Porém, um ponto que devemos levar em consideração sobre as árvores de decisões é que o fato delas obterem uma melhor acurácia nesses testes não significam que serão sempre assim. O modelo Random Forest é um conjunto de árvores e por se tratar de um método que testa várias árvores de decisão se torna muito mais confiável com os resultados mais próximos da realidade.<br/><br/>
Já as SVMs são uma familia de algoritmos muito complexas, e seu tempo de treinamento foi bastante demorado.
Por outro lado, a regressão logística se saiu bastante performática em relação aos outros algoritmos. Tanto em precisão da acurácia, quanto em velocidade de execução.


```python
logistRegression = LogisticRegression()

parametros = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'intercept_scaling': [0, 0.25, 0.4, 0.5, 0.75, 1],
    'multi_class': ['multinomial'],
    'C': [0, 0.25, 0.4, 0.5, 0.75, 1]
}

_X = list_X['X_standard1']

x_train = _X[0:3419]
y_train = Y[0:3419]
x_test = _X[3419:3799]
y_test = Y[3419:3799]
modelo = modeloGrid(logistRegression, parametros, x_train, y_train, x_test, y_test, 'Logistic Regression', 'X_standard1')

```

    Melhor performance do modelo Logistic Regression X_standard1: 52.11%
    

### 5) Utilizando o modelo para prever os resultados do campeonato

Abaixo estão listados os índices do início de algumas temporadas, sendo a nossa temporada alvo a de 2021, cujo os dados não foram apresentados para o modelo.
* Season 2013: 380-760
* Season 2020: 3040-3419
* Season 2021: 3420-3799 (principal alvo)

Em seguida temos o método preverTabelaCampeonato. Este método recebe como parâmetros o índice inicial e o índice final, o dataframe com os labels, o array com os valores e o modelo, e tem como saída a tabela do campeonato dentro desse intervalo específico. A tabela contém as seguintes informações:
* Equipe;
* Pontuação;
* Quantidade de vitórias;
* Quantidade de empates;
* Quantidade de derrotas;



```python
def preverTabelaCampeonato(i_inicial, i_final, data, X, modelo):
    x_valid = X[i_inicial:i_final] #Intervalo no array de valores
    
    previsoes = modelo.predict(x_valid) #Realizando a previsão

    #Criação de um dataframe temporário contendo a informação do jogo
    df = pd.DataFrame({
        'Id': data.index[i_inicial:i_final], #Id do jogo
        'Home': data['Home'].iloc[i_inicial:i_final], #Clube mandante
        'Away': data['Away'].iloc[i_inicial:i_final], #Clube visitante
        'Resultado': previsoes #Resultado previsto
    })


    #Criação de mais um dataframe temporário contendo os resultados da equipe como sendo mandante dos jogos
    home = df.groupby('Home').agg({'Resultado':lambda x: [3 if x == 1 else 1 if x == 0 else 0 for x in list(x)]})
    home['PontosHome'] = [sum(a) for a in list(home['Resultado'])]
    home['VitoriasHome'] = [sum([1 if x == 3 else 0 for x in a]) for a in list(home['Resultado'])]
    home['EmpatesHome'] = [sum([1 if x == 1 else 0 for x in a]) for a in list(home['Resultado'])]
    home['DerrotasHome'] = [sum([1 if x == 0 else 0 for x in a]) for a in list(home['Resultado'])]

    #Criação de mais um dataframe temporário contendo os resultados da equipe como sendo visitante dos jogos
    away = df.groupby('Away').agg({'Resultado':lambda x: [0 if x == 1 else 1 if x == 0 else 3 for x in list(x)]})
    away['PontosAway'] = [sum(a) for a in list(away['Resultado'])]
    away['VitoriasAway'] = [sum([1 if x == 3 else 0 for x in a]) for a in list(away['Resultado'])]
    away['EmpatesAway'] = [sum([1 if x == 1 else 0 for x in a]) for a in list(away['Resultado'])]
    away['DerrotasAway'] = [sum([1 if x == 0 else 0 for x in a]) for a in list(away['Resultado'])]

    #União dos resultados obtidos da equipe mandante e visitante
    home['PontosTotais'] = home['PontosHome'] + away['PontosAway']
    home['Vitorias'] = home['VitoriasHome'] + away['VitoriasAway']
    home['Empates'] = home['EmpatesHome'] + away['EmpatesAway']
    home['Derrotas'] = home['DerrotasHome'] + away['DerrotasAway']

    #Ajustes no dataframe com os resultados finais e ordenação por pontuação, seguido de vitórias e por fim empates.
    home.reset_index(inplace=True)
    home.sort_values(by=['PontosTotais', 'Vitorias', 'Empates'], ascending=False, inplace=True)
    home.set_index(np.arange(1, home.shape[0] + 1), inplace=True)

    #Obtenção do dataframe contendo os resultados finais, seleção dos campos pertinentes
    df_final = home[['Home', 'PontosTotais', 'Vitorias', 'Empates', 'Derrotas']]
    df_final.rename(columns={"Home": "Equipe", "Vitorias": "Vitórias"}, inplace=True)

    return df_final
    
```


```python
df_2013 = preverTabelaCampeonato(380, 760, data, _X, modelo)
df_2020 = preverTabelaCampeonato(3040, 3419, data, _X, modelo)
df_2021 = preverTabelaCampeonato(3419, 3799, data, _X, modelo)
```


```python
df_2013
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Equipe</th>
      <th>PontosTotais</th>
      <th>Vitórias</th>
      <th>Empates</th>
      <th>Derrotas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Cruzeiro</td>
      <td>84</td>
      <td>28</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Corinthians</td>
      <td>78</td>
      <td>26</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fluminense</td>
      <td>72</td>
      <td>24</td>
      <td>0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Atletico-MG</td>
      <td>69</td>
      <td>23</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Botafogo RJ</td>
      <td>69</td>
      <td>23</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Internacional</td>
      <td>69</td>
      <td>23</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sao Paulo</td>
      <td>69</td>
      <td>23</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Gremio</td>
      <td>66</td>
      <td>22</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Flamengo RJ</td>
      <td>63</td>
      <td>21</td>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Coritiba</td>
      <td>57</td>
      <td>19</td>
      <td>0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Atletico-PR</td>
      <td>55</td>
      <td>18</td>
      <td>1</td>
      <td>19</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Goias</td>
      <td>51</td>
      <td>17</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Vitoria</td>
      <td>51</td>
      <td>17</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Santos</td>
      <td>46</td>
      <td>15</td>
      <td>1</td>
      <td>22</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Bahia</td>
      <td>45</td>
      <td>15</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Criciuma</td>
      <td>45</td>
      <td>15</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Portuguesa</td>
      <td>45</td>
      <td>15</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Vasco</td>
      <td>42</td>
      <td>14</td>
      <td>0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Ponte Preta</td>
      <td>39</td>
      <td>13</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Nautico</td>
      <td>24</td>
      <td>8</td>
      <td>0</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2020
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Equipe</th>
      <th>PontosTotais</th>
      <th>Vitórias</th>
      <th>Empates</th>
      <th>Derrotas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Flamengo RJ</td>
      <td>105</td>
      <td>35</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Atletico-MG</td>
      <td>96</td>
      <td>32</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Palmeiras</td>
      <td>96</td>
      <td>32</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sao Paulo</td>
      <td>90</td>
      <td>30</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Internacional</td>
      <td>87</td>
      <td>29</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gremio</td>
      <td>84</td>
      <td>28</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bragantino</td>
      <td>63</td>
      <td>21</td>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ceara</td>
      <td>58</td>
      <td>19</td>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Athletico-PR</td>
      <td>51</td>
      <td>17</td>
      <td>0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bahia</td>
      <td>48</td>
      <td>16</td>
      <td>0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Fluminense</td>
      <td>48</td>
      <td>16</td>
      <td>0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Corinthians</td>
      <td>46</td>
      <td>15</td>
      <td>1</td>
      <td>22</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Fortaleza</td>
      <td>45</td>
      <td>15</td>
      <td>0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Atletico GO</td>
      <td>42</td>
      <td>14</td>
      <td>0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Santos</td>
      <td>42</td>
      <td>14</td>
      <td>0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Botafogo RJ</td>
      <td>36</td>
      <td>12</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Vasco</td>
      <td>31</td>
      <td>10</td>
      <td>1</td>
      <td>27</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Sport Recife</td>
      <td>24</td>
      <td>8</td>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Coritiba</td>
      <td>22</td>
      <td>7</td>
      <td>1</td>
      <td>30</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Goias</td>
      <td>21</td>
      <td>7</td>
      <td>0</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2021
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Equipe</th>
      <th>PontosTotais</th>
      <th>Vitórias</th>
      <th>Empates</th>
      <th>Derrotas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Flamengo RJ</td>
      <td>102</td>
      <td>34</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Atletico-MG</td>
      <td>96</td>
      <td>32</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Palmeiras</td>
      <td>78</td>
      <td>26</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sao Paulo</td>
      <td>74</td>
      <td>24</td>
      <td>2</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gremio</td>
      <td>73</td>
      <td>24</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Internacional</td>
      <td>66</td>
      <td>22</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bragantino</td>
      <td>61</td>
      <td>20</td>
      <td>1</td>
      <td>17</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Fortaleza</td>
      <td>61</td>
      <td>20</td>
      <td>1</td>
      <td>17</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Corinthians</td>
      <td>57</td>
      <td>18</td>
      <td>3</td>
      <td>17</td>
    </tr>
    <tr>
      <th>10</th>
      <td>America MG</td>
      <td>52</td>
      <td>17</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Athletico-PR</td>
      <td>52</td>
      <td>17</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bahia</td>
      <td>51</td>
      <td>17</td>
      <td>0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Ceara</td>
      <td>50</td>
      <td>16</td>
      <td>2</td>
      <td>20</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Fluminense</td>
      <td>49</td>
      <td>16</td>
      <td>1</td>
      <td>21</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Santos</td>
      <td>48</td>
      <td>16</td>
      <td>0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Atletico GO</td>
      <td>44</td>
      <td>14</td>
      <td>2</td>
      <td>22</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Cuiaba</td>
      <td>41</td>
      <td>13</td>
      <td>2</td>
      <td>23</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Juventude</td>
      <td>40</td>
      <td>13</td>
      <td>1</td>
      <td>24</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Sport Recife</td>
      <td>29</td>
      <td>9</td>
      <td>2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Chapecoense-SC</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2021.to_excel('previsao2021.xlsx', index=True)
```

### 6) Considerações finais

O modelo final obteve uma acurácia de 52.11%. O resultado pode ser considerado razoável, dado a dificuldade que é realizar esse tipo de previsão. Pontos positivos e negativos do modelo:
<br/>
Como positivo podemos considerar:
* O modelo consegue dizer de maneira geral quais são os 'níveis' das equipes. Por exemplo: Ele consegue dizer que Flamego RJ, Atlético MG e Palmeiras são as melhores equipes, tanto em 2020 quanto em 2021. E quando olhamos para o passado, em 2013 o Cruzeiro e Corinthians estavam entre os melhores. O que se confirma se olharmos a tabela daquele ano.
* Além disso podemos notar que o modelo consegue dizer com precisão quaise são os clubes que lutam para não cair. Exemplo de 2020 e 2021, onde o modelo acertou 5 dos 8 clubes que foram rebaixados nestes dois anos, e quase acertou mais dois. Botafogo RJ em 2020 e Juventude em 2021.
* O modelo foi bom em dizer quais foram as equipes que figuraram entre os 10 melhores do Brasil, acertando em 7 dos 10 casos.
<br/><br/>

Como avaliação negativa do modelo podemos considerar:
* O modelo não foi bom em dizer exatamente qual a posição que a equipe terminou o campeonato.
* O modelo também previu poucos empates, o que foi bastante negativo, dado que as equipes de meio de tabela empatam bastante seus jogos.


#### 6.1) Sugestões para estudos futuros

Como estudos futuros e sugestões para quem quer melhorar o modelo a fim de obter uma melhora acurácia e aprimorar os conhecimentos sobre cada etapa e sobre Machine Learning de maneira geral cito:
* PRÉ PROCESSAMENTO: Realizar um pré processamento diferente do que foi feito, por exemplo: Mapear os dados de maneira diferente, ou ainda padronizar os dados seguindo outro algoritmo diferente dos que foram apresentados.
* SELEÇÃO DE ATRIBUTOS: Será que a data específica considerando o dia e o mês faz diferença? E o horário que o jogo aconteceu? Será que algum clube se sai melhor jogando na quarta-feira ou no domingo (dias mais comuns para partidas do campeonato brasileiro)? Talvez aplicar uma técnica diferente do SelectKBest para selecionar os melhores atributos possa indiciar atributos melhores.
* ALGORITMOS UTILIZADOS: Foram utilizados os principais algoritmos, mas ainda poderíamos utilizar outros algoritmos, além de alguns métodos focados na otimização dos algoritmos, tais como: Inserir mais parâmetros para testagem em cada algoritmo, utilizar o ExtraTreeClassifier, utilizar a otimização dos parâmetros com o AdaBoost ou o GradientBoosting.
* DEEP LEARNING!?: Será que aplicar esses dados em uma rede neural profunda com o Tensorflow pode prever um resultado melhor?

