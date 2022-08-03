# Titanic: Machine Learning from Disaster

#### Utilizando Machine Learning para prever quais pessoas sobreviveram ao desastre
A resolução do problema segue os seguintes passos:
* Definição do problema;
* Coletando os dados;
* Análise exploratória dos dados;
* Tratamento dos dados e valores missing;
* Modelagem;
* Previsão;

Vale ressaltar que minha referência total para este notebook veio a partir deste material: https://github.com/minsuk-heo/kaggle-titanic
O meu foco principal aqui é iniciar meus estudos no tema de análise de dados, machine learning e ciência de dados.

### 1. Definição do problema

Esta é uma competição disponível no Kaggle, o foco desta competição é introduzir estudantes no mundo da análise de dados. É uma competição simples, cujos dados para treino e teste estão disponíveis no site do Kaggle.
A idéia do desafio é a partir do conjunto de dados de treino, criar um modelo capaz de prever quais foram os sobreviventes no naufrágio do Titanic que ocorreu entre os dias de 14 e 15 de abril no ano de 1912, no atlântico norte.

### 2. Coletando os dados

A primeira etapa é carregar o conjunto de dados para uma estrutura em Python que seja capaz de manipulá-los com facilidade e flexibilidade. O pacote escolhido foi o Pandas por se tratar do pacote mais famoso e mais utilizado no mercado, principalmente para quem está iniciando os estudos agora.
Os dados estão disponíveis nesse link: https://www.kaggle.com/c/titanic/data


```python
import pandas as pd

treino = pd.read_csv('data/train.csv', sep = ',')
teste = pd.read_csv('data/test.csv', sep = ',')

type(treino), type(teste)
```




    (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)



### 3. Análise exploratória dos dados
A primeira análise a ser feita é uma análise mais com teor de ver informações gerais mais superficiais sobre os dados, sem muito compromisso.


```python
treino.head()
#Alguns valores são inteiros, outros são strings, o sexo está em string, a idade é float.
#De cara da para ver que existem dados missing ali no campo 'Cabin'.
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
treino.columns #Aqui vemos todas as colunas, é importante saber o nome delas.
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')



#### Dicionário de dados
* PassengerId
* Survived: 0 = Não; 1 = Sim
* Pclass: Classe, igual em aviões: 1st class, 2nd class e 3rd class.
* Name: Nome do indivíduo: Dá para pegar informações com o nome também.
* Sexo: Male or Female: É interessante substituir por 0 e 1, para facilitar no processo de aprendizagem.
* Age: Idade do indivíduo: Ao invés de usar o próprio valor da idade por ser interessante utilizar informações como criança, adolescente, adulto, idosos por exemplo.
* Sibsp: Quantidade de irmãs(os) e esposos/esposas.
* Parch: Quantidade de pais e filhos.
* Ticket: Número do ticket: De cara parece ser uma informação meio sem sentido.
* Fare: Seria taxa, valor do ticket.
* Cabin: Cabine.
* Embarked: Informação interessante, é o porto de embarcação: C = Cherbourg; Q = Queenstown; S = Southampton


```python
treino.shape, teste.shape #Quantidade de linhas e colunas nos conjuntos de treino e teste, respectivamente.
```




    ((891, 12), (418, 11))




```python
treino.isnull().sum() #Quantidade de valores missing (NAN) no conjunto de dados treino
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
teste.isnull().sum() #Quantidade de valores missing (NAN) no conjunto de dados teste
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64



#### Utilizando gráficos visuais para mostrar os dados


```python
import matplotlib.pyplot as plt
import numpy as np
```


```python
labels = ['Sobreviveram', 'Morreram'] #As duas possibilidades, ou sobreviveram ou morreram.
#Pegando somente os homens e mulheres que sobreviveramm
homensSobreviveram = treino.loc[(treino['Survived'] == 1) & (treino['Sex'] == 'male'), 'Sex']
mulhereSobreviveram = treino.loc[(treino['Survived'] == 1) & (treino['Sex'] == 'female'), 'Sex']

#Pegando somente os homens e mulheres que morreram
homensMorreram = treino.loc[(treino['Survived'] == 0) & (treino['Sex'] == 'male'), 'Sex']
mulheresMorreram = treino.loc[(treino['Survived'] == 0) & (treino['Sex'] == 'female'), 'Sex']

#Fazendo um array com dois elementos com as qantidades dos valores acima 
homens = np.array([len(homensSobreviveram), len(homensMorreram)])
mulheres = np.array([len(mulhereSobreviveram), len(mulheresMorreram)])
```


```python
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, homens, width, label='Homens')
rects2 = ax.bar(x + width/2, mulheres, width, label='Mulheres')

ax.set_ylabel('Quantidade')
ax.set_title('Sobreviventes e mortos por sexo')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
```




    <matplotlib.legend.Legend at 0x23d343df8b0>




    
![png](titanic-solution_files/titanic-solution_15_1.png)
    


Esse gráfico mostra que a quantidade de mulheres que sobreviveram foi muito maior que os homens.


```python
#Outra forma de fazer o calculo acima de maneira mais simples e mais automatizada 
#é criar uma função que recebe somente o nome da coluna como parâmetro
def bar_chart(coluna):
    sobreviveram = treino[treino['Survived']==1][coluna].value_counts()
    morreram = treino[treino['Survived']==0][coluna].value_counts()
    df = pd.DataFrame([sobreviveram, morreram])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(6,4))
```


```python
bar_chart('Sex') #O gráfico confirma que mais mulheres sobreviveram do que homens.
```


    
![png](titanic-solution_files/titanic-solution_18_0.png)
    



```python
bar_chart('Pclass') 
#É possível notar que proporcionalmente as pessoas da primeira classe sobreviveram mais.
#Seguido da segunda classe e depois da terceira.
```


    
![png](titanic-solution_files/titanic-solution_19_0.png)
    



```python
bar_chart('SibSp')
#As pessoas com um irmão ou esposa mais comumente sobreviveram.
#As pessoas com mais do que um irmão, esposa ou sem ninguém mais comumente morreram.
```


    
![png](titanic-solution_files/titanic-solution_20_0.png)
    



```python
bar_chart('Parch')
#Praticamente metade das pessoas com pais/filhos abordo sobreviveram.
#Enquanto pessoas sem ninguém mais comumente morreram.
```


    
![png](titanic-solution_files/titanic-solution_21_0.png)
    



```python
bar_chart('Embarked')
```


    
![png](titanic-solution_files/titanic-solution_22_0.png)
    


### 4. Tratando dos dados e valores missing

Essa etapa consiste em manipular os dados para que passem de dados strings e numéricos para dados categoricos, bem como aplicar técnicas para tratar dos dados com valores NAN (Missing).


```python
from IPython.display import Image

Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")
```




<img src="https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w"/>



Com essa imagem é possível observar que as pessoas que ficaram na primeira classe ficaram em posições mais privilegiadas no momento do naufrágio, com isso é possível concluir que Pclass é um atributo muito importante para o classificador.

#### 4.1 Title e Name


```python
treino_teste = [treino, teste] #Uma lista com o Dataframe de treino e teste.

#Para cada dataframe na lista, insira a coluna Título com a string extraída da coluna Name utilizando expressão regular
#Essa expressão regular extrai qualquer string antes de um ponto. 
#Exemplo: "Braund, Mr. Owen Harris" será extraído somente o "Mr." nesse caso.
for df in treino_teste:
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
```


```python
treino['Title'].value_counts()
```




    Mr          517
    Miss        182
    Mrs         125
    Master       40
    Dr            7
    Rev           6
    Mlle          2
    Major         2
    Col           2
    Countess      1
    Capt          1
    Ms            1
    Sir           1
    Lady          1
    Mme           1
    Don           1
    Jonkheer      1
    Name: Title, dtype: int64




```python
teste['Title'].value_counts()
```




    Mr        240
    Miss       78
    Mrs        72
    Master     21
    Col         2
    Rev         2
    Ms          1
    Dr          1
    Dona        1
    Name: Title, dtype: int64



Nota-se que a quantidade de Mr, Miss, Mrs e Master estão bem acima das outras categorias, logo, é possível fazer um mapeamento como sendo:
* Mr: 0;
* Miss: 1;
* Mrs: 2;
* Master: 3;
* Others: 4;


```python
title_mapping = {
    'Mr':0,
    'Miss':1,
    'Mrs':2,
    'Master':3,
    'Dr':4, 'Rev':4, 'Major':4, 'Mlle':4, 'Col':4, 'Don':4, 'Sir':4,
    'Countess':4, 'Lady':4, 'Ms':4, 'Capt':4, 'Jonkheer':4, 'Mme':4
}

for df in treino_teste:
    df['Title'] = df['Title'].map(title_mapping)
```


```python
#Preenchendo os valores que ficaram como nulos.
treino['Title'].fillna(4, inplace=True)
teste['Title'].fillna(4, inplace=True)
```


```python
#excluindo a coluna name, pois a mesma não se faz mais relevante para o treinamento do modelo.
treino.drop('Name', axis=1, inplace=True)
teste.drop('Name', axis=1, inplace=True)
```

#### 4.2 Sex
Os valores do atributo Sex são male e female, é mais interessante trabalhar com dados categoricos numéricos, isto é 0 e 1, portanto:
* 0: Male
* 1: Female


```python
teste.Sex.unique()
```




    array(['male', 'female'], dtype=object)




```python
teste.Sex.value_counts()
```




    male      266
    female    152
    Name: Sex, dtype: int64




```python
sex_mapping = {
    'male': 0,
    'female': 1
}

for df in treino_teste:
    df['Sex'] = df['Sex'].map(sex_mapping)
```

#### 4.3 Age


```python
treino['Age'].isnull().sum(), teste['Age'].isnull().sum()
```




    (177, 86)




```python
treino['Age'].fillna(treino.groupby('Title')['Age'].transform('median'), inplace=True)
teste['Age'].fillna(teste.groupby('Title')['Age'].transform('median'), inplace=True)
```


```python
treino.head(30)
treino.groupby("Title")["Age"].transform("median")
```




    0      30.0
    1      35.0
    2      21.0
    3      35.0
    4      30.0
           ... 
    886    44.5
    887    21.0
    888    21.0
    889    30.0
    890    30.0
    Name: Age, Length: 891, dtype: float64



O próximo passo vai ser converter a variável Age de um dado numérico para um dado categórico. Vai ficar:
* Criança: 0,
* Adolescente: 1,
* Jovem: 2
* Adulto: 3,
* Meia idade: 4,
* Idoso: 5


```python
for df in treino_teste:
    df['Age'] = pd.cut(df['Age'], [0, 13, 18, 25, 35, 60,130], labels=[0,1,2,3,4,5])
```


```python
teste.Age.unique()
```




    [3, 4, 5, 2, 1, 0]
    Categories (6, int64): [0 < 1 < 2 < 3 < 4 < 5]




```python
teste.Age.value_counts()
```




    3    142
    4    105
    2    102
    0     32
    1     26
    5     11
    Name: Age, dtype: int64




```python
teste.groupby(['Age']).size().plot.bar(color = 'green')
```




    <AxesSubplot:xlabel='Age'>




    
![png](titanic-solution_files/titanic-solution_46_1.png)
    


#### 4.4 Embarked
Informação referente ao porto de embarcação. Dá para trabalhar nessa informação.


```python
Pclass1 = treino[treino['Pclass']==1]['Embarked'].value_counts()
Pclass2 = treino[treino['Pclass']==2]['Embarked'].value_counts()
Pclass3 = treino[treino['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
```




    <AxesSubplot:>




    
![png](titanic-solution_files/titanic-solution_48_1.png)
    


* S = Southampton
* C = Cherbourg;
* Q = Queenstown; 
<br> Pelo fato de a maioria das pessoas terem embarcado no porto de Southampton, vamos assumir que os valores nulos nesta coluna também embarcaram em Southampton.


```python
for df in treino_teste:
    df['Embarked'] = df['Embarked'].fillna('S')
```


```python
#Mapeamento dos dados, transformando valores char em números.
embarked_mapping = {
    'S': 0,
    'C': 1,
    'Q': 2
}

for df in treino_teste:
    df['Embarked'] = df['Embarked'].map(embarked_mapping)
```

#### 4.5 Fare
Fare é a taxa do ticket, tem alguns valores missing, vamos preencher com a mediana do conjunto.


```python
treino['Fare'].describe()
```




    count    891.000000
    mean      32.204208
    std       49.693429
    min        0.000000
    25%        7.910400
    50%       14.454200
    75%       31.000000
    max      512.329200
    Name: Fare, dtype: float64




```python
treino['Fare'].fillna(treino.groupby('Pclass')['Fare'].transform('median'), inplace=True)
teste['Fare'].fillna(teste.groupby('Pclass')['Fare'].transform('median'), inplace=True)
```


```python
tempTeste = teste.copy()
tempTreino = treino.copy()
tempTeste.fillna(tempTeste.groupby('Pclass')['Fare'].transform('median'), inplace=True)
tempTreino.fillna(tempTreino.groupby('Pclass')['Fare'].transform('median'), inplace=True)
```


```python
#Transformando a variável Fare em variável categórica
for df in treino_teste:
    df['Fare'] = pd.cut(df['Fare'], [-1, 17, 30, 100, 9999], labels=[0,1,2,3])
```

#### Cabin


```python
#Vamos pegar somente a primeira letra da frase Cabin
for df in treino_teste:
    df['Cabin'] = df['Cabin'].str[:1]
```


```python
Pclass1 = treino[treino['Pclass']==1]['Cabin'].value_counts()
Pclass2 = treino[treino['Pclass']==2]['Cabin'].value_counts()
Pclass3 = treino[treino['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
```




    <AxesSubplot:>




    
![png](titanic-solution_files/titanic-solution_59_1.png)
    



```python
#Fazendo o mapeamento das cabines também.
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for df in treino_teste:
    df['Cabin'] = df['Cabin'].map(cabin_mapping)
```


```python
#É possível notar que ainda existem muitos valores nulos nesta informação
treino.Cabin.isnull().sum(), teste.Cabin.isnull().sum()
```




    (687, 327)




```python
#Preenchendo os valores nulos do atributo
treino['Cabin'].fillna(treino.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
teste['Cabin'].fillna(teste.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
```


```python
treino.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>2</td>
      <td>0.8</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>2</td>
      <td>0.8</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### 4.6 FamilySize
Os outros atributos que também dá para trabalhar são SibSp e Parch
* SibSp: Número de irmãos e esposa(o)s
* Parch: O Parch é o número de pais e filhos
<br>Como ambos os atriutos se tratam de familiares, é possível juntar em um único atributo.


```python
treino['FamilySize'] = treino['SibSp'] + treino['Parch'] + 1
teste['FamilySize'] = teste['SibSp'] + teste['Parch'] + 1
#Mais um porque se trata do próprio indivíduo.
```


```python
#A família com maior quantidade de membros a bordo tinha 11 pessoas.
#Para não trabalhar com valores muito grandes, vamos categorizar em valores mais baixos
treino.FamilySize.max(), teste.FamilySize.max()
```




    (11, 11)




```python
family_mapping = {
    1 : 0.0, 2 : 0.3, 3 : 0.6, 4 : 0.9,
    5 : 1.2, 6 : 1.5, 7 : 1.8, 8 : 2.1,
    9 : 2.4, 10: 2.7, 11: 3.0,
}

for df in treino_teste:
    df['FamilySize'] = df['FamilySize'].map(family_mapping)
```

#### 4.7 Removendo atributos desnecessários
* Ticket: Atributo que não segue nenhuma lógica, portanto é dispensável;
* SibSp e Parch: Esses atributos foram unificados no atributo FamilySize, portanto podem ser excluídos;
* PassengerId: No conjunto de treino essa coluna não se faz necessária.


```python
features_drop = ['Ticket', 'SibSp', 'Parch']
treino = treino.drop(features_drop, axis=1)
teste = teste.drop(features_drop, axis=1)
treino = treino.drop(['PassengerId'], axis=1)
```


```python
#Vamos separar o conjunto de treinos em atributos e classe.
#Treino_data é o Pandas.DataFrame que vai servir de entrada para o modelo.
#Target é do tipo Pandas.Series que será a variável cujo modelo vai tentar prever o valor.

treino_data = treino.drop('Survived', axis=1)
target = treino['Survived']

type(treino_data), type(target)
```




    (pandas.core.frame.DataFrame, pandas.core.series.Series)



### 5. Modelagem


```python
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
```

##### 5.1 Utilizando Tensorflow
A minha primeira alternativa é tentar realizar o treinamento utilizando redes neurais profundas. Para isso vou utilizar o Tensorflow 2.x


```python
import sklearn
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
```


```python
npTreino = treino_data.to_numpy()
npTarget = target.to_numpy()
# Create a object
encoder = LabelEncoder()
# Apply the fit_transform
target_encoded = encoder.fit_transform(npTarget)
# Apply the One-Hot-Encoding on labels
target_encoded = tf.keras.utils.to_categorical(target_encoded)
```


```python
# We split the training data into two samples, training and validation
X_train, X_valid, y_train, y_valid = train_test_split(npTreino, npTarget)
```

#### _Preparing the Data_


```python
# Hyperparameters
batch_size = 32
autotune = tf.data.experimental.AUTOTUNE
```

#### _Model Building_


```python
#Pre trained model?
```


```python
model = tf.keras.models.Sequential()
```


```python
target_encoded
```




    array([[1., 0.],
           [0., 1.],
           [0., 1.],
           ...,
           [1., 0.],
           [0., 1.],
           [1., 0.]], dtype=float32)




```python
#model.add(tf.keras.Input(shape=(8,)))
model.add(tf.keras.layers.Dense(units=8, activation='relu', input_shape=(8, )))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=8, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))        
```


```python
model.summary()
```

    Model: "sequential_9"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_37 (Dense)            (None, 8)                 72        
                                                                     
     dropout_21 (Dropout)        (None, 8)                 0         
                                                                     
     dense_38 (Dense)            (None, 10)                90        
                                                                     
     dense_39 (Dense)            (None, 10)                110       
                                                                     
     dense_40 (Dense)            (None, 10)                110       
                                                                     
     dense_41 (Dense)            (None, 10)                110       
                                                                     
     dense_42 (Dense)            (None, 10)                110       
                                                                     
     dropout_22 (Dropout)        (None, 10)                0         
                                                                     
     dense_43 (Dense)            (None, 8)                 88        
                                                                     
     dropout_23 (Dropout)        (None, 8)                 0         
                                                                     
     dense_44 (Dense)            (None, 2)                 18        
                                                                     
    =================================================================
    Total params: 708
    Trainable params: 708
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
```


```python
# Hyperparameters
batch_size = 32
epochs = 500
lr = 0.01
beta1 = 0.9
beta2 = 0.9
ep = 1e-08
```


```python
# Model compilation
model.compile(optimizer = Adam(learning_rate = lr, 
                                beta_1 = beta1, 
                                beta_2 = beta2, 
                                epsilon = ep),
               loss = 'sparse_categorical_crossentropy', 
               metrics=['sparse_categorical_accuracy'])
```


```python
checkpoint1 = tf.keras.callbacks.ModelCheckpoint("best_model.h5", 
                                                verbose = 1, 
                                                save_best_only = True, 
                                                save_weights_only = True)

# Checkpoint
checkpoint2 = tf.keras.callbacks.ModelCheckpoint("last_model.h5", 
                                                verbose = 0, 
                                                save_best_only = False,
                                                save_weights_only = True,
                                                save_freq='epoch')

# Early stop
early_stop = tf.keras.callbacks.EarlyStopping(patience = 50) 
```


```python
history = model.fit(npTreino, npTarget, 
                    steps_per_epoch = len(npTreino)//batch_size, 
                    epochs = epochs, 
                    callbacks = [checkpoint1, checkpoint2, early_stop]) 
```

    


```python
#model.fit(npTreino, npTarget, epochs=100)
```

    Epoch 1/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.9510 - sparse_categorical_accuracy: 0.6251
    Epoch 2/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.7797 - sparse_categorical_accuracy: 0.6128
    Epoch 3/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.6825 - sparse_categorical_accuracy: 0.6442
    Epoch 4/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.6552 - sparse_categorical_accuracy: 0.6128
    Epoch 5/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.6439 - sparse_categorical_accuracy: 0.6341
    Epoch 6/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.6128 - sparse_categorical_accuracy: 0.6734
    Epoch 7/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.6119 - sparse_categorical_accuracy: 0.6779
    Epoch 8/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5886 - sparse_categorical_accuracy: 0.6981
    Epoch 9/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5868 - sparse_categorical_accuracy: 0.6936A: 0s - loss: 0.6015 - sparse_categorical_accuracy: 0.67
    Epoch 10/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5792 - sparse_categorical_accuracy: 0.7172
    Epoch 11/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5664 - sparse_categorical_accuracy: 0.7205
    Epoch 12/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5561 - sparse_categorical_accuracy: 0.7273
    Epoch 13/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5666 - sparse_categorical_accuracy: 0.7318
    Epoch 14/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5681 - sparse_categorical_accuracy: 0.7228
    Epoch 15/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5540 - sparse_categorical_accuracy: 0.7284
    Epoch 16/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5298 - sparse_categorical_accuracy: 0.7587
    Epoch 17/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5377 - sparse_categorical_accuracy: 0.7452
    Epoch 18/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5488 - sparse_categorical_accuracy: 0.7542
    Epoch 19/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5297 - sparse_categorical_accuracy: 0.7363
    Epoch 20/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5239 - sparse_categorical_accuracy: 0.7531
    Epoch 21/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5057 - sparse_categorical_accuracy: 0.7677
    Epoch 22/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5318 - sparse_categorical_accuracy: 0.7497
    Epoch 23/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5265 - sparse_categorical_accuracy: 0.7688
    Epoch 24/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5088 - sparse_categorical_accuracy: 0.7722
    Epoch 25/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5153 - sparse_categorical_accuracy: 0.7722
    Epoch 26/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4856 - sparse_categorical_accuracy: 0.7823
    Epoch 27/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4992 - sparse_categorical_accuracy: 0.7868
    Epoch 28/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.5020 - sparse_categorical_accuracy: 0.7856
    Epoch 29/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4956 - sparse_categorical_accuracy: 0.7733
    Epoch 30/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4916 - sparse_categorical_accuracy: 0.7834
    Epoch 31/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4705 - sparse_categorical_accuracy: 0.7856
    Epoch 32/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4856 - sparse_categorical_accuracy: 0.7969
    Epoch 33/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4788 - sparse_categorical_accuracy: 0.7879
    Epoch 34/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4785 - sparse_categorical_accuracy: 0.7890
    Epoch 35/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4785 - sparse_categorical_accuracy: 0.7991
    Epoch 36/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4798 - sparse_categorical_accuracy: 0.8002
    Epoch 37/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4701 - sparse_categorical_accuracy: 0.7845
    Epoch 38/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4616 - sparse_categorical_accuracy: 0.8013
    Epoch 39/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4682 - sparse_categorical_accuracy: 0.7935
    Epoch 40/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4561 - sparse_categorical_accuracy: 0.8070
    Epoch 41/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4678 - sparse_categorical_accuracy: 0.8103
    Epoch 42/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4571 - sparse_categorical_accuracy: 0.8013
    Epoch 43/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4566 - sparse_categorical_accuracy: 0.8036
    Epoch 44/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4692 - sparse_categorical_accuracy: 0.7991
    Epoch 45/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4687 - sparse_categorical_accuracy: 0.8047
    Epoch 46/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4682 - sparse_categorical_accuracy: 0.8103
    Epoch 47/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4620 - sparse_categorical_accuracy: 0.8036
    Epoch 48/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4554 - sparse_categorical_accuracy: 0.8148
    Epoch 49/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4542 - sparse_categorical_accuracy: 0.8013
    Epoch 50/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4469 - sparse_categorical_accuracy: 0.8294
    Epoch 51/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4443 - sparse_categorical_accuracy: 0.8204
    Epoch 52/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4521 - sparse_categorical_accuracy: 0.8070
    Epoch 53/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4478 - sparse_categorical_accuracy: 0.8103
    Epoch 54/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4551 - sparse_categorical_accuracy: 0.8103
    Epoch 55/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4493 - sparse_categorical_accuracy: 0.8092
    Epoch 56/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4573 - sparse_categorical_accuracy: 0.8092
    Epoch 57/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4332 - sparse_categorical_accuracy: 0.8238
    Epoch 58/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4431 - sparse_categorical_accuracy: 0.8227
    Epoch 59/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4484 - sparse_categorical_accuracy: 0.8137
    Epoch 60/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4420 - sparse_categorical_accuracy: 0.8238
    Epoch 61/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4428 - sparse_categorical_accuracy: 0.8238
    Epoch 62/100
    28/28 [==============================] - 0s 6ms/step - loss: 0.4402 - sparse_categorical_accuracy: 0.8294
    Epoch 63/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4571 - sparse_categorical_accuracy: 0.7991
    Epoch 64/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4357 - sparse_categorical_accuracy: 0.8171
    Epoch 65/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4294 - sparse_categorical_accuracy: 0.8272
    Epoch 66/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4344 - sparse_categorical_accuracy: 0.8204
    Epoch 67/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4401 - sparse_categorical_accuracy: 0.8103
    Epoch 68/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4294 - sparse_categorical_accuracy: 0.8316
    Epoch 69/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4293 - sparse_categorical_accuracy: 0.8249
    Epoch 70/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4294 - sparse_categorical_accuracy: 0.8182
    Epoch 71/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4411 - sparse_categorical_accuracy: 0.8081
    Epoch 72/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4618 - sparse_categorical_accuracy: 0.8159
    Epoch 73/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4298 - sparse_categorical_accuracy: 0.8215
    Epoch 74/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4306 - sparse_categorical_accuracy: 0.8316
    Epoch 75/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4412 - sparse_categorical_accuracy: 0.8148
    Epoch 76/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4341 - sparse_categorical_accuracy: 0.8148
    Epoch 77/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4303 - sparse_categorical_accuracy: 0.8260
    Epoch 78/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4446 - sparse_categorical_accuracy: 0.8137
    Epoch 79/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4375 - sparse_categorical_accuracy: 0.8238
    Epoch 80/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4366 - sparse_categorical_accuracy: 0.8249
    Epoch 81/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4304 - sparse_categorical_accuracy: 0.8316
    Epoch 82/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4295 - sparse_categorical_accuracy: 0.8260
    Epoch 83/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4234 - sparse_categorical_accuracy: 0.8204
    Epoch 84/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4401 - sparse_categorical_accuracy: 0.8092
    Epoch 85/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4201 - sparse_categorical_accuracy: 0.8260
    Epoch 86/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4391 - sparse_categorical_accuracy: 0.8171
    Epoch 87/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4177 - sparse_categorical_accuracy: 0.8215
    Epoch 88/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4193 - sparse_categorical_accuracy: 0.8260
    Epoch 89/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4208 - sparse_categorical_accuracy: 0.8316
    Epoch 90/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4169 - sparse_categorical_accuracy: 0.8305
    Epoch 91/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4192 - sparse_categorical_accuracy: 0.8227
    Epoch 92/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4261 - sparse_categorical_accuracy: 0.8137
    Epoch 93/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4302 - sparse_categorical_accuracy: 0.8171
    Epoch 94/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4314 - sparse_categorical_accuracy: 0.8294
    Epoch 95/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4193 - sparse_categorical_accuracy: 0.8272
    Epoch 96/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4294 - sparse_categorical_accuracy: 0.8283
    Epoch 97/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4327 - sparse_categorical_accuracy: 0.8283
    Epoch 98/100
    28/28 [==============================] - 0s 5ms/step - loss: 0.4159 - sparse_categorical_accuracy: 0.8373
    Epoch 99/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4219 - sparse_categorical_accuracy: 0.8249
    Epoch 100/100
    28/28 [==============================] - 0s 4ms/step - loss: 0.4079 - sparse_categorical_accuracy: 0.8305
    




    <keras.callbacks.History at 0x23dfa705fa0>




```python
npTeste = teste.to_numpy()
npTreino.shape, npTeste.shape
```




    ((891, 8), (418, 9))




```python
npTeste = teste.drop(['PassengerId'], axis=1)
npTeste = npTeste.to_numpy()
```


```python
previsoes = model.predict(npTeste)
pred = np.argmax(previsoes, axis = 1) 
```


```python
submission = pd.DataFrame({
        "PassengerId": teste["PassengerId"],
        "Survived": pred
    })

submission.to_csv('submissionTensorflow.csv', index=False)
```

<p style="color: red; font-weight: bold; font-size:20px">Ainda não trabalhei muito bem com o Tensorflow, vou voltar aqui mais para frente</p>

##### 5.2 Cross Validation (K-Fold)


```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
```

###### 5.2.1 KNN
Tutorial sobre o algoritmo KNN <br>
* Link: https://inferir.com.br/artigos/algoritimo-knn-para-classificacao/ <br>

O algoritmo KNN diferentemente dos outros algoritmos que criam um modelo para a previsão, ele faz a previsão baseado na semelhança com o vizinho mais próximo, ou seja, é em tempo real. Este vizinho mais próximo é dado pela distância euclidiana dos atributos.


```python
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, treino_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
```


```python
# kNN Score
round(np.mean(score)*100, 2)
```

###### 5.2.2 Decision Tree


É um tipo de algoritmo que cria uma estrutura de árvore até que uma decisão seja feita para um determinado registro. Inicia sempre de um nó raiz, a partir daí selecionando os outros atributos e tomando decisões. 


```python
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, treino_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
```


```python
# decision tree Score
round(np.mean(score)*100, 2)
```

###### 5.2.3 Random Forest

Modelos compostos por múltiplos modelos mais fracos que são independentemente treinados e a previsão combina esse modelos de alguma forma; Criam vários modelos de árvores diferentes, selecionando diferentes atributos com diferentes configurações. A partir daí o melhor modelo será selecionado através de um processo de votação.


```python
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, treino_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
```


```python
# Random Forest Score
round(np.mean(score)*100, 2)
```

###### 5.2.4 Naive Bayes

O Naive Bayes é um tipo de classificador que desconsidera a correlação entre os atributos, cada atributo é tratado individualmente.

A resolução de problemas relacionados a texto é muito bem resolvida com a utilização do Naive Bayes. Classificação de textos, filtragem de SPAM e análise de sentimento em redes sociais são algumas das muitas aplicações para o algoritmo.


```python
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, treino_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
```


```python
# Naive Bayes Score
round(np.mean(score)*100, 2)
```

###### 5.2.5 SVM

O algoritmo Support Vector Machine traça uma reta e tenta separar linearmente e classificar o conjunto de dados. O algoritmo tenta encontrar a reta que tenha maior distância dentre as classes prevista.

<img src="attachment:58a07f05-8b87-42f2-b5d7-b7d6795da982.png" width="300"/>

Para aplicar o SVC em conjuntos de dados não linearmente separáveis é necessário configurar o parâmetro Kernel. Este parâmetro é responsável por traçar não somente retas, mas também outros tipos de linhas no conjunto de dados.
>Possibilidades: kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} 


```python
clf = SVC(kernel='poly')
scoring = 'accuracy'
score = cross_val_score(clf, treino_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
```


```python
round(np.mean(score)*100,2)
```

### 6. Testando


```python
clf = SVC(kernel='poly')
clf.fit(treino_data, target)

teste_data = teste.drop("PassengerId", axis=1).copy()
prediction = clf.predict(teste_data)
```


```python
submission = pd.DataFrame({
        "PassengerId": teste["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
```


```python
submission = pd.read_csv('submission.csv')
submission.head()
```

### 7. Referências

Como citado no começo deste notebook, eu fiz baseado nos seguintes trabalhos:
* https://github.com/minsuk-heo/kaggle-titanic/blob/master/titanic-solution.ipynb
* https://www.kaggle.com/chapagain/titanic-solution-a-beginner-s-guide?scriptVersionId=1473689
* https://olegleyz.github.io/titanic_factors.html
* https://www.codeastar.com/data-wrangling/
* https://www.ahmedbesbes.com/blog/kaggle-titanic-competition
