<h1>PEOPLE ANALYTICS - PREVENDO SE O COLABORADOR VAI DEIXAR A EMPRESA</h1>

<h2>INTRODUÇÃO</h2>

<h3>O que é People Analytics</h3>

People analytics é um processo de coleta e análise de dados voltado para a gestão de pessoas em empresas. O conceito nasce a partir da ideia de big data, que consiste na coleta, armazenamento e análise de um volume imenso de dados. Com tantas informações disponíveis ou passíveis de serem coletadas, as empresas têm identificado oportunidades de aproveitá-las para melhorar seus processos.

É bom deixar claro desde o começo que o People Analytics não é uma ferramenta ou um software. Trata-se de uma metodologia cujo princípio é a coleta, a organização e a análise de dados aplicada à gestão de pessoas para se ter uma visão mais estratégica do papel de cada colaborador dentro de uma empresa. Seu objetivo é melhorar a qualidade da tomada de decisão sobre os profissionais a partir da coleta e do cruzamento de informações relacionadas a eles. Assim, é possível tanto reconhecer um funcionário que se destaca quanto identificar problemas que estejam ocorrendo, como baixa produtividade, pouco engajamento, insatisfação, alto índice de rotatividade, entre outros.

<h3>Objetivo do projeto</h3>
Este projeto tem por objetivo aplicar técnicas de análise descritiva, estatística e preditiva dos dados para entender o comportamento dos colaboradores e os principais fatores que levam o mesmo a deixar ou não a empresa. No final, espera-se um modelo preditivo capaz de prever se o colaborador vai deixar ou não a empresa e um relatório completo sobre os resultados da análise.

---

<h2>ORGANIZAÇÃO DOS DIRETÓRIOS</h2>

> **apresentacao/**: Apresentação dos resultados obtidos das análises descritiva e preditiva em formato de slides;<br/>
> **notebook/**: Arquivo jupyter-notebook python contendo em detalhes todo o projeto, metodologia, conceitos e técnicas aplicadas.<br/>
> **static/**: Arquivos CSS e Javascript utilizados para a construção do website<br/>
> **templates/**: Contém o arquivo index.HTML do website.<br/>
> **tools/**: Classe Python Colaborador, responsável por receber os parâmetros via Ajax e realizar a previsão utilizando a API.
> `Procfile`: Arquivo de configuração para o Deploy no servidor do Heroku;
> `main.py`: App Python responsável por executar a aplicação Flask;
> `requirements.txt`: Pacotes que serão instalados ao realizar o deploy no Heroku;
> `runtime.txt`: Versão Python que será instalada no servidor do Heroku;

---

<h2>TÉCNOLOGIAS UTILIZADAS</h2>

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

