# Modelo Para Previsão de Doenças Usando Registros Médicos Eletrônicos utilizando Amazon SageMaker

<div style="text-align: justify; font-size: 0.9rem;">
<b>Introdução e Definição do problema</b>

<p>
    Embora seja um assunto muito comum, a pressão arterial ainda gera muitas dúvidas, principalmente em relação à hipertensão, que é uma condição de saúde em que a pressão exercida nas artérias fica acima do valor de referência considerado normal. Mas para
    entender essa alteração orgânica, é preciso saber o que é pressão arterial sistólica e diastólica. A medição da pressão arterial identifica a intensidade com que o fluxo de sangue passa pelas suas artérias. Para
    isso, os valores de referência são representados em milímetros de mercúrio (mmHg), sendo composto por duas medidas, denominadas como sistólica e diastólica.
</p>
<hr/>
<b>Objetivos do projeto</b>
<p>
    Contudo, o desenvolvimento deste projeto tem um caráter que vai além da resolução de um problema em específico. O objetivo principal é a implantação deste projeto em um ambiente em nuvem. O ambiente escolhido foi o ambiente em nuvem da Amazon AWS chamado
    de SageMaker. O Amazon SageMaker é um serviço que facilita criar, treinar e efetivar modelos de machine learning (ML), de forma rápida e confiável. Ele é totalmente gerenciado pela AWS, que remove o trabalho pesado
    de cada etapa, para que o usuário possa desenvolver modelos de ML de alta qualidade.
</p>

> Todas as definições e explicações dos conceitos e técnicas utilizados estão detalhadas em cada notebook do projeto. 

<br/>
<p>O projeto contempla:</p>
<p>O projeto está divido em partes, as três primeiras partes com foco na resolução do problema e as três últimas com foco em aplicar os conceitos avançados de Machine Learning Engineering na plataforma da Amazon. </p>


* **Parte 1**: Introdução do problema, análise descritiva e estatística dos dados e preparação/transformação dos dados</li>
* **Parte 2**: Criação do modelo base aplicando algoritmos de Machine Learning: KNN, Regressão Logística e Random Forest</li>
* **Parte 3**: Criação do modelo XGBoost utilizando os recursos da Amazon aplicando Cloud Computing</li>
---
* **Parte 4**: Ajustando o Endpoint config com o objetivo de preparar o modelo para ser exportado via API para outros sistemas</li>
* **Parte 5**: Utilização do conceito de Batch Transform para dividir o problema em pequenas partes. (Dividir para conquistar)</li>
* **Parte 6**: Explicação e aplicação da configuração dos hiperparâmetros no SageMaker. Algo semelhante ao GridSearch, mas utilizando a nuvem</li>


</div>
