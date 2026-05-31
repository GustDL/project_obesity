# Roteiro para vídeo de apresentação — 4 a 10 minutos

## 1. Abertura — 30 segundos

Apresentar o problema: prever o nível de obesidade para apoiar a equipe médica na triagem e tomada de decisão.

## 2. Base de dados — 1 minuto

Explicar que a base contém variáveis como idade, altura, peso, histórico familiar, alimentação, atividade física, consumo de água, álcool, uso de tecnologia e meio de transporte.

Destacar que a variável alvo é o nível de obesidade.

## 3. Tratamento dos dados — 1 a 2 minutos

Mostrar no notebook ou script:

- `df.head()`;
- `df.shape`;
- verificação de nulos;
- verificação de duplicados;
- tratamento das variáveis de escala;
- criação do IMC;
- transformação das variáveis categóricas em numéricas.

Explicar que a modelagem usa a base `obesity_tratada_numerica.csv`.

## 4. Pipeline e modelos — 1 a 2 minutos

Mostrar a pipeline de treino:

- separação entre X e y;
- treino e teste estratificados;
- padronização das features;
- comparação de vários algoritmos;
- escolha pelo F1 macro em validação cruzada.

Modelos testados: Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVM e Naive Bayes.

## 5. Resultado do modelo — 1 minuto

Apresentar o melhor modelo: **Random Forest**.

Métricas:

- F1 macro validação cruzada: 0.9804
- Accuracy teste: 0.9904
- F1 macro teste: 0.9900

Destacar que superou o requisito mínimo de 75%.

## 6. Aplicação Streamlit — 1 a 2 minutos

Mostrar o formulário do sistema preditivo.

Inserir um exemplo de paciente e clicar em prever.

Explicar o resultado:

- classe prevista;
- IMC calculado;
- categoria de IMC;
- recomendação de apoio à decisão.

## 7. Dashboard analítico — 1 a 2 minutos

Mostrar gráficos de distribuição de obesidade, IMC por classe, histórico familiar, consumo calórico e comparação dos modelos.

Explicar em visão de negócio: o painel ajuda a equipe médica a identificar fatores associados e perfis de maior atenção.

## 8. Fechamento — 30 segundos

Reforçar que o modelo apoia a triagem, mas não substitui avaliação médica.

Finalizar mostrando o repositório GitHub e os links da aplicação e do dashboard.
