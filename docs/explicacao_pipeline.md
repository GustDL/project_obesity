# Explicação da Pipeline de Machine Learning

## 1. Objetivo

Criar um sistema preditivo capaz de classificar o nível de obesidade de uma pessoa com base em variáveis físicas, comportamentais e familiares.

## 2. Base utilizada

A base original é `Obesity.csv`. Após o tratamento, foi criada a base `obesity_tratada_numerica.csv`, usada diretamente na modelagem.

## 3. Tratamento dos dados

As principais etapas de tratamento foram:

1. Renomeação de colunas para melhorar a interpretação;
2. Remoção de duplicados exatos;
3. Limpeza de espaços extras em variáveis categóricas;
4. Arredondamento de variáveis de escala com ruído decimal;
5. Criação da variável `BMI`;
6. Criação da variável `BMI_category`;
7. Transformação de variáveis categóricas em numéricas com One-Hot Encoding;
8. Codificação da variável alvo como `obesity_level_encoded`.

## 4. Modelos testados

Foram comparados os seguintes algoritmos:

- Logistic Regression;
- KNN;
- Decision Tree;
- Random Forest;
- Extra Trees;
- Gradient Boosting;
- SVM RBF;
- Naive Bayes.

## 5. Critério de escolha

O principal critério de escolha foi o `f1_macro` médio em validação cruzada, pois o problema possui múltiplas classes de saída.

## 6. Melhor modelo

O melhor modelo foi **Random Forest**.

Métricas principais:

- F1 macro em validação cruzada: 0.9804
- Accuracy no teste: 0.9904
- F1 macro no teste: 0.9900

## 7. Uso no Streamlit

O app `app.py` recebe os dados do paciente em formato amigável, calcula o IMC, transforma a entrada para o mesmo formato numérico usado no treinamento e retorna a classe prevista.

## 8. Interpretação de negócio

O resultado deve apoiar triagem e tomada de decisão, indicando perfis que merecem acompanhamento médico, nutricional e multiprofissional. O modelo não substitui diagnóstico clínico.
