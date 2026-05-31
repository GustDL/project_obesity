#  Sistema Preditivo de Nível de Obesidade

> Ferramenta de apoio à triagem clínica baseada em Machine Learning — **não substitui avaliação médica profissional.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)


---

## Visão Geral

Este projeto entrega um **sistema preditivo de classificação de obesidade** voltado ao apoio de equipes médicas na etapa de triagem de pacientes. A partir de variáveis clínicas e comportamentais, o modelo classifica o nível de obesidade do paciente em múltiplas categorias, com alta precisão e explicabilidade.

O sistema inclui:

- Pipeline de Machine Learning com pré-processamento, treinamento e comparação de modelos
- Dashboard analítico interativo em Streamlit
- Relatórios de performance exportáveis (classification report, matriz de confusão, importância de features)

---

## Resultados do Modelo

| Métrica | Valor |
|---|---|
| Base de dados | `obesity_tratada_numerica.csv` |
| Total de registros | 2.087 |
| Features utilizadas | 38 |
| Variável alvo | `obesity_level_encoded` |
| **Melhor modelo** | **Random Forest** |
| F1 Macro (validação cruzada) | **0.9804** |
| Accuracy (teste) | **0.9904** |
| F1 Macro (teste) | **0.9900** |

---

## Estrutura do Projeto

```text
projeto_obesity_final_refeito/
│
├── app.py                          # Entrada principal do Streamlit
├── requirements.txt
│
├── data/
│   ├── Obesity.csv                             # Base original
│   ├── obesity_tratada_com_categoricas.csv     # Pós-limpeza (com categorias)
│   └── obesity_tratada_numerica.csv            # Base final para modelagem (One-Hot)
│
├── artifacts/
│   ├── melhor_pipeline_obesity_numerico.joblib     # Pipeline treinado
│   ├── comparacao_modelos_df_numerico.csv          # Comparativo de modelos
│   ├── classification_report_melhor_modelo.csv
│   ├── confusion_matrix_melhor_modelo.csv
│   ├── feature_importance_melhor_modelo.csv
│   ├── feature_columns.json
│   └── resumo_projeto.json
│
├── src/
│   ├── tratamento_dados.py         # Limpeza e pré-processamento
│   ├── train_model.py              # Treinamento e avaliação dos modelos
│   └── feature_engineering.py     # Engenharia de features
│
├── pages/
│   ├── 01_Dashboard_Analitico.py
│   └── 02_Comparacao_Modelos.py
│
├── notebooks/
│   └── pipeline_ml_df_numerico_etapa_a_etapa.ipynb
│
└── docs/
    ├── relatorio_entrega.txt
    ├── roteiro_video.md
    └── explicacao_pipeline.md
```

---

## Pipeline de Machine Learning

Esta versão utiliza a base **já tratada e totalmente numérica** (`obesity_tratada_numerica.csv`), com variáveis categóricas convertidas via **One-Hot Encoding** em etapa anterior à modelagem.

A pipeline de ML aplica, em sequência:

1. **Imputação de valores ausentes** — estratégia pela mediana
2. **Padronização** — `StandardScaler`
3. **Treinamento e comparação de modelos** — com validação cruzada estratificada


---

## Dependências Principais

| Biblioteca | Uso |
|---|---|
| `scikit-learn` | Pipeline de ML, modelos, métricas |
| `pandas` | Manipulação de dados |
| `numpy` | Operações numéricas |
| `streamlit` | Interface web interativa |
| `joblib` | Serialização do pipeline treinado |
| `matplotlib` / `seaborn` | Visualizações |

> Todas as versões estão especificadas em `requirements.txt`.

