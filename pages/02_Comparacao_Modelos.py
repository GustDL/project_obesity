from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"

st.set_page_config(page_title="Comparação de Modelos", page_icon="🤖", layout="wide")

st.title("🤖 Comparação dos Modelos de Machine Learning")

results = pd.read_csv(ARTIFACTS_DIR / "comparacao_modelos_df_numerico.csv")
st.dataframe(results, use_container_width=True)

fig = px.bar(
    results.sort_values("cv_f1_macro_media", ascending=False),
    x="modelo",
    y=["cv_f1_macro_media", "teste_f1_macro"],
    barmode="group",
    title="F1 macro: validação cruzada vs teste",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
O critério principal de escolha foi o **F1 macro médio em validação cruzada**, pois a variável alvo possui múltiplas classes.  
O teste final foi usado como confirmação do desempenho do modelo escolhido.
"""
)
