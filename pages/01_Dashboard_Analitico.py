from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

st.set_page_config(page_title="Dashboard Analítico", page_icon="📊", layout="wide")

st.title("📊 Dashboard Analítico — Obesidade")
st.markdown("Visão analítica dos principais fatores associados ao nível de obesidade na base estudada.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "obesity_tratada_com_categoricas.csv")

@st.cache_data
def load_results():
    return pd.read_csv(ARTIFACTS_DIR / "comparacao_modelos_df_numerico.csv")

df = load_data()
results = load_results()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", f"{df.shape[0]:,}".replace(",", "."))
col2.metric("Colunas", df.shape[1])
col3.metric("Média de idade", f"{df['Age'].mean():.1f}")
col4.metric("IMC médio", f"{df['BMI'].mean():.1f}")

st.divider()

left, right = st.columns(2)

with left:
    obesity_counts = df["obesity_level"].value_counts().reset_index()
    obesity_counts.columns = ["obesity_level", "quantidade"]
    fig = px.bar(
        obesity_counts,
        x="obesity_level",
        y="quantidade",
        title="Distribuição dos níveis de obesidade",
    )
    fig.update_layout(xaxis_title="Nível de obesidade", yaxis_title="Quantidade")
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.box(
        df,
        x="obesity_level",
        y="BMI",
        title="Distribuição de IMC por classe de obesidade",
    )
    fig.update_layout(xaxis_title="Nível de obesidade", yaxis_title="IMC")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    fig = px.histogram(
        df,
        x="family_history_overweight",
        color="obesity_level",
        barmode="group",
        title="Histórico familiar vs nível de obesidade",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.histogram(
        df,
        x="FAVC",
        color="obesity_level",
        barmode="group",
        title="Consumo frequente de alimentos calóricos vs nível de obesidade",
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:
    fig = px.box(
        df,
        x="obesity_level",
        y="FAF",
        title="Atividade física por nível de obesidade",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    top_results = results.sort_values("cv_f1_macro_media", ascending=False)
    fig = px.bar(
        top_results,
        x="modelo",
        y="cv_f1_macro_media",
        title="Comparação dos modelos por F1 macro em validação cruzada",
    )
    fig.update_layout(xaxis_title="Modelo", yaxis_title="F1 macro médio")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Principais insights para a equipe médica")
st.markdown(
    """
- O IMC é uma variável altamente associada à classe de obesidade, pois resume a relação entre peso e altura.
- Histórico familiar e consumo frequente de alimentos calóricos aparecem como variáveis relevantes para segmentar perfis de risco.
- A frequência de atividade física ajuda a diferenciar perfis comportamentais e pode apoiar recomendações preventivas.
- O modelo deve ser usado como apoio à triagem e priorização, não como diagnóstico médico isolado.
"""
)
