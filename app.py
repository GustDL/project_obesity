from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.feature_engineering import preparar_linha_para_modelo, traduzir_classe, recomendacao_negocio

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide",
)

ARTIFACTS_DIR = ROOT / "artifacts"

@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS_DIR / "melhor_pipeline_obesity_numerico.joblib")

@st.cache_data
def load_feature_columns():
    return json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text(encoding="utf-8"))

model = load_model()
feature_columns = load_feature_columns()

st.title("🏥 Sistema Preditivo de Nível de Obesidade")
st.markdown(
    "Aplicação de apoio à decisão médica baseada em Machine Learning. "
    "O resultado não substitui avaliação clínica, exames ou diagnóstico profissional."
)

with st.sidebar:
    st.header("Sobre o projeto")
    st.write("Modelo treinado com a base `obesity_tratada_numerica.csv`.")
    st.write("Melhor modelo: Random Forest.")
    st.write("Acesse também a página **Dashboard Analítico** no menu lateral.")

st.subheader("Entrada de dados do paciente")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gênero", ["Female", "Male"])
    age = st.slider("Idade", 14, 61, 30)
    height = st.number_input("Altura em metros", min_value=1.30, max_value=2.20, value=1.70, step=0.01)
    weight = st.number_input("Peso em kg", min_value=30.0, max_value=220.0, value=80.0, step=0.5)
    family_history = st.selectbox("Histórico familiar de excesso de peso?", ["yes", "no"])
    favc = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])

with col2:
    fcvc = st.selectbox("Consumo de vegetais nas refeições", [1, 2, 3], format_func=lambda x: {1: "1 - raramente", 2: "2 - às vezes", 3: "3 - sempre"}[x])
    ncp = st.selectbox("Número de refeições principais", [1, 2, 3, 4])
    caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Fuma?", ["no", "yes"])
    ch2o = st.selectbox("Consumo diário de água", [1, 2, 3], format_func=lambda x: {1: "1 - menos de 1 L", 2: "2 - entre 1 e 2 L", 3: "3 - mais de 2 L"}[x])

with col3:
    scc = st.selectbox("Monitora calorias?", ["no", "yes"])
    faf = st.selectbox("Frequência de atividade física", [0, 1, 2, 3], format_func=lambda x: {0: "0 - nenhuma", 1: "1 - baixa", 2: "2 - moderada", 3: "3 - alta"}[x])
    tue = st.selectbox("Tempo usando dispositivos tecnológicos", [0, 1, 2], format_func=lambda x: {0: "0 - baixo", 1: "1 - médio", 2: "2 - alto"}[x])
    calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Meio de transporte habitual", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

input_data = {
    "Gender": gender,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "family_history_overweight": family_history,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "CAEC": caec,
    "SMOKE": smoke,
    "CH2O": ch2o,
    "SCC": scc,
    "FAF": faf,
    "time_technology_devices": tue,
    "CALC": calc,
    "MTRANS": mtrans,
}

if st.button("Prever nível de obesidade", type="primary"):
    row_encoded, bmi_value, bmi_category = preparar_linha_para_modelo(input_data, feature_columns)
    pred = int(model.predict(row_encoded)[0])
    label_original, label_pt = traduzir_classe(pred)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Classe prevista", label_pt)
    c2.metric("IMC calculado", f"{bmi_value:.2f}")
    c3.metric("Categoria de IMC", bmi_category)

    st.info(recomendacao_negocio(label_original))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row_encoded)[0]
        proba_df = pd.DataFrame({
            "classe_encoded": list(range(len(probabilities))),
            "probabilidade": probabilities,
        })
        st.subheader("Distribuição de probabilidade por classe")
        st.dataframe(proba_df, use_container_width=True)

st.caption("Projeto acadêmico de Machine Learning. Use como apoio analítico, não como diagnóstico isolado.")
