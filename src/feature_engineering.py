import pandas as pd

TARGET_MAPPING = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6,
}

INVERSE_TARGET_MAPPING = {value: key for key, value in TARGET_MAPPING.items()}

TARGET_LABELS_PT = {
    "Insufficient_Weight": "Abaixo do peso",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nível I",
    "Overweight_Level_II": "Sobrepeso nível II",
    "Obesity_Type_I": "Obesidade tipo I",
    "Obesity_Type_II": "Obesidade tipo II",
    "Obesity_Type_III": "Obesidade tipo III",
}


def classify_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "Abaixo do peso"
    if bmi < 25:
        return "Peso normal"
    if bmi < 30:
        return "Sobrepeso"
    if bmi < 35:
        return "Obesidade grau I"
    if bmi < 40:
        return "Obesidade grau II"
    return "Obesidade grau III"


def tratar_base_original(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna a base tratada com categorias preservadas e a base totalmente numérica."""
    df = pd.read_csv(path)

    df = df.rename(columns={
        "family_history": "family_history_overweight",
        "TUE": "time_technology_devices",
        "Obesity": "obesity_level",
    })

    df = df.drop_duplicates().reset_index(drop=True)

    categorical_columns = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_columns:
        df[col] = df[col].astype(str).str.strip()

    scale_columns_limits = {
        "FCVC": (1, 3),
        "NCP": (1, 4),
        "CH2O": (1, 3),
        "FAF": (0, 3),
        "time_technology_devices": (0, 2),
    }

    for col, (min_value, max_value) in scale_columns_limits.items():
        df[col] = df[col].round().clip(min_value, max_value).astype(int)

    for col in ["Age", "Height", "Weight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["BMI_category"] = df["BMI"].apply(classify_bmi)
    df["obesity_level_encoded"] = df["obesity_level"].map(TARGET_MAPPING)

    X = df.drop(columns=["obesity_level", "obesity_level_encoded"])
    y = df["obesity_level_encoded"]

    categorical_features = X.select_dtypes(include="object").columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False, dtype=int)

    df_encoded = X_encoded.copy()
    df_encoded["obesity_level_encoded"] = y

    return df, df_encoded


def preparar_linha_para_modelo(input_data: dict, feature_columns: list[str]) -> tuple[pd.DataFrame, float, str]:
    """Converte os dados do formulário Streamlit para o mesmo formato da base numérica."""
    row = pd.DataFrame([input_data])
    row["BMI"] = row["Weight"] / (row["Height"] ** 2)
    bmi_value = float(row.loc[0, "BMI"])
    bmi_category = classify_bmi(bmi_value)
    row["BMI_category"] = bmi_category

    row_encoded = pd.get_dummies(row, drop_first=False, dtype=int)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)

    return row_encoded, bmi_value, bmi_category


def traduzir_classe(encoded_value: int) -> tuple[str, str]:
    label_original = INVERSE_TARGET_MAPPING[int(encoded_value)]
    label_pt = TARGET_LABELS_PT[label_original]
    return label_original, label_pt


def recomendacao_negocio(label_original: str) -> str:
    if label_original in ["Insufficient_Weight"]:
        return "Atenção para possível baixo peso. Recomenda-se avaliação clínica e nutricional."
    if label_original in ["Normal_Weight"]:
        return "Perfil compatível com peso normal. Recomenda-se manter acompanhamento preventivo."
    if label_original in ["Overweight_Level_I", "Overweight_Level_II"]:
        return "Indício de sobrepeso. Recomenda-se orientação nutricional e incentivo a hábitos saudáveis."
    return "Indício de obesidade. Recomenda-se avaliação médica, nutricional e acompanhamento multiprofissional."
