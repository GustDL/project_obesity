from pathlib import Path
from src.feature_engineering import tratar_base_original

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

if __name__ == "__main__":
    df_tratado, df_numerico = tratar_base_original(DATA_DIR / "Obesity.csv")
    df_tratado.to_csv(DATA_DIR / "obesity_tratada_com_categoricas.csv", index=False)
    df_numerico.to_csv(DATA_DIR / "obesity_tratada_numerica.csv", index=False)
    print("Tratamento finalizado.")
    print(f"Base categórica: {df_tratado.shape}")
    print(f"Base numérica: {df_numerico.shape}")
