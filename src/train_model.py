from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_DIR / "obesity_tratada_numerica.csv")

    target = "obesity_level_encoded"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    feature_columns = X.columns.tolist()
    (ARTIFACTS_DIR / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=80, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM RBF": SVC(kernel="rbf", probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []
    pipelines = {}

    for model_name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1_macro": "f1_macro",
                "precision_macro": "precision_macro",
                "recall_macro": "recall_macro",
            },
            n_jobs=1,
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        results.append({
            "modelo": model_name,
            "cv_accuracy_media": scores["test_accuracy"].mean(),
            "cv_accuracy_desvio": scores["test_accuracy"].std(),
            "cv_f1_macro_media": scores["test_f1_macro"].mean(),
            "cv_f1_macro_desvio": scores["test_f1_macro"].std(),
            "cv_precision_macro_media": scores["test_precision_macro"].mean(),
            "cv_recall_macro_media": scores["test_recall_macro"].mean(),
            "teste_accuracy": accuracy_score(y_test, pred),
            "teste_f1_macro": f1_score(y_test, pred, average="macro"),
            "teste_precision_macro": precision_score(y_test, pred, average="macro"),
            "teste_recall_macro": recall_score(y_test, pred, average="macro"),
        })
        pipelines[model_name] = pipe

    results_df = pd.DataFrame(results).sort_values(
        by=["cv_f1_macro_media", "teste_f1_macro"],
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = results_df.loc[0, "modelo"]
    best_pipeline = pipelines[best_model_name]
    best_pred = best_pipeline.predict(X_test)

    class_names = [
        "Insufficient_Weight",
        "Normal_Weight",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
        "Overweight_Level_I",
        "Overweight_Level_II",
    ]

    pd.DataFrame(
        classification_report(y_test, best_pred, output_dict=True)
    ).transpose().to_csv(ARTIFACTS_DIR / "classification_report_melhor_modelo.csv")

    pd.DataFrame(confusion_matrix(y_test, best_pred)).to_csv(
        ARTIFACTS_DIR / "confusion_matrix_melhor_modelo.csv",
        index=False,
    )

    results_df.to_csv(ARTIFACTS_DIR / "comparacao_modelos_df_numerico.csv", index=False)
    joblib.dump(best_pipeline, ARTIFACTS_DIR / "melhor_pipeline_obesity_numerico.joblib")

    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi.to_csv(ARTIFACTS_DIR / "feature_importance_melhor_modelo.csv", index=False)

    summary = {
        "base_modelagem": "data/obesity_tratada_numerica.csv",
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "features": len(feature_columns),
        "target": target,
        "melhor_modelo": best_model_name,
        "cv_f1_macro": float(results_df.loc[0, "cv_f1_macro_media"]),
        "teste_accuracy": float(results_df.loc[0, "teste_accuracy"]),
        "teste_f1_macro": float(results_df.loc[0, "teste_f1_macro"]),
    }

    (ARTIFACTS_DIR / "resumo_projeto.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Treinamento finalizado.")
    print(results_df)
    print(f"Melhor modelo: {best_model_name}")


if __name__ == "__main__":
    main()
