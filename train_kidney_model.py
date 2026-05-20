#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# -------- Configuración --------
FEATURES = ["Edad", "Peso", "Creatinina", "Urea", "BUN", "SDMA"]
LABELS = ["Bajo", "Medio", "Alto"]


# -------- Generación de datos sintéticos --------
def generate_example_data(n=1000):
    np.random.seed(42)
    data = {
        "Edad": np.random.randint(1, 15, n),
        "Peso": np.random.uniform(1, 30, n),
        "Creatinina": np.random.uniform(0.5, 6.0, n),
        "Urea": np.random.uniform(20, 180, n),
        "BUN": np.random.uniform(10, 80, n),
        "SDMA": np.random.uniform(5, 40, n),
    }
    df = pd.DataFrame(data)

    # Clasificación simple de riesgo
    def classify_row(row):
        score = 0
        if row["Creatinina"] > 1.6: score += 1
        if row["Urea"] > 60: score += 1
        if row["BUN"] > 28: score += 1
        if row["SDMA"] > 14: score += 1

        if score == 0:
            return 0  # Bajo
        elif score == 1 or score == 2:
            return 1  # Medio
        else:
            return 2  # Alto

    df["Riesgo_Renal"] = df.apply(classify_row, axis=1)

    # Comprobar balance
    print("\n📊 Distribución de clases:")
    print(df["Riesgo_Renal"].value_counts())

    df.to_csv("kidney_risk_example.csv", index=False)
    print("✅ CSV de ejemplo guardado como kidney_risk_example.csv")
    return df


# -------- Entrenamiento --------
def train_model(df):
    X = df[FEATURES]
    y = df["Riesgo_Renal"]

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelo
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    # Reporte (forzando 3 clases)
    print("\n📊 Resultados del modelo:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=LABELS,
        zero_division=0
    ))

    # Guardar modelo
    joblib.dump({"model": model, "feature_order": FEATURES}, "model_kidney.pkl")
    print("✅ Modelo multiclase guardado como model_kidney.pkl")


# -------- Main --------
if __name__ == "__main__":
    df = generate_example_data(1000)  # más datos → mejor balance
    train_model(df)
