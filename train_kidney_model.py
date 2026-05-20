#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# -------- Configuración Actualizada --------
# Incluimos la columna binaria de la especie en el entrenamiento
FEATURES = ["Es_Felino", "Edad", "Peso", "Creatinina", "Urea", "BUN", "SDMA"]
LABELS = ["Bajo", "Medio", "Alto"]


# -------- Generación de datos sintéticos --------
def generate_example_data(n=2000):
    np.random.seed(42)
    
    # 50% Caninos (0) y 50% Felinos (1) para balancear el algoritmo
    es_felino = np.random.choice([0, 1], size=n)
    
    data = {
        "Es_Felino": es_felino,
        "Edad": np.random.randint(1, 17, n),
        # Distribución de peso coherente por especie
        "Peso": np.where(es_felino == 1, np.random.uniform(2.0, 7.5, n), np.random.uniform(3.0, 42.0, n)),
        "Creatinina": np.random.uniform(0.4, 7.0, n),
        "Urea": np.random.uniform(15, 195, n),
        "BUN": np.random.uniform(6, 90, n),
        "SDMA": np.random.uniform(4, 48, n),
    }
    df = pd.DataFrame(data)

    # Clasificación estricta basada en guías clínicas internacionales IRIS
    def classify_row(row):
        score = 0
        crea = row["Creatinina"]
        sdma = row["SDMA"]
        bun = row["BUN"]
        urea = row["Urea"]
        
        if row["Es_Felino"] == 1:  # 🐱 ESTÁNDARES FILINOS
            if crea >= 2.9 or sdma >= 26 or urea > 100:
                return 2  # Alto (Estadio IRIS 3 o 4 / Emergencia Urémica)
            if 1.6 <= crea < 2.9 or 15 <= sdma < 26:
                score += 1.2
            if bun > 30 or urea > 60:
                score += 0.5
        else:                      # 🐶 ESTÁNDARES CANINOS
            if crea >= 3.5 or sdma >= 35 or urea > 120:
                return 2  # Alto (Falla Renal Avanzada Crítica)
            if 1.4 <= crea < 3.5 or 15 <= sdma < 35:
                score += 1.0
            if bun > 27 or urea > 50:
                score += 0.4

        # Introducción de ruido biológico para matizar los bordes algorítmicos
        ruido = np.random.choice([-0.2, 0.0, 0.2])
        final_score = score + ruido

        if final_score < 0.9:
            return 0  # Bajo
        elif final_score < 1.9:
            return 1  # Medio
        else:
            return 2  # Alto

    df["Riesgo_Renal"] = df.apply(classify_row, axis=1)

    print("\n📊 Nueva Distribución Clatibrada de Clases (IRIS):")
    print(df["Riesgo_Renal"].value_counts())

    df.to_csv("kidney_risk_example.csv", index=False)
    print("✅ Archivo 'kidney_risk_example.csv' actualizado con éxito.")
    return df


# -------- Entrenamiento --------
def train_model(df):
    X = df[FEATURES]
    y = df["Riesgo_Renal"]

    # División balanceada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelo con regularización leve para calibrar probabilidades neuronales/árboles
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    print("\n📊 Métricas de Desempeño del Nuevo Modelo Calibrado:")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=LABELS))

    # Empaquetado del artefacto
    bundle = {
        "model": model,
        "feature_order": FEATURES
    }
    joblib.dump(bundle, "model_kidney.pkl")
    print("🎯 ¡Súper! El archivo unificado 'model_kidney.pkl' ha sido reentrenado y guardado.")


if __name__ == "__main__":
    df_sintetico = generate_example_data(2500)
    train_model(df_sintetico)
