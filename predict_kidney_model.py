#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "model_kidney.pkl"
# Características estrictas en orden secuencial
FEATURES = ["Es_Felino", "Edad", "Peso", "Creatinina", "Urea", "BUN", "SDMA"]

RISK_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}
LABEL_TO_INDEX = {"Bajo": 0, "Medio": 1, "Alto": 2}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def safe_load_model(path):
    """
    Carga el artefacto serializado extrayendo el clasificador y su orden estructural.
    """
    art = joblib.load(path)
    if isinstance(art, dict):
        model = art.get("model") or art.get("estimator") or art.get("clf")
        feat_order = art.get("feature_order") or art.get("features") or []
    else:
        if isinstance(art, tuple) and len(art) == 2:
            model, feat_order = art
        else:
            model, feat_order = art, []
    if not feat_order:
        feat_order = FEATURES
    return model, feat_order


def _to_three_cols(proba, classes):
    """
    Formatea las matrices matemáticas para asegurar 3 salidas (Bajo, Medio, Alto)
    """
    if proba.shape[1] == 3:
        return proba
    out = np.zeros((proba.shape[0], 3))
    for i, c in enumerate(classes):
        idx = int(c)
        if 0 <= idx < 3:
            out[:, idx] = proba[:, i]
    return out


def safe_predict_proba(model, X):
    """
    Ejecuta el cálculo probabilístico matricial del modelo.
    """
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", [0, 1, 2])
    return _to_three_cols(proba, classes)


def interactive():
    print("=== Modo Interactivo Diagnóstico (Consola) ===")
    vals = {}
    
    # Capturar especie de forma numérica para pruebas manuales
    esp = input("Especie (Canino/Felino): ").strip().lower()
    vals["Es_Felino"] = 1.0 if esp == "felino" else 0.0
    
    for f in FEATURES[1:]:  # Omitir el indicador de especie ya capturado
        while True:
            try:
                vals[f] = float(input(f"{f}: ").strip())
                break
            except ValueError:
                print("❌ Entrada inválida. Digite un número válido.")
                
    model, feat_order = safe_load_model(MODEL_PATH)
    X = pd.DataFrame([vals])[feat_order]
    proba = safe_predict_proba(model, X)
    pred_class = int(np.argmax(proba, axis=1)[0])
    pred_label = INDEX_TO_LABEL.get(pred_class, str(pred_class))

    print("\n📊 Análisis del Modelo Matemático:")
    print(f"• Bajo:  {proba[0,0]*100:.1f}%")
    print(f"• Medio: {proba[0,1]*100:.1f}%")
    print(f"• Alto:  {proba[0,2]*100:.1f}%")
    print(f"📌 Predicción Final del Sistema: {pred_label}")


def main():
    ap = argparse.ArgumentParser(description="Predicción multiclase veterinaria")
    ap.add_argument("csv", nargs="?", help="Ruta al CSV")
    ap.add_argument("--interactive", action="store_true", help="Consola manual")
    args = ap.parse_args()

    if args.interactive or not args.csv:
        interactive()
    else:
        print(f"Procesando lote desde archivo: {args.csv}")


if __name__ == "__main__":
    main()
