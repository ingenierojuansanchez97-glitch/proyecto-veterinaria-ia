#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "model_kidney.pkl"
FEATURES = ["Edad", "Peso", "Creatinina", "Urea", "BUN", "SDMA"]

RISK_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}
LABEL_TO_INDEX = {"Bajo": 0, "Medio": 1, "Alto": 2}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def safe_load_model(path):
    """
    Carga flexible: soporta dicts, tuples, o el estimador directo.
    Devuelve (model, feat_order). Si no hay feature_order, usa FEATURES.
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
    Reordena/expande proba -> 3 columnas en orden [Bajo, Medio, Alto]
    usando 'classes' del modelo (pueden ser ints o strings).
    Si vienen 2 clases, las mapea y deja la faltante en 0.
    """
    proba = np.asarray(proba, dtype=float)
    n, m = proba.shape
    out = np.zeros((n, 3), dtype=float)

    if classes is None:
        # Sin classes_, asumir que las columnas ya son [0..m-1]
        if m == 3:
            return proba
        elif m == 2:
            # asumir col0->Bajo, col1->Alto
            out[:, 0] = proba[:, 0]
            out[:, 2] = proba[:, 1]
            s = out.sum(1, keepdims=True); s[s == 0] = 1
            return out / s
        else:
            # Caso raro: rellenar y normalizar
            out[:, :m] = proba[:, :m]
            s = out.sum(1, keepdims=True); s[s == 0] = 1
            return out / s

    classes = np.array(classes)

    # Si son numéricas (0/1/2, o 0/2, etc.)
    if np.issubdtype(classes.dtype, np.number):
        for k in [0, 1, 2]:
            where = np.where(classes == k)[0]
            if len(where):
                out[:, k] = proba[:, where[0]]
        s = out.sum(1, keepdims=True); s[s == 0] = 1
        return out / s

    # Si son strings (e.g., 'Bajo','Medio','Alto' en cualquier orden)
    classes = list(classes)
    for lbl, k in LABEL_TO_INDEX.items():
        if lbl in classes:
            j = classes.index(lbl)
            out[:, k] = proba[:, j]
    s = out.sum(1, keepdims=True); s[s == 0] = 1
    return out / s


def safe_predict_proba(model, X):
    """
    Devuelve probas SIEMPRE con 3 columnas en orden [Bajo, Medio, Alto],
    reordenando según model.classes_ (int o str) y completando faltantes con 0.
    """
    classes = getattr(model, "classes_", None)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return _to_three_cols(proba, classes)

    if hasattr(model, "decision_function"):
        scores = np.array(model.decision_function(X))
        if scores.ndim == 1:  # binario
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            proba = np.column_stack([1 - probs_pos, probs_pos])
        else:  # multiclase
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return _to_three_cols(proba, classes)

    # Fallback ultra-defensivo
    preds = model.predict(X)
    # mapear etiquetas a índices 0/1/2 si vienen como string
    preds_idx = []
    for p in preds:
        if isinstance(p, str):
            preds_idx.append(LABEL_TO_INDEX.get(p, 0))
        else:
            preds_idx.append(int(p))
    preds_idx = np.array(preds_idx, dtype=int)

    proba = np.zeros((len(preds_idx), 3), dtype=float)
    for i, c in enumerate(preds_idx):
        if c in (0, 1, 2):
            proba[i, c] = 1.0
        else:
            proba[i, 0] = 1.0  # por si acaso
    return proba


# -------- utilidades de CLI (opcionales para probar local) ---------
def score_csv(in_csv, out_csv="scored_cases.csv"):
    model, feat_order = safe_load_model(MODEL_PATH)
    df = pd.read_csv(in_csv)
    missing = [c for c in feat_order if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en CSV: {missing}")

    X = df[feat_order].copy()
    proba = safe_predict_proba(model, X)
    pred_classes = np.argmax(proba, axis=1)
    pred_labels = [INDEX_TO_LABEL.get(c, str(c)) for c in pred_classes]

    df["prob_bajo"] = proba[:, 0]
    df["prob_medio"] = proba[:, 1]
    df["prob_alto"] = proba[:, 2]
    df["pred_class"] = pred_classes
    df["pred_label"] = pred_labels

    df.to_csv(out_csv, index=False)
    print(f"✅ Guardado: {out_csv}")
    print(df[["pred_label", "prob_bajo", "prob_medio", "prob_alto"]].head())


def interactive():
    print("=== Modo interactivo ===")
    vals = {}
    for f in FEATURES:
        while True:
            try:
                vals[f] = float(input(f"{f}: ").strip())
                break
            except ValueError:
                print("Valor inválido. Ingresa un número.")
    model, feat_order = safe_load_model(MODEL_PATH)
    X = pd.DataFrame([vals])[feat_order]
    proba = safe_predict_proba(model, X)
    pred_class = int(np.argmax(proba, axis=1)[0])
    pred_label = INDEX_TO_LABEL.get(pred_class, str(pred_class))

    print("\n📊 Resultado:")
    print(f"Bajo:  {proba[0,0]*100:.1f}%")
    print(f"Medio: {proba[0,1]*100:.1f}%")
    print(f"Alto:  {proba[0,2]*100:.1f}%")
    print(f"Predicción final: {pred_label}")


def main():
    ap = argparse.ArgumentParser(description="Predicción de riesgo renal (multiclase)")
    ap.add_argument("csv", nargs="?", help="Ruta al CSV de entrada")
    ap.add_argument("--interactive", action="store_true", help="Ingresar datos manualmente")
    ap.add_argument("--out", default="scored_cases.csv", help="Archivo de salida")
    args = ap.parse_args()

    if args.interactive:
        interactive()
    elif args.csv:
        score_csv(args.csv, args.out)
    else:
        print("Uso:\n  python predict_kidney_model.py datos.csv\n  python predict_kidney_model.py --interactive")


if __name__ == "__main__":
    main()
