#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# =========================
# Configuración
# =========================
DB_PATH = "veterinaria.db"

FEATURES = ["Edad", "Peso", "Creatinina", "Urea", "BUN", "SDMA"]
SPECIES_OPTIONS = ["Canino", "Felino", "Ave", "Pez", "Reptil", "Caballo", "Cerdo", "Vaca", "Otro"]

# Rangos de referencia (puedes ajustar por especie)
REF_RANGES = {
    "Canino": {"Creatinina": 1.4, "Urea": 50, "BUN": 25, "SDMA": 14},
    "Felino": {"Creatinina": 1.6, "Urea": 60, "BUN": 28, "SDMA": 14},
}

RISK_LABELS = {0: ("Bajo", "#43A047"), 1: ("Medio", "#FB8C00"), 2: ("Alto", "#E53935")}

# =========================
# Reglas clínicas + probabilidades
# =========================
def classify_band(vals, especie="Canino"):
    refs = REF_RANGES.get(especie, REF_RANGES["Canino"])
    markers = ["Creatinina", "Urea", "BUN", "SDMA"]

    ratios = []
    mild_count = 0      # 1 < r <= 1.3
    severe_count = 0    # r > 1.3

    for m in markers:
        v = float(vals[m])
        r = v / float(refs[m])
        ratios.append(r)
        if r > 1.3:
            severe_count += 1
        elif r > 1.0:
            mild_count += 1

    # Reglas de banda
    if severe_count >= 1 or (mild_count + severe_count) >= 2:
        band_idx = 2  # Alto
    elif (mild_count == 1) and (severe_count == 0):
        band_idx = 1  # Medio
    else:
        band_idx = 0  # Bajo

    # Probabilidades suaves:
    # - 0.80 a la clase asignada
    # - 0.20 repartido entre las otras dos según severidad
    probs = np.zeros(3, dtype=float)
    probs[band_idx] = 0.80

    # Pesos para distribuir el 0.20
    w = np.zeros(3, dtype=float)
    mean_excess = float(np.mean([max(0.0, r - 1.0) for r in ratios]))  # exceso medio sobre el límite
    # cercanía a "bajo": mientras menor sea el exceso, más peso para bajo
    closeness_low = max(0.0, 1.0 - (mean_excess / 0.30))  # 0 si está >=30% por encima en promedio
    # severidad hacia alto: nº de severos
    w[2] = float(severe_count)                             # peso hacia alto
    w[1] = float(mild_count)                               # peso hacia medio
    w[0] = closeness_low                                   # peso hacia bajo

    # quitamos el peso de la clase ya asignada
    w[band_idx] = 0.0
    if w.sum() == 0:
        # repartir equitativamente entre las dos restantes
        for i in range(3):
            if i != band_idx:
                probs[i] = probs[i] + 0.10
    else:
        w = w / w.sum()
        for i in range(3):
            if i != band_idx:
                probs[i] += 0.20 * w[i]

    # Asegurar que sumen 1 por redondeos
    probs = probs / probs.sum()
    return band_idx, probs

# =========================
# DB
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS casos_renales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha_hora TEXT,
        nombre TEXT,
        especie TEXT,
        raza TEXT,
        edad REAL, peso REAL, creatinina REAL, urea REAL, bun REAL, sdma REAL,
        risk_label TEXT,
        prob_bajo REAL, prob_medio REAL, prob_alto REAL
    )
    """)
    conn.commit()
    conn.close()

def insert_case(vals, nombre, especie, raza, risk_label, probs):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO casos_renales
            (fecha_hora, nombre, especie, raza, edad, peso, creatinina, urea, bun, sdma,
             risk_label, prob_bajo, prob_medio, prob_alto)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            nombre, especie, raza,
            vals["Edad"], vals["Peso"], vals["Creatinina"], vals["Urea"], vals["BUN"], vals["SDMA"],
            risk_label, float(probs[0]), float(probs[1]), float(probs[2])
        ))
        conn.commit()
    except Exception as e:
        messagebox.showerror("DB", f"No se pudo guardar en la base de datos: {e}")
    finally:
        conn.close()

def fetch_last5():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT nombre, especie, risk_label, prob_bajo, prob_medio, prob_alto
        FROM casos_renales
        ORDER BY id DESC LIMIT 5
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

# =========================
# PDF
# =========================
def generate_risk_chart(vals, especie, filename="grafico.png"):
    refs = REF_RANGES.get(especie, REF_RANGES["Canino"])
    parametros = ["Creatinina", "Urea", "BUN", "SDMA"]
    valores = [float(vals[p]) for p in parametros]
    normal_max = [float(refs[p]) for p in parametros]

    plt.figure(figsize=(6, 4))
    plt.bar(parametros, valores, label="Paciente")
    plt.plot(parametros, normal_max, marker="o", linestyle="--", label="Límite normal")
    plt.ylabel("Valor")
    plt.title("Parámetros renales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

from fpdf import FPDF
def generate_pdf(vals, nombre, especie, raza, riesgo, probs, filename="reporte.pdf"):
    chart_file = "grafico.png"
    generate_risk_chart(vals, especie, chart_file)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(25, 118, 210)
    pdf.cell(0, 10, f"Reporte de Riesgo Renal - {nombre}", ln=True, align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.ln(4)
    pdf.cell(0, 8, f"Especie: {especie}    Raza: {raza}", ln=True)
    pdf.cell(0, 8, f"Riesgo: {riesgo}", ln=True)
    pdf.cell(0, 8, f"Probabilidades -> Bajo: {probs[0]*100:.0f}% | Medio: {probs[1]*100:.0f}% | Alto: {probs[2]*100:.0f}%", ln=True)

    pdf.ln(4)
    pdf.image(chart_file, x=25, w=160)

    pdf.ln(78)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, "Recomendaciones", ln=True)
    pdf.set_font("Arial", '', 12)
    if riesgo == "Bajo":
        pdf.multi_cell(0, 6, "- Mantener controles de rutina.\n- Dieta balanceada y agua fresca.\n- Repetir perfil renal si aparecen sintomas.")
    elif riesgo == "Medio":
        pdf.multi_cell(0, 6, "- Control cada 3-6 meses.\n- Considerar dieta renal y rehidratacion guiada.\n- Reevaluar si aumentan creatinina/SDMA.")
    else:
        pdf.multi_cell(0, 6, "- Evaluacion y tratamiento inmediato.\n- Dieta renal estricta y seguimiento cercano.\n- Considerar estudios complementarios.")
    pdf.output(filename)


# =========================
# GUI
# =========================
def fmt_pct(x): return f"{x*100:.0f}%"

class KidneyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Predicción de Riesgo Renal - Veterinaria")
        self.geometry("720x710")
        self.resizable(False, False)
        self.configure(bg="#F2F5F9")

        init_db()

        self.inputs = {}
        self.last_vals = None
        self.last_probs = None
        self.last_pred = None
        self.last_meta = None

        self._build_ui()
        self._load_quick_history()

    # ---------- UI ----------
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Header.TFrame", background="#1976D2")
        style.configure("Header.TLabel", background="#1976D2", foreground="white",
                        font=("Segoe UI", 16, "bold"))

        header = ttk.Frame(self, padding=10, style="Header.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Predicción de Riesgo Renal - Veterinaria",
                  style="Header.TLabel").pack(anchor="center")

        container = ttk.Frame(self, padding=14)
        container.pack(fill="both", expand=True)

        # Datos básicos
        ttk.Label(container, text="Nombre:").grid(row=0, column=0, sticky="w")
        self.entry_nombre = ttk.Entry(container, width=24)
        self.entry_nombre.grid(row=0, column=1, sticky="w")

        ttk.Label(container, text="Especie:").grid(row=0, column=2, sticky="w")
        self.combo_especie = ttk.Combobox(container, values=SPECIES_OPTIONS,
                                          state="readonly", width=18)
        self.combo_especie.current(0)
        self.combo_especie.grid(row=0, column=3, sticky="w")

        ttk.Label(container, text="Raza:").grid(row=1, column=0, sticky="w")
        self.entry_raza = ttk.Entry(container, width=24)
        self.entry_raza.grid(row=1, column=1, sticky="w")

        placeholders = {
            "Edad": "años", "Peso": "kg",
            "Creatinina": "mg/dL", "Urea": "mg/dL", "BUN": "mg/dL", "SDMA": "μg/dL"
        }
        for i, feat in enumerate(FEATURES, start=2):
            ttk.Label(container, text=f"{feat} ({placeholders.get(feat,'')}):")\
                .grid(row=i, column=0, sticky="w")
            e = ttk.Entry(container, width=18, validate="key")
            e["validatecommand"] = (e.register(self._validate_number), "%P", "%V")
            e.grid(row=i, column=1, sticky="w")
            self.inputs[feat] = e

        # Botones
        btns = ttk.Frame(container)
        btns.grid(row=2+len(FEATURES), column=0, columnspan=4, pady=12)
        ttk.Button(btns, text="Predecir", command=self.predict_single)\
            .pack(side="left", padx=6)
        ttk.Button(btns, text="Limpiar", command=self.reset_fields)\
            .pack(side="left", padx=6)
        self.btn_pdf = ttk.Button(btns, text="Generar PDF",
                                  command=self.generate_pdf_action, state="disabled")
        self.btn_pdf.pack(side="left", padx=6)

        # Resultado principal
        self.lbl_result = tk.Label(container, text="Riesgo: —",
                                   font=("Segoe UI", 16, "bold"), bg="#F2F5F9")
        self.lbl_result.grid(row=3+len(FEATURES), column=0, columnspan=4, pady=8)

        # Barras con gradiente (Canvas)
        self.prog_bars = {}
        for idx, (label, color) in RISK_LABELS.items():
            frame = ttk.Frame(container)
            frame.grid(row=4+len(FEATURES)+idx, column=0, columnspan=4, sticky="we", pady=2)

            tk.Label(frame, text=label, width=8, anchor="w").pack(side="left")
            canvas = tk.Canvas(frame, width=550, height=20, bg="#E0E0E0", highlightthickness=0)
            canvas.pack(side="left", padx=5)
            pct_label = tk.Label(frame, text="0%", width=6, anchor="e")
            pct_label.pack(side="left")

            self.prog_bars[idx] = (canvas, pct_label, color)

        # Historial
        ttk.Label(container, text="Últimas 5 predicciones:")\
            .grid(row=8+len(FEATURES), column=0, columnspan=4, pady=(6, 2))
        self.tree = ttk.Treeview(container, columns=("nombre", "especie", "riesgo", "prob"),
                                 show="headings", height=6)
        self.tree.heading("nombre", text="Nombre");  self.tree.column("nombre", width=180, anchor="w")
        self.tree.heading("especie", text="Especie"); self.tree.column("especie", width=120, anchor="center")
        self.tree.heading("riesgo", text="Riesgo");  self.tree.column("riesgo", width=100, anchor="center")
        self.tree.heading("prob", text="Probabilidad"); self.tree.column("prob", width=120, anchor="center")
        self.tree.grid(row=9+len(FEATURES), column=0, columnspan=4, sticky="we")

        self.tree.tag_configure("row_green", background="#C8E6C9")
        self.tree.tag_configure("row_orange", background="#FFE0B2")
        self.tree.tag_configure("row_red", background="#FFCDD2")

    # ---------- Validaciones ----------
    def _validate_number(self, P, V):
        if V == "key":
            return (P == "" or P.replace(".", "", 1).isdigit())
        return True

    def reset_fields(self):
        for e in self.inputs.values():
            e.delete(0, tk.END)
        self.entry_nombre.delete(0, tk.END)
        self.entry_raza.delete(0, tk.END)
        self.lbl_result.config(text="Riesgo: —", bg="#F2F5F9", fg="black")
        for idx, (canvas, lbl, _) in self.prog_bars.items():
            canvas.delete("all"); lbl.config(text="0%")
        self.btn_pdf.config(state="disabled")
        self.last_vals = self.last_probs = self.last_pred = self.last_meta = None

    # ---------- Lógica ----------
    def _draw_gradient_bar(self, canvas, pct, color):
        canvas.delete("all")
        w = int(canvas.winfo_width())
        h = int(canvas.winfo_height())
        fill_w = int(w * pct)
        r, g, b = self.winfo_rgb(color)
        r //= 256; g //= 256; b //= 256
        for i in range(fill_w):
            frac = i / max(fill_w, 1)
            rr = int(255 + (r - 255) * frac)
            gg = int(255 + (g - 255) * frac)
            bb = int(255 + (b - 255) * frac)
            canvas.create_line(i, 0, i, h, fill=f"#{rr:02x}{gg:02x}{bb:02x}")

    def predict_single(self):
        try:
            nombre = self.entry_nombre.get().strip() or "—"
            especie = self.combo_especie.get()
            raza = self.entry_raza.get().strip() or "—"
            vals = {feat: float(self.inputs[feat].get()) for feat in FEATURES}
        except ValueError:
            messagebox.showerror("Error", "Por favor ingresa todos los valores numéricos.")
            return

        # Clasificación por reglas clínicas
        band_idx, probs = classify_band(vals, especie)
        pred_label, color = RISK_LABELS[band_idx]

        # UI
        self.lbl_result.config(text=f"Riesgo: {pred_label}", bg=color, fg="white")
        for idx, (canvas, lbl, col) in self.prog_bars.items():
            p = float(probs[idx])
            lbl.config(text=fmt_pct(p))
            self._draw_gradient_bar(canvas, p, col)

        # Guardar estado para PDF
        self.last_vals = vals
        self.last_probs = probs
        self.last_pred = pred_label
        self.last_meta = {"nombre": nombre, "especie": especie, "raza": raza}
        self.btn_pdf.config(state="normal")

        # Guardar en DB + refrescar historial
        insert_case(vals, nombre, especie, raza, pred_label, probs)
        self._load_quick_history()

    def _load_quick_history(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

        rows = fetch_last5()
        for nombre, especie, risk_label, pb, pm, pa in rows:
            # asignar color por etiqueta
            if risk_label == "Alto":
                tag, prob = "row_red", pa
            elif risk_label == "Medio":
                tag, prob = "row_orange", pm
            else:
                tag, prob = "row_green", pb
            self.tree.insert("", "end",
                             values=(nombre, especie, risk_label, fmt_pct(prob)),
                             tags=(tag,))

    # ---------- PDF ----------
    def generate_pdf_action(self):
        if not self.last_vals:
            messagebox.showinfo("PDF", "No hay predicción reciente.")
            return
        try:
            filename = f"reporte_{self.last_meta['nombre']}.pdf"
            generate_pdf(self.last_vals,
                         self.last_meta["nombre"],
                         self.last_meta["especie"],
                         self.last_meta["raza"],
                         self.last_pred,
                         self.last_probs,
                         filename=filename)
            messagebox.showinfo("PDF", f"PDF generado: {filename}")
        except Exception as e:
            messagebox.showerror("Error PDF", f"No se pudo generar el PDF: {e}")

# =========================
# Run
# =========================
if __name__ == "__main__":
    app = KidneyApp()
    app.mainloop()
