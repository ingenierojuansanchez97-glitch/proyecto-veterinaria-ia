#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicción de Riesgo Renal (Veterinaria) – Versión 2 (MODIFICADA)
Mejoras:
- Splash screen con logo + icono de la app
- Sistema de login/registro (médico o clínica) con verificación por correo (OTP)
- Historial por usuario/clinica
- Integración de modelo ML + reglas clínicas + (opcional) validador IA vía API REST
- **Validación de datos con validator.py**
- **Generación de PDF con fpdf2**
- **Envío automático del PDF por correo al email del usuario (SMTP)**
- **Historial editable: doble clic carga el último registro del paciente**

Requisitos sugeridos:
  pip install pillow fpdf2 joblib scikit-learn python-dotenv
Variables de entorno (para envío de correo):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
Opcional:
  AI_VALIDATOR_URL (endpoint REST que recibe JSON y devuelve {"probs": [p_bajo, p_medio, p_alto]})

Para compilar ejecutable con icono (Windows):
  pyinstaller --noconsole --onefile --icon=logo.ico app_kidney_gui_v2.py

Coloca los archivos de medios en la misma carpeta:
  - logo.png  (para splash)
  - logo.ico  (para icono de ventana/ejecutable en Windows)
"""
import os
import sys
import json
import sqlite3
import smtplib, ssl
from email.message import EmailMessage
from datetime import datetime, timedelta

import tkinter as tk
from tkinter import ttk, messagebox

# Dependencias opcionales
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF
from dotenv import load_dotenv

# ✅ NUEVO: validación y envío de correo
from validator import validate_inputs, ValidationError
from mailer import send_email_with_pdf

# ✅ NUEVO: integración con OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Cargar variables del .env
load_dotenv()

# =========================
# Configuración
# =========================
def resource_path(relative_path):
    """Obtiene la ruta absoluta, tanto si corres como .py o como .exe"""
    if hasattr(sys, '_MEIPASS'):  # cuando es .exe
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
APP_TITLE = "Predicción Renal Veterinaria — v2"
DB_PATH = os.environ.get("KIDNEY_APP_DB", "veterinaria.db")
MODEL_PATH = os.environ.get("KIDNEY_MODEL", "model_kidney.pkl")
AI_VALIDATOR_URL = os.environ.get("AI_VALIDATOR_URL")

FEATURES = ["Edad", "Peso", "Creatinina", "Urea", "BUN", "SDMA"]
SPECIES_OPTIONS = ["Canino", "Felino", "Ave", "Pez", "Reptil", "Caballo", "Cerdo", "Vaca", "Otro"]

REF_RANGES = {
    "Canino": {"Creatinina": 1.4, "Urea": 50, "BUN": 25, "SDMA": 14},
    "Felino": {"Creatinina": 1.6, "Urea": 60, "BUN": 28, "SDMA": 14},
}

RISK_LABELS = {0: ("Bajo", "#43A047"), 1: ("Medio", "#FB8C00"), 2: ("Alto", "#E53935")}

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER or "")

# =========================
# Utilidades
# =========================
def resource_path(path: str) -> str:
    """ Soporta PyInstaller (carpeta temporal _MEIPASS). """
    base = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base, path)


def fmt_pct(x: float) -> str:
    return f"{x*100:.0f}%"

# =========================
# Reglas clínicas (bandas) + probabilidades suaves
# =========================
def classify_band(vals: dict, especie: str = "Canino"):
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

    if severe_count >= 1 or (mild_count + severe_count) >= 2:
        band_idx = 2  # Alto
    elif (mild_count == 1) and (severe_count == 0):
        band_idx = 1  # Medio
    else:
        band_idx = 0  # Bajo

    probs = np.zeros(3, dtype=float)
    probs[band_idx] = 0.80

    w = np.zeros(3, dtype=float)
    mean_excess = float(np.mean([max(0.0, r - 1.0) for r in ratios]))
    closeness_low = max(0.0, 1.0 - (mean_excess / 0.30))
    w[2] = float(severe_count)
    w[1] = float(mild_count)
    w[0] = closeness_low
    w[band_idx] = 0.0

    if w.sum() == 0:
        for i in range(3):
            if i != band_idx:
                probs[i] += 0.10
    else:
        w = w / w.sum()
        for i in range(3):
            if i != band_idx:
                probs[i] += 0.20 * w[i]

    probs = probs / probs.sum()
    return band_idx, probs

# =========================
# Modelo ML (carga segura)
# =========================
def safe_load_model(path: str):
    art = joblib.load(path)
    if isinstance(art, dict):
        model = art.get('model') or art.get('estimator') or art.get('clf')
        feat_order = art.get('feature_order') or art.get('features') or []
    else:
        if isinstance(art, tuple) and len(art) == 2:
            model, feat_order = art
        else:
            model, feat_order = art, []
    if not feat_order:
        feat_order = FEATURES
    return model, feat_order


def safe_predict_proba(model, X: pd.DataFrame):
    classes = getattr(model, 'classes_', None)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if classes is None:
            return proba
        import numpy as _np
        sorted_idx = _np.argsort(classes)
        return proba[:, sorted_idx]
    elif hasattr(model, 'decision_function'):
        import numpy as _np
        scores = _np.array(model.decision_function(X))
        if scores.ndim == 1:
            probs_pos = 1 / (1 + _np.exp(-scores))
            return _np.column_stack([1 - probs_pos, probs_pos])
        else:
            exp_scores = _np.exp(scores)
            return exp_scores / _np.sum(exp_scores, axis=1, keepdims=True)
    else:
        preds = model.predict(X)
        n_classes = int(max(preds)) + 1 if len(set(preds))>0 else 3
        proba = np.zeros((len(preds), n_classes))
        for i, p in enumerate(preds):
            proba[i, int(p)] = 1.0
        return proba

# =========================
# Envío de correos (OTP de verificación)
# =========================
def send_verification_email(to_email: str, code: str) -> bool:
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and SMTP_FROM):
        print("[WARN] SMTP no configurado; modo simulación. Código:", code)
        return True
    try:
        msg = EmailMessage()
        msg["Subject"] = "Verificación de cuenta — App Renal"
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        msg.set_content(f"Tu código de verificación es: {code}\nExpira en 15 minutos.")
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return True
    except Exception as e:
        print("[ERROR] SMTP:", e)
        return False


def send_test_email():
    """Envía un correo de prueba usando la configuración SMTP."""
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and SMTP_FROM):
        messagebox.showwarning("Correo", "SMTP no configurado. Revisa tu .env")
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = "Prueba de correo — App Renal"
        msg["From"] = SMTP_FROM
        msg["To"] = SMTP_USER
        msg.set_content("Este es un correo de prueba desde la aplicación de predicción renal.")
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        messagebox.showinfo("Correo", f"Correo de prueba enviado a {SMTP_USER}")
    except Exception as e:
        messagebox.showerror("Correo", f"Error al enviar: {e}")

# =========================
# DB & migraciones
# =========================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Usuarios
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            clinic TEXT,
            role TEXT DEFAULT 'doctor',
            email_verified INTEGER DEFAULT 0,
            verification_code TEXT,
            verification_expires TEXT
        )
        """
    )

    # Casos (extensión con user_id y clinic)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS casos_renales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_hora TEXT,
            nombre TEXT,
            especie TEXT,
            raza TEXT,
            edad REAL, peso REAL, creatinina REAL, urea REAL, bun REAL, sdma REAL,
            risk_label TEXT,
            prob_bajo REAL, prob_medio REAL, prob_alto REAL,
            user_id INTEGER,
            clinic TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    # Asegurar columnas nuevas si venimos de v1
    def ensure_column(table: str, col: str, ddl: str):
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
        except Exception:
            pass

    ensure_column("casos_renales", "user_id", "INTEGER")
    ensure_column("casos_renales", "clinic", "TEXT")

    conn.commit()
    conn.close()

# =========================
# Seguridad (hash de contraseña)
# =========================
import hashlib, secrets

def hash_password(pw: str, salt: str = None) -> str:
    if not salt:
        salt = secrets.token_hex(8)
    h = hashlib.sha256((salt + pw).encode('utf-8')).hexdigest()
    return f"{salt}${h}"

def verify_password(pw: str, stored: str) -> bool:
    try:
        salt, h = stored.split("$")
        return hash_password(pw, salt) == stored
    except Exception:
        return False

# =========================
# Lógica de predicción (blending reglas + ML + IA)
# =========================

def predict_blended(vals: dict, especie: str, model_bundle):
    # Reglas clínicas
    band_idx, probs_rules = classify_band(vals, especie)

    # Modelo ML
    model, feat_order = model_bundle
    X = pd.DataFrame([vals])[feat_order]
    probs_ml = safe_predict_proba(model, X)[0]
    probs_ml = _pad3(probs_ml)

    # Validación IA externa (opcional)
    probs_ai = None
    if AI_VALIDATOR_URL:
        try:
            import urllib.request
            req = urllib.request.Request(
                AI_VALIDATOR_URL,
                data=json.dumps({"vals": vals, "especie": especie}).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, dict) and "probs" in data:
                    probs_ai = np.array(data["probs"], dtype=float)
        except Exception as e:
            print("[WARN] AI_VALIDATOR_URL fallo:", e)

    # Pesos de mezcla
    w_rules = 0.30
    w_ml    = 0.60
    w_ai    = 0.10 if probs_ai is not None else 0.0
    norm = w_rules + w_ml + w_ai
    w_rules, w_ml, w_ai = w_rules/norm, w_ml/norm, w_ai/norm

    if probs_ai is None:
        probs_ai = np.zeros(3)

    final_probs = w_rules*np.array(probs_rules) + w_ml*np.array(probs_ml) + w_ai*np.array(probs_ai)
    final_probs = final_probs / final_probs.sum()
    pred_idx = int(np.argmax(final_probs))
    return pred_idx, final_probs, {
        "probs_rules": probs_rules,
        "probs_ml": probs_ml,
        "probs_ai": probs_ai if w_ai>0 else None,
        "weights": (w_rules, w_ml, w_ai)
    }


def _pad3(arr):
    arr = np.array(arr, dtype=float)
    if arr.shape[0] == 3:
        return arr
    if arr.shape[0] == 2:
        # Si el modelo fue binario, clavamos prob "Medio" como 0 y re-normalizamos
        arr = np.array([arr[0], 0.0, arr[1]])
        s = arr.sum()
        return arr/s if s>0 else np.array([1/3,1/3,1/3])
    # Otra forma/orden
    arr = arr[:3]
    s = arr.sum()
    return arr/s if s>0 else np.array([1/3,1/3,1/3])

# =========================
# PDF
# =========================

def generate_risk_chart(vals, especie, filename="grafico.png"):
    import matplotlib.pyplot as plt
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


def generate_pdf(vals, meta, riesgo, probs, filename="reporte.pdf"):
    chart_file = "grafico.png"
    generate_risk_chart(vals, meta["especie"], chart_file)

    pdf = FPDF()
    pdf.add_page()

    # Logo superior (si existe)
    logo_path = resource_path("logo.png")
    if os.path.exists(logo_path):
        try:
            pdf.image(logo_path, x=10, y=8, w=20)
        except Exception:
            pass

    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(25, 118, 210)
    pdf.cell(0, 10, f"Reporte de Riesgo Renal - {meta['nombre']}", ln=True, align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.ln(2)
    pdf.cell(0, 8, f"Especie: {meta['especie']}    Raza: {meta['raza']}", ln=True)
    pdf.cell(0, 8, f"Riesgo: {riesgo}", ln=True)
    pdf.cell(0, 8, f"Probabilidades -> Bajo: {probs[0]*100:.0f}% | Medio: {probs[1]*100:.0f}% | Alto: {probs[2]*100:.0f}%", ln=True)

    pdf.ln(2)
    pdf.cell(0, 8, f"Atendido por: {meta.get('doctor_name','—')} ({meta.get('clinic','—')})", ln=True)

    pdf.ln(4)
    pdf.image(chart_file, x=25, w=160)

    pdf.ln(78)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, "Recomendaciones", ln=True)
    pdf.set_font("Arial", '', 12)
    if riesgo == "Bajo":
        pdf.multi_cell(0, 6, "- Mantener controles de rutina.\n- Dieta balanceada y agua fresca.\n- Repetir perfil renal si aparecen síntomas.")
    elif riesgo == "Medio":
        pdf.multi_cell(0, 6, "- Control cada 3-6 meses.\n- Considerar dieta renal y rehidratación guiada.\n- Reevaluar si aumentan creatinina/SDMA.")
    else:
        pdf.multi_cell(0, 6, "- Evaluación y tratamiento inmediato.\n- Dieta renal estricta y seguimiento cercano.\n- Considerar estudios complementarios.")
    pdf.output(filename)

# =========================
# UI: Splash
# =========================
class Splash(tk.Toplevel):
    def __init__(self, master, duration=1800):
        super().__init__(master)
        self.overrideredirect(True)
        self.configure(bg="#ffffff")
        w, h = 420, 260
        self.geometry(f"{w}x{h}+{self._cx(w)}+{self._cy(h)}")

        frame = tk.Frame(self, bg="#ffffff")
        frame.pack(expand=True, fill="both", padx=16, pady=16)

        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            try:
                if PIL_AVAILABLE:
                    img = Image.open(logo_path)
                    img = img.resize((128, 128), Image.LANCZOS)
                    self._img = ImageTk.PhotoImage(img)
                else:
                    self._img = tk.PhotoImage(file=logo_path)
                tk.Label(frame, image=self._img, bg="#ffffff").pack(pady=8)
            except Exception:
                tk.Label(frame, text="🐾", font=("Segoe UI", 48), bg="#ffffff").pack(pady=8)
        else:
            tk.Label(frame, text="🐾", font=("Segoe UI", 48), bg="#ffffff").pack(pady=8)

        tk.Label(frame, text=APP_TITLE, font=("Segoe UI", 14, "bold"), bg="#ffffff").pack()
        tk.Label(frame, text="Cargando…", font=("Segoe UI", 10), bg="#ffffff").pack(pady=6)

        self.after(duration, self.destroy)

    def _cx(self, w):
        return int(self.winfo_screenwidth()/2 - w/2)
    def _cy(self, h):
        return int(self.winfo_screenheight()/2 - h/2)

# =========================
# UI: Login & Registro
# =========================
class AuthWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Acceso — App Renal")
        self.geometry("420x450")
        self.resizable(False, False)
        self.configure(bg="#F7FAFC")
        self.transient(master)
        self.grab_set()

        try:
            ico_path = os.path.join(os.path.dirname(__file__), "logo.ico")
            if os.path.exists(ico_path):
                self.iconbitmap(ico_path)
        except Exception:
            pass

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Login ---
        frm_login = ttk.Frame(nb, padding=12)
        nb.add(frm_login, text="Iniciar sesión")

        ttk.Label(frm_login, text="Email:").grid(row=0, column=0, sticky="w")
        self.e_login_email = ttk.Entry(frm_login, width=34)
        self.e_login_email.grid(row=0, column=1, sticky="w")

        ttk.Label(frm_login, text="Contraseña:").grid(row=1, column=0, sticky="w")
        self.e_login_pw = ttk.Entry(frm_login, width=34, show="•")
        self.e_login_pw.grid(row=1, column=1, sticky="w")

        self.var_keep = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_login, text="Recordarme (solo este equipo)", variable=self.var_keep).grid(row=2, column=1, sticky="w", pady=(6,2))

        ttk.Button(frm_login, text="Entrar", command=self._do_login).grid(row=3, column=1, sticky="e", pady=8)
        ttk.Button(frm_login, text="Probar correo", command=send_test_email).grid(row=4, column=1, sticky="e", pady=4)

        # --- Registro ---
        frm_reg = ttk.Frame(nb, padding=12)
        nb.add(frm_reg, text="Registrarme")

        ttk.Label(frm_reg, text="Nombre completo:").grid(row=0, column=0, sticky="w")
        self.e_name = ttk.Entry(frm_reg, width=34); self.e_name.grid(row=0, column=1, sticky="w")
        ttk.Label(frm_reg, text="Email:").grid(row=1, column=0, sticky="w")
        self.e_email = ttk.Entry(frm_reg, width=34); self.e_email.grid(row=1, column=1, sticky="w")
        ttk.Label(frm_reg, text="Clínica (opcional):").grid(row=2, column=0, sticky="w")
        self.e_clinic = ttk.Entry(frm_reg, width=34); self.e_clinic.grid(row=2, column=1, sticky="w")
        ttk.Label(frm_reg, text="Contraseña:").grid(row=3, column=0, sticky="w")
        self.e_pw = ttk.Entry(frm_reg, width=34, show="•"); self.e_pw.grid(row=3, column=1, sticky="w")
        ttk.Label(frm_reg, text="Confirmar:").grid(row=4, column=0, sticky="w")
        self.e_pw2 = ttk.Entry(frm_reg, width=34, show="•"); self.e_pw2.grid(row=4, column=1, sticky="w")

        ttk.Button(frm_reg, text="Crear cuenta", command=self._do_register).grid(row=5, column=1, sticky="e", pady=8)

        # --- Verificación ---
        frm_ver = ttk.Frame(nb, padding=12)
        nb.add(frm_ver, text="Verificar correo")

        ttk.Label(frm_ver, text="Email:").grid(row=0, column=0, sticky="w")
        self.e_vemail = ttk.Entry(frm_ver, width=34); self.e_vemail.grid(row=0, column=1, sticky="w")
        ttk.Label(frm_ver, text="Código recibido:").grid(row=1, column=0, sticky="w")
        self.e_vcode = ttk.Entry(frm_ver, width=34); self.e_vcode.grid(row=1, column=1, sticky="w")
        ttk.Button(frm_ver, text="Verificar", command=self._do_verify).grid(row=2, column=1, sticky="e", pady=8)

        self.user = None

    # --- DB helpers ---
    def _get_user_by_email(self, email):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id,name,email,password_hash,clinic,role,email_verified FROM users WHERE email=?", (email.lower().strip(),))
        row = cur.fetchone()
        conn.close()
        return row

    def _insert_user(self, name, email, pw_hash, clinic):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users(name,email,password_hash,clinic,role,email_verified) VALUES (?,?,?,?,?,0)",
                    (name, email.lower().strip(), pw_hash, clinic, 'doctor'))
        uid = cur.lastrowid
        conn.commit()
        conn.close()
        return uid

    def _set_verification(self, email, code, expires):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET verification_code=?, verification_expires=? WHERE email=?",
                    (code, expires.isoformat(), email.lower().strip()))
        conn.commit()
        conn.close()

    # --- Actions ---
    def _do_login(self):
        email = self.e_login_email.get().strip().lower()
        pw = self.e_login_pw.get()
        row = self._get_user_by_email(email)
        if not row:
            messagebox.showerror("Login", "Usuario no encontrado.")
            return
        uid, name, email, pw_hash, clinic, role, verified = row
        if not verify_password(pw, pw_hash):
            messagebox.showerror("Login", "Contraseña incorrecta.")
            return
        if not verified:
            messagebox.showwarning("Login", "Cuenta no verificada. Ingresa el código en la pestaña 'Verificar correo'.")
            return
        self.user = {"id": uid, "name": name, "email": email, "clinic": clinic, "role": role}
        self.destroy()

    def _do_register(self):
        name = self.e_name.get().strip()
        email = self.e_email.get().strip().lower()
        clinic = self.e_clinic.get().strip() or None
        pw = self.e_pw.get()
        pw2 = self.e_pw2.get()

        if not name or not email or not pw:
            messagebox.showerror("Registro", "Completa los campos obligatorios.")
            return
        if pw != pw2:
            messagebox.showerror("Registro", "Las contraseñas no coinciden.")
            return
        if self._get_user_by_email(email):
            messagebox.showerror("Registro", "Ya existe una cuenta con ese correo.")
            return

        pw_hash = hash_password(pw)
        self._insert_user(name, email, pw_hash, clinic)

        # Enviar OTP
        code = f"{np.random.randint(100000,999999)}"
        expires = datetime.utcnow() + timedelta(minutes=15)
        self._set_verification(email, code, expires)
        ok = send_verification_email(email, code)
        if ok:
            messagebox.showinfo("Registro", "Cuenta creada. Revisa tu correo y verifica en la pestaña correspondiente.")
        else:
            messagebox.showwarning("Registro", "No se pudo enviar el correo. Puedes reintentar más tarde.")

    def _do_verify(self):
        email = self.e_vemail.get().strip().lower()
        code  = self.e_vcode.get().strip()

        if not email or not code:
            messagebox.showerror("Verificar", "Completa email y código.")
            return

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT verification_code, verification_expires FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        if not row:
            conn.close()
            messagebox.showerror("Verificar", "Usuario no encontrado.")
            return

        vcode, vexp = row
        try:
            vexp_dt = datetime.fromisoformat(vexp) if vexp else None
        except Exception:
            vexp_dt = None

        if (code == vcode) and vexp_dt and datetime.utcnow() <= vexp_dt:
            cur.execute("UPDATE users SET email_verified=1, verification_code=NULL, verification_expires=NULL WHERE email=?", (email,))
            conn.commit()
            conn.close()
            messagebox.showinfo("Verificar", "Correo verificado. Ya puedes iniciar sesión.")
        else:
            conn.close()
            messagebox.showerror("Verificar", "Código inválido o expirado.")

# =========================
# UI: App principal
# =========================
class KidneyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("760x760")
        self.resizable(False, False)
        self.configure(bg="#F2F5F9")

        try:
            ico_path = resource_path("logo.ico")
            if os.path.exists(ico_path):
                self.iconbitmap(ico_path)
        except Exception:
            pass

        init_db()

        # Splash
        self.withdraw()
        Splash(self, duration=1600)
        self.after(1650, self._show_auth)

        # Estado
        self.user = None
        self.model_bundle = None
        self.inputs = {}
        self.last_vals = None
        self.last_probs = None
        self.last_pred = None
        self.last_meta = None

    def _show_auth(self):
        self.deiconify()
        auth = AuthWindow(self)
        self.wait_window(auth)
        if not auth.user:
            self.destroy(); return
        self.user = auth.user
        self._build_ui()
        self._load_model()
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
        ttk.Label(header, text=f"{APP_TITLE}", style="Header.TLabel").pack(anchor="center")
        ttk.Label(self, text=f"Sesión: {self.user['name']} — {self.user.get('clinic','(sin clínica)')}",
                  background="#F2F5F9").pack(pady=(4,0))

        container = ttk.Frame(self, padding=14)
        container.pack(fill="both", expand=True)

        # Datos básicos
        ttk.Label(container, text="Nombre paciente:").grid(row=0, column=0, sticky="w")
        self.entry_nombre = ttk.Entry(container, width=24)
        self.entry_nombre.grid(row=0, column=1, sticky="w")

        ttk.Label(container, text="Especie:").grid(row=0, column=2, sticky="w")
        self.combo_especie = ttk.Combobox(container, values=SPECIES_OPTIONS, state="readonly", width=18)
        self.combo_especie.current(0)
        self.combo_especie.grid(row=0, column=3, sticky="w")

        ttk.Label(container, text="Raza:").grid(row=1, column=0, sticky="w")
        self.entry_raza = ttk.Entry(container, width=24)
        self.entry_raza.grid(row=1, column=1, sticky="w")

        placeholders = {"Edad":"años", "Peso":"kg", "Creatinina":"mg/dL", "Urea":"mg/dL", "BUN":"mg/dL", "SDMA":"μg/dL"}
        for i, feat in enumerate(FEATURES, start=2):
            ttk.Label(container, text=f"{feat} ({placeholders.get(feat,'')}):").grid(row=i, column=0, sticky="w")
            e = ttk.Entry(container, width=18, validate="key")
            e["validatecommand"] = (e.register(self._validate_number), "%P", "%V")
            e.grid(row=i, column=1, sticky="w")
            self.inputs[feat] = e

        # Botones
        btns = ttk.Frame(container)
        btns.grid(row=2+len(FEATURES), column=0, columnspan=4, pady=12)
        ttk.Button(btns, text="Predecir", command=self.predict_single).pack(side="left", padx=6)
        ttk.Button(btns, text="Limpiar", command=self.reset_fields).pack(side="left", padx=6)
        self.btn_pdf = ttk.Button(btns, text="Generar PDF", command=self.generate_pdf_action, state="disabled")
        self.btn_pdf.pack(side="left", padx=6)
        self.btn_ai = ttk.Button(btns, text="Consultar con IA", command=self.consult_with_ai, state="disabled")
        self.btn_ai.pack(side="left", padx=6)


        # Resultado
        self.lbl_result = tk.Label(container, text="Riesgo: —", font=("Segoe UI", 16, "bold"), bg="#F2F5F9")
        self.lbl_result.grid(row=3+len(FEATURES), column=0, columnspan=4, pady=8)

        # Barras
        self.prog_bars = {}
        for idx, (label, color) in RISK_LABELS.items():
            frame = ttk.Frame(container)
            frame.grid(row=4+len(FEATURES)+idx, column=0, columnspan=4, sticky="we", pady=2)
            tk.Label(frame, text=label, width=8, anchor="w").pack(side="left")
            canvas = tk.Canvas(frame, width=580, height=20, bg="#E0E0E0", highlightthickness=0)
            canvas.pack(side="left", padx=5)
            pct_label = tk.Label(frame, text="0%", width=6, anchor="e")
            pct_label.pack(side="left")
            self.prog_bars[idx] = (canvas, pct_label, color)

        # Historial
        ttk.Label(container, text="Últimas 5 predicciones (mi cuenta):").grid(row=8+len(FEATURES), column=0, columnspan=4, pady=(6, 2))
        self.tree = ttk.Treeview(container, columns=("nombre","especie","riesgo","prob"), show="headings", height=6)
        self.tree.heading("nombre", text="Nombre");  self.tree.column("nombre", width=200, anchor="w")
        self.tree.heading("especie", text="Especie"); self.tree.column("especie", width=120, anchor="center")
        self.tree.heading("riesgo", text="Riesgo");  self.tree.column("riesgo", width=100, anchor="center")
        self.tree.heading("prob", text="Prob."); self.tree.column("prob", width=120, anchor="center")
        self.tree.grid(row=9+len(FEATURES), column=0, columnspan=4, sticky="we")
        self.tree.tag_configure("row_green", background="#C8E6C9")
        self.tree.tag_configure("row_orange", background="#FFE0B2")
        self.tree.tag_configure("row_red", background="#FFCDD2")
        # ✅ NUEVO: doble clic para cargar último caso del paciente
        self.tree.bind("<Double-1>", self._load_case_from_history)

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
        for _, (canvas, lbl, _) in self.prog_bars.items():
            canvas.delete("all"); lbl.config(text="0%")
        self.btn_pdf.config(state="disabled")
        self.last_vals = self.last_probs = self.last_pred = self.last_meta = None

    def _draw_gradient_bar(self, canvas, pct, color):
        canvas.delete("all")
        w = int(canvas.winfo_width()); h = int(canvas.winfo_height())
        fill_w = int(w * pct)
        r, g, b = self.winfo_rgb(color)
        r //= 256; g //= 256; b //= 256
        for i in range(fill_w):
            frac = i / max(fill_w, 1)
            rr = int(255 + (r - 255) * frac)
            gg = int(255 + (g - 255) * frac)
            bb = int(255 + (b - 255) * frac)
            canvas.create_line(i, 0, i, h, fill=f"#{rr:02x}{gg:02x}{bb:02x}")

    def _load_model(self):
        try:
            self.model_bundle = safe_load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showwarning("Modelo", f"No se pudo cargar el modelo ML: {e}\nSe usará solo la regla clínica.")
            self.model_bundle = None

    def predict_single(self):
        try:
            nombre = self.entry_nombre.get().strip() or "—"
            especie = self.combo_especie.get()
            raza = self.entry_raza.get().strip() or "—"
            raw_vals = {feat: self.inputs[feat].get() for feat in FEATURES}

            # ✅ Validación de entradas (rango/tipo)
            vals = validate_inputs(raw_vals)
        except ValidationError as ve:
            messagebox.showerror("Validación", str(ve))
            return
        except Exception:
            messagebox.showerror("Error", "Por favor ingresa todos los valores numéricos.")
            return

        if self.model_bundle:
            pred_idx, probs, dbg = predict_blended(vals, especie, self.model_bundle)
        else:
            pred_idx, probs = classify_band(vals, especie)
            dbg = {"probs_rules": probs, "probs_ml": None, "probs_ai": None, "weights": (1,0,0)}

        pred_label, color = RISK_LABELS[pred_idx]
        self.lbl_result.config(text=f"Riesgo: {pred_label}", bg=color, fg="white")
        for idx, (canvas, lbl, col) in self.prog_bars.items():
            p = float(probs[idx] if idx < len(probs) else 0)
            lbl.config(text=fmt_pct(p))
            self._draw_gradient_bar(canvas, p, col)

        self.last_vals = vals
        self.last_probs = probs
        self.last_pred = pred_label
        self.last_meta = {
            "nombre": nombre,
            "especie": especie,
            "raza": raza,
            "doctor_name": self.user["name"],
            "clinic": self.user.get("clinic")
        }
        self.btn_pdf.config(state="normal")
        self.btn_ai.config(state="normal")
        self._save_case(vals, pred_label, probs)
        self._load_quick_history()

    def _save_case(self, vals, risk_label, probs):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO casos_renales
                (fecha_hora, nombre, especie, raza, edad, peso, creatinina, urea, bun, sdma,
                 risk_label, prob_bajo, prob_medio, prob_alto, user_id, clinic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    self.last_meta["nombre"], self.last_meta["especie"], self.last_meta["raza"],
                    vals["Edad"], vals["Peso"], vals["Creatinina"], vals["Urea"], vals["BUN"], vals["SDMA"],
                    risk_label, float(probs[0]), float(probs[1]), float(probs[2]),
                    self.user["id"], self.user.get("clinic")
                )
            )
            conn.commit()
        except Exception as e:
            messagebox.showerror("DB", f"No se pudo guardar en la base de datos: {e}")
        finally:
            conn.close()

    def _load_quick_history(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT nombre, especie, risk_label, prob_bajo, prob_medio, prob_alto
            FROM casos_renales
            WHERE user_id=?
            ORDER BY id DESC LIMIT 5
            """,
            (self.user["id"],)
        )
        rows = cur.fetchall(); conn.close()
        for nombre, especie, risk_label, pb, pm, pa in rows:
            if risk_label == "Alto":
                tag, prob = "row_red", pa
            elif risk_label == "Medio":
                tag, prob = "row_orange", pm
            else:
                tag, prob = "row_green", pb
            self.tree.insert("", "end", values=(nombre, especie, risk_label, fmt_pct(prob)), tags=(tag,))

    # ✅ NUEVO: Cargar último caso del paciente al hacer doble clic
    def _load_case_from_history(self, event):
        selected = self.tree.focus()
        if not selected:
            return
        values = self.tree.item(selected, "values")
        if not values:
            return
        nombre, especie, riesgo, _ = values

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT edad, peso, creatinina, urea, bun, sdma, raza
            FROM casos_renales
            WHERE nombre=? AND especie=? AND user_id=?
            ORDER BY id DESC LIMIT 1
            """,
            (nombre, especie, self.user["id"]) 
        )
        row = cur.fetchone()
        conn.close()

        if row:
            edad, peso, creatinina, urea, bun, sdma, raza = row
            self.entry_nombre.delete(0, tk.END); self.entry_nombre.insert(0, nombre)
            self.combo_especie.set(especie)
            self.entry_raza.delete(0, tk.END); self.entry_raza.insert(0, raza or "—")
            vals = [edad, peso, creatinina, urea, bun, sdma]
            for feat, val in zip(FEATURES, vals):
                e = self.inputs[feat]
                e.delete(0, tk.END)
                e.insert(0, str(val))
            messagebox.showinfo("Historial", f"Datos de {nombre} cargados para edición.")

    def generate_pdf_action(self):
        if not self.last_vals:
            messagebox.showinfo("PDF", "No hay predicción reciente.")
            return
        try:
            safe_name = "".join([c for c in self.last_meta['nombre'] if c.isalnum() or c in (' ','_','-')]).strip() or 'paciente'
            filename = f"reporte_{safe_name}.pdf"
            generate_pdf(self.last_vals, self.last_meta, self.last_pred, self.last_probs, filename=filename)

            # ✅ Enviar correo automáticamente al usuario logueado
            to_email = self.user.get("email")
            if to_email:
                send_email_with_pdf(to_email, filename, subject="Reporte de Predicción Renal", body="Adjunto el reporte del paciente.")
                messagebox.showinfo("PDF", f"PDF generado y enviado a {to_email}")
            else:
                messagebox.showinfo("PDF", f"PDF generado: {filename}")
        except Exception as e:
            messagebox.showerror("Error PDF", f"No se pudo generar/enviar el PDF: {e}")



    def consult_with_ai(self):
        if self.last_vals is None or self.last_probs is None:
            messagebox.showerror("IA", "Primero realiza una predicción.")
            return
        try:
            prompt = (
                f"Sos un asistente experto en nefrología veterinaria.\n"
                f"Paciente: {self.last_meta['nombre']}, especie: {self.last_meta['especie']}, raza: {self.last_meta['raza']}.\n"
                f"Edad: {self.last_vals['Edad']} años, Peso: {self.last_vals['Peso']} kg.\n"
                f"Creatinina: {self.last_vals['Creatinina']}, Urea: {self.last_vals['Urea']}, "
                f"BUN: {self.last_vals['BUN']}, SDMA: {self.last_vals['SDMA']}.\n"
                f"Resultado del modelo: Riesgo {self.last_pred} con probabilidades "
                f"Bajo {fmt_pct(self.last_probs[0])}, Medio {fmt_pct(self.last_probs[1])}, Alto {fmt_pct(self.last_probs[2])}.\n\n"
                "Da una interpretación breve y recomendaciones clínicas."
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )

            answer = response.choices[0].message.content

            win = tk.Toplevel(self)
            win.title("Opinión IA")
            txt = tk.Text(win, wrap="word", width=80, height=20)
            txt.insert("1.0", answer)
            txt.pack(padx=10, pady=10)
        except Exception as e:
            messagebox.showerror("IA", f"No se pudo consultar la IA: {e}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    app = KidneyApp()
    app.mainloop()
