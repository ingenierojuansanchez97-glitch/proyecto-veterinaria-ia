# app.py — Web app completa con auth, dashboard, predicción, PDF e IA
import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Se incluye BackgroundTasks para evitar bloqueos por envíos síncronos de correo
from fastapi import FastAPI, Request, Form, HTTPException, status, BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# ==== módulos locales ====
from validator import validate_inputs, ValidationError
from mailer import send_email
from ai_consult import consult_ai
from report_pdf import generate_pdf as generate_report_pdf
from predict_kidney_model import safe_load_model, safe_predict_proba


# ======================= Config =======================
APP_TITLE = "Predicción Renal Veterinaria"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.getenv("KIDNEY_APP_DB", os.path.join(BASE_DIR, "veterinaria.db"))
MODEL_PATH = os.getenv("KIDNEY_MODEL", os.path.join(BASE_DIR, "model_kidney.pkl"))

SECRET_KEY = os.getenv("APP_SECRET_KEY", "supersecreto123")

# Orden exacto con el que el nuevo modelo fue entrenado
FEATURES = ["Es_Felino", "Edad", "Peso", "Creatinina", "Urea", "BUN", "SDMA"]
SPECIES_OPTIONS = ["Canino", "Felino", "Ave", "Pez", "Reptil", "Caballo", "Cerdo", "Vaca", "Otro"]

RISK_LABELS: Dict[int, Tuple[str, str]] = {
    0: ("Bajo", "#43A047"),
    1: ("Medio", "#FB8C00"),
    2: ("Alto", "#E53935"),
}

# Límites diagnósticos específicos por especie según guías internacionales IRIS
REF_RANGES = {
    "Canino": {"Creatinina": 1.4, "Urea": 50.0, "BUN": 27.0, "SDMA": 14.0},
    "Felino": {"Creatinina": 1.6, "Urea": 60.0, "BUN": 30.0, "SDMA": 15.0},
}


# ==================== App & assets ====================
app = FastAPI(title=APP_TITLE)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


# ====================== DB utils ======================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    # Tabla de usuarios
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

    # Tabla de casos
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS casos_renales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_hora TEXT,
            nombre TEXT,
            especie TEXT,
            raza TEXT,
            edad REAL,
            peso REAL,
            creatinina REAL,
            urea REAL,
            bun REAL,
            sdma REAL,
            risk_label TEXT,
            prob_bajo REAL,
            prob_medio REAL,
            prob_alto REAL,
            user_id INTEGER,
            clinic TEXT,
            ai_text TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    # Migraciones defensivas (idempotentes)
    def ensure(table, col, ddl):
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
        except Exception:
            pass

    ensure("casos_renales", "user_id", "INTEGER")
    ensure("casos_renales", "clinic", "TEXT")
    ensure("casos_renales", "ai_text", "TEXT")

    conn.commit()
    conn.close()


init_db()


# ============= Seguridad (hash de password) ============
def hash_password(pw: str, salt: Optional[str] = None) -> str:
    if not salt:
        salt = secrets.token_hex(8)
    h = hashlib.sha256((salt + pw).encode("utf-8")).hexdigest()
    return f"{salt}${h}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        salt, _h = stored.split("$")
        return hash_password(pw, salt) == stored
    except Exception:
        return False


# ======== Reglas clínicas + predicción combinada ========
def classify_band(vals: dict, especie: str = "Canino"):
    # Si la especie no está en el mapa, por defecto usamos los límites caninos
    esp_key = "Felino" if especie.lower() == "felino" else "Canino"
    refs = REF_RANGES[esp_key]
    
    crea = float(vals.get("Creatinina", 0.0))
    sdma = float(vals.get("SDMA", 0.0))
    bun = float(vals.get("BUN", 0.0))
    urea = float(vals.get("Urea", 0.0))

    # Evaluación y asignación probabilística adaptada a guías reales IRIS
    if esp_key == "Felino":
        if crea >= 2.9 or sdma >= 26.0 or urea > 100.0:
            band_idx = 2  # Alto
            p_dist = [0.00, 0.05, 0.95]
        elif 1.6 <= crea < 2.9 or 15.0 <= sdma < 26.0:
            band_idx = 1  # Medio (Estadio IRIS 2 temprano/moderado)
            p_dist = [0.15, 0.70, 0.15]
        else:
            band_idx = 0  # Bajo
            p_dist = [0.85, 0.10, 0.05]
    else:  # Caninos
        if crea >= 3.5 or sdma >= 35.0 or urea > 120.0:
            band_idx = 2  # Alto
            p_dist = [0.00, 0.05, 0.95]
        elif 1.4 <= crea < 3.5 or 15.0 <= sdma < 35.0:
            band_idx = 1  # Medio
            p_dist = [0.15, 0.70, 0.15]
        else:
            band_idx = 0  # Bajo
            p_dist = [0.85, 0.10, 0.05]

    return band_idx, np.array(p_dist, dtype=float)


# Cargar modelo de forma segura de predict_kidney_model
MODEL_BUNDLE = safe_load_model(MODEL_PATH)


def predict_blended(vals: dict, especie: str, model_bundle: tuple) -> Tuple[int, np.ndarray]:
    # 1) Reglas clínicas balanceadas
    _, probs_rules = classify_band(vals, especie)

    # 2) Procesamiento con el Modelo ML
    model, feat_order = model_bundle
    
    # Preparar el vector inyectando el valor binario de la especie para el clasificador
    input_data = vals.copy()
    input_data["Es_Felino"] = 1.0 if especie.lower() == "felino" else 0.0
    
    # Asegurar el DataFrame con las 7 columnas exactas requeridas
    X = pd.DataFrame([input_data])[feat_order]
    
    probs_ml = safe_predict_proba(model, X)[0]
    probs_ml = np.array(probs_ml, dtype=float)
    
    s = probs_ml.sum()
    if s <= 0:
        probs_ml = np.array([1/3, 1/3, 1/3])
    else:
        probs_ml = probs_ml / s

    # 3) Mezcla calibrada (Blending): 40% Reglas Expertas, 60% Aprendizaje Automático
    w_rules = 0.40
    w_ml = 0.60
    final_probs = w_rules * probs_rules + w_ml * probs_ml
    final_probs = final_probs / final_probs.sum()
    
    pred_idx = int(np.argmax(final_probs))

    try:
        print(f"[DEBUG CLÍNICO] Especie: {especie} -> Rules: {probs_rules.tolist()} | ML: {probs_ml.tolist()} | Blended: {final_probs.tolist()}")
    except Exception:
        pass

    return pred_idx, final_probs


# =============== Utilidades de sesión ==================
def current_user(request: Request) -> Optional[Dict[str, Any]]:
    return request.session.get("user")


def require_login_or_redirect(request: Request) -> Optional[RedirectResponse]:
    if not current_user(request):
        return RedirectResponse("/auth", status_code=status.HTTP_303_SEE_OTHER)
    return None


def _history_for(user_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, nombre, especie, raza, risk_label, prob_bajo, prob_medio, prob_alto, fecha_hora
        FROM casos_renales
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT 10
        """,
        (user_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ====================== Rutas HTML ======================
@app.get("/")
def home(request: Request):
    if current_user(request):
        return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse("/auth", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/auth")
def auth_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="auth.html",
        context={"title": APP_TITLE}
    )


@app.post("/auth/register")
def auth_register(
    request: Request,
    background_tasks: BackgroundTasks,  # Inyección de tareas en segundo plano
    name: str = Form(...),
    email: str = Form(...),
    clinic: str = Form(""),
    password: str = Form(...),
    password2: str = Form(...),
):
    if password != password2:
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={"title": APP_TITLE, "error": "Las contraseñas no coinciden."}
        )

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email=?", (email.lower().strip(),))
    if cur.fetchone():
        conn.close()
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={"title": APP_TITLE, "error": "El correo ya está registrado."}
        )

    pw_hash = hash_password(password)
    code = secrets.token_hex(3).upper()
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    cur.execute(
        "INSERT INTO users (name,email,password_hash,clinic,verification_code,verification_expires) VALUES (?,?,?,?,?,?)",
        (name.strip(), email.lower().strip(), pw_hash, clinic.strip(), code, expires),
    )
    conn.commit()
    conn.close()

    # En vez de ejecutar send_email de forma síncrona aquí, lo delegamos a FastAPI:
    try:
        background_tasks.add_task(
            send_email, 
            email.lower().strip(), 
            "Verificación de cuenta", 
            f"Tu código de verificación es: {code}"
        )
    except Exception as e:
        print("⚠️ Error al delegar la tarea de correo de verificación:", e)

    return templates.TemplateResponse(
        request=request,
        name="auth.html",
        context={
            "title": APP_TITLE,
            "success": "Cuenta creada con éxito. Ingresa el código enviado a tu correo corporativo.",
        }
    )


@app.post("/auth/verify")
def auth_verify(request: Request, email: str = Form(...), code: str = Form(...)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, verification_code, verification_expires FROM users WHERE email=?",
        (email.lower().strip(),),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={"title": APP_TITLE, "error": "Usuario no encontrado."}
        )

    uid, real_code, exp = row
    if (real_code or "").upper().strip() != code.upper().strip():
        conn.close()
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={"title": APP_TITLE, "error": "Código inválido."}
        )

    if exp and datetime.now(timezone.utc) > datetime.fromisoformat(exp).replace(tzinfo=timezone.utc):
        conn.close()
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={"title": APP_TITLE, "error": "El código ha expirado."}
        )

    cur.execute(
        "UPDATE users SET email_verified=1, verification_code=NULL, verification_expires=NULL WHERE id=?",
        (uid,),
    )
    conn.commit()
    conn.close()

    return templates.TemplateResponse(
        request=request,
        name="auth.html",
        context={"title": APP_TITLE, "success": "Correo verificado, ahora puedes iniciar sesión."}
    )


@app.post("/auth/login")
def auth_login(request: Request, email: str = Form(...), password: str = Form(...)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, email, password_hash, clinic, role, email_verified FROM users WHERE email=?",
        (email.lower().strip(),),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={"title": APP_TITLE, "error": "Usuario no encontrado."}
        )

    uid, name, email, pw_hash, clinic, role, verified = row
    if not verify_password(password, pw_hash):
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={"title": APP_TITLE, "error": "Contraseña incorrecta."}
        )

    if not verified:
        return templates.TemplateResponse(
            request=request,
            name="auth.html",
            context={"title": APP_TITLE, "error": "Cuenta pendiente por verificación por correo."}
        )

    request.session["user"] = {"id": uid, "name": name, "email": email, "clinic": clinic, "role": role}
    return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/auth", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/dashboard")
def dashboard(request: Request):
    redirect = require_login_or_redirect(request)
    if redirect:
        return redirect

    user = current_user(request)
    vals = {}
    result = None
    case_id = None
    
    try:
        q = dict(request.query_params)
        if "case_id" in q and q["case_id"]:
            case_id = int(q["case_id"])
    except Exception:
        case_id = None

    if case_id:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM casos_renales WHERE id=? AND user_id=?", (case_id, user["id"]))
        row = cur.fetchone()
        conn.close()
        if row:
            vals = {
                "Nombre": row["nombre"],
                "Especie": row["especie"],
                "Raza": row["raza"],
                "Edad": row["edad"],
                "Peso": row["peso"],
                "Creatinina": row["creatinina"],
                "Urea": row["urea"],
                "BUN": row["bun"],
                "SDMA": row["sdma"],
            }
            result = {
                "nombre": row["nombre"],
                "especie": row["especie"],
                "raza": row["raza"],
                "pred_label": row["risk_label"],
                "probs": [row["prob_bajo"], row["prob_medio"], row["prob_alto"]],
                "ai_text": row["ai_text"],
            }

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": APP_TITLE,
            "user": user,
            "species": SPECIES_OPTIONS,
            "vals": vals,
            "result": result,
            "history": _history_for(user["id"]),
        }
    )


# =================== Predicción (HTML) ===================
@app.post("/predict")
def predict_form(
    request: Request,
    nombre: str = Form("Paciente"),
    especie: str = Form("Canino"),
    raza: str = Form("—"),
    Edad: float = Form(..., ge=0, le=40),
    Peso: float = Form(..., ge=0.5, le=120),
    Creatinina: float = Form(..., ge=0),
    Urea: float = Form(..., ge=0),
    BUN: float = Form(..., ge=0),
    SDMA: float = Form(..., ge=0),
):
    redirect = require_login_or_redirect(request)
    if redirect:
        return redirect

    user = current_user(request)
    raw_vals = {
        "Edad": Edad,
        "Peso": Peso,
        "Creatinina": Creatinina,
        "Urea": Urea,
        "BUN": BUN,
        "SDMA": SDMA,
    }

    try:
        vals = validate_inputs(raw_vals)
    except ValidationError as e:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "title": APP_TITLE,
                "user": user,
                "species": SPECIES_OPTIONS,
                "error": str(e),
                "vals": {},
                "history": _history_for(user["id"]),
            }
        )

    pred_idx, probs = predict_blended(vals, especie, MODEL_BUNDLE)
    pred_label, _ = RISK_LABELS[pred_idx]

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO casos_renales
            (fecha_hora, nombre, especie, raza, edad, peso, creatinina, urea, bun, sdma,
             risk_label, prob_bajo, prob_medio, prob_alto, user_id, clinic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                nombre.strip(),
                especie,
                raza.strip(),
                float(vals["Edad"]),
                float(vals["Peso"]),
                float(vals["Creatinina"]),
                float(vals["Urea"]),
                float(vals["BUN"]),
                float(vals["SDMA"]),
                pred_label,
                float(probs[0]),
                float(probs[1]),
                float(probs[2]),
                user["id"],
                user.get("clinic"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "title": APP_TITLE,
                "user": user,
                "species": SPECIES_OPTIONS,
                "error": f"Error de Base de Datos: {e}",
                "vals": {},
                "history": _history_for(user["id"]),
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": APP_TITLE,
            "user": user,
            "species": SPECIES_OPTIONS,
            "success": f"Análisis completado: Riesgo {pred_label}",
            "vals": vals,
            "result": {
                "nombre": nombre,
                "especie": especie,
                "raza": raza,
                "pred_label": pred_label,
                "probs": [float(x) for x in probs],
            },
            "history": _history_for(user["id"]),
        }
    )


# ========================= PDF =========================
@app.post("/predict/pdf")
def predict_and_pdf(
    request: Request,
    nombre: str = Form("Paciente"),
    especie: str = Form("Canino"),
    raza: str = Form("—"),
    Edad: float = Form(...),
    Peso: float = Form(...),
    Creatinina: float = Form(...),
    Urea: float = Form(...),
    BUN: float = Form(...),
    SDMA: float = Form(...),
):
    redirect = require_login_or_redirect(request)
    if redirect:
        return redirect

    user = current_user(request)
    raw_vals = {
        "Edad": Edad,
        "Peso": Peso,
        "Creatinina": Creatinina,
        "Urea": Urea,
        "BUN": BUN,
        "SDMA": SDMA,
    }

    try:
        vals = validate_inputs(raw_vals)
    except ValidationError as e:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "title": APP_TITLE,
                "user": user,
                "species": SPECIES_OPTIONS,
                "error": str(e),
                "vals": {},
                "history": _history_for(user["id"]),
            }
        )

    pred_idx, probs = predict_blended(vals, especie, MODEL_BUNDLE)
    pred_label, _ = RISK_LABELS[pred_idx]

    # Atributos limpios para renderizado del PDF corporativo
    patient_data = {
        "Nombre": nombre,
        "Especie": especie,
        "Raza": raza,
        "Edad": vals["Edad"],
        "Peso": vals["Peso"],
        "Creatinina": vals["Creatinina"],
        "Urea": vals["Urea"],
        "BUN": vals["BUN"],
        "SDMA": vals["SDMA"]
    }
    prob_dict = {
        "Bajo": float(probs[0]),
        "Medio": float(probs[1]),
        "Alto": float(probs[2]),
    }

    safe_name = "".join([c for c in nombre if c.isalnum() or c in (" ", "_", "-")]).strip() or "paciente"
    pdf_path = os.path.join(BASE_DIR, f"reporte_{safe_name}.pdf")

    try:
        generate_report_pdf(
            patient_data,
            pred_label,
            prob_dict,
            filename=pdf_path,
            doctor=user.get("name", "Veterinario"),
            clinica=user.get("clinic", "Clínica"),
            comentario_ia=None,
        )

        from shutil import copyfile
        public_path = os.path.join(static_dir, os.path.basename(pdf_path))
        copyfile(pdf_path, public_path)
        pdf_url = f"/static/{os.path.basename(pdf_path)}"

        try:
            send_email(
                to=user.get("email"),
                subject=f"Reporte Digital de Diagnóstico Renal - {nombre}",
                body=f"Estimado Dr. {user.get('name')},\n\nAdjunto encontrará el informe clínico automatizado en formato PDF correspondiente al paciente '{nombre}'.",
                attachment_path=pdf_path,
            )
        except Exception as e:
            print("⚠️ Error al despachar correo electrónico:", e)

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "title": APP_TITLE,
                "user": user,
                "species": SPECIES_OPTIONS,
                "error": f"No se pudo estructurar el reporte PDF: {e}",
                "vals": {},
                "history": _history_for(user["id"]),
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": APP_TITLE,
            "user": user,
            "species": SPECIES_OPTIONS,
            "success": "✅ PDF generado y enviado con éxito al correo del remitente.",
            "vals": {
                "Nombre": nombre,
                "Especie": especie,
                "Raza": raza,
                **vals,
            },
            "result": {
                "nombre": nombre,
                "especie": especie,
                "raza": raza,
                "pred_label": pred_label,
                "probs": [float(x) for x in probs],
            },
            "pdf_url": pdf_url,
            "history": _history_for(user["id"]),
        }
    )


# ==================== Consulta con IA ====================
@app.post("/ai/consult")
def ai_consult_route(
    request: Request,
    nombre: str = Form(...),
    especie: str = Form(...),
    raza: str = Form(""),
    Edad: Optional[float] = Form(None),
    Peso: Optional[float] = Form(None),
    Creatinina: Optional[float] = Form(None),
    Urea: Optional[float] = Form(None),
    BUN: Optional[float] = Form(None),
    SDMA: Optional[float] = Form(None),
):
    redirect = require_login_or_redirect(request)
    if redirect:
        return redirect

    user = current_user(request)
    raw_vals = {
        "Edad": Edad or 0.0,
        "Peso": Peso or 0.0,
        "Creatinina": Creatinina or 0.0,
        "Urea": Urea or 0.0,
        "BUN": BUN or 0.0,
        "SDMA": SDMA or 0.0,
    }

    try:
        vals = validate_inputs(raw_vals)
    except ValidationError as e:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "title": APP_TITLE,
                "user": user,
                "species": SPECIES_OPTIONS,
                "error": str(e),
                "vals": {},
                "history": _history_for(user["id"]),
            }
        )

    pred_idx, probs = predict_blended(vals, especie, MODEL_BUNDLE)
    pred_label, _ = RISK_LABELS[pred_idx]

    patient_data = {"Nombre": nombre, "Especie": especie, "Raza": raza, **vals}
    prob_dict = {"Bajo": float(probs[0]), "Medio": float(probs[1]), "Alto": float(probs[2])}

    try:
        ai_text = consult_ai(patient_data, pred_label, prob_dict)
        
        # Persistencia del reporte clínico generado por IA en la base de datos local
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE casos_renales 
            SET ai_text = ? 
            WHERE user_id = ? AND nombre = ? 
            ORDER BY id DESC LIMIT 1
            """,
            (ai_text, user["id"], nombre.strip())
        )
        conn.commit()
        conn.close()
        
    except Exception as e:
        ai_text = f"No se pudo consultar a la IA: {e}"

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": APP_TITLE,
            "user": user,
            "species": SPECIES_OPTIONS,
            "success": "Análisis avanzado de soporte de IA completado.",
            "vals": {
                "Nombre": nombre,
                "Especie": especie,
                "Raza": raza,
                **vals,
            },
            "result": {
                "nombre": nombre,
                "especie": especie,
                "raza": raza,
                "pred_label": pred_label,
                "probs": [float(x) for x in probs],
                "ai_text": ai_text,
            },
            "history": _history_for(user["id"]),
        }
    )


# ==================== Healthcheck simple ====================
@app.get("/health")
def health():
    return {"ok": True}
