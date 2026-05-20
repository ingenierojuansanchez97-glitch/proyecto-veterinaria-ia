# create_env.py
import os

env_content = """# Configuración SMTP para Gmail
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=sebaxto@gmail.com
SMTP_PASS=tu_contraseña_de_aplicacion
SMTP_FROM=sebaxto@gmail.com

# Rutas del modelo y la base de datos
KIDNEY_MODEL=./model_kidney.pkl
KIDNEY_APP_DB=./veterinaria.db

# Opcional: correo adicional de resultados
RESULTS_EMAIL=sebaxto@gmail.com
"""

# Guardar el archivo .env
with open(".env", "w", encoding="utf-8") as f:
    f.write(env_content)

print("✅ Archivo .env creado correctamente en:", os.path.abspath(".env"))
