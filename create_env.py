# create_env.py
import os

# Definimos el contenido del archivo .env
# Reemplaza 'tu_contraseña_de_aplicacion_aqui' con el código de 16 letras que obtendrás de Google
env_content = """# ==============================================================================
# Configuración SMTP para Gmail
# ==============================================================================
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=sebaxto@gmail.com
SMTP_PASS=vsytovtqrbjnvuvi
SMTP_FROM=sebaxto@gmail.com

# ==============================================================================
# Rutas del Modelo y Base de Datos
# ==============================================================================
KIDNEY_MODEL=./model_kidney.pkl
KIDNEY_APP_DB=./veterinaria.db

# ==============================================================================
# Configuración Opcional
# ==============================================================================
RESULTS_EMAIL=sebaxto@gmail.com
"""

# Guardar el archivo .env con codificación UTF-8
with open(".env", "w", encoding="utf-8") as f:
    f.write(env_content)

print("✅ Archivo .env creado correctamente en:", os.path.abspath(".env"))
print("⚠️ Recuerda reemplazar 'tu_contraseña_de_aplicacion_aqui' por tu clave de 16 dígitos de Google.")
