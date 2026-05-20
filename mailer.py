# mailer.py
# Envío de correos optimizado para producción (Soporta TLS / Puerto 587 para Render/Cloud)

import os
import smtplib
import ssl
from email.message import EmailMessage

# Configuración desde variables de entorno (Render, Railway o local .env)
# Se cambia el puerto por defecto a 587 ya que el 465 suele estar bloqueado en la nube.
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))  
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)


def send_email(to_email: str, subject: str, body: str, pdf_path: str = None) -> bool:
    """
    Envía un correo electrónico. Si se pasa `pdf_path`, adjunta el archivo PDF.
    Retorna True si el envío fue exitoso, False si falló.
    """
    if not to_email:
        print("❌ Error: No se especificó el correo de destino.")
        return False

    if not SMTP_USER or not SMTP_PASS:
        print("❌ Error de configuración: SMTP_USER o SMTP_PASS no definidos en las variables de entorno.")
        return False

    # Crear el objeto del mensaje
    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Adjuntar el archivo PDF si se proporciona una ruta válida
    if pdf_path:
        if os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as f:
                    file_data = f.read()
                    msg.add_attachment(
                        file_data,
                        maintype="application",
                        subtype="pdf",
                        filename=os.path.basename(pdf_path),
                    )
                print(f"📎 Archivo adjunto cargado correctamente: {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"⚠️ No se pudo leer o adjuntar el archivo PDF: {e}")
        else:
            print(f"⚠️ Advertencia: La ruta del PDF no existe ({pdf_path}). Se enviará el correo sin el adjunto.")

    try:
        # Crear un contexto SSL seguro por defecto
        context = ssl.create_default_context()
        
        # Conexión usando el puerto estándar SMTP (587)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()             # Identificarse con el servidor SMTP
            server.starttls(context=context)  # Cifrar la conexión usando TLS (Obligatorio para evitar bloqueos)
            server.ehlo()             # Re-identificarse de forma segura
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
            
        adjunto_status = f"con adjunto [{os.path.basename(pdf_path)}]" if (pdf_path and os.path.exists(pdf_path)) else "(sin adjunto)"
        print(f"✅ Correo enviado con éxito a {to_email} {adjunto_status}")
        return True
        
    except Exception as e:
        print(f"⚠️ Error al enviar correo: {e}")
        return False
