# mailer.py
# Envío de correos con o sin PDF adjunto

import os
import smtplib
import ssl
from email.message import EmailMessage

# Configuración desde variables de entorno (Railway las inyecta)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)


def send_email(to_email: str, subject: str, body: str, pdf_path: str = None) -> bool:
    """
    Envía un correo. Si se pasa `pdf_path`, adjunta el PDF.
    Retorna True si el envío fue exitoso, False si falló.
    """
    if not to_email:
        raise ValueError("❌ No se especificó correo destino")

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Adjuntar PDF si existe
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            file_data = f.read()
            msg.add_attachment(
                file_data,
                maintype="application",
                subtype="pdf",
                filename=os.path.basename(pdf_path),
            )

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"✅ Correo enviado a {to_email} {'con adjunto ' + pdf_path if pdf_path else '(sin adjunto)'}")
        return True
    except Exception as e:
        print(f"⚠️ Error al enviar correo: {e}")
        return False
