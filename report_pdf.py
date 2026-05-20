# report_pdf.py actualizado y mejorado

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime


def generate_pdf(
    patient_data: dict,
    prediction: str,
    probabilities: dict,
    filename: str = "reporte_renal.pdf",
    doctor: str = "Juan Sanchez",
    clinica: str = "San Sebastian",
    comentario_ia: str = None,
):
    """
    Genera un PDF estilo GUI (igual al de app_kidney_gui_v2).
    patient_data: dict con datos del paciente (Nombre, Especie, Raza, Edad, etc.)
    prediction: etiqueta de riesgo ("Bajo", "Medio", "Alto")
    probabilities: dict con probabilidades {"Bajo": x, "Medio": y, "Alto": z}
    doctor: nombre del veterinario
    clinica: nombre de la clínica
    comentario_ia: texto opcional de IA
    """
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    nombre = patient_data.get("Nombre", "Paciente")
    especie = patient_data.get("Especie", "-")
    raza = patient_data.get("Raza", "-")

    # === Encabezado ===
    elements.append(Paragraph(f"Reporte de Riesgo Renal - {nombre}", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Especie: {especie}    Raza: {raza}", styles["Normal"]))
    
    # Riesgo resaltado con color
    color_map = {"Bajo": colors.green, "Medio": colors.orange, "Alto": colors.red}
    risk_color = color_map.get(prediction, colors.black)
    elements.append(Paragraph(f"<b><font color='{risk_color}'>Riesgo: {prediction}</font></b>", styles["Normal"]))

    # Probabilidades en tabla
    probs_table = Table(
        [[k, f"{v*100:.1f}%"] for k, v in probabilities.items()],
        colWidths=[100, 80],
    )
    probs_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    elements.append(probs_table)
    elements.append(Spacer(1, 12))

    # === Datos clínicos ===
    clinical_data = [[k, str(v)] for k, v in patient_data.items() if k not in ("Nombre", "Especie", "Raza")]
    if clinical_data:
        elements.append(Paragraph("Parámetros clínicos", styles["Heading2"]))
        table = Table(clinical_data, colWidths=[120, 120])
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 12))

    # === Doctor y clínica ===
    elements.append(Paragraph(f"Atendido por: {doctor} ({clinica})", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # === Recomendaciones estándar ===
    elements.append(Paragraph("Recomendaciones", styles["Heading2"]))
    recs = [
        "- Mantener controles de rutina.",
        "- Dieta balanceada y agua fresca.",
        "- Repetir perfil renal si aparecen síntomas.",
    ]
    for r in recs:
        elements.append(Paragraph(r, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # === Comentario de IA ===
    if comentario_ia:
        elements.append(Paragraph("Comentario de IA", styles["Heading2"]))
        elements.append(Paragraph(comentario_ia, styles["Normal"]))
        elements.append(Spacer(1, 12))

    # === Fecha ===
    elements.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))

    doc.build(elements)
    return filename
