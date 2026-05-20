import os
from google import genai
from google.genai import types

def consult_ai(patient_data: dict, pred_label: str, prob_dict: dict) -> str:
    """
    Intenta consultar a Gemini 2.5. Si falla por saturación (503) u otro motivo,
    activa un fallback local para que la app nunca se quede sin servicio.
    """
    nombre = patient_data.get('Nombre', 'Paciente')
    especie = patient_data.get('Especie', 'No especificada')
    raza = patient_data.get('Raza', 'No especificada')
    edad = patient_data.get('Edad')
    peso = patient_data.get('Peso')
    
    # 1. Intentar la vía principal con Inteligencia Artificial
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Falta GEMINI_API_KEY")

        client = genai.Client(api_key=api_key)

        prompt = f"""
        Eres un experto en nefrología veterinaria. Analiza el siguiente caso clínico:
        
        INFORMACIÓN DEL PACIENTE:
        - Nombre: {nombre}
        - Especie: {especie}
        - Raza: {raza}
        - Edad: {edad} años
        - Peso: {peso} kg
        
        BIOMARCADORES RENALES:
        - Creatinina: {patient_data.get('Creatinina')} mg/dL
        - Urea: {patient_data.get('Urea')} mg/dL
        - BUN: {patient_data.get('BUN')} mg/dL
        - SDMA: {patient_data.get('SDMA')} ug/dL
        
        RESULTADO DEL ALGORITMO:
        - Clasificación de Riesgo: {pred_label}
        - Probabilidades: Bajo ({prob_dict.get('Bajo', 0)*100:.1f}%), Medio ({prob_dict.get('Medio', 0)*100:.1f}%), Alto ({prob_dict.get('Alto', 0)*100:.1f}%)
        
        INSTRUCCIONES:
        Genera un informe analítico, profesional y directo para el veterinario.
        1. Interpreta los biomarcadores específicos para su especie.
        2. Explica la coherencia del riesgo '{pred_label}'.
        3. Da recomendaciones de manejo clínico, diagnóstico diferencial y pautas terapéuticas iniciales.
        Estructura con viñetas limpias.
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text

    # 2. Plan B: Si la IA está saturada (503), bloqueada o da error, se ejecuta esto de inmediato
    except Exception as e:
        print(f"⚠️ Servidor de Gemini saturado o no disponible ({e}). Activando fallback local.")
        
        # Estructuramos un reporte clínico dinámico local simulando el flujo de la IA
        if pred_label == "Alto":
            fallback_text = (
                f"📋 **[Informe Clínico Automatizado - Contingencia por Alta Demanda]**\n\n"
                f"El paciente **{nombre}** ({especie}, {raza}) presenta un perfil bioquímico compatible con "
                f"un **Riesgo ALTO** de disfunción renal aguda o crónica reagudizada.\n\n"
                f"**Interpretación de Biomarcadores:**\n"
                f"• Los niveles de Creatinina, Urea/BUN y SDMA ingresados superan el umbral crítico tolerado. "
                f"La probabilidad matemática de daño estructural es de un {prob_dict.get('Alto', 0)*100:.1f}%.\n\n"
                f"**Recomendaciones Clínicas de Urgencia:**\n"
                f"1. **Estabilización:** Iniciar fluidoterapia intravenosa adaptada para restaurar la perfusión renal, vigilando la producción de orina.\n"
                f"2. **Seguridad farmacológica:** Retirar inmediatamente AINEs, aminoglucósidos o cualquier agente nefrotóxico.\n"
                f"3. **Diagnóstico complementario:** Realizar ecografía abdominal/renal para evaluar arquitectura y un uroanálisis completo (UPC y densidad).\n"
                f"4. **Manejo:** Evaluar la colocación de protectores gastrointestinales por posible gastritis urémica secundaria."
            )
        elif pred_label == "Medio":
            fallback_text = (
                f"📋 **[Informe Clínico Automatizado - Contingencia por Alta Demanda]**\n\n"
                f"El paciente **{nombre}** ({especie}, {raza}) se encuentra clasificado en **Riesgo MEDIO** "
                f"con una probabilidad del {prob_dict.get('Medio', 0)*100:.1f}%.\n\n"
                f"**Interpretación de Biomarcadores:**\n"
                f"• Se observa una elevación incipiente o moderada en los analitos renales. Esto puede sugerir una fase "
                f"temprana de enfermedad renal (IRIS Estadio 1 o 2) o azotemia prerrenal por deshidratación/cardiopatías.\n\n"
                f"**Recomendaciones de Monitoreo:**\n"
                f"1. **Repetición:** Revaluar el perfil renal completo bajo condiciones óptimas de hidratación en un lapso de 48 a 72 horas.\n"
                f"2. **Estudio urinario:** La densidad urinaria es crucial aquí para confirmar si el riñón conserva su capacidad de concentración.\n"
                f"3. **Soporte Nutricional:** Considerar el inicio preventivo de una dieta húmeda de soporte renal o quelantes de fósforo si la bioquímica lo amerita.\n"
                f"4. **Presión arterial:** Tomar la presión sistólica para descartar hipertensión sistémica concomitante."
            )
        else:
            fallback_text = (
                f"📋 **[Informe Clínico Automatizado - Contingencia por Alta Demanda]**\n\n"
                f"El paciente **{nombre}** ({especie}) ha sido catalogado en **Riesgo BAJO** "
                f"(Probabilidad: {prob_dict.get('Bajo', 0)*100:.1f}%).\n\n"
                f"**Interpretación de Biomarcadores:**\n"
                f"• Los valores de Creatinina, BUN y el biomarcador de detección temprana SDMA se localizan "
                f"dentro de los límites fisiológicos estándar o presentan variaciones no significativas.\n\n"
                f"**Recomendaciones Preventivas:**\n"
                f"1. **Seguimiento:** Mantener el protocolo de medicina preventiva anual o semestral recomendado para su grupo de edad.\n"
                f"2. **Hidratación:** Asegurar acceso irrestricto a fuentes de agua limpia.\n"
                f"3. **Nota:** Este resultado evalúa la función de filtración actual; no sustituye la monitorización clínica si existen signos físicos presentes."
            )
            
        return fallback_text
