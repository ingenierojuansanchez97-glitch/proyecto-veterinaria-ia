import os
from google import genai
from google.genai import types

def consult_ai(patient_data: dict, pred_label: str, prob_dict: dict) -> str:
    """
    Consulta a la IA de Gemini (Gratis desde Google AI Studio) para obtener
    un análisis clínico veterinario personalizado.
    """
    # 1. Obtener la API Key desde las variables de entorno
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró la variable de entorno GEMINI_API_KEY")

    # 2. Inicializar el cliente de Google GenAI
    client = genai.Client(api_key=api_key)

    # 3. Diseñar el contexto y los datos del paciente para el prompt
    prompt = f"""
    Eres un experto en nefrología veterinaria. Analiza el siguiente caso clínico:
    
    INFORMACIÓN DEL PACIENTE:
    - Nombre: {patient_data.get('Nombre', 'Paciente')}
    - Especie: {patient_data.get('Especie', 'No especificada')}
    - Raza: {patient_data.get('Raza', 'No especificada')}
    - Edad: {patient_data.get('Edad')} años
    - Peso: {patient_data.get('Peso')} kg
    
    BIOMARCADORES RENALES:
    - Creatinina: {patient_data.get('Creatinina')} mg/dL
    - Urea: {patient_data.get('Urea')} mg/dL
    - BUN: {patient_data.get('BUN')} mg/dL
    - SDMA: {patient_data.get('SDMA')} ug/dL
    
    RESULTADO DEL ALGORITMO COMBINADO:
    - Clasificación de Riesgo: {pred_label}
    - Probabilidades estimadas: Bajo ({prob_dict.get('Bajo', 0)*100:.1f}%), Medio ({prob_dict.get('Medio', 0)*100:.1f}%), Alto ({prob_dict.get('Alto', 0)*100:.1f}%)
    
    INSTRUCCIONES DE RESPUESTA:
    Genera un informe analítico, profesional y directo para el médico veterinario a cargo. 
    1. Interpreta qué significan estos biomarcadores específicos para su especie (ej. la importancia del SDMA o la relación BUN/Creatinina).
    2. Explica brevemente la coherencia del nivel de riesgo '{pred_label}'.
    3. Proporciona recomendaciones de manejo clínico, diagnóstico diferencial (ej. descartar injuria renal aguda vs crónica) y pautas terapéuticas o nutricionales iniciales.
    
    Mantén un tono clínico formal, estructurado con viñetas y limpio.
    """

    # 4. Realizar la petición usando el modelo rápido y optimizado (gemini-1.5-flash)
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3, # Temperatura baja para un comportamiento más clínico y preciso
        )
    )

    return response.text
