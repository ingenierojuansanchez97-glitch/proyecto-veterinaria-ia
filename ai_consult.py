# ai_consult.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None

def consult_ai(patient_data: dict, prediction: str, probabilities: dict) -> str:
    """
    Consulta a la IA con los datos del paciente y la predicción.
    """
    prompt = f"""
    Soy una IA veterinaria especializada en nefrología.
    Evalúa estos datos del paciente y comenta si el riesgo predicho es coherente
    o si hay observaciones adicionales que el médico veterinario debería tener en cuenta.

    Datos del paciente:
    {patient_data}

    Predicción del modelo: {prediction}
    Probabilidades: {probabilities}

    Por favor da una explicación clara en lenguaje veterinario.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # puedes cambiar por el modelo que tengas disponible
        messages=[
            {"role": "system", "content": "Eres un asistente experto en nefrología veterinaria."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
    )

    return response.choices[0].message.content.strip()
