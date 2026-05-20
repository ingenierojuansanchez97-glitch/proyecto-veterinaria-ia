# validator.py
# Validación de datos de entrada para la IA de predicción renal

class ValidationError(Exception):
    """Error personalizado de validación"""
    pass

def validate_inputs(vals: dict) -> dict:
    """
    Valida y normaliza los valores ingresados por el usuario.
    vals: dict con claves -> Edad, Peso, Creatinina, Urea, BUN, SDMA
    Retorna dict validado.
    """
    validated = {}

    # Rangos fisiológicos de referencia en animales (puedes ajustar según especie)
    ranges = {
        "Edad": (0, 30),            # años
        "Peso": (0.5, 120),         # kg
        "Creatinina": (0.2, 10),    # mg/dL
        "Urea": (5, 200),           # mg/dL
        "BUN": (2, 80),             # mg/dL
        "SDMA": (0, 100)            # µg/dL
    }

    for k, v in vals.items():
        if k not in ranges:
            raise ValidationError(f"❌ Campo inesperado: {k}")

        try:
            num = float(v)
        except ValueError:
            raise ValidationError(f"❌ {k} debe ser un número.")

        min_val, max_val = ranges[k]
        if not (min_val <= num <= max_val):
            raise ValidationError(f"❌ {k} fuera de rango ({num}). Esperado entre {min_val} y {max_val}.")

        validated[k] = num

    return validated
