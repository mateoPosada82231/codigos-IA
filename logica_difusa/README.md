# Lógica Difusa — Levitación de Pelota

Controladores basados en lógica difusa (Fuzzy PD + Integral) para el sistema de levitación magnética con ESP32. Cada archivo usa un método de defuzzificación diferente.

## Archivos

| Archivo | Descripción |
|---|---|
| `levitacion_fuzzy_bisector.py` | Controlador fuzzy con defuzzificación **Bisector** |
| `levitacion_fuzzy_centroide.py` | Controlador fuzzy con defuzzificación **Centroide** |
| `levitacion_fuzzy_mom.py` | Controlador fuzzy con defuzzificación **Mean of Maximum (MOM)** |

## Hardware requerido

- ESP32
- Sensor ultrasónico HC-SR04
- Electroimán + driver de potencia (PWM)
- Pelota de levitación

## Uso

Copiar el archivo deseado al ESP32 con **MicroPico** y ejecutarlo:

```python
# En el REPL de MicroPython
import levitacion_fuzzy_centroide  # o bisector / mom
```

## Lógica de control

- **Entradas:** error de posición (cm), derivada del error
- **Reglas:** 9 niveles de error → 9 niveles de corrección de PWM
- **Salida:** señal PWM al electroimán
- **Anti-windup** en el término integral
- **Rechazo de outliers** en la lectura del sensor

## Comparación de métodos

| Método | Característica |
|---|---|
| Centroide | Más suave, promedia toda la zona activa |
| Bisector | Divide el área en dos mitades iguales |
| MOM | Más agresivo, usa el máximo de activación |
