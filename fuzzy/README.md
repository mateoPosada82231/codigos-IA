# Lógica Difusa — Controlador Fuzzy PD+I

Controladores basados en lógica difusa para la levitación de una pelota en un tubo de acrílico usando ESP32. Implementa un controlador PD+I con matriz FAM de 9×7 reglas y tres métodos de defuzzificación.

---

## Archivos del Módulo

| Archivo | Descripción |
|---------|-------------|
| `controller_centroid.py` | Defuzzificación por **Centroide** — salida suave y continua |
| `controller_bisector.py` | Defuzzificación por **Bisector** — respuesta intermedia |
| `controller_mom.py` | Defuzzificación por **Mean of Maximum (MOM)** — respuesta agresiva |
| `maestro.py` | Orquestador interactivo PC ↔ ESP32 |
| `data_centroid.csv` | Datos experimentales del controlador Centroide |
| `data_bisector.csv` | Datos experimentales del controlador Bisector |
| `data_mom.csv` | Datos experimentales del controlador MOM |

---

## Hardware Requerido

- ESP32 con MicroPython
- Sensor ultrasónico HC-SR04 (TRIG: GPIO 27, ECHO: GPIO 26)
- Ventilador DC (PWM: GPIO 14, 25 kHz)
- Tubo de acrílico vertical
- Pelota de icopor (~0.5 g)

---

## Uso Rápido

### Opción 1: Orquestador (recomendado)

```bash
cd fuzzy
python maestro.py
```

El menú interactivo permite:
1. Elegir método de defuzzificación (Centroide, Bisector, MOM)
2. Fijar setpoint
3. Subir automáticamente el firmware al ESP32
4. Abrir terminal serie para monitoreo
5. Descargar datos CSV al finalizar

### Opción 2: Manual (subir con MicroPico)

1. Copiar el controlador deseado al ESP32.
2. Ejecutar en el REPL de MicroPython:

```python
import controller_centroid  # o controller_bisector / controller_mom
```

---

## Lógica de Control

### Entradas
- **Error de posición** (`e = setpoint - distancia`): 9 niveles lingüísticos (NV a PV)
- **Derivada del error**: 7 niveles lingüísticos

### Procesamiento de Señal
- Mediana deslizante de 7 lecturas del HC-SR04
- Rechazo de valores atípicos (> 5 cm de salto)
- Filtro EMA (α = 0.40) sobre la derivada
- Acumulador integral con decaimiento (0.998) y anti-windup

### Matriz FAM
Matriz asimétrica 9×7 con incrementos de PWM que reflejan la dinámica del sistema:
- Incrementos positivos grandes cuando la pelota cae (gravedad asistida)
- Decrementos negativos pequeños cuando la pelota sube

### Salida
- PWM saturado al rango [170, 900]
- Frecuencia de control: ~20 Hz

---

## Métodos de Defuzzificación

| Método | Comportamiento | Mejor Para |
|--------|---------------|------------|
| **Centroide** | Promedio ponderado de todos los singletons activos. Salida suave y continua. | Error promedio mínimo (1.89 mm) |
| **Bisector** | Valor que divide el área total de activación en dos mitades iguales. | Respuesta intermedia |
| **MOM** | Promedio de los singletons con máxima activación. Cambios bruscos de PWM. | Respuesta rápida, menos estable |

---

## Arquitectura del Controlador (ESP32)

```
HC-SR04 → Mediana 7 → Error → Fuzzy PD+I → PWM → Ventilador
                        ↓
                   Derivada (EMA)
                        ↓
                   Integral (decaimiento + anti-windup)
```

---

## Datos CSV

Cada archivo `data_*.csv` contiene 301 filas con las columnas:

| Columna | Descripción |
|---------|-------------|
| `tiempo` | Tiempo transcurrido (ms) |
| `distancia` | Distancia medida (cm) |
| `setpoint` | Punto de consigna (cm) |
| `error` | Error de posición (cm) |
| `derivada` | Derivada filtrada del error |
| `integral` | Acumulador integral |
| `delta_pwm` | Incremento calculado |
| `pwm` | PWM aplicado al ventilador |
