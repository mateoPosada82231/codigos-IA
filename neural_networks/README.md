# Redes Neuronales — Controlador por Red Neuronal Feedforward

Red neuronal completamente conectada FCLayer(3→16→12→8→1) entrenada en PC con datos reales del sistema de levitación y desplegada en ESP32 para inferencia en tiempo real. Soporta tres funciones de activación: **ReLU**, **Sigmoid** y **Tanh**.

---

## Archivos del Módulo

### ESP32 (MicroPython) — Controladores
| Archivo | Descripción |
|---------|-------------|
| `controller_relu.py` | Activación oculta **ReLU** (no acotada, eficiente) |
| `controller_sigmoid.py` | Activación oculta **Sigmoid** (acotada [0,1]) |
| `controller_tanh.py` | Activación oculta **Tanh** (acotada [-1,1], centrada en 0) |

### PC (Python) — Entrenamiento y Exportación
| Archivo | Descripción |
|---------|-------------|
| `train.py` | Entrena la red con datos CSV, genera `weights.json` y gráficas |
| `export_weights.py` | Exporta pesos desde `weights.json` hacia el controlador MicroPython |
| `maestro.py` | Orquestador interactivo: sube, captura, re-entrena en ciclo |

### Datos y Modelos
| Archivo | Descripción |
|---------|-------------|
| `weights.json` | Pesos pre-entrenados de las 3 variantes (JSON) |
| `data_relu.csv` | Datos experimentales — controlador ReLU |
| `data_sigmoid.csv` | Datos experimentales — controlador Sigmoid |
| `data_tanh.csv` | Datos experimentales — controlador Tanh |

---

## Hardware Requerido

- ESP32 con MicroPython
- Sensor ultrasónico HC-SR04 (TRIG: GPIO 27, ECHO: GPIO 26)
- Ventilador DC (PWM: GPIO 14, 25 kHz)
- Tubo de acrílico vertical
- Pelota de icopor (~0.5 g)

### Dependencias PC

```bash
pip install numpy matplotlib scikit-fuzzy
```

---

## Uso Rápido

### Opción 1: Orquestador (recomendado)

```bash
cd neural_networks
python maestro.py
```

El menú guía todo el ciclo: elegir activación → subir a ESP32 → monitorear → descargar datos → re-entrenar.

### Opción 2: Entrenar desde cero

```bash
python train.py
```

Esto entrena las 3 variantes con los datos CSV, genera `weights.json` y produce gráficas de pérdida y comparación contra fuzzy.

### Opción 3: Exportar pesos manualmente

```bash
python export_weights.py weights.json sigmoid
```

Esto inyecta los pesos (W1..W4, B1..B4) y parámetros de normalización dentro del archivo `controller_sigmoid.py`.

### Opción 4: Despliegue manual

1. Copiar el controlador deseado (`controller_*.py`) al ESP32.
2. Ejecutar en el REPL:

```python
import controller_sigmoid  # o controller_relu / controller_tanh
```

---

## Arquitectura de la Red

```
Entrada (3) → FC 16 → Activación → FC 12 → Activación → FC 8 → Activación → FC 1 (lineal)
```

| Capa | Neuronas | Pesos | Activación |
|------|----------|-------|------------|
| Entrada | 3 | — | error, derivada, integral |
| Oculta 1 | 16 | 3×16 + 16 bias | ReLU / Sigmoid / Tanh |
| Oculta 2 | 12 | 16×12 + 12 bias | ReLU / Sigmoid / Tanh |
| Oculta 3 | 8 | 12×8 + 8 bias | ReLU / Sigmoid / Tanh |
| Salida | 1 | 8×1 + 1 bias | Lineal (ΔPWM) |

Total: **391 parámetros entrenables** (~1.5 KB en punto flotante).

---

## Flujo de Trabajo Completo

```
                        ┌─────────────┐
                        │  Datos CSV  │
                        │  (ESP32)    │
                        └──────┬──────┘
                               ▼
                     ┌─────────────────┐
                     │   train.py      │
                     │  (entrenamiento)│
                     └───────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │  weights.json   │
                    │  (pesos + stats)│
                    └───────┬─────────┘
                            ▼
                   ┌──────────────────┐
                   │ export_weights.py│
                   └───────┬──────────┘
                           ▼
              ┌────────────────────────┐
              │ controller_*.py        │
              │ (código + pesos incrustados) │
              └───────────┬────────────┘
                          ▼
                    ┌──────────┐
                    │  ESP32   │
                    │ (inferencia 20 Hz) │
                    └──────────┘
```

---

## Forward Pass en el ESP32

El ESP32 ejecuta inferencia con aritmética de punto flotante de 32 bits:

```
x_norm = (x - X_MEAN) / X_STD
h1 = ReLU(W1 @ x_norm + B1)
h2 = ReLU(W2 @ h1 + B2)
h3 = ReLU(W3 @ h2 + B3)
y = W4 @ h3 + B4
Δpwm = y * Y_STD + Y_MEAN
```

Se usan buffers pre-asignados (`array.array('f')`) para evitar fragmentación del heap y garantizar ejecución dentro del período de 50 ms.

---

## Procesamiento de Señal (idéntico en las 3 variantes)

- Mediana deslizante de 7 lecturas del HC-SR04
- Rechazo de outliers (> 5 cm)
- Filtro EMA (α = 0.40) sobre la derivada
- Acumulador integral con decaimiento (0.998) y anti-windup
- PWM saturado a [170, 900]

---

## Datos CSV

Cada archivo `data_*.csv` contiene 301 filas con:

| Columna | Descripción |
|---------|-------------|
| `tiempo` | Tiempo transcurrido (ms) |
| `distancia` | Distancia medida (cm) |
| `setpoint` | Punto de consigna (cm) |
| `error` | Error de posición (cm) |
| `derivada` | Derivada filtrada |
| `integral` | Acumulador integral |
| `delta_pwm` | ΔPWM calculado por la red |
| `pwm` | PWM final aplicado |
