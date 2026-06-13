# Aprendizaje por Refuerzo — Q-Learning y DQN

Implementaciones de aprendizaje por refuerzo para el levitador vertical (un sensor HC-SR04, un ventilador DC). Se contrastan dos enfoques:

- **Q-Learning tabular** (11 estados × 6 acciones) ejecutado íntegramente en el ESP32.
- **Deep Q-Network (DQN)** (2→24→24→6) entrenada en PC con PyTorch sobre un simulador 1D y desplegada al ESP32 para inferencia a 20 Hz.

---

## Archivos del Módulo

### Q-Learning — ESP32
| Archivo | Descripción |
|---------|-------------|
| `qlearning_esp32.py` | Q-Learning tabular con ε-greedy, recompensa penalizante y persistencia a `qtable.json` |
| `qtable.json` | Tabla Q aprendida (generada por `qlearning_esp32.py`) |

### DQN — PC (Entrenamiento)
| Archivo | Descripción |
|---------|-------------|
| `dqn_train.py` | Simulador 1D `LevitadorEnv`, entrenamiento DQN + Q-Learning baseline, exporta modelo |
| `export_dqn_weights.py` | Convierte `dqn_model.pth` → `dqn_weights.py` (MicroPython) |
| `dqn_weights.py` | Pesos DQN en listas planas (generado, ~3.1 KB) |

### DQN — ESP32 (Inferencia)
| Archivo | Descripción |
|---------|-------------|
| `dqn_esp32.py` | Forward pass manual, normalización, lectura HC-SR04, lazo 20 Hz |

### Orquestador
| Archivo | Descripción |
|---------|-------------|
| `maestro.py` | Menú interactivo: subir controlador, monitorear serie, re-entrenar |

### Datos Experimentales
| Archivo | Descripción |
|---------|-------------|
| `data_qlearning.csv` | Respuesta del sistema con Q-Learning |
| `data_dqn.csv` | Respuesta del sistema con DQN |

---

## Hardware Requerido

- ESP32 con MicroPython
- Sensor HC-SR04 (TRIG: GPIO 27, ECHO: GPIO 26)
- Ventilador DC (PWM: GPIO 14, 25 kHz)
- Tubo de acrílico vertical
- Pelota de icopor (~0.5 g)

### Dependencias PC

```bash
pip install torch numpy matplotlib gymnasium
```

---

## Uso Rápido

### Opción 1: Orquestador (recomendado)

```bash
cd reinforcement_learning
python maestro.py
```

Menú interactivo que permite elegir Q-Learning o DQN, subir el firmware al ESP32, monitorear la salida serie, descargar datos y re-entrenar.

### Opción 2: Q-Learning directo en ESP32

1. Copiar `qlearning_esp32.py` al ESP32.
2. Ejecutar en el REPL:

```python
import qlearning_esp32
```

### Opción 3: DQN — entrenar y desplegar

```bash
# 1. Entrenar en PC
python dqn_train.py
# Genera: dqn_model.pth, qtable.json, figures/dqn_training.png

# 2. Exportar pesos a MicroPython
python export_dqn_weights.py
# Genera: dqn_weights.py

# 3. Copiar al ESP32: dqn_weights.py + dqn_esp32.py

# 4. En el REPL del ESP32:
import dqn_esp32
```

---

## Q-Learning Tabular

### Estado
Discretización del espacio de posiciones [10, 20] cm en 11 estados:

| Estado | Rango (cm) |
|--------|-----------|
| 0 | < 10.0 |
| 1 | [10.0, 10.9) |
| 2 | [10.9, 11.8) |
| ... | ... |
| 10 | > 19.1 |

### Acciones
6 valores de PWM: [200, 280, 360, 440, 520, 600]

### Algoritmo
- Política ε-greedy con ε creciente: 0.20 → 1.0 (+0.20 cada 100 pasos)
- Tasa de aprendizaje α = 0.10
- Factor de descuento γ = 0.90
- Recompensa: R = -\|dist - setpoint\|
- 800 pasos por sesión, guarda Q-table cada 100 pasos a `qtable.json`
- Persistencia: la tabla continúa aprendiendo entre reinicios

---

## Deep Q-Network

### Simulador 1D (`LevitadorEnv`)

Modelo de primer orden:
```
p_eq(PWM) = 25 − 20 × (PWM − 200) / 400
τ = 0.35 s
```

Ecuación de actualización:
```
p(t+1) = p(t) + (p_eq(PWM) − p(t)) / τ + ruido_σ=0.15cm
```

Recompensa:
```
r = −0.3×|e| + { +7 si |e| < 0.3cm, +3 si |e| < 1.0cm, 0 en otro caso }
```

Episodio: termina al salir del tubo o tras 150 pasos (7.5 s).

### Arquitectura de la Red

```
Estado [p, v] → FC 24 (ReLU) → FC 24 (ReLU) → FC 6 (lineal) → Q-values
```

| Capa | Dimensiones | Parámetros |
|------|------------|------------|
| Entrada | 2 (posición normalizada, velocidad) | — |
| Oculta 1 | 2×24 + 24 bias | 72 |
| Oculta 2 | 24×24 + 24 bias | 600 |
| Salida | 24×6 + 6 bias | 150 |
| **Total** | | **786** (~3.1 KB) |

### Hiperparámetros de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Episodios | 800 |
| Pasos por episodio | 150 |
| Replay buffer | 2000 transiciones |
| Target network sync | cada 10 episodios |
| ε inicial / final | 1.0 / 0.01 (decaimiento 0.995) |
| Optimizador | Adam, lr = 1e-3 |
| Pérdida | MSE sobre Bellman, γ = 0.95 |
| Seed de experiencia | Transiciones expertas desde controlador fuzzy |

### Inferencia en ESP32

```
HC-SR04 → mediana 3 → EMA → [p, v] normalizado → DQN forward → argmax → PWM
```

El forward pass manual implementa producto matriz-vector + ReLU en punto flotante de 32 bits con buffers pre-asignados.

---

## Comparativa (Simulador)

| Política | Error Medio | Pasos dentro de ±1 cm |
|----------|------------|----------------------|
| Q-Learning tabular (11×6) | ~0.9 cm | ~55 % |
| **DQN (2→24→24→6)** | **~0.2 cm** | **~99 %** |

La DQN supera al baseline tabular porque:
- Usa estado continuo (posición + velocidad)
- Generaliza entre estados discretos vecinos
- Evita el control bang-bang típico de la tabla

---

## Pines HC-SR04

| Señal | GPIO |
|-------|------|
| TRIG | GPIO 27 |
| ECHO | GPIO 26 |
| PWM Ventilador | GPIO 14 (25 kHz) |
