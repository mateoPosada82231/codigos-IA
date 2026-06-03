# Aprendizaje por Refuerzo — Q-Learning y DQN

Implementaciones de aprendizaje por refuerzo para control autónomo: Q-Learning clásico para levitación en ESP32 y Deep Q-Network (DQN) para navegación con tres sensores.

## Archivos

### Q-Learning — ESP32 (MicroPython)
| Archivo | Descripción |
|---|---|
| `aprendizaje1.py` | Q-Learning para levitación de pelota, corre directamente en el ESP32 |

### DQN — PC (Python)
| Archivo | Descripción |
|---|---|
| `dqn_tres_sensores.py` | Entorno y entrenamiento DQN con tres sensores de distancia (PC) |
| `exportar_pesos_dqn.py` | Exporta pesos del modelo `.pth` a `pesos_dqn.py` para el ESP32 |
| `dqn_tres_sensores.pth` | Modelo DQN entrenado (generado tras el entrenamiento) |

### DQN — ESP32 (MicroPython)
| Archivo | Descripción |
|---|---|
| `dqn_esp32.py` | Inferencia DQN en el ESP32 con tres sensores HC-SR04 |
| `pesos_dqn.py` | Pesos de la red DQN exportados (generado por `exportar_pesos_dqn.py`) |

## Flujo de trabajo DQN

```
1. Entrenar y exportar en PC:
   python exportar_pesos_dqn.py
   (Si no existe el .pth, entrena automáticamente primero)

2. Copiar al ESP32 con MicroPico:
   - pesos_dqn.py
   - dqn_esp32.py

3. Ejecutar en el ESP32:
   import dqn_esp32
```

## Q-Learning (aprendizaje1.py)

- **Estados:** 11 posiciones discretas (10–20 cm)
- **Acciones:** 6 valores de PWM (200–600)
- **Algoritmo:** Q-Learning con ε-greedy
- **Persistencia:** tabla Q guardada en `qtable.json`
- **Corre completamente en el ESP32** (sin PC)

## DQN (dqn_tres_sensores.py + dqn_esp32.py)

- **Entorno:** 3 sensores ultrasónicos (izquierdo, frente, derecho)
- **Acciones:** 5 (avanzar, retroceder, girar izquierda, girar derecha, mantener)
- **Red:** 3 → 64 → 64 → 5 (PyTorch en PC, inferencia manual en ESP32)
- **Entrenamiento:** PC con gymnasium + torch
- **Inferencia:** ESP32 con aritmética pura de listas (sin librerías externas)

## Pines HC-SR04 (dqn_esp32.py)

| Sensor | TRIG | ECHO |
|---|---|---|
| Izquierdo | GPIO 5 | GPIO 4 |
| Frontal | GPIO 18 | GPIO 19 |
| Derecho | GPIO 21 | GPIO 22 |

Ajustar según el cableado real en `dqn_esp32.py`.

## Dependencias PC

```
pip install gymnasium torch numpy
```
