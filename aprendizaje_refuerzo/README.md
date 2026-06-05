# Aprendizaje por Refuerzo — Q-Learning y DQN

Implementaciones de aprendizaje por refuerzo para el levitador vertical
(un sensor HC-SR04, un ventilador DC). Se contrastan dos enfoques:

- **Q-Learning tabular** ejecutado íntegramente en el ESP32.
- **Deep Q-Network (DQN)** entrenada en PC y desplegada al ESP32 para
  inferencia. La DQN aproxima la misma tarea que la Q-table usando estado
  continuo y velocidad, lo que mejora notablemente el desempeño.

Adicionalmente se conserva un ejemplo académico de DQN para navegación
con tres sensores, que demuestra la flexibilidad del marco DQN sobre
un problema distinto al de levitación.

## Archivos

### Q-Learning — ESP32 (MicroPython)
| Archivo | Descripción |
|---|---|
| `aprendizaje1.py` | Q-Learning tabular (11 estados × 6 acciones PWM). Corre directamente en el ESP32. |

### DQN levitador — PC (Python)
| Archivo | Descripción |
|---|---|
| `dqn_levitador.py` | Simulador `LevitadorEnv` (1D, sistema de primer orden hacia `p_eq(PWM)`), entrenamiento Q-Learning baseline, entrenamiento DQN con replay buffer y target network, comparación cuantitativa y gráfica. |
| `qtable_levitador.npy` | Tabla Q 11×6 aprendida por el baseline tabular (generada). |
| `dqn_levitador.pth` | Pesos de la DQN entrenada (generados). |
| `resultados/dqn_levitador.png` | Curva de aprendizaje y distribución final (generada). |
| `exportar_pesos_dqn_levitador.py` | Exporta `dqn_levitador.pth` a `pesos_dqn_levitador.py` en formato MicroPython. |
| `pesos_dqn_levitador.py` | Pesos DQN exportados, en listas planas listas para el ESP32 (generado). |

### DQN levitador — ESP32 (MicroPython)
| Archivo | Descripción |
|---|---|
| `dqn_levitador_esp32.py` | Inferencia DQN en el ESP32 con 1 sensor HC-SR04, derivada de velocidad filtrada, normalización y forward pass manual. |

### Ejemplo DQN con tres sensores (académico, no aplica al levitador)
| Archivo | Descripción |
|---|---|
| `dqn_tres_sensores.py` | Entorno simulado con tres sensores (izq./frente/der.) y cinco acciones (avanzar, retroceder, girar, mantener). Demuestra el marco DQN sobre un problema distinto. |
| `dqn_tres_sensores.pth` | Pesos del ejemplo (3 → 64 → 64 → 5). |
| `exportar_pesos_dqn.py` | Exporta esos pesos a `pesos_dqn.py`. |
| `dqn_esp32.py` | Inferencia en ESP32 con tres HC-SR04 (TRIG 5/4, 18/19, 21/22). |
| `pesos_dqn.py` | Pesos exportados del ejemplo. |

## Flujo de trabajo — DQN del levitador

```
1. Entrenar y validar en PC:
   python dqn_levitador.py
   (Genera qtable_levitador.npy, dqn_levitador.pth y la grafica)

2. Exportar pesos a MicroPython:
   python exportar_pesos_dqn_levitador.py
   (Genera pesos_dqn_levitador.py)

3. Copiar al ESP32 con MicroPico:
   - pesos_dqn_levitador.py
   - dqn_levitador_esp32.py

4. Ejecutar en el ESP32:
   import dqn_levitador_esp32
```

## Comparativa DQN vs Q-Learning (simulador)

Tras 1500 episodios de Q-Learning y 800 de DQN, evaluando las políticas
greedy en 200 episodios de test:

| Política | Error medio | Pasos dentro de ±1 cm |
|---|---|---|
| Q-Learning tabular (11×6) | ~0.9 cm | ~55 % |
| **DQN (2 → 24 → 24 → 6)** | **~0.2 cm** | **~99 %** |

La DQN supera al baseline tabular gracias a que su estado continuo
incluye la **velocidad** estimada y generaliza entre estados discretos
vecinos, evitando la oscilación típica del control bang-bang con tabla.

## Pines HC-SR04 (levitador, igual que el resto del proyecto)

| Sensor | TRIG | ECHO |
|---|---|---|
| Único | GPIO 27 | GPIO 26 |
| Ventilador (PWM 25 kHz) | — | GPIO 14 |

## Dependencias PC

```
pip install torch numpy matplotlib gymnasium
```
