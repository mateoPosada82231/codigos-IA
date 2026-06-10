# Redes Neuronales — Levitación de Pelota

Red neuronal entrenada en PC con datos reales del sistema de levitación, exportada al ESP32 para inferencia en tiempo real.

## Archivos

### ESP32 (MicroPython)
| Archivo | Descripción |
|---|---|
| `levitacion_red_neuronal_relu.py` | Controlador con red neuronal activación **ReLU** |
| `levitacion_red_neuronal_sigmoid.py` | Controlador con red neuronal activación **Sigmoid** |
| `levitacion_red_neuronal_tanh.py` | Controlador con red neuronal activación **Tanh** |

### PC (Python)
| Archivo | Descripción |
|---|---|
| `entrenar_red_levitador.py` | Entrena la red neuronal con los datos CSV y guarda `pesos_red_levitador.json` |
| `exportar_pesos_esp32.py` | Exporta pesos de `pesos_red_levitador.json` (o `.pkl`) al controlador MicroPython |
| `control_maestro.py` | Orquestador interactivo: sube controlador al ESP32, captura datos y re-entrena en ciclo |

### Modelos guardados
| Archivo | Descripción |
|---|---|
| `pesos_red_levitador.json` | Pesos de las tres variantes (relu/sigmoid/tanh) en formato JSON |

### Datos capturados (ESP32)
| Archivo | Descripción |
|---|---|
| `datos_esp32_relu.csv` | Datos de respuesta del controlador ReLU en el sistema real |
| `datos_esp32_sigmoid.csv` | Datos de respuesta del controlador Sigmoid en el sistema real |
| `datos_esp32_tanh.csv` | Datos de respuesta del controlador Tanh en el sistema real |

### Gráficas generadas
| Archivo | Descripción |
|---|---|
| `entrenamiento_levitador_relu.png` | Curva de entrenamiento — activación ReLU |
| `entrenamiento_levitador_sigmoid.png` | Curva de entrenamiento — activación Sigmoid |
| `entrenamiento_levitador_tanh.png` | Curva de entrenamiento — activación Tanh |
| `comparacion_fuzzy_vs_rn_relu.png` | Comparación Fuzzy vs Red Neuronal ReLU |
| `comparacion_fuzzy_vs_rn_sigmoid.png` | Comparación Fuzzy vs Red Neuronal Sigmoid |
| `comparacion_fuzzy_vs_rn_tanh.png` | Comparación Fuzzy vs Red Neuronal Tanh |

## Flujo de trabajo

```
1. Entrenar en PC con datos capturados:
   python entrenar_red_levitador.py
   (Genera pesos_red_levitador.json y gráficas de entrenamiento)

2. Exportar pesos al controlador ESP32:
   python exportar_pesos_esp32.py pesos_red_levitador.json sigmoid
   (Inyecta pesos en levitacion_red_neuronal_sigmoid.py)

3. Copiar controlador al ESP32 con MicroPico:
   - levitacion_red_neuronal_sigmoid.py  (o relu / tanh)

4. Ejecutar en el ESP32:
   import levitacion_red_neuronal_sigmoid

5. (Opcional) Usar el orquestador para todo el ciclo:
   python control_maestro.py
```

## Arquitectura de la red

- **Capas:** FCLayer(3 → 16 → 12 → 8 → 1)
- **Entradas:** distancia actual, error, derivada del error
- **Salida:** PWM al electroimán
- **Entrenamiento:** datos reales del sistema físico

## Hardware requerido

- ESP32
- Sensor ultrasónico HC-SR04
- Electroimán + driver de potencia (PWM)
