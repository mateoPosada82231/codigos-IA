# Redes Neuronales — Levitación de Pelota

Red neuronal entrenada en PC con datos reales del sistema de levitación, exportada al ESP32 para inferencia en tiempo real.

## Archivos

### ESP32 (MicroPython)
| Archivo | Descripción |
|---|---|
| `levitacion_red_neuronal_relu.py` | Controlador con red neuronal activación **ReLU** |
| `levitacion_red_neuronal_sigmoid.py` | Controlador con red neuronal activación **Sigmoid** |
| `levitacion_red_neuronal_tanh.py` | Controlador con red neuronal activación **Tanh** |
| `pesos.txt` | Pesos exportados listos para cargar en el ESP32 |

### PC (Python)
| Archivo | Descripción |
|---|---|
| `entrenar_red_levitador.py` | Entrena la red neuronal con los datos CSV |
| `exportar_pesos_esp32.py` | Exporta pesos del modelo `.pkl` a código MicroPython |

### Datos de entrenamiento
| Archivo | Descripción |
|---|---|
| `datos_levitacion_10cm.csv` | Datos capturados a 10 cm de objetivo |
| `datos_levitacion_15cm.csv` | Datos capturados a 15 cm de objetivo |
| `datos_levitacion_20cm.csv` | Datos capturados a 20 cm de objetivo |

### Modelos guardados
| Archivo | Descripción |
|---|---|
| `pesos_levitador_relu.pkl` | Modelo entrenado con ReLU |
| `pesos_levitador_sigmoid.pkl` | Modelo entrenado con Sigmoid |
| `pesos_levitador_tanh.pkl` | Modelo entrenado con Tanh |

## Flujo de trabajo

```
1. Capturar datos → datos_levitacion_Xcm.csv
2. Entrenar en PC → python entrenar_red_levitador.py
3. Exportar pesos → python exportar_pesos_esp32.py pesos_levitador_sigmoid.pkl
4. Copiar pesos.txt al ESP32 con MicroPico
5. Ejecutar levitacion_red_neuronal_sigmoid.py en el ESP32
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
