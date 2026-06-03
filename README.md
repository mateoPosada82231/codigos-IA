# codigos-IA

Repositorio de algoritmos de control inteligente para ESP32: lógica difusa, redes neuronales y aprendizaje por refuerzo aplicados a sistemas embebidos.

## Estructura del proyecto

```
codigos-IA/
├── logica_difusa/          # Controladores Fuzzy PD + Integral
├── redes_neuronales/       # Red neuronal entrenada en PC, inferencia en ESP32
├── aprendizaje_refuerzo/   # Q-Learning y DQN (tres sensores)
└── graficas-informe-1/     # Gráficas y visualizaciones del informe
```

## Módulos

### [logica_difusa/](logica_difusa/README.md)
Controladores fuzzy con tres métodos de defuzzificación para levitación de pelota.
- `levitacion_fuzzy_centroide.py` — defuzzificación por Centroide
- `levitacion_fuzzy_bisector.py` — defuzzificación por Bisector
- `levitacion_fuzzy_mom.py` — defuzzificación por Mean of Maximum

### [redes_neuronales/](redes_neuronales/README.md)
Red neuronal FCLayer(3→16→12→8→1) entrenada con datos reales, exportada al ESP32.
- Tres variantes de activación: ReLU, Sigmoid, Tanh
- Scripts de entrenamiento y exportación de pesos
- Datos CSV de captura real del sistema

### [aprendizaje_refuerzo/](aprendizaje_refuerzo/README.md)
Q-Learning clásico en ESP32 y DQN para navegación con tres sensores ultrasónicos.
- `aprendizaje1.py` — Q-Learning directo en ESP32
- `dqn_tres_sensores.py` — entrenamiento DQN en PC (gymnasium + torch)
- `dqn_esp32.py` — inferencia DQN en ESP32 (sin dependencias externas)

## Hardware

- **Microcontrolador:** ESP32
- **Sensores:** HC-SR04 (ultrasónico)
- **Actuador:** Electroimán + driver PWM

## Dependencias PC

```bash
pip install numpy matplotlib scikit-fuzzy gymnasium torch
```
