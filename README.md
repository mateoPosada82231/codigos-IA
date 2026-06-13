# codigos-IA

Sistemas de Control Inteligente para Levitación en un Tubo de Acrílico Usando un Microcontrolador ESP32.

Este repositorio implementa y compara tres estrategias de control inteligente —**lógica difusa**, **redes neuronales** y **aprendizaje por refuerzo**— para estabilizar una pelota de icopor dentro de un tubo de acrílico vertical mediante un ventilador DC controlado por PWM. Incluye además un sistema de consulta RAG para interrogar el proyecto en lenguaje natural.

---

## Estructura del Repositorio

```
codigos-IA/
├── fuzzy/                   # Controladores Fuzzy PD+I (3 métodos de defuzzificación)
├── neural_networks/         # Red neuronal FCLayer(3→16→12→8→1) entrenada en PC
├── reinforcement_learning/  # Q-Learning tabular (ESP32) y DQN (PC + ESP32)
├── llm/                     # Sistema RAG con ChromaDB + Ollama
├── figures/                 # Gráficas e imágenes del informe
├── scripts/                 # Scripts de generación de gráficas
├── informe.tex              # Informe académico en LaTeX (IEEEtran)
└── README.md
```

---

## Hardware

| Componente         | Descripción                          | Pin ESP32      |
|--------------------|--------------------------------------|----------------|
| Microcontrolador   | ESP32 (MicroPython)                  | —              |
| Sensor ultrasónico | HC-SR04                              | TRIG: GPIO 27  |
|                    |                                      | ECHO: GPIO 26  |
| Actuador           | Ventilador DC sin escobillas         | GPIO 14 (PWM)  |
| Estructura         | Tubo de acrílico 40 cm               | —              |
| Pelota             | Icopor 0.5 g (no perfectamente esférica) | —          |

---

## Dependencias PC

```bash
pip install numpy matplotlib scikit-fuzzy gymnasium torch pandas
```

Para el módulo RAG se requieren adicionalmente:

```bash
pip install ollama chromadb langchain langchain-ollama langchain-chroma langchain-core
ollama pull llama3.2
ollama pull nomic-embed-text
```

---

## Guía Rápida

### 1. Probar un controlador en el ESP32

Cada módulo tiene un **orquestador** (`maestro.py`) que automatiza el ciclo completo:

```bash
cd fuzzy
python maestro.py        # Menú: elegir método, subir a ESP32, monitorear
```

```bash
cd neural_networks
python maestro.py        # Menú: elegir activación, entrenar, exportar, desplegar
```

```bash
cd reinforcement_learning
python maestro.py        # Menú: Q-Learning o DQN, subir, capturar datos
```

### 2. Entrenar desde cero

```bash
cd neural_networks
python train.py          # Entrena las 3 variantes con datos CSV
```

```bash
cd reinforcement_learning
python dqn_train.py      # Entrena DQN en simulador 1D
```

### 3. Consultar el proyecto con IA

```bash
cd llm
python rag.py "¿Cuál es el error promedio del controlador Centroide?"
```

---

## Trabajo por Módulos

| Módulo | Descripción | Documentación |
|--------|-------------|---------------|
| [`fuzzy/`](fuzzy/README.md) | 3 controladores difusos (Centroide, Bisector, MOM) | FAM 9×7, PD+I, defuzzificación |
| [`neural_networks/`](neural_networks/README.md) | Red neuronal entrenada en PC, inferencia en ESP32 | 3 activaciones (ReLU, Sigmoid, Tanh) |
| [`reinforcement_learning/`](reinforcement_learning/README.md) | Q-Learning en ESP32 y DQN híbrida PC/ESP32 | Tabla 11×6 vs red 2→24→24→6 |
| [`llm/`](llm/README.md) | Sistema RAG con ChromaDB y Ollama | Consultas en lenguaje natural |

---

## Resultados Experimentales

| Método | Error Promedio | Desv. Est. |
|--------|---------------|------------|
| DQN | 1.67 mm | 1.23 mm |
| Fuzzy Centroide | 1.89 mm | 1.45 mm |
| Q-Learning | 1.92 mm | 1.58 mm |
| Fuzzy MOM | 2.12 mm | 1.62 mm |
| Fuzzy Bisector | 2.34 mm | 1.87 mm |
| NN Tanh | 2.56 mm | 2.15 mm |
| NN Sigmoid | 2.78 mm | 2.34 mm |
| NN ReLU | 3.45 mm | 2.91 mm |

---

## Informe Académico

El archivo [`informe.tex`](informe.tex) contiene el documento completo en formato IEEEtran (español) con descripción detallada del proyecto, análisis de cada técnica, resultados y conclusiones. Las imágenes referenciadas están en [`figures/`](figures/).

---

## Licencia

Código publicado con fines académicos. Libre para uso educativo y experimental.
