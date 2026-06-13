# Módulo LLM — Sistema RAG del Proyecto de Levitación

Sistema de Retrieval-Augmented Generation (RAG) que permite realizar consultas en lenguaje natural sobre el proyecto. Indexa el informe LaTeX y el código fuente en una base de datos vectorial ChromaDB y responde preguntas contextualizadas usando Ollama.

---

## Requisitos

### 1. Instalar Ollama

Descargar e instalar desde [ollama.com](https://ollama.com). Asegurarse de que el servicio esté corriendo.

### 2. Descargar modelos locales

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

---

## Estructura del Módulo

```
llm/
├── rag.py              # Sistema RAG principal
├── test_rag.py         # Pruebas del vectorstore
├── requirements.txt    # Dependencias Python
├── .gitignore          # Ignora vectordb/ y __pycache__/
└── README.md
```

La base de datos vectorial se crea automáticamente en `vectordb/` en el primer uso.

---

## Uso

### Consulta interactiva

```bash
python rag.py
```

Inicia un REPL donde se pueden hacer preguntas consecutivas. Escribir `salir` para terminar.

### Consulta directa

```bash
python rag.py "¿Cuál es el error promedio del controlador Centroide?"
```

### Reconstruir el índice

Si el informe o el código cambian, forzar la re-indexación:

```bash
python rag.py --rebuild
```

---

## ¿Cómo funciona?

```
                ┌──────────────┐
                │  informe.tex │
                │  *.py        │
                └──────┬───────┘
                       ▼
              ┌─────────────────┐
              │  Chunking       │
              │  (secciones,    │
              │   funciones)    │
              └───────┬─────────┘
                      ▼
              ┌─────────────────┐
              │  nomic-embed-text│
              │  → embeddings   │
              └───────┬─────────┘
                      ▼
              ┌─────────────────┐
              │  ChromaDB       │
              │  (vectorstore)  │
              └───────┬─────────┘
                      ▼
   Pregunta ──→ embedding ──→ búsqueda coseno ──→ fragmentos relevantes
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  llama3.2       │
                                              │  (generación)   │
                                              └───────┬─────────┘
                                                      ▼
                                              Respuesta en lenguaje natural
```

### Etapas

1. **Indexación**: Extrae texto plano de cada sección del `informe.tex` y analiza los archivos `.py` identificando funciones y clases. Cada fragmento se almacena en ChromaDB con su embedding (768 dimensiones) y metadatos de origen.

2. **Búsqueda**: La pregunta se convierte en embedding. Se recuperan los $k$ fragmentos más similares por similitud coseno.

3. **Generación**: Los fragmentos se concatenan como contexto y se envían a `llama3.2` (via Ollama), que genera una respuesta fundamentada en la documentación real del proyecto.

---

## Preguntas de Ejemplo

- ¿Qué controladores se implementaron en el proyecto?
- ¿Cuál es la diferencia entre Q-Learning y DQN?
- ¿Qué función de activación usó la red neuronal?
- Describe el hardware del proyecto.
- ¿Cuál algoritmo tiene la mejor precisión?
- ¿Cómo funciona el filtro EMA en el lazo de control?
- ¿Qué pines del ESP32 se usan para el HC-SR04?
- ¿Cuántas reglas tiene la matriz FAM?
