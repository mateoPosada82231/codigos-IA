# Módulo LLM — RAG del Proyecto de Levitación

Sistema de consulta basado en RAG (Retrieval-Augmented Generation) para que el profesor haga preguntas sobre el proyecto usando el informe en LaTeX.

## Requisitos

1. **Ollama** instalado y corriendo: https://ollama.com
2. Modelos locales descargados:
   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Dependencias Python:
   ```
   pip install -r requirements.txt
   ```

## Uso

```bash
# Modo interactivo
python rag_levitador.py

# Consulta directa
python rag_levitador.py "¿cuál es el algoritmo que mejor funciona?"

# Reconstruir vectorstore si el informe cambió
python rag_levitador.py --rebuild
```

## Preguntas de ejemplo

- ¿Qué controladores se implementaron?
- ¿Cuál es la diferencia entre Q-learning y DQN?
- ¿Qué función de activación se usó en la red neuronal?
- Describe el hardware del proyecto.
- ¿Cuál algoritmo tiene mejor precisión?

## Estructura

```
LLM/
├── rag_levitador.py    # Sistema RAG (procesa .tex → ChromaDB → consultas)
├── requirements.txt    # Dependencias Python
├── .gitignore
└── README.md
```

El vectorstore persistente se crea en `LLM/vectordb/` en el primer uso.
