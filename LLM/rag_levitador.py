"""
rag_levitador.py
================
Sistema RAG (Retrieval-Augmented Generation) para consultar el informe
del proyecto de levitación usando Ollama (modelos locales, gratuitos).

Flujo:
  1. Procesa el archivo .tex del informe → texto plano
  2. Divide en chunks y los almacena en ChromaDB (persistente)
  3. Por cada pregunta, busca los chunks más relevantes
  4. El LLM responde basado en el contexto recuperado

Requiere:
    pip install ollama chromadb langchain langchain-community
    ollama pull llama3.2     (o mistral, phi3, etc.)
    ollama pull nomic-embed-text

Uso:
    python rag_levitador.py
"""

import os
import sys
import re
import textwrap
import argparse

# ── Configuración ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROYECTO_DIR = os.path.dirname(SCRIPT_DIR)  # carpeta raíz del proyecto

RUTA_INFORME = os.path.join(PROYECTO_DIR, "informe.tex")
RUTA_VECTORSTORE = os.path.join(SCRIPT_DIR, "vectordb")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MODELO_LLM = "llama3.2"       # cambiar a "mistral", "phi3", etc.
MODELO_EMBED = "nomic-embed-text"

# ── 1. Extraer texto del .tex ─────────────────────────────────────────────

def limpiar_tex(texto):
    """Elimina comandos LaTeX y devuelve texto plano legible."""
    # Eliminar comentarios
    texto = re.sub(r'(?<!\\)%.*', '', texto)
    # Eliminar \begin{...} y \end{...}
    texto = re.sub(r'\\begin\{[^}]*\}', '', texto)
    texto = re.sub(r'\\end\{[^}]*\}', '', texto)
    # Eliminar \appendix
    texto = re.sub(r'\\appendix', '', texto)
    # Eliminar \label, \ref, \cite con sus argumentos
    texto = re.sub(r'\\(?:label|ref|cite)\{[^}]*\}', '', texto)
    # Eliminar \url{...}
    texto = re.sub(r'\\url\{[^}]*\}', '', texto)
    # Eliminar \href{...}{...}
    texto = re.sub(r'\\href\{[^}]*\}\{[^}]*\}', '', texto)
    # Eliminar \texttt, \textit, \textbf, \emph, \textsc, \textsf, \textrm
    texto = re.sub(r'\\(?:texttt|textit|textbf|emph|textsc|textsf|textrm|textsuperscript)\{([^}]*)\}', r'\1', texto)
    # Eliminar \includegraphics[opts]{file}
    texto = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}', '', texto)
    # Eliminar \lstdefinestyle, \lstset completos (bloques enteros)
    texto = re.sub(r'\\lstdefinestyle\{[^}]*\}[^%]*?(?=\\begin\{document\}|\\section)', '', texto, flags=re.DOTALL)
    # Eliminar \definecolor, \color, \textcolor
    texto = re.sub(r'\\(?:definecolor|color|textcolor)\{?[^}]*\}?', '', texto)
    # Eliminar \hypersetup{...}
    texto = re.sub(r'\\hypersetup\{[^}]*\}', '', texto)
    # Eliminar $...$ (matemáticas en línea)
    texto = re.sub(r'\$[^$]*\$', '', texto)
    # Eliminar $$...$$ (matemáticas en bloque)
    texto = re.sub(r'\$\$[^$]*\$\$', '', texto)
    # Eliminar \[ ... \] (matemáticas en bloque)
    texto = re.sub(r'\\\[.*?\\\]', '', texto, flags=re.DOTALL)
    # Eliminar \(...\)
    texto = re.sub(r'\\\(.*?\\\)', '', texto, flags=re.DOTALL)
    # Eliminar \Big, \big, \left, \right, \displaystyle, etc.
    texto = re.sub(r'\\(?:Big|big|Bigg|bigg|left|right|displaystyle|text|mbox|small|large|Large)\s*', '', texto)
    # Eliminar \begin{thebibliography} ... \end{thebibliography} (mantener las citas como texto)
    texto = re.sub(r'\\bibitem\{[^}]*\}', '• ', texto)
    # Eliminar llaves sueltas
    texto = texto.replace('{', '').replace('}', '')
    # Eliminar ~ (espacio duro)
    texto = texto.replace('~', ' ')
    # Eliminar comandos \_ \- \&
    texto = texto.replace('\\_', '_')
    texto = texto.replace('\\-', '')
    texto = texto.replace('\\&', '&')
    # Comillas dobles LaTeX `` ''
    texto = texto.replace('``', '"').replace("''", '"')
    # Guiones largos
    texto = texto.replace('---', ' — ').replace('--', ' – ')
    # Eliminar saltos de línea múltiples
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    # Eliminar espacios múltiples
    texto = re.sub(r' +', ' ', texto)
    # Eliminar \ sobrantes que preceden letras
    texto = re.sub(r'\\([a-zA-Z])', r'\1', texto)
    return texto.strip()


def extraer_secciones(texto_tex):
    """
    Divide el .tex en secciones (section, subsection).
    Solo procesa el cuerpo del documento (entre \begin{document} y \end{document}).
    Devuelve lista de (titulo_seccion, contenido_texto_plano).
    """
    # Extraer solo el cuerpo del documento
    match_body = re.search(
        r'\\begin\{document\}(.*?)\\end\{document\}',
        texto_tex, re.DOTALL
    )
    if match_body:
        cuerpo = match_body.group(1)
    else:
        cuerpo = texto_tex

    # Eliminar thebibliography (mantener solo las citas como texto plano)
    cuerpo = re.sub(
        r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
        '',
        cuerpo, flags=re.DOTALL
    )

    # pattern títulos de sección
    patron_seccion = re.compile(
        r'\\(?:section|subsection|subsubsection)\*?\s*\{(.*?)\}',
        re.DOTALL
    )
    partes = patron_seccion.split(cuerpo)
    secciones = []
    # Las partes van en pares: título, contenido (empezando desde índice 0 si hay contenido antes de la primera sección)
    # Ignoramos la primera parte si está vacía o es solo preámbulo
    start_idx = 0
    if partes:
        primero = limpiar_tex(partes[0])
        if primero.strip():
            secciones.append(("Resumen / Abstract", primero.strip()))
            start_idx = 1
        else:
            start_idx = 1
    for i in range(start_idx, len(partes) - 1, 2):
        titulo = partes[i].strip()
        contenido_raw = partes[i + 1] if i + 1 < len(partes) else ""
        contenido = limpiar_tex(contenido_raw)
        if contenido.strip():
            secciones.append((titulo, contenido.strip()))
    return secciones


def extraer_texto_completo():
    """Lee el .tex y devuelve texto plano completo."""
    if not os.path.isfile(RUTA_INFORME):
        print(f"[ERROR] No se encuentra el informe en: {RUTA_INFORME}")
        print("  Asegúrate de que 'informe.tex' esté en la raíz del proyecto.")
        sys.exit(1)

    with open(RUTA_INFORME, 'r', encoding='utf-8') as f:
        raw = f.read()

    secciones = extraer_secciones(raw)
    return secciones


# ── 2. Chunking ───────────────────────────────────────────────────────────

def chunk_secciones(secciones, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Divide cada sección en chunks de tamaño aproximado.
    """
    chunks = []
    for titulo, contenido in secciones:
        palabras = contenido.split()
        if not palabras:
            continue
        start = 0
        while start < len(palabras):
            end = min(start + chunk_size, len(palabras))
            fragmento = ' '.join(palabras[start:end])
            chunks.append({
                "text": fragmento,
                "metadata": {"seccion": titulo}
            })
            if end == len(palabras):
                break
            start = end - overlap
    return chunks


# ── 3. Vector Store (ChromaDB + Ollama embeddings) ────────────────────────

def _crear_vectorstore(chunks):
    """Crea ChromaDB con embeddings de Ollama."""
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document

    docs = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in chunks
    ]

    embeddings = OllamaEmbeddings(model=MODELO_EMBED)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=RUTA_VECTORSTORE,
    )
    vectorstore.persist()
    print(f"  Vectorstore creado con {len(docs)} chunks en: {RUTA_VECTORSTORE}")
    return vectorstore


def _cargar_vectorstore():
    """Carga ChromaDB existente."""
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = OllamaEmbeddings(model=MODELO_EMBED)
    vectorstore = Chroma(
        persist_directory=RUTA_VECTORSTORE,
        embedding_function=embeddings,
    )
    return vectorstore


def obtener_vectorstore(forzar_recrear=False):
    """Retorna vectorstore, creándolo si es necesario."""
    if forzar_recrear:
        print("  Forzando recreación del vectorstore...")
        secciones = extraer_texto_completo()
        chunks = chunk_secciones(secciones)
        return _crear_vectorstore(chunks)

    if os.path.isdir(RUTA_VECTORSTORE) and os.listdir(RUTA_VECTORSTORE):
        print(f"  Cargando vectorstore existente desde: {RUTA_VECTORSTORE}")
        try:
            return _cargar_vectorstore()
        except Exception:
            print("  No se pudo cargar. Recreando...")

    print("  Creando vectorstore desde el informe...")
    secciones = extraer_texto_completo()
    chunks = chunk_secciones(secciones)
    return _crear_vectorstore(chunks)


# ── 4. Consulta RAG ───────────────────────────────────────────────────────

def consultar(pregunta, vectorstore, k=5):
    """
    Recupera chunks relevantes y genera respuesta con Ollama.
    """
    import ollama

    # Recuperar documentos relevantes
    docs = vectorstore.similarity_search(pregunta, k=k)
    if not docs:
        print("  No se encontraron documentos relevantes.")
        return

    contexto = ""
    fuentes = []
    for i, doc in enumerate(docs):
        contexto += f"[Fuente {i+1} - {doc.metadata.get('seccion', 'N/A')}]\n{doc.page_content}\n\n"
        fuentes.append(doc.metadata.get('seccion', 'N/A'))

    # Construir prompt
    prompt = f"""Eres un asistente experto en sistemas de control inteligente para levitación.
Tu única fuente de información es el siguiente extracto del informe del proyecto.
Responde ÚNICAMENTE con la información contenida en el contexto proporcionado.
Si la información no está en el contexto, di que no la tienes disponible.
Responde en español, de forma clara y concisa.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""

    print("  Generando respuesta (esto puede tomar unos segundos)...\n")

    try:
        respuesta = ollama.chat(
            model=MODELO_LLM,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 512}
        )
        texto = respuesta["message"]["content"].strip()
    except Exception as e:
        print(f"\n[ERROR] Falló la consulta a Ollama: {e}")
        print("  Asegúrate de que Ollama esté corriendo y el modelo esté descargado.")
        print("  Comandos:  ollama serve  y  ollama pull {MODELO_LLM}")
        return

    # Mostrar respuesta
    print("─" * 60)
    print(texto)
    print("─" * 60)
    print(f"\n  Fuentes consultadas: {', '.join(set(fuentes))}")


# ── 5. CLI interactiva ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG para consultar el informe del levitador"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Forzar recreación del vectorstore desde el .tex"
    )
    parser.add_argument(
        "pregunta", nargs="*",
        help="Pregunta directa (opcional). Si no se provee, entra en modo interactivo."
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  RAG — Sistema de Consulta del Informe de Levitación")
    print("═" * 60)
    print(f"  Modelo LLM:     {MODELO_LLM}")
    print(f"  Modelo Embed:   {MODELO_EMBED}")
    print(f"  Informe:        {RUTA_INFORME}")
    print("─" * 60)

    vs = obtener_vectorstore(forzar_recrear=args.rebuild)

    if args.pregunta:
        pregunta = ' '.join(args.pregunta)
        print(f"\n  Pregunta: {pregunta}")
        consultar(pregunta, vs)
        return

    print("\n  Modo interactivo. Escribe 'salir' o 'exit' para terminar.")
    print("  Escribe 'rebuild' para reconstruir el vectorstore.")
    print("─" * 60)

    while True:
        try:
            pregunta = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not pregunta:
            continue
        if pregunta.lower() in ('salir', 'exit', 'quit', 'q'):
            break
        if pregunta.lower() == 'rebuild':
            vs = obtener_vectorstore(forzar_recrear=True)
            print("  Vectorstore reconstruido.")
            continue

        consultar(pregunta, vs)

    print("\n¡Hasta luego!\n")


if __name__ == "__main__":
    main()
