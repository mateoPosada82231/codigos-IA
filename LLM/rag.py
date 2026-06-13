"""
rag.py
================
Sistema RAG (Retrieval-Augmented Generation) para consultar el proyecto
de levitación: informe LaTeX, código fuente Python y datos CSV de pruebas.
Usa Ollama (modelos locales, gratuitos) y ChromaDB como vector store.

Flujo:
  1. Procesa informe .tex → texto plano (preservando ecuaciones)
  2. Procesa código .py → chunks por función/clase
  3. Procesa datos .csv → resúmenes estadísticos
  4. Almacena todo en ChromaDB (persistente)
  5. Por cada pregunta, busca los chunks más relevantes
  6. El LLM responde basado en el contexto recuperado

Requiere:
    pip install ollama chromadb langchain langchain-community
    ollama pull llama3.2     (o mistral, phi3, etc.)
    ollama pull nomic-embed-text

Uso:
    python rag.py
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

# ── Configuración de fuentes adicionales ─────────────────────────────────────
RUTAS_CODIGO = [
    os.path.join(PROYECTO_DIR, "logica_difusa"),
    os.path.join(PROYECTO_DIR, "redes_neuronales"),
    os.path.join(PROYECTO_DIR, "aprendizaje_refuerzo"),
]
EXCLUIR_ARCHIVOS = ["pesos_", "qtable"]
EXCLUIR_DIRS = ["graficas", "graficos", ".idea", "__pycache__", "resultados", "LLM"]

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
    # ── Limpiar comandos matemáticos antes de quitar delimitadores ──
    # Nota: \frac no se trata explícitamente porque solo hay 1 instancia
    # en el informe (\frac{\text{PWM}-200}{400}) con anidamiento que
    # rompe regex simples; el barrido genérico \\([a-zA-Z]) lo reduce a
    # "frac..." que el LLM interpreta igual.
    texto = texto.replace('\\cdot', ' * ')
    texto = texto.replace('\\times', ' x ')
    texto = texto.replace('\\to', ' -> ')
    texto = texto.replace('\\partial', ' d')
    # Letras griegas comunes (en ASCII para compatibilidad con terminal)
    for cmd, char in [
        ('\\alpha', 'alpha'), ('\\beta', 'beta'), ('\\gamma', 'gamma'),
        ('\\delta', 'delta'), ('\\epsilon', 'epsilon'), ('\\sigma', 'sigma'),
        ('\\tau', 'tau'), ('\\theta', 'theta'), ('\\mu', 'mu'),
        ('\\pi', 'pi'), ('\\omega', 'omega'), ('\\phi', 'phi'),
        ('\\Delta', 'Delta'), ('\\Sigma', 'Sigma'),
    ]:
        texto = texto.replace(cmd, char)
    # Conservar contenido de ecuaciones (solo quitar delimitadores)
    texto = re.sub(r'\$\$(.*?)\$\$', r'\1', texto, flags=re.DOTALL)
    texto = re.sub(r'\$(.*?)\$', r'\1', texto)
    texto = re.sub(r'\\\[(.*?)\\\]', r'\1', texto, flags=re.DOTALL)
    texto = re.sub(r'\\\((.*?)\\\)', r'\1', texto, flags=re.DOTALL)
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
    Divide el .tex en secciones, respetando la jerarquía section/subsection.
    Devuelve lista de (ruta_completa_seccion, contenido_texto_plano).
    """
    match_body = re.search(
        r'\\begin\{document\}(.*?)\\end\{document\}',
        texto_tex, re.DOTALL
    )
    if match_body:
        cuerpo = match_body.group(1)
    else:
        cuerpo = texto_tex

    # Eliminar thebibliography
    cuerpo = re.sub(
        r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
        '',
        cuerpo, flags=re.DOTALL
    )

    # Dividir respetando niveles jerárquicos
    # Para cada match, capturamos: tipo (section/subsection), título, y todo hasta el próximo header
    patron = re.compile(
        r'\\(section|subsection|subsubsection)\*?\s*\{(.*?)\}',
        re.DOTALL
    )
    pos = 0
    secciones = []
    current_section = None
    current_subsection = None

    # Contenido antes del primer header
    primero = limpiar_tex(cuerpo[:cuerpo.find('\\section')] if '\\section' in cuerpo else cuerpo)
    if primero.strip():
        secciones.append(("Resumen / Abstract", primero.strip()))

    for match in patron.finditer(cuerpo):
        nivel = match.group(1)
        titulo = match.group(2).strip()

        # Actualizar jerarquía ANTES de verificar contenido
        if nivel == 'section':
            current_section = titulo
            current_subsection = None
        elif nivel == 'subsection':
            current_subsection = titulo
        # subsubsection no cambia current_subsection

        inicio = match.end()
        fin = patron.search(cuerpo, inicio)
        contenido_raw = cuerpo[inicio:fin.start() if fin else len(cuerpo)]
        contenido = limpiar_tex(contenido_raw)
        if not contenido.strip():
            continue

        if nivel == 'section':
            secciones.append((current_section, contenido.strip()))
        elif nivel == 'subsection':
            ruta = f"{current_section} > {titulo}" if current_section else titulo
            secciones.append((ruta, contenido.strip()))
        elif nivel == 'subsubsection':
            if current_subsection:
                ruta = f"{current_section} > {current_subsection} > {titulo}"
            elif current_section:
                ruta = f"{current_section} > {titulo}"
            else:
                ruta = titulo
            secciones.append((ruta, contenido.strip()))

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
    Incluye el título de la sección como prefijo para mejorar la recuperación semántica.
    """
    chunks = []
    for titulo, contenido in secciones:
        palabras = contenido.split()
        if not palabras:
            continue
        prefijo = f"[{titulo}]\n"
        start = 0
        while start < len(palabras):
            end = min(start + chunk_size, len(palabras))
            fragmento = ' '.join(palabras[start:end])
            chunks.append({
                "text": prefijo + fragmento,
                "metadata": {"seccion": titulo, "tipo": "informe"}
            })
            if end == len(palabras):
                break
            start = end - overlap
    return chunks


# ── 3. Procesar código fuente Python ────────────────────────────────────────

def _archivo_excluido(nombre):
    """True si el archivo debe ser excluido."""
    return any(nombre.startswith(p) for p in EXCLUIR_ARCHIVOS)


def _extraer_primeras_lineas(texto, n=3):
    """Devuelve las primeras n líneas no vacías como resumen."""
    lineas = [l for l in texto.split('\n') if l.strip()][:n]
    return ' | '.join(l.strip() for l in lineas)


def procesar_codigo_py():
    """
    Escanea directorios de código, divide cada .py en funciones/clases
    y retorna chunks con metadatos (archivo, módulo, tecnología).
    """
    chunks = []
    for ruta_dir in RUTAS_CODIGO:
        if not os.path.isdir(ruta_dir):
            continue
        tecnologia = os.path.basename(ruta_dir)
        for archivo in sorted(os.listdir(ruta_dir)):
            ruta_py = os.path.join(ruta_dir, archivo)
            if not os.path.isfile(ruta_py):
                continue
            if not archivo.endswith('.py'):
                continue
            if _archivo_excluido(archivo):
                continue

            with open(ruta_py, 'r', encoding='utf-8') as f:
                contenido = f.read()

            modulo = archivo.replace('.py', '')

            # Dividir por definiciones de nivel superior (def / class)
            patron = re.compile(r'^(?=def |class )', re.MULTILINE)
            bloques = patron.split(contenido)

            for bloque in bloques:
                bloque = bloque.strip()
                if not bloque:
                    continue
                resumen = _extraer_primeras_lineas(bloque, 2)
                prefijo = f"[{modulo} - {tecnologia}]\n"
                chunks.append({
                    "text": prefijo + bloque,
                    "metadata": {
                        "seccion": f"codigo/{tecnologia}/{modulo}",
                        "tipo": "codigo",
                        "archivo": archivo,
                        "funcion": resumen,
                        "modulo": modulo,
                        "tecnologia": tecnologia,
                    }
                })

    cant_por_tec = {}
    for c in chunks:
        t = c['metadata'].get('tecnologia', 'otro')
        cant_por_tec[t] = cant_por_tec.get(t, 0) + 1
    for tec, cant in sorted(cant_por_tec.items()):
        print(f"  {tec}: {cant} bloques de código")
    return chunks


# ── 4. Resumir CSVs de pruebas ─────────────────────────────────────────────

def resumir_csvs():
    """
    Lee cada datos_esp32_*.csv del proyecto, calcula estadísticas
    por columna numérica y genera un chunk de texto descriptivo.
    """
    import csv

    chunks = []
    for ruta_dir in RUTAS_CODIGO:
        if not os.path.isdir(ruta_dir):
            continue
        tecnologia = os.path.basename(ruta_dir)
        for archivo in sorted(os.listdir(ruta_dir)):
            if not archivo.endswith('.csv') or not archivo.startswith('datos_esp32'):
                continue
            ruta_csv = os.path.join(ruta_dir, archivo)
            with open(ruta_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                filas = list(reader)

            if not filas:
                continue

            columnas = list(filas[0].keys())
            stats = {}
            for col in columnas:
                valores = []
                for row in filas:
                    try:
                        v = float(row[col])
                        valores.append(v)
                    except (ValueError, TypeError):
                        pass
                if not valores:
                    continue
                n = len(valores)
                media = sum(valores) / n
                var = sum((x - media) ** 2 for x in valores) / n
                stats[col] = {
                    'n': n, 'media': media, 'std': var ** 0.5,
                    'min': min(valores), 'max': max(valores),
                }

            lineas = [
                f"[Resumen de {archivo}]  ({tecnologia})",
                f"Columnas: {', '.join(columnas)}",
                f"Lecturas: {len(filas)}",
                "",
            ]
            for col, s in stats.items():
                lineas.append(
                    f"  {col}: media={s['media']:.4f}, std={s['std']:.4f}, "
                    f"min={s['min']:.4f}, max={s['max']:.4f}, n={s['n']}"
                )

            prefijo = f"[Datos: {archivo} - {tecnologia}]\n"
            chunks.append({
                "text": prefijo + '\n'.join(lineas),
                "metadata": {
                    "seccion": f"datos/{archivo}",
                    "tipo": "datos_csv",
                    "archivo": archivo,
                    "tecnologia": tecnologia,
                }
            })

    if chunks:
        print(f"  Generados {len(chunks)} resúmenes de datos CSV")
    return chunks


# ── 5. Vector Store (ChromaDB + Ollama embeddings) ────────────────────────

def _crear_vectorstore(chunks):
    """Crea ChromaDB con embeddings de Ollama."""
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

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
    print(f"  Vectorstore creado con {len(docs)} chunks en: {RUTA_VECTORSTORE}")
    return vectorstore


def _cargar_vectorstore():
    """Carga ChromaDB existente."""
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma

    embeddings = OllamaEmbeddings(model=MODELO_EMBED)
    vectorstore = Chroma(
        persist_directory=RUTA_VECTORSTORE,
        embedding_function=embeddings,
    )
    return vectorstore


def _reunir_chunks():
    """Reúne chunks de todas las fuentes (informe, código, datos)."""
    chunks = []

    # 1. Informe .tex
    print("  Procesando informe LaTeX...")
    secciones = extraer_texto_completo()
    chunks += chunk_secciones(secciones)
    print(f"    {len(secciones)} secciones extraídas")

    # 2. Código Python
    print("  Procesando código fuente...")
    chunks += procesar_codigo_py()

    # 3. Resúmenes de CSV
    print("  Procesando datos de pruebas...")
    chunks += resumir_csvs()

    return chunks


def obtener_vectorstore(forzar_recrear=False):
    """Retorna vectorstore, creándolo si es necesario."""
    if forzar_recrear:
        print("  Forzando recreación del vectorstore...")
        chunks = _reunir_chunks()
        return _crear_vectorstore(chunks)

    if os.path.isdir(RUTA_VECTORSTORE) and os.listdir(RUTA_VECTORSTORE):
        print(f"  Cargando vectorstore existente desde: {RUTA_VECTORSTORE}")
        try:
            return _cargar_vectorstore()
        except Exception:
            print("  No se pudo cargar. Recreando...")

    print("  Creando vectorstore desde todas las fuentes...")
    chunks = _reunir_chunks()
    return _crear_vectorstore(chunks)


# ── 4. Consulta RAG ───────────────────────────────────────────────────────

def _expandir_consulta(pregunta):
    """Genera variantes de la consulta para mejorar recuperación."""
    variantes = [pregunta]
    p_lower = pregunta.lower()

    if 'mejor' in p_lower or 'resultado' in p_lower or 'funcion' in p_lower:
        variantes.append(f"resultados error promedio desviación estándar {pregunta}")
        variantes.append(f"desempeño controladores métricas error {pregunta}")

    if 'método' in p_lower or 'metodo' in p_lower or 'técnica' in p_lower or 'tecnica' in p_lower:
        variantes.append(f"comparación controladores DQN Q-Learning fuzzy resultados {pregunta}")

    if 'código' in p_lower or 'codigo' in p_lower:
        variantes.append(f"implementación código fuente {pregunta}")

    return variantes


def _diversificar_fuentes(pregunta, vectorstore, k=10):
    """
    Recupera chunks combinando varias estrategias de búsqueda.
    Siempre forza la inclusión de chunks del informe con consultas
    específicas para cubrir preguntas sobre resultados y métricas.
    """
    vistos = set()
    docs = []

    # 1. Búsqueda forzada en el informe con consultas dirigidas
    #    (usa el vocabulario exacto de la sección Resultados)
    consultas_informe = [
        pregunta,
        "error promedio desviación estándar distancia máxima controladores",
        "resultados DQN Fuzzy Q-Learning comparación error método",
        "desempeño controladores levitación métricas tabla",
        "mejor resultado global error promedio menor",
    ]
    for q in consultas_informe:
        try:
            resultados = vectorstore.similarity_search(
                q, k=3,
                filter={"tipo": {"$eq": "informe"}}
            )
        except Exception as e:
            print(f"  [DEBUG] Error en búsqueda con filtro: {type(e).__name__}: {e}")
            resultados = []
        for d in resultados:
            id_unico = d.page_content[:100]
            if id_unico not in vistos:
                vistos.add(id_unico)
                docs.append(d)

    # 2. Búsqueda general con expansión (todas las fuentes)
    consultas = _expandir_consulta(pregunta)
    for q in consultas:
        try:
            resultados = vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=k * 2)
        except Exception as e:
            print(f"  [DEBUG] Error en MMR search: {type(e).__name__}: {e}")
            resultados = []
        for d in resultados:
            id_unico = d.page_content[:100]
            if id_unico not in vistos:
                vistos.add(id_unico)
                docs.append(d)
        if len(docs) >= k * 2:
            break

    return docs[:k]


def consultar(pregunta, vectorstore, k=10):
    """
    Recupera chunks relevantes y genera respuesta con Ollama.
    """
    import ollama

    docs = _diversificar_fuentes(pregunta, vectorstore, k)

    if not docs:
        print("  No se encontraron documentos relevantes.")
        return

    contexto = ""
    fuentes = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        seccion = meta.get('seccion', 'N/A')
        tipo = meta.get('tipo', 'informe')
        contexto += f"[Fuente {i+1} - {seccion}]\n{doc.page_content}\n\n"
        fuentes.append(seccion)

    # Construir prompt
    prompt = f"""Eres un asistente experto en sistemas de control inteligente para levitación.
Tus fuentes de información son:
- El informe del proyecto (informe.tex)
- El código fuente de las implementaciones (Python, MicroPython)
- Los resúmenes estadísticos de las pruebas experimentales (datos CSV)
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

    # Mostrar respuesta (con protección de encoding)
    import sys
    safe = texto.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
    print("-" * 60)
    print(safe)
    print("-" * 60)
    tipos = set(d.metadata.get('tipo', 'informe') for d in docs)
    print(f"\n  Tipos de fuente: {', '.join(sorted(tipos))}")
    print(f"  Secciones consultadas: {', '.join(dict.fromkeys(fuentes))}")


# ── 5. CLI interactiva ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG para consultar el proyecto de levitación (.tex + .py + .csv)"
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

    print("\n" + "=" * 60)
    print("  RAG — Sistema de Consulta del Proyecto de Levitacion")
    print("=" * 60)
    print(f"  Modelo LLM:     {MODELO_LLM}")
    print(f"  Modelo Embed:   {MODELO_EMBED}")
    print(f"  Fuentes:        .tex + .py + .csv")
    print(f"  Informe:        {RUTA_INFORME}")
    print("-" * 60)

    # ── Verificar conectividad con Ollama ──
    import ollama as _ollama_check
    try:
        _ollama_check.list()
        print("  Ollama:          conectado")
    except Exception as e:
        print(f"  [ERROR] No se puede conectar a Ollama: {e}")
        print("  Asegúrate de que Ollama esté corriendo:  ollama serve")
        print("  Y que los modelos estén descargados:     ollama pull {MODELO_LLM}")
        print("                                            ollama pull {MODELO_EMBED}")
        sys.exit(1)

    vs = obtener_vectorstore(forzar_recrear=args.rebuild)

    if args.pregunta:
        pregunta = ' '.join(args.pregunta)
        print(f"\n  Pregunta: {pregunta}")
        consultar(pregunta, vs)
        return

    print("\n  Modo interactivo. Escribe 'salir' o 'exit' para terminar.")
    print("  Escribe 'rebuild' para reconstruir el vectorstore.")
    print("-" * 60)

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
