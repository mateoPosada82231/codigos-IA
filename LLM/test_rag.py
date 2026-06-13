import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rag import _cargar_vectorstore, MODELO_EMBED
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embed = OllamaEmbeddings(model=MODELO_EMBED)

# Try loading with explicit embedding function
try:
    vs = Chroma(persist_directory=r"C:\Users\mateo\OneDrive\Documentos\uni\codigos-IA\LLM\vectordb", embedding_function=embed)
    print("Vectorstore loaded OK")
except Exception as e:
    print(f"Load error: {e}")
    sys.exit(1)

# Test basic search
try:
    r = vs.similarity_search("error promedio red neuronal", k=5)
    print(f"Basic search: {len(r)} docs")
    for d in r:
        print(f"  [{d.metadata.get('tipo')}] {d.metadata.get('seccion','')[:60]}")
        print(f"  text: {d.page_content[:150]}")
except Exception as e:
    print(f"Basic search error: {type(e).__name__}: {e}")

# Test with filter
try:
    r = vs.similarity_search("error promedio", k=5, filter={"tipo": {"$eq": "informe"}})
    print(f"Filtered search (informe): {len(r)} docs")
    for d in r:
        print(f"  [{d.metadata.get('tipo')}] {d.metadata.get('seccion','')[:60]}")
except Exception as e:
    print(f"Filtered search error: {type(e).__name__}: {e}")

# Test MMR
try:
    r = vs.max_marginal_relevance_search("error promedio red neuronal", k=5, fetch_k=10)
    print(f"MMR search: {len(r)} docs")
    for d in r:
        print(f"  [{d.metadata.get('tipo')}] {d.metadata.get('seccion','')[:60]}")
except Exception as e:
    print(f"MMR search error: {type(e).__name__}: {e}")

# Check what's in the collection
try:
    count = vs._collection.count()
    print(f"Total docs in collection: {count}")
except Exception as e:
    print(f"Count error: {e}")
