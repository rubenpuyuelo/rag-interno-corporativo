import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = Path("chroma_db")

SYSTEM_PROMPT = """Eres un asistente interno de una empresa.
Responde SOLO usando el CONTEXTO proporcionado.

Reglas:
- Si la respuesta no está explícitamente en el contexto, responde exactamente: NO ENCONTRADO
- No uses conocimiento externo.
- Sé breve y claro.
- No inventes números, plazos o políticas.
"""

def pretty(doc):
    src = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", None)
    page_str = f"p.{page+1}" if isinstance(page, int) else "p.?"
    text = " ".join(doc.page_content.strip().split())
    return src, page_str, text

def filter_same_source_as_top1(results, abs_thresh: float, rel_delta: float):
    if not results:
        return []
    top_doc, top_score = results[0]
    if top_score is None or top_score < abs_thresh:
        return []
    top_source = top_doc.metadata.get("source", "unknown")
    rel_thresh = top_score - rel_delta

    good = []
    for doc, score in results:
        if score is None:
            continue
        if score >= abs_thresh and score >= rel_thresh:
            if doc.metadata.get("source", "unknown") == top_source:
                good.append((doc, score))
    if not good:
        good = [(top_doc, top_score)]
    return good

def build_context(good):
    blocks = []
    cites = []
    for idx, (doc, score) in enumerate(good, start=1):
        src, page_str, text = pretty(doc)
        blocks.append(f"[{idx}] ({src} {page_str}) {text}")
        cites.append({"id": idx, "source": src, "page": page_str, "score": float(score)})
    return "\n\n".join(blocks), cites

def ollama_chat(model: str, prompt: str) -> str:
    """
    Llama a Ollama por CLI. Esto evita dependencias de LangChain wrappers.
    """
    cmd = ["ollama", "run", model, prompt]
    out = subprocess.check_output(cmd, text=True)
    return out.strip()

def main():
    load_dotenv()

    ABS_THRESH = float(os.getenv("RELEVANCE_THRESHOLD", "0.24"))
    TOP_K = int(os.getenv("TOP_K", "4"))
    REL_DELTA = float(os.getenv("RELATIVE_DELTA", "0.02"))
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

    if not CHROMA_DIR.exists():
        raise SystemExit("No existe chroma_db. Ejecuta: python src/build_index.py")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="novaworks_docs",
    )

    print(f"Ollama model: {OLLAMA_MODEL}")
    print(f"Config: TOP_K={TOP_K} | ABS_THRESH={ABS_THRESH} | REL_DELTA={REL_DELTA}")
    print("Pregunta (ENTER para salir):")

    while True:
        q = input("> ").strip()
        if not q:
            break

        results = db.similarity_search_with_relevance_scores(q, k=TOP_K)
        good = filter_same_source_as_top1(results, abs_thresh=ABS_THRESH, rel_delta=REL_DELTA)

        if not good:
            print("\nRespuesta: NO ENCONTRADO\n")
            continue

        context, cites = build_context(good)

        prompt = f"""{SYSTEM_PROMPT}

PREGUNTA:
{q}

CONTEXTO:
{context}

Respuesta:
"""

        answer = ollama_chat(OLLAMA_MODEL, prompt)

        print("\nRespuesta:")
        print(answer)

        print("\nCitas:")
        for c in cites:
            print(f"- [{c['id']}] {c['source']} {c['page']} (score={c['score']:.3f})")
        print("")

if __name__ == "__main__":
    main()
