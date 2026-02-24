import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = Path("chroma_db")

def pretty(doc):
    src = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", None)
    page_str = f"p.{page+1}" if isinstance(page, int) else "p.?"
    text = " ".join(doc.page_content.strip().split())
    return src, page_str, text

def extract_bullets(text: str, max_lines: int = 4) -> str:
    """
    Extrae líneas tipo bullet para dar una respuesta corta sin usar LLM.
    Como nuestros PDFs tienen guiones '-', esto funciona bien.
    """
    # reconstruimos con saltos aproximados
    raw = text.replace(" - ", "\n- ").replace("  - ", "\n- ")
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    bullets = [ln for ln in lines if ln.startswith("-")]
    if bullets:
        return "\n".join(bullets[:max_lines])
    # fallback: primera frase corta
    return text[:240] + ("..." if len(text) > 240 else "")

def filter_results(results, abs_thresh: float, rel_delta: float):
    """
    Estrategia didactica y muy util en RAG con pocos documentos:
    1) Elegimos el TOP1 (si pasa abs_thresh).
    2) Solo aceptamos otros resultados si:
       - pasan abs_thresh
       - estan cerca del TOP1 (rel_delta)
       - y son del MISMO source que el TOP1

    Esto evita "citas raras" de otros PDFs cuando el dataset es pequeno.
    """
    if not results:
        return []

    top_doc, top_score = results[0]
    if top_score is None or top_score < abs_thresh:
        return []

    top_source = top_doc.metadata.get("source", "unknown")
    max_score = top_score
    rel_thresh = max_score - rel_delta

    good = []
    for doc, score in results:
        if score is None:
            continue
        if score >= abs_thresh and score >= rel_thresh:
            if doc.metadata.get("source", "unknown") == top_source:
                good.append((doc, score))

    # Garantizamos al menos el TOP1
    if not good:
        good = [(top_doc, top_score)]

    return good

def main():
    load_dotenv()

    ABS_THRESH = float(os.getenv("RELEVANCE_THRESHOLD", "0.24"))
    TOP_K = int(os.getenv("TOP_K", "4"))
    REL_DELTA = float(os.getenv("RELATIVE_DELTA", "0.02"))  # cuanto cerca del top debe estar

    if not CHROMA_DIR.exists():
        raise SystemExit("No existe chroma_db. Ejecuta: python src/build_index.py")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="novaworks_docs",
    )

    print(f"Config: TOP_K={TOP_K} | ABS_THRESH={ABS_THRESH} | REL_DELTA={REL_DELTA}")
    print("Pregunta (ENTER para salir):")

    while True:
        q = input("> ").strip()
        if not q:
            break

        results = db.similarity_search_with_relevance_scores(q, k=TOP_K)

        print("\n--- Top-k (debug) ---")
        for i, (doc, score) in enumerate(results, start=1):
            src, page_str, text = pretty(doc)
            print(f"{i}. score={score:.3f} | {src} {page_str} | {text[:110]}{'...' if len(text)>110 else ''}")

        good = filter_results(results, abs_thresh=ABS_THRESH, rel_delta=REL_DELTA)

        if not good:
            print("\nRespuesta: No encuentro esta información en los documentos proporcionados.\n")
            continue

        # respuesta corta basada en el mejor doc
        best_doc, best_score = good[0]
        src, page_str, text = pretty(best_doc)
        answer_short = extract_bullets(text)

        print("\nRespuesta (corta, basada en evidencia):")
        print(answer_short)

        print("\nCitas usadas:")
        for doc, score in good:
            src, page_str, _ = pretty(doc)
            print(f"- {src} {page_str} (score={score:.3f})")
        print("")

if __name__ == "__main__":
    main()
