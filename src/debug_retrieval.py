from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = Path("chroma_db")

def main():
    if not CHROMA_DIR.exists():
        raise SystemExit("No existe chroma_db. Ejecuta: python src/build_index.py")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="novaworks_docs",
    )

    q = "¿Cuántos días de vacaciones tengo?"
    results = db.similarity_search_with_relevance_scores(q, k=6)

    print("Query:", q)
    print("\nTOP-K resultados:")
    for i, (doc, score) in enumerate(results, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        page_str = f"p.{page+1}" if isinstance(page, int) else "p.?"
        text = " ".join(doc.page_content.strip().split())
        print(f"{i}. score={score} | {src} {page_str} | {text[:160]}{'...' if len(text)>160 else ''}")

if __name__ == "__main__":
    main()
