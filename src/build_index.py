from pathlib import Path
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_DIR = Path("data/pdfs")
CHROMA_DIR = Path("chroma_db")

def load_pdfs():
    docs = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        # Añadimos metadatos útiles
        for p in pages:
            p.metadata["source"] = pdf.name
        docs.extend(pages)
    return docs

def main():
    if not PDF_DIR.exists():
        raise SystemExit("No existe data/pdfs")

    docs = load_pdfs()
    if not docs:
        raise SystemExit("No hay PDFs para indexar")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Embeddings locales (sin API keys)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Si quieres regenerar desde cero, borra la carpeta antes:
    # shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="novaworks_docs",
    )
    vectordb.persist()

    print(f"OK: Index creado en {CHROMA_DIR.resolve()}")
    print(f"Docs: {len(docs)} | Chunks: {len(chunks)}")

if __name__ == "__main__":
    main()
