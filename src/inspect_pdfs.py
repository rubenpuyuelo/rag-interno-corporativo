from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

PDF_DIR = Path("data/pdfs")

def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No hay PDFs en data/pdfs")

    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        print("=" * 80)
        print(f"{pdf.name} -> {len(pages)} paginas")
        preview = pages[0].page_content.strip().replace("\n", " ")
        print("Preview:", preview[:250], "...")
    print("\nOK: lectura PDF funcionando.")

if __name__ == "__main__":
    main()
