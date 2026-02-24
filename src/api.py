import os
import subprocess
import requests
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_DIR = Path("chroma_db")

SYSTEM_PROMPT = """Eres un asistente interno de una empresa.
Responde SOLO usando el CONTEXTO proporcionado.

Reglas:
- Si la respuesta no está explícitamente en el contexto, responde exactamente: NO ENCONTRADO
- No uses conocimiento externo.
- Sé breve y claro.
- No inventes números, plazos o políticas.
"""

# ---------- Modelos de entrada/salida (Pydantic) ----------
class AskRequest(BaseModel):
    question: str

class Citation(BaseModel):
    id: int
    source: str
    page: str
    score: float

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]

# ---------- Utilidades ----------
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
        cites.append(Citation(id=idx, source=src, page=page_str, score=float(score)))
    return "\n\n".join(blocks), cites

def extractive_answer_from_doc(doc) -> str:
    """
    Respuesta extractiva simple: devuelve el texto del chunk top.
    (Sin LLM, sin inventar.)
    """
    _, _, text = pretty(doc)
    return text

def ollama_chat(model: str, prompt: str) -> str:
    """
    Llama a Ollama por HTTP (ideal para Docker/Compose).
    """
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=240)
    r.raise_for_status()
    return r.json().get("response", "").strip()

# ---------- Inicialización (se hace 1 vez al arrancar) ----------
ABS_THRESH = float(os.getenv("RELEVANCE_THRESHOLD", "0.24"))
TOP_K = int(os.getenv("TOP_K", "4"))
REL_DELTA = float(os.getenv("RELATIVE_DELTA", "0.02"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
USE_LLM = os.getenv("USE_LLM", "true").lower() in ("1", "true", "yes", "y")

if not CHROMA_DIR.exists():
    raise SystemExit("No existe chroma_db. Ejecuta primero: python src/build_index.py")

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_db = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=_embeddings,
    collection_name="novaworks_docs",
)

# ---------- FastAPI ----------
app = FastAPI(title="RAG Interno Corporativo", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}
    
@app.post("/preguntar", response_model=AskResponse)
def preguntar(payload: AskRequest):
    q = payload.question.strip()
    if not q:
        return AskResponse(answer="NO ENCONTRADO", citations=[])

    results = _db.similarity_search_with_relevance_scores(q, k=TOP_K)
    if not results:
        return AskResponse(answer="NO ENCONTRADO", citations=[])

    # Guardrail 1: score mínimo del TOP1 (evita llamar al LLM con evidencias flojas)
    min_top1 = float(os.getenv("MIN_TOP1_SCORE", "0.26"))  # baja un poco vs 0.28
    top1_doc, top1_score = results[0]
    if top1_score is None or top1_score < min_top1:
        return AskResponse(answer="NO ENCONTRADO", citations=[])

    # Guardrail 2: consolidación por source (evita "citas raras")
    good = filter_same_source_as_top1(results, abs_thresh=ABS_THRESH, rel_delta=REL_DELTA)
    if not good:
        return AskResponse(answer="NO ENCONTRADO", citations=[])

    context, cites = build_context(good)

    prompt = f"""{SYSTEM_PROMPT}

PREGUNTA:
{q}

CONTEXTO:
{context}

Respuesta:
"""
    # Modo CI / rápido: sin LLM
    if not USE_LLM:
        best_doc, best_score = good[0]
        answer = extractive_answer_from_doc(best_doc).strip()
        return AskResponse(answer=answer, citations=cites)

    # Modo normal: con LLM
    answer = ollama_chat(OLLAMA_MODEL, prompt).strip()

    if answer.startswith("NO ENCONTRADO"):
        return AskResponse(answer="NO ENCONTRADO", citations=[])

    return AskResponse(answer=answer, citations=cites)

@app.get("/ready")
def ready():
    # 1) comprobar que hay colección/documentos
    try:
        # Chroma puede devolver vacío si no indexaste
        test = _db.similarity_search("test", k=1)
    except Exception as e:
        return {"ready": False, "reason": f"chroma_error: {e}"}

    # 2) Ollama listo (solo si usamos LLM)
    if USE_LLM:
        try:
            import requests
            base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
            r = requests.get(f"{base}/api/tags", timeout=5)
            r.raise_for_status()
            tags = r.json()
            model_names = [m.get("name") for m in tags.get("models", [])]
            if OLLAMA_MODEL not in model_names:
                return {"ready": False, "reason": f"model_not_pulled: {OLLAMA_MODEL}"}
        except Exception as e:
            return {"ready": False, "reason": f"ollama_error: {e}"}