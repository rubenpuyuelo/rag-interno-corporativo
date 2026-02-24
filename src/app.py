import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Interno NovaWorks", page_icon="📚")
st.title("📚 Buscador interno (RAG) - NovaWorks")

st.caption("Responde solo con información de los documentos internos. Si no está, dirá: NO ENCONTRADO.")

# Estado del chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render mensajes previos
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Ver citas"):
                for c in msg["citations"]:
                    st.write(f"- [{c['id']}] {c['source']} {c['page']} (score={c['score']:.3f})")

# Input del usuario
user_input = st.chat_input("Escribe tu pregunta...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Llamada a la API
    try:
        r = requests.post(f"{API_URL}/preguntar", json={"question": user_input}, timeout=60)
        r.raise_for_status()
        data = r.json()
        answer = data.get("answer", "NO ENCONTRADO")
        citations = data.get("citations", [])
    except Exception as e:
        answer = f"Error llamando a la API: {e}"
        citations = []

    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})

    with st.chat_message("assistant"):
        st.markdown(answer)
        if citations:
            with st.expander("Ver citas"):
                for c in citations:
                    st.write(f"- [{c['id']}] {c['source']} {c['page']} (score={c['score']:.3f})")
