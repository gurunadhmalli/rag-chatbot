import streamlit as st
import cohere
import pdfplumber
import faiss
import numpy as np
import tempfile
import os

# ================================
# Load Cohere API Key from secrets
# ================================
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
co = cohere.Client(COHERE_API_KEY)

# ================================
# Custom CSS for colorful UI
# ================================
st.markdown('''
    <style>
    body {
        background-color: #f4f6f8;
    }
    .chat-bubble {
        padding: 12px;
        margin: 8px 0;
        border-radius: 12px;
        max-width: 80%;
        word-wrap: break-word;
    }
    .user-bubble {
        background-color: #4f9eed;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .bot-bubble {
        background-color: #e6e6e6;
        color: black;
        margin-right: auto;
        text-align: left;
    }
    .stApp {
        background: linear-gradient(120deg, #a6c0fe, #f68084);
        background-attachment: fixed;
    }
    </style>
''', unsafe_allow_html=True)

# ================================
# Helper Functions
# ================================
def load_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_texts(texts, mode="search_document"):
    response = co.embed(
        model="embed-english-v3.0",
        input_type=mode,
        texts=texts
    )
    return np.array(response.embeddings, dtype="float32")

def build_vector_db(chunks):
    embeddings = embed_texts(chunks, mode="search_document")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

def search_db(query, index, chunks, top_k=3):
    query_embedding = embed_texts([query], mode="search_query")
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"Answer the question using the context below:\n{context}\n\nQuestion: {query}"
    response = co.chat(model="command-r", message=prompt)
    return response.text

# ================================
# Streamlit App
# ================================
def main():
    st.set_page_config(page_title="🎨 Creative RAG Chatbot", layout="wide")
    st.title("🤖 Creative Retrieval-Augmented Generation Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            text = load_pdf_text(tmp_file.name)

        chunks = chunk_text(text)
        index, stored_chunks = build_vector_db(chunks)
        st.success("✅ PDF processed successfully! Ask your question below.")

        query = st.chat_input("💬 Type your message...")
        if query:
            results = search_db(query, index, stored_chunks)
            answer = generate_answer(query, results)
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("bot", answer))

    for role, chat_text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-bubble user-bubble'>🙋‍♂️ {chat_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble bot-bubble'>🤖 {chat_text}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
