import streamlit as st
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Imposta la configurazione della pagina Streamlit
st.set_page_config(page_title="Motore di Ricerca del Bando", page_icon="📄")
st.title("Motore di Ricerca del Bando EDISU")
st.markdown("Fai una domanda sul bando A.A. 2024/25. L'IA risponde solo sul contenuto del bando PDF.")

# Percorso del PDF
pdf_path = "bando.pdf"
CACHE_PATH = "bando_cache.pkl"

# Carica e crea gli embeddings solo se non esiste la cache
def create_embeddings():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            embeddings, index = pickle.load(f)
        st.write("Modello caricato dalla cache!")
    else:
        # Carica il PDF e creazione degli embeddings
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Carica il modello pre-addestrato Sentence-BERT
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Crea gli embeddings per il documento
        embeddings = [model.encode(doc.page_content) for doc in documents]

        # Crea un indice FAISS per la ricerca
        faiss_index = faiss.IndexFlatL2(len(embeddings[0]))  # Indice per la ricerca (distanza euclidea)
        faiss_index.add(np.array(embeddings))  # Aggiungi gli embeddings all'indice

        with open(CACHE_PATH, "wb") as f:
            pickle.dump((embeddings, faiss_index), f)
        
        st.write("Modello creato e memorizzato nella cache!")
    return embeddings, faiss_index

# Crea o carica gli embeddings
embeddings, faiss_index = create_embeddings()

# Input della query
query = st.text_input("Scrivi la tua domanda:", placeholder="Esempio: Quali sono i requisiti per la borsa?")
if query:
    with st.spinner("Sto cercando la risposta nel bando..."):
        # Calcola l'embedding della query
        query_embedding = SentenceTransformer('paraphrase-MiniLM-L6-v2').encode([query])

        # Cerca la query nell'indice FAISS
        distances, indices = faiss_index.search(np.array(query_embedding), k=3)

        # Ottieni le risposte più pertinenti dai documenti
        st.write("Le risposte più pertinenti:")
        for i in range(len(indices[0])):
            st.write(f"**Risposta {i + 1}:** {documents[indices[0][i]].page_content[:500]}...")
