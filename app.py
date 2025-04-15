import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

# Configurazione Streamlit
st.set_page_config(page_title="IA Bando Gratuita", page_icon="📄")
st.title("IA Bando Gratuita")
st.markdown("Fai una domanda sul bando PDF. Nessuna API a pagamento necessaria!")

@st.cache_resource
def create_qa():
    # Carica il PDF
    loader = PyPDFLoader("bando.pdf")
    documents = loader.load()

    # Usa un embedding gratuito
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Costruzione FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Modello di risposta (usa un modello di HuggingFace ospitato via HuggingFaceHub)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )

    return qa_chain

qa = create_qa()

# Interfaccia utente
query = st.text_input("Scrivi la tua domanda:", placeholder="Esempio: Quali sono i requisiti per la borsa?")
if query:
    with st.spinner("Sto cercando la risposta nel bando..."):
        try:
            risposta = qa.run(query)
            st.success(risposta)
        except Exception as e:
            st.error(f"Errore: {str(e)}")
