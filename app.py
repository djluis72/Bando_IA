import streamlit as st
import PyPDF2
from haystack.document_stores import FAISSDocumentStore  # Verifica questa importazione
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import load_pdf_text

# Funzione per caricare il testo da un file PDF
def load_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Inizializza il document store
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

# Prepara il modello di retrieval
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="distilbert-base-uncased")

# Funzione per rispondere alla domanda
def answer_question(question, context):
    # Aggiungi il documento al document store
    document_store.write_documents([{"text": context, "meta": {}}])
    
    # Recupera la risposta dalla pipeline di Haystack
    qa_pipeline = ExtractiveQAPipeline(retriever=retriever)
    result = qa_pipeline.run(query=question, params={"Retriever": {"top_k": 1}})
    
    return result["answers"][0]["answer"] if result["answers"] else "Nessuna risposta trovata"

# App Streamlit
st.title("Domande e Risposte sui Documenti PDF")

# Carica il file PDF
uploaded_file = st.file_uploader("Carica il tuo documento PDF", type="pdf")

if uploaded_file is not None:
    # Estrai il testo dal PDF
    text = load_pdf_text(uploaded_file)

    # Visualizza una parte del testo per verificarne il contenuto
    st.write(text[:1000])  # Mostra solo i primi 1000 caratteri del testo

    # Input per la domanda
    question = st.text_input("Inserisci la tua domanda:")

    if question:
        answer = answer_question(question, text)
        st.write(f"Risposta: {answer}")
