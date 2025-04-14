import time
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import openai
from openai import error  # Aggiungi questa importazione per gestire gli errori di OpenAI

# Carica la chiave da Streamlit Cloud (segreti)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Configurazione pagina
st.set_page_config(page_title="IA Bando EDISU", page_icon="📄")
st.title("IA Bando EDISU")
st.markdown("Fai una domanda sul bando A.A. 2024/25. L'IA risponde solo sul contenuto del bando PDF.")

# Funzione per gestire i retry in caso di RateLimitError
def embed_with_retry(texts, embedding_model, max_retries=5):
    retry_attempts = 0
    while retry_attempts < max_retries:
        try:
            return embedding_model.embed_documents(texts)
        except error.RateLimitError as e:  # Usa 'error.RateLimitError' invece di 'openai.error.RateLimitError'
            retry_attempts += 1
            wait_time = 2 ** retry_attempts  # Backoff esponenziale
            print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
            time.sleep(wait_time)  # Aspetta prima di ritentare
        except Exception as e:
            raise e  # Rilancia l'errore se non è un RateLimitError
    raise Exception("Max retry attempts reached. Unable to proceed with embedding.")

# Creazione della classe di embedding personalizzata con retry
class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        return embed_with_retry(texts, self)

@st.cache_resource
def create_qa():
    pdf_path = "bando.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Creazione delle embeddings usando la versione con retry
    embeddings = CustomOpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Creazione dell'indice FAISS
    faiss_index = FAISS.from_documents(documents, embeddings)
    
    # Recupero dei dati tramite l'indice FAISS
    retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

    # Configurazione del modello di chat
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa

qa = create_qa()

query = st.text_input("Scrivi la tua domanda:", placeholder="Esempio: Quali sono i requisiti per la borsa?")
if query:
    with st.spinner("Sto cercando la risposta nel bando..."):
        result = qa({"query": query})
        st.success(result["result"])
