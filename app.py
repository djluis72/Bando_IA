import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Gestione compatibilità RateLimitError (versioni diverse della libreria openai)
try:
    from openai import RateLimitError
except ImportError:
    from openai.error import RateLimitError

# Carica la chiave API OpenAI da Streamlit Cloud (segreti)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Configurazione pagina Streamlit
st.set_page_config(page_title="IA Bando EDISU", page_icon="📄")
st.title("IA Bando EDISU")
st.markdown("Fai una domanda sul bando A.A. 2024/25. L'IA risponde solo sul contenuto del bando PDF.")

# Estensione della classe OpenAIEmbeddings con retry automatico
class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        retry_attempts = 0
        while retry_attempts < 5:
            try:
                # Usa il metodo originale della classe OpenAIEmbeddings
                return super().embed_documents(texts)
            except RateLimitError:
                retry_attempts += 1
                wait_time = 2 ** retry_attempts
                st.warning(f"Rate limit raggiunto. Attesa di {wait_time} secondi...")
                time.sleep(wait_time)
            except Exception as e:
                raise e
        raise Exception("Errore: superato numero massimo di tentativi per embedding.")

# Funzione per creare il sistema di domanda-risposta
@st.cache_resource
def create_qa():
    pdf_path = "bando.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    embeddings = CustomOpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    faiss_index = FAISS.from_documents(documents, embeddings)
    retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=Chat
