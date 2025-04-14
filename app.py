import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Gestione compatibilità RateLimitError
try:
    from openai import RateLimitError
except ImportError:
    from openai._exceptions import RateLimitError

# Carica la chiave API OpenAI da Streamlit Cloud (segreti)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Configurazione pagina Streamlit
st.set_page_config(page_title="IA Bando EDISU", page_icon="📄")
st.title("IA Bando EDISU")
st.markdown("Fai una domanda sul bando A.A. 2024/25. L'IA risponde solo sul contenuto del bando PDF.")

# Funzione con retry in caso di RateLimitError
def embed_with_retry(texts, embedding_model, max_retries=5):
    retry_attempts = 0
    while retry_attempts < max_retries:
        try:
            return embedding_model.embed_documents(texts)
        except RateLimitError:
            retry_attempts += 1
            wait_time = 2 ** retry_attempts
            st.warning(f"Rate limit raggiunto. Attesa di {wait_time} secondi...")
            time.sleep(wait_time)
        except Exception as e:
            raise e
    raise Exception("Errore: superato numero massimo di tentativi per embedding.")

# Estensione Embeddings con retry
class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        return embed_with_retry(texts, self)

@st.cache_resource
def create_qa_
