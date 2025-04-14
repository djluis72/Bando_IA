import time
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from openai.error import RateLimitError  # ✅ CORRETTO

# Carica la chiave da Streamlit Cloud (segreti)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Funzione per gestire retry
def embed_with_retry(texts, embedding_model, max_retries=5):
    retry_attempts = 0
    while retry_attempts < max_retries:
        try:
            return embedding_model.embed_documents(texts)
        except RateLimitError as e:  # ✅ CORRETTO
            retry_attempts += 1
            wait_time = 2 ** retry_attempts
            print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            raise e
    raise Exception("Max retry attempts reached. Unable to proceed with embedding.")

# Embeddings personalizzati con retry
class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        return embed_with_retry(texts, self)

# Il resto del codice rimane invariato (caricamento PDF, costruzione FAISS, ecc.)
