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

# Funzione per eseguire l'embedding con retry in caso di RateLimitError
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

# Estensione della classe OpenAIEmbeddings con retry automatico
class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        return embed_with_retry(texts, self)

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
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa

# Inizializza il sistema
qa = create_qa()

# Interfaccia utente per la domanda
query = st.text_input("Scrivi la tua domanda:", placeholder="Esempio: Quali sono i requisiti per la borsa?")
if query:
    with st.spinner("Sto cercando la risposta nel bando..."):
        try:
            result = qa({"query": query})
            st.success(result["result"])
        except Exception as e:
            st.error(f"Errore durante la risposta: {e}")
