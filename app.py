
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Carica la chiave da Streamlit Cloud (segreti)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Configurazione pagina
st.set_page_config(page_title="IA Bando EDISU", page_icon="📄")
st.title("IA Bando EDISU")
st.markdown("Fai una domanda sul bando A.A. 2024/25. L'IA risponde solo sul contenuto del bando PDF.")

@st.cache_resource
def create_qa():
    pdf_path = "bando.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

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
