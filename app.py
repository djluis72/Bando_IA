import streamlit as st
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from utils import extract_text_from_pdf
from haystack import Document

st.title("Domande su PDF - Gratis con Haystack")

uploaded_file = st.file_uploader("Carica un PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    if not text:
        st.warning("Il PDF non contiene testo.")
    else:
        with st.spinner("Indicizzazione in corso..."):
            # 1. Crea documenti
            docs = [Document(content=text)]

            # 2. Crea document store
            document_store = InMemoryDocumentStore()
            document_store.write_documents(docs)

            # 3. Retriever + Reader
            retriever = BM25Retriever(document_store=document_store)
            reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

            # 4. Pipeline QA
            pipe = ExtractiveQAPipeline(reader, retriever)

        query = st.text_input("Fai una domanda:")

        if query:
            with st.spinner("Sto cercando la risposta..."):
                prediction = pipe.run(
                    query=query,
                    params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}},
                )
                answer = prediction["answers"][0].answer
                st.success(f"Risposta: {answer}")
