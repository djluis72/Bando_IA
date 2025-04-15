import streamlit as st
from transformers import pipeline

# Carica il modello solo una volta
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="deepset/tinyroberta-squad2")

qa_pipeline = load_model()

st.title("Domande su un testo PDF")

uploaded_file = st.file_uploader("Carica un file PDF", type="pdf")
if uploaded_file is not None:
    import PyPDF2
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    st.text_area("Testo del bando", full_text, height=300)

    question = st.text_input("Fai una domanda sul bando:")
    if question:
        with st.spinner("Sto cercando la risposta..."):
            result = qa_pipeline(question=question, context=full_text)
            st.markdown(f"**Risposta:** {result['answer']}")
