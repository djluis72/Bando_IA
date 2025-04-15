import streamlit as st
import PyPDF2
from io import BytesIO

# Funzione per leggere il contenuto del PDF
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Funzione per rispondere alle domande (molto semplice, basato su ricerca nel testo)
def simple_answer(text, query):
    if query.lower() in text.lower():
        return f"La domanda '{query}' è presente nel documento!"
    else:
        return "La domanda non è presente nel documento."

# Interfaccia utente Streamlit
st.title("Domande sul PDF")

# Caricamento del PDF
uploaded_file = st.file_uploader("Carica il tuo documento PDF", type="pdf")

if uploaded_file is not None:
    # Leggi il contenuto del PDF
    pdf_text = read_pdf(uploaded_file)

    # Mostra il contenuto del PDF (opzionale)
    st.subheader("Contenuto del PDF:")
    st.text(pdf_text[:1000])  # Mostra solo i primi 1000 caratteri per evitare troppi dati

    # Campo per inserire la domanda
    query = st.text_input("Fai una domanda sul contenuto del PDF:")

    if query:
        # Rispondi alla domanda (funzione di ricerca semplice)
        answer = simple_answer(pdf_text, query)
        st.write(answer)
