import PyPDF2

# Funzione per caricare il testo da un file PDF
def load_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
