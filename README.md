# Motore di Ricerca del Bando EDISU

Questa applicazione Streamlit permette di fare domande relative a un bando universitaro (A.A. 2024/25) a partire da un PDF. Utilizza tecniche di embedding per restituire risposte pertinenti basate sul contenuto del bando.

## Funzionalità
- Caricamento di un file PDF.
- Estrazione del testo e generazione di embeddings tramite **Sentence-BERT**.
- Motore di ricerca tramite **FAISS** per trovare le risposte più pertinenti alla tua domanda.

## Installazione

1. Clona questo repository:
    ```bash
    git clone https://github.com/tuo-username/bando_ia.git
    ```

2. Installa le dipendenze:
    ```bash
    pip install -r requirements.txt
    ```

3. Avvia l'applicazione Streamlit:
    ```bash
    streamlit run app.py
    ```

4. Carica il file PDF del bando e fai delle domande nel campo di input.

## Licenza
Questo progetto è open source e distribuito sotto la Licenza MIT.
