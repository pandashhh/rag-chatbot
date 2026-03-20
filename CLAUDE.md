# Projekt: RAG Chatbot (Projekt 1 von 5)

## Was ist das hier?
Ein RAG-Chatbot (Retrieval Augmented Generation) mit LlamaIndex.
Der User kann PDFs hochladen und dann per Chat Fragen dazu stellen.
Dieses Projekt ist Teil eines AI Engineering Portfolios für GitHub.

## Tech Stack
- **LlamaIndex** – Dokumente laden, indexieren, RAG-Queries
- **ChromaDB** – Vektordatenbank (lokal, kein Account nötig)
- **Streamlit** – einfaches Web-UI
- **Python 3.11+** mit virtualenv

## Projektstruktur
```
rag-chatbot/
├── CLAUDE.md           ← diese Datei (Claude Code liest sie automatisch)
├── README.md           ← GitHub-Beschreibung mit Demo-Screenshots
├── requirements.txt    ← alle Python-Abhängigkeiten
├── .env.example        ← zeigt welche API-Keys gebraucht werden
├── .gitignore          ← schützt .env und __pycache__ etc.
├── src/
│   ├── app.py          ← Streamlit UI (Hauptdatei, hier starten)
│   ├── indexer.py      ← Dokumente laden + in ChromaDB indexieren
│   └── querier.py      ← RAG-Query-Logik
└── docs/
    └── architecture.md ← technische Erklärung der Architektur
```

## Wie starten
```bash
# 1. Virtuelle Umgebung aktivieren
source venv/bin/activate  # Mac/Linux
# oder: venv\Scripts\activate  # Windows

# 2. App starten
streamlit run src/app.py
```

## Was Claude Code wissen soll
- Wir verwenden LlamaIndex (NICHT LangChain) für dieses Projekt
- ChromaDB läuft lokal – kein Cloud-Account nötig
- Streamlit für das UI – kein React/JS
- Alle Kommentare auf Deutsch (Lernprojekt)
- API-Keys kommen aus .env (nie hardcoden!)
- Der Lernende ist Python-Mittelstufe – Kommentare ausführlich halten

## Lernziele dieses Projekts
1. RAG-Pipeline verstehen (Dokument → Chunks → Embeddings → Vektorstore → Query)
2. LlamaIndex API kennenlernen
3. ChromaDB als Vektordatenbank nutzen
4. Streamlit für schnelle UIs verwenden
5. Professionelle Projektstruktur auf GitHub zeigen
