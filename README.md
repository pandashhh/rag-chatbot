# 📚 RAG Chatbot

> Projekt 1/5 meines AI Engineering Portfolios

Ein Chatbot, der deine PDF-Dokumente versteht und Fragen dazu beantwortet.
Basiert auf **Retrieval Augmented Generation (RAG)** mit LlamaIndex und ChromaDB.

## 🎯 Was kann das?

- PDFs hochladen (einzeln oder mehrere gleichzeitig)
- Natürliche Fragen in Deutsch oder Englisch stellen
- Antworten basieren **nur** auf den hochgeladenen Dokumenten (keine Halluzinationen)
- Quellen mit Ähnlichkeitswert werden zu jeder Antwort angezeigt

## 🏗️ Architektur

```
PDF Upload → Chunks (512 Tokens) → Embeddings (HuggingFace, lokal) → ChromaDB
                                                                           ↓
User Frage → Embedding → Ähnlichkeitssuche → Top-3 Chunks → Claude → Antwort
```

**RAG erklärt in einem Satz:**
Statt dem LLM alles zu sagen, suchen wir erst die relevantesten Textstellen
und geben nur diese als Kontext mit – das macht Antworten präziser und günstiger.

## 🛠️ Tech Stack

| Tool | Zweck |
|------|-------|
| [LlamaIndex](https://www.llamaindex.ai/) | RAG-Framework: Laden, Indexieren, Abfragen |
| [ChromaDB](https://www.trychroma.com/) | Vektordatenbank (lokal, kein Cloud-Account nötig) |
| [Anthropic Claude API](https://www.anthropic.com/) | LLM für die Antwortgenerierung (claude-sonnet) |
| [HuggingFace sentence-transformers](https://huggingface.co/) | Embeddings (lokal, kostenlos, kein API-Key nötig) |
| [Streamlit](https://streamlit.io/) | Web-UI |
| Python 3.11 | Programmiersprache |

## 🚀 Lokale Installation

### Voraussetzungen
- Python 3.11+
- Anthropic API Key ([hier holen](https://console.anthropic.com/))

### Setup

```bash
# 1. Repository klonen
git clone https://github.com/DEIN-USERNAME/rag-chatbot.git
cd rag-chatbot

# 2. Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
source venv/bin/activate        # Mac/Linux
# oder: venv\Scripts\activate   # Windows

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. API-Key konfigurieren
cp .env.example .env
# Öffne .env und trage deinen ANTHROPIC_API_KEY ein

# 5. App starten
streamlit run src/app.py
```

Die App öffnet sich automatisch unter http://localhost:8501

## 📖 Nutzung

1. PDF(s) in der linken Sidebar hochladen
2. Auf **"Dokumente indexieren"** klicken (dauert je nach Größe 10–60 Sekunden)
3. Fragen im Chat eingeben
4. Unter jeder Antwort die verwendeten Quellen aufklappen

## 💡 Lernziele (für Interessierte)

Dieses Projekt demonstriert:
- **RAG-Pipeline**: Von rohem PDF bis zur LLM-Antwort
- **Vektordatenbanken**: Wie ChromaDB Embeddings speichert und sucht
- **LlamaIndex API**: Documents, Nodes, Retrievers, QueryEngine
- **Streamlit**: Schnelle Web-UIs für ML-Projekte
- **Professionelle Projektstruktur**: .env, .gitignore, Modularisierung

## 📂 Projektstruktur

```
rag-chatbot/
├── src/
│   ├── app.py          # Streamlit UI (Einstiegspunkt)
│   ├── indexer.py      # Dokumente laden + indexieren
│   └── querier.py      # RAG-Query-Logik
├── .env.example        # API-Key Vorlage
├── requirements.txt    # Abhängigkeiten
└── CLAUDE.md           # Kontext für Claude Code
```

## 🔮 Mögliche Erweiterungen

- [ ] Unterstützung für .docx, .txt, .csv
- [ ] Mehrsprachige Embeddings
- [ ] Deployment auf GCP Cloud Run (→ Projekt 4)
- [ ] Konversationsspeicher (Follow-up Fragen)

---

*Teil meines [AI Engineering Portfolios](https://github.com/DEIN-USERNAME)*
