# ============================================================
# src/app.py – Streamlit Web-UI für den RAG Chatbot
# ============================================================
#
# Das ist die Hauptdatei – hier startest du die App:
#   streamlit run src/app.py
#
# Streamlit funktioniert so:
# - Die Datei wird von oben nach unten ausgeführt
# - Bei jeder User-Interaktion (Button klicken, Text eingeben)
#   wird die GESAMTE Datei neu ausgeführt
# - st.session_state speichert Daten zwischen den Ausführungen
#
# AUFBAU DER APP:
# ┌─────────────────────────────────┐
# │  📚 RAG Chatbot                 │
# ├──────────┬──────────────────────┤
# │ Sidebar  │  Chat-Bereich        │
# │ (Upload) │  (Fragen stellen)    │
# └──────────┴──────────────────────┘

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

# Unsere eigenen Module importieren
from indexer import build_index, clear_index, load_existing_index
from querier import ask_question, create_query_engine

# .env laden (API-Keys)
load_dotenv()

# ============================================================
# SEITENKONFIGURATION
# ============================================================
# Muss der ERSTE Streamlit-Aufruf in der Datei sein!
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",           # Volle Breite nutzen
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE INITIALISIERUNG
# ============================================================
# session_state = persistenter Speicher zwischen Streamlit-Reruns
# Ohne session_state würde bei jedem Klick alles zurückgesetzt

if "messages" not in st.session_state:
    # Chat-Verlauf: Liste von {"role": "user"/"assistant", "content": "..."}
    st.session_state.messages = []

if "index" not in st.session_state:
    # Der LlamaIndex VectorStoreIndex (None wenn noch keine Docs geladen)
    st.session_state.index = None

if "query_engine" not in st.session_state:
    # Die Query Engine (wird aus dem Index erstellt)
    st.session_state.query_engine = None

if "docs_loaded" not in st.session_state:
    # Anzahl geladener Dokumente
    st.session_state.docs_loaded = 0

# ============================================================
# BEIM START: Bestehenden Index laden (falls vorhanden)
# ============================================================
# Wenn wir die App neu starten und schon Dokumente indexiert haben,
# laden wir den gespeicherten Index aus ChromaDB
if st.session_state.index is None:
    existing_index = load_existing_index()
    if existing_index:
        st.session_state.index = existing_index
        st.session_state.query_engine = create_query_engine(existing_index)

# ============================================================
# SIDEBAR – Dokument-Upload
# ============================================================
with st.sidebar:
    st.title("📁 Dokumente")
    st.markdown("Lade PDFs hoch, die der Chatbot kennen soll.")
    
    # API-Key Warnung
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("⚠️ ANTHROPIC_API_KEY fehlt in der .env Datei!")
        st.code("cp .env.example .env\n# Dann API-Key eintragen")
        st.stop()  # App stoppen wenn kein API-Key vorhanden
    
    # ---- Datei-Uploader ----
    uploaded_files = st.file_uploader(
        "PDFs auswählen",
        type=["pdf"],
        accept_multiple_files=True,  # Mehrere PDFs auf einmal möglich
        help="Du kannst mehrere PDF-Dateien gleichzeitig hochladen",
    )
    
    # ---- Index-Button ----
    if uploaded_files:
        if st.button("🔄 Dokumente indexieren", type="primary", use_container_width=True):
            
            # Fortschrittsanzeige
            with st.spinner("Verarbeite Dokumente..."):
                
                # Temporäre Dateien erstellen
                # (Streamlit gibt uns UploadedFile-Objekte, keine echten Pfade)
                # Wir müssen sie kurz auf die Festplatte schreiben
                temp_paths = []
                for uploaded_file in uploaded_files:
                    # Temporäre Datei erstellen (wird automatisch gelöscht)
                    with tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix=".pdf",
                        prefix=f"{uploaded_file.name}_"
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_paths.append(tmp.name)
                
                try:
                    # Alten Index löschen und neuen aufbauen
                    clear_index()
                    index = build_index(temp_paths)
                    
                    # Index und Query Engine in session_state speichern
                    st.session_state.index = index
                    st.session_state.query_engine = create_query_engine(index)
                    st.session_state.docs_loaded = len(uploaded_files)
                    
                    # Chat-Verlauf zurücksetzen (neue Dokumente = neues Gespräch)
                    st.session_state.messages = []
                    
                    st.success(f"✅ {len(uploaded_files)} Dokument(e) indexiert!")
                    
                except Exception as e:
                    st.error(f"❌ Fehler beim Indexieren: {e}")
                
                finally:
                    # Temporäre Dateien aufräumen
                    for path in temp_paths:
                        os.unlink(path)
    
    # ---- Status-Anzeige ----
    st.divider()
    
    if st.session_state.index is not None:
        st.success("✅ Dokumente bereit")
        
        # Index löschen Button
        if st.button("🗑️ Index löschen", use_container_width=True):
            clear_index()
            st.session_state.index = None
            st.session_state.query_engine = None
            st.session_state.messages = []
            st.session_state.docs_loaded = 0
            st.rerun()  # App neu laden
    else:
        st.info("ℹ️ Noch keine Dokumente geladen")
    
    # ---- Einstellungen ----
    st.divider()
    st.subheader("⚙️ Einstellungen")
    top_k = st.slider(
        "Anzahl Quellen (top_k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Wie viele Textabschnitte sollen für die Antwort genutzt werden?",
    )
    
    # Query Engine neu erstellen wenn top_k geändert wird
    if st.session_state.index and top_k != 3:
        st.session_state.query_engine = create_query_engine(
            st.session_state.index, 
            top_k=top_k
        )

# ============================================================
# HAUPTBEREICH – Chat Interface
# ============================================================
st.title("📚 RAG Chatbot")
st.caption("Stelle Fragen zu deinen hochgeladenen Dokumenten")

# Trennlinie
st.divider()

# ---- Chat-Verlauf anzeigen ----
# Alle bisherigen Nachrichten aus session_state anzeigen
for message in st.session_state.messages:
    # st.chat_message erstellt einen Chat-Bubble
    # role="user" = rechts, role="assistant" = links (mit Bot-Icon)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Quellen anzeigen (nur bei Assistenten-Nachrichten)
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📎 {len(message['sources'])} Quelle(n) anzeigen"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Quelle {i}** (Ähnlichkeit: {source['score']})")
                    
                    # Metadaten anzeigen (Dateiname, Seite etc.)
                    if source["metadata"]:
                        meta_str = " | ".join(
                            f"{k}: {v}" for k, v in source["metadata"].items()
                            if k in ["file_name", "page_label", "file_path"]
                        )
                        if meta_str:
                            st.caption(meta_str)
                    
                    # Textvorschau
                    st.text(source["text"])
                    
                    if i < len(message["sources"]):
                        st.divider()

# ---- Eingabefeld ----
# st.chat_input = Texteingabe am unteren Rand (wie ChatGPT)
if prompt := st.chat_input(
    "Stelle eine Frage zu deinen Dokumenten...",
    disabled=st.session_state.query_engine is None,  # Deaktiviert wenn kein Index
):
    # ---- User-Nachricht hinzufügen ----
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # User-Bubble direkt anzeigen
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ---- Antwort generieren ----
    with st.chat_message("assistant"):
        
        if st.session_state.query_engine is None:
            # Fehlerfall: kein Index vorhanden
            response_text = "⚠️ Bitte lade zuerst Dokumente in der Sidebar hoch!"
            st.markdown(response_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
            })
        else:
            # Ladeanimation während der Anfrage
            with st.spinner("Suche in Dokumenten..."):
                try:
                    # RAG-Query ausführen (aus querier.py)
                    result = ask_question(st.session_state.query_engine, prompt)
                    
                    # Antwort anzeigen
                    st.markdown(result.answer)
                    
                    # Quellen anzeigen
                    if result.sources:
                        with st.expander(f"📎 {len(result.sources)} Quelle(n) anzeigen"):
                            for i, source in enumerate(result.sources, 1):
                                st.markdown(f"**Quelle {i}** (Ähnlichkeit: {source['score']})")
                                if source["metadata"]:
                                    meta_str = " | ".join(
                                        f"{k}: {v}" for k, v in source["metadata"].items()
                                        if k in ["file_name", "page_label", "file_path"]
                                    )
                                    if meta_str:
                                        st.caption(meta_str)
                                st.text(source["text"])
                                if i < len(result.sources):
                                    st.divider()
                    
                    # Antwort in Chat-Verlauf speichern
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Fehler bei der Anfrage: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

# ---- Hinweis wenn noch keine Dokumente geladen ----
if st.session_state.query_engine is None:
    st.info("👈 Lade zuerst ein PDF in der Sidebar hoch, dann kannst du Fragen stellen.")
    
    # Beispiel-Use-Cases anzeigen
    st.markdown("### 💡 Beispiele was du damit machen kannst:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📖 Forschungsarbeiten**\nFrage deine PDFs nach spezifischen Informationen")
    with col2:
        st.markdown("**📋 Verträge**\nFinde wichtige Klauseln ohne alles lesen zu müssen")
    with col3:
        st.markdown("**📚 Lehrbücher**\nStelle Lernfragen zu deinen Unterlagen")
