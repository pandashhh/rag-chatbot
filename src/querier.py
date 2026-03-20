# ============================================================
# src/querier.py – RAG-Queries ausführen (Fragen beantworten)
# ============================================================
#
# WAS PASSIERT HIER?
# Das ist das Herzstück des RAG-Systems:
#
# 1. RETRIEVAL (Abrufen):
#    Die Frage des Users wird ebenfalls in einen Vektor umgewandelt.
#    ChromaDB sucht die k ähnlichsten Chunks aus unseren Dokumenten.
#    (k=3 bedeutet: die 3 relevantesten Textabschnitte)
#
# 2. AUGMENTED (Angereichert):
#    Die gefundenen Chunks werden dem LLM als Kontext mitgegeben.
#    Das Prompt sieht dann so aus:
#    "Hier ist relevanter Kontext: [Chunk1] [Chunk2] [Chunk3]
#     Beantworte die Frage: [Frage des Users]"
#
# 3. GENERATION (Generierung):
#    Das LLM (GPT-4o-mini) generiert eine Antwort NUR basierend
#    auf dem gegebenen Kontext – keine Halluzinationen!

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import VectorStoreIndex

load_dotenv()


@dataclass
class QueryResult:
    """
    Ergebnis einer RAG-Query.
    
    Dataclass = einfache Klasse nur für Daten, kein Code.
    Wir geben nicht nur die Antwort zurück, sondern auch die
    Quellen (welche Textstellen wurden verwendet).
    """
    answer: str          # Die generierte Antwort
    sources: list[dict]  # Liste der verwendeten Quellen mit Text und Metadaten


def setup_llm_and_embeddings():
    """
    LLM und Embedding-Modell global konfigurieren.
    
    LlamaIndex hat globale Settings – einmal gesetzt, gelten sie überall.
    Das spart uns, bei jedem Funktionsaufruf die Modelle neu anzugeben.
    """
    # LLM konfigurieren: Claude claude-sonnet-4-20250514 – smart und schnell
    # Für noch bessere Qualität: "claude-opus-4-20250514" (teurer)
    Settings.llm = Anthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1,  # 0 = deterministisch, 1 = kreativ; 0.1 = meistens konsistent
    )

    # Embedding-Modell: HuggingFace – kostenlos, läuft lokal
    # Muss IDENTISCH mit dem in indexer.py sein!
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_query_engine(index: VectorStoreIndex, top_k: int = 3) -> RetrieverQueryEngine:
    """
    Query Engine aus dem Index erstellen.
    
    Die Query Engine kombiniert:
    - Retriever: findet relevante Chunks
    - Response Synthesizer: generiert die Antwort aus den Chunks
    
    Args:
        index: Der VectorStoreIndex aus indexer.py
        top_k: Wie viele Chunks sollen abgerufen werden? (Standard: 3)
               Mehr = umfassendere Antworten, aber mehr Token-Kosten
    
    Returns:
        RetrieverQueryEngine: Fertige Engine zum Abfragen
    """
    # LLM und Embeddings global setzen
    setup_llm_and_embeddings()
    
    # --- Retriever konfigurieren ---
    # VectorIndexRetriever sucht die k ähnlichsten Chunks
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    
    # --- Response Synthesizer konfigurieren ---
    # "compact" = fasst alle Chunks in einem Prompt zusammen (effizient)
    # Alternativen: "tree_summarize" (für lange Docs), "refine" (höchste Qualität)
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        # Deutsch als Antwortsprache erzwingen
        text_qa_template=None,  # Standard-Template verwenden
    )
    
    # Query Engine aus Retriever + Synthesizer zusammenbauen
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    
    return query_engine


def ask_question(query_engine: RetrieverQueryEngine, question: str) -> QueryResult:
    """
    Eine Frage an die Dokumente stellen und Antwort mit Quellen zurückgeben.
    
    Args:
        query_engine: Die Query Engine aus create_query_engine()
        question: Die Frage des Users als String
    
    Returns:
        QueryResult mit Antwort und verwendeten Quellen
    """
    print(f"\n❓ Frage: {question}")
    print("🔄 Suche relevante Stellen...")
    
    # Query ausführen – hier passiert alles:
    # 1. Frage → Embedding
    # 2. ChromaDB-Suche nach ähnlichen Chunks
    # 3. Chunks + Frage → GPT-4o-mini
    # 4. GPT generiert Antwort
    response = query_engine.query(question)
    
    # Quellen extrahieren (welche Textstellen wurden verwendet?)
    sources = []
    for source_node in response.source_nodes:
        sources.append({
            # Der Textausschnitt selbst (erste 300 Zeichen zur Vorschau)
            "text": source_node.node.text[:300] + "..." if len(source_node.node.text) > 300 else source_node.node.text,
            # Ähnlichkeitswert: 1.0 = perfekte Übereinstimmung, 0.0 = keine
            "score": round(source_node.score, 3) if source_node.score else 0,
            # Metadaten: Dateiname, Seitenzahl etc.
            "metadata": source_node.node.metadata,
        })
    
    print(f"✅ Antwort generiert (basierend auf {len(sources)} Quellen)")
    
    return QueryResult(
        answer=str(response),
        sources=sources,
    )
