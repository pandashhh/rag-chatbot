# ============================================================
# src/indexer.py – Dokumente laden und in ChromaDB indexieren
# ============================================================
#
# WAS PASSIERT HIER?
# 1. PDF-Dateien werden geladen und in kleine Textabschnitte (Chunks) aufgeteilt
# 2. Jeder Chunk wird durch ein Embedding-Modell in einen Zahlenvektor umgewandelt
#    (z.B. "Der Hund bellt" → [0.23, -0.41, 0.87, ...] mit 1536 Zahlen)
# 3. Diese Vektoren werden in ChromaDB gespeichert (lokal auf der Festplatte)
#
# WARUM DAS GANZE?
# Später können wir eine Frage stellen, die auch in einen Vektor umgewandelt wird.
# ChromaDB findet dann die Chunks, deren Vektoren am ähnlichsten zur Frage sind.
# Das ist das "R" in RAG – Retrieval (Abrufen).

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# .env Datei laden (damit OPENAI_API_KEY verfügbar ist)
load_dotenv()


def get_chroma_collection():
    """
    Verbindung zur ChromaDB herstellen und Collection zurückgeben.
    
    Eine 'Collection' ist wie eine Tabelle in einer Datenbank –
    dort werden alle unsere Dokument-Chunks gespeichert.
    
    Returns:
        chromadb.Collection: Die ChromaDB Collection
    """
    # ChromaDB Client erstellen – speichert Daten im Ordner 'chroma_db'
    # PersistentClient = Daten bleiben nach dem Neustart erhalten
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Collection-Name aus .env lesen (Standard: "rag_documents")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
    
    # Collection holen oder erstellen (get_or_create = kein Fehler wenn sie schon existiert)
    collection = client.get_or_create_collection(collection_name)
    
    return collection


def build_index(uploaded_files: list[str]) -> VectorStoreIndex:
    """
    Aus einer Liste von PDF-Dateien einen durchsuchbaren Index bauen.
    
    ABLAUF:
    1. PDFs einlesen
    2. In Chunks aufteilen (je ~512 Tokens)
    3. Embeddings berechnen (OpenAI API)
    4. In ChromaDB speichern
    5. LlamaIndex VectorStoreIndex zurückgeben
    
    Args:
        uploaded_files: Liste von Dateipfaden zu den PDFs
        
    Returns:
        VectorStoreIndex: Der fertige Index, den wir zum Suchen nutzen
    """
    print(f"📄 Lade {len(uploaded_files)} Dokument(e)...")
    
    # --- SCHRITT 1: Dokumente einlesen ---
    # SimpleDirectoryReader liest alle Dateien aus einem Ordner
    # Wir geben direkt die Dateipfade an
    documents = SimpleDirectoryReader(input_files=uploaded_files).load_data()
    print(f"✅ {len(documents)} Seite(n) geladen")
    
    # --- SCHRITT 2: Chunks erstellen ---
    # SentenceSplitter teilt den Text in überlappende Abschnitte auf
    # chunk_size=512: jeder Chunk hat ca. 512 Tokens (~400 Wörter)
    # chunk_overlap=50: 50 Tokens Überlappung zwischen Chunks
    #   → verhindert, dass wichtige Infos an Chunk-Grenzen verloren gehen
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    
    # --- SCHRITT 3: Embedding-Modell einrichten ---
    # HuggingFace all-MiniLM-L6-v2: kostenlos, läuft lokal auf deinem Mac
    # Kein API-Key nötig! Beim ersten Start wird das Modell heruntergeladen (~90MB)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # --- SCHRITT 4: ChromaDB als Speicher einrichten ---
    chroma_collection = get_chroma_collection()
    
    # ChromaVectorStore = Adapter zwischen LlamaIndex und ChromaDB
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # StorageContext sagt LlamaIndex: "Speicher alles in diesem VectorStore"
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # --- SCHRITT 5: Index bauen ---
    # Hier passiert die eigentliche Arbeit:
    # - Chunks werden erstellt
    # - Embeddings werden berechnet (OpenAI API-Aufruf!)
    # - Alles wird in ChromaDB gespeichert
    print("🔄 Berechne Embeddings und speichere in ChromaDB...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        embed_model=embed_model,
        show_progress=True,
    )
    
    print("✅ Index erfolgreich erstellt!")
    return index


def load_existing_index() -> VectorStoreIndex | None:
    """
    Einen bereits gespeicherten Index aus ChromaDB laden.
    
    Wenn wir die App neu starten, müssen wir nicht alle PDFs nochmal
    verarbeiten – wir laden einfach den gespeicherten Index.
    
    Returns:
        VectorStoreIndex wenn Daten vorhanden, sonst None
    """
    chroma_collection = get_chroma_collection()
    
    # Prüfen ob überhaupt Daten in der Collection sind
    if chroma_collection.count() == 0:
        print("ℹ️ Keine gespeicherten Dokumente gefunden")
        return None
    
    print(f"📚 Lade gespeicherten Index ({chroma_collection.count()} Chunks)...")
    
    # Embedding-Modell einrichten (muss dasselbe sein wie beim Speichern!)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # VectorStore und Index aus ChromaDB laden
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    print("✅ Index geladen!")
    return index


def clear_index():
    """
    Alle gespeicherten Dokumente aus ChromaDB löschen.
    Nützlich wenn man neue Dokumente laden möchte.
    """
    chroma_collection = get_chroma_collection()
    
    # Alle IDs holen und dann löschen
    all_ids = chroma_collection.get()["ids"]
    if all_ids:
        chroma_collection.delete(ids=all_ids)
        print(f"🗑️ {len(all_ids)} Chunks gelöscht")
    else:
        print("ℹ️ Index war bereits leer")
