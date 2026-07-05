import sys
import os

# 🚨 CRITICAL CLOUD WORKAROUND: Linux SQLite Version Override (Must be on top)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

print("🔄 Initializing native cloud-compatible ChromaDB space...")
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "PatentLens_DB")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 🚨 Secure environment isolation for automated cloud writing
chroma_settings = Settings(
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)

# settings object explicitly injected inside client runtime
chroma_client = chromadb.PersistentClient(path=db_path, settings=chroma_settings)

collection = chroma_client.get_or_create_collection(
    name="universal_global_patents", 
    embedding_function=sentence_transformer_ef 
)

chunks_file = os.path.join(current_dir, "patent_chunks.csv")

if not os.path.exists(chunks_file):
    print(f"❌ ERROR: Could not find your chunk file at: {chunks_file}")
    exit()

print(f"📖 Reading structural chunk parameters from {chunks_file}...")
df = pd.read_csv(chunks_file, dtype={'chunk_id': str, 'patent_id': str, 'text': str, 'title': str, 'date': str})

df['text'] = df['text'].fillna("")
df['title'] = df['title'].fillna("Unknown Title")

documents = df['text'].tolist()
ids = df['chunk_id'].astype(str).tolist()

metadatas = []
for idx, row in df.iterrows():
    metadatas.append({
        "patent_id": str(row.get('patent_id')),
        "title": str(row.get('title')),
        "date": str(row.get('date'))
    })

print(f"🧠 Indexing {len(documents)} universal claims sentences natively...")

# Safe processing in structural chunks
batch_size = 500
for i in range(0, len(documents), batch_size):
    end_idx = min(i + batch_size, len(documents))
    collection.add(
        documents=documents[i:end_idx],
        metadatas=metadatas[i:end_idx],
        ids=ids[i:end_idx]
    )

print("✅ Ingestion Completed Flawlessly!")