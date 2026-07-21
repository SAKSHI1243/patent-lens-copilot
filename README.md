# 🌐 PatentLens AI

PatentLens AI is a hybrid patent search application that helps users compare a technical idea against a collection of patent documents. It combines traditional keyword search with semantic vector search to retrieve relevant patent chunks and reranks the results using a Cross-Encoder for improved relevance.

Live Demo: https://patent-lens-copilot-dz2lmhk66htmw8jenely54.streamlit.app/

## Features

* 🔍 **Hybrid Retrieval:** Combines BM25 keyword search with ChromaDB semantic vector search.
* 🧠 **Cross-Encoder Reranking:** Improves search accuracy by reranking retrieved patent chunks based on semantic relevance.
* 📑 **Patent Similarity Analysis:** Displays the most relevant patent chunks along with similarity scores and patent metadata.
* 📊 **Interactive Dashboard:** Built with Streamlit for searching, comparing, and exploring patent documents.
* 📄 **Report Generation:** Export search results as a downloadable text report.

---

## Tech Stack

| Category         | Technologies                           |
| ---------------- | -------------------------------------- |
| Language         | Python                                 |
| Frontend         | Streamlit                              |
| Vector Database  | ChromaDB                               |
| Sparse Retrieval | BM25 (`rank-bm25`)                     |
| Embedding Model  | `all-MiniLM-L6-v2`                     |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Data Processing  | Pandas, NumPy                          |

---

## System Architecture

```text
Patent Dataset (CSV)
        │
        ▼
Document Chunking
        │
        ▼
Generate Embeddings
        │
        ▼
Store in ChromaDB
        │
        ▼
─────────────────────────────────────
              User Query
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
     BM25 Search      ChromaDB Search
        │                     │
        └──────────┬──────────┘
                   ▼
        Merge Candidate Results
                   │
                   ▼
      Cross-Encoder Reranking
                   │
                   ▼
          Top Relevant Results
                   │
                   ▼
     Streamlit Dashboard & Report
```

---

## Project Structure

```text
PatentLens-AI/
│
├── app.py                  # Streamlit application
├── ingest_patents.py       # Builds the ChromaDB vector database
├── patent_chunks.csv       # Patent dataset
├── PatentLens_DB/          # Persistent ChromaDB database
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SAKSHI1243/patent-lens-copilot.git
cd patent-lens-copilot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the vector database

```bash
python ingest_patents.py
```

### 5. Run the application

```bash
streamlit run app.py
```

---

## How It Works

1. Patent chunks are read from the dataset.
2. Sentence embeddings are generated using the MiniLM embedding model.
3. Embeddings are stored in ChromaDB.
4. A user enters a technical idea.
5. BM25 retrieves keyword-based matches.
6. ChromaDB retrieves semantically similar matches.
7. Candidate results are merged.
8. A Cross-Encoder reranks the candidates.
9. The top results are displayed along with patent metadata and similarity scores.

---

## Future Improvements

* Integrate an LLM to generate personalized patent analysis and recommendations.
* Support larger patent datasets with scalable vector storage.
* Add metadata filtering (date, domain, inventor, etc.).
* Export reports in PDF format.
* Add support for multiple embedding models.

---

## Key Concepts Demonstrated

* Hybrid Retrieval (Sparse + Dense Search)
* Semantic Search
* Vector Databases
* Sentence Embeddings
* Cross-Encoder Reranking
* Information Retrieval (IR)
* Streamlit Application Development

---

## License

This project is intended for educational and research purposes.

